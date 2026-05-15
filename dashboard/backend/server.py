import warnings
warnings.filterwarnings("ignore")

import sys, os, json, glob, threading, subprocess
import signal
import time, uuid, re, shutil, math
from pathlib import Path
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
BASE_STD_SUMOCFG = os.path.join(PROJECT_ROOT, "mehd_tol.sumocfg")
BASE_INDIA_SUMOCFG = os.path.join(PROJECT_ROOT, "india_stress.sumocfg")

app = Flask(__name__)
CORS(app, origins="*")

_sim_state = {
    "running": False,
    "run_id": None,
    "scenario": None,
    "progress": 0,
    "step": 0,
    "total": 1800,
    "log_lines": [],
    "process": None,
    "result": None,
    "latest_csv": None,
    "error": None,
}
_state_lock = threading.Lock()


@dataclass
class SimConfig:
    scenario_name: str
    mode: str
    use_rl_signals: bool
    use_rl_glosa: bool
    sim_steps: int
    traffic_type: str
    ev_count: int


def _append_log(line: str):
    with _state_lock:
        _sim_state["log_lines"].append({"line": line, "ts": time.strftime("%H:%M:%S")})
        if len(_sim_state["log_lines"]) > 500:
            _sim_state["log_lines"] = _sim_state["log_lines"][-500:]


def _latest_kpi_csv():
    candidates = []
    for pat in ("india_*_kpi.csv", "*_kpi.csv", "v2x_kpi_log.csv"):
        candidates.extend(glob.glob(os.path.join(OUTPUTS_DIR, pat)))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _latest_run_csv():
    candidates = glob.glob(os.path.join(OUTPUTS_DIR, "run_*.csv"))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _run_csv_path(run_id: str) -> str:
    return os.path.join(OUTPUTS_DIR, f"run_{run_id}.csv")


def _run_meta_path(run_id: str) -> str:
    return os.path.join(OUTPUTS_DIR, f"run_{run_id}.meta.json")


def _load_run_meta(run_id: str) -> dict:
    path = _run_meta_path(run_id)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _pick_vehicle_type(index: int, ev_count: int, traffic_type: str) -> str:
    if index < ev_count:
        return "ev"
    if traffic_type == "indian":
        return "two_wheeler" if index % 3 else "car"
    return "car"


def _prepare_runtime_config(config: SimConfig, run_id: str) -> str:
    base_cfg = BASE_INDIA_SUMOCFG if config.traffic_type == "indian" else BASE_STD_SUMOCFG
    cfg_tree = ET.parse(base_cfg)
    cfg_root = cfg_tree.getroot()
    input_node = cfg_root.find("input")
    if input_node is None:
        raise RuntimeError("SUMO config missing <input>")

    def _abs_path_list(value: str) -> str:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        abs_parts = []
        for p in parts:
            if os.path.isabs(p):
                abs_parts.append(p.replace("\\", "/"))
            else:
                abs_parts.append(os.path.join(PROJECT_ROOT, p).replace("\\", "/"))
        return ",".join(abs_parts)

    # Normalize base config file references to absolute paths so runtime cfg can
    # live under outputs/ without breaking relative path resolution.
    for tag in ("net-file", "route-files", "additional-files"):
        node = input_node.find(tag)
        if node is not None and node.get("value"):
            node.set("value", _abs_path_list(node.get("value", "")))

    gui_node = cfg_root.find("gui-settings-file")
    if gui_node is not None and gui_node.get("value"):
        gui_node.set("value", _abs_path_list(gui_node.get("value", "")))

    route_node = input_node.find("route-files")
    if route_node is None:
        raise RuntimeError("SUMO config missing <route-files>")
    original_routes = [p.strip() for p in route_node.get("value", "").split(",") if p.strip()]
    base_route_path = original_routes[0]
    if not os.path.exists(base_route_path):
        raise FileNotFoundError(f"Route file not found: {base_route_path}")

    route_tree = ET.parse(base_route_path)
    route_root = route_tree.getroot()
    vehicles = route_root.findall("vehicle")

    ev_count = max(0, min(int(config.ev_count), len(vehicles)))
    for idx, vehicle in enumerate(vehicles):
        vehicle.set("type", _pick_vehicle_type(idx, ev_count, config.traffic_type))

    runtime_route_path = os.path.join(OUTPUTS_DIR, f"run_{run_id}.rou.xml")
    route_tree.write(runtime_route_path, encoding="utf-8", xml_declaration=True)

    preserved_routes = [p.replace("\\", "/") for p in original_routes[1:]]
    route_node.set("value", ",".join([runtime_route_path.replace("\\", "/"), *preserved_routes]))

    time_node = cfg_root.find("time")
    if time_node is not None:
        end_node = time_node.find("end")
        if end_node is not None:
            end_node.set("value", str(max(1, int(config.sim_steps))))

    runtime_cfg_path = os.path.join(OUTPUTS_DIR, f"run_{run_id}.sumocfg")
    cfg_tree.write(runtime_cfg_path, encoding="utf-8", xml_declaration=True)
    return runtime_cfg_path


def _run_simulation_thread(config: SimConfig, run_id: str):
    try:
        runtime_cfg = _prepare_runtime_config(config, run_id)
        mode = config.mode if config.mode in {"null_baseline", "rule_based_v2x", "rule_based_no_ev"} else "rule_based_v2x"
        output_csv = _run_csv_path(run_id)
        py = f"""
import sys
sys.path.insert(0, r'{PROJECT_ROOT}')
import config as cfg
import main
cfg.SUMO_CFG = r'{runtime_cfg}'
main.run(no_gui=True, mode={mode!r}, sim_steps={int(config.sim_steps)}, output_csv={output_csv!r}, use_rl_signals={bool(config.use_rl_signals)}, use_rl_glosa={bool(config.use_rl_glosa)})
"""
        cmd = [sys.executable, "-u", "-c", py]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=PROJECT_ROOT,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        with _state_lock:
            _sim_state["process"] = process

        rgx = re.compile(r"step (\d+)/(\d+)")
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            _append_log(line)
            m = rgx.search(line)
            if m:
                step, total = int(m.group(1)), int(m.group(2))
                total = total if total > 0 else 1
                with _state_lock:
                    _sim_state["step"] = step
                    _sim_state["total"] = total
                    _sim_state["progress"] = int(step / total * 100)

        process.wait()
        if process.returncode == 0:
            avg_speed = avg_wait = 0.0
            collisions = 0
            if os.path.exists(output_csv):
                df = pd.read_csv(output_csv)
                avg_speed = float(df["avg_speed_mps"].mean()) if "avg_speed_mps" in df else 0.0
                avg_wait = float(df["avg_wait_s"].mean()) if "avg_wait_s" in df else 0.0
                if "collision_rate" in df:
                    collisions = int(pd.to_numeric(df["collision_rate"], errors="coerce").fillna(0).sum())

            with _state_lock:
                _sim_state["result"] = {
                    "run_id": run_id,
                    "avg_speed": round(avg_speed, 3),
                    "avg_wait": round(avg_wait, 3),
                    "collisions": int(collisions),
                    "csv": f"run_{run_id}.csv",
                }
                _sim_state["latest_csv"] = output_csv
                _sim_state["step"] = int(config.sim_steps)
                _sim_state["total"] = int(config.sim_steps)
                _sim_state["progress"] = 100

            with open(_run_meta_path(run_id), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_id": run_id,
                        "scenario_name": config.scenario_name,
                        "mode": mode,
                        "use_rl_signals": bool(config.use_rl_signals),
                        "use_rl_glosa": bool(config.use_rl_glosa),
                        "sim_steps": int(config.sim_steps),
                        "traffic_type": config.traffic_type,
                        "ev_count": int(config.ev_count),
                        "csv": os.path.basename(output_csv),
                    },
                    f,
                    ensure_ascii=True,
                    indent=2,
                )
            _append_log(f"Simulation complete: {run_id}")
        else:
            with _state_lock:
                _sim_state["error"] = "Process exited with error"
            _append_log("ERROR: Simulation failed")
    except Exception as e:
        with _state_lock:
            _sim_state["error"] = str(e)
        _append_log(f"ERROR: {e}")
    finally:
        with _state_lock:
            _sim_state["running"] = False
            _sim_state["process"] = None


@app.route("/")
def serve_dashboard():
    return send_from_directory(os.path.join(os.path.dirname(__file__), ".."), "index.html")


@app.get("/api/status")
def api_status():
    with _state_lock:
        payload = dict(_sim_state)
        payload.pop("process", None)
    return jsonify(payload)


@app.post("/api/run")
def api_run():
    with _state_lock:
        if _sim_state["running"]:
            return jsonify({"error": "Already running"}), 409
    body = request.get_json(force=True, silent=True) or {}
    config = SimConfig(
        scenario_name=str(body.get("scenario_name", "run")).strip() or "run",
        mode=str(body.get("mode", "rule_based_v2x")),
        use_rl_signals=bool(body.get("use_rl_signals", False)),
        use_rl_glosa=bool(body.get("use_rl_glosa", False)),
        sim_steps=int(body.get("sim_steps", 1800)),
        traffic_type=str(body.get("traffic_type", "standard")),
        ev_count=int(body.get("ev_count", 5)),
    )
    run_id = str(uuid.uuid4())[:8]
    with _state_lock:
        _sim_state.update(
            {
                "running": True,
                "run_id": run_id,
                "scenario": config.scenario_name,
                "progress": 0,
                "step": 0,
                "total": config.sim_steps,
                "log_lines": [],
                "result": None,
                "error": None,
                "latest_csv": None,
            }
        )
    threading.Thread(target=_run_simulation_thread, args=(config, run_id), daemon=True).start()
    return jsonify({"run_id": run_id, "status": "started"})


@app.post("/api/cancel")
def api_cancel():
    with _state_lock:
        p = _sim_state["process"]
    if p:
        try:
            if os.name == "nt":
                try:
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                    p.wait(timeout=5)
                except Exception:
                    pass
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=3)
                    except Exception:
                        pass
                if p.poll() is None:
                    subprocess.run(
                        ["taskkill", "/PID", str(p.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
            else:
                p.terminate()
        except Exception:
            pass
    with _state_lock:
        _sim_state["running"] = False
    _append_log("Simulation cancelled by user")
    return jsonify({"status": "cancelled"})


@app.get("/api/stream")
def api_stream():
    def generate():
        last_index = 0
        while True:
            with _state_lock:
                lines = list(_sim_state["log_lines"])
                new_lines = lines[last_index:]
                last_index = len(lines)
                running = _sim_state["running"]
                progress = _sim_state["progress"]
                step = _sim_state["step"]
                total = _sim_state["total"]
                result = _sim_state["result"]
                error = _sim_state["error"]

            for line_dict in new_lines:
                yield f"data: {json.dumps(line_dict)}\n\n"
            yield f"data: {json.dumps({'type':'status','running':running,'progress':progress,'step':step,'total':total,'result':result,'error':error})}\n\n"
            time.sleep(0.5)
            if not running and last_index >= len(lines):
                break

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


def _n(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _clean_json_value(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _clean_row(d):
    return {k: _clean_json_value(v) for k, v in d.items()}


@app.get("/api/results")
def api_results():
    rows = []
    base_csv = os.path.join(OUTPUTS_DIR, "btp_summary_table.csv")
    if os.path.exists(base_csv):
        df = pd.read_csv(base_csv)
        for _, r in df.iterrows():
            rows.append(
                _clean_row(
                    {
                        "run_id": str(r.get("scenario", "baseline")),
                        "scenario_name": r.get("scenario"),
                        "avg_speed_mps": _n(r.get("avg_speed_mps")),
                        "avg_wait_s": _n(r.get("avg_wait_s")),
                        "throughput": _n(r.get("throughput")),
                        "ev_travel_time_s": _n(r.get("ev_travel_time_s")),
                        "collisions": _n(r.get("collisions")),
                        "notes": r.get("notes"),
                        "source": "baseline",
                    }
                )
            )

    for fp in glob.glob(os.path.join(OUTPUTS_DIR, "run_*.csv")):
        run_id = Path(fp).stem.replace("run_", "", 1)
        meta = _load_run_meta(run_id)
        try:
            df = pd.read_csv(fp)
            rows.append(
                _clean_row(
                    {
                        "run_id": run_id,
                        "scenario_name": meta.get("scenario_name", run_id),
                        "avg_speed_mps": float(df["avg_speed_mps"].mean()) if "avg_speed_mps" in df else None,
                        "avg_wait_s": float(df["avg_wait_s"].mean()) if "avg_wait_s" in df else None,
                        "throughput": float(df["total_arrived_cum"].max()) if "total_arrived_cum" in df and not df.empty else None,
                        "ev_travel_time_s": None,
                        "collisions": float(pd.to_numeric(df.get("collision_rate", pd.Series([0])), errors="coerce").fillna(0).sum()),
                        "notes": "Interactive run",
                        "source": "run",
                        "mtime": os.path.getmtime(fp),
                    }
                )
            )
        except Exception:
            pass
    run_rows = [r for r in rows if r.get("source") == "run"]
    base_rows = [r for r in rows if r.get("source") != "run"]
    run_rows.sort(key=lambda r: float(r.get("mtime") or 0.0))
    for r in run_rows:
        r.pop("mtime", None)
    rows = base_rows + run_rows
    return jsonify(rows)


@app.get("/api/history")
def api_history():
    rows = []
    for fp in glob.glob(os.path.join(OUTPUTS_DIR, "run_*.csv")):
        run_id = Path(fp).stem.replace("run_", "", 1)
        meta = _load_run_meta(run_id)
        try:
            df = pd.read_csv(fp)
            rows.append(
                {
                    "run_id": run_id,
                    "scenario_name": meta.get("scenario_name", run_id),
                    "mode": meta.get("mode", "rule_based_v2x"),
                    "use_rl_signals": bool(meta.get("use_rl_signals", False)),
                    "use_rl_glosa": bool(meta.get("use_rl_glosa", False)),
                    "sim_steps": int(meta.get("sim_steps", int(df["step"].max()) if "step" in df and not df.empty else 0)),
                    "traffic_type": meta.get("traffic_type", "standard"),
                    "ev_count": int(meta.get("ev_count", 0)),
                    "avg_speed": float(df["avg_speed_mps"].mean()) if "avg_speed_mps" in df else 0.0,
                    "avg_wait": float(df["avg_wait_s"].mean()) if "avg_wait_s" in df else 0.0,
                    "collisions": int(pd.to_numeric(df.get("collision_rate", pd.Series([0])), errors="coerce").fillna(0).sum()),
                    "mtime": os.path.getmtime(fp),
                }
            )
        except Exception:
            pass
    rows.sort(key=lambda x: x["mtime"], reverse=True)
    for r in rows:
        r.pop("mtime", None)
    return jsonify(rows)


@app.get("/api/kpi_log")
def api_kpi_log():
    run_id = request.args.get("run_id", "").strip()
    if run_id:
        explicit = _run_csv_path(run_id)
        if os.path.exists(explicit):
            df = pd.read_csv(explicit).tail(60)
            return jsonify(df.where(pd.notna(df), None).to_dict(orient="records"))
    with _state_lock:
        latest_csv = _sim_state.get("latest_csv")
    path = latest_csv or _latest_run_csv() or os.path.join(OUTPUTS_DIR, "v2x_kpi_log.csv")
    if not os.path.exists(path):
        return jsonify([])
    df = pd.read_csv(path).tail(60)
    return jsonify(df.where(pd.notna(df), None).to_dict(orient="records"))


@app.delete("/api/history/<run_id>")
def api_delete(run_id: str):
    for suffix in (".csv", ".meta.json", ".sumocfg", ".rou.xml"):
        fp = os.path.join(OUTPUTS_DIR, f"run_{run_id}{suffix}")
        if os.path.exists(fp):
            os.remove(fp)
    return jsonify({"status": "deleted"})


if __name__ == "__main__":
    print("* V2X Backend running on http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
