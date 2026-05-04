import asyncio
import json
from pathlib import Path
from typing import Optional
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.inference   import predict, load_models
from api.alert_store import alert_store
from api.simulator   import simulator
from utils.logger    import get_logger

log = get_logger("main_api")

app = FastAPI(
    title      = "Behavioral Security Layer API",
    description= "Real-time threat detection with explainable AI",
    version    = "1.0.0",
)

# Allow React dev server to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5173", "http://localhost:3000"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        log.info(f"WS connected — {len(self.active)} active")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        log.info(f"WS disconnected — {len(self.active)} active")

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()


# ── Startup ──────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    log.info("Loading models...")
    load_models()
    log.info("API ready")


# ── REST Routes ──────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name"   : "Behavioral Security Layer",
        "version": "1.0.0",
        "status" : "running",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "alerts": alert_store.count()}


@app.post("/ingest")
async def ingest_event(event: dict):
    """
    Accept a raw network/endpoint event and return an alert.
    This is the main inference endpoint.

    POST /ingest
    Body: {feature1: value1, feature2: value2, ...}
    """
    alert = predict(event)
    alert_store.add(alert)

    # Broadcast to all connected WebSocket clients
    await manager.broadcast({
        "type" : "new_alert",
        "alert": alert,
    })

    return alert


@app.get("/alerts")
async def get_alerts(n: int = 20):
    """Get last n alerts."""
    return {
        "alerts": alert_store.get_recent(n),
        "stats" : alert_store.stats(),
    }


@app.get("/alerts/stats")
async def get_stats():
    """Get alert statistics."""
    return alert_store.stats()


@app.post("/alerts/{alert_id}/action")
async def analyst_action(alert_id: str, action: dict):
    """
    Analyst feedback endpoint.
    action: {"decision": "confirm" | "dismiss" | "escalate"}
    """
    decision = action.get("decision", "unknown")
    log.info(f"Analyst action: {alert_id} → {decision}")
    return {
        "alert_id": alert_id,
        "decision": decision,
        "status"  : "recorded",
    }


@app.get("/metrics")
async def get_metrics():
    """Return model performance metrics."""
    return {
        "models": [
            {"model": "Network IDS",      "f1": 0.9936,
             "auc": 0.9998, "fpr": 0.0029},
            {"model": "LightGBM",         "f1": 0.7893,
             "auc": 0.9595, "fpr": 0.0381},
            {"model": "Ensemble",         "f1": 0.7787,
             "auc": 0.9156, "fpr": 0.0299},
            {"model": "CERT Insider",     "f1": 0.7496,
             "auc": 0.9908, "fpr": 0.0561},
            {"model": "GenAI Detector",   "f1": 0.7467,
             "auc": 0.9996, "fpr": 0.0520},
            {"model": "Isolation Forest", "f1": 0.2854,
             "auc": 0.8421, "fpr": 0.0470},
        ],
        "zero_day_detection" : 0.996,
        "federated_gap"      : 0.0196,
        "privacy_epsilon"    : 0.119,
    }


@app.delete("/alerts")
async def clear_alerts():
    """Clear all alerts — for demo reset."""
    alert_store.clear()
    return {"status": "cleared"}


# ── WebSocket ────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for live alert streaming.
    Dashboard connects here to receive alerts in real time.
    """
    await manager.connect(websocket)

    # Send existing alerts on connect
    existing = alert_store.get_recent(20)
    await websocket.send_json({
        "type"  : "history",
        "alerts": existing,
        "stats" : alert_store.stats(),
    })

    try:
        while True:
            # Keep connection alive — wait for client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ── Simulator Routes ─────────────────────────────────────
@app.post("/simulator/start")
async def start_simulator(
    background_tasks: BackgroundTasks,
    speed: float = 2.0,
    max_events: Optional[int] = 200,
):
    """
    Start replaying test data as live events.
    speed      : events per second
    max_events : stop after this many (None = run forever)
    """
    if simulator.running:
        return {"status": "already running"}

    simulator.speed = speed
    simulator.load()

    async def run():
        async def on_event(event: dict):
            alert = predict(event)
            if alert["ensemble_score"] > 0.20:
                alert_store.add(alert)
                await manager.broadcast({
                    "type" : "new_alert",
                    "alert": alert,
                })

        await simulator.stream(on_event, max_events=max_events)

    background_tasks.add_task(run)
    return {
        "status"    : "started",
        "speed"     : speed,
        "max_events": max_events,
    }


@app.post("/simulator/stop")
async def stop_simulator():
    """Stop the data simulator."""
    simulator.stop()
    return {"status": "stopped"}


@app.get("/simulator/status")
async def simulator_status():
    return {
        "running": simulator.running,
        "speed"  : simulator.speed,
    }