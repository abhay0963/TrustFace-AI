"""
main.py
========
FastAPI application entry point for TrustFace AI.

Run with (from the backend/ folder):
    uvicorn main:app --reload

Or simply:
    python main.py

ROUTES
------
Pages (return HTML):
    GET  /                -> dashboard
    GET  /register         -> registration page
    GET  /attendance       -> live recognition / attendance page
    GET  /users             -> manage registered users
    GET  /logs               -> recognition logs / audit trail

API (return JSON):
    POST /api/register       -> register a new user from a captured frame
    POST /api/recognize       -> run recognition on a captured frame
    GET  /api/users            -> list registered users
    DELETE /api/users/{id}      -> remove a user
    GET  /api/dashboard-summary   -> today's attendance + stats for charts
    GET  /api/logs                 -> recent recognition logs
    GET  /api/history                -> full attendance history
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import database as db
import recognition_service
import presence
import ml_trust_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(title="TrustFace AI", description="Privacy-Preserving Intelligent Presence Verification System")

app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(FRONTEND_DIR, "templates"))

db.init_db()


# -----------------------------------------------------------------------
# Request / response schemas
# -----------------------------------------------------------------------
class RegisterRequest(BaseModel):
    name: str
    external_id: str
    image: str  # base64 data URL


class RecognizeRequest(BaseModel):
    image: str  # base64 data URL


class LabelRequest(BaseModel):
    label: str  # "accept" | "retry" | "reject"


class TrainRequest(BaseModel):
    model_type: str = "random_forest"  # "random_forest" | "decision_tree"


# -----------------------------------------------------------------------
# PAGE ROUTES
# -----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "active": "dashboard"})


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "active": "register"})


@app.get("/attendance", response_class=HTMLResponse)
def attendance_page(request: Request):
    return templates.TemplateResponse("attendance.html", {"request": request, "active": "attendance"})


@app.get("/users", response_class=HTMLResponse)
def users_page(request: Request):
    users = db.get_all_users()
    return templates.TemplateResponse("users.html", {"request": request, "active": "users", "users": users})


@app.get("/logs", response_class=HTMLResponse)
def logs_page(request: Request):
    return templates.TemplateResponse("logs.html", {"request": request, "active": "logs"})


# -----------------------------------------------------------------------
# API ROUTES
# -----------------------------------------------------------------------
@app.post("/api/register")
def api_register(payload: RegisterRequest):
    image_bgr = recognition_service.decode_base64_image(payload.image)
    if image_bgr is None:
        return JSONResponse({"success": False, "error": "Could not decode image."}, status_code=400)

    name = payload.name.strip()
    external_id = payload.external_id.strip()
    if not name or not external_id:
        return JSONResponse({"success": False, "error": "Name and ID are required."}, status_code=400)

    result = recognition_service.register_face(image_bgr, name, external_id)
    status_code = 200 if result.get("success") else 400
    return JSONResponse(result, status_code=status_code)


@app.post("/api/recognize")
def api_recognize(payload: RecognizeRequest):
    image_bgr = recognition_service.decode_base64_image(payload.image)
    if image_bgr is None:
        return JSONResponse({"error": "Could not decode image."}, status_code=400)
    result = recognition_service.recognize_frame(image_bgr)
    return JSONResponse(result)


@app.get("/api/users")
def api_list_users():
    return JSONResponse(db.get_all_users())


@app.delete("/api/users/{user_id}")
def api_delete_user(user_id: int):
    db.delete_user(user_id)
    return JSONResponse({"success": True})


@app.get("/api/dashboard-summary")
def api_dashboard_summary():
    today_attendance = presence.get_today_attendance_enriched()
    total_users = len(db.get_all_users())
    recent_logs = db.get_recent_logs(limit=10)

    present_count = len(today_attendance)
    avg_trust = (
        round(sum(r["avg_trust_score"] for r in today_attendance) / present_count, 1)
        if present_count else 0
    )
    avg_presence_pct = (
        round(sum(r["presence_percentage"] for r in today_attendance) / present_count, 1)
        if present_count else 0
    )

    return JSONResponse({
        "total_registered_users": total_users,
        "present_today": present_count,
        "avg_trust_score": avg_trust,
        "avg_presence_percentage": avg_presence_pct,
        "today_attendance": today_attendance,
        "recent_logs": recent_logs,
    })


@app.get("/api/logs")
def api_logs(limit: int = 50):
    return JSONResponse(db.get_recent_logs(limit=limit))


@app.get("/api/history")
def api_history(limit: int = 200):
    return JSONResponse(presence.get_history_enriched(limit=limit))


# -----------------------------------------------------------------------
# ML TRUST ENGINE ROUTES (Module 3b)
# -----------------------------------------------------------------------
@app.post("/api/logs/{log_id}/label")
def api_label_log(log_id: int, payload: LabelRequest):
    """
    Human-in-the-loop labeling: a reviewer confirms the TRUE correct
    decision for a logged recognition attempt. This is what builds the
    (X, y) dataset the ML Trust Engine trains on - see ml_trust_engine.py.
    """
    if payload.label not in ("accept", "retry", "reject"):
        return JSONResponse({"success": False, "error": "label must be accept, retry, or reject"}, status_code=400)
    db.set_human_label(log_id, payload.label)
    return JSONResponse({"success": True, "status": ml_trust_engine.training_status()})


@app.get("/api/ml-status")
def api_ml_status():
    status = ml_trust_engine.training_status()
    status["metrics"] = ml_trust_engine.load_metrics()
    return JSONResponse(status)


@app.post("/api/train-model")
def api_train_model(payload: TrainRequest):
    """
    Trains the ML Trust Engine on all human-labeled recognition_logs rows
    collected so far. Returns real accuracy/precision/recall/confusion-
    matrix numbers computed on a held-out test split - see
    ml_trust_engine.train_model() for the full methodology.
    """
    result = ml_trust_engine.train_model(model_type=payload.model_type)
    status_code = 200 if result.get("success") else 400
    return JSONResponse(result, status_code=status_code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
