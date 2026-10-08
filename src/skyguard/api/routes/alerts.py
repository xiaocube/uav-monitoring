"""Alerts API Routes.

Endpoints for alert management backed by database.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status

from skyguard.api.converters import alert_from_api, alert_to_api
from skyguard.api.dependencies import get_db_manager
from skyguard.api.models import (
    APIResponse,
    AlertInfo,
    AlertLevel,
    AlertType,
)
from skyguard.db import DatabaseManager
from skyguard.db.manager import create_uuid

router = APIRouter()


def _get_db(request: Request) -> Optional[DatabaseManager]:
    """Get database manager from request."""
    return get_db_manager(request)


@router.get("/count/total", response_model=APIResponse)
async def get_alert_count(
    request: Request,
    type: Optional[AlertType] = None,
    level: Optional[AlertLevel] = None,
    acknowledged: Optional[bool] = None,
):
    """Get alert count with optional filtering."""
    db = _get_db(request)

    if db and db.is_connected:
        models = db.get_alerts(
            type=type.value if type else None,
            level=level.value if level else None,
            acknowledged=acknowledged,
            limit=100000,
        )
        return APIResponse.ok(data={"count": len(models)})

    return APIResponse.ok(data={"count": 0})


@router.get("/unacknowledged/list", response_model=APIResponse)
async def get_unacknowledged_alerts(request: Request):
    """Get all unacknowledged alerts."""
    db = _get_db(request)

    if db and db.is_connected:
        models = db.get_alerts(acknowledged=False, limit=10000)
        return APIResponse.ok(data=[alert_to_api(m) for m in models])

    return APIResponse.ok(data=[])


@router.get("/", response_model=APIResponse)
async def list_alerts(
    request: Request,
    type: Optional[AlertType] = None,
    level: Optional[AlertLevel] = None,
    acknowledged: Optional[bool] = None,
    device_id: Optional[str] = None,
    limit: int = 100,
):
    """List alerts with optional filtering."""
    db = _get_db(request)

    if db and db.is_connected:
        models = db.get_alerts(
            type=type.value if type else None,
            level=level.value if level else None,
            acknowledged=acknowledged,
            device_id=device_id,
            limit=limit,
        )
        return APIResponse.ok(data=[alert_to_api(m) for m in models])

    return APIResponse.ok(data=[])


@router.get("/{alert_id}", response_model=APIResponse)
async def get_alert(request: Request, alert_id: str):
    """Get alert by ID."""
    db = _get_db(request)

    if db and db.is_connected:
        models = db.get_alerts(limit=10000)
        for model in models:
            if model.id == alert_id:
                return APIResponse.ok(data=alert_to_api(model))

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Alert {alert_id} not found",
    )


@router.post("/", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(request: Request, alert: AlertInfo):
    """Create a new alert."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    alert.id = create_uuid()
    alert.timestamp = datetime.utcnow()
    alert.acknowledged = False

    model = db.add_alert(alert_from_api(alert))
    return APIResponse.ok(data=alert_to_api(model), message="Alert created")


@router.put("/{alert_id}/acknowledge", response_model=APIResponse)
async def acknowledge_alert(request: Request, alert_id: str):
    """Acknowledge an alert."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    success = db.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found",
        )

    return APIResponse.ok(message="Alert acknowledged")


@router.put("/{alert_id}/unacknowledge", response_model=APIResponse)
async def unacknowledge_alert(request: Request, alert_id: str):
    """Unacknowledge an alert."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    model = db.update_alert(alert_id, {"acknowledged": False})
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found",
        )

    return APIResponse.ok(message="Alert unacknowledged")


@router.delete("/{alert_id}", response_model=APIResponse)
async def delete_alert(request: Request, alert_id: str):
    """Delete an alert."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    deleted = db.delete_alert(alert_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found",
        )

    return APIResponse.ok(message="Alert deleted")


@router.delete("/", response_model=APIResponse)
async def delete_all_alerts(request: Request):
    """Delete all alerts."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    models = db.get_alerts(limit=100000)
    for model in models:
        db.delete_alert(model.id)

    return APIResponse.ok(message="All alerts deleted")
