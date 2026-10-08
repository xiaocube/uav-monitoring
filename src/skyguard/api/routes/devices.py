"""Devices API Routes.

CRUD endpoints for device management backed by database.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from skyguard.api.converters import device_from_api, device_to_api
from skyguard.api.dependencies import get_db_manager
from skyguard.api.models import (
    APIResponse,
    DeviceInfo,
    DeviceStatus,
    DeviceType,
)
from skyguard.db import DatabaseManager
from skyguard.db.manager import create_uuid

router = APIRouter()


def _get_db(request: Request) -> Optional[DatabaseManager]:
    """Get database manager from request."""
    return get_db_manager(request)


@router.get("/", response_model=APIResponse)
async def list_devices(
    request: Request,
    type: Optional[DeviceType] = None,
    status: Optional[DeviceStatus] = None,
):
    """List all devices."""
    db = _get_db(request)
    
    if db and db.is_connected:
        models = db.list_devices(status=status.value if status else None)
        devices = [device_to_api(m) for m in models]
        if type:
            devices = [d for d in devices if d.type == type]
        return APIResponse.ok(data=devices)
    
    return APIResponse.ok(data=[])


@router.get("/{device_id}", response_model=APIResponse)
async def get_device(request: Request, device_id: str):
    """Get device by ID."""
    db = _get_db(request)
    
    if db and db.is_connected:
        model = db.get_device(device_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device {device_id} not found",
            )
        return APIResponse.ok(data=device_to_api(model))
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Device {device_id} not found",
    )


@router.post("/", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_device(request: Request, device: DeviceInfo):
    """Create a new device."""
    db = _get_db(request)
    
    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )
    
    existing = db.get_device(device.id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Device {device.id} already exists",
        )
    
    model = db.add_device(device_from_api(device))
    return APIResponse.ok(data=device_to_api(model), message="Device created")


@router.put("/{device_id}", response_model=APIResponse)
async def update_device(request: Request, device_id: str, device: DeviceInfo):
    """Update an existing device."""
    db = _get_db(request)
    
    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )
    
    existing = db.get_device(device_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device {device_id} not found",
        )
    
    updates = device_from_api(device)
    updates["id"] = device_id
    model = db.update_device(device_id, updates)
    return APIResponse.ok(data=device_to_api(model), message="Device updated")


@router.delete("/{device_id}", response_model=APIResponse)
async def delete_device(request: Request, device_id: str):
    """Delete a device."""
    db = _get_db(request)
    
    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )
    
    deleted = db.delete_device(device_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device {device_id} not found",
        )
    
    return APIResponse.ok(message="Device deleted")


@router.post("/{device_id}/connect", response_model=APIResponse)
async def connect_device(request: Request, device_id: str):
    """Connect to a device."""
    db = _get_db(request)
    
    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )
    
    device = db.get_device(device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device {device_id} not found",
        )
    
    updated = db.update_device(device_id, {
        "status": DeviceStatus.CONNECTING.value,
        "last_seen": datetime.utcnow(),
    })
    
    # Simulate successful connection
    updated = db.update_device(device_id, {
        "status": DeviceStatus.ONLINE.value,
        "last_seen": datetime.utcnow(),
    })
    
    return APIResponse.ok(data=device_to_api(updated), message="Device connected")


@router.post("/{device_id}/disconnect", response_model=APIResponse)
async def disconnect_device(request: Request, device_id: str):
    """Disconnect from a device."""
    db = _get_db(request)
    
    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )
    
    device = db.get_device(device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device {device_id} not found",
        )
    
    updated = db.update_device(device_id, {
        "status": DeviceStatus.OFFLINE.value,
    })
    
    return APIResponse.ok(data=device_to_api(updated), message="Device disconnected")


@router.get("/cameras/list", response_model=APIResponse)
async def list_cameras(request: Request):
    """List all cameras."""
    db = _get_db(request)
    
    if db and db.is_connected:
        models = db.list_devices()
        cameras = [device_to_api(m) for m in models if m.type == DeviceType.CAMERA.value]
        return APIResponse.ok(data=cameras)
    
    return APIResponse.ok(data=[])


@router.get("/ptz/list", response_model=APIResponse)
async def list_ptz_devices(request: Request):
    """List all PTZ devices."""
    db = _get_db(request)
    
    if db and db.is_connected:
        models = db.list_devices()
        ptzs = [device_to_api(m) for m in models if m.type == DeviceType.PTZ.value]
        return APIResponse.ok(data=ptzs)
    
    return APIResponse.ok(data=[])
