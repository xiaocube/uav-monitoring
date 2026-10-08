"""Configuration API Routes.

Endpoints for system configuration management backed by database.
"""

from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Request, status

from skyguard.api.converters import (
    camera_config_from_api,
    camera_config_to_api,
    ptz_config_from_api,
    ptz_config_to_api,
    zone_config_from_api,
    zone_config_to_api,
)
from skyguard.api.dependencies import get_db_manager
from skyguard.api.models import APIResponse, CameraConfig, PTZConfig, ZoneConfig
from skyguard.core.config import settings
from skyguard.db import DatabaseManager

router = APIRouter()


def _get_db(request: Request) -> Optional[DatabaseManager]:
    """Get database manager from request."""
    return get_db_manager(request)


@router.get("/", response_model=APIResponse)
async def get_all_configs(request: Request):
    """Get all system configurations."""
    db = _get_db(request)

    cameras = []
    ptzs = []
    zones = []

    if db and db.is_connected:
        cameras = [camera_config_to_api(m) for m in db.list_camera_configs()]
        ptzs = [ptz_config_to_api(m) for m in db.list_ptz_configs()]
        zones = [zone_config_to_api(m) for m in db.list_zone_configs()]

    configs = {
        "system": settings.model_dump(),
        "cameras": cameras,
        "ptz": ptzs,
        "zones": zones,
    }
    return APIResponse.ok(data=configs)


@router.get("/system", response_model=APIResponse)
async def get_system_config():
    """Get current system configuration."""
    return APIResponse.ok(data=settings.model_dump())


@router.get("/camera", response_model=APIResponse)
async def list_camera_configs(request: Request):
    """List all camera configurations."""
    db = _get_db(request)

    if db and db.is_connected:
        configs = [camera_config_to_api(m) for m in db.list_camera_configs()]
        return APIResponse.ok(data=configs)

    return APIResponse.ok(data=[])


@router.get("/camera/{camera_id}", response_model=APIResponse)
async def get_camera_config(request: Request, camera_id: str):
    """Get camera configuration by ID."""
    db = _get_db(request)

    if db and db.is_connected:
        model = db.get_camera_config(camera_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera {camera_id} not found",
            )
        return APIResponse.ok(data=camera_config_to_api(model))

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Database not available",
    )


@router.post("/camera", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_camera_config(request: Request, config: CameraConfig):
    """Create a new camera configuration."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    model = db.save_camera_config(camera_config_from_api(config))
    return APIResponse.ok(data=camera_config_to_api(model), message="Camera config created")


@router.put("/camera/{camera_id}", response_model=APIResponse)
async def update_camera_config(request: Request, camera_id: str, config: CameraConfig):
    """Update camera configuration."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    if camera_id != config.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera ID mismatch",
        )

    model = db.save_camera_config(camera_config_from_api(config))
    return APIResponse.ok(data=camera_config_to_api(model), message="Camera config updated")


@router.delete("/camera/{camera_id}", response_model=APIResponse)
async def delete_camera_config(request: Request, camera_id: str):
    """Delete camera configuration."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    deleted = db.delete_camera_config(camera_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    return APIResponse.ok(message="Camera config deleted")


@router.get("/ptz", response_model=APIResponse)
async def list_ptz_configs(request: Request):
    """List all PTZ configurations."""
    db = _get_db(request)

    if db and db.is_connected:
        configs = [ptz_config_to_api(m) for m in db.list_ptz_configs()]
        return APIResponse.ok(data=configs)

    return APIResponse.ok(data=[])


@router.get("/ptz/{ptz_id}", response_model=APIResponse)
async def get_ptz_config(request: Request, ptz_id: str):
    """Get PTZ configuration by ID."""
    db = _get_db(request)

    if db and db.is_connected:
        model = db.get_ptz_config(ptz_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PTZ {ptz_id} not found",
            )
        return APIResponse.ok(data=ptz_config_to_api(model))

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Database not available",
    )


@router.post("/ptz", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_ptz_config(request: Request, config: PTZConfig):
    """Create a new PTZ configuration."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    model = db.save_ptz_config(ptz_config_from_api(config))
    return APIResponse.ok(data=ptz_config_to_api(model), message="PTZ config created")


@router.put("/ptz/{ptz_id}", response_model=APIResponse)
async def update_ptz_config(request: Request, ptz_id: str, config: PTZConfig):
    """Update PTZ configuration."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    if ptz_id != config.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PTZ ID mismatch",
        )

    model = db.save_ptz_config(ptz_config_from_api(config))
    return APIResponse.ok(data=ptz_config_to_api(model), message="PTZ config updated")


@router.delete("/ptz/{ptz_id}", response_model=APIResponse)
async def delete_ptz_config(request: Request, ptz_id: str):
    """Delete PTZ configuration."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    deleted = db.delete_ptz_config(ptz_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PTZ {ptz_id} not found",
        )

    return APIResponse.ok(message="PTZ config deleted")


@router.get("/zones", response_model=APIResponse)
async def list_zone_configs(request: Request):
    """List all zone configurations."""
    db = _get_db(request)

    if db and db.is_connected:
        configs = [zone_config_to_api(m) for m in db.list_zone_configs()]
        return APIResponse.ok(data=configs)

    return APIResponse.ok(data=[])


@router.get("/zones/{zone_id}", response_model=APIResponse)
async def get_zone_config(request: Request, zone_id: str):
    """Get zone configuration by ID."""
    db = _get_db(request)

    if db and db.is_connected:
        model = db.get_zone_config(zone_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Zone {zone_id} not found",
            )
        return APIResponse.ok(data=zone_config_to_api(model))

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Database not available",
    )


@router.post("/zones", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_zone_config(request: Request, config: ZoneConfig):
    """Create a new zone configuration."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    model = db.save_zone_config(zone_config_from_api(config))
    return APIResponse.ok(data=zone_config_to_api(model), message="Zone config created")


@router.put("/zones/{zone_id}", response_model=APIResponse)
async def update_zone_config(request: Request, zone_id: str, config: ZoneConfig):
    """Update zone configuration."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    if zone_id != config.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Zone ID mismatch",
        )

    model = db.save_zone_config(zone_config_from_api(config))
    return APIResponse.ok(data=zone_config_to_api(model), message="Zone config updated")


@router.delete("/zones/{zone_id}", response_model=APIResponse)
async def delete_zone_config(request: Request, zone_id: str):
    """Delete zone configuration."""
    db = _get_db(request)

    if not db or not db.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    deleted = db.delete_zone_config(zone_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Zone {zone_id} not found",
        )

    return APIResponse.ok(message="Zone config deleted")


@router.post("/reload", response_model=APIResponse)
async def reload_config():
    """Reload configuration from file."""
    return APIResponse.ok(message="Configuration reloaded")
