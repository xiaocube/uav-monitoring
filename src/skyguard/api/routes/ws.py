"""WebSocket Routes.

Real-time streaming endpoints for detection, tracking, and alerts.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from skyguard.api.models import WSMessage, WSMessageType

router = APIRouter()

# Active WebSocket connections
_connections: Dict[str, List[WebSocket]] = {
    "detection": [],
    "tracking": [],
    "alert": [],
    "status": [],
}


class ConnectionManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {
            "detection": [],
            "tracking": [],
            "alert": [],
            "status": [],
        }
    
    async def connect(self, websocket: WebSocket, channel: str):
        """Connect to a channel."""
        await websocket.accept()
        if channel not in self._connections:
            self._connections[channel] = []
        self._connections[channel].append(websocket)
    
    def disconnect(self, websocket: WebSocket, channel: str):
        """Disconnect from a channel."""
        if channel in self._connections and websocket in self._connections[channel]:
            self._connections[channel].remove(websocket)
    
    async def broadcast(self, channel: str, message: WSMessage):
        """Broadcast message to all connected clients in a channel."""
        if channel not in self._connections:
            return
        
        data = message.model_dump_json()
        disconnected = []
        
        for websocket in self._connections[channel]:
            try:
                await websocket.send_text(data)
            except (WebSocketDisconnect, Exception):
                disconnected.append((websocket, channel))
        
        for ws, ch in disconnected:
            self.disconnect(ws, ch)
    
    def get_connection_count(self, channel: str) -> int:
        """Get number of connections in a channel."""
        return len(self._connections.get(channel, []))


_manager = ConnectionManager()


@router.websocket("/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    """WebSocket endpoint for real-time data.
    
    Channels:
    - detection: Real-time detection results
    - tracking: Real-time tracking results
    - alert: Real-time alerts
    - status: System status updates
    """
    await _manager.connect(websocket, channel)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client messages
            if message.get("type") == "ping":
                await websocket.send_text(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()})
                )
    
    except WebSocketDisconnect:
        _manager.disconnect(websocket, channel)


@router.get("/connections", response_model=Dict[str, int])
async def get_connection_counts():
    """Get number of active WebSocket connections per channel."""
    return {
        channel: _manager.get_connection_count(channel)
        for channel in ["detection", "tracking", "alert", "status"]
    }


@router.post("/broadcast/{channel}")
async def broadcast_message(channel: str, message: dict):
    """Broadcast a message to all clients in a channel."""
    ws_message = WSMessage(
        type=WSMessageType(channel),
        timestamp=datetime.now(),
        data=message,
    )
    await _manager.broadcast(channel, ws_message)
    return {"success": True, "channel": channel, "connections": _manager.get_connection_count(channel)}


# ========== Helper functions for broadcasting ==========

async def broadcast_detection(data: dict):
    """Broadcast detection data."""
    message = WSMessage(
        type=WSMessageType.DETECTION,
        timestamp=datetime.now(),
        data=data,
    )
    await _manager.broadcast("detection", message)


async def broadcast_tracking(data: dict):
    """Broadcast tracking data."""
    message = WSMessage(
        type=WSMessageType.TRACKING,
        timestamp=datetime.now(),
        data=data,
    )
    await _manager.broadcast("tracking", message)


async def broadcast_alert(data: dict):
    """Broadcast alert data."""
    message = WSMessage(
        type=WSMessageType.ALERT,
        timestamp=datetime.now(),
        data=data,
    )
    await _manager.broadcast("alert", message)


async def broadcast_status(data: dict):
    """Broadcast status data."""
    message = WSMessage(
        type=WSMessageType.STATUS,
        timestamp=datetime.now(),
        data=data,
    )
    await _manager.broadcast("status", message)