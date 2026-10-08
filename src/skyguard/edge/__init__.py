"""SkyGuard Edge Deployment Module (Sprint 11).

Provides Jetson edge device support:
  * Hardware detection and capability probing
  * Power mode management (nvpmodel / jetson_clocks)
  * TensorRT engine build and optimization (FP16 / INT8)
  * Edge device health monitoring (GPU / thermal / memory)
  * Deployment orchestration with graceful lifecycle

Designed to degrade gracefully on non-Jetson hosts so unit tests
can run on macOS / Linux CI without real hardware.
"""
from skyguard.edge.hardware import JetsonHardware, detect_jetson
from skyguard.edge.power import PowerManager, PowerMode
from skyguard.edge.monitor import EdgeMonitor, EdgeHealth
from skyguard.edge.deploy import EdgeDeploymentManager, DeploymentConfig

__all__ = [
    "JetsonHardware",
    "detect_jetson",
    "PowerManager",
    "PowerMode",
    "EdgeMonitor",
    "EdgeHealth",
    "EdgeDeploymentManager",
    "DeploymentConfig",
]
