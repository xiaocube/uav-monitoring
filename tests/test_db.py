"""Database module tests."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from skyguard.db.models import (
    AlertModel,
    Base,
    CameraConfigModel,
    DetectionModel,
    DeviceModel,
    PTZConfigModel,
    TrackModel,
    ZoneConfigModel,
)
from skyguard.db.manager import DatabaseManager, create_uuid
from skyguard.db.types import DatabaseConfig, TimescaleConfig


class TestDatabaseTypes:
    """Test database configuration types."""
    
    def test_database_config_default(self):
        """Test default database config."""
        cfg = DatabaseConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.database == "skyguard"
        assert cfg.username == "skyguard"
        assert cfg.password == ""
    
    def test_database_config_dsn(self):
        """Test DSN generation."""
        cfg = DatabaseConfig(
            host="192.168.1.100",
            port=5433,
            database="testdb",
            username="user",
            password="pass",
        )
        dsn = cfg.dsn
        assert "192.168.1.100" in dsn
        assert "5433" in dsn
        assert "testdb" in dsn
        assert "user" in dsn
        assert "pass" in dsn
    
    def test_timescale_config_default(self):
        """Test default Timescale config."""
        cfg = TimescaleConfig()
        assert cfg.enabled is True
        assert cfg.chunk_interval == "1 day"
        assert cfg.compression_enabled is False


class TestDatabaseModels:
    """Test database models."""
    
    def test_device_model(self):
        """Test DeviceModel creation."""
        device = DeviceModel(
            id="test-dev-001",
            name="Test Camera",
            type="camera",
            status="online",
            host="192.168.1.10",
            port=554,
            config={"fps": 30},
        )
        assert device.id == "test-dev-001"
        assert device.name == "Test Camera"
        assert device.type == "camera"
        assert device.status == "online"
        assert device.host == "192.168.1.10"
        assert device.port == 554
        assert device.config == {"fps": 30}
    
    def test_detection_model(self):
        """Test DetectionModel creation."""
        detection = DetectionModel(
            id="det-001",
            frame_id="frame_001",
            device_id="cam-001",
            class_id=0,
            class_name="drone",
            confidence=0.95,
            bbox_x1=100.0,
            bbox_y1=200.0,
            bbox_x2=300.0,
            bbox_y2=400.0,
            frame_width=1920,
            frame_height=1080,
            fps=30.0,
        )
        assert detection.id == "det-001"
        assert detection.class_name == "drone"
        assert detection.confidence == 0.95
        assert detection.bbox_x2 - detection.bbox_x1 == 200.0
    
    def test_track_model(self):
        """Test TrackModel creation."""
        track = TrackModel(
            id="track-001",
            track_id=1,
            device_id="cam-001",
            class_id=0,
            class_name="drone",
            confidence=0.92,
            bbox_x1=150.0,
            bbox_y1=250.0,
            bbox_x2=350.0,
            bbox_y2=450.0,
            velocity_x=10.5,
            velocity_y=-5.3,
            age=15.0,
            status="active",
        )
        assert track.track_id == 1
        assert track.velocity_x == 10.5
        assert track.velocity_y == -5.3
        assert track.age == 15.0
    
    def test_alert_model(self):
        """Test AlertModel creation."""
        alert = AlertModel(
            id="alert-001",
            type="drone_detected",
            level="warning",
            message="Drone detected in restricted area",
            device_id="cam-001",
            track_id=1,
            acknowledged=False,
            alert_metadata={"coords": [116.4, 39.9]},
        )
        assert alert.type == "drone_detected"
        assert alert.level == "warning"
        assert alert.acknowledged is False
        assert alert.alert_metadata == {"coords": [116.4, 39.9]}
    
    def test_camera_config_model(self):
        """Test CameraConfigModel creation."""
        config = CameraConfigModel(
            id="cam-config-001",
            name="Main Camera",
            host="192.168.1.20",
            port=554,
            username="admin",
            password="password",
            rtsp_url="rtsp://192.168.1.20:554/stream",
            enabled=True,
            fps=30,
            resolution="1920x1080",
        )
        assert config.name == "Main Camera"
        assert config.rtsp_url == "rtsp://192.168.1.20:554/stream"
        assert config.enabled is True
    
    def test_ptz_config_model(self):
        """Test PTZConfigModel creation."""
        config = PTZConfigModel(
            id="ptz-config-001",
            name="PTZ Camera",
            host="192.168.1.30",
            port=80,
            username="admin",
            password="secret",
            enabled=True,
            auto_tracking=True,
        )
        assert config.name == "PTZ Camera"
        assert config.auto_tracking is True
    
    def test_zone_config_model(self):
        """Test ZoneConfigModel creation."""
        config = ZoneConfigModel(
            id="zone-config-001",
            name="Restricted Zone",
            coordinates=[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
            enabled=True,
            alert_on_entry=True,
            alert_on_exit=False,
        )
        assert config.name == "Restricted Zone"
        assert len(config.coordinates) == 4


class TestCreateUUID:
    """Test UUID generation."""
    
    def test_create_uuid_format(self):
        """Test UUID format."""
        uuid_str = create_uuid()
        assert len(uuid_str) == 36
        assert uuid_str.count("-") == 4
    
    def test_create_uuid_unique(self):
        """Test UUID uniqueness."""
        uuids = [create_uuid() for _ in range(100)]
        assert len(uuids) == len(set(uuids))


class TestDatabaseManagerInMemory:
    """Test DatabaseManager with in-memory SQLite."""
    
    @pytest.fixture
    def db_manager(self):
        """Create an in-memory database manager."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        db = DatabaseManager()
        db.engine = engine
        db.SessionLocal = SessionLocal
        db._connected = True
        
        yield db
    
    def test_add_device(self, db_manager):
        """Test adding a device."""
        device = db_manager.add_device({
            "id": "test-dev-001",
            "name": "Test Device",
            "type": "camera",
            "status": "online",
            "host": "192.168.1.10",
            "port": 554,
        })
        assert device.id == "test-dev-001"
        assert device.name == "Test Device"
    
    def test_get_device(self, db_manager):
        """Test getting a device."""
        db_manager.add_device({
            "id": "test-dev-002",
            "name": "Test Device 2",
            "type": "ptz",
            "status": "online",
            "host": "192.168.1.11",
            "port": 80,
        })
        
        device = db_manager.get_device("test-dev-002")
        assert device is not None
        assert device.name == "Test Device 2"
    
    def test_get_device_not_found(self, db_manager):
        """Test getting non-existent device."""
        device = db_manager.get_device("nonexistent")
        assert device is None
    
    def test_list_devices(self, db_manager):
        """Test listing devices."""
        db_manager.add_device({
            "id": "dev-001",
            "name": "Device 1",
            "type": "camera",
            "status": "online",
            "host": "192.168.1.1",
            "port": 554,
        })
        db_manager.add_device({
            "id": "dev-002",
            "name": "Device 2",
            "type": "ptz",
            "status": "offline",
            "host": "192.168.1.2",
            "port": 80,
        })
        
        devices = db_manager.list_devices()
        assert len(devices) == 2
        
        online_devices = db_manager.list_devices(status="online")
        assert len(online_devices) == 1
        assert online_devices[0].id == "dev-001"
    
    def test_update_device(self, db_manager):
        """Test updating a device."""
        db_manager.add_device({
            "id": "test-dev-003",
            "name": "Original Name",
            "type": "camera",
            "status": "online",
            "host": "192.168.1.12",
            "port": 554,
        })
        
        updated = db_manager.update_device("test-dev-003", {
            "name": "Updated Name",
            "status": "offline",
        })
        
        assert updated is not None
        assert updated.name == "Updated Name"
        assert updated.status == "offline"
    
    def test_update_device_not_found(self, db_manager):
        """Test updating non-existent device."""
        updated = db_manager.update_device("nonexistent", {"name": "New Name"})
        assert updated is None
    
    def test_delete_device(self, db_manager):
        """Test deleting a device."""
        db_manager.add_device({
            "id": "test-dev-004",
            "name": "To Delete",
            "type": "camera",
            "status": "online",
            "host": "192.168.1.13",
            "port": 554,
        })
        
        result = db_manager.delete_device("test-dev-004")
        assert result is True
        
        device = db_manager.get_device("test-dev-004")
        assert device is None
    
    def test_delete_device_not_found(self, db_manager):
        """Test deleting non-existent device."""
        result = db_manager.delete_device("nonexistent")
        assert result is False
    
    def test_add_detection(self, db_manager):
        """Test adding a detection."""
        db_manager.add_device({
            "id": "det-cam-001",
            "name": "Detection Camera",
            "type": "camera",
            "status": "online",
            "host": "192.168.1.20",
            "port": 554,
        })
        
        detection = db_manager.add_detection({
            "id": "det-001",
            "frame_id": "frame_001",
            "device_id": "det-cam-001",
            "class_id": 0,
            "class_name": "drone",
            "confidence": 0.95,
            "bbox_x1": 100.0,
            "bbox_y1": 200.0,
            "bbox_x2": 300.0,
            "bbox_y2": 400.0,
            "frame_width": 1920,
            "frame_height": 1080,
        })
        
        assert detection.id == "det-001"
        assert detection.class_name == "drone"
    
    def test_get_detections(self, db_manager):
        """Test getting detections."""
        db_manager.add_device({
            "id": "det-cam-002",
            "name": "Camera",
            "type": "camera",
            "status": "online",
            "host": "192.168.1.21",
            "port": 554,
        })
        
        for i in range(5):
            db_manager.add_detection({
                "id": f"det-{i:03d}",
                "frame_id": f"frame_{i:03d}",
                "device_id": "det-cam-002",
                "class_id": i % 2,
                "class_name": "drone" if i % 2 == 0 else "bird",
                "confidence": 0.8 + i * 0.02,
                "bbox_x1": 100.0,
                "bbox_y1": 200.0,
                "bbox_x2": 300.0,
                "bbox_y2": 400.0,
                "frame_width": 1920,
                "frame_height": 1080,
            })
        
        detections = db_manager.get_detections(device_id="det-cam-002")
        assert len(detections) == 5
        
        drone_detections = db_manager.get_detections(device_id="det-cam-002", class_id=0)
        assert len(drone_detections) == 3
    
    def test_add_track(self, db_manager):
        """Test adding a track."""
        db_manager.add_device({
            "id": "track-cam-001",
            "name": "Track Camera",
            "type": "camera",
            "status": "online",
            "host": "192.168.1.30",
            "port": 554,
        })
        
        track = db_manager.add_track({
            "id": "track-001",
            "track_id": 1,
            "device_id": "track-cam-001",
            "class_id": 0,
            "class_name": "drone",
            "confidence": 0.92,
            "bbox_x1": 150.0,
            "bbox_y1": 250.0,
            "bbox_x2": 350.0,
            "bbox_y2": 450.0,
            "velocity_x": 10.5,
            "velocity_y": -5.3,
            "age": 15.0,
        })
        
        assert track.track_id == 1
        assert track.velocity_x == 10.5
    
    def test_add_alert(self, db_manager):
        """Test adding an alert."""
        alert = db_manager.add_alert({
            "id": "alert-001",
            "type": "drone_detected",
            "level": "warning",
            "message": "Drone detected",
            "device_id": "cam-001",
            "track_id": 1,
            "acknowledged": False,
        })
        
        assert alert.type == "drone_detected"
        assert alert.level == "warning"
        assert alert.acknowledged is False
    
    def test_get_alerts(self, db_manager):
        """Test getting alerts."""
        for i in range(5):
            level = "warning" if i % 2 == 0 else "critical"
            db_manager.add_alert({
                "id": f"alert-{i:03d}",
                "type": "drone_detected",
                "level": level,
                "message": f"Alert {i}",
                "acknowledged": i < 2,
            })
        
        alerts = db_manager.get_alerts()
        assert len(alerts) == 5
        
        unacknowledged = db_manager.get_alerts(acknowledged=False)
        assert len(unacknowledged) == 3
        
        critical = db_manager.get_alerts(level="critical")
        assert len(critical) == 2
    
    def test_acknowledge_alert(self, db_manager):
        """Test acknowledging an alert."""
        db_manager.add_alert({
            "id": "alert-ack-001",
            "type": "drone_detected",
            "level": "warning",
            "message": "Test alert",
            "acknowledged": False,
        })
        
        result = db_manager.acknowledge_alert("alert-ack-001")
        assert result is True
        
        alert = db_manager.get_alerts(acknowledged=True)
        assert len(alert) == 1
    
    def test_save_camera_config(self, db_manager):
        """Test saving camera config."""
        config = db_manager.save_camera_config({
            "id": "cam-config-001",
            "name": "Front Camera",
            "host": "192.168.1.100",
            "port": 554,
            "username": "admin",
            "password": "pass",
            "enabled": True,
        })
        
        assert config.name == "Front Camera"
        
        retrieved = db_manager.get_camera_config("cam-config-001")
        assert retrieved is not None
        assert retrieved.name == "Front Camera"
    
    def test_save_ptz_config(self, db_manager):
        """Test saving PTZ config."""
        config = db_manager.save_ptz_config({
            "id": "ptz-config-001",
            "name": "PTZ 1",
            "host": "192.168.1.101",
            "port": 80,
            "auto_tracking": True,
        })
        
        assert config.auto_tracking is True
        
        retrieved = db_manager.get_ptz_config("ptz-config-001")
        assert retrieved is not None
    
    def test_save_zone_config(self, db_manager):
        """Test saving zone config."""
        config = db_manager.save_zone_config({
            "id": "zone-config-001",
            "name": "Zone A",
            "coordinates": [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]],
            "enabled": True,
        })
        
        assert config.name == "Zone A"
        
        retrieved = db_manager.get_zone_config("zone-config-001")
        assert retrieved is not None
    
    def test_get_detection_statistics(self, db_manager):
        """Test getting detection statistics."""
        db_manager.add_device({
            "id": "stats-cam-001",
            "name": "Stats Camera",
            "type": "camera",
            "status": "online",
            "host": "192.168.1.50",
            "port": 554,
        })
        
        for i in range(10):
            db_manager.add_detection({
                "id": f"stat-det-{i:03d}",
                "frame_id": f"frame_{i:03d}",
                "device_id": "stats-cam-001",
                "class_id": 0,
                "class_name": "drone",
                "confidence": 0.9,
                "bbox_x1": 100.0,
                "bbox_y1": 200.0,
                "bbox_x2": 300.0,
                "bbox_y2": 400.0,
                "frame_width": 1920,
                "frame_height": 1080,
            })
        
        stats = db_manager.get_detection_statistics(device_id="stats-cam-001", hours=1)
        assert stats["total_detections"] == 10
        assert len(stats["by_class"]) == 1
    
    def test_get_alert_statistics(self, db_manager):
        """Test getting alert statistics."""
        for i in range(5):
            db_manager.add_alert({
                "id": f"stat-alert-{i:03d}",
                "type": "drone_detected",
                "level": "warning",
                "message": f"Stat alert {i}",
            })
        
        stats = db_manager.get_alert_statistics(hours=1)
        assert stats["total_alerts"] == 5
    
    def test_cleanup_old_data(self, db_manager):
        """Test cleaning up old data."""
        db_manager.add_device({
            "id": "cleanup-cam-001",
            "name": "Cleanup Camera",
            "type": "camera",
            "status": "online",
            "host": "192.168.1.60",
            "port": 554,
        })
        
        for i in range(10):
            db_manager.add_detection({
                "id": f"cleanup-det-{i:03d}",
                "frame_id": f"frame_{i:03d}",
                "device_id": "cleanup-cam-001",
                "class_id": 0,
                "class_name": "drone",
                "confidence": 0.9,
                "bbox_x1": 100.0,
                "bbox_y1": 200.0,
                "bbox_x2": 300.0,
                "bbox_y2": 400.0,
                "frame_width": 1920,
                "frame_height": 1080,
            })
            db_manager.add_track({
                "id": f"cleanup-track-{i:03d}",
                "track_id": 1,
                "device_id": "cleanup-cam-001",
                "class_id": 0,
                "class_name": "drone",
                "confidence": 0.9,
                "bbox_x1": 100.0,
                "bbox_y1": 200.0,
                "bbox_x2": 300.0,
                "bbox_y2": 400.0,
            })
        
        result = db_manager.cleanup_old_data(days_to_keep=0)
        assert result["detections_deleted"] == 10
        assert result["tracks_deleted"] == 10