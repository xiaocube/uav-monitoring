"""Database manager for PostgreSQL + TimescaleDB operations."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

from loguru import logger
from sqlalchemy import (
    create_engine,
    func,
    select,
)
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    AlertModel,
    Base,
    CameraConfigModel,
    DetectionModel,
    DeviceModel,
    PTZConfigModel,
    TrackModel,
    ZoneConfigModel,
)
from .types import DatabaseConfig, TimescaleConfig

ModelType = TypeVar("ModelType", bound=Base)


class DatabaseManager:
    """Database manager for SkyGuard."""
    
    def __init__(
        self,
        config: Optional[DatabaseConfig] = None,
        timescale_config: Optional[TimescaleConfig] = None,
    ):
        self.config = config or DatabaseConfig()
        self.timescale_config = timescale_config or TimescaleConfig()
        self.engine = None
        self.SessionLocal = None
        self._connected = False
    
    def connect(self) -> bool:
        """Connect to the database."""
        try:
            self.engine = create_engine(
                self.config.dsn,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                connect_args={"connect_timeout": self.config.connect_timeout},
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )
            
            with self.engine.connect() as conn:
                conn.execute(select(1))
                conn.commit()
            
            self._connected = True
            logger.info(f"Connected to database: {self.config.host}:{self.config.port}/{self.config.database}")
            
            self._setup_timescale()
            
            return True
        except OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            return False
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the database."""
        if self.engine:
            self.engine.dispose()
            self._connected = False
            logger.info("Disconnected from database")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._connected
    
    def create_tables(self) -> None:
        """Create all database tables."""
        if not self.engine:
            raise RuntimeError("Database not connected")
        
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")
    
    def drop_tables(self) -> None:
        """Drop all database tables."""
        if not self.engine:
            raise RuntimeError("Database not connected")
        
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")
    
    def get_session(self) -> Session:
        """Get a database session."""
        if not self.SessionLocal:
            raise RuntimeError("Database not connected")
        return self.SessionLocal()
    
    def _setup_timescale(self) -> None:
        """Setup TimescaleDB hypertables."""
        if not self.timescale_config.enabled:
            return
        
        try:
            with self.get_session() as session:
                session.execute(
                    "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"
                )
                session.commit()
                
                for table_name in ["detections", "tracks"]:
                    try:
                        session.execute(
                            f"""
                            SELECT create_hypertable(
                                '{table_name}',
                                'timestamp',
                                chunk_time_interval => INTERVAL '{self.timescale_config.chunk_interval}'
                            )
                            """
                        )
                        session.commit()
                        logger.info(f"Created hypertable for {table_name}")
                    except SQLAlchemyError:
                        pass
                
                if self.timescale_config.compression_enabled:
                    for table_name in ["detections", "tracks"]:
                        try:
                            session.execute(
                                f"""
                                ALTER TABLE {table_name} SET (
                                    timescaledb.compress,
                                    timescaledb.compress_segmentby = 'device_id',
                                    timescaledb.compress_orderby = 'timestamp DESC'
                                )
                                """
                            )
                            session.commit()
                            session.execute(
                                f"""
                                SELECT add_compression_policy(
                                    '{table_name}',
                                    INTERVAL '{self.timescale_config.compression_interval}'
                                )
                                """
                            )
                            session.commit()
                        except SQLAlchemyError:
                            pass
        except SQLAlchemyError as e:
            logger.warning(f"TimescaleDB setup failed (continuing without): {e}")
    
    def add_device(self, device: Dict[str, Any]) -> DeviceModel:
        """Add a device."""
        with self.get_session() as session:
            model = DeviceModel(**device)
            session.add(model)
            session.commit()
            session.refresh(model)
            return model
    
    def get_device(self, device_id: str) -> Optional[DeviceModel]:
        """Get a device by ID."""
        with self.get_session() as session:
            return session.get(DeviceModel, device_id)
    
    def list_devices(self, status: Optional[str] = None) -> List[DeviceModel]:
        """List devices."""
        with self.get_session() as session:
            query = select(DeviceModel)
            if status:
                query = query.where(DeviceModel.status == status)
            return list(session.execute(query).scalars().all())
    
    def update_device(self, device_id: str, updates: Dict[str, Any]) -> Optional[DeviceModel]:
        """Update a device."""
        with self.get_session() as session:
            device = session.get(DeviceModel, device_id)
            if not device:
                return None
            for key, value in updates.items():
                setattr(device, key, value)
            session.commit()
            session.refresh(device)
            return device
    
    def delete_device(self, device_id: str) -> bool:
        """Delete a device."""
        with self.get_session() as session:
            device = session.get(DeviceModel, device_id)
            if not device:
                return False
            session.delete(device)
            session.commit()
            return True
    
    def add_detection(self, detection: Dict[str, Any]) -> DetectionModel:
        """Add a detection."""
        with self.get_session() as session:
            model = DetectionModel(**detection)
            session.add(model)
            session.commit()
            session.refresh(model)
            return model
    
    def get_detections(
        self,
        device_id: Optional[str] = None,
        class_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[DetectionModel]:
        """Get detections with filters."""
        with self.get_session() as session:
            query = select(DetectionModel).order_by(DetectionModel.timestamp.desc())
            
            if device_id:
                query = query.where(DetectionModel.device_id == device_id)
            if class_id is not None:
                query = query.where(DetectionModel.class_id == class_id)
            if start_time:
                query = query.where(DetectionModel.timestamp >= start_time)
            if end_time:
                query = query.where(DetectionModel.timestamp <= end_time)
            
            query = query.limit(limit)
            return list(session.execute(query).scalars().all())
    
    def add_track(self, track: Dict[str, Any]) -> TrackModel:
        """Add a track."""
        with self.get_session() as session:
            model = TrackModel(**track)
            session.add(model)
            session.commit()
            session.refresh(model)
            return model
    
    def get_tracks(
        self,
        device_id: Optional[str] = None,
        track_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[TrackModel]:
        """Get tracks with filters."""
        with self.get_session() as session:
            query = select(TrackModel).order_by(TrackModel.timestamp.desc())
            
            if device_id:
                query = query.where(TrackModel.device_id == device_id)
            if track_id:
                query = query.where(TrackModel.track_id == track_id)
            if start_time:
                query = query.where(TrackModel.timestamp >= start_time)
            if end_time:
                query = query.where(TrackModel.timestamp <= end_time)
            
            query = query.limit(limit)
            return list(session.execute(query).scalars().all())
    
    def add_alert(self, alert: Dict[str, Any]) -> AlertModel:
        """Add an alert."""
        with self.get_session() as session:
            model = AlertModel(**alert)
            session.add(model)
            session.commit()
            session.refresh(model)
            return model
    
    def get_alerts(
        self,
        type: Optional[str] = None,
        level: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        device_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AlertModel]:
        """Get alerts with filters."""
        with self.get_session() as session:
            query = select(AlertModel).order_by(AlertModel.timestamp.desc())
            
            if type:
                query = query.where(AlertModel.type == type)
            if level:
                query = query.where(AlertModel.level == level)
            if acknowledged is not None:
                query = query.where(AlertModel.acknowledged == acknowledged)
            if device_id:
                query = query.where(AlertModel.device_id == device_id)
            if start_time:
                query = query.where(AlertModel.timestamp >= start_time)
            if end_time:
                query = query.where(AlertModel.timestamp <= end_time)
            
            query = query.limit(limit)
            return list(session.execute(query).scalars().all())
    
    def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> Optional[AlertModel]:
        """Update an alert."""
        with self.get_session() as session:
            alert = session.get(AlertModel, alert_id)
            if not alert:
                return None
            for key, value in updates.items():
                setattr(alert, key, value)
            session.commit()
            session.refresh(alert)
            return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self.get_session() as session:
            alert = session.get(AlertModel, alert_id)
            if not alert:
                return False
            alert.acknowledged = True
            session.commit()
            return True
    
    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert."""
        with self.get_session() as session:
            alert = session.get(AlertModel, alert_id)
            if not alert:
                return False
            session.delete(alert)
            session.commit()
            return True
    
    def get_camera_config(self, config_id: str) -> Optional[CameraConfigModel]:
        """Get camera configuration."""
        with self.get_session() as session:
            return session.get(CameraConfigModel, config_id)
    
    def list_camera_configs(self) -> List[CameraConfigModel]:
        """List camera configurations."""
        with self.get_session() as session:
            return list(session.execute(select(CameraConfigModel)).scalars().all())
    
    def save_camera_config(self, config: Dict[str, Any]) -> CameraConfigModel:
        """Save camera configuration."""
        with self.get_session() as session:
            model = CameraConfigModel(**config)
            existing = session.get(CameraConfigModel, model.id)
            if existing:
                session.delete(existing)
            session.add(model)
            session.commit()
            session.refresh(model)
            return model
    
    def delete_camera_config(self, config_id: str) -> bool:
        """Delete camera configuration."""
        with self.get_session() as session:
            config = session.get(CameraConfigModel, config_id)
            if not config:
                return False
            session.delete(config)
            session.commit()
            return True
    
    def get_ptz_config(self, config_id: str) -> Optional[PTZConfigModel]:
        """Get PTZ configuration."""
        with self.get_session() as session:
            return session.get(PTZConfigModel, config_id)
    
    def list_ptz_configs(self) -> List[PTZConfigModel]:
        """List PTZ configurations."""
        with self.get_session() as session:
            return list(session.execute(select(PTZConfigModel)).scalars().all())
    
    def save_ptz_config(self, config: Dict[str, Any]) -> PTZConfigModel:
        """Save PTZ configuration."""
        with self.get_session() as session:
            model = PTZConfigModel(**config)
            existing = session.get(PTZConfigModel, model.id)
            if existing:
                session.delete(existing)
            session.add(model)
            session.commit()
            session.refresh(model)
            return model
    
    def delete_ptz_config(self, config_id: str) -> bool:
        """Delete PTZ configuration."""
        with self.get_session() as session:
            config = session.get(PTZConfigModel, config_id)
            if not config:
                return False
            session.delete(config)
            session.commit()
            return True
    
    def get_zone_config(self, config_id: str) -> Optional[ZoneConfigModel]:
        """Get zone configuration."""
        with self.get_session() as session:
            return session.get(ZoneConfigModel, config_id)
    
    def list_zone_configs(self) -> List[ZoneConfigModel]:
        """List zone configurations."""
        with self.get_session() as session:
            return list(session.execute(select(ZoneConfigModel)).scalars().all())
    
    def save_zone_config(self, config: Dict[str, Any]) -> ZoneConfigModel:
        """Save zone configuration."""
        with self.get_session() as session:
            model = ZoneConfigModel(**config)
            existing = session.get(ZoneConfigModel, model.id)
            if existing:
                session.delete(existing)
            session.add(model)
            session.commit()
            session.refresh(model)
            return model
    
    def delete_zone_config(self, config_id: str) -> bool:
        """Delete zone configuration."""
        with self.get_session() as session:
            config = session.get(ZoneConfigModel, config_id)
            if not config:
                return False
            session.delete(config)
            session.commit()
            return True
    
    def get_detection_statistics(
        self,
        device_id: Optional[str] = None,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """Get detection statistics."""
        with self.get_session() as session:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            query = select(
                DetectionModel.class_id,
                DetectionModel.class_name,
                func.count(DetectionModel.id).label("count"),
            ).where(DetectionModel.timestamp.between(start_time, end_time))
            
            if device_id:
                query = query.where(DetectionModel.device_id == device_id)
            
            query = query.group_by(DetectionModel.class_id, DetectionModel.class_name)
            
            results = session.execute(query).all()
            
            return {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_detections": sum(r.count for r in results),
                "by_class": [
                    {
                        "class_id": r.class_id,
                        "class_name": r.class_name,
                        "count": r.count,
                    }
                    for r in results
                ],
            }
    
    def get_alert_statistics(
        self,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """Get alert statistics."""
        with self.get_session() as session:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            query = select(
                AlertModel.level,
                AlertModel.type,
                func.count(AlertModel.id).label("count"),
            ).where(AlertModel.timestamp.between(start_time, end_time))
            
            query = query.group_by(AlertModel.level, AlertModel.type)
            
            results = session.execute(query).all()
            
            return {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_alerts": sum(r.count for r in results),
                "by_level": {
                    level: sum(r.count for r in results if r.level == level)
                    for level in ["info", "warning", "critical"]
                },
                "by_type": [
                    {
                        "type": r.type,
                        "level": r.level,
                        "count": r.count,
                    }
                    for r in results
                ],
            }
    
    def cleanup_old_data(
        self,
        days_to_keep: int = 30,
    ) -> Dict[str, int]:
        """Clean up old data."""
        with self.get_session() as session:
            cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)
            
            detections_deleted = session.execute(
                DetectionModel.__table__.delete().where(
                    DetectionModel.timestamp < cutoff_time
                )
            ).rowcount
            
            tracks_deleted = session.execute(
                TrackModel.__table__.delete().where(
                    TrackModel.timestamp < cutoff_time
                )
            ).rowcount
            
            session.commit()
            
            return {
                "detections_deleted": detections_deleted,
                "tracks_deleted": tracks_deleted,
            }


def create_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())