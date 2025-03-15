"""
Centralized database access for the Clue server.
"""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Callable

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, col, create_engine, select

from cluedb.models import (
    BLEUUID,
    AndroidAppCreate,
    AndroidAppUUID,
    BLEDeviceCreate,
    BLEDeviceUUID,
    SyncState,
    SyncStatus,
)


class Database:
    """Centralized database access for the Cluetooth data."""

    _listeners: dict[type[SQLModel], list[Callable[["Database", SQLModel], None]]] = {}

    def __init__(self, postgres_dsn: str) -> None:
        self.engine = create_engine(postgres_dsn)
        Database.init_db(self.engine)

    @classmethod
    def listen_for[T: SQLModel](cls, model: type[T]):
        """Decorator to register a listener for a model."""

        def decorator(func: Callable[[Database, T], None]) -> Callable[[Database, T], None]:
            if not cls._listeners.get(model):
                cls._listeners[model] = []
            cls._listeners[model].append(func)  # type: ignore
            return func

        return decorator

    @staticmethod
    def init_db(engine: Engine) -> None:
        """Initialize the database schema."""
        SQLModel.metadata.create_all(engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Session context manager.

        Usage:
        .. code-block:: python
            with db.session() as session:
                session.add(model)
                session.commit()
        """
        with Session(self.engine) as session:
            yield session

    def get_blob_names(self, *states: SyncState) -> list[str]:
        """Get all blob names or filter by states."""
        with self.session() as session:
            if not states:
                return [x.blob_name for x in session.exec(select(SyncStatus)).all()]
            return [
                x.blob_name
                for x in session.exec(
                    select(SyncStatus).where(col(SyncStatus.state).in_(states))
                ).all()
            ]

    def update_sync_status(
        self, blob_name: str, state: SyncState, message: str | None = None
    ) -> None:
        """Update the sync status for the given blob name."""
        with self.session() as session:
            sync_status = session.exec(
                select(SyncStatus).where(col(SyncStatus.blob_name) == blob_name)
            ).first()
            if not sync_status:
                raise ValueError(f"SyncStatus for {blob_name} not found")
            sync_status.state = state
            sync_status.message = message
            sync_status.process_time = datetime.now(UTC)
            session.commit()

    def upsert_ble_uuid(self, ble_uuid: BLEUUID) -> None:
        """Update or insert a BLEUUID."""
        with self.session() as session:
            existing = session.exec(
                select(BLEUUID).where(col(BLEUUID.full_uuid) == ble_uuid.full_uuid)
            ).first()
            if existing:
                if existing.short_uuid == ble_uuid.short_uuid and existing.name == ble_uuid.name:
                    return
                existing.short_uuid = ble_uuid.short_uuid
                existing.name = ble_uuid.name
                session.commit()
            else:
                self.insert_model(ble_uuid)

    def insert_model(self, model: SQLModel) -> None:
        """
        Insert a single model into the database, running pre-processors and post-processors.
        """
        with self.session() as session:
            session.add(model)
            session.commit()
            # refresh the model to get the primary key
            session.refresh(model)
            # run any listeners for this model
            for listener in self._listeners.get(type(model), []):
                listener(self, model)

    def create_ble_device(self, ble_device_create: BLEDeviceCreate) -> None:
        """Insert UUIDs and the BLEDevice into the database"""
        uuids, device = ble_device_create.create()
        for uuid in uuids:
            self.upsert_ble_uuid(uuid)
        self.insert_model(device)
        assert device.id is not None
        # Update the many-to-many relationship
        for uuid in uuids:
            self.insert_model(BLEDeviceUUID(ble_device_id=device.id, uuid=uuid.full_uuid))

    def create_android_app(self, android_app: AndroidAppCreate) -> None:
        """Insert UUIDs and the AndroidApp into the database"""
        uuids, app = android_app.create()
        for uuid in uuids:
            self.upsert_ble_uuid(uuid)
        self.insert_model(app)
        assert app.app_id is not None
        # Update the many-to-many relationship
        for uuid in uuids:
            self.insert_model(AndroidAppUUID(app_id=app.app_id, uuid=uuid.full_uuid))
