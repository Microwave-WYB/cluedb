"""
Centralized database access for the Clue server.
"""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Callable
from warnings import deprecated

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, col, create_engine, select

from cluedb.models import (
    BLEUUID,
    AndroidApp,
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

    @deprecated("Use upsert_model instead")
    def upsert_ble_uuid(self, ble_uuid: BLEUUID) -> None:
        """Update or insert a BLEUUID."""
        self.upsert_model(ble_uuid, ble_uuid.full_uuid)

    def insert_model(self, model: SQLModel) -> None:
        """
        Insert a single model into the database, notifying any listeners.
        """
        with self.session() as session:
            session.add(model)
            session.commit()
            # refresh the model to get the primary key
            session.refresh(model)
            # run any listeners for this model
            for listener in self._listeners.get(type(model), []):
                listener(self, model)

    def upsert_model(self, model: SQLModel, pkey: Any) -> None:
        """
        Update or insert a single model into the database, notifying any listeners.
        """
        if not pkey:
            raise ValueError(
                "Primary key value must be provided to update or insert a model into the database. "
                "Use the insert_model method to insert a new model."
            )
        existing = self.get(type(model), pkey)
        if existing:
            for key, value in model.model_dump().items():
                setattr(existing, key, value)
            self.insert_model(existing)
        else:
            self.insert_model(model)

        for listener in self._listeners.get(type(model), []):
            listener(self, model)

    def get[T: SQLModel](self, model: type[T], pkey: Any) -> T | None:
        """Get a model by primary key."""
        with self.session() as session:
            return session.get(model, pkey)

    def create_ble_device(self, ble_device_create: BLEDeviceCreate) -> None:
        """Insert UUIDs and the BLEDevice into the database using a single transaction"""
        uuids, device = ble_device_create.create()

        with self.session() as session:
            # Handle all UUIDs in bulk
            for uuid in uuids:
                existing = session.get(type(uuid), uuid.full_uuid)
                if existing:
                    for key, value in uuid.model_dump().items():
                        setattr(existing, key, value)
                else:
                    session.add(uuid)

            # Add the device
            session.add(device)
            session.flush()  # Get the ID without committing

            # Add all relationships at once
            assert device.id is not None
            for uuid in uuids:
                session.add(BLEDeviceUUID(ble_device_id=device.id, uuid=uuid.full_uuid))

            # Commit everything in one go
            session.commit()

            # Now notify listeners (after the transaction)
            for uuid in uuids:
                for listener in self._listeners.get(type(uuid), []):
                    listener(self, uuid)

            for listener in self._listeners.get(type(device), []):
                listener(self, device)

    def create_android_app(self, android_app: AndroidAppCreate, overwrite: bool = False) -> None:
        """Insert UUIDs and the AndroidApp into the database using a single transaction"""
        with self.session() as session:
            # Check if app exists first
            existing_app = session.get(AndroidApp, android_app.app_id)
            if existing_app and not overwrite:
                raise ValueError(f"AndroidApp with app_id {android_app.app_id} already exists")

            # Create entities
            uuids, app = android_app.create()

            # Handle all UUIDs in bulk
            for uuid in uuids:
                existing_uuid = session.get(type(uuid), uuid.full_uuid)
                if existing_uuid:
                    for key, value in uuid.model_dump().items():
                        setattr(existing_uuid, key, value)
                else:
                    session.add(uuid)

            # Handle app
            if existing_app and overwrite:
                for key, value in app.model_dump().items():
                    setattr(existing_app, key, value)
            else:
                session.add(app)

            session.flush()  # Ensure app has ID without committing

            # Add all relationships at once
            for uuid in uuids:
                # Check if relationship already exists
                relationship = session.exec(
                    select(AndroidAppUUID).where(
                        (col(AndroidAppUUID.app_id) == app.app_id)
                        & (col(AndroidAppUUID.uuid) == uuid.full_uuid)
                    )
                ).first()

                if not relationship:
                    session.add(AndroidAppUUID(app_id=app.app_id, uuid=uuid.full_uuid))

            # Commit everything in one go
            session.commit()

            # Now notify listeners (after the transaction)
            for uuid in uuids:
                for listener in self._listeners.get(type(uuid), []):
                    listener(self, uuid)

            for listener in self._listeners.get(type(app), []):
                listener(self, app)
