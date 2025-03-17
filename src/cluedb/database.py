"""
Centralized database access for the Clue server.
"""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Callable
from warnings import deprecated

from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, col, create_engine, select

from cluedb.models import (
    BLEUUID,
    AndroidApp,
    AndroidAppCreate,
    AndroidAppUUID,
    BLEDevice,
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

    def insert_model(self, model: SQLModel, exist_ok: bool = False) -> None:
        """
        Insert a single model into the database, notifying any listeners.
        Set exist_ok to True to ignore unique constraint errors.
        """
        try:
            with self.session() as session:
                session.add(model)
                session.commit()
                # refresh the model to get the primary key
                session.refresh(model)
                # run any listeners for this model
                for listener in self._listeners.get(type(model), []):
                    listener(self, model)
        except IntegrityError:
            if not exist_ok:
                raise

    def upsert_model(self, model: SQLModel, pkey: Any) -> None:
        """
        Update or insert a single model into the database, notifying any listeners.
        This is different from insert_model with exist_ok=True
        because it will update the model if it already exists.
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

    def create_ble_devices(self, ble_device_creates: list[BLEDeviceCreate]) -> None:
        """Insert multiple BLE devices and their UUIDs into the database in a single transaction

        Args:
            ble_device_creates: List of BLEDeviceCreate objects to process
        """
        if not ble_device_creates:
            return

        # Prepare all devices and UUIDs before opening transaction
        all_devices: list[BLEDevice] = []
        all_uuids: list[BLEUUID] = []
        device_uuid_pairs: list[tuple[BLEDevice, set[BLEUUID]]] = []

        for ble_device_create in ble_device_creates:
            uuids, device = ble_device_create.create()
            all_devices.append(device)
            all_uuids.extend(uuids)
            device_uuid_pairs.append((device, uuids))

        with self.session() as session:
            # Process all UUIDs in bulk
            uuid_cache = {}  # Cache for UUID lookups to avoid repeated queries

            # Get all existing UUIDs in a single query to minimize database roundtrips
            existing_uuid_objects = {}
            for uuid_type in {type(uuid) for uuid in all_uuids}:
                uuid_values = [uuid.full_uuid for uuid in all_uuids if isinstance(uuid, uuid_type)]
                if uuid_values:
                    # Query all UUIDs of this type in a single query
                    for existing in session.exec(
                        select(uuid_type).where(col(uuid_type.full_uuid).in_(uuid_values))
                    ).all():
                        existing_uuid_objects[existing.full_uuid] = existing

            # Update or add all UUIDs
            uuids_to_add = []
            for uuid in all_uuids:
                existing = existing_uuid_objects.get(uuid.full_uuid)
                if existing:
                    # Update existing UUID
                    for key, value in uuid.model_dump().items():
                        setattr(existing, key, value)
                    uuid_cache[uuid.full_uuid] = existing
                else:
                    # Add new UUID
                    uuids_to_add.append(uuid)
                    uuid_cache[uuid.full_uuid] = uuid

            # Add all new UUIDs in bulk
            if uuids_to_add:
                session.add_all(uuids_to_add)

            # Add all devices
            session.add_all(all_devices)
            session.flush()  # Get IDs without committing

            # Create all device-UUID relationships at once
            device_uuid_relationships = []
            for device, uuids in device_uuid_pairs:
                assert device.id is not None
                for uuid in uuids:
                    device_uuid_relationships.append(
                        BLEDeviceUUID(ble_device_id=device.id, uuid=uuid.full_uuid)
                    )

            # Add all relationships in bulk
            if device_uuid_relationships:
                session.add_all(device_uuid_relationships)

            # Commit everything in one transaction
            session.commit()

            # Notify listeners (after transaction completes)
            for uuid in all_uuids:
                for listener in self._listeners.get(type(uuid), []):
                    listener(self, uuid)

            for device in all_devices:
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
