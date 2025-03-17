"""
Centralized database access for the Clue server.
"""

from collections.abc import Generator, Iterable
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
        with self.session() as session:
            try:
                session.add(model)
                session.commit()
                session.refresh(model)
            except IntegrityError as e:
                if not exist_ok:
                    raise e

        self._notify_listeners([model], type(model))

    def insert_models[T: SQLModel](self, models: Iterable[T], exist_ok: bool = False) -> None:
        """
        Insert multiple models into the database in a single transaction.
        Set exist_ok to True to ignore unique constraint errors.
        """
        if not models:
            return

        if not exist_ok:
            with self.session() as session:
                session.add_all(models)
                session.commit()
                for model in models:
                    session.refresh(model)
                    for listener in self._listeners.get(type(model), []):
                        listener(self, model)
            return

        self._notify_listeners(models, type(next(iter(models))))

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

        self._notify_listeners([model], type(model))

    def get[T: SQLModel](self, model: type[T], pkey: Any) -> T | None:
        """Get a model by primary key."""
        with self.session() as session:
            return session.get(model, pkey)

    def _notify_listeners[T: SQLModel](self, models: Iterable[T], model_type: type[T]) -> None:
        """Notify listeners of a model change."""
        if not self._listeners.get(model_type):
            return
        for model in models:
            for listener in self._listeners[model_type]:
                listener(self, model)

    def create_ble_devices(self, ble_device_creates: list[BLEDeviceCreate]) -> None:
        """Insert multiple BLE devices and their UUIDs into the database in a single transaction

        Args:
            ble_device_creates: List of BLEDeviceCreate objects to process
        """
        if not ble_device_creates:
            return

        with self.session() as session:
            # Prepare all devices and UUIDs
            all_devices: list[BLEDevice] = []
            all_uuids: list[BLEUUID] = []
            device_uuid_pairs: list[tuple[BLEDevice, set[BLEUUID]]] = []

            for ble_device_create in ble_device_creates:
                uuids, device = ble_device_create.create()
                all_devices.append(device)
                all_uuids.extend(uuids)
                device_uuid_pairs.append((device, uuids))

            # First insert all UUIDs with exist_ok=True to handle duplicates
            for uuid in all_uuids:
                if not session.get(BLEUUID, uuid.full_uuid):
                    session.add(uuid)

            session.flush()

            # Then insert all devices
            session.add_all(all_devices)
            session.flush()

            # Finally create and insert all device-UUID relationships
            device_uuid_relationships = []
            for device, uuids in device_uuid_pairs:
                assert device.id is not None
                for uuid in uuids:
                    device_uuid_relationships.append(
                        BLEDeviceUUID(ble_device_id=device.id, uuid=uuid.full_uuid)
                    )

            if device_uuid_relationships:
                session.add_all(device_uuid_relationships)
            session.commit()

            for device in all_devices:
                session.refresh(device)

            for relationship in device_uuid_relationships:
                session.refresh(relationship)

            for uuid in all_uuids:
                session.refresh(uuid)

        self._notify_listeners(all_devices, BLEDevice)
        self._notify_listeners(all_uuids, BLEUUID)
        self._notify_listeners(device_uuid_relationships, BLEDeviceUUID)

    def create_android_apps(self, android_app_creates: list[AndroidAppCreate]) -> None:
        """Insert multiple Android apps and their UUIDs into the database in a single transaction

        Args:
            android_app_creates: List of AndroidAppCreate objects to process
        """
        if not android_app_creates:
            return

        with self.session() as session:
            # Prepare all apps and UUIDs
            all_apps: list[AndroidApp] = []
            all_uuids: list[BLEUUID] = []
            app_uuid_pairs: list[tuple[AndroidApp, set[BLEUUID]]] = []

            # Create entities
            for app_create in android_app_creates:
                uuids, app = app_create.create()
                all_apps.append(app)
                all_uuids.extend(uuids)
                app_uuid_pairs.append((app, uuids))

            # First insert all UUIDs with exist_ok=True to handle duplicates
            for uuid in all_uuids:
                if not session.get(BLEUUID, uuid.full_uuid):
                    session.add(uuid)
            session.flush()

            # Then insert all apps with exist_ok=True to skip existing ones
            for app in all_apps:
                if not session.get(AndroidApp, app.app_id):
                    session.add(app)
            session.flush()

            # Finally create and insert all app-UUID relationships
            app_uuid_relationships = []
            for app, uuids in app_uuid_pairs:
                for uuid in uuids:
                    app_uuid_relationships.append(
                        AndroidAppUUID(app_id=app.app_id, uuid=uuid.full_uuid)
                    )

            if app_uuid_relationships:
                for relationship in app_uuid_relationships:
                    existing = session.get(AndroidAppUUID, (relationship.app_id, relationship.uuid))
                    if not existing:
                        session.add(relationship)

                session.commit()

        self._notify_listeners(all_apps, AndroidApp)
        self._notify_listeners(all_uuids, BLEUUID)
        self._notify_listeners(app_uuid_relationships, AndroidAppUUID)

    def dispose(self) -> None:
        """Close the database connection."""
        self.engine.dispose()

    def __del__(self) -> None:
        self.dispose()
