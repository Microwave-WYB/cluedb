from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pytest import fixture
from sqlmodel import create_engine, select
from testcontainers.postgres import PostgresContainer

from cluedb.database import Database
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


@fixture
def database():
    with PostgresContainer("postgis/postgis") as db:
        postgres_dsn = db.get_connection_url().replace("+psycopg2", "")
        db.with_exposed_ports(5432)
        engine = create_engine(postgres_dsn)
        Database.init_db(engine)
        yield Database(postgres_dsn)


def test_get_blob_names(database: Database):
    # Setup: Add sync status entries
    for i in range(3):
        database.insert_model(SyncStatus(blob_name=f"blob_{i}"))

    # Test: Get all blob names
    result = database.get_blob_names()
    assert len(result) == 3
    assert set(result) == {"blob_0", "blob_1", "blob_2"}


def test_get_blob_names_with_states(database: Database):
    # Setup: Add sync status entries with different states
    database.insert_model(SyncStatus(blob_name="blob_pending", state=SyncState.PENDING))
    database.insert_model(SyncStatus(blob_name="blob_synced", state=SyncState.SYNCED))
    database.insert_model(SyncStatus(blob_name="blob_failed", state=SyncState.FAILED))

    # Test: Filter by states
    assert set(database.get_blob_names(SyncState.PENDING)) == {"blob_pending"}
    assert set(database.get_blob_names(SyncState.SYNCED, SyncState.FAILED)) == {
        "blob_synced",
        "blob_failed",
    }


def test_update_sync_status(database: Database):
    # Setup: Add a sync status entry
    database.insert_model(SyncStatus(blob_name="test_blob"))

    # Test: Update the status
    database.update_sync_status("test_blob", SyncState.SYNCED, "Success")

    # Verify the update
    with database.session() as session:
        status = session.exec(select(SyncStatus).where(SyncStatus.blob_name == "test_blob")).one()
        assert status.state == SyncState.SYNCED
        assert status.message == "Success"
        assert status.process_time is not None


def test_update_sync_status_not_found(database: Database):
    # Test: Update a non-existent sync status
    with pytest.raises(ValueError, match="SyncStatus for nonexistent_blob not found"):
        database.update_sync_status("nonexistent_blob", SyncState.SYNCED)


def test_upsert_ble_uuid(database: Database):
    # Test insert new UUID
    uuid_val = uuid4()
    ble_uuid = BLEUUID(full_uuid=uuid_val, short_uuid=1234, name="Test UUID")
    database.upsert_ble_uuid(ble_uuid)

    # Test update existing UUID
    updated_uuid = BLEUUID(full_uuid=uuid_val, short_uuid=5678, name="Updated UUID")
    database.upsert_ble_uuid(updated_uuid)

    # Verify
    with database.session() as session:
        result = session.exec(select(BLEUUID).where(BLEUUID.full_uuid == uuid_val)).one()
        assert result.short_uuid == 5678
        assert result.name == "Updated UUID"


def test_upsert_model(database: Database):
    uuid_val = uuid4()
    ble_uuid = BLEUUID(full_uuid=uuid_val, short_uuid=1234, name="Test UUID")
    database.upsert_model(ble_uuid, uuid_val)

    updated_uuid = BLEUUID(full_uuid=uuid_val, short_uuid=5678, name="Updated UUID")
    database.upsert_model(updated_uuid, uuid_val)

    with database.session() as session:
        result = session.exec(select(BLEUUID).where(BLEUUID.full_uuid == uuid_val)).one()
        assert result.short_uuid == 5678
        assert result.name == "Updated UUID"


def test_get(database: Database):
    # Setup: Add a BLE device
    ble_uuid = BLEUUID(
        full_uuid=UUID("00000000-0000-0000-0000-000000000001"), short_uuid=0, name="Test UUID"
    )
    database.insert_model(ble_uuid)

    assert database.get(BLEUUID, UUID("00000000-0000-0000-0000-000000000001"))


def test_create_ble_devices(database: Database):
    # Create a BLE device
    ble_device = BLEDeviceCreate(
        mac="AA:BB:CC:DD:EE:FF",
        rssi=-70,
        time=datetime.now(UTC),
        lat=37.7749,
        lon=-122.4194,
        accuracy=10.0,
        blob_name="test_blob",
        uuids=str(uuid4()),
    )
    database.create_ble_devices([ble_device])

    # Verify
    with database.session() as session:
        devices = session.exec(select(BLEDevice)).all()
        assert len(devices) == 1
        assert devices[0].mac == "AA:BB:CC:DD:EE:FF"

        # Check the many-to-many relationship
        device_uuids = session.exec(select(BLEDeviceUUID)).all()
        assert len(device_uuids) == 1


def test_create_android_app(database: Database):
    # Create an Android app
    uuid_val = str(uuid4())
    app_create = AndroidAppCreate(
        app_id="com.example.app", name="Test App", description="A test app", uuids=uuid_val
    )
    database.create_android_apps([app_create])

    # Verify
    with database.session() as session:
        apps = session.exec(select(AndroidApp)).all()
        assert len(apps) == 1
        assert apps[0].app_id == "com.example.app"
        assert apps[0].name == "Test App"

        # Check the UUID was created
        uuids = session.exec(select(BLEUUID)).all()
        assert len(uuids) == 1
        assert str(uuids[0].full_uuid) == uuid_val

        # Check the many-to-many relationship
        app_uuids = session.exec(select(AndroidAppUUID)).all()
        assert len(app_uuids) == 1
        assert app_uuids[0].app_id == "com.example.app"

    another_app_create = AndroidAppCreate(
        app_id="com.example.app2", name="Test App 2", description="Another test app", uuids=uuid_val
    )
    database.create_android_apps([another_app_create])

    with database.session() as session:
        apps = session.exec(select(AndroidApp)).all()
        assert len(apps) == 2
        assert apps[1].app_id == "com.example.app2"
        assert apps[1].name == "Test App 2"

        # Check the UUID was created
        uuids = session.exec(select(BLEUUID)).all()
        assert len(uuids) == 1
        assert str(uuids[0].full_uuid) == uuid_val

        # Check the many-to-many relationship
        app_uuids = session.exec(select(AndroidAppUUID)).all()
        assert len(app_uuids) == 2
        assert app_uuids[1].app_id == "com.example.app2"


listener_called = False


@Database.listen_for(BLEDevice)
def ble_device_listener(db: Database, model: BLEDevice) -> None:
    if model.name and model.name == "Listener Test":
        global listener_called
        listener_called = True

        special_device = BLEDevice(
            mac=model.mac,
            rssi=model.rssi,
            time=model.time,
            lat=model.lat,
            lon=model.lon,
            accuracy=model.accuracy,
            blob_name=model.blob_name,
            name="Listener Called",
        )
        db.insert_model(special_device)


def test_insert_model_listener(database: Database):
    # Create a BLE device
    ble_device = BLEDeviceCreate(
        mac="AA:BB:CC:DD:EE:FF",
        rssi=-70,
        time=datetime.now(UTC),
        lat=37.7749,
        lon=-122.4194,
        accuracy=10.0,
        blob_name="test_blob",
        uuids=str(uuid4()),
        name="Listener Test",
    )
    database.create_ble_devices([ble_device])

    # Verify the listener was called
    assert listener_called

    # Verify the listener modified the model
    with database.session() as session:
        devices = session.exec(select(BLEDevice).where(BLEDevice.name == "Listener Called")).all()
        assert len(devices) == 1
