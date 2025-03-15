"""
This module contains all database schema definitions for the Clue server.
"""

import base64
import struct
from collections.abc import Iterable
from datetime import datetime
from enum import IntEnum, StrEnum, auto
from typing import Any, Optional, Self, override
from uuid import UUID

from bluenumbers import AdPacket
from geoalchemy2 import Geometry
from geoalchemy2.shape import from_shape
from pydantic import field_validator, model_validator
from shapely import Point
from sqlalchemy import JSON, Column, Enum, LargeBinary
from sqlmodel import Field, Relationship, SQLModel


class BLEDeviceUUID(SQLModel, table=True):
    """Association table for BLEDevice and BLEUUID"""

    __tablename__: str = "ble_device_uuid"  # type: ignore

    ble_device_id: int = Field(foreign_key="ble_device.id", primary_key=True)
    uuid: UUID = Field(foreign_key="ble_uuid.full_uuid", primary_key=True)


class AndroidAppUUID(SQLModel, table=True):
    """Association table for AndroidApp and BLEUUID"""

    __tablename__: str = "android_app_uuid"  # type: ignore

    app_id: str = Field(foreign_key="android_app.app_id", primary_key=True)
    uuid: UUID = Field(foreign_key="ble_uuid.full_uuid", primary_key=True)


class BLEUUID(SQLModel, table=True):
    """
    Schema for all BLE UUIDs discovered. This includes:
        Bluetooth SIG assigned UUIDs
        Scanned UUIDs from BLE devices
        Extracted UUIDs from Android apps
    """

    __tablename__: str = "ble_uuid"  # type: ignore

    full_uuid: UUID = Field(primary_key=True)
    short_uuid: int | None = Field(default=None)
    name: str | None = Field(default=None)

    # Relationships
    ble_devices: list["BLEDevice"] = Relationship(
        back_populates="uuids",
        link_model=BLEDeviceUUID,
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    android_apps: list["AndroidApp"] = Relationship(
        back_populates="uuids",
        link_model=AndroidAppUUID,
        sa_relationship_kwargs={"lazy": "selectin"},
    )

    def __hash__(self) -> int:
        return hash(self.full_uuid)


class BLEDeviceBase(SQLModel):
    """Base schema for BLE devices"""

    mac: str = Field(min_length=17, max_length=17)
    rssi: int = Field(...)
    time: datetime = Field(...)
    lat: float = Field(...)
    lon: float = Field(...)
    accuracy: float = Field(...)
    blob_name: str = Field(...)
    speed: float | None = Field(default=None)
    name: str | None = Field(default=None)
    manufacturer_id: int | None = Field(default=None)

    @field_validator("mac")
    def validate_mac(cls, value: str) -> str:
        """Ensure mac address is in uppercase"""
        return value.upper()


class BLEDeviceCreate(BLEDeviceBase):
    """Schema for creating BLE devices"""

    uuids: str | None = Field(default=None)
    raw_data: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_with_raw_data(self) -> Self:
        if self.raw_data is None:
            return self

        raw_data_bytes = base64.b64decode(self.raw_data)
        ad_packet = AdPacket.from_bytes(raw_data_bytes)
        self.name = ad_packet.name
        self.manufacturer_id = ad_packet.manufacturer_id
        self.uuids = ",".join(str(uuid) for uuid in ad_packet.uuids)
        self.raw_data = self.raw_data
        self.uuids = self.uuids
        return self

    def create(self) -> tuple[set[BLEUUID], "BLEDevice"]:
        """Create a BLEDevice instance from raw data"""
        uuid_list = self.uuids.split(",") if self.uuids else []
        ble_uuids = set(BLEUUID(full_uuid=UUID(uuid)) for uuid in uuid_list)
        ble_device = BLEDevice.from_create(self)
        return ble_uuids, ble_device


class BLEDevice(BLEDeviceBase, table=True):
    """Schema for BLE devices"""

    __tablename__: str = "ble_device"  # type: ignore

    # Data not included in the raw advertisement data
    id: int | None = Field(default=None, primary_key=True)
    coordinates: Any | None = Field(
        default=None,
        sa_column=Column(
            Geometry(geometry_type="POINT", srid=4326),
        ),
    )
    raw_data: bytes | None = Field(default=None, sa_column=Column(LargeBinary))
    ad_packet: dict | None = Field(default=None, sa_column=Column(JSON))

    # Relationships
    uuids: list[BLEUUID] = Relationship(
        back_populates="ble_devices",
        link_model=BLEDeviceUUID,
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    qt_device: Optional["QTDevice"] = Relationship(
        back_populates="ble_device", sa_relationship_kwargs={"lazy": "selectin", "uselist": False}
    )

    @override
    def model_post_init(self, __context: Any) -> None:
        """Add the missing coordinates field"""
        if self.lat and self.lon:
            self.coordinates = self.coordinates or from_shape(Point(self.lon, self.lat), srid=4326)

    @classmethod
    def from_create(cls: type[Self], raw_device: BLEDeviceCreate) -> Self:
        """Create a BLEDevice instance from raw data"""
        raw_data_bytes = base64.b64decode(raw_device.raw_data) if raw_device.raw_data else None
        return cls(
            mac=raw_device.mac,
            rssi=raw_device.rssi,
            time=raw_device.time,
            lat=raw_device.lat,
            lon=raw_device.lon,
            accuracy=raw_device.accuracy,
            blob_name=raw_device.blob_name,
            speed=raw_device.speed,
            name=raw_device.name,
            manufacturer_id=raw_device.manufacturer_id,
            raw_data=raw_data_bytes,
            ad_packet=AdPacket.from_bytes(raw_data_bytes).model_dump() if raw_data_bytes else None,
        )

    @property
    def raw_data_b64(self) -> str | None:
        """Return the raw data as a base64 encoded string"""
        return base64.b64encode(self.raw_data).decode() if self.raw_data else None


class AndroidAppCreate(SQLModel):
    app_id: str
    name: str
    description: str | None = None
    uuids: str | None = Field(default=None, description="Comma-separated list of UUIDs")

    def create(self) -> tuple[set[BLEUUID], "AndroidApp"]:
        """Create an AndroidApp instance from raw data"""
        uuid_list = self.uuids.split(",") if self.uuids else []
        ble_uuids = set(BLEUUID(full_uuid=UUID(uuid)) for uuid in uuid_list)
        android_app = AndroidApp(app_id=self.app_id, name=self.name, description=self.description)
        return ble_uuids, android_app


class AndroidApp(SQLModel, table=True):
    """Schema for Android apps"""

    __tablename__: str = "android_app"  # type: ignore

    app_id: str = Field(primary_key=True)
    name: str = Field(...)
    description: str | None = Field(default=None)

    # Relationships
    uuids: list[BLEUUID] = Relationship(
        back_populates="android_apps",
        link_model=AndroidAppUUID,
        sa_relationship_kwargs={"lazy": "selectin"},
    )


class SyncState(StrEnum):
    """Sync states"""

    PENDING = auto()  # Found in the GCS bucket but not yet processed
    QUEUED = auto()  # Queued for processing
    PROCESSING = auto()  # Currently being processed
    SYNCED = auto()  # Successfully processed
    FAILED = auto()  # Failed to process


class SyncStatus(SQLModel, table=True):
    """Schema for syncing status"""

    __tablename__: str = "sync_status"  # type: ignore

    blob_name: str = Field(primary_key=True)
    state: SyncState = Field(default=SyncState.PENDING)
    process_time: datetime | None = Field(default=None)
    message: str | None = Field(default=None)

    # Note: A blob can contain many BLE devices
    # The relationship is from BLE device to blob_name, not the other way around


class NoSQLData(SQLModel, table=True):
    """Schema for NoSQL data"""

    __tablename__: str = "nosql_data"  # type: ignore

    key: str = Field(primary_key=True)
    value: dict = Field(sa_column=Column(JSON))


class QTMode(IntEnum):
    """Mode of the device"""

    UNKNOWN = 0
    INSTALLER = 1
    DEALER = 2
    USER = 3
    NOSALE = 4
    BCA = 5
    VALET = 6
    BOOTLOAD = 7
    UNCONFIGURED = 8


class QTColor(IntEnum):
    """Color of the device"""

    ORANGE = 1
    BLUE = 2
    LIGHTGREEN = 3
    RED = 4
    MEDIUMPURPLE = 5
    LIGHTGREY = 6


class QTDevice(SQLModel, table=True):
    __tablename__: str = "qt_device"  # type: ignore

    ble_device_id: int = Field(foreign_key="ble_device.id", primary_key=True)
    name: str
    mac: str
    color: QTColor = Field(sa_column=Column(Enum(QTColor)))
    mode: QTMode = Field(sa_column=Column(Enum(QTMode)))
    armed: bool
    snowmode: bool
    vbat: float

    # Relationships
    ble_device: BLEDevice = Relationship(back_populates="qt_device")

    @classmethod
    def from_ble_device(cls: type["QTDevice"], ble_device: BLEDevice) -> "QTDevice":
        assert ble_device.name is not None, "QTDevice requires a name"
        assert ble_device.raw_data is not None, "QTDevice requires raw data"
        assert ble_device.id is not None, "QTDevice requires a BLE device ID"

        def iter_fields(data: bytes) -> Iterable[tuple[int, bytes]]:
            offset = 0
            while offset < len(data):
                length = data[offset]
                if length == 0:
                    return
                # Get type from the byte after length
                type_id = data[offset + 1]
                # Get value excluding length and type bytes
                value = data[offset + 2 : offset + length + 1]
                yield type_id, value
                offset += length + 1

        fields: dict[int, bytes] = {}
        for field_type, value in iter_fields(ble_device.raw_data):
            fields[field_type] = value

        manufacturer_data = fields.get(255, b"")
        name = fields.get(9, b"").decode("utf-8")
        mac = ble_device.mac
        unpacked = struct.unpack("<BBBBBB", manufacturer_data)
        color = QTColor(((unpacked[0] & 0xC0) >> 6) | ((unpacked[1] & 0xC0) >> 4))
        mode = QTMode(((unpacked[2] & 0xC0) >> 6) | ((unpacked[3] & 0x40) >> 4))
        armed = bool(unpacked[3] & 0x80)
        snowmode = bool(unpacked[4] & 0x40)
        vbat = unpacked[5] * 60 / 1000
        return cls(
            name=name,
            mac=mac,
            color=color,
            mode=mode,
            armed=armed,
            snowmode=snowmode,
            vbat=vbat,
            ble_device_id=ble_device.id,
            ble_device=ble_device,
        )
