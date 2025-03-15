from cluedb.database import Database
from cluedb.models import BLEDevice, QTDevice


@Database.listen_for(BLEDevice)
def insert_qt_device(db: Database, model: BLEDevice) -> None:
    """
    On BLEDevice insert, check if the device is a QT device and insert it into the database.
    """
    if model.name and model.name.startswith("QT "):
        return
    qt_device = QTDevice.from_ble_device(model)
    db.insert_model(qt_device)
