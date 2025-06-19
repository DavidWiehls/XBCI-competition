"""Custom exceptions for the ble_sdk package."""

class BleError(Exception):
    """Base exception for all BLE-related errors."""
    pass


class ConnectionError(BleError):
    """Raised when a connection to a BLE device fails."""
    pass


class DisconnectionError(BleError):
    """Raised when disconnecting from a BLE device fails."""
    pass


class NotificationError(BleError):
    """Raised when enabling or disabling notifications fails."""
    pass


class DeviceNotFoundError(BleError):
    """Raised when a device with the specified address is not found."""
    pass


class WriteError(BleError):
    """Raised when writing to a GATT characteristic fails."""
    pass 


class HandleParseError(BleError):
    """Raised when a handle parse error occurs."""
    pass


    