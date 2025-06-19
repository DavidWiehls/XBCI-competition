"""BLE SDK for IoT devices.

This SDK provides a simple interface for communicating with BLE devices,
focusing on battery monitoring, signal strength, and electrode status.
"""

# Version info
__version__ = '0.0.4'


# Import main classes and functions
from .client import BleClient
from .constants import DeviceType
from .exceptions import (
    BleError,
    ConnectionError,
    DisconnectionError,
    NotificationError,
    DeviceNotFoundError,
    WriteError
)

from .scanner import scan_ble_devices

# The individual handlers and parsers are not typically part of the primary SDK interface
# if BleClient provides dedicated stream methods. They can be used internally or
# exposed if direct access to parsing is a design goal.



# Define the public API of the SDK
__all__ = [
    'BleClient',
    'DeviceType',
    'BleError',
    'ConnectionError',
    'DisconnectionError',
    'NotificationError',
    'DeviceNotFoundError',
    'WriteError',
    
    'scan_ble_devices',
]

