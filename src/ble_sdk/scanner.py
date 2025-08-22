from bleak import BleakScanner

from .constants import SCAN_TIMEOUT, DEVICE_PREFIX_TUPLE
from .log import sdk_logger as logger


async def scan_ble_devices():
    """
    Scan for BLE devices with a timeout.
    Returns:
        list: A list of dictionaries containing device information.
            Each dictionary contains:
                - 'name': Device name (lower case)
                - 'address': Device address
                - 'rssi': Received signal strength indicator (or None if not available)
    """
    logger.debug(f"Scanning devices timeout {SCAN_TIMEOUT} seconds...")
    devices = await BleakScanner.discover(timeout=SCAN_TIMEOUT, return_adv=False)
    # devices = []
    res = []
    for device in devices:
        if device.name:
            # Try to get RSSI safely
            rssi = None
            try:
                # Try different ways to access RSSI
                if hasattr(device, '_rssi'):
                    rssi = device._rssi
                elif hasattr(device, 'rssi'):
                    rssi = device.rssi
                elif hasattr(device, 'metadata') and 'rssi' in device.metadata:
                    rssi = device.metadata['rssi']
            except Exception as e:
                logger.debug(f"Could not get RSSI for device {device.name}: {e}")
            
            logger.debug(f"Found device {device.name} with address {device.address} and RSSI {rssi}")
            name_lower = device.name.lower()
            if name_lower.startswith(DEVICE_PREFIX_TUPLE):
                res.append({"name": device.name, "address": str(device.address), "rssi": rssi})
    logger.info(f"Scan results: {len(res)} valid BLE devices")
    return res
