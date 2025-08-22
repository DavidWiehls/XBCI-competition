import asyncio
from bleak import BleakScanner, BleakClient
from bleak.exc import BleakError
from loguru import logger
import sys

# Configure logging for better visibility
logger.remove()
logger.add(sys.stderr, level="INFO")

# Define the device name prefixes from your constants.py file
DEVICE_PREFIX_TUPLE = ("jxj", "brainup", "music", "xiaomi")

async def test_ble_connection():
    """
    Scans for a compatible BLE device by name and attempts to connect.
    """
    logger.info("Scanning for compatible BLE devices...")
    device_found = None

    # Scan for a short period of time to find devices
    devices = await BleakScanner.discover(timeout=10.0)

    for device in devices:
        if device.name and device.name.lower().startswith(DEVICE_PREFIX_TUPLE):
            logger.info(f"Found a compatible device: {device.name} at {device.address}")
            device_found = device
            break

    if not device_found:
        logger.error("No compatible device found. Please ensure your device is on and in range.")
        return

    # Attempt to connect to the found device
    logger.info(f"Attempting to connect to {device_found.name} ({device_found.address})...")
    try:
        async with BleakClient(device_found.address) as client:
            if client.is_connected:
                logger.success(f"Successfully connected to {device_found.name}!")
            else:
                logger.error(f"Failed to connect to {device_found.name} after a successful discovery.")
    except BleakError as e:
        # Corrected line: removed the extraneous closing parenthesis
        logger.error(f"Connection to {device_found.name} failed due to a BLE error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during connection: {e}")

if __name__ == "__main__":
    asyncio.run(test_ble_connection())