"""Example demonstrating the use of BleClient to stream battery data."""

import sys
import asyncio
from typing import Dict, Any
from loguru import logger
# Configure logger for better output
logger.remove()
logger.add(sys.stderr, level="INFO", filter=lambda record: not record["extra"].get("sdk_internal", False))

from ble_sdk.client import BleClient
from ble_sdk.constants import DeviceType
from ble_sdk.exceptions import BleError, ConnectionError as BleConnectionError
from ble_sdk.handles.eeg_handler import eeg_builtin_callback

from ble_sdk.log import sdk_logger_manager
sdk_logger_manager.set_level("DEBUG")


DEFAULT_DEVICE_ADDRESS = "E9:2C:7B:52:03:EF"  # jxj_4-bci_03ef
# DEFAULT_DEVICE_ADDRESS = "E5:6A:D5:08:A5:24"  # jxj_2-bci_a524
# DEFAULT_DEVICE_ADDRESS = "D8:50:96:79:15:DA"  # jxj_1-bci_15da
DEFAULT_DEVICE_TYPE = DeviceType.BLE_4
# DEFAULT_DEVICE_ADDRESS = "D2:43:D5:88:4D:9A"  # Replace with your device address or use None for discovery



def user_handle_data(data: Dict[str, Any]):
    """Callback function to handle incoming battery data."""
    # logger.info(f"Received Battery Data: Level={data.get('battery_level', 'N/A')}%, RSSI={data.get('rssi', 'N/A')} dBm")
    logger.info(f"Received Data: {data}")


async def main():
    """Main function to connect, stream battery data, and disconnect."""

    logger.info(f"Attempting to use BLE SDK with device: {DEFAULT_DEVICE_ADDRESS}")

    # Initialize BleClient with the device address and type
    # Assuming BLE_8 is the correct type as per previous contexts
    client = BleClient(address=DEFAULT_DEVICE_ADDRESS, device_type=DEFAULT_DEVICE_TYPE)

    try:
        async with client:  # Handles connect and disconnect automatically
            logger.info(f"Successfully connected to {client.address}.")

            # Start streaming battery data
            logger.info("Starting battery data stream...")
            # await client.start_battery_stream(user_handle_data)
            # await client.start_impedance_stream(user_handle_data, raw_data_only=False)
            # await client.start_eeg_stream(eeg_builtin_callback, raw_data_only=False)
            # await client.start_imu_stream(user_handle_data, raw_data_only=True)
            # await client.start_lead_stream(user_handle_data, raw_data_only=False)
            logger.info("Battery stream started. Monitoring for 20 seconds...")

            # Keep the stream running for 10 seconds
            await asyncio.sleep(10)

            # Stop streaming battery data
            logger.info("Stopping battery data stream...")
            await client.stop_battery_stream()
            logger.info("Battery stream stopped.")

    except BleConnectionError as e:
        logger.error(f"Connection Error: {e}. Ensure the device is powered on and in range.")
    except BleError as e:
        logger.error(f"A BLE SDK error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        logger.info("Example finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.") 
