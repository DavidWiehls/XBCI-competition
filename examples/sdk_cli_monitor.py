"""CLI-based BLE Data Stream Monitor.

This script allows users to select and monitor various data streams (battery, EEG,
IMU, impedance, or a mixed stream) from a BLE device via command-line arguments.
"""

import sys
import asyncio
import argparse
from typing import Dict, Any
from loguru import logger

# 配置日志管理器(一定要在带入ble_sdk模块之前)
logger.remove()
logger.add(sys.stderr, level="INFO", filter=lambda record: not record["extra"].get("sdk_internal", False))


from ble_sdk.client import BleClient
from ble_sdk.constants import DeviceType
from ble_sdk.exceptions import BleError, ConnectionError as BleConnectionError

# 导入 sdk 内置的数据处理回调函数（可选）
from ble_sdk.handles.eeg_handler import eeg_builtin_callback  # sdk 内置的 eeg 数据处理回调函数
from ble_sdk.handles.imu_handler import imu_builtin_callback  # sdk 内置的 imu 数据处理回调函数
from ble_sdk.handles.lead_handler import lead_builtin_callback  # sdk 内置的 lead 数据处理回调函数
from ble_sdk.handles.impedance_handler import impedance_builtin_callback  # sdk 内置的 impedance 数据处理回调函数

# 导入 sdk 的日志管理器(可选)
from ble_sdk.log import sdk_logger_manager
sdk_logger_manager.set_level("INFO")


# DEFAULT_DEVICE_ADDRESS = "D8:50:96:79:15:DA"  # jxj_1 - bci_15da
# DEFAULT_DEVICE_ADDRESS = "E5:6A:D5:08:A5:24"  # jxj_2-bci_a524
DEFAULT_DEVICE_ADDRESS = "E9:2C:7B:52:03:EF"  # jxj_4-bci_03ef
# DEFAULT_DEVICE_ADDRESS = "D8:50:96:79:15:DA"  # Replace with your device's address


# 用户自定义的数据处理回调函数
def user_data_handler(data: Dict[str, Any]):
    """Generic callback function to handle incoming data from any stream."""
    logger.info(f"Received Data: {data}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="BLE SDK Data Stream Monitor")
    
    parser.add_argument(
        "--address", "-a",
        type=str,
        default=DEFAULT_DEVICE_ADDRESS,
        help=f"Bluetooth address of the BLE device (default: {DEFAULT_DEVICE_ADDRESS})"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=10,
        help="Duration of data streaming in seconds (default: 10)"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["battery", "eeg", "imu", "impedance", "lead", "mixed"],
        required=True,
        help="Data stream mode to activate."
    )
    parser.add_argument(
        "--raw_data", "-r",
        action='store_true',
        help="Whether to return raw data only. Default: False"
    )
    parser.add_argument(
        "--mixed_primary",
        type=str,
        choices=["eeg", "impedance"],
        default="eeg",
        help="Primary data source for 'mixed' mode (eeg or impedance). Default: eeg."
    )
    parser.add_argument(
        "--device_type",
        type=str,
        default="BLE_8",
        choices=[dt.name for dt in DeviceType], # Use names from DeviceType enum
        help=f"Type of the BLE device (default: BLE_8). Choices: {[dt.name for dt in DeviceType]}"
    )
    return parser.parse_args()


async def main_logic(args):
    """Main logic for connecting, streaming, and disconnecting."""
    
    selected_device_type = DeviceType[args.device_type] # Convert string name to DeviceType enum member
    logger.info(f"Attempting to connect to device: {args.address} (Type: {selected_device_type.name})")
    logger.info(f"Selected stream mode: {args.mode.upper()}")
    if args.mode == "mixed":
        logger.info(f"Mixed stream primary source: {args.mixed_primary.upper()}")
    logger.info(f"Streaming duration: {args.duration} seconds")

    client = BleClient(address=args.address, device_type=selected_device_type)

    try:
        async with client:  # Handles connect and disconnect automatically
            logger.info(f"Successfully connected to {client.address}.")

            stream_active = False
            if args.mode == "battery":
                logger.info("Starting battery stream...")
                await client.start_battery_stream(user_data_handler)
                stream_active = True
            elif args.mode == "eeg":
                logger.info("Starting EEG stream...")
                await client.start_eeg_stream(user_data_handler, raw_data_only=args.raw_data)
                # await client.start_eeg_stream(eeg_builtin_callback)
                stream_active = True
            elif args.mode == "imu":
                logger.info("Starting IMU stream...")
                await client.start_imu_stream(user_data_handler, raw_data_only=args.raw_data)
                # await client.start_imu_stream(imu_builtin_callback)
                stream_active = True
            elif args.mode == "impedance":
                logger.info("Starting Impedance stream...")
                await client.start_impedance_stream(user_data_handler, raw_data_only=args.raw_data)
                stream_active = True
            elif args.mode == "lead":
                logger.info("Starting Lead stream...")
                await client.start_lead_stream(user_data_handler, raw_data_only=args.raw_data)
                stream_active = True
            elif args.mode == "mixed":
                logger.info(f"Starting Mixed stream (Primary: {args.mixed_primary})...")
                await client.start_mixed_stream(mode=args.mixed_primary, user_callback=user_data_handler, raw_main_data_only=args.raw_data)
                stream_active = True
            
            if stream_active:
                logger.info(f"Stream active. Monitoring for {args.duration} seconds...")
                await asyncio.sleep(args.duration)
                logger.info("Monitoring duration ended.")
            else:
                logger.error(f"Invalid mode selected: {args.mode}")
                return # Exit if mode was not matched

            # Stop the stream
            logger.info(f"Stopping {args.mode.upper()} stream...")
            if args.mode == "battery":
                await client.stop_battery_stream()
            elif args.mode == "eeg":
                await client.stop_eeg_stream()
            elif args.mode == "imu":
                await client.stop_imu_stream()
            elif args.mode == "impedance":
                await client.stop_impedance_stream()
            elif args.mode == "mixed":
                await client.stop_mixed_stream()
            logger.info(f"{args.mode.upper()} stream stopped.")

    except BleConnectionError as e:
        logger.error(f"Connection Error: {e}. Ensure the device is on and in range.")
    except BleError as e:
        logger.error(f"A BLE SDK error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.exception("Traceback:") # Log full traceback for unexpected errors
    finally:
        logger.info("CLI monitor finished.")


async def main():
    args = parse_arguments()
    await main_logic(args)

if __name__ == "__main__":
    # Remove the sys.path modification if SDK is installed or PYTHONPATH is set
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(current_dir)
    # if project_root not in sys.path:
    #    sys.path.insert(0, project_root)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.") 
