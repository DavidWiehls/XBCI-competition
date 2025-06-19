"""IMU data handler for the ble_sdk package.

This module provides functionality for parsing and processing IMU data
from BLE devices that collect motion data.
"""

from typing import Dict, Any, Optional, Callable, Union
import datetime

from ..constants import DeviceType
from ..log import sdk_logger as logger
from ..utils.cache import cached_config, build_imu_config


@cached_config(build_imu_config)
def parse_imu_data(data: bytearray, device_type: Optional[DeviceType] = None,
                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Parse IMU data from the device.
    
    Data packet format (total 18 bytes):
    AA D8 0C 0C 02 [AccX(2)] [AccY(2)] [AccZ(2)] [GyroX(2)] [GyroY(2)] [GyroZ(2)] DD
    Acc/Gyro values are 2-byte little-endian signed integers.
    
    Args:
        data: Raw data received from the device
        device_type: Device type (currently not used in IMU parsing, but kept for consistency)
        config: Cached configuration (automatically injected by decorator)
        
    Returns:
        A dictionary containing parsed IMU data:
        {
            "acceleration": (acc_x, acc_y, acc_z),  # in m/s²
            "gyroscope": (gyro_x, gyro_y, gyro_z),  # in °/s
            "valid": bool                           # Whether the data is valid
        }
    """
    result = {
        "acceleration": (0.0, 0.0, 0.0),
        "gyroscope": (0.0, 0.0, 0.0),
        "valid": False
    }

    # Use cached configuration or fallback to defaults
    if config:
        acc_sensitivity = config["acc_sensitivity"]
        gyro_sensitivity = config["gyro_sensitivity"]
    else:
        logger.warning(f"No config available for device_type: {device_type}, using defaults")
        acc_sensitivity = 0.00059814
        gyro_sensitivity = 0.061037

    try:
        # Extract acceleration data (in m/s²)
        acc_x = int.from_bytes(data[5:7], byteorder='little', signed=True) * acc_sensitivity
        acc_y = int.from_bytes(data[7:9], byteorder='little', signed=True) * acc_sensitivity
        acc_z = int.from_bytes(data[9:11], byteorder='little', signed=True) * acc_sensitivity

        # Extract gyroscope data (in °/s)
        gyro_x = int.from_bytes(data[11:13], byteorder='little', signed=True) * gyro_sensitivity
        gyro_y = int.from_bytes(data[13:15], byteorder='little', signed=True) * gyro_sensitivity
        gyro_z = int.from_bytes(data[15:17], byteorder='little', signed=True) * gyro_sensitivity

        result = {
            "acceleration": (round(acc_x, 4), round(acc_y, 4), round(acc_z, 4)),
            "gyroscope": (round(gyro_x, 4), round(gyro_y, 4), round(gyro_z, 4)),
            "valid": True
        }

        logger.debug(f"Parsed IMU data: acc=({acc_x:.2f}, {acc_y:.2f}, {acc_z:.2f}) m/s², gyro=({gyro_x:.2f}, {gyro_y:.2f}, {gyro_z:.2f}) °/s")
        return result

    except Exception as e:
        logger.error(f"Error parsing IMU data: {str(e)}")
        return result


@cached_config(build_imu_config)
def create_imu_data_processor(user_callback: Callable[[Union[Dict[str, Any], bytearray]], None], device_type: Optional[DeviceType] = None, config: Optional[Dict[str, Any]] = None, raw_data_only: bool = False) -> Callable[[Any, bytearray], None]:
    """
    Creates a callback for Bleak's start_notify that parses IMU data
    and passes the parsed dictionary or raw data to the user_callback.
    
    Args:
        user_callback: User callback function to receive parsed IMU data or raw bytes
        device_type: Device type (passed to parse_imu_data for consistency)
        raw_data_only: If True, callback receives raw bytearray (header/tail removed).
                       Otherwise, callback receives parsed dictionary.
        config: Cached configuration (automatically injected by decorator)
    """
    # Extract configuration once at processor creation time
    if config:
        frame_header = config["frame_header"]
        frame_tail = config["frame_tail"] 
        expected_length = config["expected_length"]
        header_size = config["header_size"]
    else:
        logger.warning(f"No config available for device_type: {device_type}, using defaults")
        frame_header = bytes([0xAA, 0xD8])
        frame_tail = 0xDD
        expected_length = 18

    def _validate_imu_data(data: bytearray) -> bool:
        """Internal validation function for IMU data format.
        
        Args:
            data: Raw IMU data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if len(data) != expected_length:
            logger.warning(f"Invalid IMU data length: {len(data)}, expected {expected_length}")
            return False

        if data[:len(frame_header)] != frame_header or data[-1] != frame_tail:
            logger.warning(f"Invalid IMU frame format: header=0x{data[0]:02X}, tail=0x{data[-1]:02X}")
            return False

        return True

    def processor(sender: Any, data: bytearray) -> None:
        logger.debug(f"Received data[len={len(data)}]: {data.hex()}")

        # Validate data format first
        if not _validate_imu_data(data):
            return

        if raw_data_only:
            # Remove header and tail
            trimmed_data = data[header_size:-1]
            user_callback(trimmed_data)
        else:
            parsed_data = parse_imu_data(data, device_type)
            if parsed_data.get("valid"):
                parsed_data["timestamp"] = datetime.datetime.now()
                user_callback(parsed_data)

    return processor


def imu_builtin_callback(imu_data: Dict[str, Any]) -> None:
    """Built-in preset callback function for handling parsed IMU data.
    
    This function receives IMU data in a dictionary format, where the 'valid' key indicates
    whether the data is valid. If the data is valid, it logs the acceleration and gyroscope
    values. If the data is invalid, it logs a warning.
    
    Args:
        imu_data: Parsed IMU data
    """
    
    if imu_data["valid"]:
        acc = imu_data["acceleration"]
        gyro = imu_data["gyroscope"]
        
        logger.info(f"IMU Data:")
        logger.info(f"  Acceleration (m/s²): X={acc[0]:.2f}, Y={acc[1]:.2f}, Z={acc[2]:.2f}")
        logger.info(f"  Gyroscope (°/s): X={gyro[0]:.2f}, Y={gyro[1]:.2f}, Z={gyro[2]:.2f}")
    else:
        logger.warning("Received invalid IMU data") 
