"""Impedance data handler for the ble_sdk package.

This module provides functionality for parsing and processing electrode impedance
data from BLE devices that measure skin-electrode contact quality.
"""

from typing import Dict, Any, List, Optional, Callable, Union
import datetime

from ..constants import DeviceType
from ..log import sdk_logger as logger
from ..algo.impedance import imp_conversion
from ..utils.cache import cached_config, build_impedance_config


@cached_config(build_impedance_config)
def parse_impedance_data(data: bytearray, device_type: Optional[DeviceType] = None,
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Parse impedance data from the device.
    Each packet length should be 244 bytes, includes 1 channel data eg(C1), per channel data include 80 samples.

    Data packet format:
    Starts with 0xAB 0x04 0xC1 ends with 0xDD
    header: 3 bytes (0xAB 0x04 0xC1)    # 0xAB is frame header, 0x04 is channel count, 0xC1 is channel index
    data: 240 bytes (3 bytes * 80 samples) # 3 bytes per sample, 80 samples per channel
    footer: 1 byte (0xDD)

    Args:
        data: Raw data received from the device
        device_type: Device type (currently not used in impedance parsing, but kept for consistency)
        config: Cached configuration (automatically injected by decorator)

    Returns:
        A dictionary containing parsed impedance data:
        {
            "channel_count": int,     # 通道数量
            "channel_index": int,     # 通道索引
            "impedance_value": float, # 阻抗值
            "valid": bool             # Whether the data is invalid
        }
    """
    result = {
        "channel_count": "",
        "channel_index": "",
        "impedance_value": "",
        "valid": False
    }

    # Use cached configuration or fallback to defaults
    if config:
        frame_header = config["frame_header"]
        header_size = config["header_size"]
        channel_count = config["channel_count"]
        valid_channel_range = config["valid_channel_range"]
        expected_length = config["expected_length"]
    else:
        logger.warning(f"No config available for device_type: {device_type}, using defaults")
        frame_header = bytes([0xAA, 0xD8])
        header_size = 3
        valid_channel_range = (0xC1, 0xC8)  # 只处理C1-C8通道

    # # Validate basic frame format
    # if data[:2] != frame_header:
    #     logger.warning(f"Invalid impedance frame format: header={data[:2].hex()}, expected={frame_header.hex()}")
    #     return result

    try:
        # Extract device type and channel index
        header_bytes = data[0:header_size]
        channel_byte = header_bytes[2]

        # # Validate channel range
        # if not (valid_channel_range[0] <= channel_byte <= valid_channel_range[1]):
        #     logger.warning(f"Invalid impedance data format: channel=0x{channel_byte:02X}")
        #     return result

        channel = channel_byte - 0xC0

        impedance_payload = data[header_size:]   
        impedance_value = imp_conversion(impedance_payload.hex())

        result = {
            "channel_count": channel_count,
            "channel_index": channel,
            "impedance_value": impedance_value,
            "valid": True
        }

        logger.debug(f"Parsed impedance data: channel_count={channel_count}, channel_index={channel}, impedance_value={impedance_value}")
        return result

    except Exception as e:
        logger.exception(f"Error parsing impedance data: {str(e)}")
        return result


@cached_config(build_impedance_config)
def create_impedance_data_processor(user_callback: Callable[[Union[Dict[str, Any], bytearray]], None], device_type: Optional[DeviceType] = None, config: Optional[Dict[str, Any]] = None, raw_data_only: bool = False) -> Callable[[Any, bytearray], None]:
    """
    Creates a callback for Bleak's start_notify that parses impedance data
    and passes the parsed dictionary or raw data to the user_callback.
    
    Args:
        user_callback: User callback function to receive parsed impedance data or raw bytes
        device_type: Device type (passed to parse_impedance_data for consistency)
        raw_data_only: If True, callback receives raw bytearray (header/tail removed).
                       Otherwise, callback receives parsed dictionary.
        config: Cached configuration (automatically injected by decorator)
    """
    # Extract configuration once at processor creation time
    if config:
        frame_header = config["frame_header"]
        valid_channel_range = config["valid_channel_range"]
        header_size = config["header_size"]
        expected_length = config["expected_length"]
    else:
        logger.warning(f"No config available for device_type: {device_type}, using defaults")
        frame_header = bytes([0xAA, 0xD8])
        valid_channel_range = (0xC1, 0xC8)
        header_size = 3
        expected_length = 244

    def _validate_impedance_data(data: bytearray) -> bool:
        """Internal validation function for impedance data format.
        
        Args:
            data: Raw impedance data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if data[:2] != frame_header:
            logger.warning(f"Invalid impedance frame format: header={data[:2].hex()}, expected={frame_header.hex()}")
            return False

        if len(data) != expected_length:
            logger.warning(f"Invalid impedance data length: {len(data)}, expected >= {expected_length}")
            return False

        # Validate channel range
        channel_byte = data[2]
        if not (valid_channel_range[0] <= channel_byte <= valid_channel_range[1]):
            logger.warning(f"Invalid impedance data format: channel=0x{channel_byte:02X}")
            return False

        return True

    def processor(sender: Any, data: bytearray) -> None:
        logger.debug(f"Received raw data[len={len(data)}]: {data.hex()}")

        # Validate data format first
        if not _validate_impedance_data(data):
            return

        if raw_data_only:
            # Remove header and tail
            trimmed_data = data[header_size:-1]
            user_callback(trimmed_data)
        else:
            parsed_data = parse_impedance_data(data, device_type)
            if parsed_data.get("valid"):
                parsed_data["timestamp"] = datetime.datetime.now()
                user_callback(parsed_data)

    return processor


def impedance_builtin_callback(impedance_data: Dict[str, Any]) -> None:
    """Built-in preset callback function for handling parsed impedance data.
    
    This function receives impedance data in a dictionary format, where the 'valid' key indicates
    whether the data is valid. If the data is valid, it logs the impedance value. If the data is invalid, it logs a warning.
    
    Args:
        impedance_data: Parsed impedance data
    """
    if impedance_data["valid"]:
        logger.info(f"Impedance Data:")
        logger.info(f"  Channel: {impedance_data['channel_index']}")
        logger.info(f"  Impedance Value: {impedance_data['impedance_value']} Ω")
    else:
        logger.warning("Received invalid impedance data")
