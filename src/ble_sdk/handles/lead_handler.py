"""Lead status data handler for the ble_sdk package.

This module provides functionality for parsing and processing lead status data
from BLE devices that monitor electrode connections.
"""

from typing import List, Tuple, Dict, Callable, Any, Optional, Union
import datetime

from ..constants import DeviceType
from ..log import sdk_logger as logger
from ..utils.cache import cached_config, build_lead_config


@cached_config(build_lead_config)
def parse_lead_status(data: bytearray, device_type: Optional[DeviceType] = None,
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Parse lead status data from the device.
    
    Args:
        data: Raw data received from the device
        device_type: Device type (currently not used in lead parsing, but kept for consistency)
        config: Cached configuration (automatically injected by decorator)
        
    Returns:
        A dictionary containing parsed lead status information:
        {
            "raw_value": int,      # Raw lead status value (16-bit)
            "p_status": List[int], # Status of P electrodes (1P-8P), 1=connected, 0=disconnected
            "n_status": List[int], # Status of N electrodes (1N-8N), 1=connected, 0=disconnected
            "valid": bool          # Whether the data is valid
        }
    """
    result = {
        "raw_value": 0,
        "p_status": [0] * 8,
        "n_status": [0] * 8,
        "valid": False
    }

    # Use cached configuration or fallback to defaults
    if config:
        frame_header = config["frame_header"]
        frame_tail = config["frame_tail"]
        expected_length = config["expected_length"]
        header_size = config["header_size"]
        channels_count = config["channels_count"]
        lead_command = config["lead_command"]
    else:
        logger.warning(f"No config available for device_type: {device_type}, using defaults")
        frame_header = bytes([0xAA, 0xD8])
        frame_tail = 0xDD
        expected_length = 8
        header_size = 5
        channels_count = 8
        lead_command = frame_header + bytes([0x02, 0x0C, 0x03])
        ## frame_header + 长度  + 命令码
        ## 0xAA 0xD8     0x02   0x0C 0x03

    # # Validate data length
    # if len(data) != expected_length:
    #     logger.warning(f"Invalid data length: {len(data)}, expected {expected_length}")
    #     return result

    # # Validate frame header and command
    # if data[:2] != frame_header or data[-1] != frame_tail:
    #     logger.warning(f"Invalid frame header/tail: {data[:2].hex()} {data[-1].hex()}, expected {frame_header.hex()} {frame_tail.hex()}")
    #     return result

    # # command = (data[3], data[4])
    # if data[:5] != lead_command:  # 导联状态命令码
    #     logger.warning(f"Invalid lead status command: {data[:5].hex()}")
    #     return result

    p_byte = data[header_size]
    n_byte = data[header_size + 1]
    lead_status = (p_byte << 8) | n_byte

    # Parse P electrodes (1P-8P) status - bits 0-7
    p_status = [(p_byte >> i) & 0x01 for i in range(8)]

    # Parse N electrodes (1N-8N) status - bits 8-15
    n_status = [(n_byte >> i) & 0x01 for i in range(8)]

    # Pad with zeros if channels_count < 8
    while len(p_status) < 8:
        p_status.append(0)
    while len(n_status) < 8:
        n_status.append(0)

    result = {
        "raw_value": lead_status,
        "p_status": p_status,
        "n_status": n_status,
        "valid": True
    }
    logger.debug(f"Parsed lead status: 0x{lead_status:04X}, P: {p_status}, N: {n_status}")
    return result
    # lead_status 的使用
    # p_byte = (lead_status >> 8) & 0xFF   # 取高 8 位
    # n_byte = lead_status & 0xFF          # 取低 8 位
    # print(f"p_byte = {p_byte} → {format(p_byte, '08b')}")
    # print(f"n_byte = {n_byte} → {format(n_byte, '08b')}")


@cached_config(build_lead_config)
def create_lead_data_processor(user_callback: Callable[[Union[Dict[str, Any], bytearray]], None], device_type: Optional[DeviceType] = None, config: Optional[Dict[str, Any]] = None, raw_data_only: bool = False) -> Callable[[Any, bytearray], None]:
    """
    Creates a callback for Bleak's start_notify that parses lead status data
    and passes the parsed dictionary or raw data to the user_callback.
    
    Args:
        user_callback: User callback function to receive parsed lead status data or raw bytes
        device_type: Device type (passed to parse_lead_status for consistency)
        config: Cached configuration (automatically injected by decorator)
        raw_data_only: If True, callback receives raw bytearray (header/tail removed).
                       Otherwise, callback receives parsed dictionary.
    """
    # Extract configuration once at processor creation time
    if config:
        frame_header = config["frame_header"]
        frame_tail = config["frame_tail"]
        expected_length = config["expected_length"]
        header_size = config["header_size"]
        lead_command = config["lead_command"]
    else:
        logger.warning(f"No config available for device_type: {device_type}, using defaults")
        frame_header = bytes([0xAA, 0xD8])
        frame_tail = 0xDD
        expected_length = 8
        header_size = 5
        lead_command = frame_header + bytes([0x02, 0x0C, 0x03])

    def _validate_lead_data(data: bytearray) -> bool:
        """Internal validation function for lead data format.
        
        Args:
            data: Raw lead data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if len(data) != expected_length:
            logger.warning(f"Invalid lead data length: {len(data)}, expected {expected_length}")
            return False

        if data[:2] != frame_header or data[-1] != frame_tail:
            logger.warning(f"Invalid lead frame header/tail: {data[:2].hex()} {data[-1].hex()}, expected {frame_header.hex()} {frame_tail.hex()}")
            return False

        if data[:5] != lead_command:
            logger.warning(f"Invalid lead status command: {data[:5].hex()}")
            return False

        return True

    def processor(sender: Any, data: bytearray) -> None:
        logger.debug(f"Received data[len={len(data)}]: {data.hex()}")

        # Validate data format first
        if not _validate_lead_data(data):
            return

        if raw_data_only:
            # Remove header and tail
            trimmed_data = data[header_size:-1]
            user_callback(trimmed_data)
        else:
            parsed_data = parse_lead_status(data, device_type)
            if parsed_data.get("valid"):
                parsed_data["timestamp"] = datetime.datetime.now()
                user_callback(parsed_data)

    return processor


# Direct function that can be used as a notification handler
def lead_builtin_callback(lead_data: Dict[str, Any]) -> None:
    """
    Built-in preset callback function for handling parsed lead status data.
    This function receives lead status data in a dictionary format, which contains "raw_value", "p_status", "n_status", and "valid".
    If the data is valid, it logs the lead status, P electrodes, and N electrodes. If the data is invalid, it logs a warning.
    """
    
    if lead_data["valid"]:
        logger.info(f"Lead Data:")
        logger.info(f"  Raw Value: 0x{lead_data['raw_value']:04X}")
        logger.info(f"  P Electrode Status (1P-8P): {lead_data['p_status']}")
        logger.info(f"  N Electrode Status (1N-8N): {lead_data['n_status']}")
    else:
        logger.warning("Received invalid lead status data")
