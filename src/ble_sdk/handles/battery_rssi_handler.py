"""Battery and RSSI data handler for the ble_sdk package.

This module provides functionality for parsing and processing battery level
and RSSI (Received Signal Strength Indicator) data from BLE devices.
"""

from typing import Dict, Callable, Any, Optional

from ..log import sdk_logger as logger
from ..constants import DeviceType
from ..utils.cache import cached_config, build_battery_rssi_config


def to_int8(value: int) -> int:
    """Convert 8-bit unsigned integer to 8-bit signed integer (int8).

    Args:
        value: 8-bit unsigned integer (0-255)

    Returns:
        Converted signed integer (-128 to 127)
    """
    return value if value < 0x80 else value - 0x100


@cached_config(build_battery_rssi_config)
def parse_battery_rssi(data: bytearray, device_type: Optional[DeviceType] = None,
                      config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Parse battery level and RSSI data from the device.
    
    Args:
        data: Raw data received from the device
        device_type: Device type (currently not used in battery/RSSI parsing, but kept for consistency)
        config: Cached configuration (automatically injected by decorator)
        
    Returns:
        A dictionary containing parsed information:
        {
            "battery_level": int,  # Battery level percentage (0-100)
            "rssi": int,           # RSSI value in dBm (typically negative)
            "valid": bool          # Whether the data is valid
        }
    """

    # Use cached configuration or fallback to defaults
    if config:
        sub_instruction_pos = config["sub_instruction_pos"]
    else:
        logger.warning(f"No config available for device_type: {device_type}, using defaults")
        sub_instruction_pos = 4

    # Validate data length
    # if len(data) == min_data_length:
    #     logger.warning(f"Invalid data length: {len(data)}, expected at least {min_data_length} bytes")
    #     return result

    result = {"valid": False}
    flag = data[sub_instruction_pos]  # 第5个字节是子指令标识符（0x07 表示 RSSI，0x08 表示电量）
    match flag:
        case 0x02:
            result.update( {
                "device_name": data[sub_instruction_pos + 1 : -1].decode("utf-8"),
                "valid": True,
            })
            logger.debug(f"Parsed Device Name: {result['device_name']}")
        case 0x03:
            result.update({
                "device_id": data[sub_instruction_pos + 1: -1].hex(),
                "valid": True
            })
            logger.debug(f"Parsed Device ID: {result['device_id']}")
        case 0x04:
            result.update( {
                "hardware_version": data[sub_instruction_pos + 1 : -1].decode("ascii"),
                "valid": True,
            })
            logger.debug(f"Parsed Hardware Version: {result['hardware_version']}")
        case 0x05:
            result.update({
                "firmware_version": data[sub_instruction_pos + 1 : -1].decode("ascii"),
                "valid": True,
            })
            logger.debug(f"Parsed Firmware Version: {result['firmware_version']}")
        case 0x06:
            result.update({
                "software_version": data[sub_instruction_pos + 1 : -1].decode("ascii"),
                "valid": True,
            })
            logger.debug(f"Parsed Software Version: {result['software_version']}")
        case 0x07:
            result.update({
                "rssi": to_int8(data[sub_instruction_pos + 1]),
                "valid": True
            })
            logger.debug(f"Parsed RSSI: {result['rssi']} dBm")
        case 0x08:
            result.update({
                "battery_level": data[sub_instruction_pos + 1],
                "valid": True
            })
            logger.debug(f"Parsed Battery Level: {result['battery_level']}%")
        case _:
            logger.warning(f"Unknown sub instruction: {flag}")

    # battery_level = data[battery_offset]  # First byte: Battery level (uint8)
    # rssi_raw = data[rssi_offset]          # Second byte: RSSI (uint8 in [0, 255])
    # rssi = to_int8(rssi_raw)              # Convert to int8

    # result = {
    #     "battery_level": battery_level,
    #     "rssi": rssi,
    #     "valid": True
    # }

    # logger.debug(f"Parsed battery level: {battery_level}%, RSSI: {rssi} dBm")
    return result


def create_battery_rssi_data_processor(user_callback: Callable[[Dict[str, Any]], None], device_type: Optional[DeviceType] = None) -> Callable[[Any, bytearray], None]:
    """
    Creates a callback for Bleak's start_notify that parses battery/RSSI data
    and passes the parsed dictionary to the user_callback.
    
    Args:
        user_callback: User callback function to receive parsed battery/RSSI data
        device_type: Device type (passed to parse_battery_rssi for consistency)
    """

    def processor(sender: Any, data: bytearray) -> None:
        logger.debug(f"Received data[len={len(data)}]: {data.hex()}")
        parsed_data = parse_battery_rssi(data, device_type)
        if parsed_data.get("valid"):
            user_callback(parsed_data)
    return processor
