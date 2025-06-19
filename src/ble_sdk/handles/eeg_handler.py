"""EEG data handler for the ble_sdk package.

This module provides functionality for parsing and processing EEG data
from BLE devices that collect brain activity data.
"""

from typing import Dict, Any, List, Optional, Callable, Union
import datetime

from ..constants import DeviceType
from ..log import sdk_logger as logger
from ..algo.eeg import one_ch_one_sampling_process
from ..utils.cache import cached_config, build_eeg_config


@cached_config(build_eeg_config)
def parse_eeg_data(data: bytearray, device_type: Optional[DeviceType] = None, 
                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Parse EEG data from the device.
    Each packet length should be 198 bytes, includes all channels data, per channel data include N samples.
    
    Data packet format:
    - Header: 5 bytes (0xAA 0xD8 0xC0 0x0C 0x01)
    - Data: num_channels × 8 samples = (num_channels * 8 * 3) bytes
    - Each sample is a 3-byte big-endian signed integer
    - Tail: 0xDD
    
    Args:
        data: Raw data received from the device
        device_type: Device type to get num_channels from DEVICE_CONFIGS
        config: Cached configuration (automatically injected by decorator)
        
    Returns:
        A dictionary containing parsed EEG data:
        {
            "channels": List[List[float]],  # 2D list where each sublist contains the voltage values (in μV) for one channel
            "valid": bool                   # Whether the data is valid
        }
    """
    result = {
        "channels": [[]],   #  2维数组，每个子数组表示一个通道的采样值
        "valid": False
    }

    # Use cached configuration or fallback to defaults
    if not config:
        logger.warning(f"No config available for device_type: {device_type}, using defaults")
        channels_count = 8
        eeg_header = b'\xAA\xD8\xC0\x0C\x01'
        expected_length = 198
        header_size = 5
        footer_size = 1
        sample_size = 3
    else:
        channels_count = config["channels_count"]
        eeg_header = config["frame_header"] + bytes([0xC0, 0x0C, 0x01])
        expected_length = config["expected_length"]
        header_size = config["header_size"]
        footer_size = config["footer_size"]
        sample_size = config["sample_size"]

    # # Fast validation
    # if len(data) < expected_length:
    #     logger.warning(f"Invalid EEG data length: {len(data)}, expected >= {expected_length}")
    #     return [[] for _ in range(channels_count)]

    # if data[:5] != eeg_header:
    #     logger.warning(
    #         f"Invalid EEG frame header: {data[:5].hex(' ')}, expected: {eeg_header.hex(' ')}"
    #     )
    #     return [[] for _ in range(channels_count)]

    try:
        # Calculate samples per channel
        samples_per_channel = (len(data) - header_size - footer_size) // sample_size // channels_count

        # Pre-allocate result lists for better performance
        channels = [[] for _ in range(channels_count)]
        for ch in channels:
            ch.extend([0.0] * samples_per_channel)

        # Extract and process EEG data
        eeg_bytes = data[header_size:-footer_size]
        for i in range(0, len(eeg_bytes), sample_size):
            if i + sample_size - 1 < len(eeg_bytes):
                sample_bytes = eeg_bytes[i:i + sample_size]
                voltage = one_ch_one_sampling_process(sample_bytes.hex())
                ## 业界算法
                # value = int.from_bytes(sample_bytes, byteorder="big", signed=True)
                # voltage = value * 0.0223517  # 1LSB=0.0223517uV

                sample_idx = i // sample_size
                channel_idx = sample_idx // samples_per_channel
                sample_in_channel = sample_idx % samples_per_channel

                if channel_idx < channels_count and sample_in_channel < samples_per_channel:
                    channels[channel_idx][sample_in_channel] = voltage

        logger.debug(f"Parsed EEG data: {channels_count} channels, {samples_per_channel} samples per channel")
        result = {
            "channels": channels,
            "valid": True
        }
        return result

    except Exception as e:
        logger.error(f"Error parsing EEG data: {str(e)}")
        return result


@cached_config(build_eeg_config)
def create_eeg_data_processor(user_callback: Callable[[Union[Dict[str, Any], bytearray]], None], device_type: Optional[DeviceType] = None, config: Optional[Dict[str, Any]] = None, raw_data_only: bool = False) -> Callable[[Any, bytearray], None]:
    """
    Creates a callback for Bleak's start_notify that parses EEG data
    and passes the parsed dictionary or raw data to the user_callback.
    
    Args:
        user_callback: User callback function to receive parsed EEG data or raw bytes
        device_type: Device type to get num_channels from DEVICE_CONFIGS
        config: Cached configuration (automatically injected by decorator)
        raw_data_only: If True, callback receives raw bytearray (header/tail removed).
                       Otherwise, callback receives parsed dictionary.
    """
    # Extract configuration once at processor creation time
    if config:
        eeg_header = config["frame_header"] + bytes([0xC0, 0x0C, 0x01])
        expected_length = config["expected_length"]
        header_size = config["header_size"]
        footer_size = config["footer_size"]
    else:
        logger.warning(f"No config available for device_type: {device_type}, using defaults")
        eeg_header = b'\xAA\xD8\xC0\x0C\x01'
        expected_length = 198
        header_size = 5
        footer_size = 1

    def _validate_eeg_data(data: bytearray) -> bool:
        """Internal validation function for EEG data format.
        
        Args:
            data: Raw EEG data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if len(data) < expected_length:
            logger.warning(f"Invalid EEG data length: {len(data)}, expected >= {expected_length}")
            return False

        if data[:5] != eeg_header:
            logger.warning(f"Invalid EEG frame header: {data[:5].hex(' ')}, expected: {eeg_header.hex(' ')}")
            return False

        return True

    def processor(sender: Any, data: bytearray) -> None:
        logger.debug(f"Received data[len={len(data)}]: {data.hex()}")

        # Validate data format first
        if not _validate_eeg_data(data):
            return

        if raw_data_only:
            # Remove header and tail
            trimmed_data = data[header_size:-footer_size]
            user_callback(trimmed_data)
        else:
            parsed_eeg_data = parse_eeg_data(data, device_type)
            if parsed_eeg_data.get("valid"):
                parsed_eeg_data["timestamp"] = datetime.datetime.now()
                user_callback(parsed_eeg_data)

    return processor


def eeg_builtin_callback(eeg_data: Dict[str, Any], num_channels: Optional[int] = None) -> None:
    """
    Built-in preset callback function for handling parsed EEG data.
    This function receives EEG data in a dictionary format, which contains "channels" and "valid".
    If the data is valid, it logs a preview of samples from each channel. If the data is invalid, it logs a warning.

    Args:
        eeg_data: EEG data in a dictionary format, which contains "channels" and "valid"
        num_channels (Optional[int]): Number of channels (if None, will use len(channels))

    Returns:
        None

    Example:
        >>> eeg_builtin_callback({
        ...     "channels": [
        ...         [12.3, 12.5, 12.4, 12.6, 12.7, 12.8, 12.9, 13.0],
        ...         [20.1, 20.0, 19.9, 19.8, 19.7, 19.6, 19.5, 19.4],
        ...         # ... other channels
        ...     ],
        ...     "valid": True
        ... })
        INFO: EEG Data: 8 channels
        INFO: Channel 1 Sample Values (μV): [12.3, 12.5, 12.4, 12.6, 12.7, 12.8, 12.9, 13.0]
        INFO: Channel 2 Sample Values (μV): [20.1, 20.0, 19.9, 19.8, 19.7, 19.6, 19.5, 19.4]
        ...
    """

    if eeg_data["valid"]:
        channels = eeg_data["channels"]

        actual_num_channels = num_channels if num_channels is not None else len(channels)
        logger.info(f"EEG Data: {actual_num_channels} channels")
        for idx, values in enumerate(channels):
            if values:
                preview = values
                logger.info(
                    f"Channel {idx + 1} Sample Values (μV): {[round(v, 3) for v in preview]}"
                )
    else:
        logger.warning("Received invalid EEG data")


def create_legacy_eeg_handler(num_channels: int = 8) -> Callable[[Any, bytearray], None]:
    """
    Creates a legacy callback for direct use with Bleak's start_notify.
    
    This function is for backward compatibility with examples that use the old pattern
    of directly passing a handler to start_notify instead of using BleClient's start_eeg_stream.
    
    Args:
        num_channels: Number of EEG channels for this device type
        
    Returns:
        A callback function that can be used directly with Bleak's start_notify
    """
    def legacy_handler(sender: Any, data: bytearray) -> None:
        logger.debug(f"Legacy EEG handler received data[len={len(data)}]: {data.hex()}")
        parsed_channels = parse_eeg_data(data, None)  # Use None to trigger default behavior
        if any(parsed_channels):
            eeg_builtin_callback(parsed_channels, num_channels)
        else:
            logger.debug("Legacy EEG handler: No valid channels parsed, not calling eeg_data_handler.")
    
    return legacy_handler
