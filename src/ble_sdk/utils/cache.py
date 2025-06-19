"""Configuration caching utilities for BLE SDK handlers.

This module provides decorators and utilities to cache device configurations
for improved performance in data parsing functions.
"""

from functools import wraps
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from ..constants import DEVICE_CONFIGS, DeviceType

T = TypeVar('T')

class ConfigCache(Generic[T]):
    """Generic configuration cache manager."""
    
    def __init__(self, config_builder: Callable[[Optional[DeviceType]], Optional[T]]):
        """Initialize cache with a config builder function.
        
        Args:
            config_builder: Function that builds config from device_type
        """
        self._cache: Dict[Optional[DeviceType], Optional[T]] = {}
        self._config_builder = config_builder
    
    def get(self, device_type: Optional[DeviceType]) -> Optional[T]:
        """Get configuration, build and cache if not exists.
        
        Args:
            device_type: Device type to get config for
            
        Returns:
            Cached or newly built configuration
        """
        if device_type not in self._cache:
            self._cache[device_type] = self._config_builder(device_type)
        return self._cache[device_type]
    
    def clear(self):
        """Clear all cached configurations."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "total_entries": len(self._cache),
            "cached_device_types": list(self._cache.keys()),
            "cache_size_bytes": sum(
                len(str(config)) if config else 0 
                for config in self._cache.values()
            )
        }

def cached_config(config_builder: Callable[[Optional[DeviceType]], Optional[Dict[str, Any]]]):
    """Universal configuration caching decorator.
    
    This decorator caches device configurations to avoid repeated lookups
    and computations in data parsing functions.
    
    Args:
        config_builder: Function that builds configuration from device_type
        
    Returns:
        Decorator function
        
    Example:
        @cached_config(build_eeg_config)
        def parse_eeg_data(data, device_type=None, config=None):
            # config parameter will be automatically injected
            pass
    """
    def decorator(func):
        cache = ConfigCache(config_builder)
        
        @wraps(func)
        def wrapper(data: bytearray, device_type: Optional[DeviceType] = None, *args, **kwargs):
            config = cache.get(device_type)
            return func(data, device_type, config, *args, **kwargs)
        
        # Expose cache management interfaces
        wrapper.clear_cache = cache.clear
        wrapper.get_cache_stats = cache.get_stats
        wrapper._cache_instance = cache  # For advanced usage
        
        return wrapper
    return decorator

# Predefined configuration builders for different handler types

def build_eeg_config(device_type: Optional[DeviceType]) -> Optional[Dict[str, Any]]:
    """Build EEG-specific configuration.
    
    Args:
        device_type: Device type to build config for
        
    Returns:
        EEG configuration dictionary or None if device_type is invalid
    """
    if not device_type:
        return None
    
    device_config = DEVICE_CONFIGS.get(device_type)
    if not device_config:
        return None
    
    return {
        "channels_count": device_config.num_channels,
        "frame_header": device_config.frame_header,
        "expected_length": 198, 
        "header_size": 5,
        "footer_size": 1,
        "sample_size": 3,
    }

def build_impedance_config(device_type: Optional[DeviceType]) -> Optional[Dict[str, Any]]:
    """Build impedance-specific configuration.
    
    Args:
        device_type: Device type to build config for
        
    Returns:
        Impedance configuration dictionary or None if device_type is invalid
    """
    if not device_type:
        return None
    
    device_config = DEVICE_CONFIGS.get(device_type)
    if not device_config:
        return None
    
    return {
        "channel_count": device_config.num_channels,
        "frame_header": device_config.frame_header,
        "valid_channel_range": (0xC1, 0xC1 + device_config.num_channels - 1),
        "header_size": 3,
        "expected_length": 244,  # 每1个通道： 240 = 3 * 80 (3bytes * 80 次采样) + header(3bytes) + footer(1byte)
    }

def build_imu_config(device_type: Optional[DeviceType]) -> Optional[Dict[str, Any]]:
    """Build IMU-specific configuration.
    
    Args:
        device_type: Device type to build config for
        
    Returns:
        IMU configuration dictionary or None if device_type is invalid
    """
    if not device_type:
        return None
    
    device_config = DEVICE_CONFIGS.get(device_type)
    if not device_config:
        return None
    
    return {
        "frame_header": device_config.frame_header,
        "frame_tail": 0xDD,
        "header_size": 5,
        "expected_length": 18,
        "acc_sensitivity": 0.00059814,  # m/s^2 per LSB
        "gyro_sensitivity": 0.061037,   # °/s per LSB
    }

def build_lead_config(device_type: Optional[DeviceType]) -> Optional[Dict[str, Any]]:
    """Build lead status-specific configuration.
    
    Args:
        device_type: Device type to build config for
        
    Returns:
        Lead configuration dictionary or None if device_type is invalid
    """
    if not device_type:
        return None
    
    device_config = DEVICE_CONFIGS.get(device_type)
    if not device_config:
        return None
    
    return {
        "channels_count": device_config.num_channels,
        "frame_header": device_config.frame_header,
        "frame_tail": 0xDD,
        "expected_length": 8,
        "header_size": 5,
        "lead_command": device_config.frame_header + bytes([0x02, 0x0C, 0x03]),
    }

def build_battery_rssi_config(device_type: Optional[DeviceType]) -> Optional[Dict[str, Any]]:
    """Build battery/RSSI-specific configuration.
    
    Args:
        device_type: Device type to build config for
        
    Returns:
        Battery/RSSI configuration dictionary or None if device_type is invalid
    """
    if not device_type:
        return None

    device_config = DEVICE_CONFIGS.get(device_type)
    if not device_config:
        return None

    return {
        "frame_header": device_config.frame_header,
        "sub_instruction_pos": 4,
    }
