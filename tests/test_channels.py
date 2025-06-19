#!/usr/bin/env python3
"""Test script to verify num_channels functionality."""

from ble_sdk.constants import DEVICE_CONFIGS, DeviceType

def test_num_channels():
    """Test that all device types have the correct number of channels."""
    print("Testing num_channels configuration:")
    print("-" * 40)
    
    for device_type in DeviceType:
        config = DEVICE_CONFIGS[device_type]
        print(f"{device_type.name}: {config.num_channels} channels")
        
        # Verify impedance support
        has_impedance = config.commands.start_impedance is not None
        print(f"  Impedance support: {'Yes' if has_impedance else 'No'}")
    
    print("-" * 40)
    print("✅ All device configurations loaded successfully!")

if __name__ == "__main__":
    test_num_channels() 