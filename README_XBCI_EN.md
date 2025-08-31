# BLE SDK

A Python SDK for communicating with BLE (Bluetooth Low Energy) IoT devices.

## Features

- **Simple and Easy-to-Use API**: Easily start and stop various data streams through high-level methods
- **Multiple Data Support**: Built-in parsing for battery/RSSI, EEG, impedance, IMU, and lead status
- **Raw Data Mode**: Return raw hexadecimal data without frame headers for custom processing
- **Mixed Data Mode**: Synchronously collect multiple data types through a single callback
- **Asynchronous Design**: Fully based on `asyncio` with proper resource management
- **Flexible Callback Mechanism**: Support both parsed and raw data modes
- **Comprehensive Exception Handling**: Clear exception classes for error handling
- **Isolated Logging System**: SDK logs don't interfere with application logs
- **Dynamic Channel Configuration**: Support 1, 2, 4, 8 channel devices
- **Command Line Monitoring Tool**: Out-of-the-box CLI tool for testing

## Installation

```bash
# From PyPI (when available)
pip install ble-sdk

# From source for development
git clone https://github.com/your-repo/ble_sdk.git
cd ble_sdk
pip install -e .
```

## Quick Start

### Basic Usage

```python
import asyncio
from ble_sdk.client import BleClient
from ble_sdk.constants import DeviceType

DEVICE_ADDRESS = "D2:43:D5:88:4D:9A"  # Replace with your device address

def data_handler(data):
    print(f"Received: {data}")

async def main():
    client = BleClient(address=DEVICE_ADDRESS, device_type=DeviceType.BLE_8)
    async with client:
        await client.start_battery_stream(data_handler)
        await asyncio.sleep(10)
        await client.stop_battery_stream()

asyncio.run(main())
```

### Raw Data Mode

```python
def raw_handler(data):
    if isinstance(data, bytearray):
        print(f"Raw data: {data.hex()}")
    else:
        print(f"Parsed: {data}")

# Use raw_data_only=True for raw data
await client.start_eeg_stream(raw_handler, raw_data_only=True)
```

### Mixed Data Stream

```python
def mixed_handler(data):
    print(f"Type: {data['type']}")
    print(f"EEG: {data.get('eeg_data')}")
    print(f"IMU: {data.get('imu_data')}")

# Collect EEG + IMU + Lead data simultaneously
await client.start_mixed_stream("eeg", mixed_handler)
```

## Command Line Tool

```bash
# Start EEG stream for 20 seconds
python examples/sdk_cli_monitor.py --mode eeg --duration 20

# Raw data mode
python examples/sdk_cli_monitor.py --mode eeg --raw_data

# Mixed mode with impedance
python examples/sdk_cli_monitor.py --mode mixed --mixed_primary impedance

# View all options
python examples/sdk_cli_monitor.py --help
```

## Finding Your Device Address

Use the provided script to discover your Bluetooth device:

```bash
python find_device_address.py
```

This will scan for compatible devices and show their addresses. Make sure your device is turned on and in pairing mode.

## API Reference

### BleClient Methods

- `start_eeg_stream(callback, raw_data_only=False)`
- `start_imu_stream(callback, raw_data_only=False)`
- `start_impedance_stream(callback, raw_data_only=False)`
- `start_lead_stream(callback, raw_data_only=False)`
- `start_battery_stream(callback)`
- `start_mixed_stream(mode, callback, raw_main_data_only=False)`

### Device Types

- `DeviceType.BLE_8` - 8-channel device
- `DeviceType.BLE_4` - 4-channel device  
- `DeviceType.BLE_2` - 2-channel device
- `DeviceType.BLE_1` - 1-channel device

## Best Practices

1. **Use async context managers**: Always use `async with client:` for proper cleanup
2. **Handle exceptions**: Wrap operations in try-catch blocks
3. **Choose data mode appropriately**: Use parsed mode for analysis, raw mode for custom processing
4. **Implement efficient callbacks**: Keep callbacks fast to avoid blocking
5. **Use logging**: Configure proper logging for debugging

## Project Structure

```
ble_sdk/
├── examples/             # Example scripts
├── src/ble_sdk/         # Source code
│   ├── client.py        # Main BleClient class
│   ├── scanner.py       # Device discovery
│   ├── algo/            # Data processing algorithms
│   ├── handles/         # Data stream handlers
│   └── constants.py     # Device configurations
├── tests/               # Test suite
└── docs/                # Documentation
```

## License

[MIT](LICENSE)
