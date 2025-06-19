# BLE SDK API 结构

本文档详细描述 BLE SDK 的 API 结构和使用方法。

## 主要类和函数

### BleClient 类

`BleClient` 是 SDK 的核心类，提供了与 BLE 设备交互的所有功能。

```python
from ble_sdk import BleClient

client = BleClient(address=DEFAULT_DEVICE_ADDRESS, device_type=DeviceType.BLE_8)
```

#### 主要方法

- `connect(timeout=10.0)`
  - **描述**: 连接到 BLE 设备。此方法会尝试与指定地址的 BLE 设备建立连接。
  - **参数**:
    - `timeout` (float, 可选): 连接超时时间，单位为秒。默认为 `DEFAULT_CONNECTION_TIMEOUT`。
  - **返回值**: `bool` - 如果成功连接则返回 `True`，否则返回 `False`。
  - **抛出**: `ConnectionError` - 如果连接失败或超时。

- `disconnect()`
  - **描述**: 断开与 BLE 设备的连接。此方法会终止当前的 BLE 连接。
  - **返回值**: `bool` - 如果成功断开连接则返回 `True`，否则返回 `False`。
  - **抛出**: `DisconnectionError` - 如果断开连接失败。

- `start_notify(char_uuid, callback)`
  - **描述**: 启用特定特征的通知。设备会周期性地通过此特征发送数据，数据将通过回调函数处理。
  - **参数**:
    - `char_uuid` (str): 需要启用通知的特征的 UUID。通常使用预定义的常量如 `TX3_CHARACTERISTIC_UUID`。
    - `callback` (Callable): 一个异步回调函数，用于处理接收到的通知数据。该函数应接受一个参数，即包含通知数据的字典。
  - **抛出**: `NotificationError` - 如果启用通知失败。

- `stop_notify(char_uuid)`
  - **描述**: 停止特定特征的通知。
  - **参数**:
    - `char_uuid` (str): 需要停止通知的特征的 UUID。
  - **抛出**: `NotificationError` - 如果停止通知失败。

- `write_gatt_char(char_uuid='', data=None, response=True)`
  - **描述**: 写入 GATT 特性。
  - **参数**:
    - `char_uuid` (str): 要写入的特性 UUID。
    - `data` (bytes): 要发送的数据。
    - `response` (bool): 如果为 `True`，则需要响应；否则不等待响应。
  - **返回值**: `bool` - 如果写入成功则返回 `True`，否则返回 `False`。
  - **抛出**: `WriteError` - 写入特性失败时抛出。

- `start_battery_stream(user_callback)`
  - **描述**: 开始传输电池电量和 RSSI 数据。
  - **参数**:
    - `user_callback` (Callable): 用于接收解析后的电池和 RSSI 数据的回调函数。
  - **返回值**: `bool` - 如果成功启动传输则返回 `True`，否则返回 `False`。

- `stop_battery_stream()`
  - **描述**: 停止传输电池电量和 RSSI 数据。
  - **返回值**: `bool` - 如果成功停止传输则返回 `True`，否则返回 `False`。

- `start_eeg_stream(user_callback)`
  - **描述**: 开始传输 EEG 数据。
  - **参数**:
    - `user_callback` (Callable): 用于接收解析后的 EEG 数据的回调函数。
  - **返回值**: `bool` - 如果成功启动传输则返回 `True`，否则返回 `False`。

- `stop_eeg_stream()`
  - **描述**: 停止传输 EEG 数据。
  - **返回值**: `bool` - 如果成功停止传输则返回 `True`，否则返回 `False`。

- `start_impedance_stream(user_callback)`
  - **描述**: 开始传输电极阻抗数据。
  - **参数**:
    - `user_callback` (Callable): 用于接收解析后的阻抗数据的回调函数。
  - **返回值**: `bool` - 如果成功启动传输则返回 `True`，否则返回 `False`。

- `stop_impedance_stream()`
  - **描述**: 停止传输电极阻抗数据。
  - **返回值**: `bool` - 如果成功停止传输则返回 `True`，否则返回 `False`。

- `start_imu_stream(user_callback)`
  - **描述**: 开始传输 IMU (加速度计和陀螺仪) 数据。
  - **参数**:
    - `user_callback` (Callable): 用于接收解析后的 IMU 数据的回调函数。
  - **返回值**: `bool` - 如果成功启动传输则返回 `True`，否则返回 `False`。

- `stop_imu_stream()`
  - **描述**: 停止传输 IMU 数据。
  - **返回值**: `bool` - 如果成功停止传输则返回 `True`，否则返回 `False`。

- `start_lead_stream(user_callback)`
  - **描述**: 开始传输导联状态数据。
  - **参数**:
    - `user_callback` (Callable): 用于接收解析后的导联状态数据的回调函数。
  - **返回值**: `bool` - 如果成功启动传输则返回 `True`，否则返回 `False`。

- `stop_lead_stream()`
  - **描述**: 停止传输导联状态数据。
  - **返回值**: `bool` - 如果成功停止传输则返回 `True`，否则返回 `False`。

- `start_mixed_stream(mode, user_callback)`
  - **描述**: 开始传输混合数据流 (EEG/阻抗, IMU, 导联)。
  - **参数**:
    - `mode` (str): 主要数据流模式 ('eeg' 或 'impedance')。
    - `user_callback` (Callable): 用于接收捆绑混合数据的回调函数。
  - **返回值**: `bool` - 如果所有流都成功启动则返回 `True`，否则返回 `False`。

- `stop_mixed_stream()`
  - **描述**: 停止混合数据流。
  - **返回值**: `bool` - 如果成功停止所有流则返回 `True`，否则返回 `False`。

BleClient 类支持异步上下文管理器：

```python
async with BleClient(device_address) as client:
    # 设备已连接
    await client.start_notify(callback=data_callback)
    # 执行其他操作
# 退出上下文后，设备自动断开连接
```

### 工具函数

- `to_int8(value)`
  - **描述**: 将 8 位无符号整数转换为 8 位有符号整数。
  - **参数**: `value` (int) - 8 位无符号整数。
  - **返回值**: `int` - 转换后的 8 位有符号整数。

- `create_notification_handler(callback)`
  - **描述**: 创建一个通知处理函数，该函数将从 BLE 通知中解析原始字节数据并将其传递给用户提供的回调函数。
  - **参数**: `callback` (Callable) - 用户提供的异步回调函数，用于处理解析后的数据。
  - **返回值**: `Callable` - 可直接用于 `start_notify` 的通知处理函数。

#### 通知处理示例

```python
from ble_sdk import BleClient, create_notification_handler, TX3_CHARACTERISTIC_UUID
import asyncio

async def my_data_callback(data):
    """
    处理从 BLE 设备接收到的数据。
    """
    print(f"Received data: Battery Level = {data.get('battery_level')}%, RSSI = {data.get('rssi')} dBm")

async def main():
    device_address = "XX:XX:XX:XX:XX:XX" # 替换为您的设备地址
    async with BleClient(device_address) as client:
        # 创建通知处理函数
        handler = create_notification_handler(my_data_callback)
        
        # 启动通知
        await client.start_notify(TX3_CHARACTERISTIC_UUID, handler)
        print("Notifications started. Waiting for data...")
        
        # 在这里执行其他操作，或者保持连接活跃
        await asyncio.sleep(60) # 等待60秒接收数据
        
        # 停止通知 (如果需要)
        await client.stop_notify(TX3_CHARACTERISTIC_UUID)
        print("Notifications stopped.")

if __name__ == "__main__":
    asyncio.run(main())
```

### 常量

- `DEFAULT_CONNECTION_TIMEOUT` - 建立连接时的默认超时时间（秒）。
**用途**: 这些常量为 SDK 提供了预定义的值，确保在不同模块和功能中保持一致性。

### 异常类

所有异常继承自基础的 `BleError` 类：

- `ConnectionError` - 连接失败时抛出。
- `DisconnectionError` - 断开连接失败时抛出。
- `NotificationError` - 启用或停止通知失败时抛出。
- `DeviceNotFoundError` - 未找到指定地址的设备时抛出。
- `WriteError` - 写入特性失败时抛出。

#### 异常处理示例

```python
from ble_sdk import BleClient, ConnectionError, DeviceNotFoundError
import asyncio

async def connect_to_device_safely(device_address):
    try:
        async with BleClient(device_address) as client:
            print(f"Successfully connected to {device_address}")
            # ... 执行其他操作 ...
    except ConnectionError as e:
        print(f"Connection failed: {e}")
    except DeviceNotFoundError as e:
        print(f"Device not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_device_address = "XX:XX:XX:XX:XX:XX" # 替换为您的设备地址
    asyncio.run(connect_to_device_safely(test_device_address))
```

## 数据格式

通知回调函数接收一个包含以下键的字典：

```python
{
    "battery_level": 85,  # 电池电量百分比 (0-100)
    "rssi": -75,         # 信号强度 (dBm)
}
```