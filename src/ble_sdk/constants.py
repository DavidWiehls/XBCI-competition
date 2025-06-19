from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional

# Default connection parameters
DEFAULT_CONNECTION_TIMEOUT = 10.0   
# Ble scan timeout
SCAN_TIMEOUT = 15.0
# Ble device_name prefix
DEVICE_PREFIX_TUPLE = ("jxj", "brainup", "music", "xiaomi")


class DeviceType(IntEnum):
    BLE_8 = 8
    BLE_4 = 4
    BLE_2 = 2
    BLE_1 = 1


@dataclass
class DeviceCharacteristics:
    battery: str
    lead: str
    imu: str
    write: str
    eeg: str
    impedance: Optional[str] = None


@dataclass
class DeviceCommands:
    start_eeg: bytes
    start_imu: bytes
    start_lead: bytes
    stop_xxx: bytes
    start_impedance: Optional[bytes] = None
    stop_impedance: Optional[bytes] = None
    adjust_sample_rate: Optional[bytes] = None
    adjust_scaling_factor: Optional[bytes] = None


@dataclass
class DeviceConfig:
    characteristics: DeviceCharacteristics
    commands: DeviceCommands
    num_channels: int  # Number of EEG channels for this device type
    frame_header: bytes # 帧头，用于验证数据包的完整性

BLE_8_CONFIG = DeviceConfig(
    characteristics=DeviceCharacteristics(
        battery="6e400006-b5a3-f393-e0a9-e50e24dcca9e",
        lead="6e400007-b5a3-f393-e0a9-e50e24dcca9e",
        imu="6e400005-b5a3-f393-e0a9-e50e24dcca9e",
        write="6e400002-b5a3-f393-e0a9-e50e24dcca9e",
        eeg="6e400004-b5a3-f393-e0a9-e50e24dcca9e",
        impedance="6e400008-b5a3-f393-e0a9-e50e24dcca9e",
    ),
    commands=DeviceCommands(
        start_eeg=bytes([0xAB, 0x08, 0x01, 0x0C, 0x0A, 0x01, 0xDD]),
        start_imu=bytes([0xAB, 0x08, 0x01, 0x0C, 0x0A, 0x01, 0xDD]),
        start_lead=bytes([0xAB, 0x08, 0x01, 0x0C, 0x0A, 0x01, 0xDD]),
        stop_xxx=bytes([0xAB, 0x08, 0x01, 0x0C, 0x0B, 0x01, 0xDD]),
        start_impedance=bytes([0xAB, 0x08, 0x01, 0x0C, 0x0A, 0x02, 0xDD]),
        stop_impedance=bytes([0xAB, 0x08, 0x01, 0x0C, 0x0B, 0x02, 0xDD]),
    ),
    num_channels=8,
    frame_header=bytes([0xAB, 0x08]),
)

BLE_4_CONFIG = DeviceConfig(
    characteristics=DeviceCharacteristics(
        battery="6e400006-b5a3-f393-e0a9-e50e24dcca9e",
        lead="6e400007-b5a3-f393-e0a9-e50e24dcca9e",
        imu="6e400005-b5a3-f393-e0a9-e50e24dcca9e",
        write="6e400002-b5a3-f393-e0a9-e50e24dcca9e",
        eeg="6e400004-b5a3-f393-e0a9-e50e24dcca9e",
        impedance="6e400008-b5a3-f393-e0a9-e50e24dcca9e",
    ),
    commands=DeviceCommands(
        start_eeg=bytes([0xAB, 0x04, 0x01, 0x0C, 0x0A, 0x01, 0xDD]),
        start_imu=bytes([0xAB, 0x04, 0x01, 0x0C, 0x0A, 0x01, 0xDD]),
        start_lead=bytes([0xAB, 0x04, 0x01, 0x0C, 0x0A, 0x01, 0xDD]),
        stop_xxx=bytes([0xAB, 0x04, 0x01, 0x0C, 0x0B, 0x01, 0xDD]),
        start_impedance=bytes([0xAB, 0x04, 0x01, 0x0C, 0x0A, 0x02, 0xDD]),
        stop_impedance=bytes([0xAB, 0x04, 0x01, 0x0C, 0x0B, 0x02, 0xDD]),
    ),
    num_channels=4,
    frame_header=bytes([0xAB, 0x04]),
)

BLE_2_CONFIG = DeviceConfig(
    characteristics=DeviceCharacteristics(
        battery="6e400006-b5a3-f393-e0a9-e50e24dcca9e",
        lead="6e400007-b5a3-f393-e0a9-e50e24dcca9e",
        imu="6e400005-b5a3-f393-e0a9-e50e24dcca9e",
        write="6e400002-b5a3-f393-e0a9-e50e24dcca9e",
        eeg="6e400004-b5a3-f393-e0a9-e50e24dcca9e",
    ),
    commands=DeviceCommands(
        start_eeg=bytes([0xAB, 0x02, 0x01, 0x0C, 0x0A, 0x01, 0xDD]),
        start_imu=bytes([0xAB, 0x02, 0x01, 0x0C, 0x0A, 0x01, 0xDD]),
        start_lead=bytes([0xAB, 0x02, 0x01, 0x0C, 0x0A, 0x01, 0xDD]),
        stop_xxx=bytes([0xAB, 0x02, 0x01, 0x0C, 0x0B, 0x01, 0xDD]),
        start_impedance=None,  # 2通道设备不支持阻抗测量
        stop_impedance=None,   # 2通道设备不支持阻抗测量
    ),
    num_channels=2,
    frame_header=bytes([0xAB, 0x02]),
)

BLE_1_CONFIG = DeviceConfig(
    characteristics=DeviceCharacteristics(
        battery="6e400006-b5a3-f393-e0a9-e50e24dcca9e",
        lead="6e400007-b5a3-f393-e0a9-e50e24dcca9e",
        imu="6e400005-b5a3-f393-e0a9-e50e24dcca9e",
        write="6e400002-b5a3-f393-e0a9-e50e24dcca9e",
        eeg="6e400004-b5a3-f393-e0a9-e50e24dcca9e",
    ),
    commands=DeviceCommands(
        start_eeg=bytes([0xAB, 0x01, 0x01, 0x0C, 0x0A, 0x01, 0xDD]),
        start_imu=bytes([0xAB, 0x01, 0x01, 0x0C, 0x0A, 0x01, 0xDD]),
        start_lead=bytes([0xAB, 0x01, 0x01, 0x0C, 0x0A, 0x01, 0xDD]),
        stop_xxx=bytes([0xAB, 0x01, 0x01, 0x0C, 0x0B, 0x01, 0xDD]),
        start_impedance=None,  # 1通道设备不支持阻抗测量
        stop_impedance=None,   # 1通道设备不支持阻抗测量
        
        # 1个字节表示型号（0x01蓝牙；0x02WiFi），1个字节表示更改采样率数值（0x01=250HZ,0x02=500Hz,0x03=1000Hz）
        adjust_sample_rate=bytes([0xAB, 0x01, 0x02, 0x0E, 0x01, 0x01, 0x02, 0xDD]),
        # 1字节；1个字节表示放大倍数数值（0x01=放大1倍,0x02=放大2倍,0x03=放大4倍,0x04=放大8倍,0x05=放大12倍,0x06=放大24倍）
        adjust_scaling_factor=bytes([0xAB, 0x01, 0x01, 0x0E, 0x02, 0x06, 0xDD]),
    ),
    num_channels=1,
    frame_header=bytes([0xAB, 0x01]),
)

# 字节位置	值	说明
# 1-2	0xAB 0x01	固定包头
# 3	0x01	数据长度（Len）：1字节（仅0x05一个数据字节）
# 4	0x0E	指令类型：参数下发指令（0x0E）
# 5	0x02	参数类型：0x02表示放大倍数
# 6	0x05	放大倍数值：0x05对应12倍（协议定义见下表）
# 7	0xDD	固定包尾

# 字节位置	值（Hex）	说明
# 1-2	0xAB 0x01	包头：固定标识符，表示指令起始。
# 3	0x01	数据长度（Len）：0x01表示后续数据部分长度为1字节（即0x01）。
# 4	0x0C	指令类型：0x0C表示这是数据指令（控制数据传输开关）。
# 5	0x0A	参数类型：0x0A表示开启EEG数据传输。
# 6	0x01	数据值：0x01表示"开启"操作（0x00或其他值可能无效）。
# 7	0xDD	包尾：固定标识符，表示指令结束。


# 所有设备配置映射
DEVICE_CONFIGS: Dict[DeviceType, DeviceConfig] = {
    DeviceType.BLE_8: BLE_8_CONFIG,
    DeviceType.BLE_4: BLE_4_CONFIG,
    DeviceType.BLE_2: BLE_2_CONFIG,
    DeviceType.BLE_1: BLE_1_CONFIG,
}


if __name__ == "__main__":
    ble8_config = DEVICE_CONFIGS[DeviceType.BLE_8]
    print("BLE 8 Battery UUID:", ble8_config.characteristics.battery)
    print("BLE 8 Start EEG Command:", ble8_config.commands.start_eeg)
