import asyncio
from bleak import BleakClient, BleakScanner

# 设备地址
DEVICE_ADDRESS = "D2:43:D5:88:4D:9A"

# 特征 UUID
BATTERY_CHAR_UUID = "6e400006-b5a3-f393-e0a9-e50e24dcca9e"


def battery_notification_handler(sender, data):
    print(f"[通知] 收到原始数据 from {sender}: {data.hex(' ')}")

    if len(data) < 7:
        print("数据长度不足")
        return

    # 解析帧头
    header = data[:5]
    if header != b"\xaa\xd8\x0c\x0c\x02":
        print("无效的电量数据帧头")
        return

    battery_level = data[5]
    rssi = int.from_bytes(data[6:7], byteorder="big", signed=True)

    print(f"🔋 电池电量: {battery_level}%")
    print(f"📶 蓝牙信号强度: {rssi} dBm")


async def main():
    print("正在扫描设备...")
    scanner = BleakScanner()
    device = await scanner.find_device_by_address(DEVICE_ADDRESS, timeout=10)
    if not device:
        print("未找到设备！请确认蓝牙连接是否正常。")
        return

    async with BleakClient(device) as client:
        print(f"已连接到设备: {device.address}")

        # 打印服务和特征（可选调试用）
        for service in client.services:
            print(f"[服务] {service.uuid}")
            for char in service.characteristics:
                print(f"  [特征] {char.uuid} | 属性: {char.properties}")

        # 启动电量数据监听
        await client.start_notify(BATTERY_CHAR_UUID, battery_notification_handler)

        print("等待电量数据... 按 Ctrl+C 停止。")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("停止监听...")
            await client.stop_notify(BATTERY_CHAR_UUID)


asyncio.run(main())
