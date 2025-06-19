import sys
import asyncio
from loguru import logger

# 配置上层应用日志，排除SDK日志 （必须在导入SDK之前）
logger.remove()
logger.add(sys.stderr, level="INFO", filter=lambda record: not record["extra"].get("sdk_internal", False))  # 排除SDK日志
logger.info("application log init ...")

# 导入SDK
from ble_sdk.scanner import scan_ble_devices
# 设置SDK日志级别(可选)
from ble_sdk.log import sdk_logger_manager
sdk_logger_manager.set_level("DEBUG")


async def main():
    logger.info("Starting BLE device scan...")
    res = await scan_ble_devices()
    if not res:
        logger.warning("No BLE devices found")
        return
    for device in res:
        logger.info(f"Device: {device}")


if __name__ == "__main__":
    asyncio.run(main())
