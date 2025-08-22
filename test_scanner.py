import asyncio
import sys
from loguru import logger

# Configure application logging
logger.remove()
logger.add(sys.stderr, level="INFO", filter=lambda record: not record["extra"].get("sdk_internal", False))

# Import SDK components
from ble_sdk.scanner import scan_ble_devices
from ble_sdk.log import sdk_logger_manager

# Set SDK log level
sdk_logger_manager.set_level("INFO")

async def test_scanner():
    """Test the BLE scanner functionality"""
    logger.info("Testing BLE Scanner...")
    
    try:
        devices = await scan_ble_devices()
        
        if not devices:
            logger.warning("No BLE devices found!")
            logger.info("Please ensure:")
            logger.info("1. Your BCI device is turned on")
            logger.info("2. Bluetooth is enabled on your computer")
            logger.info("3. The device is in pairing mode")
            return False
        
        logger.info(f"Found {len(devices)} device(s):")
        for i, device in enumerate(devices):
            logger.info(f"  {i+1}. {device['name']} ({device['address']}) - RSSI: {device['rssi']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        return False

async def main():
    """Main function"""
    logger.info("BLE Scanner Test")
    logger.info("=" * 30)
    
    success = await test_scanner()
    
    if success:
        logger.info("Scanner test completed successfully!")
        logger.info("You can now run the BCI classifier.")
    else:
        logger.error("Scanner test failed. Please check your setup.")

if __name__ == "__main__":
    asyncio.run(main())
