"""BLE client for connecting to and interacting with BLE devices."""

import asyncio
import datetime
from typing import Callable, Optional, Dict, Any, Union, List

from bleak import BleakClient, BleakError

from .log import sdk_logger as logger
from .handles.battery_rssi_handler import create_battery_rssi_data_processor
from .handles.eeg_handler import create_eeg_data_processor, parse_eeg_data
from .handles.impedance_handler import create_impedance_data_processor, parse_impedance_data
from .handles.imu_handler import create_imu_data_processor, parse_imu_data
from .handles.lead_handler import create_lead_data_processor, parse_lead_status
from .constants import DEFAULT_CONNECTION_TIMEOUT, DEVICE_CONFIGS, DeviceType
from .exceptions import ConnectionError, DisconnectionError, NotificationError, DeviceNotFoundError, WriteError


class BleClient:
    """BLE client for connecting and interacting with BLE devices.

    Attributes:
        address: BLE device address
        client: BleakClient instance
        is_connected: Whether the device is connected
        device_config: Configuration for the specified device type
        active_notifications: Currently active notification characteristic UUIDs and callback functions
    """

    def __init__(self, address: str, device_type: DeviceType = DeviceType.BLE_8):
        """Initialize the BLE client.

        Args:
            address: BLE device address
            device_type: Type of the BLE device (e.g., DeviceType.BLE_8)
        """
        self.address = address
        self.device_type = device_type
        self.client: Optional[BleakClient] = None
        self.is_connected = False
        self.active_notifications: Dict[str, Callable] = {}
        self.device_config = DEVICE_CONFIGS.get(device_type)
        if not self.device_config:
            raise ValueError(f"Unsupported device type: {device_type}. No configuration found.")

        # For mixed data stream
        self._mixed_user_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._mixed_latest_imu_data: Optional[Union[Dict[str, Any], bytearray]] = None
        self._mixed_latest_lead_data: Optional[Dict[str, Any]] = None
        self._mixed_current_mode: Optional[str] = None # 'eeg' or 'impedance'

    async def connect(self, timeout: float = DEFAULT_CONNECTION_TIMEOUT) -> bool:
        """Connect to the BLE device.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            Whether the connection was successful

        Raises:
            ConnectionError: When connection fails
        """
        try:
            logger.info(f"Connecting to device {self.address} ...")
            self.client = BleakClient(self.address, timeout=timeout)
            # The next line is where the connection attempt happens.
            self.is_connected = await self.client.connect()
            
            print("self.is_connected")
            #asyncio.sleep(1)
            if not self.is_connected:
                # Add more specific logging here to help debug
                logger.error(f"Bleak client returned 'False' for connection status. This could be due to a timeout, device rejection, or another device already connected.")
                raise ConnectionError(f"Unable to connect to device {self.address}")

            logger.info(f"Successfully connected to device {self.address}")
            return True

        except BleakError as e:
            # The BleakError exception provides detailed information on why the connection failed.
            # You can log this specific error for better confirmation.
            logger.error(f"Error connecting to device {self.address}: {str(e)}")
            if "org.freedesktop.DBus.Error.UnknownObject" in str(e):
                logger.error("This is often a low-level Bluetooth service issue. Try restarting your computer or the device.")
            elif "Connection refused" in str(e):
                logger.error("The device is likely already connected to another host. Please disconnect it.")

            raise ConnectionError(f"Error connecting to device: {str(e)}")
    async def disconnect(self) -> bool:
        """Disconnect from the BLE device.

        Returns:
            Whether disconnection was successful

        Raises:
            DisconnectionError: When disconnection fails
        """
        if not self.client or not self.is_connected:
            logger.warning("Attempting to disconnect a device that is not connected")
            return False

        try:
            # Stop all notifications first
            for char_uuid in list(self.active_notifications.keys()):
                await self.stop_notify(char_uuid)

            await self.client.disconnect()
            self.is_connected = False
            logger.info(f"Disconnected from device {self.address}")
            return True

        except BleakError as e:
            logger.error(f"Error disconnecting from device {self.address}: {str(e)}")
            raise DisconnectionError(f"Error disconnecting: {str(e)}")

    async def _start_notify_internal(
        self, 
        char_uuid: str, 
        internal_callback: Callable[[Any, bytearray], None]
    ) -> bool:
        """Internal method to enable BLE notifications using a raw Bleak callback.

        Args:
            char_uuid: Characteristic UUID
            internal_callback: Callback function for Bleak (sender, data)

        Returns:
            Whether enabling notifications was successful
        """
        if not self.client or not self.is_connected:
            logger.error("Device not connected, cannot enable notifications")
            raise NotificationError("Device not connected, cannot enable notifications")

        try:
            logger.info(f"Enabling notifications for characteristic {char_uuid}...")
            await self.client.start_notify(char_uuid, internal_callback)
            self.active_notifications[char_uuid] = internal_callback
            logger.info(f"Notifications enabled for characteristic {char_uuid}")
            return True

        except BleakError as e:
            logger.error(f"Error enabling notifications for {char_uuid}: {str(e)}")
            raise NotificationError(f"Error enabling notifications for {char_uuid}: {str(e)}")

    async def stop_notify(self, char_uuid: str = '') -> bool:
        """Stop BLE notifications for a specific characteristic UUID.

        Args:
            char_uuid: Characteristic UUID

        Returns:
            Whether stopping notifications was successful

        Raises:
            NotificationError: When stopping notifications fails
        """
        if not self.client or not self.is_connected:
            logger.warning("Device not connected, cannot stop notifications")
            return False

        if char_uuid not in self.active_notifications:
            logger.warning(f"Notifications for characteristic {char_uuid} not enabled or already stopped.")
            return False

        try:
            logger.info(f"Stopping notifications for characteristic {char_uuid}...")
            await self.client.stop_notify(char_uuid)
            del self.active_notifications[char_uuid]
            logger.info(f"Notifications stopped for characteristic {char_uuid}")
            return True

        except BleakError as e:
            logger.error(f"Error stopping notifications for {char_uuid}: {str(e)}")
            raise NotificationError(f"Error stopping notifications for {char_uuid}: {str(e)}")

    async def start_battery_stream(self, user_callback: Callable[[Dict[str, Any]], None]) -> bool:
        """Start streaming battery level and RSSI data.

        Args:
            user_callback: Callback function to receive parsed battery and RSSI data.
                         Example: def my_callback(data: dict):
                                      print(f"Battery: {data['battery_level']}%, RSSI: {data['rssi']}")

        Returns:
            True if streaming was started successfully, False otherwise.
        """
        if not self.device_config or not self.device_config.characteristics.battery:
            logger.error("Battery characteristic UUID not configured for this device type.")
            return False

        battery_char_uuid = self.device_config.characteristics.battery
        internal_bleak_callback = create_battery_rssi_data_processor(user_callback, self.device_type)
        return await self._start_notify_internal(battery_char_uuid, internal_bleak_callback)

    async def stop_battery_stream(self) -> bool:
        """Stop streaming battery level and RSSI data."""
        if not self.device_config or not self.device_config.characteristics.battery:
            logger.error("Battery characteristic UUID not configured for this device type.")
            return False
        return await self.stop_notify(self.device_config.characteristics.battery)

    async def start_eeg_stream(self, user_callback: Callable[[Union[Dict[str, Any], bytearray]], None], raw_data_only: bool = False) -> bool:
        """Start streaming EEG data.

        Args:
            user_callback: Callback function to receive parsed EEG data or raw bytes.
            raw_data_only: If True, callback receives raw bytearray (header/tail removed).
                           Otherwise, callback receives parsed dictionary.

        Returns:
            True if streaming was started successfully, False otherwise.
        """
        if not self.device_config or \
           not self.device_config.characteristics.eeg or \
           not self.device_config.characteristics.write or \
           not self.device_config.commands.start_eeg:
            logger.error("EEG characteristics or commands not fully configured for this device type.")
            return False

        eeg_char_uuid = self.device_config.characteristics.eeg
        write_char_uuid = self.device_config.characteristics.write
        start_cmd = self.device_config.commands.start_eeg

        internal_bleak_callback = create_eeg_data_processor(user_callback, self.device_type, raw_data_only=raw_data_only)

        # Start notifications first
        if not await self._start_notify_internal(eeg_char_uuid, internal_bleak_callback):
            return False # Notification setup failed

        # Then send the start command
        try:
            await self.write_gatt_char(write_char_uuid, start_cmd)
            logger.info("EEG stream started successfully.")
            return True
        except WriteError as e:
            logger.error(f"Failed to send EEG start command: {e}. Stopping notifications.")
            await self.stop_notify(eeg_char_uuid) # Clean up notifications
            return False

    async def stop_eeg_stream(self) -> bool:
        """Stop streaming EEG data."""
        if not self.device_config or \
           not self.device_config.characteristics.eeg or \
           not self.device_config.characteristics.write or \
           not self.device_config.commands.stop_xxx: # Assuming stop_xxx is for EEG
            logger.error("EEG characteristics or stop command not fully configured.")
            return False

        eeg_char_uuid = self.device_config.characteristics.eeg
        write_char_uuid = self.device_config.characteristics.write
        stop_cmd = self.device_config.commands.stop_xxx

        try:
            await self.write_gatt_char(write_char_uuid, stop_cmd)
            logger.info("EEG stop command sent.")
        except WriteError as e:
            logger.error(f"Failed to send EEG stop command: {e}. Proceeding to stop notifications.")
            # Fall through to stop notifications even if command fails

        return await self.stop_notify(eeg_char_uuid)

    async def write_gatt_char(
        self,
        char_uuid: str = '',
        data: bytes = None,
        response: bool = True
    ) -> bool:
        """Write to a GATT characteristic.

        Args:
            char_uuid: Characteristic UUID to write to
            data: The data to send
            response: If True, require a response, otherwise don't wait for a response

        Returns:
            Whether the write was successful

        Raises:
            WriteError: When writing to the characteristic fails
        """
        if not self.client or not self.is_connected:
            logger.error("Device not connected, cannot write to characteristic")
            raise WriteError("Device not connected, cannot write to characteristic")

        if data is None:
            logger.error("No data provided for write operation")
            raise WriteError("No data provided for write operation")

        try:
            logger.info(f"Writing to characteristic {char_uuid}... (data: {data.hex()})")
            await self.client.write_gatt_char(char_uuid, data, response)
            logger.info(f"Successfully wrote to characteristic {char_uuid}")
            return True

        except BleakError as e:
            logger.error(f"Error writing to characteristic {char_uuid}: {str(e)}")
            raise WriteError(f"Error writing to characteristic: {str(e)}")

    async def start_impedance_stream(self, user_callback: Callable[[Union[Dict[str, Any], bytearray]], None], raw_data_only: bool = False) -> bool:
        """Start streaming electrode impedance data.

        Args:
            user_callback: Callback function to receive parsed impedance data or raw bytes.
            raw_data_only: If True, callback receives raw bytearray (header/tail removed).
                           Otherwise, callback receives parsed dictionary.

        Returns:
            True if streaming was started successfully, False otherwise.
        """
        if not self.device_config or \
           not self.device_config.characteristics.impedance or \
           not self.device_config.characteristics.write or \
           not self.device_config.commands.start_impedance:
            logger.error("Impedance characteristics or commands not fully configured for this device type.")
            return False

        imp_char_uuid = self.device_config.characteristics.impedance
        write_char_uuid = self.device_config.characteristics.write
        start_cmd = self.device_config.commands.start_impedance

        internal_bleak_callback = create_impedance_data_processor(user_callback, self.device_type, raw_data_only=raw_data_only)

        if not await self._start_notify_internal(imp_char_uuid, internal_bleak_callback):
            return False 

        try:
            await self.write_gatt_char(write_char_uuid, start_cmd)
            logger.info("Impedance stream started successfully.")
            return True
        except WriteError as e:
            logger.error(f"Failed to send Impedance start command: {e}. Stopping notifications.")
            await self.stop_notify(imp_char_uuid) 
            return False

    async def stop_impedance_stream(self) -> bool:
        """Stop streaming electrode impedance data."""
        if not self.device_config or \
           not self.device_config.characteristics.impedance or \
           not self.device_config.characteristics.write or \
           not self.device_config.commands.stop_impedance: 
            logger.error("Impedance characteristics or stop command not fully configured.")
            return False

        imp_char_uuid = self.device_config.characteristics.impedance
        write_char_uuid = self.device_config.characteristics.write
        stop_cmd = self.device_config.commands.stop_impedance

        try:
            await self.write_gatt_char(write_char_uuid, stop_cmd)
            logger.info("Impedance stop command sent.")
        except WriteError as e:
            logger.error(f"Failed to send Impedance stop command: {e}. Proceeding to stop notifications.")

        return await self.stop_notify(imp_char_uuid)

    async def start_imu_stream(self, user_callback: Callable[[Union[Dict[str, Any], bytearray]], None], raw_data_only: bool = False) -> bool:
        """Start streaming IMU (accelerometer and gyroscope) data.

        Args:
            user_callback: Callback function to receive parsed IMU data or raw bytes.
            raw_data_only: If True, callback receives raw bytearray (header/tail removed).
                           Otherwise, callback receives parsed dictionary.

        Returns:
            True if streaming was started successfully, False otherwise.
        """
        if not self.device_config or \
           not self.device_config.characteristics.imu or \
           not self.device_config.characteristics.write or \
           not self.device_config.commands.start_imu:
            logger.error("IMU characteristics or commands not fully configured for this device type.")
            return False

        imu_char_uuid = self.device_config.characteristics.imu
        write_char_uuid = self.device_config.characteristics.write
        start_cmd = self.device_config.commands.start_imu

        internal_bleak_callback = create_imu_data_processor(user_callback, self.device_type, raw_data_only=raw_data_only)

        if not await self._start_notify_internal(imu_char_uuid, internal_bleak_callback):
            return False

        try:
            await self.write_gatt_char(write_char_uuid, start_cmd)
            logger.info("IMU stream started successfully.")
            return True
        except WriteError as e:
            logger.error(f"Failed to send IMU start command: {e}. Stopping notifications.")
            await self.stop_notify(imu_char_uuid)
            return False

    async def stop_imu_stream(self) -> bool:
        """Stop streaming IMU data."""
        if not self.device_config or \
           not self.device_config.characteristics.imu or \
           not self.device_config.characteristics.write or \
           not self.device_config.commands.stop_xxx: # Assuming stop_xxx is also for IMU
            logger.error("IMU characteristics or stop command not fully configured.")
            return False

        imu_char_uuid = self.device_config.characteristics.imu
        write_char_uuid = self.device_config.characteristics.write
        stop_cmd = self.device_config.commands.stop_xxx # Re-evaluate if IMU needs a specific stop cmd

        try:
            await self.write_gatt_char(write_char_uuid, stop_cmd)
            logger.info("IMU stop command sent.")
        except WriteError as e:
            logger.error(f"Failed to send IMU stop command: {e}. Proceeding to stop notifications.")

        return await self.stop_notify(imu_char_uuid)

    async def start_lead_stream(self, user_callback: Callable[[Union[Dict[str, Any], bytearray]], None], raw_data_only: bool = False) -> bool:
        """Start streaming lead status data.

        Args:
            user_callback: Callback function to receive parsed lead status data or raw bytes.
            raw_data_only: If True, callback receives raw bytearray (header/tail removed).
                           Otherwise, callback receives parsed dictionary.

        Returns:
            True if streaming was started successfully, False otherwise.
        """
        if not self.device_config or \
           not self.device_config.characteristics.lead or \
           not self.device_config.characteristics.write or \
           not self.device_config.commands.start_lead:
            logger.error("Lead status characteristics or commands not fully configured for this device type.")
            return False

        lead_char_uuid = self.device_config.characteristics.lead
        write_char_uuid = self.device_config.characteristics.write
        start_cmd = self.device_config.commands.start_lead

        internal_bleak_callback = create_lead_data_processor(user_callback, self.device_type, raw_data_only=raw_data_only)

        if not await self._start_notify_internal(lead_char_uuid, internal_bleak_callback):
            return False

        try:
            await self.write_gatt_char(write_char_uuid, start_cmd)
            logger.info("Lead status stream started successfully.")
            return True
        except WriteError as e:
            logger.error(f"Failed to send lead status start command: {e}. Stopping notifications.")
            await self.stop_notify(lead_char_uuid)
            return False

    async def stop_lead_stream(self) -> bool:
        """Stop streaming lead status data."""
        if not self.device_config or not self.device_config.characteristics.lead:
            logger.error("Lead status characteristic not configured for this device type.")
            return False

        lead_char_uuid = self.device_config.characteristics.lead

        return await self.stop_notify(lead_char_uuid)

    # --- Mixed Data Stream Methods ---
    def _create_internal_mixed_processor(self, data_source_type: str, raw_data_only_for_imu: bool = False, raw_data_only_for_lead: bool = False, raw_data_only_for_main: bool = False) -> Callable[[Any, bytearray], None]:
        """Creates an internal callback for mixed data stream processing."""
        def processor(sender: Any, raw_data: bytearray) -> None:
            if not self._mixed_user_callback: # User callback not set
                return

            parsed_data: Optional[Dict[str, Any]] = None
            current_timestamp = datetime.datetime.now()

            if data_source_type == "imu":
                if raw_data_only_for_imu:
                    # Validate and trim raw IMU data
                    if not self.device_config:
                        logger.error("Device config not loaded for raw IMU data processing.")
                        return
                    trimmed_data = raw_data[5:-1]
                    self._mixed_latest_imu_data = trimmed_data
                else:
                    parsed_data = parse_imu_data(raw_data, self.device_type)
                    if parsed_data and parsed_data.get("valid"):
                        self._mixed_latest_imu_data = parsed_data
                # IMU data itself doesn't trigger the main callback
                return 

            elif data_source_type == "lead":
                if raw_data_only_for_lead:
                    trimmed_data = raw_data[5:-1]
                    self._mixed_latest_lead_data = trimmed_data
                else:
                    parsed_data = parse_lead_status(raw_data, self.device_type)
                    if parsed_data and parsed_data.get("valid"):
                        self._mixed_latest_lead_data = parsed_data
                # Lead data itself doesn't trigger the main callback
                return

            elif data_source_type == "eeg_impedance":
                main_data_type = self._mixed_current_mode
                main_data_parsed: Optional[Union[List[List[float]], Dict[str, Any], bytearray]] = None

                if raw_data_only_for_main:
                    if main_data_type == "eeg":
                        # EEG: header=5 bytes, footer=1 byte
                        main_data_parsed = raw_data[5:-1]  # Remove header and footer
                    elif main_data_type ==  "impedance":
                        # Impedance: header=3 bytes, footer=1 byte
                        main_data_parsed = raw_data[3:-1]  # Remove header and footer
                    else:
                        logger.error("No suppord main_data_type")
                else:
                    # Parse data normally
                    if main_data_type == "eeg":
                        main_data_parsed = parse_eeg_data(raw_data, self.device_type)
                        if not (main_data_parsed and main_data_parsed.get("valid")):
                            main_data_parsed = None # Invalid parse
                    elif main_data_type == "impedance":
                        main_data_parsed = parse_impedance_data(raw_data, self.device_type)
                        if not (main_data_parsed and main_data_parsed.get("valid")):
                            main_data_parsed = None # Invalid parse

                # Set mode_key for both raw and parsed modes
                mode_key = f"{main_data_type}_data"

                if main_data_parsed is not None:
                    output_payload = {
                        "timestamp": current_timestamp,
                        "type": main_data_type,
                        mode_key: main_data_parsed,
                        "imu_data": self._mixed_latest_imu_data,
                        "lead_data": self._mixed_latest_lead_data
                    }
                    self._mixed_user_callback(output_payload)
                else:
                    logger.debug(f"Mixed stream: Invalid {main_data_type} data, not calling user_callback.")
            else:
                logger.warning(f"Unknown data_source_type in mixed processor: {data_source_type}")

        return processor

    async def start_mixed_stream(self, mode: str, user_callback: Callable[[Dict[str, Union[Any, bytearray]]], None], raw_main_data_only: bool = False) -> bool:
        """Start streaming mixed data (EEG/Impedance, IMU, Lead).

        Args:
            mode: 'eeg' or 'impedance' for the main data stream.
            user_callback: Callback function to receive bundled mixed data.
                           All data types can be parsed dictionary or raw bytes.
            raw_main_data_only: If True, main data (EEG/Impedance、imu、lead) in callback will be raw bytearray (header/tail removed).
                                Otherwise, main data will be parsed dictionary.
        
        Returns:
            True if all streams were started successfully, False otherwise.
        """
        if mode.lower() not in ["eeg", "impedance"]:
            logger.error(f"Invalid mode for mixed stream: {mode}. Must be 'eeg' or 'impedance'.")
            return False

        # Check if impedance mode is supported for this device
        if mode.lower() == "impedance" and not self.device_config.commands.start_impedance:
            logger.error(f"Impedance mode is not supported for device type {self.device_config}.")
            return False

        if not self.device_config or \
           not self.device_config.characteristics.eeg or \
           not self.device_config.characteristics.imu or \
           not self.device_config.characteristics.lead or \
           not self.device_config.characteristics.write or \
           not self.device_config.commands.start_eeg or \
           not self.device_config.commands.start_imu or \
           not self.device_config.commands.start_lead:
            logger.error("Mixed stream characteristics or commands not fully configured.")
            return False

        self._mixed_user_callback = user_callback
        self._mixed_current_mode = mode.lower()
        # Reset cached latest data based on the new raw data flags
        self._mixed_latest_imu_data = None  # Will store bytearray or Dict based on raw_imu_data_only
        self._mixed_latest_lead_data = None  # Will store bytearray or Dict based on raw_lead_data_only

        # Characteristic UUIDs
        eeg_char = self.device_config.characteristics.eeg
        impedance_char = self.device_config.characteristics.impedance
        imu_char = self.device_config.characteristics.imu
        lead_char = self.device_config.characteristics.lead
        write_char = self.device_config.characteristics.write

        # Create internal processors
        eeg_imp_processor = self._create_internal_mixed_processor("eeg_impedance", raw_data_only_for_main=raw_main_data_only)
        imu_processor = self._create_internal_mixed_processor("imu", raw_data_only_for_imu=raw_main_data_only)
        lead_processor = self._create_internal_mixed_processor("lead", raw_data_only_for_lead=raw_main_data_only)

        # Determine which characteristic to use for EEG/Impedance based on mode
        if self._mixed_current_mode == "eeg":
            main_char = eeg_char
        else:  # impedance mode
            main_char = impedance_char

        # Start notifications
        success = True
        try:
            logger.info("Starting notifications for mixed data stream...")
            if not await self._start_notify_internal(main_char, eeg_imp_processor):
                success = False
            if success and not await self._start_notify_internal(imu_char, imu_processor):
                # Try to stop previous if this one fails
                await self.stop_notify(main_char) 
                success = False
            if success and not await self._start_notify_internal(lead_char, lead_processor):
                await self.stop_notify(main_char)
                await self.stop_notify(imu_char)
                success = False

            if not success:
                logger.error("Failed to start all notifications for mixed stream.")
                return False
            logger.info("All notifications for mixed stream enabled.")

            # Send start commands
            logger.info("Sending start commands for mixed data stream...")
            if self._mixed_current_mode == "eeg":
                await self.write_gatt_char(write_char, self.device_config.commands.start_eeg)
            else: # impedance mode
                await self.write_gatt_char(write_char, self.device_config.commands.start_impedance)

            await self.write_gatt_char(write_char, self.device_config.commands.start_imu)
            await self.write_gatt_char(write_char, self.device_config.commands.start_lead)

            logger.info("Mixed data stream started successfully.")
            return True

        except (WriteError, NotificationError) as e:
            logger.error(f"Error starting mixed stream: {e}. Cleaning up...")
            # Attempt to stop any notifications that might have started
            if main_char in self.active_notifications: await self.stop_notify(main_char)
            if imu_char in self.active_notifications: await self.stop_notify(imu_char)
            if lead_char in self.active_notifications: await self.stop_notify(lead_char)
            return False

    async def stop_mixed_stream(self) -> bool:
        """Stop the mixed data stream."""
        logger.info("Stopping mixed data stream...")
        if not self.device_config:
            logger.error("Device config not loaded, cannot stop mixed stream.")
            return False

        write_char = self.device_config.characteristics.write
        imu_char = self.device_config.characteristics.imu
        lead_char = self.device_config.characteristics.lead

        # Determine which characteristic was used for EEG/Impedance based on mode
        if self._mixed_current_mode == "eeg":
            main_char = self.device_config.characteristics.eeg
        else:  # impedance mode
            main_char = self.device_config.characteristics.impedance

        # Send stop commands (assuming stop_xxx for EEG/IMU and specific for impedance/lead if available)
        # Error handling for write_gatt_char is important here, but for simplicity, direct calls shown.
        # In a robust implementation, check for command existence in device_config.
        try:
            if self._mixed_current_mode == "eeg" and self.device_config.commands.stop_xxx:
                await self.write_gatt_char(write_char, self.device_config.commands.stop_xxx)
            elif self._mixed_current_mode == "impedance" and self.device_config.commands.stop_impedance:
                await self.write_gatt_char(write_char, self.device_config.commands.stop_impedance)

            # Assuming IMU also uses stop_xxx, and Lead might not have a specific stop command other than notification stop
            if self.device_config.commands.stop_xxx: # For IMU
                await self.write_gatt_char(write_char, self.device_config.commands.stop_xxx)
            # If lead has a stop_lead command, add it here:
            # if self.device_config.commands.stop_lead:
            #     await self.write_gatt_char(write_char, self.device_config.commands.stop_lead)

        except WriteError as e:
            logger.warning(f"Failed to send one or more stop commands for mixed stream: {e}")
            # Continue to stop notifications regardless

        # Stop notifications
        stopped_all_notifications = True
        if main_char in self.active_notifications and not await self.stop_notify(main_char):
            stopped_all_notifications = False
        if imu_char in self.active_notifications and not await self.stop_notify(imu_char):
            stopped_all_notifications = False
        if lead_char in self.active_notifications and not await self.stop_notify(lead_char):
            stopped_all_notifications = False

        if stopped_all_notifications:
            logger.info("Mixed data stream stopped successfully.")
        else:
            logger.warning("Mixed data stream stopped, but failed to stop one or more notifications.")

        self._mixed_user_callback = None
        self._mixed_current_mode = None
        self._mixed_latest_imu_data = None
        self._mixed_latest_lead_data = None
        return stopped_all_notifications

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect() 
