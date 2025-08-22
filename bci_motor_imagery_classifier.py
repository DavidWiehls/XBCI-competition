import asyncio
import numpy as np
import sys
from typing import List, Dict, Any
from loguru import logger
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Configure application logging (must be before importing SDK)
logger.remove()
logger.add(
    sys.stderr, 
    level="INFO", 
    filter=lambda record: not record["extra"].get("sdk_internal", False)
)

# Import SDK components
from src.ble_sdk.client import BleClient
from src.ble_sdk.constants import DeviceType
from src.ble_sdk.scanner import scan_ble_devices
from src.ble_sdk.log import sdk_logger_manager

# Set SDK log level
sdk_logger_manager.set_level("INFO")

class BCIMotorImageryClassifier:
    def __init__(self, device_address: str = "D6:5A:01:75:CD:7E", sample_rate: int = 250):
        """
        Initialize BCI Motor Imagery Classifier
        
        Args:
            device_address: BLE device address (if None, will scan for devices)
            sample_rate: EEG sampling rate in Hz
        """
        self.device_address = device_address
        self.sample_rate = sample_rate
        self.client = None
        self.raw_data_buffer = []
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_collecting = False
        
        # Filter parameters
        self.low_freq = 2.0
        self.high_freq = 40.0
        self.notch_freq = 50.0
        
        # Classification labels
        self.classes = ['right_hand', 'left_hand', 'right_feet', 'left_feet']
        
    async def find_device(self) -> str:
        """Scan for BLE devices and return the first compatible device address"""
        logger.info("Scanning for BLE devices...")
        devices = await scan_ble_devices()
        
        if not devices:
            raise Exception("No BLE devices found. Please ensure your device is turned on and in pairing mode.")
        
        logger.info(f"Found {len(devices)} device(s):")
        for device in devices:
            logger.info(f"  - {device['name']} ({device['address']})")
        
        # Use the first device found
        device_address = devices[0]['address']
        logger.info(f"Using device: {device_address}")
        return device_address
    
    def eeg_data_handler(self, data):
        """Callback function for EEG data collection"""
        if self.is_collecting:
            if isinstance(data, bytearray):
                # Raw data mode - store the raw bytes
                self.raw_data_buffer.append(data)
            else:
                # Parsed data mode - extract channel data
                if 'channels' in data:
                    channel_data = data['channels']
                    # Convert to numpy array and store
                    self.raw_data_buffer.append(np.array(channel_data))
    
    def preprocess_eeg_data(self, raw_data: List) -> np.ndarray:
        """
        Preprocess EEG data with bandpass and notch filters
        
        Args:
            raw_data: List of raw EEG data chunks
            
        Returns:
            Preprocessed EEG data array (channels x samples)
        """
        logger.info("Preprocessing EEG data...")
        
        # Combine all raw data chunks
        if isinstance(raw_data[0], bytearray):
            # Handle raw byte data - convert to numpy array
            # Assuming 8 channels, 2 bytes per sample, little endian
            combined_data = b''.join(raw_data)
            num_samples = len(combined_data) // (8 * 2)  # 8 channels * 2 bytes per sample
            eeg_data = np.frombuffer(combined_data, dtype=np.int16)
            eeg_data = eeg_data.reshape(8, -1)  # Reshape to (channels, samples)
        else:
            # Handle parsed data
            eeg_data = np.column_stack(raw_data)
        
        logger.info(f"Raw EEG data shape: {eeg_data.shape}")
        
        # Apply bandpass filter (2-40 Hz)
        nyquist = self.sample_rate / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply notch filter (50 Hz)
        notch_b, notch_a = signal.iirnotch(self.notch_freq, 30, self.sample_rate)
        
        # Apply filters to each channel
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            # Bandpass filter
            bandpass_filtered = signal.filtfilt(b, a, eeg_data[ch, :])
            # Notch filter
            filtered_data[ch, :] = signal.filtfilt(notch_b, notch_a, bandpass_filtered)
        
        logger.info(f"Filtered EEG data shape: {filtered_data.shape}")
        return filtered_data
    
    def extract_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract features from preprocessed EEG data
        
        Args:
            eeg_data: Preprocessed EEG data (channels x samples)
            
        Returns:
            Feature vector
        """
        features = []
        
        for ch in range(eeg_data.shape[0]):
            channel_data = eeg_data[ch, :]
            
            # Time domain features
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.var(channel_data),
                np.max(channel_data),
                np.min(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
            ])
            
            # Frequency domain features
            fft_vals = np.abs(np.fft.fft(channel_data))
            freqs = np.fft.fftfreq(len(channel_data), 1/self.sample_rate)
            
            # Power in different frequency bands
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            mu_mask = (freqs >= 8) & (freqs <= 12)
            
            features.extend([
                np.sum(fft_vals[alpha_mask]),
                np.sum(fft_vals[beta_mask]),
                np.sum(fft_vals[mu_mask]),
            ])
        
        return np.array(features)
    
    async def collect_calibration_data(self, class_label: str, duration: int = 10) -> np.ndarray:
        """
        Collect calibration data for a specific motor imagery class
        
        Args:
            class_label: Class label ('right_hand', 'left_hand', 'right_feet', 'left_feet')
            duration: Collection duration in seconds
            
        Returns:
            Preprocessed EEG data
        """
        logger.info(f"Collecting calibration data for: {class_label}")
        logger.info(f"Please think about moving your {class_label} for {duration} seconds...")
        
        # Clear buffer and start collection
        self.raw_data_buffer = []
        self.is_collecting = True
        
        # Start EEG stream
        await self.client.start_eeg_stream(self.eeg_data_handler, raw_data_only=True)
        
        # Wait for specified duration
        await asyncio.sleep(duration)
        
        # Stop collection
        await self.client.stop_eeg_stream()
        self.is_collecting = False
        
        # Preprocess the collected data
        preprocessed_data = self.preprocess_eeg_data(self.raw_data_buffer)
        
        logger.info(f"Collected {len(self.raw_data_buffer)} data chunks for {class_label}")
        return preprocessed_data
    
    async def train_classifier(self):
        """Collect calibration data and train the classifier"""
        logger.info("Starting calibration phase...")
        
        X = []  # Features
        y = []  # Labels
        
        # Collect data for each class
        for i, class_label in enumerate(self.classes):
            logger.info(f"\n=== Calibration Session {i+1}/4 ===")
            logger.info(f"Class: {class_label}")
            
            # Give user time to prepare
            logger.info("Get ready... Starting in 3 seconds...")
            await asyncio.sleep(3)
            
            # Collect data
            eeg_data = await self.collect_calibration_data(class_label, duration=10)
            
            # Extract features
            features = self.extract_features(eeg_data)
            X.append(features)
            y.append(i)  # Use index as label
            
            logger.info(f"Completed calibration for {class_label}")
            await asyncio.sleep(2)  # Brief pause between sessions
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Calibration data shape: X={X.shape}, y={y.shape}")
        
        # Train classifier
        logger.info("Training classifier...")
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X, y)
        
        # Test accuracy on training data
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        logger.info(f"Training accuracy: {accuracy:.2f}")
        
        logger.info("Classifier training completed!")
    
    async def classify_realtime(self, duration: int = 10) -> str:
        """
        Collect real-time data and classify it
        
        Args:
            duration: Collection duration in seconds
            
        Returns:
            Predicted class label
        """
        if self.classifier is None:
            raise Exception("Classifier not trained. Please run calibration first.")
        
        logger.info("Starting real-time classification...")
        logger.info("Please perform the motor imagery task...")
        
        # Clear buffer and start collection
        self.raw_data_buffer = []
        self.is_collecting = True
        
        # Start EEG stream
        await self.client.start_eeg_stream(self.eeg_data_handler, raw_data_only=True)
        
        # Wait for specified duration
        await asyncio.sleep(duration)
        
        # Stop collection
        await self.client.stop_eeg_stream()
        self.is_collecting = False
        
        # Preprocess the collected data
        preprocessed_data = self.preprocess_eeg_data(self.raw_data_buffer)
        
        # Extract features
        features = self.extract_features(preprocessed_data)
        
        # Make prediction
        prediction = self.classifier.predict([features])[0]
        predicted_class = self.classes[prediction]
        
        logger.info(f"Classification result: {predicted_class}")
        return predicted_class
    
    async def run_complete_session(self):
        """Run the complete BCI motor imagery session"""
        try:
            # Find device if not provided
            if self.device_address is None:
                self.device_address = await self.find_device()
            
            # Initialize client
            #self.client = BleClient(address=self.device_address, device_type=DeviceType.BLE_8)
            device_types_to_test = [DeviceType.BLE_8, DeviceType.BLE_4, DeviceType.BLE_2, DeviceType.BLE_1]
            connected = False

            for device_type in device_types_to_test:
                try:
                    logger.info(f"Attempting to connect with DeviceType.{device_type.name}...")
                    self.client = BleClient(address=self.device_address, device_type=device_type)

                    async with self.client:
                        logger.info(f"Successfully connected with DeviceType.{device_type.name}!")
                        self.device_type = device_type
                        connected = True
                        break  # Exit the loop once a successful connection is made

                except ConnectionError:
                    logger.warning(f"Connection failed with DeviceType.{device_type.name}. Trying next type.")
                    continue

                if not connected:
                        raise Exception("Failed to connect with all tested device types.")
            async with self.client:
                logger.info("Connected to BLE device")
                
                # Phase 1: Calibration
                logger.info("\n" + "="*50)
                logger.info("PHASE 1: CALIBRATION")
                logger.info("="*50)
                await self.train_classifier()
                
                # Phase 2: Real-time Classification
                logger.info("\n" + "="*50)
                logger.info("PHASE 2: REAL-TIME CLASSIFICATION")
                logger.info("="*50)
                
                while True:
                    try:
                        # Collect and classify new data
                        result = await self.classify_realtime(duration=10)
                        
                        # Print the result
                        print("\n" + "="*30)
                        print(f"PREDICTION: {result.upper()}")
                        print("="*30)
                        
                        # Ask if user wants to continue
                        logger.info("Press Enter to continue classification or 'q' to quit...")
                        # Note: In a real application, you might want to use a different input method
                        await asyncio.sleep(1)  # Give user time to see the result
                        
                    except KeyboardInterrupt:
                        logger.info("Classification stopped by user")
                        break
                        
        except Exception as e:
            logger.error(f"Error during BCI session: {e}")
            raise

async def main():
    """Main function to run the BCI motor imagery classifier"""
    logger.info("BCI Motor Imagery Classifier")
    logger.info("This script will help you calibrate and classify motor imagery tasks")
    
    # You can specify a device address here if you know it
    # device_address = "D2:43:D5:88:4D:9A"  # Replace with your device address
    device_address = None  # Will scan for devices
    
    # Create and run the classifier
    classifier = BCIMotorImageryClassifier(device_address=device_address)
    await classifier.run_complete_session()

if __name__ == "__main__":
    asyncio.run(main())
