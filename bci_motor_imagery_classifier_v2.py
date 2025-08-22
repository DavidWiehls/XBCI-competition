import asyncio
import numpy as np
import sys
from typing import List, Dict, Any
from loguru import logger
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# Configure application logging (must be before importing SDK)
logger.remove()
logger.add(
    sys.stderr, 
    level="INFO", 
    filter=lambda record: not record["extra"].get("sdk_internal", False)
)

# Import SDK components
from ble_sdk.client import BleClient
from ble_sdk.constants import DeviceType
from ble_sdk.scanner import scan_ble_devices
from ble_sdk.log import sdk_logger_manager
from ble_sdk.algo.eeg import one_ch_one_sampling_process

# Set SDK log level
sdk_logger_manager.set_level("INFO")

class BCIMotorImageryClassifier:
    def __init__(self, device_address: str = None, sample_rate: int = 500):
        """
        Initialize BCI Motor Imagery Classifier
        
        Args:
            device_address: BLE device address (if None, will scan for devices)
            sample_rate: EEG sampling rate in Hz (default 500Hz for this device)
        """
        self.device_address = device_address
        self.sample_rate = sample_rate
        self.client = None
        self.raw_data_buffer = []
        self.classifier = None
        self.is_collecting = False
        
        # Filter parameters
        self.low_freq = 2.0
        self.high_freq = 40.0
        self.notch_freq = 50.0
        
        # Classification labels
        self.classes = ['right_hand', 'left_hand', 'right_feet', 'left_feet']
        
        # EEG data parameters
        self.num_channels = 8
        self.bytes_per_sample = 3
        self.samples_per_packet = 64  # 192 bytes / (8 channels * 3 bytes per sample)
        
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
    
    def parse_raw_eeg_data(self, raw_data: List[bytearray]) -> np.ndarray:
        """
        Parse raw EEG data from bytearrays to numpy array
        
        Args:
            raw_data: List of raw EEG data chunks (bytearrays)
            
        Returns:
            EEG data array (channels x samples)
        """
        logger.info("Parsing raw EEG data...")
        
        # Constants for data parsing
        HEADER_SIZE = 5  # Frame header size
        FOOTER_SIZE = 1  # Frame footer size
        BYTES_PER_SAMPLE = 3
        
        all_channels_data = [[] for _ in range(self.num_channels)]
        
        for packet in raw_data:
            if len(packet) < HEADER_SIZE + FOOTER_SIZE:
                continue
                
            # Extract EEG data (remove header and footer)
            eeg_bytes = packet[HEADER_SIZE:-FOOTER_SIZE]
            
            # Calculate samples per channel in this packet
            samples_per_channel = len(eeg_bytes) // (self.num_channels * BYTES_PER_SAMPLE)
            
            # Parse each sample
            for sample_idx in range(samples_per_channel):
                for ch_idx in range(self.num_channels):
                    # Calculate position in the packet
                    start_pos = (sample_idx * self.num_channels + ch_idx) * BYTES_PER_SAMPLE
                    end_pos = start_pos + BYTES_PER_SAMPLE
                    
                    if end_pos <= len(eeg_bytes):
                        # Extract 3 bytes for this sample
                        sample_bytes = eeg_bytes[start_pos:end_pos]
                        # Convert to hex string for processing
                        hex_str = sample_bytes.hex()
                        # Process using the SDK's algorithm
                        voltage = one_ch_one_sampling_process(hex_str)
                        all_channels_data[ch_idx].append(voltage)
        
        # Convert to numpy array
        eeg_data = np.array(all_channels_data, dtype=np.float64)
        logger.info(f"Parsed EEG data shape: {eeg_data.shape}")
        return eeg_data
    
    def preprocess_eeg_data(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Preprocess EEG data with bandpass and notch filters
        
        Args:
            eeg_data: Raw EEG data array (channels x samples)
            
        Returns:
            Preprocessed EEG data array (channels x samples)
        """
        logger.info("Preprocessing EEG data...")
        
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
                np.median(channel_data),
            ])
            
            # Frequency domain features
            fft_vals = np.abs(np.fft.fft(channel_data))
            freqs = np.fft.fftfreq(len(channel_data), 1/self.sample_rate)
            
            # Power in different frequency bands
            delta_mask = (freqs >= 0.5) & (freqs <= 4)
            theta_mask = (freqs >= 4) & (freqs <= 8)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            mu_mask = (freqs >= 8) & (freqs <= 12)
            
            features.extend([
                np.sum(fft_vals[delta_mask]),
                np.sum(fft_vals[theta_mask]),
                np.sum(fft_vals[alpha_mask]),
                np.sum(fft_vals[beta_mask]),
                np.sum(fft_vals[mu_mask]),
            ])
            
            # Spectral features
            features.extend([
                np.argmax(fft_vals),  # Dominant frequency
                np.max(fft_vals),     # Peak amplitude
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
        
        # Parse and preprocess the collected data
        raw_eeg_data = self.parse_raw_eeg_data(self.raw_data_buffer)
        preprocessed_data = self.preprocess_eeg_data(raw_eeg_data)
        
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
        
        # Parse and preprocess the collected data
        raw_eeg_data = self.parse_raw_eeg_data(self.raw_data_buffer)
        preprocessed_data = self.preprocess_eeg_data(raw_eeg_data)
        
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
            self.client = BleClient(address=self.device_address, device_type=DeviceType.BLE_8)
            
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
                        logger.info("Press Ctrl+C to stop classification...")
                        await asyncio.sleep(2)  # Give user time to see the result
                        
                    except KeyboardInterrupt:
                        logger.info("Classification stopped by user")
                        break
                        
        except Exception as e:
            logger.error(f"Error during BCI session: {e}")
            raise

async def main():
    """Main function to run the BCI motor imagery classifier"""
    logger.info("BCI Motor Imagery Classifier v2")
    logger.info("This script will help you calibrate and classify motor imagery tasks")
    
    # You can specify a device address here if you know it
    # device_address = "D2:43:D5:88:4D:9A"  # Replace with your device address
    device_address = None  # Will scan for devices
    
    # Create and run the classifier
    classifier = BCIMotorImageryClassifier(device_address=device_address)
    await classifier.run_complete_session()

if __name__ == "__main__":
    asyncio.run(main())
