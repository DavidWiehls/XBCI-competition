"""
Offline BCI Motor Imagery Classifier

This module provides functionality to classify offline EEG data for motor imagery tasks.
It outputs class labels 0, 1, 2, 3 corresponding to:
0: right_hand
1: left_hand  
2: right_feet
3: left_feet

Usage:
    from offline_bci_classifier import OfflineBCIClassifier
    
    # Initialize classifier
    classifier = OfflineBCIClassifier()
    
    # Train on your data
    classifier.train_classifier_from_files("path/to/your/data")
    
    # Classify a single file
    prediction = classifier.classify_offline_file("data_file.npy")
    print(f"Predicted class: {prediction}")
    
    # Classify all files in a directory
    results = classifier.classify_offline_directory("path/to/data/directory")
    for filename, prediction in results:
        print(f"{filename}: Class {prediction}")
"""

import numpy as np
import os
from typing import List, Tuple, Union
from loguru import logger
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class OfflineBCIClassifier:
    """
    Offline BCI Motor Imagery Classifier
    
    This class provides functionality to classify offline EEG data for motor imagery tasks.
    It outputs class labels 0, 1, 2, 3 corresponding to:
    0: right_hand
    1: left_hand  
    2: right_feet
    3: left_feet
    """
    
    def __init__(self, sample_rate: int = 500):
        """
        Initialize the offline BCI classifier
        
        Args:
            sample_rate: EEG sampling rate in Hz (default 500Hz)
        """
        self.sample_rate = sample_rate
        self.classifier = None
        self.classes = ['right_hand', 'left_hand', 'right_feet', 'left_feet']
        
        # Filter parameters
        self.low_freq = 2.0
        self.high_freq = 40.0
        self.notch_freq = 50.0
        
        # EEG data parameters
        self.num_channels = 8
        
    def preprocess_eeg_data(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Preprocess EEG data with bandpass and notch filters
        
        Args:
            eeg_data: Raw EEG data array (channels x samples)
            
        Returns:
            Preprocessed EEG data array (channels x samples)
        """
        # Apply bandpass filter (2-40 Hz)
        nyquist = self.sample_rate / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply notch filter (50 Hz)
        notch_b, notch_a = signal.iirnotch(self.notch_freq, 30, self.sample_rate)
        
        # Apply filters to each channel
        filtered_data = np.zeros_like(eeg_data, dtype=np.float64)
        for ch in range(eeg_data.shape[0]):
            # Convert to float64 for filtering
            channel_data = eeg_data[ch, :].astype(np.float64)
            # Bandpass filter
            bandpass_filtered = signal.filtfilt(b, a, channel_data)
            # Notch filter
            filtered_data[ch, :] = signal.filtfilt(notch_b, notch_a, bandpass_filtered)
        
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
            
            # Basic statistical features
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.var(channel_data),
                np.max(channel_data) - np.min(channel_data),  # Range
            ])
            
            # Frequency domain features - focus on motor imagery relevant bands
            fft_vals = np.abs(np.fft.fft(channel_data))
            freqs = np.fft.fftfreq(len(channel_data), 1/self.sample_rate)
            
            # Motor imagery relevant frequency bands
            mu_mask = (freqs >= 8) & (freqs <= 12)    # Mu rhythm (8-12 Hz)
            beta_mask = (freqs >= 13) & (freqs <= 30)  # Beta rhythm (13-30 Hz)
            
            # Power in motor imagery relevant bands
            mu_power = np.sum(fft_vals[mu_mask])
            beta_power = np.sum(fft_vals[beta_mask])
            
            features.extend([mu_power, beta_power])
            
            # Spectral centroid (center of mass of the spectrum)
            if np.sum(fft_vals) > 0:
                spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
            else:
                spectral_centroid = 0
            features.append(spectral_centroid)
        
        return np.array(features)
    
    def train_classifier_from_files(self, data_dir: str):
        """
        Train the classifier using the available offline data files
        Each file is treated as a separate class sample
        
        Args:
            data_dir: Directory containing .npy files
            
        Returns:
            Training accuracy
        """
        logger.info("Training offline classifier...")
        
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        files.sort()
        
        if len(files) < 4:
            logger.warning(f"Only {len(files)} files found. Need at least 4 files for 4 classes.")
        
        X_features = []
        y_labels = []
        
        # Process each file as a separate sample
        for i, file in enumerate(files):
            file_path = os.path.join(data_dir, file)
            data = np.load(file_path)
            
            # Preprocess data
            preprocessed_data = self.preprocess_eeg_data(data)
            
            # Extract features
            features = self.extract_features(preprocessed_data)
            X_features.append(features)
            
            # Assign label based on file index (cycling through 0,1,2,3)
            label = i % 4
            y_labels.append(label)
            
            logger.info(f"Processed {file}: shape={data.shape}, assigned label={label} ({self.classes[label]})")
        
        X_features = np.array(X_features)
        y_labels = np.array(y_labels)
        
        logger.info(f"Feature matrix shape: {X_features.shape}")
        logger.info(f"Labels: {y_labels}")
        
        # Train classifier with simple parameters to avoid overfitting
        self.classifier = RandomForestClassifier(
            n_estimators=50,  # Fewer trees
            max_depth=3,      # Limit depth
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1
        )
        
        self.classifier.fit(X_features, y_labels)
        
        # Check training accuracy
        train_accuracy = self.classifier.score(X_features, y_labels)
        logger.info(f"Training accuracy: {train_accuracy:.3f}")
        
        # Print feature importance
        feature_importance = self.classifier.feature_importances_
        logger.info(f"Top 5 most important features: {np.argsort(feature_importance)[-5:]}")
        
        logger.info("Classifier training completed!")
        return train_accuracy
    
    def classify_offline_data(self, data: np.ndarray) -> int:
        """
        Classify offline EEG data
        
        Args:
            data: EEG data array (channels x samples)
            
        Returns:
            Predicted class label (0, 1, 2, or 3)
        """
        if self.classifier is None:
            raise Exception("Classifier not trained. Please train the classifier first.")
        
        # Preprocess data
        preprocessed_data = self.preprocess_eeg_data(data)
        
        # Extract features
        features = self.extract_features(preprocessed_data)
        
        # Make prediction
        prediction = self.classifier.predict([features])[0]
        
        return prediction
    
    def classify_offline_file(self, file_path: str) -> int:
        """
        Classify a single offline data file
        
        Args:
            file_path: Path to .npy file
            
        Returns:
            Predicted class label (0, 1, 2, or 3)
        """
        data = np.load(file_path)
        return self.classify_offline_data(data)
    
    def classify_offline_directory(self, data_dir: str) -> List[Tuple[str, int]]:
        """
        Classify all .npy files in a directory
        
        Args:
            data_dir: Directory containing .npy files
            
        Returns:
            List of (filename, predicted_class) tuples
        """
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        files.sort()
        
        results = []
        for file in files:
            file_path = os.path.join(data_dir, file)
            prediction = self.classify_offline_file(file_path)
            results.append((file, prediction))
            logger.info(f"{file}: predicted class {prediction} ({self.classes[prediction]})")
        
        return results
    
    def get_prediction_probabilities(self, data: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities for all classes
        
        Args:
            data: EEG data array (channels x samples)
            
        Returns:
            Array of probabilities for each class [0, 1, 2, 3]
        """
        if self.classifier is None:
            raise Exception("Classifier not trained. Please train the classifier first.")
        
        # Preprocess data
        preprocessed_data = self.preprocess_eeg_data(data)
        
        # Extract features
        features = self.extract_features(preprocessed_data)
        
        # Get prediction probabilities
        probabilities = self.classifier.predict_proba([features])[0]
        
        return probabilities
    
    def get_class_name(self, class_id: int) -> str:
        """
        Get the class name for a given class ID
        
        Args:
            class_id: Class ID (0, 1, 2, or 3)
            
        Returns:
            Class name string
        """
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        else:
            return f"Unknown class {class_id}"
    
    def save_classifier(self, file_path: str):
        """Save the trained classifier to disk"""
        if self.classifier is None:
            raise Exception("No classifier to save. Please train first.")
        
        joblib.dump(self.classifier, file_path)
        logger.info(f"Classifier saved to {file_path}")
    
    def load_classifier(self, file_path: str):
        """Load a trained classifier from disk"""
        self.classifier = joblib.load(file_path)
        logger.info(f"Classifier loaded from {file_path}")


def main():
    """Example usage of the offline classifier"""
    # Initialize classifier
    classifier = OfflineBCIClassifier(sample_rate=500)
    
    # Train classifier
    data_dir = "off_line_data"
    accuracy = classifier.train_classifier_from_files(data_dir)
    
    # Save trained classifier
    classifier.save_classifier("offline_bci_classifier.pkl")
    
    # Classify all files in the directory
    print("\n" + "="*50)
    print("OFFLINE CLASSIFICATION RESULTS")
    print("="*50)
    
    results = classifier.classify_offline_directory(data_dir)
    
    print("\nSummary:")
    for filename, predicted_class in results:
        class_name = classifier.get_class_name(predicted_class)
        print(f"{filename}: Class {predicted_class} ({class_name})")
    
    # Show prediction probabilities for the first file
    print("\n" + "="*50)
    print("PREDICTION PROBABILITIES (First File)")
    print("="*50)
    
    first_file = os.path.join(data_dir, "data_000001.npy")
    data = np.load(first_file)
    probabilities = classifier.get_prediction_probabilities(data)
    
    for i, (class_name, prob) in enumerate(zip(classifier.classes, probabilities)):
        print(f"Class {i} ({class_name}): {prob:.3f}")


if __name__ == "__main__":
    main()
