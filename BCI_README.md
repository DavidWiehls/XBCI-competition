# BCI Motor Imagery Classifier

This script implements a complete Brain-Computer Interface (BCI) system for motor imagery classification using the BLE SDK. It can classify four different motor imagery tasks: right hand, left hand, right feet, and left feet movements.

## Features

- **4-Class Motor Imagery Classification**: Classifies right hand, left hand, right feet, and left feet motor imagery
- **Real-time EEG Data Collection**: Collects EEG data from 8 channels using the BLE device
- **Advanced Signal Processing**: 
  - Bandpass filter (2-40 Hz) for relevant frequency bands
  - Notch filter (50 Hz) for power line interference removal
- **Feature Extraction**: Time and frequency domain features for each channel
- **Machine Learning**: Random Forest classifier for robust classification
- **Calibration System**: User-specific calibration for improved accuracy

## Prerequisites

1. **Hardware Requirements**:
   - BCI device compatible with the BLE SDK (8-channel EEG device)
   - Computer with Bluetooth capability

2. **Software Requirements**:
   - Python 3.8 or higher
   - Required Python packages (see requirements_bci.txt)

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_bci.txt
   ```

2. **Install the BLE SDK** (if not already installed):
   ```bash
   pip install -e .
   ```

## Usage

### Step 1: Test Device Connection

First, test if your BCI device can be detected:

```bash
python test_scanner.py
```

This will scan for BLE devices and show you the available devices. Make sure your BCI device is:
- Turned on
- In pairing mode
- Within range of your computer

### Step 2: Run the BCI Classifier

```bash
python bci_motor_imagery_classifier_v2.py
```

## How It Works

### Phase 1: Calibration (4 Sessions)

The system will guide you through 4 calibration sessions, each lasting 10 seconds:

1. **Right Hand Session**: Think about moving your right hand
2. **Left Hand Session**: Think about moving your left hand  
3. **Right Feet Session**: Think about moving your right foot
4. **Left Feet Session**: Think about moving your left foot

**Instructions for each session**:
- Get comfortable and relax
- When prompted, focus on imagining the specific movement
- Try to maintain consistent mental imagery throughout the 10 seconds
- Avoid physical movements - only mental imagery

### Phase 2: Real-time Classification

After calibration, the system will:
1. Collect 10 seconds of EEG data
2. Process and classify the data
3. Display the prediction: "RIGHT HAND", "LEFT HAND", "RIGHT FEET", or "LEFT FEET"
4. Repeat the process until you stop it (Ctrl+C)

## Technical Details

### Signal Processing Pipeline

1. **Data Collection**: Raw EEG data from 8 channels at 500 Hz
2. **Parsing**: Convert raw byte data to voltage values using device-specific algorithm
3. **Filtering**: 
   - Bandpass filter (2-40 Hz) to focus on relevant brain activity
   - Notch filter (50 Hz) to remove power line interference
4. **Feature Extraction**:
   - **Time Domain**: Mean, std, variance, min, max, percentiles, median
   - **Frequency Domain**: Power in delta, theta, alpha, beta, and mu bands
   - **Spectral**: Dominant frequency and peak amplitude
5. **Classification**: Random Forest classifier with 100 trees

### Data Format

The system handles the specific data format from your BLE device:
- **Packet Structure**: 198 bytes per packet
- **Header**: 5 bytes (frame header)
- **Data**: 192 bytes (8 channels × 64 samples × 3 bytes per sample)
- **Footer**: 1 byte
- **Sample Rate**: 500 Hz
- **Resolution**: 3 bytes per sample (24-bit)

## Troubleshooting

### Common Issues

1. **No devices found**:
   - Ensure Bluetooth is enabled
   - Check if device is in pairing mode
   - Verify device is within range

2. **Connection errors**:
   - Restart the device
   - Check battery level
   - Ensure proper electrode placement

3. **Poor classification accuracy**:
   - Ensure good electrode contact
   - Minimize movement during calibration
   - Focus on consistent mental imagery
   - Consider longer calibration sessions

### Performance Tips

1. **For Better Accuracy**:
   - Ensure electrodes are properly placed and have good contact
   - Minimize physical movement during data collection
   - Practice consistent mental imagery
   - Use longer calibration sessions (15-20 seconds each)

2. **For Real-time Performance**:
   - Close unnecessary applications
   - Ensure stable Bluetooth connection
   - Use shorter classification windows (5-10 seconds)

## Customization

### Modifying Classification Classes

To change the motor imagery tasks, modify the `classes` list in the `BCIMotorImageryClassifier` class:

```python
self.classes = ['right_hand', 'left_hand', 'right_feet', 'left_feet']
```

### Adjusting Filter Parameters

Modify the filter frequencies in the `__init__` method:

```python
self.low_freq = 2.0      # Lower bandpass frequency
self.high_freq = 40.0    # Upper bandpass frequency  
self.notch_freq = 50.0   # Notch filter frequency
```

### Changing Collection Duration

Modify the `duration` parameter in the calibration and classification methods:

```python
eeg_data = await self.collect_calibration_data(class_label, duration=15)  # 15 seconds
```

## Files Description

- `bci_motor_imagery_classifier_v2.py`: Main BCI classifier script
- `test_scanner.py`: Simple scanner test script
- `requirements_bci.txt`: Required Python packages
- `BCI_README.md`: This documentation file

## Safety Notes

- This system is for research and educational purposes
- Do not use while driving or operating machinery
- If you experience any discomfort, stop using the system immediately
- Consult a healthcare professional if you have any concerns

## License

This project is licensed under the MIT License - see the LICENSE file for details.
