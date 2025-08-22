# BCI Motor Imagery Competition - Setup Guide

This folder contains the implementation for the BCI Motor Imagery Classification competition. The system can classify four different motor imagery tasks: right hand, left hand, right feet, and left feet movements using EEG data from an 8-channel BLE device.

## 🚀 Quick Start

### Prerequisites

1. **Hardware Requirements**:
   - BCI device compatible with the BLE SDK (8-channel EEG device)
   - Computer with Bluetooth capability
   - Proper electrode placement for EEG recording

2. **Software Requirements**:
   - Python 3.8 or higher
   - All required dependencies (see installation below)

### Installation

1. **Navigate to the project root**:
   ```bash
   cd /path/to/XBCI-competition
   ```

2. **Install the BLE SDK**:
   ```bash
   pip install -e .
   ```

3. **Install BCI dependencies**:
   ```bash
   pip install -r requirements_bci.txt
   ```

## 📁 File Structure

The main BCI implementation files are located in the project root:

```
XBCI-competition/
├── bci_motor_imagery_classifier_v2.py    # Main BCI classifier
├── test_scanner.py                       # Device connection test
├── requirements_bci.txt                  # Python dependencies
├── BCI_README.md                         # Detailed documentation
```

## 🧠 Running the BCI System

### Step 1: Test Device Connection

First, ensure your BCI device is properly connected:

```bash
# From the project root directory
python test_scanner.py
```

**Expected Output**:
```
BLE Scanner Test
==============================
Testing BLE Scanner...
Found 1 device(s):
  1. OurDeviceName (D6:5A:01:75:CD:7E) - CD7E ID39 Nervoviden
Scanner test completed successfully!
You can now run the BCI classifier.
```

**If no devices are found**:
- Ensure your BCI device is turned on
- Check that Bluetooth is enabled on your computer
- Verify the device is in pairing mode
- Make sure electrodes are properly placed

### Step 2: Run the BCI Classifier

```bash
# From the project root directory
python bci_motor_imagery_classifier_v2.py
```

## 🎯 How the System Works

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
- Keep your eyes open and minimize blinking

### Phase 2: Real-time Classification

After calibration, the system will:
1. Collect 10 seconds of EEG data
2. Process and classify the data
3. Display the prediction: "RIGHT HAND", "LEFT HAND", "RIGHT FEET", or "LEFT FEET"
4. Repeat the process until you stop it (Ctrl+C)

## 🔧 Technical Details

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

## 🎯 Competition Guidelines

### Best Practices for High Accuracy

1. **Electrode Placement**:
   - Ensure all electrodes have good contact with the scalp
   - Use conductive gel if necessary
   - Check impedance levels if your device supports it

2. **Mental Imagery Technique**:
   - Practice consistent mental imagery before the competition
   - Focus on the specific movement without physical execution
   - Maintain concentration throughout each session

3. **Environment**:
   - Minimize external distractions
   - Ensure stable Bluetooth connection
   - Avoid movement during data collection

### Performance Optimization

1. **For Better Accuracy**:
   - Use longer calibration sessions (15-20 seconds each)
   - Practice mental imagery techniques beforehand
   - Ensure good electrode contact throughout the session

2. **For Real-time Performance**:
   - Close unnecessary applications
   - Use shorter classification windows (5-10 seconds)
   - Maintain stable system performance

## 🛠️ Troubleshooting

### Common Issues

1. **No devices found**:
   ```
   No BLE devices found!
   Please ensure:
   1. Your BCI device is turned on
   2. Bluetooth is enabled on your computer
   3. The device is in pairing mode
   ```
   - Check device power and pairing mode
   - Verify Bluetooth is enabled
   - Ensure device is within range

2. **Connection errors**:
   ```
   Error during BCI session: Connection failed
   ```
   - Restart the device
   - Check battery level
   - Ensure proper electrode placement

3. **Poor classification accuracy**:
   - Ensure good electrode contact
   - Minimize movement during calibration
   - Focus on consistent mental imagery
   - Consider longer calibration sessions

### Error Messages and Solutions

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'ble_sdk'` | Run `pip install -e .` from project root |
| `No BLE devices found` | Check device power and pairing mode |
| `Connection failed` | Restart device and check Bluetooth |
| `Training accuracy: 0.25` | Improve electrode contact and mental imagery |

## 📊 Expected Results

### Typical Performance

- **Training Accuracy**: 0.75 - 1.0 (depends on user and setup)
- **Real-time Classification**: Should correctly identify motor imagery tasks
- **Response Time**: ~10 seconds per classification

### Success Indicators

- Scanner finds your device successfully
- Calibration completes without errors
- Training accuracy > 0.75
- Real-time predictions are consistent

## 🔄 Customization

### Modifying Classification Classes

To change the motor imagery tasks, edit `bci_motor_imagery_classifier_v2.py`:

```python
# Line ~40: Change the classes list
self.classes = ['right_hand', 'left_hand', 'right_feet', 'left_feet']
```

### Adjusting Parameters

```python
# Filter frequencies (lines ~35-37)
self.low_freq = 2.0      # Lower bandpass frequency
self.high_freq = 40.0    # Upper bandpass frequency  
self.notch_freq = 50.0   # Notch filter frequency

# Collection duration (lines ~180, ~250)
duration=10  # Change to desired duration in seconds
```