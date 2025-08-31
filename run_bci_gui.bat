@echo off
echo Starting XBCI Motor Imagery Classifier GUI...
echo.
echo This GUI allows you to:
echo - Train and use offline classification on saved data files
echo - View classification results with class labels 0, 1, 2, 3
echo - Save and load trained classifiers
echo - Professional interface with XBCI logo
echo.
echo Make sure you have Python and the required packages installed.
echo Required packages: numpy, scipy, scikit-learn, loguru, joblib, Pillow
echo.
pause
python simple_bci_gui.py
pause
