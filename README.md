
# Image Processing Toolkit

## Overview

This repository contains a comprehensive set of image processing functions implemented in Python. The toolkit includes various image enhancement, filtering, and analysis techniques, leveraging libraries such as NumPy, OpenCV, and PIL.

## Features

- **Basic Image Operations**: Brightness, contrast adjustment, flipping, enlarging, and shrinking.
- **Filtering**: Arithmetic mean filter, adaptive median filter, universal filter, optimized filter, and Rosenfeld operator.
- **Morphological Operations**: Dilation, erosion, opening, closing, and hit-or-miss transformation.
- **Histogram Analysis**: Grayscale and RGB histogram creation, histogram-based image enhancement.
- **Statistical Analysis**: Mean, variance, standard deviation, variation coefficients, asymmetry coefficient, flattening coefficient, and entropy.
- **Error Metrics**: Mean Square Error (MSE), Peak Mean Square Error (PMSE), Signal-to-Noise Ratio (SNR), Peak Signal-to-Noise Ratio (PSNR), and Maximum Difference.
- **Frequency Domain Analysis**: Discrete Fourier Transform (DFT), Fast Fourier Transform (FFT), lowpass, highpass, bandpass, bandcut, directional highpass, and phase modifying filters.
- **Region Growing**: Segmentation based on seed point and intensity threshold.

## Setup

### Prerequisites

- Python 3.x
- Required Libraries: NumPy, OpenCV, PIL, Matplotlib, SciPy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-processing-toolkit.git
   cd image-processing-toolkit
   ```

2. Install the required libraries:
   ```bash
   pip install numpy opencv-python pillow matplotlib scipy
   ```

## Usage

### Command Line Interface

The toolkit can be run from the command line with various options. Here are some examples:

- **Basic Operations**:
  ```bash
  python main.py --brightness 50
  python main.py --contrast 1.5
  python main.py --enlarge 2.0
  python main.py --shrink 2
  ```

- **Filtering**:
  ```bash
  python main.py --arithmeticMeanFilter 3
  python main.py --adaptiveMedianFilter 5
  python main.py --sexdetii
  python main.py --sexdet
  ```

- **Morphological Operations**:
  ```bash
  python main.py --dilate 1
  python main.py --erode 2
  python main.py --opening 3
  python main.py --closing 4
  python main.py --hmt 5
  ```

- **Histogram Analysis**:
  ```bash
  python main.py --histogram
  python main.py --histogram R
  python main.py --hpower 10 240
  python main.py --hpowerrgb 10 240
  ```

- **Statistical Analysis**:
  ```bash
  python main.py --cmean
  python main.py --cvariance
  python main.py --cstdev
  python main.py --cvarcoi
  python main.py --casyco
  python main.py --cflatco
  python main.py --cvarcoii
  python main.py --centropy
  ```

- **Error Metrics**:
  ```bash
  python main.py --mse
  python main.py --pmse
  python main.py --snr
  python main.py --psnr
  python main.py --maxdiff
  ```

- **Frequency Domain Analysis**:
  ```bash
  python main.py --dft-slow
  python main.py --fft-time
  python main.py --dft-spectrum
  python main.py --fft-spectrum
  python main.py --lowpass 30
  python main.py --highpass 10
  python main.py --bandpass 10 30
  python main.py --bandcut 10 30
  python main.py --directionalHighpass 30 0 0.5
  python main.py --phase 1 1
  ```

- **Region Growing**:
  ```bash
  python main.py --regiongrow 100 150 20
  ```

### Output

The processed image is saved as `result.bmp` or `result1.bmp` (for binary images) in the repository directory.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
