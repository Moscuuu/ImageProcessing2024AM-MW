
from PIL import Image
import numpy as np
import sys


######################
# FUNCTION DEFINITIONS
######################

def doBrightness(arr, param):
    print("Function doBrightness invoked with param: " + param)
    arr = (arr + float(param)).astype(np.int64)
    arr[arr > 255] = 255
    arr[arr < 0] = 0
    return arr

def doContrast(arr, param):
    print("Function doContrast invoked with param: " + param)
    arr = ((arr - 128) * float(param) + 128).astype(np.int64)
    arr[arr > 255] = 255
    arr[arr < 0] = 0
    return arr

def doNegation(arr):
    arr = 255 - arr
    arr[arr > 255] = 255
    arr[arr < 0] = 0
    return arr

def doVerticalFlip(arr):
    arr = arr[::-1]
    return arr

def doHorizontalFlip(arr):
    arr = arr[:, ::-1]
    return arr

def doDiagonalFlip(arr):
    arr = arr[::-1, ::-1]
    return arr

def doEnlarge(arr, factor):
    print("Function doEnlarge invoked with factor: " + factor)
    factor = float(factor)
    height = len(arr)
    width = len(arr[0])
    new_height = int(height * factor)
    new_width = int(width * factor)

    new_arr = [[0] * new_width for _ in range(new_height)]

    for i in range(new_height):
        for j in range(new_width):
            new_arr[i][j] = arr[int(i / factor)][int(j / factor)]

    return new_arr

def doShrink(arr, factor):
    print("Function doShrink invoked with factor: " + factor)
    arr = arr[::int(factor), ::int(factor)]
    return arr

def doArithmeticMeanFilter(arr, filter_size):
    print("Applying Arithmetic Mean Filter with size:", filter_size)
    filter_size = int(filter_size)
    pad_size = filter_size // 2

    if arr.ndim == 2:  # grayscale
        padded_arr = np.pad(arr, pad_size, mode='constant', constant_values=0)
    else:  # color
        padded_arr = np.pad(arr, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant',
                            constant_values=0)

    new_arr = np.zeros_like(arr)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr.ndim == 2:  # grayscale
                new_arr[i, j] = np.mean(padded_arr[i:i + filter_size, j:j + filter_size])
            else:  # color
                for c in range(arr.shape[2]):
                    new_arr[i, j, c] = np.mean(padded_arr[i:i + filter_size, j:j + filter_size, c])

    return new_arr

def doAdaptiveMedianFilter(arr, max_filter_size):
    print("Applying Adaptive Median Filter with max size:", max_filter_size)
    max_filter_size = int(max_filter_size)
    pad_size = max_filter_size // 2
    if arr.ndim == 2:  # grayscale
        padded_arr = np.pad(arr, pad_size, mode='constant', constant_values=0)
    else:  # color
        padded_arr = np.pad(arr, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant',
                            constant_values=0)

    new_arr = np.zeros_like(arr)

    def adaptive_median(window, Smax):
        window_flat = window.flatten()
        zmin = np.min(window_flat)
        zmax = np.max(window_flat)
        zmed = np.median(window_flat)
        zxy = window[len(window) // 2, len(window) // 2]

        # Stage A
        A1 = zmed - zmin
        A2 = zmed - zmax

        if A1 > 0 and A2 < 0:
            # Stage B
            B1 = zxy - zmin
            B2 = zxy - zmax
            if np.all(B1 > 0) and np.all(B2 < 0):
                return zxy
            else:
                return zmed
        else:
            return None

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            window_size = 3
            result = None
            while window_size <= max_filter_size:
                half_size = window_size // 2
                window = padded_arr[i:i + window_size, j:j + window_size]
                result = adaptive_median(window, max_filter_size)
                if result is not None:
                    break
                window_size += 2

            if result is None:
                result = arr[i, j]
            new_arr[i, j] = result

    return new_arr

def mean_square_error(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same dimensions.")

    error = 0.0
    height, width, *channels = arr1.shape
    num_elements = height * width * (channels[0] if channels else 1)

    for i in range(height):
        for j in range(width):
            if channels:
                for c in range(channels[0]):
                    error += (arr1[i][j][c] - arr2[i][j][c]) ** 2
            else:
                error += (arr1[i][j] - arr2[i][j]) ** 2

    mse = error / num_elements
    return mse


def peak_mean_square_error(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same dimensions.")

    # Calculate MSE
    mse = np.mean((arr1 - arr2) ** 2)

    # Normalize by the square of the maximum possible pixel value
    max_pixel_value = 255.0
    pmse = mse / (max_pixel_value ** 2)

    return pmse

def signal_to_noise_ratio(original, noisy):
    if original.shape != noisy.shape:
        raise ValueError("Input arrays must have the same dimensions.")

    # Calculate the mean signal value
    mean_signal = np.mean(original)

    # Calculate the noise
    noise = original - noisy

    # Calculate the mean noise value
    mean_noise = np.mean(noise)

    # Compute SNR
    snr = 10 * np.log10((mean_signal ** 2) / (mean_noise ** 2))

    return snr

def peak_signal_to_noise_ratio(original, noisy):
    if original.shape != noisy.shape:
        raise ValueError("Input arrays must have the same dimensions.")

    # Calculate MSE
    mse = np.mean((original - noisy) ** 2)
    if mse == 0:
        return float('inf')

    # Maximum possible pixel value
    max_pixel_value = 255.0

    # Compute PSNR
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    return psnr

def maximum_difference(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same dimensions.")

    # Calculate the absolute differences
    differences = np.abs(arr1 - arr2)

    # Find the maximum difference
    max_diff = np.max(differences)

    return max_diff

###########################
# HERE THE MAIN PART STARTS
###########################
#im = Image.open("lena.bmp")
im = Image.open("lenac.bmp")
im2 = Image.open("result.bmp")

arr = np.array(im.getdata())
arr2 = np.array(im2.getdata())

if arr.ndim == 1: #grayscale
    numColorChannels = 1
    arr = arr.reshape(im.size[1], im.size[0])
    arr2 = arr2.reshape(im.size[1], im.size[0])
else:
    numColorChannels = arr.shape[1]
    arr = arr.reshape(im.size[1], im.size[0], numColorChannels)
    arr2 = arr2.reshape(im.size[1], im.size[0], numColorChannels)
if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

if len(sys.argv) == 2:
    if sys.argv[1] == '--negation':
        arr = doNegation(arr)
    elif sys.argv[1] == '--verticalFlip':
        arr = doVerticalFlip(arr)
    elif sys.argv[1] == '--horizontalFlip':
        arr = doHorizontalFlip(arr)
    elif sys.argv[1] == '--diagonalFlip':
        arr = doDiagonalFlip(arr)
    elif sys.argv[1] == '--mse':
        mse = mean_square_error(arr, arr2)
        print("Mean Square Error:", mse)
    elif sys.argv[1] == '--pmse':
        pmse = peak_mean_square_error(arr, arr2)
        print("Peak Mean Square Error:", pmse)
    elif sys.argv[1] == '--snr':
        snr = signal_to_noise_ratio(arr, arr2)
        print("Signal to Noise Ratio:", snr)
    elif sys.argv[1] == '--psnr':
        psnr = peak_signal_to_noise_ratio(arr, arr2)
        print("Peak Signal to Noise Ratio:", psnr)
    elif sys.argv[1] == '--maxdiff':
        max_diff = maximum_difference(arr, arr2)
        print("Maximum Difference:", max_diff)
    else:
        print("Too few command line parameters given.\n")
        sys.exit()
else:
    command = sys.argv[1]
    param = sys.argv[2]
    if command == '--brightness':
        arr = doBrightness(arr, param)
    elif command == '--contrast':
        arr = doContrast(arr, param)
    elif command == '--enlarge':
        arr = doEnlarge(arr, param)
    elif command == '--shrink':
        arr = doShrink(arr, param)
    elif command == '--arithmeticMeanFilter':
        arr = doArithmeticMeanFilter(arr, param)
    elif command == '--adaptiveMedianFilter':
        arr = doAdaptiveMedianFilter(arr, param)
    else:
        print("Unknown command: " + command)
        sys.exit()

newIm = Image.fromarray(arr.astype(np.uint8))
newIm.save("result.bmp")