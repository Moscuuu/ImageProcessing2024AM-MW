
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d


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
    factor = float(factor)
    print("Function doEnlarge invoked with factor:", factor)

    height = len(arr)
    width = len(arr[0])
    if arr.ndim == 3:
        num_channels = arr.shape[2]
    else:
        num_channels = 1

    new_height = int(height * factor)
    new_width = int(width * factor)

    if num_channels == 1:
        new_arr = np.zeros((new_height, new_width), dtype=arr.dtype)
    else:
        new_arr = np.zeros((new_height, new_width, num_channels), dtype=arr.dtype)

    for i in range(new_height):
        for j in range(new_width):
            if num_channels == 1:
                new_arr[i, j] = arr[int(i / factor), int(j / factor)]
            else:
                for c in range(num_channels):
                    new_arr[i, j, c] = arr[int(i / factor), int(j / factor), c]

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
    print("Applying Enhanced Adaptive Median Filter with max size:", max_filter_size)
    max_filter_size = int(max_filter_size)
    pad_size = max_filter_size // 2

    # Efficient padding
    if arr.ndim == 2:  # Grayscale
        padded_arr = np.pad(arr, pad_size, mode='reflect')
        channels = 1
    else:  # Color
        padded_arr = np.pad(arr, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
        channels = arr.shape[2]

    new_arr = np.zeros_like(arr)
    height, width = arr.shape[:2]

    # Precompute window ranges for efficiency
    windows = [np.arange(-(size // 2), (size // 2) + 1) for size in range(3, max_filter_size + 1, 2)]

    def adaptive_median(window):
        window_flat = window.flatten()
        zmin = np.min(window_flat)
        zmax = np.max(window_flat)
        zmed = np.median(window_flat)
        zxy = window[window.shape[0] // 2, window.shape[1] // 2]

        # Ensure scalar comparisons
        if zmin < zmed < zmax:
            return zxy if zmin < zxy < zmax else zmed
        return None

    for c in range(channels):
        for i in range(height):
            for j in range(width):
                result = None
                for window_offsets in windows:
                    x, y = i + pad_size, j + pad_size
                    if channels == 1:  # Grayscale
                        window = padded_arr[x + window_offsets[:, None], y + window_offsets]
                    else:  # Color
                        window = padded_arr[x + window_offsets[:, None], y + window_offsets, c]

                    result = adaptive_median(window)
                    if result is not None:
                        break
                if channels == 1:
                    new_arr[i, j] = result if result is not None else arr[i, j]
                else:
                    new_arr[i, j, c] = result if result is not None else arr[i, j, c]

    return new_arr


def mean_square_error(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same dimensions.")

    if arr1.ndim == 3 and arr1.shape[2] == 3:  # Obraz RGB
        mse_channels = []
        for c in range(3):  # Obliczanie dla każdego kanału
            mse_channel = np.mean((arr1[:, :, c] - arr2[:, :, c]) ** 2)
            mse_channels.append(mse_channel)
        overall_mse = np.mean(mse_channels)  # Uśredniona wartość
        return mse_channels, overall_mse
    else:  # Obraz w odcieniach szarości
        mse = np.mean((arr1 - arr2) ** 2)
        return mse

def peak_mean_square_error(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same dimensions.")

    max_pixel_value = 255.0

    if arr1.ndim == 3 and arr1.shape[2] == 3:  # Obraz RGB
        pmse_channels = []
        for c in range(3):  # Obliczanie dla każdego kanału
            mse_channel = np.mean((arr1[:, :, c] - arr2[:, :, c]) ** 2)
            pmse_channel = mse_channel / (max_pixel_value ** 2)
            pmse_channels.append(pmse_channel)
        overall_pmse = np.mean(pmse_channels)  # Uśredniona wartość
        return pmse_channels, overall_pmse
    else:  # Obraz w odcieniach szarości
        mse = np.mean((arr1 - arr2) ** 2)
        pmse = mse / (max_pixel_value ** 2)
        return pmse

def signal_to_noise_ratio(original, noisy):
    if original.shape != noisy.shape:
        raise ValueError("Input arrays must have the same dimensions.")

    epsilon = 1e-10  # Small value to avoid division by zero

    if original.ndim == 3 and original.shape[2] == 3:  # RGB image
        snr = []
        for channel in range(3):  # Per-channel computation
            mean_signal = np.mean(original[:, :, channel])
            noise_power = np.mean((original[:, :, channel] - noisy[:, :, channel]) ** 2)
            if noise_power < epsilon:
                snr.append(float('inf'))  # Perfect reconstruction case
            else:
                snr_channel = 10 * np.log10((mean_signal ** 2) / (noise_power + epsilon))
                snr.append(snr_channel)
        return snr
    else:  # Grayscale image
        mean_signal = np.mean(original)
        noise_power = np.mean((original - noisy) ** 2)
        if noise_power < epsilon:
            return float('inf')  # Perfect reconstruction case
        else:
            snr = 10 * np.log10((mean_signal ** 2) / (noise_power + epsilon))
            return snr

def peak_signal_to_noise_ratio(original, noisy):
    if original.shape != noisy.shape:
        raise ValueError("Input arrays must have the same dimensions.")

    max_pixel_value = 255.0

    if original.ndim == 3 and original.shape[2] == 3:  # RGB image
        psnr = []
        for channel in range(3):
            mse = np.mean((original[:, :, channel] - noisy[:, :, channel]) ** 2)
            if mse == 0:
                psnr_channel = float('inf')
            else:
                psnr_channel = 10 * np.log10((max_pixel_value ** 2) / mse)
            psnr.append(psnr_channel)
        return psnr
    else:  # Grayscale image
        mse = np.mean((original - noisy) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
        return psnr

def maximum_difference(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same dimensions.")

    # Calculate the absolute differences
    differences = np.abs(arr1 - arr2)

    # If the image is multi-channel (e.g., RGB)
    if arr1.ndim == 3 and arr1.shape[2] == 3:  # 3 channels (R, G, B)
        max_diffs = []
        for channel in range(arr1.shape[2]):
            max_diffs.append(np.max(differences[:, :, channel]))
        return max_diffs  # Return a list of maximum differences per channel
    else:  # Grayscale image
        max_diff = np.max(differences)
        return max_diff  # Return a single value for grayscale images


###########################

# TASK 2 PART

###########################

def create_histogram(image, output_filename, channel=None):

    if image.ndim == 2:  # Grayscale image
        histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
        plt.figure()
        plt.plot(histogram, color='black')
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.savefig(output_filename)
        plt.close()
        return {"grayscale": histogram.tolist()}

    elif image.ndim == 3 and image.shape[2] == 3:  # RGB image
        histograms = {}
        plt.figure()

        # Check if a specific channel is requested
        if channel:
            channel_map = {'R': 0, 'G': 1, 'B': 2}
            if channel.upper() in channel_map:
                idx = channel_map[channel.upper()]
                histogram, bins = np.histogram(image[:, :, idx].flatten(), bins=256, range=[0, 256])
                plt.plot(histogram, color=channel.lower())
                plt.title(f"{channel.upper()} Channel Histogram")
                plt.xlabel("Pixel Intensity")
                plt.ylabel("Frequency")
                histograms[channel.upper()] = histogram.tolist()
            else:
                raise ValueError("Invalid channel specified. Use 'R', 'G', or 'B'.")
        else:  # Generate histograms for all channels
            colors = ['r', 'g', 'b']
            for i, color in enumerate(colors):
                histogram, bins = np.histogram(image[:, :, i].flatten(), bins=256, range=[0, 256])
                plt.plot(histogram, color=color, label=f"{color.upper()} Channel")
                histograms[color.upper()] = histogram.tolist()
            plt.title("RGB Histogram")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.legend()

        plt.savefig(output_filename)
        plt.close()
        return histograms

    else:
        raise ValueError("Unsupported image format.")

def enhance_image_based_on_histogram(image, g_min, g_max):
    """
    Enhance the image brightness based on the provided histogram equation.

    :param image: Input grayscale image as a 2D numpy array.
    :param g_min: Minimum brightness for the output image.
    :param g_max: Maximum brightness for the output image.
    :return: Enhanced image as a 2D numpy array.
    """
    # Ensure the image is grayscale
    if image.ndim != 2:
        raise ValueError("This method works only for grayscale images.")

    # Calculate histogram and cumulative histogram
    L = 256  # Number of grayscale levels
    hist, bins = np.histogram(image.flatten(), bins=L, range=[0, L])
    cumulative_hist = np.cumsum(hist)
    N = image.size  # Total number of pixels

    # Calculate g(f) transformation function
    g_min_cubed = g_min ** (1/3)
    g_max_cubed = g_max ** (1/3)
    g_f = np.zeros(L)  # Array to store g(f) for each intensity value f

    for f in range(L):
        cumulative_sum = cumulative_hist[f] / N
        g_f[f] = (g_min_cubed + (g_max_cubed - g_min_cubed) * cumulative_sum) ** 3

        # Apply the transformation to the image
    enhanced_image = np.zeros_like(image, dtype=np.float32)
    for f in range(L):
        enhanced_image[image == f] = g_f[f]

    # Normalize the enhanced image to 0-255 range for saving
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

    return enhanced_image

def enhance_image_rgb(image, g_min, g_max):
    """
    Enhance the brightness of an RGB image using histogram-based transformation.
    The operation is performed on the Lightness channel in HSL color space.

    :param image: Input RGB image as a 3D numpy array.
    :param g_min: Minimum brightness for the output image.
    :param g_max: Maximum brightness for the output image.
    :return: Enhanced RGB image as a 3D numpy array.
    """
    # Upewnij się, że obraz jest w formacie uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Convert RGB image to HSL
    hsl_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    L = hsl_image[:, :, 1]  # Extract Lightness channel

    # Apply grayscale enhancement function to the Lightness channel
    enhanced_L = enhance_image_based_on_histogram(L, g_min, g_max)

    # Replace the Lightness channel in the HSL image
    hsl_image[:, :, 1] = enhanced_L

    # Convert the image back to RGB
    enhanced_image = cv2.cvtColor(hsl_image, cv2.COLOR_HLS2RGB)

    return enhanced_image

def compute_mean(arr):
    """
    Compute the mean for each channel in an image array.

    Parameters:
        arr (numpy.ndarray): Image array.

    Returns:
        list: Mean for each channel.
    """
    if arr.ndim == 3:  # RGB image
        return [np.mean(arr[:, :, c]) for c in range(arr.shape[2])]
    else:  # Grayscale image
        return [np.mean(arr)]


def compute_variance(arr):
    """
    Compute the variance for each channel in an image array.

    Parameters:
        arr (numpy.ndarray): Image array.

    Returns:
        list: Variance for each channel.
    """
    if arr.ndim == 3:  # RGB image
        return [np.var(arr[:, :, c]) for c in range(arr.shape[2])]
    else:  # Grayscale image
        return [np.var(arr)]


def compute_std_dev(arr):
    """
    Compute the standard deviation for each channel in an image array.

    Parameters:
        arr (numpy.ndarray): Image array.

    Returns:
        list: Standard deviation for each channel.
    """
    if arr.ndim == 3:  # RGB image
        return [np.std(arr[:, :, c]) for c in range(arr.shape[2])]
    else:  # Grayscale image
        return [np.std(arr)]


def compute_var_coefficient_i(arr):
    """
    Compute the variation coefficient I for each channel.

    Parameters:
        arr (numpy.ndarray): Image array.

    Returns:
        list: Variation coefficient I for each channel.
    """
    means = compute_mean(arr)
    std_devs = compute_std_dev(arr)
    return [std_dev / mean if mean != 0 else 0 for std_dev, mean in zip(std_devs, means)]


def compute_asymmetry_coefficient(arr):
    """
    Compute the asymmetry coefficient for each channel.

    Parameters:
        arr (numpy.ndarray): Image array.

    Returns:
        list: Asymmetry coefficient for each channel.
    """
    means = compute_mean(arr)
    std_devs = compute_std_dev(arr)
    if arr.ndim == 3:
        return [
            np.mean((arr[:, :, c] - mean) ** 3) / (std_dev ** 3) if std_dev != 0 else 0
            for c, (mean, std_dev) in enumerate(zip(means, std_devs))
        ]
    else:
        return [
            np.mean((arr - means[0]) ** 3) / (std_devs[0] ** 3) if std_devs[0] != 0 else 0
        ]


def compute_flattening_coefficient(arr):
    """
    Compute the flattening coefficient for each channel.

    Parameters:
        arr (numpy.ndarray): Image array.

    Returns:
        list: Flattening coefficient for each channel.
    """
    means = compute_mean(arr)
    std_devs = compute_std_dev(arr)
    if arr.ndim == 3:
        return [
            np.mean((arr[:, :, c] - mean) ** 4) / (std_dev ** 4) - 3 if std_dev != 0 else 0
            for c, (mean, std_dev) in enumerate(zip(means, std_devs))
        ]
    else:
        return [
            np.mean((arr - means[0]) ** 4) / (std_devs[0] ** 4) - 3 if std_devs[0] != 0 else 0
        ]


def compute_var_coefficient_ii(arr):
    """
    Compute the variation coefficient II for each channel.

    Parameters:
        arr (numpy.ndarray): Image array.

    Returns:
        list: Variation coefficient II for each channel.
    """
    means = compute_mean(arr)
    variances = compute_variance(arr)
    return [var / (mean ** 2) if mean != 0 else 0 for var, mean in zip(variances, means)]


def compute_entropy(arr):
    """
    Compute the information source entropy for each channel.

    Parameters:
        arr (numpy.ndarray): Image array.

    Returns:
        list: Entropy for each channel.
    """
    if arr.ndim == 3:  # RGB image
        entropies = []
        for c in range(arr.shape[2]):
            values, counts = np.unique(arr[:, :, c], return_counts=True)
            probabilities = counts / np.sum(counts)
            entropies.append(-np.sum(probabilities * np.log2(probabilities)))
        return entropies
    else:  # Grayscale image
        values, counts = np.unique(arr, return_counts=True)
        probabilities = counts / np.sum(counts)
        return [-np.sum(probabilities * np.log2(probabilities))]

def apply_filter_universal(image_array, mask):
    """
    Apply a universal convolution filter to an image.

    Parameters:
        image_array (numpy.ndarray): Image array (grayscale or RGB).
        mask (numpy.ndarray): Convolution mask.

    Returns:
        numpy.ndarray: Filtered image array.
    """
    if image_array.ndim == 3:  # RGB Image
        filtered_image = np.zeros_like(image_array)
        for c in range(3):  # Apply the mask to each channel separately
            filtered_image[:, :, c] = convolve2d(image_array[:, :, c], mask, mode='same', boundary='symm')
        return np.clip(filtered_image, 0, 255).astype(np.uint8)
    else:  # Grayscale Image
        filtered_image = convolve2d(image_array, mask, mode='same', boundary='symm')
        return np.clip(filtered_image, 0, 255).astype(np.uint8)

def optimized_filter(image_array):
    """
    Optimized filter for a specific mask (hₛ).

    Parameters:
        image_array (numpy.ndarray): Input image (grayscale or RGB).

    Returns:
        numpy.ndarray: Filtered image.
    """
    # Predefined mask (hₛ)
    mask = np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]])

    if image_array.ndim == 3:  # RGB Image
        # Process channel-by-channel without creating full intermediate arrays
        filtered_channels = []
        for c in range(3):  # Iterate over R, G, B
            channel = image_array[:, :, c]
            filtered_channel = convolve2d(channel, mask, mode='same', boundary='symm')
            filtered_channels.append(np.clip(filtered_channel, 0, 255).astype(np.uint8))
        return np.stack(filtered_channels, axis=-1)  # Combine channels
    else:  # Grayscale Image
        # Directly process grayscale
        filtered_image = convolve2d(image_array, mask, mode='same', boundary='symm')
        return np.clip(filtered_image, 0, 255).astype(np.uint8)


def rosenfeld_operator_optimized(image_array, P):
    """
    Optimized Rosenfeld operator using cumulative sums.

    Parameters:
        image_array (numpy.ndarray): Input image (grayscale or RGB).
        P (int): Neighborhood size parameter (e.g., 1, 2, 4, 8, etc.).

    Returns:
        numpy.ndarray: Filtered image.
    """
    if P <= 0:
        raise ValueError("P must be a positive integer.")

    if image_array.ndim == 3:  # RGB Image
        # Process each channel independently
        filtered_image = np.zeros_like(image_array, dtype=np.float32)
        for c in range(3):
            filtered_image[:, :, c] = rosenfeld_single_channel_optimized(image_array[:, :, c], P)
        return np.clip(filtered_image, 0, 255).astype(np.uint8)
    else:  # Grayscale Image
        return rosenfeld_single_channel_optimized(image_array, P)


def rosenfeld_single_channel_optimized(channel, P):
    """
    Optimized Rosenfeld operator for a single channel using cumulative sums.

    Parameters:
        channel (numpy.ndarray): Single-channel input image.
        P (int): Neighborhood size parameter.

    Returns:
        numpy.ndarray: Filtered channel.
    """
    height, width = channel.shape

    # Pad the image to handle edges (P rows on both top and bottom)
    padded_channel = np.pad(channel, ((P, P), (0, 0)), mode='edge')

    # Compute cumulative sum along the vertical axis
    cumsum = np.cumsum(padded_channel, axis=0)

    # Compute forward and backward sums using cumulative sums
    forward_sum = cumsum[P:height + P, :] - cumsum[:height, :]
    backward_sum = cumsum[2 * P:height + 2 * P, :] - cumsum[P:height + P, :]

    # Compute the Rosenfeld output
    filtered_channel = (1 / P) * (forward_sum - backward_sum)

    return filtered_channel

###########################
# HERE THE MAIN PART STARTS
###########################
#im = Image.open("lena.bmp")
im = Image.open("lenac.bmp")
im2 = Image.open("lenac.bmp")

arr = np.array(im.getdata())
arr2 = np.array(im2.getdata())

mask = np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]]) # mask for "W" filter

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
    elif sys.argv[1] == '--histogram':
        output_filename = "histogram.png"
        if numColorChannels == 1:
            hist_data = create_histogram(arr, output_filename)
            print("Grayscale Histogram saved to:", output_filename)
            print("Histogram data:", hist_data)
        else:
            hist_data = create_histogram(arr, output_filename)
            print("RGB Histogram saved to:", output_filename)
            print("Histogram data:", hist_data)
    elif sys.argv[1] == '--cmean':
        mean = compute_mean(arr)
        print("Mean:", mean)
    elif sys.argv[1] == '--cvariance':
        variance = compute_variance(arr)
        print("Variance:", variance)
    elif sys.argv[1] == '--cstdev':
        std_dev = compute_std_dev(arr)
        print("Standard Deviation:", std_dev)
    elif sys.argv[1] == '--cvarcoi':
        var_coeff_i = compute_var_coefficient_i(arr)
        print("Variation Coefficient I:", var_coeff_i)
    elif sys.argv[1] == '--casyco':
        asymmetry_coeff = compute_asymmetry_coefficient(arr)
        print("Asymmetry Coefficient:", asymmetry_coeff)
    elif sys.argv[1] == '--cflatco':
        flattening_coeff = compute_flattening_coefficient(arr)
        print("Flattening Coefficient:", flattening_coeff)
    elif sys.argv[1] == '--cvarcoii':
        var_coeff_ii = compute_var_coefficient_ii(arr)
        print("Variation Coefficient II:", var_coeff_ii)
    elif sys.argv[1] == '--centropy':
        entropy = compute_entropy(arr)
        print("Entropy:", entropy)
    elif sys.argv[1] == '--sexdetii':
        arr = apply_filter_universal(arr, mask)
        print("Universal Filter applied.")
    elif sys.argv[1] == '--sexdet':
        arr = optimized_filter(arr)
        print("Optimized Filter applied.")
    else:
        print("Too few command line parameters given.\n")
        sys.exit()
elif len(sys.argv) == 4:  # Adding parameters for histogram power transformation
    command = sys.argv[1]
    g_min = float(sys.argv[2])
    g_max = float(sys.argv[3])

    if command == '--hpower':
        if numColorChannels == 1:  # Grayscale image
            arr = enhance_image_based_on_histogram(arr, g_min, g_max)
        else:
            print("Error: --hpower only works on grayscale images.")
            sys.exit()

    elif command == '--hpowerrgb':
        if numColorChannels == 3:
            arr = enhance_image_rgb(arr, g_min, g_max)
    else:
        print("Unknown command or incorrect parameters.")
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
    elif command == '--histogram' and numColorChannels > 1:
        output_filename = "channel_histogram.png"
        hist_data = create_histogram(arr, output_filename, param)
        print(f"{param.upper()} Channel Histogram saved to:", output_filename)
        print(f"{param.upper()} Histogram data:", hist_data)
    elif command == '--orosenfeld':
        arr = rosenfeld_operator_optimized(arr, int(param))
        print("Rosenfeld Operator applied.")
    else:
        print("Unknown command: " + command)
        sys.exit()

newIm = Image.fromarray(arr.astype(np.uint8))
newIm.save("result.bmp")