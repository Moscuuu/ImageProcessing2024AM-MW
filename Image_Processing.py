from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from collections import deque

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
    if image.ndim == 2:
        histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
        plt.figure()
        plt.plot(histogram, color='black')
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.savefig(output_filename)
        plt.close()
        return {"grayscale": histogram.tolist()}

    elif image.ndim == 3 and image.shape[2] == 3:
        histograms = {}
        plt.figure()

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
        else:
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
    if image.ndim != 2:
        raise ValueError("This method works only for grayscale images.")

    L = 256
    hist, bins = np.histogram(image.flatten(), bins=L, range=[0, L])
    cumulative_hist = np.cumsum(hist)
    N = image.size

    g_min_cubed = g_min ** (1 / 3)
    g_max_cubed = g_max ** (1 / 3)
    g_f = np.zeros(L)

    for f in range(L):
        cumulative_sum = cumulative_hist[f] / N
        g_f[f] = (g_min_cubed + (g_max_cubed - g_min_cubed) * cumulative_sum) ** 3

    enhanced_image = np.zeros_like(image, dtype=np.float32)
    for f in range(L):
        enhanced_image[image == f] = g_f[f]

    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

    return enhanced_image


def enhance_image_rgb(image, g_min, g_max):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    hsl_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    L = hsl_image[:, :, 1]

    enhanced_L = enhance_image_based_on_histogram(L, g_min, g_max)

    hsl_image[:, :, 1] = enhanced_L

    enhanced_image = cv2.cvtColor(hsl_image, cv2.COLOR_HLS2RGB)

    return enhanced_image


def compute_mean(arr):
    if arr.ndim == 3:
        return [np.mean(arr[:, :, c]) for c in range(arr.shape[2])]
    else:
        return [np.mean(arr)]


def compute_variance(arr):
    if arr.ndim == 3:
        return [np.var(arr[:, :, c]) for c in range(arr.shape[2])]
    else:
        return [np.var(arr)]


def compute_std_dev(arr):
    if arr.ndim == 3:
        return [np.std(arr[:, :, c]) for c in range(arr.shape[2])]
    else:
        return [np.std(arr)]


def compute_var_coefficient_i(arr):
    means = compute_mean(arr)
    std_devs = compute_std_dev(arr)
    return [std_dev / mean if mean != 0 else 0 for std_dev, mean in zip(std_devs, means)]


def compute_asymmetry_coefficient(arr):
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
    means = compute_mean(arr)
    variances = compute_variance(arr)
    return [var / (mean ** 2) if mean != 0 else 0 for var, mean in zip(variances, means)]


def compute_entropy(arr):
    if arr.ndim == 3:
        entropies = []
        for c in range(arr.shape[2]):
            values, counts = np.unique(arr[:, :, c], return_counts=True)
            probabilities = counts / np.sum(counts)
            entropies.append(-np.sum(probabilities * np.log2(probabilities)))
        return entropies
    else:
        values, counts = np.unique(arr, return_counts=True)
        probabilities = counts / np.sum(counts)
        return [-np.sum(probabilities * np.log2(probabilities))]


def apply_filter_universal(image_array, mask):
    if image_array.ndim == 3:
        filtered_image = np.zeros_like(image_array)
        for c in range(3):
            filtered_image[:, :, c] = convolve2d(image_array[:, :, c], mask, mode='same', boundary='symm')
        return np.clip(filtered_image, 0, 255).astype(np.uint8)
    else:
        filtered_image = convolve2d(image_array, mask, mode='same', boundary='symm')
        return np.clip(filtered_image, 0, 255).astype(np.uint8)


def optimized_filter(image_array):
    mask = np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]])

    if image_array.ndim == 3:
        filtered_channels = []
        for c in range(3):
            channel = image_array[:, :, c]
            filtered_channel = convolve2d(channel, mask, mode='same', boundary='symm')
            filtered_channels.append(np.clip(filtered_channel, 0, 255).astype(np.uint8))
        return np.stack(filtered_channels, axis=-1)
    else:
        filtered_image = convolve2d(image_array, mask, mode='same', boundary='symm')
        return np.clip(filtered_image, 0, 255).astype(np.uint8)


def rosenfeld_operator_optimized(image_array, P):
    if P <= 0:
        raise ValueError("P must be a positive integer.")

    if image_array.ndim == 3:
        filtered_image = np.zeros_like(image_array, dtype=np.float32)
        for c in range(3):
            filtered_image[:, :, c] = rosenfeld_single_channel_optimized(image_array[:, :, c], P)
        return np.clip(filtered_image, 0, 255).astype(np.uint8)
    else:
        return rosenfeld_single_channel_optimized(image_array, P)


def rosenfeld_single_channel_optimized(channel, P):
    height, width = channel.shape

    padded_channel = np.pad(channel, ((P, P), (0, 0)), mode='edge')

    cumsum = np.cumsum(padded_channel, axis=0)

    forward_sum = cumsum[P:height + P, :] - cumsum[:height, :]
    backward_sum = cumsum[2 * P:height + 2 * P, :] - cumsum[P:height + P, :]

    filtered_channel = (1 / P) * (forward_sum - backward_sum)
    filtered_channel = np.abs(filtered_channel)

    return filtered_channel


###########################
# TASK 3
###########################

def dilate(image, structuring_element):

    se_height, se_width = structuring_element.shape
    pad_h = se_height // 2
    pad_w = se_width // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    output = np.zeros_like(image, dtype=np.uint8)
    for i in range(se_height):
        for j in range(se_width):
            if structuring_element[i, j] == 1:
                output = np.maximum(output, padded_image[i:i + image.shape[0], j:j + image.shape[1]])

    return output


def erode(image, structuring_element):

    se_height, se_width = structuring_element.shape
    pad_h = se_height // 2
    pad_w = se_width // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=1)

    output = np.ones_like(image, dtype=np.uint8)
    for i in range(se_height):
        for j in range(se_width):
            if structuring_element[i, j] == 1:
                output = np.minimum(output, padded_image[i:i + image.shape[0], j:j + image.shape[1]])

    return output


def opening(image, structuring_element):

    eroded = erode(image, structuring_element)
    opened = dilate(eroded, structuring_element)
    return opened


def closing(image, structuring_element):

    dilated = dilate(image, structuring_element)
    closed = erode(dilated, structuring_element)
    return closed


def hit_or_miss_single_optimized(image, structuring_element):


    se_foreground = (structuring_element == 1).astype(np.uint8)
    se_background = (structuring_element == 0).astype(np.uint8)

    image = (image > 0).astype(np.uint8)

    pad_h, pad_w = structuring_element.shape[0] // 2, structuring_element.shape[1] // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    padded_complement = np.pad(1 - image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)


    strides = padded_image.strides
    region_shape = (image.shape[0], image.shape[1], structuring_element.shape[0], structuring_element.shape[1])
    image_windows = np.lib.stride_tricks.as_strided(
        padded_image,
        shape=region_shape,
        strides=strides + strides,
    )
    complement_windows = np.lib.stride_tricks.as_strided(
        padded_complement,
        shape=region_shape,
        strides=strides + strides,
    )

    foreground_match = np.all(image_windows * se_foreground == se_foreground, axis=(-1, -2))
    background_match = np.all(complement_windows * se_background == se_background, axis=(-1, -2))

    output = (foreground_match & background_match).astype(np.uint8)

    return output


def get_predefined_structuring_elements():

    return {
        1: np.array([
            [0, 0, 0],
            [-1, 1, -1],
            [1, 1, 1]
        ]),
        2: np.array([
            [-1, 0, 0],
            [1, 1, 0],
            [1, 1, -1]
        ]),
        3: np.array([
            [1, -1, 0],
            [1, 1, 0],
            [1, -1, 0]
        ]),
        4: np.array([
            [1, 1, -1],
            [1, 1, 0],
            [-1, 0, 0]
        ]),
        5: np.array([
            [1, 1, 1],
            [-1, 1, -1],
            [0, 0, 0]
        ]),
        6: np.array([
            [-1, 1, 1],
            [0, 1, 1],
            [0, 0, -1]
        ]),
        7: np.array([
            [0, -1, 1],
            [0, 1, 1],
            [0, -1, 1]
        ]),
        8: np.array([
            [0, 0, -1],
            [0, 1, 1],
            [-1, 1, 1]
        ]),
        9: np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]),
        10: np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]),
        11: np.array([
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]),
        12: np.array([
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 0]
        ]),
        13: np.array([
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 0]
        ]),
    }


def hit_or_miss_repeated(image, structuring_elements, max_elements=8):

    prev_image = np.zeros_like(image)
    current_image = image.copy()
    iteration = 0

    while not np.array_equal(prev_image, current_image):
        prev_image = current_image.copy()
        iteration += 1
        for idx, (se_id, structuring_element) in enumerate(structuring_elements.items()):
            if idx >= max_elements:
                break

            print(f"Applying HMT with structuring element {se_id} (Iteration {iteration}):")
            print(structuring_element)

            hmt_result = hit_or_miss_single_optimized(current_image, structuring_element)

            current_image -= hmt_result
            current_image = np.clip(current_image, 0, 1)

        diff = np.sum(np.abs(prev_image - current_image))
        print(f"Iteration {iteration}: Difference = {diff}")
        if diff == 21:
            break

    return current_image


def region_growing_rgb(image, seed, threshold):
    """
    Perform region growing segmentation on an RGB image.
    Args:
        image: RGB image as a numpy array of shape (rows, cols, 3).
        seed: Tuple (x, y) specifying the starting seed point.
        threshold: Intensity difference threshold for growing the region.
    Returns:
        Binary mask of the segmented region (same dimensions as the input image, 2D).
    """
    rows, cols, _ = image.shape
    segmented = np.zeros((rows, cols), dtype=bool)
    queue = deque([seed])
    seed_color = image[seed]  # The RGB color at the seed point

    while queue:
        x, y = queue.popleft()

        if segmented[x, y]:
            continue

        segmented[x, y] = True

        # Check 8-neighbor pixels
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not segmented[nx, ny]:
                neighbor_color = image[nx, ny]
                # Calculate Euclidean distance in RGB space
                color_diff = np.linalg.norm(neighbor_color - seed_color)
                if color_diff <= threshold:
                    queue.append((nx, ny))

    return segmented.astype(np.uint8) * 255  # Convert boolean mask to binary image

###########################
# TASK 4
###########################

def apply_per_channel(img, transform_fn):
    if img.ndim == 2:
        return transform_fn(img)
    elif img.ndim == 3:
        channels = []
        for c in range(img.shape[2]):
            single_channel = img[..., c]  # shape [H, W]
            transformed = transform_fn(single_channel)
            channels.append(transformed)
        return np.stack(channels, axis=2)
    else:
        raise ValueError("Image must be 2D or 3D.")

def dft2d_slow_single(img2d):

    H, W = img2d.shape
    img2d = img2d.astype(complex)
    dft = np.zeros((H, W), dtype=complex)
    for u in range(H):
        for v in range(W):
            s = 0+0j
            for x in range(H):
                for y in range(W):
                    angle = -2j * np.pi * ((u*x)/H + (v*y)/W)
                    s += img2d[x, y] * np.exp(angle)
            dft[u, v] = s
    return dft

def idft2d_slow_single(freq2d):

    H, W = freq2d.shape
    out = np.zeros((H, W), dtype=complex)
    for x in range(H):
        for y in range(W):
            s = 0+0j
            for u in range(H):
                for v in range(W):
                    angle = 2j * np.pi * ((u*x)/H + (v*y)/W)
                    s += freq2d[u, v] * np.exp(angle)
            # scale by 1/(H*W)
            out[x, y] = s / (H * W)
    return out

def dft2d_slow(img):
    return apply_per_channel(img, dft2d_slow_single)

def idft2d_slow(freq):
    return apply_per_channel(freq, idft2d_slow_single)

def fft1d_spatial(x):

    x = x.astype(complex)
    N = len(x)
    if N <= 1:
        return x
    even = fft1d_spatial(x[0::2])
    odd  = fft1d_spatial(x[1::2])
    X = np.zeros(N, dtype=complex)
    half = N // 2
    for k in range(half):
        twiddle = np.exp(-2j * np.pi * k / N) * odd[k]
        X[k] = even[k] + twiddle
        X[k + half] = even[k] - twiddle
    return X

def ifft1d_spatial(X):

    conjX = np.conjugate(X)
    Y = fft1d_spatial(conjX)
    y = np.conjugate(Y)
    return y / len(X)

def fft2d_spatial_single(img2d):

    H, W = img2d.shape
    img2d = img2d.astype(complex)
    # 1) FFT each row
    row_fft = np.zeros((H, W), dtype=complex)
    for r in range(H):
        row_fft[r, :] = fft1d_spatial(img2d[r, :])
    # 2) FFT each column
    out = np.zeros((H, W), dtype=complex)
    for c in range(W):
        out[:, c] = fft1d_spatial(row_fft[:, c])
    return out

def ifft2d_spatial_single(freq2d):

    H, W = freq2d.shape
    # 1) IFFT each column
    col_ifft = np.zeros((H, W), dtype=complex)
    for c in range(W):
        col_ifft[:, c] = ifft1d_spatial(freq2d[:, c])
    # 2) IFFT each row
    out = np.zeros((H, W), dtype=complex)
    for r in range(H):
        out[r, :] = ifft1d_spatial(col_ifft[r, :])
    return out

def fft2d_spatial(img):
    return apply_per_channel(img, fft2d_spatial_single)

def ifft2d_spatial(freq):
    return apply_per_channel(freq, ifft2d_spatial_single)

def spectrum_single_channel(freq2d, use_log=True):

    magnitude = np.abs(freq2d)

    if use_log:
        magnitude = np.log1p(magnitude)
    magnitude -= magnitude.min()
    max_val = magnitude.max()
    if max_val > 0:
        magnitude /= max_val
    magnitude_8u = (magnitude * 255).astype(np.uint8)

    return magnitude_8u

def visualize_spectrum(freq, use_log=True):

    return apply_per_channel(freq, lambda f2d: spectrum_single_channel(f2d, use_log))

def frequency_mesh(N, M):

    u = np.arange(N) - (N // 2)   # shape [N,]
    v = np.arange(M) - (M // 2)   # shape [M,]

    U, V = np.meshgrid(u, v, indexing='ij')
    return U, V

def apply_filter(freq, mask):

    if freq.ndim == 2:
        return freq * mask
    elif freq.ndim == 3:
        freq_filtered = np.zeros_like(freq)
        for c in range(freq.shape[2]):
            freq_filtered[..., c] = freq[..., c] * mask
        return freq_filtered
    else:
        raise ValueError("freq must be 2D or 3D.")

def create_lowpass_mask(N, M, cutoff):

    U, V = frequency_mesh(N, M)
    R = np.sqrt(U**2 + V**2)  # distance from center
    mask = np.zeros((N, M), dtype=float)
    mask[R <= cutoff] = 1.0
    return mask

def create_highpass_mask(N, M, cutoff):
    U, V = frequency_mesh(N, M)
    R = np.sqrt(U**2 + V**2)
    mask = np.ones((N, M), dtype=float)
    mask[R <= cutoff] = 0.0
    return mask

def lowpass_filter_image(img, cutoff):

    F = fft2d_spatial(img)
    F_shift = np.fft.fftshift(F, axes=(0,1))

    N, M = F.shape[:2]
    mask = create_lowpass_mask(N, M, cutoff)

    F_filt_shift = apply_filter(F_shift, mask)

    F_filt = np.fft.ifftshift(F_filt_shift, axes=(0,1))

    img_filtered_complex = ifft2d_spatial(F_filt)
    img_filtered = np.real(img_filtered_complex)
    img_filtered = np.clip(img_filtered, 0, 255)
    return img_filtered.astype(np.uint8)

def highpass_filter_image(img, cutoff):

    F = fft2d_spatial(img)
    F_shift = np.fft.fftshift(F, axes=(0,1))

    N, M = F.shape[:2]
    mask = create_highpass_mask(N, M, cutoff)

    F_filt_shift = apply_filter(F_shift, mask)

    F_filt = np.fft.ifftshift(F_filt_shift, axes=(0,1))

    img_filtered_complex = ifft2d_spatial(F_filt)
    img_filtered = np.real(img_filtered_complex)
    img_filtered = np.clip(img_filtered, 0, 255)
    return img_filtered.astype(np.uint8)

def create_bandpass_mask(N, M, lowRadius, highRadius):
    U, V = frequency_mesh(N, M)
    R = np.sqrt(U**2 + V**2)
    mask = np.zeros((N, M), dtype=float)
    mask[(R >= lowRadius) & (R <= highRadius)] = 1.0
    return mask

def create_bandcut_mask(N, M, lowRadius, highRadius):
    U, V = frequency_mesh(N, M)
    R = np.sqrt(U**2 + V**2)
    mask = np.ones((N, M), dtype=float)
    mask[(R >= lowRadius) & (R <= highRadius)] = 0.0
    return mask

def bandpass_filter_image(img, lowRadius, highRadius):
    F = fft2d_spatial(img)
    F_shift = np.fft.fftshift(F, axes=(0,1))

    N, M = F.shape[:2]
    mask = create_bandpass_mask(N, M, lowRadius, highRadius)

    F_filt_shift = apply_filter(F_shift, mask)

    F_filt = np.fft.ifftshift(F_filt_shift, axes=(0,1))

    img_filtered_complex = ifft2d_spatial(F_filt)
    img_filtered = np.real(img_filtered_complex)
    img_filtered = np.clip(img_filtered, 0, 255)
    return img_filtered.astype(np.uint8)

def bandcut_filter_image(img, lowRadius, highRadius):
    F = fft2d_spatial(img)
    F_shift = np.fft.fftshift(F, axes=(0,1))

    N, M = F.shape[:2]
    mask = create_bandcut_mask(N, M, lowRadius, highRadius)

    F_filt_shift = apply_filter(F_shift, mask)

    F_filt = np.fft.ifftshift(F_filt_shift, axes=(0,1))

    img_filtered_complex = ifft2d_spatial(F_filt)
    img_filtered = np.real(img_filtered_complex)
    img_filtered = np.clip(img_filtered, 0, 255)
    return img_filtered.astype(np.uint8)

def create_directional_highpass_mask(N, M, cutoff, angle_center, angle_width):

    u = np.arange(N) - (N//2)
    v = np.arange(M) - (M//2)
    U, V = np.meshgrid(u, v, indexing='ij')

    R = np.sqrt(U**2 + V**2)
    angles = np.arctan2(V, U)

    radius_mask = (R > cutoff)


    angle_diff = np.abs( (angles - angle_center + np.pi) % (2*np.pi) - np.pi )
    angle_mask = (angle_diff <= angle_width)

    mask = np.zeros((N, M), dtype=float)
    mask[radius_mask & angle_mask] = 1.0

    return mask

def load_mask_image(mask_filename):

    mask_im = Image.open(mask_filename).convert('L')
    mask_arr = np.array(mask_im)

    mask_bin = (mask_arr > 128).astype(float)
    return mask_bin


def get_directional_highpass_mask(
        N, M,
        cutoff=None,
        angle_center=None,
        angle_width=None,
        mask_filename=None
):
    if mask_filename is not None:
        mask = load_mask_image(mask_filename)

        if mask.shape != (N, M):
            raise ValueError(f"Mask file size {mask.shape} does not match expected {N}x{M}")
        return mask

    else:
        if cutoff is None or angle_center is None or angle_width is None:
            raise ValueError("To generate a mask, you must provide cutoff, angle_center, and angle_width.")
        mask = create_directional_highpass_mask(N, M, cutoff, angle_center, angle_width)
        return mask

def directional_highpass_filter_image (image, cutoff, angle_center, angle_width):


    F = fft2d_spatial(image)
    F_shift = np.fft.fftshift(F, axes=(0, 1))

    N, M = F.shape[:2]
    # Option A: Load mask from file
    #mask = get_directional_highpass_mask(N, M, mask_filename="F5mask1.bmp")

    # Option B: Generate mask from user params
    mask = get_directional_highpass_mask(N, M, cutoff, angle_center, angle_width)

    F_filt_shift = apply_filter(F_shift, mask)
    F_filt = np.fft.ifftshift(F_filt_shift, axes=(0, 1))
    img_filtered_complex = ifft2d_spatial(F_filt)
    img_filtered = np.real(img_filtered_complex)
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)

    return img_filtered


def create_phase_mask(N, M, k, l):
    n_idx = np.arange(N).reshape(N, 1)  # shape [N,1]
    m_idx = np.arange(M).reshape(1, M)  # shape [1,M]

    phase = (
            -2 * np.pi * k * n_idx / N
            - 2 * np.pi * l * m_idx / M
            + (k + l) * np.pi
    )
    mask = np.exp(1j * phase)
    return mask

def phase_modifying_filter(image, k, l):

    F = fft2d_spatial(image)
    F_shift = np.fft.fftshift(F, axes=(0, 1))

    N, M = F.shape[:2]

    mask = create_phase_mask(N, M, k, l)

    F_filt_shift = apply_filter(F_shift, mask)
    F_filt = np.fft.ifftshift(F_filt_shift, axes=(0, 1))
    img_filtered_complex = ifft2d_spatial(F_filt)
    img_filtered = np.real(img_filtered_complex)
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)

    return img_filtered

###########################
# HERE THE MAIN PART STARTS
###########################
#im = Image.open("lena.bmp")
im = Image.open("result.bmp")
im2 = Image.open("result.bmp")

arr = np.array(im.getdata())
arr2 = np.array(im2.getdata())

mask = np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]])


structuring_elements = get_predefined_structuring_elements()

if arr.ndim == 1:
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
    elif sys.argv[1] == '--hmtm5':
        arr = hit_or_miss_repeated((arr > 0).astype(np.uint8), structuring_elements, max_elements=8)
        print("Hit-or-Miss Transformation applied with 8 structuring elements.")
    elif sys.argv[1] == '--dft-slow':
        freq = dft2d_slow(arr)
        print("Slow DFT done. Now 'freq' is complex.")
        recon = idft2d_slow(freq)
        print("Slow IDFT done.")
        arr = np.real(recon)

    elif sys.argv[1] == '--fft-time':
        freq = fft2d_spatial(arr)
        print("Fast FFT done (decimation in time).")
        recon = ifft2d_spatial(freq)
        arr = np.real(recon)
    elif sys.argv[1] == '--dft-spectrum':

        freq = dft2d_slow(arr)
        print("Slow DFT done.")
        freq_shifted = np.fft.fftshift(freq, axes=(0, 1))
        spectrum = visualize_spectrum(freq_shifted)
        if numColorChannels == 1:
            Image.fromarray(spectrum, mode='L').save("spectrum.bmp")
            print("Spectrum saved as grayscale image.")
        else:
            Image.fromarray(spectrum, mode='RGB').save("spectrum.bmp")
            print("Spectrum saved as RGB image.")
    elif sys.argv[1] == '--fft-spectrum':
        freq = fft2d_spatial(arr)
        print("Fast FFT done (decimation in time).")
        freq_shifted = np.fft.fftshift(freq, axes=(0, 1))
        spectrum = visualize_spectrum(freq_shifted,use_log=True)
        if numColorChannels == 1:
            Image.fromarray(spectrum, mode='L').save("spectrum.bmp")
            print("Spectrum saved as grayscale image.")
        else:
            Image.fromarray(spectrum, mode='RGB').save("spectrum.bmp")
            print("Spectrum saved as RGB image.")
    elif sys.argv[1] == '--directionalHighpass':
        arr = directional_highpass_filter_image(arr, cutoff=0, angle_center=0, angle_width=0)
        print("Directional Highpass Filter applied.")
    else:
        print("Too few command line parameters given.\n")
        sys.exit()
elif len(sys.argv) == 4:
    command = sys.argv[1]

    if command == '--hpower':
        g_min = float(sys.argv[2])
        g_max = float(sys.argv[3])
        if numColorChannels == 1:
            arr = enhance_image_based_on_histogram(arr, g_min, g_max)
        else:
            print("Error: --hpower only works on grayscale images.")
            sys.exit()

    elif command == '--hpowerrgb':
        g_min = float(sys.argv[2])
        g_max = float(sys.argv[3])
        if numColorChannels == 3:
            arr = enhance_image_rgb(arr, g_min, g_max)

    elif command == '--bandpass':
        lowRadius = float(sys.argv[2])
        highRadius = float(sys.argv[3])
        arr = bandpass_filter_image(arr, lowRadius, highRadius)
        print("Bandpass Filter applied.")

    elif command == '--bandcut':
        lowRadius = float(sys.argv[2])
        highRadius = float(sys.argv[3])
        arr = bandcut_filter_image(arr, lowRadius, highRadius)
        print("Bandcut Filter applied.")
    elif command == '--phase':
        k = int(sys.argv[2])
        l = int(sys.argv[3])
        arr = phase_modifying_filter(arr, k, l)
        print("Phase Modifying Filter applied.")
    else:
        print("Unknown command or incorrect parameters.")
        sys.exit()
elif len(sys.argv) == 5:  # For commands requiring 4 parameters
        command = sys.argv[1]
        if command == '--regiongrow':
            if numColorChannels == 3:  # Ensure the image is RGB
                seed_x = int(sys.argv[2])
                seed_y = int(sys.argv[3])
                threshold = int(sys.argv[4])

                # Apply region growing
                arr = region_growing_rgb(arr, (seed_x, seed_y), threshold)

                print(f"Region Growing applied with seed=({seed_x}, {seed_y}) and threshold={threshold}.")
            else:
                print("Error: --regiongrow only works on RGB images.")
                sys.exit()
        elif command == '--directionalHighpass':
            arr = directional_highpass_filter_image(arr, float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
            print("Directional Highpass Filter applied.")
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
    elif command == '--hmt':
        param = int(param)
        if param not in structuring_elements:
            print("Invalid structuring element ID. Choose from:", list(structuring_elements.keys()))
            sys.exit()

        selected_se = structuring_elements[param]
        print(f"Applying HMT with structuring element {param}:")
        print(selected_se)


        arr = hit_or_miss_single_optimized((arr > 0).astype(np.uint8), selected_se)
    elif sys.argv[1] == '--dilate':
        arr = (arr > 0).astype(np.uint8)
        selected_se = structuring_elements[int(param)]
        arr = dilate(arr, selected_se)
        print("Dilation applied.")
    elif sys.argv[1] == '--erode':
        arr = (arr > 0).astype(np.uint8)
        selected_se = structuring_elements[int(param)]
        arr = erode(arr, selected_se)
        print("Erosion applied.")
    elif sys.argv[1] == '--opening':
        arr = (arr > 0).astype(np.uint8)
        selected_se = structuring_elements[int(param)]
        arr = opening(arr, selected_se)
        print("Opening applied.")
    elif sys.argv[1] == '--closing':
        arr = (arr > 0).astype(np.uint8)
        selected_se = structuring_elements[int(param)]
        arr = closing(arr, selected_se)
        print("Closing applied.")
    elif sys.argv[1] == '--lowpass':
        arr = lowpass_filter_image(arr, int(param))
        print("Lowpass Filter applied.")
    elif sys.argv[1] == '--highpass':
        arr = highpass_filter_image(arr, int(param))
        print("Highpass Filter applied.")
    else:
        print("Unknown command: " + command)
        sys.exit()

if np.array_equal(np.unique(arr), [0, 1]):
    newIm = Image.fromarray(arr.astype(np.uint8) * 255).convert("1")
    newIm.save("result1.bmp")
    print("Saved as 1-bit binary image (mode '1').")
else:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    newIm = Image.fromarray(arr.astype(np.uint8))
    newIm.save("result.bmp")
    print("Saved as 8-bit image.")
