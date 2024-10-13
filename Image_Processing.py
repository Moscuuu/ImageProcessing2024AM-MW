
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
    arr[arr > 255] = 255;
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


def doArithmeticMeanFilter(arr, filter_size):
    print("Applying Arithmetic Mean Filter with size:", filter_size)
    filter_size = int(filter_size)
    pad_size = filter_size // 2

    # Pad the array to handle the edges
    if arr.ndim == 2:  # grayscale image
        padded_arr = np.pad(arr, pad_size, mode='constant', constant_values=0)
    else:  # color image
        padded_arr = np.pad(arr, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant',
                            constant_values=0)

    new_arr = np.zeros_like(arr)

    # Apply the arithmetic mean filter
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr.ndim == 2:  # grayscale image
                new_arr[i, j] = np.mean(padded_arr[i:i + filter_size, j:j + filter_size])
            else:  # color image
                for c in range(arr.shape[2]):  # loop through color channels
                    new_arr[i, j, c] = np.mean(padded_arr[i:i + filter_size, j:j + filter_size, c])

    return new_arr

###########################
# HERE THE MAIN PART STARTS
###########################
#im = Image.open("lena.bmp")
im = Image.open("lenac.bmp")

arr = np.array(im.getdata())
if arr.ndim == 1: #grayscale
    numColorChannels = 1
    arr = arr.reshape(im.size[1], im.size[0])
else:
    numColorChannels = arr.shape[1]
    arr = arr.reshape(im.size[1], im.size[0], numColorChannels)

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
    elif command == '--arithmeticMeanFilter':
        arr = doArithmeticMeanFilter(arr, param)  # param is the filter size (e.g., 3, 5, etc.)
    else:
        print("Unknown command: " + command)
        sys.exit()

newIm = Image.fromarray(arr.astype(np.uint8))
newIm.save("result.bmp")