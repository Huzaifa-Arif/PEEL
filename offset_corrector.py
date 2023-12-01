import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Set the QT_QPA_PLATFORM environment variable
os.environ['QT_QPA_PLATFORM'] = 'off'

# Load the original and reconstructed images
original_image = cv2.imread('orig_3.png')
reconstructed_image = cv2.imread('recons_3.png')

# Check if the images were loaded correctly
if original_image is None or reconstructed_image is None:
    print("Error: unable to load the image.")
else:
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    reconstructed_gray = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2GRAY)

    # Match the template (find the reconstructed image in the original)
    result = cv2.matchTemplate(original_gray, reconstructed_gray, cv2.TM_CCOEFF_NORMED)

    # Find the maximum position from the match template result
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # The top-left corner of the matched area will give us the offset
    offset_x, offset_y = max_loc

    # Create the translation matrix
    translation_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])

    # Perform the translation
    translated_image = cv2.warpAffine(reconstructed_image, translation_matrix,
                                      (reconstructed_image.shape[1], reconstructed_image.shape[0]))

    # Concatenate images horizontally for side-by-side comparison
    comparison_image = np.concatenate((original_image, translated_image), axis=1)

    # Convert BGR to RGB for matplotlib display
    comparison_image = cv2.cvtColor(comparison_image, cv2.COLOR_BGR2RGB)

    # Show the concatenated image using matplotlib
    plt.imshow(comparison_image)
    plt.title('Original vs Reconstructed')
    plt.axis('off')  # Hide axes
    plt.show()

