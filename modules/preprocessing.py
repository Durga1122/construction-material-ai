import cv2
import os

def preprocess_image(input_path, output_path):
    img = cv2.imread(input_path)

    # Resize
    img = cv2.resize(img, (800, 600))

    # Noise reduction
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Brightness/contrast correction
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    corrected = cv2.merge((l, a, b))
    corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

    # Sharpening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sharpened = cv2.filter2D(corrected, -1, kernel)

    cv2.imwrite(output_path, sharpened)
    return output_path