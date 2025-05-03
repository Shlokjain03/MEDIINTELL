import pytesseract
import cv2
import os
import platform
import numpy as np

def set_tesseract_path():
    if platform.system() == 'Windows':
        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_path):
            pytesseract.pytesseract.tesseract_cmd = default_path
        else:
            print("[⚠] Tesseract not found! Install from: https://github.com/UB-Mannheim/tesseract")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # sharpen
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    # blur to reduce noise
    blur = cv2.medianBlur(sharp, 3)
    # thresholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_text_local(image_path):
    set_tesseract_path()
    processed = preprocess_image(image_path)
    if processed is None:
        return ""
    text = pytesseract.image_to_string(processed)
    print("[🔍] OCR Output:", text)
    lines = text.split('\n')
    clean_lines = [line.strip() for line in lines if line.strip()]
    return clean_lines[0] if clean_lines else ""
