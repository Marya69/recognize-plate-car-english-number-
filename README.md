import cv2
import pytesseract
import numpy as np  # Import NumPy

# If Tesseract is not in your PATH, add it here
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

def recognize_license_plate(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Pre-process the image (convert to grayscale and apply Gaussian blur)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection
    edged = cv2.Canny(blurred, 50, 200)

    # Find contours on edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming we are interested in the largest contour (license plate)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    license_plate_contour = None
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:  # A rectangle
            license_plate_contour = approx
            break

    if license_plate_contour is None:
        print("License plate contour not found")
        return

    # Mask for the license plate
    mask = np.zeros(gray.shape, dtype="uint8")  # Use np.zeros to create the mask
    cv2.drawContours(mask, [license_plate_contour], -1, 255, -1)

    # Extract the license plate
    plate_image = cv2.bitwise_and(image, image, mask=mask)

    # Crop the image to the bounding rectangle of the license plate
    (x, y, w, h) = cv2.boundingRect(license_plate_contour)
    license_plate = plate_image[y:y + h, x:x + w]

    # Convert to grayscale and apply thresholding
    license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    _, thresholded_plate = cv2.threshold(license_plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use Tesseract to extract text
    config = '--psm 8 -c tessedit_char_whitelist=0123456789أبجدية'
    text = pytesseract.image_to_string(thresholded_plate, config=config)

    # Print the recognized text
    print("Recognized License Plate Number:", text)

# Example usage
image_path = r"C:\Users\ZETTA-\Downloads\FzEajuyq.jpg"  # Change this to your image path
recognize_license_plate(image_path)
