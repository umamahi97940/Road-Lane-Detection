import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slop_intercept(image, lines):
    left_fit = []
    right_fit = []
    left_line = np.array([293, 720, 473, 432])
    right_line = np.array([])
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def area_of_interest(image):
    height = image.shape[0]
    polygon = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def dis_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 8)
    return line_image

def select_image():
    # Open a file dialog to select an image file
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(title="Select an Image", 
                                            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    return file_path

# Main processing
if __name__ == "__main__":
    # Allow the user to select an image
    image_path = select_image()
    
    if image_path:  # Check if a file was selected
        image = cv2.imread(image_path)
        lane_image = np.copy(image)
        cany = canny(lane_image)
        cropped = area_of_interest(cany)
        lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average_slop_intercept(lane_image, lines)
        lined_image = dis_lines(lane_image, averaged_lines)
        combo_image = cv2.addWeighted(lane_image, 0.8, lined_image, 1, 1)
        
        # Show the result
        cv2.imshow('Result', combo_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image selected.")
