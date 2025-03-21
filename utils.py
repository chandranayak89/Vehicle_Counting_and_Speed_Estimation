import cv2
import numpy as np
import imutils
import time
import csv
import os
from datetime import datetime


def resize_frame(frame, width=None):
    """
    Resize a frame to a specified width while maintaining aspect ratio.
    """
    if width is None:
        return frame
    return imutils.resize(frame, width=width)


def apply_preprocessing(frame, blur_size=(5, 5)):
    """
    Apply preprocessing to a frame:
    1. Convert to grayscale
    2. Apply Gaussian blur for noise reduction
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_size, 0)
    return blurred


def draw_roi_lines(frame, line1_position, line2_position):
    """
    Draw lines on the frame to visualize regions of interest for speed calculation.
    """
    h, w = frame.shape[:2]
    # Draw the first line
    cv2.line(frame, (0, line1_position), (w, line1_position), (0, 0, 255), 2)
    # Draw the second line
    cv2.line(frame, (0, line2_position), (w, line2_position), (0, 0, 255), 2)
    return frame


def draw_bounding_boxes(frame, detections, colors=None):
    """
    Draw bounding boxes around detected vehicles.
    """
    if colors is None:
        colors = [(0, 255, 0)]  # Default color is green
    
    for (i, (object_id, centroid, bbox, speed)) in enumerate(detections):
        # Draw the bounding box
        x, y, w, h = bbox
        color = colors[i % len(colors)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw the centroid
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
        
        # Put ID and speed text
        text = f"ID: {object_id}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if speed is not None:
            speed_text = f"{speed:.1f} km/h"
            cv2.putText(frame, speed_text, (centroid[0] - 10, centroid[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame


def draw_vehicle_count(frame, count):
    """
    Display vehicle count on the frame.
    """
    text = f"Vehicle Count: {count}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 0, 255), 2)
    return frame


def log_speed_violation(vehicle_id, speed, speed_limit, output_file="speed_violations.csv"):
    """
    Log speed violations to a CSV file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'vehicle_id', 'speed', 'speed_limit']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'timestamp': timestamp,
            'vehicle_id': vehicle_id,
            'speed': f"{speed:.1f}",
            'speed_limit': speed_limit
        })

        
def calculate_fps(prev_time):
    """
    Calculate and return the frames per second and new time reference.
    """
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return fps, current_time


def draw_fps(frame, fps):
    """
    Display FPS on the frame.
    """
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 0), 2)
    return frame


def calculate_distance_in_pixels(point1, point2):
    """
    Calculate Euclidean distance between two points.
    """
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2) 