import cv2
import numpy as np
import torch
import os


class ContourDetector:
    """
    Detector that uses background subtraction and contour detection to identify vehicles.
    """
    def __init__(self, min_area=500, detect_shadows=True, threshold=200):
        """
        Initialize the ContourDetector.
        
        Args:
            min_area (int): Minimum area (in pixels) for a contour to be considered a vehicle
            detect_shadows (bool): Whether to detect shadows in background subtraction
            threshold (int): Threshold for binary image conversion
        """
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=16, detectShadows=detect_shadows
        )
        self.min_area = min_area
        self.threshold = threshold
    
    def detect(self, frame):
        """
        Detect vehicles in a frame using background subtraction and contour detection.
        
        Args:
            frame: The input frame (BGR format)
            
        Returns:
            list: A list of bounding boxes in format [x, y, w, h]
        """
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Threshold the mask to get binary image
        _, binary = cv2.threshold(fg_mask, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and create bounding boxes
        bounding_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append([x, y, w, h])
        
        return bounding_boxes


class HaarCascadeDetector:
    """
    Vehicle detector using Haar Cascades.
    """
    def __init__(self, cascade_path="haarcascade_car.xml"):
        """
        Initialize the Haar Cascade detector.
        
        Args:
            cascade_path (str): Path to the Haar Cascade XML file
        """
        # Check if the cascade file exists
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")
            
        self.car_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect(self, frame):
        """
        Detect vehicles in a frame using Haar Cascades.
        
        Args:
            frame: The input frame (BGR format)
            
        Returns:
            list: A list of bounding boxes in format [x, y, w, h]
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect cars
        cars = self.car_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        bounding_boxes = []
        for (x, y, w, h) in cars:
            bounding_boxes.append([x, y, w, h])
            
        return bounding_boxes


class YOLODetector:
    """
    Vehicle detector using YOLOv5/YOLOv8 deep learning model.
    """
    def __init__(self, model_path=None, confidence=0.5, device='cpu'):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path (str): Path to the YOLO model (if None, will use pretrained model)
            confidence (float): Confidence threshold for detections
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        self.confidence = confidence
        self.device = device
        
        # Load YOLOv5 model using torch hub
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            # Set device and confidence threshold
            self.model.to(device)
            self.model.conf = confidence
            # Filter classes to only include vehicles (car, bus, truck, motorcycle)
            self.model.classes = [2, 3, 5, 7]  # COCO dataset indices for vehicle classes
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")
            print("Falling back to OpenCV DNN YOLO implementation")
            self._load_opencv_yolo(model_path)
            
    def _load_opencv_yolo(self, model_path):
        """Load YOLO model using OpenCV DNN when torch is not available."""
        # This is a fallback method using OpenCV DNN when torch hub is not available
        self.use_torch = False
        # Path to YOLO weights and configuration
        weights_path = 'yolov3.weights'
        config_path = 'yolov3.cfg'
        
        if model_path is not None:
            weights_path = model_path
            
        # Load YOLO network
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # COCO class names
        self.classes = ["car", "bus", "truck", "motorcycle"]
    
    def detect(self, frame):
        """
        Detect vehicles in a frame using YOLO.
        
        Args:
            frame: The input frame (BGR format)
            
        Returns:
            list: A list of bounding boxes in format [x, y, w, h]
        """
        try:
            # Run inference
            results = self.model(frame)
            
            # Extract bounding boxes
            bounding_boxes = []
            for detection in results.xyxy[0]:  # Get detections for first image
                x1, y1, x2, y2, conf, cls = detection
                if conf >= self.confidence:
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)
                    bounding_boxes.append([x, y, w, h])
            
            return bounding_boxes
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
            return []


def get_detector(method='contour'):
    """
    Factory function to get the appropriate detector based on method.
    
    Args:
        method (str): Detection method ('contour', 'haar', 'yolo')
        
    Returns:
        Object: An instance of the detector class
    """
    if method.lower() == 'contour':
        return ContourDetector()
    elif method.lower() == 'haar':
        return HaarCascadeDetector()
    elif method.lower() == 'yolo':
        return YOLODetector()
    else:
        print(f"Unknown detection method: {method}. Using contour detector as fallback.")
        return ContourDetector() 