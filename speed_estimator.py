import cv2
import numpy as np
import time
from utils import calculate_distance_in_pixels


class SpeedEstimator:
    """
    Class for estimating the speed of vehicles.
    """
    def __init__(self, fps=25.0, line1_pos=300, line2_pos=500, real_distance_meters=10.0):
        """
        Initialize the SpeedEstimator.
        
        Args:
            fps (float): Frames per second of the video
            line1_pos (int): Y-coordinate of the first line (pixels from top)
            line2_pos (int): Y-coordinate of the second line (pixels from top)
            real_distance_meters (float): Real-world distance between the two lines in meters
        """
        self.fps = fps
        self.line1_pos = line1_pos
        self.line2_pos = line2_pos
        self.real_distance_meters = real_distance_meters
        
        # Dictionary to store the frame numbers when an object crosses each line
        self.line1_crossings = {}  # format: {object_id: frame_number}
        self.line2_crossings = {}  # format: {object_id: frame_number}
        
        # Dictionary to store the speeds of objects
        self.speeds = {}  # format: {object_id: speed_kmh}
        
        # Current frame number
        self.current_frame = 0
    
    def update(self, tracked_objects):
        """
        Update the speed estimator with the current tracked objects.
        
        Args:
            tracked_objects (list): List of tuples in format (object_id, centroid, bbox, speed)
        
        Returns:
            dict: Dictionary mapping object IDs to speeds in km/h
        """
        self.current_frame += 1
        
        # Check if objects have crossed the lines
        for (object_id, centroid, _, _) in tracked_objects:
            cy = centroid[1]  # y-coordinate of the centroid
            
            # Check if the object has crossed line 1 (from top to bottom)
            if object_id not in self.line1_crossings and self.line1_pos - 5 <= cy <= self.line1_pos + 5:
                self.line1_crossings[object_id] = self.current_frame
            
            # Check if the object has crossed line 2 (from top to bottom)
            if object_id not in self.line2_crossings and self.line2_pos - 5 <= cy <= self.line2_pos + 5:
                # Only record crossing if it already crossed line 1
                if object_id in self.line1_crossings:
                    self.line2_crossings[object_id] = self.current_frame
                    # Calculate speed
                    self._calculate_speed(object_id)
        
        # Return the current speeds
        return self.speeds
    
    def _calculate_speed(self, object_id):
        """
        Calculate the speed of an object based on the time it took to travel between the two lines.
        
        Args:
            object_id (int): The ID of the object
        """
        # Get the frame numbers when the object crossed each line
        frame1 = self.line1_crossings[object_id]
        frame2 = self.line2_crossings[object_id]
        
        # Calculate the time difference (in seconds)
        time_diff = (frame2 - frame1) / self.fps
        
        # To avoid division by zero
        if time_diff > 0:
            # Calculate speed in meters per second
            speed_ms = self.real_distance_meters / time_diff
            
            # Convert to kilometers per hour
            speed_kmh = speed_ms * 3.6
            
            # Store the speed
            self.speeds[object_id] = speed_kmh
    
    def get_speed(self, object_id):
        """
        Get the speed of a specific object.
        
        Args:
            object_id (int): The ID of the object
            
        Returns:
            float or None: The speed in km/h, or None if not available
        """
        return self.speeds.get(object_id, None)
    
    def get_all_speeds(self):
        """
        Get all calculated speeds.
        
        Returns:
            dict: Dictionary mapping object IDs to speeds in km/h
        """
        return self.speeds


class AdaptiveSpeedEstimator:
    """
    A more advanced speed estimator that doesn't rely on fixed lines
    but calculates speed based on movement between consecutive frames.
    """
    def __init__(self, fps=25.0, pixels_per_meter=12.0, smoothing_factor=0.7):
        """
        Initialize the AdaptiveSpeedEstimator.
        
        Args:
            fps (float): Frames per second of the video
            pixels_per_meter (float): Conversion factor from pixels to meters
            smoothing_factor (float): Factor for exponential smoothing (0-1)
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.smoothing_factor = smoothing_factor
        
        # Dictionary to store previous positions
        self.prev_positions = {}  # format: {object_id: (x, y)}
        
        # Dictionary to store speeds
        self.speeds = {}  # format: {object_id: speed_kmh}
    
    def update(self, tracked_objects):
        """
        Update the speed estimation for all tracked objects.
        
        Args:
            tracked_objects (list): List of tuples in format (object_id, centroid, bbox, speed)
        
        Returns:
            dict: Dictionary mapping object IDs to speeds in km/h
        """
        current_speeds = {}
        
        for (object_id, centroid, _, _) in tracked_objects:
            # Calculate speed if we have a previous position
            if object_id in self.prev_positions:
                prev_pos = self.prev_positions[object_id]
                
                # Calculate distance in pixels
                distance_pixels = calculate_distance_in_pixels(prev_pos, centroid)
                
                # Convert to meters
                distance_meters = distance_pixels / self.pixels_per_meter
                
                # Calculate time between frames in seconds
                time_seconds = 1.0 / self.fps
                
                # Calculate speed in meters per second
                speed_ms = distance_meters / time_seconds
                
                # Convert to kilometers per hour
                speed_kmh = speed_ms * 3.6
                
                # Apply exponential smoothing if we have a previous speed
                if object_id in self.speeds:
                    speed_kmh = (self.smoothing_factor * speed_kmh + 
                                (1 - self.smoothing_factor) * self.speeds[object_id])
                
                # Store the speed
                current_speeds[object_id] = speed_kmh
            
            # Update the previous position
            self.prev_positions[object_id] = centroid
        
        # Update speeds
        self.speeds.update(current_speeds)
        
        # Return the current speeds
        return self.speeds
    
    def get_speed(self, object_id):
        """
        Get the speed of a specific object.
        
        Args:
            object_id (int): The ID of the object
            
        Returns:
            float or None: The speed in km/h, or None if not available
        """
        return self.speeds.get(object_id, None)
    
    def get_all_speeds(self):
        """
        Get all calculated speeds.
        
        Returns:
            dict: Dictionary mapping object IDs to speeds in km/h
        """
        return self.speeds


def get_speed_estimator(method='fixed_lines', **kwargs):
    """
    Factory function to get the appropriate speed estimator based on method.
    
    Args:
        method (str): Speed estimation method ('fixed_lines', 'adaptive')
        **kwargs: Additional parameters for the specific estimator
        
    Returns:
        Object: An instance of the speed estimator class
    """
    if method.lower() == 'fixed_lines':
        return SpeedEstimator(**kwargs)
    elif method.lower() == 'adaptive':
        return AdaptiveSpeedEstimator(**kwargs)
    else:
        print(f"Unknown speed estimation method: {method}. Using fixed lines as fallback.")
        return SpeedEstimator(**kwargs) 