import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class CentroidTracker:
    """
    A tracker that uses centroids of bounding boxes to track objects across frames.
    """
    def __init__(self, max_disappeared=50, max_distance=50):
        """
        Initialize the CentroidTracker.
        
        Args:
            max_disappeared (int): Maximum number of consecutive frames an object can
                be missing before it's deregistered
            max_distance (int): Maximum distance (in pixels) between centroids for
                them to be considered the same object
        """
        # Initialize the next unique object ID and two ordered dictionaries to
        # keep track of mapping a given object ID to its centroid
        self.next_object_id = 0
        self.objects = OrderedDict()  # format: {object_id: (centroid_x, centroid_y)}
        self.bboxes = OrderedDict()   # format: {object_id: (x, y, w, h)}
        self.speeds = OrderedDict()   # format: {object_id: speed}
        self.disappeared = OrderedDict()  # format: {object_id: disappear_count}
        
        # Number of consecutive frames an object is allowed to be
        # missing until we deregister it
        self.max_disappeared = max_disappeared
        
        # Maximum distance between centroids to associate an object
        self.max_distance = max_distance
    
    def register(self, centroid, bbox):
        """
        Register a new object with the next available ID.
        
        Args:
            centroid (tuple): (x, y) coordinates of the centroid
            bbox (tuple): (x, y, w, h) coordinates of the bounding box
        """
        # Register the new object
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.speeds[self.next_object_id] = None
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """
        Deregister an object by deleting its ID from all tracking dictionaries.
        
        Args:
            object_id (int): The ID of the object to deregister
        """
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.speeds[object_id]
        del self.disappeared[object_id]
    
    def update(self, bboxes):
        """
        Update the tracker with new bounding boxes.
        
        Args:
            bboxes (list): List of (x, y, w, h) bounding boxes
            
        Returns:
            list: List of tuples in format (object_id, centroid, bbox, speed)
        """
        # Check if the list of input bounding boxes is empty
        if len(bboxes) == 0:
            # Loop over any existing tracked objects and mark them as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # If we have reached a maximum number of consecutive frames
                # where a given object has been marked as missing, deregister it
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Return early as there are no centroids or tracking information to update
            return self.get_tracked_objects()
        
        # Calculate centroids of the new bounding boxes
        input_centroids = np.zeros((len(bboxes), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(bboxes):
            # Calculate the centroid
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)
        
        # If we are currently not tracking any objects, register all of them
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], bboxes[i])
        
        # Otherwise, we need to match the input centroids to existing object centroids
        else:
            # Get the set of object IDs and corresponding centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute the distance between each pair of existing centroids and input centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Find the smallest value in each row and the corresponding column indices
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # To determine if we need to update, register, or deregister
            # an object, we need to keep track of which rows and column
            # indexes we have already examined
            used_rows = set()
            used_cols = set()
            
            # Loop over the combination of the row and column indices
            for (row, col) in zip(rows, cols):
                # If this row or column has already been used, ignore it
                if row in used_rows or col in used_cols:
                    continue
                
                # If the distance is greater than the maximum distance,
                # don't associate the two centroids to the same object
                if D[row, col] > self.max_distance:
                    continue
                
                # Otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared counter
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = bboxes[col]
                self.disappeared[object_id] = 0
                
                # Mark that we've used both row and column
                used_rows.add(row)
                used_cols.add(col)
            
            # Compute both the row and column indices that we haven't used yet
            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)
            
            # If the number of object centroids is greater than or equal
            # to the number of input centroids, we need to check if some
            # of these objects have potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # Loop over the unused row indices
                for row in unused_rows:
                    # Grab the object ID for the corresponding row index and
                    # increment the disappeared counter
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    # Check if the number of consecutive frames the object
                    # has been marked as "disappeared" exceeds our threshold
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # Otherwise, if the number of input centroids is greater
            # than the number of existing object centroids, we need to
            # register each new input centroid as a trackable object
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], bboxes[col])
        
        # Return the set of trackable objects
        return self.get_tracked_objects()
    
    def get_tracked_objects(self):
        """
        Return the current tracking information for all objects.
        
        Returns:
            list: List of tuples in format (object_id, centroid, bbox, speed)
        """
        tracked_objects = []
        for object_id in self.objects.keys():
            tracked_objects.append(
                (object_id, self.objects[object_id], self.bboxes[object_id], self.speeds[object_id])
            )
        return tracked_objects
    
    def set_speed(self, object_id, speed):
        """
        Set the speed for a tracked object.
        
        Args:
            object_id (int): The ID of the object
            speed (float): The speed in km/h
        """
        if object_id in self.speeds:
            self.speeds[object_id] = speed 