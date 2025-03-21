import cv2
import numpy as np
import argparse
import time
import os
from collections import defaultdict
import random

# Import custom modules
from detector import get_detector
from tracker import CentroidTracker
from speed_estimator import get_speed_estimator
import utils


def main():
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input video")
    ap.add_argument("-o", "--output", required=True, help="path to output video")
    ap.add_argument("-d", "--detector", default="contour", 
                    choices=["contour", "haar", "yolo"],
                    help="detection method to use")
    ap.add_argument("-s", "--speed_method", default="fixed_lines", 
                    choices=["fixed_lines", "adaptive"],
                    help="speed estimation method")
    ap.add_argument("--distance", type=float, default=10.0,
                    help="real-world distance in meters between measurement lines")
    ap.add_argument("--show", action="store_true",
                    help="display the processed frames")
    ap.add_argument("--speed_limit", type=float, default=40.0,
                    help="speed limit in km/h for violation detection")
    ap.add_argument("--resolution", type=int, default=720,
                    help="processing resolution width (maintains aspect ratio)")
    args = vars(ap.parse_args())
    
    # Print banner
    print("="*50)
    print("Vehicle Counting & Speed Estimation System")
    print("="*50)
    
    # Check if input file exists
    if not os.path.isfile(args["input"]):
        print(f"Error: Input file '{args['input']}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args["output"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize video capture
    cap = cv2.VideoCapture(args["input"])
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args['input']}'.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    # Calculate scaling factor for visualization (keeping aspect ratio)
    if args["resolution"] > 0:
        scale_factor = args["resolution"] / width
        output_width = args["resolution"]
        output_height = int(height * scale_factor)
    else:
        output_width, output_height = width, height
        scale_factor = 1.0
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args["output"], fourcc, fps, (output_width, output_height))
    
    # Initialize detector, tracker, and speed estimator
    detector = get_detector(args["detector"])
    tracker = CentroidTracker(max_disappeared=50, max_distance=100)
    
    # Calculate line positions based on the frame height
    line1_pos = int(output_height * 0.3)  # 30% from the top
    line2_pos = int(output_height * 0.7)  # 70% from the top
    
    # Initialize speed estimator
    speed_estimator_args = {
        "fps": fps,
        "line1_pos": line1_pos,
        "line2_pos": line2_pos,
        "real_distance_meters": args["distance"]
    }
    if args["speed_method"] == "adaptive":
        speed_estimator_args = {
            "fps": fps,
            "pixels_per_meter": 12.0,  # This should be calibrated per video
            "smoothing_factor": 0.7
        }
    
    speed_estimator = get_speed_estimator(args["speed_method"], **speed_estimator_args)
    
    # Create random colors for visualization
    colors = []
    for i in range(100):
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    
    # Counter for vehicles
    vehicle_count = 0
    counted_ids = set()
    
    # Variables for FPS calculation
    prev_time = time.time()
    fps_counter = 0
    display_fps = 0
    
    # Process the video frame by frame
    frame_count = 0
    
    print("Starting video processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for processing
        frame = utils.resize_frame(frame, width=output_width)
        
        # Update progress every 100 frames
        frame_count += 1
        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100
            print(f"Processing: {progress:.1f}% complete ({frame_count}/{total_frames})")
        
        # Detect vehicles
        bboxes = detector.detect(frame)
        
        # Update tracker
        tracked_objects = tracker.update(bboxes)
        
        # Update speed estimator
        speeds = speed_estimator.update(tracked_objects)
        
        # Update tracker with speed information
        for object_id, speed in speeds.items():
            tracker.set_speed(object_id, speed)
        
        # Count vehicles that have crossed the second line
        for (object_id, _, _, _) in tracked_objects:
            if object_id in speed_estimator.line2_crossings and object_id not in counted_ids:
                vehicle_count += 1
                counted_ids.add(object_id)
        
        # Draw ROI lines for speed estimation
        if args["speed_method"] == "fixed_lines":
            frame = utils.draw_roi_lines(frame, line1_pos, line2_pos)
        
        # Draw bounding boxes, centroids, and speeds
        frame = utils.draw_bounding_boxes(frame, tracked_objects, colors)
        
        # Draw vehicle count
        frame = utils.draw_vehicle_count(frame, vehicle_count)
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 30:  # Update FPS every 30 frames
            current_time = time.time()
            display_fps = fps_counter / (current_time - prev_time)
            fps_counter = 0
            prev_time = current_time
        
        # Draw FPS
        frame = utils.draw_fps(frame, display_fps)
        
        # Check for speed violations and log them
        for (object_id, _, _, speed) in tracked_objects:
            if speed is not None and speed > args["speed_limit"]:
                # Log violation
                utils.log_speed_violation(
                    object_id, speed, args["speed_limit"], output_file="speed_violations.csv"
                )
        
        # Write the frame to the output video
        out.write(frame)
        
        # Display the frame if show flag is set
        if args["show"]:
            cv2.imshow("Vehicle Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    # Cleanup
    cap.release()
    out.release()
    if args["show"]:
        cv2.destroyAllWindows()
    
    print("="*50)
    print(f"Processing complete! Output saved to {args['output']}")
    print(f"Total vehicles counted: {vehicle_count}")
    print("="*50)


if __name__ == "__main__":
    main() 