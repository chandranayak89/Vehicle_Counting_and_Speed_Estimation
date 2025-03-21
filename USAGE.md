# Vehicle Counting and Speed Estimation - Usage Guide

This guide provides instructions on how to use the vehicle counting and speed estimation system.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download a sample traffic video (or use your own):

```bash
python download_sample_video.py
```

## Basic Usage

The most basic way to run the program is:

```bash
python main.py --input videos/traffic_sample.mp4 --output output.mp4
```

This will:
- Process the input video (videos/traffic_sample.mp4)
- Detect and count vehicles using the default contour detector
- Estimate speeds using the fixed lines method
- Save the results to output.mp4

## Display the Results in Real-Time

To display the processed video in real-time:

```bash
python main.py --input videos/traffic_sample.mp4 --output output.mp4 --show
```

## Change Detection Method

The program supports three detection methods:

### 1. Contour Detection (Default)

This method uses background subtraction and contour detection:

```bash
python main.py --input videos/traffic_sample.mp4 --output output.mp4 --detector contour
```

### 2. Haar Cascade

This method uses pre-trained Haar cascades for car detection:

```bash
# First, download a car haar cascade XML file
curl -o haarcascade_car.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_car.xml

# Then run with the haar detector
python main.py --input videos/traffic_sample.mp4 --output output.mp4 --detector haar
```

### 3. YOLO (Requires PyTorch)

This method uses a deep learning model for object detection:

```bash
python main.py --input videos/traffic_sample.mp4 --output output.mp4 --detector yolo
```

## Change Speed Estimation Method

### 1. Fixed Lines Method (Default)

This method calculates speed based on the time it takes a vehicle to cross two predefined lines:

```bash
python main.py --input videos/traffic_sample.mp4 --output output.mp4 --speed_method fixed_lines
```

You can specify the real-world distance between the lines:

```bash
python main.py --input videos/traffic_sample.mp4 --output output.mp4 --speed_method fixed_lines --distance 15.0
```

### 2. Adaptive Method

This method calculates speed based on frame-to-frame movement:

```bash
python main.py --input videos/traffic_sample.mp4 --output output.mp4 --speed_method adaptive
```

## Speed Limit and Violation Detection

You can set a speed limit to log violations:

```bash
python main.py --input videos/traffic_sample.mp4 --output output.mp4 --speed_limit 30.0
```

Violations will be logged to `speed_violations.csv`.

## Processing Resolution

You can specify the processing resolution to speed up the detection:

```bash
python main.py --input videos/traffic_sample.mp4 --output output.mp4 --resolution 480
```

## All Options Combined

Here's an example using multiple options:

```bash
python main.py --input videos/traffic_sample.mp4 --output output.mp4 --detector yolo --speed_method adaptive --speed_limit 40.0 --resolution 640 --show
```

## Troubleshooting

1. **Error: No module named 'cv2'**
   - Make sure you've installed all dependencies: `pip install -r requirements.txt`

2. **Error: Haar cascade file not found**
   - Download the file: `curl -o haarcascade_car.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_car.xml`

3. **YOLO model loading error**
   - The program will automatically fall back to contour detection if YOLO fails to load

4. **Poor detection results**
   - Try different detection methods (`--detector contour`, `--detector haar`, or `--detector yolo`)
   - Adjust the camera angle or video source
   - For contour detection, ensure there's good contrast between vehicles and the background 