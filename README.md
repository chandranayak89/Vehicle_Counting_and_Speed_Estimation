# Vehicle Counting and Speed Estimation

A computer vision project for detecting, counting, and estimating the speed of vehicles in traffic videos.

## Features

- Vehicle detection using background subtraction and/or deep learning models
- Vehicle tracking with unique ID assignment
- Speed estimation based on real-world distance calibration
- Traffic count statistics
- Speed violation detection and logging

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/vehicle-counting-speed-estimation.git
cd vehicle-counting-speed-estimation
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Download a sample traffic video or use your own.

## Usage

### Basic Usage

```
python main.py --input videos/traffic_video.mp4 --output output.mp4
```

### Additional Options

```
python main.py --input videos/traffic_video.mp4 --output output.mp4 --detector yolo --distance 15 --speed_limit 40
```

Parameters:
- `--input`: Path to input video file
- `--output`: Path to output video file
- `--detector`: Detection method (options: 'yolo', 'contour', 'haar', default: 'contour')
- `--distance`: Real-world distance (in meters) for calibration
- `--speed_limit`: Speed limit in km/h for violation detection
- `--show`: Display the processed frames (default: False)

## Project Structure

- `main.py`: Main script to run the application
- `detector.py`: Contains different vehicle detection methods
- `tracker.py`: Implements vehicle tracking algorithms
- `speed_estimator.py`: Handles speed calculation
- `utils.py`: Various utility functions

## License

This project is licensed under the MIT License - see the LICENSE file for details. 