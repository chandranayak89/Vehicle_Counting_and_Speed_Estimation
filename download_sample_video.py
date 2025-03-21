import os
import urllib.request
import sys

# Create videos directory if it doesn't exist
if not os.path.exists('videos'):
    os.makedirs('videos')

# URL of a sample traffic video (this is a small sample video from Pexels)
video_url = "https://www.pexels.com/download/video/854671/?fps=25.0&h=720&w=1280"
output_file = "videos/traffic_sample.mp4"

print("Downloading sample traffic video...")
print(f"URL: {video_url}")
print(f"Output file: {output_file}")

# Function to show download progress
def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB")
        sys.stdout.flush()

try:
    # Download the file with progress reporting
    urllib.request.urlretrieve(video_url, output_file, show_progress)
    print("\nDownload complete!")
    print(f"The sample video is saved to {output_file}")
    
    # Print usage instructions
    print("\nTo run the vehicle detection and speed estimation:")
    print(f"python main.py -i {output_file} -o output.mp4 --show")
    
except Exception as e:
    print(f"\nError downloading the file: {e}")
    print("If the download fails, you can try to manually download a traffic video and save it to the 'videos' folder.") 