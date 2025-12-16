import sys
import cv2
import flask
import numpy

print(f"Python Version: {sys.version}")
print(f"OpenCV Version: {cv2.__version__}")
print(f"Flask Version: {flask.__version__}")
print(f"Numpy Version: {numpy.__version__}")

try:
    import picamera2
    print("✅ Picamera2 imported successfully!")
except ImportError as e:
    print(f"❌ Picamera2 failed: {e}")