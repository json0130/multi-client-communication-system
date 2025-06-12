import pyrealsense2 as rs
import sys
import os

# Debug info
print("Python version:", sys.version)
try:
    # Check pyrealsense2 version differently
    print("PyRealSense2 loaded from:", rs.__file__)
except AttributeError:
    print("PyRealSense2 loaded but version not available")

# List available RealSense devices
print("Checking for RealSense devices...")
ctx = rs.context()
devices = ctx.query_devices()
device_list = list(devices)

if len(device_list) == 0:
    print("No RealSense devices detected!")
    
    # Let's check if USB devices are visible at all
    print("\nChecking USB devices...")
    os.system("lsusb")
    
    print("\nChecking video devices...")
    os.system("ls -la /dev/video*")
    
    print("\nChecking USB bus...")
    os.system("ls -la /dev/bus/usb/*/* | grep -i intel")
else:
    print(f"Found {len(device_list)} RealSense devices:")
    for i, dev in enumerate(device_list):
        try:
            print(f"Device {i+1}:")
            print(f"  Name: {dev.get_info(rs.camera_info.name)}")
            print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")
            print(f"  USB Type: {dev.get_info(rs.camera_info.usb_type_descriptor)}")
        except Exception as e:
            print(f"  Error getting device info: {e}")

# Try to start a pipeline
print("\nAttempting to start a basic pipeline...")
try:
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("Starting pipeline...")
    pipeline.start(config)
    print("Pipeline started successfully!")
    
    # Get a single frame to test
    print("Waiting for a frame...")
    frames = pipeline.wait_for_frames(5000)  # 5 second timeout
    print("Received frame successfully!")
    
    # Stop the pipeline
    pipeline.stop()
    print("Pipeline stopped. Test completed successfully!")
    
except Exception as e:
    print(f"Error in pipeline: {e}")
    
    # Check device permissions in case of failure
    print("\nChecking device permissions...")
    os.system("ls -la /dev/video*")
