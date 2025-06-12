# arduino_handler.py
import serial
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ArduinoHandler:
    def __init__(self, serial_port=None, baud_rate=9600):
        # Get serial port from environment or use default
        self.serial_port = serial_port or os.getenv("ARDUINO_PORT", "/dev/ttyUSB0")
        self.baud_rate = baud_rate
        self.arduino = None
        self.connected = False
        
        # Try to connect to Arduino
        self.connect()
    
    def connect(self):
        """Connect to Arduino"""
        try:
            self.arduino = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            self.connected = True
            print(f" Connected to Arduino on {self.serial_port}")
            return True
        except serial.SerialException as e:
            self.arduino = None
            self.connected = False
            print(f"  Warning: Could not connect to Arduino on {self.serial_port}")
            print(f"    Error: {e}")
            return False
        except Exception as e:
            self.arduino = None
            self.connected = False
            print(f" Arduino connection error: {e}")
            return False
    
    def send_emotion(self, emotion):
        """Send emotion to Arduino - from original chatbot.py"""
        if not self.connected or not self.arduino or not self.arduino.is_open:
            print(f"  Arduino not connected, cannot send emotion: {emotion}")
            return False
        
        try:
            command = f"{emotion}\n"
            self.arduino.write(command.encode('utf-8'))
            print(f" Sent to Arduino: {emotion}")
            return True
        except Exception as e:
            print(f" Serial write failed: {e}")
            self.connected = False
            return False
    
    def send_bot_emotion(self, bot_emotion_tag):
        """Send bot emotion tag to Arduino"""
        # Map bot emotion tags to Arduino commands if needed
        # You can customize this mapping based on your Arduino code
        emotion_mapping = {
            "GREETING": "GREETING",
            "WAVE": "WAVE", 
            "POINT": "POINT",
            "CONFUSED": "CONFUSED",
            "SHRUG": "SHRUG",
            "ANGRY": "ANGRY",
            "SAD": "SAD",
            "SLEEP": "SLEEP",
            "DEFAULT": "DEFAULT",
            "POSE": "POSE"
        }
        
        arduino_command = emotion_mapping.get(bot_emotion_tag, "DEFAULT")
        return self.send_emotion(arduino_command)
    
    def test_connection(self):
        """Test Arduino connection by sending a test command"""
        if not self.connected:
            print(" Arduino not connected")
            return False
        
        print(" Testing Arduino connection...")
        success = self.send_emotion("DEFAULT")
        if success:
            print(" Arduino test successful")
        else:
            print(" Arduino test failed")
        return success
    
    def reconnect(self):
        """Try to reconnect to Arduino"""
        print(" Attempting to reconnect to Arduino...")
        if self.arduino:
            try:
                self.arduino.close()
            except:
                pass
        
        return self.connect()
    
    def close(self):
        """Close Arduino connection"""
        if self.arduino:
            try:
                self.arduino.close()
                print(" Arduino connection closed")
            except:
                pass
        self.connected = False
    
    def get_status(self):
        """Get Arduino connection status"""
        return {
            "connected": self.connected,
            "port": self.serial_port,
            "baud_rate": self.baud_rate,
            "arduino_open": self.arduino.is_open if self.arduino else False
        }

# Standalone function for easy import
def create_arduino_handler(port=None, baud_rate=9600):
    """Create and return an Arduino handler instance"""
    return ArduinoHandler(port, baud_rate)

# Test function
def test_arduino_handler():
    """Test Arduino handler functionality"""
    print(" Testing Arduino Handler...")
    
    # Create handler
    handler = ArduinoHandler()
    
    # Test connection
    if handler.connected:
        # Test different emotions
        test_emotions = ["GREETING", "WAVE", "HAPPY", "SAD", "DEFAULT"]
        
        for emotion in test_emotions:
            print(f"Testing emotion: {emotion}")
            handler.send_emotion(emotion)
            time.sleep(1)
        
        print(" Arduino handler test completed")
    else:
        print(" Arduino handler test failed - no connection")
    
    # Clean up
    handler.close()

if __name__ == "__main__":
    test_arduino_handler()
