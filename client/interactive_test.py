import serial
import time

# Configuration
ESP32_PORT = '/dev/ttyUSB0'  # Update this to your ESP32's port
BAUD_RATE = 115200
TIMEOUT = 2

# Valid expressions
VALID_EXPRESSIONS = [
    "greeting", "wave", "point", "confused", "shrug", 
    "angry", "sad", "sleep", "default", "pose"
]

def connect_to_esp32():
    """Connect to ESP32 and return serial object"""
    try:
        ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)  # Wait for ESP32 to initialize
        print("✅ Connected to ESP32 ChatBox successfully!")
        
        # Read any startup messages
        while ser.in_waiting > 0:
            startup_msg = ser.readline().decode('utf-8', errors='ignore').strip()
            if startup_msg:
                print(f"ESP32: {startup_msg}")
        
        return ser
    except serial.SerialException as e:
        print(f"❌ Error connecting to ESP32: {e}")
        return None

def send_command(ser, command):
    """Send command to ESP32 and read response"""
    try:
        ser.write(f"{command}\n".encode())
        print(f"📤 Sent: {command}")
        
        # Read responses for a few seconds
        start_time = time.time()
        while time.time() - start_time < 3:
            if ser.in_waiting > 0:
                response = ser.readline().decode('utf-8', errors='ignore').strip()
                if response:
                    print(f"📥 ESP32: {response}")
            time.sleep(0.1)
                    
    except Exception as e:
        print(f"❌ Error sending command: {e}")

def interactive_mode(ser):
    """Interactive command mode"""
    print("\n🎭 ESP32 ChatBox Interactive Mode")
    print("Valid commands:", ", ".join(VALID_EXPRESSIONS))
    print("Type 'exit' to quit, 'help' for commands list\n")
    
    while True:
        try:
            command = input("Enter command: ").strip().lower()
            
            if command == 'exit':
                print("👋 Exiting...")
                break
            elif command == 'help':
                print("Valid expressions:", ", ".join(VALID_EXPRESSIONS))
                continue
            elif command == '':
                continue
                
            send_command(ser, command)
            
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break

def automated_test(ser):
    """Run automated test sequence"""
    print("\n🤖 Running Automated Test Sequence...")
    
    test_sequence = [
        "default",
        "greeting", 
        "wave",
        "happy",  # Invalid command test
        "sad",
        "angry",
        "confused",
        "shrug",
        "point",
        "pose",
        "sleep",
        "default"
    ]
    
    for i, command in enumerate(test_sequence, 1):
        print(f"\n--- Test {i}/{len(test_sequence)}: {command} ---")
        send_command(ser, command)
        time.sleep(6)  # Wait for expression to complete
    
    print("\n✅ Automated test sequence completed!")

def main():
    print("🤖 ESP32 ChatBox Control System")
    print("=" * 40)
    
    # Connect to ESP32
    ser = connect_to_esp32()
    if not ser:
        return
    
    try:
        while True:
            print("\nChoose test mode:")
            print("1. Interactive Mode")
            print("2. Automated Test")
            print("3. Exit")
            
            choice = input("Enter choice (1-3): ").strip()
            
            if choice == '1':
                interactive_mode(ser)
            elif choice == '2':
                automated_test(ser)
            elif choice == '3':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    finally:
        if ser and ser.is_open:
            ser.close()
            print("🔌 Serial connection closed.")

if __name__ == "__main__":
    main()
