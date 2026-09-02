# debug_speech.py - Speech Processing Debug Tool
import os
import sys
import base64
from modules.speech_processor import SpeechProcessor

def debug_speech_processing(audio_file_path=None):
    """Debug speech processing with detailed logging"""
    
    print("🔍 Speech Processing Debug Tool")
    print("=" * 50)
    
    # Initialize speech processor
    config = {
        'whisper_model_size': 'base',
        'whisper_device': 'cpu',  # Force CPU for debugging
        'whisper_compute_type': 'int8',
        'max_audio_length': 30,
        'sample_rate': 16000
    }
    
    processor = SpeechProcessor(config)
    
    print("🔄 Initializing speech processor...")
    if not processor.initialize():
        print("❌ Failed to initialize speech processor")
        return
    
    print("✅ Speech processor initialized successfully")
    print(f"   Model loaded: {processor.model_loaded}")
    print(f"   Device: {'cpu' if processor.device == 'auto' else processor.device}")
    
    # Get audio file
    if not audio_file_path:
        if len(sys.argv) > 1:
            audio_file_path = sys.argv[1]
        else:
            print("\n📁 No audio file provided")
            print("Usage: python debug_speech.py <audio_file.wav>")
            
            # Try to find recent temp files
            import glob
            temp_files = glob.glob("/tmp/tmp*.wav")
            if temp_files:
                print(f"\n🔍 Found recent temp files:")
                for i, f in enumerate(temp_files[-5:]):  # Show last 5
                    print(f"   {i+1}: {f}")
                
                try:
                    choice = input("Enter number to test (or Enter to skip): ").strip()
                    if choice and choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(temp_files):
                            audio_file_path = temp_files[idx]
                except:
                    pass
            
            if not audio_file_path:
                print("❌ No audio file to test")
                return
    
    # Check if file exists
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return
    
    print(f"\n🎵 Testing audio file: {audio_file_path}")
    
    # Read audio file
    try:
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()
        
        file_size = len(audio_bytes)
        print(f"📊 File size: {file_size} bytes ({file_size/1024:.1f} KB)")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Test validation
    print("\n🔍 Validating audio file...")
    try:
        is_valid = processor.validate_wav_file(audio_bytes)
        print(f"   Valid WAV: {is_valid}")
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return
    
    if not is_valid:
        print("❌ Audio file validation failed")
        return
    
    # Test transcription
    print("\n🎤 Testing transcription...")
    try:
        success, transcription, confidence = processor.transcribe_audio(audio_bytes)
        
        print(f"\n📊 Results:")
        print(f"   Success: {success}")
        print(f"   Transcription: '{transcription}'")
        print(f"   Confidence: {confidence:.1f}%")
        
        if success:
            print("✅ Speech processing successful!")
        else:
            print("❌ Speech processing failed")
            
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        print(f"   Full traceback:")
        traceback.print_exc()
    
    print("\n🔍 Debug complete")

def test_base64_processing():
    """Test base64 audio processing"""
    print("\n🔍 Testing base64 processing...")
    
    # This would test the exact same flow as the server
    if len(sys.argv) > 1:
        audio_file_path = sys.argv[1]
        
        if os.path.exists(audio_file_path):
            with open(audio_file_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Encode to base64
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            print(f"   Base64 length: {len(audio_b64)} characters")
            
            # Test processing
            config = {'whisper_device': 'cpu', 'whisper_compute_type': 'int8'}
            processor = SpeechProcessor(config)
            
            if processor.initialize():
                success, transcription, confidence = processor.transcribe_audio_base64(audio_b64)
                print(f"   Base64 processing result: {success}")
                print(f"   Transcription: '{transcription}'")
                print(f"   Confidence: {confidence:.1f}%")

if __name__ == "__main__":
    debug_speech_processing()
    test_base64_processing()