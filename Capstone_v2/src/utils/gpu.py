# src/utils/gpu.py

import tensorflow as tf

def setup_metal_gpu():
    """Configure Metal GPU settings for Apple """
    try:
        import platform
        if platform.processor() != 'arm':
            print("Not running on Apple Silicon")
            return False

        # Clear any existing GPU memory
        tf.keras.backend.clear_session()
        
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                # Basic configuration 
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print(f"\nMetal GPU device found:")
                print(f"Number of devices: {len(physical_devices)}")
                
                # Enable Mixed Precision
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("Mixed precision enabled (float16)")
                
                return True
            except RuntimeError as e:
                # This error is expected and can be ignored
                if "Virtual devices cannot be modified after being initialized" in str(e):
                    print("GPU already initialized (this is normal in Jupyter)")
                    return True
                else:
                    print(f"Unexpected GPU error: {e}")
                    return False
        else:
            print("No Metal GPU devices found")
            return False
            
    except Exception as e:
        print(f"Error setting up Metal GPU: {e}")
        return False
