import librosa
import numpy as np

def extract_features(file_path):
    # Load audio file with a sample rate of 16,000 Hz
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Extract MFCC features (commonly used in voice analysis)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Take the mean of MFCC over time (one value per coefficient)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    return mfccs_scaled

# Example usage
file_path = r'C:\Users\ikras\OneDrive\Desktop\V\audio_file.wav'  # Provide the actual path to your audio file
features = extract_features(file_path)
print(f"Extracted features: {features}")
