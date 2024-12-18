import os
import librosa  # For audio processing
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.preprocessing.sequence import pad_sequences
import glob

# Initialize lists to store features and labels
features = []
labels = []

# Map emotions (you may need to adjust this based on your dataset labeling)
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to extract features from audio
def extract_features(audio_file, fixed_length=310):
    try:
        # Load audio file using librosa
        audio, sr = librosa.load(audio_file, sr=None)
        
        # Example feature extraction: MFCCs (Mel Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Fix the length of the MFCC sequence
        mfccs = librosa.util.fix_length(mfccs, size=fixed_length)  # Correct usage of fix_length
        
        # Reshape MFCCs for LSTM (add channel dimension if needed for CNN)
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {audio_file}: {e}")
        return None

# Path to the dataset directory
dataset_dir = "D:/Businesses/Bolo/Code/bolo_speech_emotion_app/data/RAVDESS/archive/audio_speech_actors_01-24"

# Iterate through all actor subdirectories
for actor_dir in os.listdir(dataset_dir):
    actor_path = os.path.join(dataset_dir, actor_dir)
    if os.path.isdir(actor_path):  # Ensure it's a directory
        # Process all .wav files in the current actor directory
        for wav_file in glob.glob(os.path.join(actor_path, "*.wav")):
            # Get the emotion label from the file name (assuming format: XX-XX-XX-XX-XX-XX-XX-XX.wav)
            try:
                emotion_code = wav_file.split('-')[2]  # Extract the emotion code from the file name
                emotion = emotion_map.get(emotion_code, 'unknown')
                
                # Extract features from the audio file
                feature = extract_features(wav_file)
                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Debugging: print the length of features and first few samples
print("Length of features:", len(features))
if len(features) > 0:
    print("First few samples of features:", features[:2])

# Check if features are extracted correctly
if len(features) == 0:
    print("No features extracted. Please check your dataset and feature extraction.")

# Convert labels to integers
label_map = {emotion: idx for idx, emotion in enumerate(emotion_map.values())}
labels = np.array([label_map[label] for label in labels])

# Ensure features have the same length (pad if needed)
# Find the maximum length of sequences only if there are features
if len(features) > 0:
    max_length = max([seq.shape[0] for seq in features])
    features = pad_sequences(features, padding='post', dtype='float32', maxlen=max_length)

    print("Shape of features after padding:", features.shape)

# Reshape data for LSTM layer: we need (batch_size, timesteps, features)
features = np.expand_dims(features, axis=-1)  # Add the channel dimension for LSTM (features)
print("Shape of features after expansion:", features.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# LSTM Model (without CNN)
model = models.Sequential()

# LSTM Layer
model.add(layers.LSTM(128, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))

# Dense layer for classification
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(emotion_map), activation='softmax'))  # Softmax output for multi-class classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
