import cv2
import numpy as np
import tensorflow
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Function to preprocess text scenario
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return tokens

# Function CNN Extraction
def extract_cnn_features(frame):
    # Load pre-trained VGG16 model
    model = VGG16(weights='imagenet', include_top=False)

    # Preprocess input frame
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Extract features
    features = model.predict(img)
    return features.flatten()

# Function to compare text scenario with video frames
def match_scenario(video_path, scenario):
    cap = cv2.VideoCapture(video_path)
    c=0
    scenario_keywords = preprocess_text(scenario)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        #  image processing
        frame_features = extract_cnn_features(frame)
        
        
        # Dummy matching (replace with a machine learning model)
        match_score = np.dot(frame_features, frame_features)
        
        # Define a threshold for matching
        threshold = 1000  # Adjust according to your scenario
        
        if match_score > threshold:
            print("Scenario matched in the video!")
            c+=1
            break
        
    if c==0 :
        print("NOT FOUND")

    cap.release()

# Example usage
video_path = 'Etawah Train Crashes Into Bike _ Bike Accident _ Uttar Pradesh News _ #Shorts _ English News.mp4'
scenario = "there is a car"

match_scenario(video_path, scenario)
