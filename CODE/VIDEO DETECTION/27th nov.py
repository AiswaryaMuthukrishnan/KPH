import cv2
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras import applications
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Function to preprocess text scenario
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return tokens

# Function to extract features using a pre-trained CNN
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

# Function to prepare dataset for training
def prepare_dataset(video_paths, scenarios):
    X = []
    y = []

    for video_path, scenario in zip(video_paths, scenarios):
        cap = cv2.VideoCapture(video_path)

        scenario_keywords = preprocess_text(scenario)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract features using pre-trained CNN
            frame_features = extract_cnn_features(frame)

            X.append(frame_features)
            y.append(scenario_keywords)

        cap.release()

    # Flatten the y array to make it one-dimensional
    y_flat = [item for sublist in y for item in sublist]

    return np.array(X), np.array(y_flat)

# Function to train SVM model
def train_svm(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    return model

# Function to match scenario using the trained SVM model
def match_scenario(video_path, model):
    cap = cv2.VideoCapture(video_path)
    predicted_labels = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features using pre-trained CNN
        frame_features = extract_cnn_features(frame)

        # Predict using the trained SVM model
        predicted_label = model.predict([frame_features])[0]
        predicted_labels.append(predicted_label)

    cap.release()
    return predicted_labels

# Example usage
video_paths = ['Etawah Train Crashes Into Bike _ Bike Accident _ Uttar Pradesh News _ #Shorts _ English News.mp4', 'Israeli police kill two armed Hamas terrorists in dramatic car chase near Netivot.mp4']
scenarios = ["bike crashes into train", "Gun shot and killing two people"]

# Prepare dataset for training
X, y = prepare_dataset(video_paths, scenarios)

# Split dataset into training and testing sets
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, np._FlatIterSelf , test_size=0.2, random_state=42)


# Train SVM model
svm_model = train_svm(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Match scenario in a new video
new_video_path = 'Israeli police kill two armed Hamas terrorists in dramatic car chase near Netivot.mp4'
predicted_labels = match_scenario(new_video_path, svm_model)

# Print the predicted labels for each frame
print(predicted_labels)
