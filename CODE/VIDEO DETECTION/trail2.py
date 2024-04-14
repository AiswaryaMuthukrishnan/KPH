import spacy
import cv2
import time

# Load an NLP model (e.g., spaCy)
nlp = spacy.load("en_core_web_sm")

def process_description(description):
    doc = nlp(description)
    
    # Extract keywords or entities from the description
    keywords = [token.text for token in doc if token.is_alpha]
    
    # Define some predefined criteria for detecting a real incident
    predefined_criteria = ["robbery", "assault", "burglary", "gunshot", "girl"]
    
    # Check if any of the predefined criteria are in the description
    return keywords

def analyze_cctv_realtime(video_path, description_keywords, max_analysis_time_seconds):
    # Access the video input
    cap = cv2.VideoCapture(video_path)
    
    start_time = time.time()
    incident_detected = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if incident_detected_in_frame(frame, description_keywords):
            incident_detected = True
            break  # Match found, exit the loop
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time >= max_analysis_time_seconds:
            print("Analysis time exceeded 1 minute. No match found.")
            break
    
    cap.release()
    
    return incident_detected

def incident_detected_in_frame(frame, description_keywords):
    # Match description keywords with the video frame
    frame_text = " ".join(description_keywords)
    
    if frame_text in frame:
        return True
    
    return False

def send_alert(message):
    # Implement your alerting mechanism (e.g., send an email, SMS, or notification)
    print(f"ALERT: {message}")

# Get user input for incident description
incident_description = "gunshot incident with car chasing"

# Process the description and determine if it's a real incident
description_keywords = process_description(incident_description)

if description_keywords:
    video_path = "Israeli police kill two armed Hamas terrorists in dramatic car chase near Netivot.mp4"
    max_analysis_time_seconds = 200  # 1 minute
    
    incident_detected = analyze_cctv_realtime(video_path, description_keywords, max_analysis_time_seconds)
    
    if incident_detected:
        send_alert("Incident detected in video feed!")
    else:
        print("No incident detected in the video feed.")
else:
    print("No real incident detected in this description.")
