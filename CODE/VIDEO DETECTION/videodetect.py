import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
import cv2


# Load NLU model (BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nlu_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
nlu_model.eval()

# Load video analysis model (I3D)
video_model = r3d_18(pretrained=True)
video_model.eval()

# Load and preprocess video frames
# Load and preprocess video frames
# Load and preprocess video frames
# Load and preprocess video frames
def preprocess_frames(frames):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor()
    ])

    processed_frames = []
    for frame in frames:
        frame_tensor = transform(frame)
        frame_tensor = frame_tensor.unsqueeze(0)
        processed_frames.append(frame_tensor)

    return torch.stack(processed_frames)

# Process user input
user_input = "police kill two armed Hamas terrorists"
inputs = tokenizer(user_input, return_tensors="pt")

# Get NLU model predictions
with torch.no_grad():
    nlu_outputs = nlu_model(**inputs)

# Load and process the video
video_path = "Israeli police kill two armed Hamas terrorists in dramatic car chase near Netivot.mp4"
cap = cv2.VideoCapture(video_path)

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess video frame
    video_frames = preprocess_frames([frame])

    # Get video analysis model predictions
    with torch.no_grad():
        video_outputs = video_model(video_frames)

    # Fusion of NLU and video features
    concatenated_features = torch.cat((nlu_outputs.logits, video_outputs), dim=1)

    # Use a final classifier for making predictions
    num_classes=2
    final_classifier = torch.nn.Linear(concatenated_features.size(1), num_classes)
    predictions = final_classifier(concatenated_features)

    # Check if the prediction indicates a violent event
    if predictions.argmax(dim=1) == 1:
        print("Violent event detected in the video.")
        break

cap.release()
