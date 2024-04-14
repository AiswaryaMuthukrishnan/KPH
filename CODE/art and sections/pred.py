import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load your labeled dataset
df = pd.read_csv('output.csv')  # Replace with your actual dataset path

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['section_desc'],  # Assuming 'section_description' is the column containing incident descriptions
    df['section'],   # Assuming 'section' is the column containing section labels
    test_size=0.2,
    random_state=42
)

# Create a pipeline with TF-IDF vectorizer and Multinomial Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(train_data, train_labels)

# Example input
new_incident = input("Details of the incident").split()

# Make predictions on the new incident
predicted_section = model.predict(new_incident)[0]

# Get the description for the predicted section
predicted_description = df[df['section'] == predicted_section]['section_desc'].iloc[0]

# Print the predicted section and its description
print(f'Predicted Section: {predicted_section}')
print(f'Section Description: {predicted_description}')
