
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import re

# Load the CCTNS datasets for FIRs in Hindi
hindi_firs = pd.read_csv('hindi_firs.csv')

# Create an OCR function to extract text from images
def ocr(image_path):
    """
    Performs OCR on an image and returns the extracted text.

    Args:
    image_path: Path to the image file.

    Returns:
    Extracted text from the image.
    """

    # Use an OCR library to extract text from the image
    text = pytesseract.image_to_string(image_path)
    return text

# Define a function to preprocess the text
def preprocess_text(text):
    """
    Preprocesses the given text by removing stop words, stemming, and converting to lowercase.

    Args:
    text: Text to be preprocessed.

    Returns:
    Preprocessed text.
    """

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('hindi'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stem the tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Convert the tokens to lowercase
    tokens = [token.lower() for token in tokens]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Define a function to extract crucial details from the text
def extract_crucial_details(text):
    """
    Extracts crucial details from the given text, such as the names of the complainant, accused, and witnesses, as well as the date, time, and location of the incident.

    Args:
    text: Text to extract details from.

    Returns:
    A dictionary containing the extracted details.
    """

    # Create a dictionary to store the extracted details
    details = {}

    # Use regular expressions to extract the names of the complainant, accused, and witnesses
    complainant_pattern = r"फरियादी का नाम: (.*)"
    accused_pattern = r"आरोपी का नाम: (.*)"
    witnesses_pattern = r"गवाहों के नाम: (.*)"

    complainant = re.findall(complainant_pattern, text)
    accused = re.findall(accused_pattern, text)
    witnesses = re.findall(witnesses_pattern, text)

    # Use regular expressions to extract the date, time, and location of the incident
    date_pattern = r"तारीख: (.*)"
    time_pattern = r"समय: (.*)"
    location_pattern = r"स्थान: (.*)"

    date = re.findall(date_pattern, text)
    time = re.findall(time_pattern, text)
    location = re.findall(location_pattern, text)

    # Add the extracted details to the dictionary
    details['complainant'] = complainant
    details['accused'] = accused
    details['witnesses'] = witnesses
    details['date'] = date
    details['time'] = time
    details['location'] = location

    return details

# Define a function to generate a FIR based on the extracted details
def generate_fir(details):
    """
    Generates a FIR based on the given extracted details.

    Args:
    details: A dictionary containing the extracted details.

    Returns:
    A string containing the generated FIR.
    """

    # Create a string to store the FIR
    fir = ""

    # Add the FIR header
    fir += "प्रथम सूचना रिपोर्ट (FIR)\n"

    # Add the details of the complainant
    fir += "फरियादी का नाम: {}\n".format(details['complainant'])

    # Add the details of the accused
    fir += "आरोपी का नाम: {}\n".format(details['accused'])

    # Add the details of the witnesses
    fir += "गवाहों के नाम: {}\n".format(details['witnesses'])

    # Add the date, time, and location of the incident
    fir += "तारीख: {}\n".format(details['date'])
    fir += "समय: {}\n".format(details['time'])
    fir += "स्थान: {}\n".format(details['location'])

    # Add the description of the incident
    fir += "घटना का विवरण: {}\n".format(text)

    # Add the signature of the complainant
    fir += "फरियादी का हस्ताक्षर: \n"

    # Add the signature of the police officer
    fir += "पुलिस अधिकारी का हस्ताक्षर: \n"

    return fir

# Get the incident details in description format
text = ocr('incident_details.jpg')

# Preprocess the text
text = preprocess_text(text)

# Extract crucial details from the text
details = extract_crucial_details(text)

# Generate a FIR based on the extracted details
fir = generate_fir(details)

# Print the generated FIR
print(fir)
