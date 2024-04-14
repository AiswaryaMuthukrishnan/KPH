
import datetime
import os
import re
import spacy
from spacy.matcher import Matcher

# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")

# Define the rules for matching important entities and sections
matcher = Matcher(nlp.vocab)

# Define the patterns for matching names, dates, and locations
name_pattern = [{"POS": "PROPN", "OP": "?"}, {"POS": "PROPN"}]
date_pattern = [{"POS": "DATE"}]
location_pattern = [{"POS": "PROPN", "OP": "?"}, {"POS": "NOUN", "OP": "?"}]

# Define the patterns for matching sections of the Indian Penal Code (IPC) and other acts
ipc_pattern = [{"POS": "NOUN", "OP": "?"}, {"POS": "NUM", "OP": "?"}]
other_acts_pattern = [{"POS": "NOUN", "OP": "?"}, {"POS": "NOUN", "OP": "?"}]

# Add the patterns to the matcher
matcher.add("NAME", [name_pattern])
matcher.add("DATE", [date_pattern])
matcher.add("LOCATION", [location_pattern])
matcher.add("IPC", [ipc_pattern])
matcher.add("OTHER_ACTS", [other_acts_pattern])

# Function to extract important entities and sections from the complaint
def extract_info(complaint):
    # Process the complaint text using spaCy
    doc = nlp(complaint)

    # Match the entities and sections using the matcher
    matches = matcher(doc)

    # Extract the matched entities and sections
    entities = {}
    sections = {}
    for match_id, start, end in matches:
        span = doc[start:end]
        if match_id == "NAME":
            entities["name"] = span.text
        elif match_id == "DATE":
            entities["date"] = span.text
        elif match_id == "LOCATION":
            entities["location"] = span.text
        elif match_id == "IPC":
            sections["ipc"] = span.text
        elif match_id == "OTHER_ACTS":
            sections["other_acts"] = span.text

    # Return the extracted entities and sections
    return entities, sections

# Function to generate the FIR based on the extracted information
def generate_fir(entities, sections):
    # Prepare the FIR header
    fir_header = f"FIR No: {os.urandom(16).hex()}\n"
    

    # Prepare the FIR body
    fir_body = ""
    fir_body += f"Complainant Name: {entities['name']}\n"
    fir_body += f"Date of Incident: {entities['date']}\n"
    fir_body += f"Location of Incident: {entities['location']}\n"
    fir_body += f"Sections of IPC Applied: {sections['ipc']}\n"
    fir_body += f"Sections of Other Acts Applied: {sections['other_acts']}\n"
    fir_body += f"Description of Incident: {complaint}\n"

    # Prepare the FIR footer
    fir_footer = f"Signature of Complainant: ______________\n"
    fir_footer += f"Signature of Investigating Officer: ______________\n"

    # Combine the header, body, and footer to form the complete FIR
    fir = fir_header + fir_body + fir_footer

    # Return the generated FIR
    return fir

# Get the complaint from the user
complaint = input("Enter the complaint: ")

# Extract the important entities and sections from the complaint
entities, sections = extract_info(complaint)

# Generate the FIR based on the extracted information
fir = generate_fir(entities, sections)

# Print the generated FIR
print(fir)
