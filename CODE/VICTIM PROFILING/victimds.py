import pandas as pd
import numpy as np
import random

# Define the number of samples
num_samples = 1000

# Generate synthetic data for victim demographics
victim_data = {
    'victim_age': np.random.randint(18, 80, size=num_samples),
    'victim_gender': np.random.choice(['Male', 'Female'], size=num_samples),
    'victim_race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], size=num_samples)
}

# Generate synthetic data for crime type
crime_types = ['Theft', 'Assault', 'Vandalism', 'Burglary', 'Robbery', 'Drug Offense', 'Fraud']
victim_data['crime_type'] = np.random.choice(crime_types, size=num_samples)

# Generate synthetic data for location
locations = ['Urban', 'Suburban', 'Rural']
victim_data['location'] = np.random.choice(locations, size=num_samples)

# Generate synthetic data for outcome (arrest, warning, no action)
outcomes = ['Arrest', 'Warning', 'No Action']
victim_data['outcome'] = np.random.choice(outcomes, size=num_samples)

# Create a DataFrame
crime_df = pd.DataFrame(victim_data)

# Display the first few rows of the DataFrame
print(crime_df.head())

# Save the dataset to a CSV file
crime_df.to_csv('synthetic_crime_data.csv', index=False)
