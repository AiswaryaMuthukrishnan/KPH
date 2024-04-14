import pandas as pd
import numpy as np
import random

# Define the number of samples
num_samples = 1000

# Generate synthetic data for crime incidents
crime_data = {
    'crime_type': np.random.choice(['Theft', 'Assault', 'Vandalism', 'Burglary', 'Robbery', 'Drug Offense', 'Fraud'], size=num_samples),
    'location_type': np.random.choice(['Urban', 'Suburban', 'Rural'], size=num_samples),
    'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], size=num_samples),
    'severity': np.random.choice(['Low', 'Medium', 'High'], size=num_samples)
}

# Generate synthetic data for police resources
police_resources = {
    'police_station': ['Station A', 'Station B', 'Station C'],
    'patrol_cars': [random.randint(1, 10) for _ in range(3)],
    'officers': [random.randint(5, 50) for _ in range(3)]
}

# Create DataFrame for crime data
crime_df = pd.DataFrame(crime_data)

# Create DataFrame for police resources
resources_df = pd.DataFrame(police_resources)

# Join the crime data with police resources
crime_df['key'] = 0
resources_df['key'] = 0
crime_police_df = pd.merge(crime_df, resources_df, on='key').drop('key', axis=1)

# Add random noise to resources to simulate allocation
crime_police_df['patrol_cars'] = crime_police_df['patrol_cars'] + np.random.randint(-2, 3, size=len(crime_police_df))
crime_police_df['officers'] = crime_police_df['officers'] + np.random.randint(-3, 4, size=len(crime_police_df))
crime_police_df['patrol_cars'] = crime_police_df['patrol_cars'].clip(0)  # Ensure no negative values
crime_police_df['officers'] = crime_police_df['officers'].clip(0)  # Ensure no negative values

# Save the dataset to a CSV file
crime_police_df.to_csv('dummy_crime_police_data.csv', index=False)
