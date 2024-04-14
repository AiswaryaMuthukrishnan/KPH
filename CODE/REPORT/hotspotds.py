import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Generate dummy data
np.random.seed(0)

# Generate random latitude and longitude for 1000 locations in India
latitude = np.random.uniform(8.4, 37.6, 1000)  # Approximate latitude range of India
longitude = np.random.uniform(68.1, 97.4, 1000)  # Approximate longitude range of India

# Generate dummy features
crime_types = ['Theft', 'Assault', 'Burglary', 'Robbery', 'Vandalism']
types_of_crime = [random.choice(crime_types) for _ in range(1000)]

time_of_crime = [datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23)) for _ in range(1000)]

population_density = np.random.randint(50, 500, 1000)  # Dummy population density
median_income = np.random.randint(20000, 100000, 1000)  # Dummy median income
education_level = np.random.uniform(0, 1, 1000)  # Dummy education level (normalized)
unemployment_rate = np.random.uniform(0, 0.2, 1000)  # Dummy unemployment rate (normalized)

proximity_to_transport = np.random.uniform(0, 1, 1000)  # Dummy proximity to public transportation (normalized)
proximity_to_parks = np.random.uniform(0, 1, 1000)  # Dummy proximity to parks (normalized)
proximity_to_schools = np.random.uniform(0, 1, 1000)  # Dummy proximity to schools (normalized)
proximity_to_bars = np.random.uniform(0, 1, 1000)  # Dummy proximity to bars (normalized)

historical_crime_data = np.random.randint(0, 50, 1000)  # Dummy historical crime data
poverty_rate = np.random.uniform(0, 0.5, 1000)  # Dummy poverty rate (normalized)
housing_density = np.random.uniform(0, 1, 1000)  # Dummy housing density (normalized)

police_stations = np.random.randint(1, 10, 1000)  # Dummy number of police stations
patrol_routes = np.random.randint(1, 20, 1000)  # Dummy number of patrol routes
response_time = np.random.uniform(1, 30, 1000)  # Dummy police response time (minutes)

weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog', 'Thunderstorm']
weather = [random.choice(weather_conditions) for _ in range(1000)]

# Create DataFrame
dummy_data = pd.DataFrame({
    'latitude': latitude,
    'longitude': longitude,
    'crime_type': types_of_crime,
    'time_of_crime': time_of_crime,
    'population_density': population_density,
    'median_income': median_income,
    'education_level': education_level,
    'unemployment_rate': unemployment_rate,
    'proximity_to_transport': proximity_to_transport,
    'proximity_to_parks': proximity_to_parks,
    'proximity_to_schools': proximity_to_schools,
    'proximity_to_bars': proximity_to_bars,
    'historical_crime_data': historical_crime_data,
    'poverty_rate': poverty_rate,
    'housing_density': housing_density,
    'police_stations': police_stations,
    'patrol_routes': patrol_routes,
    'response_time': response_time,
    'weather_conditions': weather
})

# Save dummy data to a CSV file
dummy_data.to_csv('dummy_crime_data.csv', index=False)
