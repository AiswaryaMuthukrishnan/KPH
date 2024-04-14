import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium

# Load the dummy crime data
crime_data = pd.read_csv('dummy_crime_data.csv')

# Feature selection
features = ['latitude', 'longitude', 'population_density', 'median_income', 'education_level', 
            'unemployment_rate', 'proximity_to_transport', 'proximity_to_parks', 'proximity_to_schools', 
            'proximity_to_bars', 'historical_crime_data', 'poverty_rate', 'housing_density', 
            'police_stations', 'patrol_routes', 'response_time']

X = crime_data[features]

# Perform K-means clustering
k = 10  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=k, random_state=42)
crime_data['cluster'] = kmeans.fit_predict(X)

# Create a Folium map centered around India
india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Mark crime hotspots (cluster centroids) on the map
for centroid in kmeans.cluster_centers_:
    folium.Marker(location=[centroid[0], centroid[1]], popup='Cluster Center').add_to(india_map)

# Save the map as an HTML file
india_map.save('crime_hotspots_map.html')
