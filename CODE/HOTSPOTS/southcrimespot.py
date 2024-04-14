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

# Filter data points in South India
south_india_data = crime_data[(crime_data['latitude'] >= 8.4) & (crime_data['latitude'] <= 12.4) &
                              (crime_data['longitude'] >= 74.5) & (crime_data['longitude'] <= 77.5)]

# Perform K-means clustering
k = 10  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=k, random_state=42)
south_india_data['cluster'] = kmeans.fit_predict(south_india_data[['latitude', 'longitude']])

# Create a Folium map centered around South India
south_india_map = folium.Map(location=[10.4, 76], zoom_start=6, tiles='OpenStreetMap')  # Specify 'OpenStreetMap' tiles

# Mark crime hotspots (cluster centroids) in South India on the map
for centroid in kmeans.cluster_centers_:
    folium.Marker(location=[centroid[0], centroid[1]], popup='Cluster Center').add_to(south_india_map)

# Save the map as an HTML file
south_india_map.save('south_india_crime_hotspots_map.html')
