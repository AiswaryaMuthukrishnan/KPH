import pandas as pd
import numpy as np
import random

 
crime_police_df = pd.read_csv('POLICE RESOURCE ALLOCATION/dummy_crime_police_data.csv')

 
def allocate_resources(data):
    crime_counts = data.groupby(['location_type', 'time_of_day']).size().unstack(fill_value=0)
    average_crimes = crime_counts.mean(axis=1)
    total_patrol_cars = data['patrol_cars'].sum()
    total_officers = data['officers'].sum()
    patrol_allocation = (crime_counts / crime_counts.values.sum()) * total_patrol_cars
    officer_allocation = (crime_counts / crime_counts.values.sum()) * total_officers
    return patrol_allocation, officer_allocation

 
patrol_allocation, officer_allocation = allocate_resources(crime_police_df)
print("Patrol Car Allocation:")
print(patrol_allocation)

print("\nOfficer Allocation:")
print(officer_allocation)
