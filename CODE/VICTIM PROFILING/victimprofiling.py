import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the synthetic crime dataset
crime_df = pd.read_csv('VICTIM PROFILING/synthetic_crime_data.csv')

# 1. Descriptive Statistics
print(crime_df['victim_age'].describe())
print(crime_df['victim_gender'].value_counts())
print(crime_df['victim_race'].value_counts())

# 2. Data Visualization
# Plot the distribution of victim age
plt.figure(figsize=(8, 6))
sns.histplot(data=crime_df, x='victim_age', bins=20, kde=True)
plt.title('Distribution of Victim Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot the count of each crime type by victim gender
plt.figure(figsize=(10, 6))
sns.countplot(data=crime_df, x='crime_type', hue='victim_gender')
plt.title('Count of Each Crime Type by Victim Gender')
plt.xlabel('Crime Type')
plt.ylabel('Count')
plt.legend(title='Victim Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Plot the count of each crime type by victim race
plt.figure(figsize=(10, 6))
sns.countplot(data=crime_df, x='crime_type', hue='victim_race')
plt.title('Count of Each Crime Type by Victim Race')
plt.xlabel('Crime Type')
plt.ylabel('Count')
plt.legend(title='Victim Race', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# You can train machine learning models to predict victim demographics based on crime characteristics
# Example:
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Define features and target variable
X = crime_df[['crime_type', 'location']]
y = crime_df['victim_gender']

# Convert categorical variables into numerical format (e.g., one-hot encoding)
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_classifier.predict(X_test)
print(classification_report(y_test, y_pred))

