import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import datetime
import json
from pyspark.ml.feature import StringIndexer
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pyspark.ml.feature import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# Load your dataset into a pandas DataFrame
df = pd.read_csv('/Users/mac/Downloads/predictive.csv')

# Extract 'oil_value' and 'fuel_liters' from 'details' column
df['details'] = df['details'].apply(lambda x: json.loads(x.replace("'", "\"")))
df['oil_value'] = df['details'].apply(lambda x: x.get('oil_value', None))
df['fuel_liters'] = df['details'].apply(lambda x: x.get('fuel_liters', None))

# Convert 'date_insertion' to datetime and create time-based features
df['date_insertion'] = pd.to_datetime(df['date_insertion'])
df['day_of_week'] = df['date_insertion'].dt.dayofweek
df['hour_of_day'] = df['date_insertion'].dt.hour

# Aggregate readings on a daily basis
daily_avg_df = df.groupby(["thing_id", "date_insertion"])['power_supply_voltage'].mean().reset_index()
daily_avg_df.rename(columns={'power_supply_voltage': 'daily_avg_voltage'}, inplace=True)
df = pd.merge(df, daily_avg_df, on=["thing_id", "date_insertion"], how="left")

# Create binary indicator for 'engine_status'
df['engine_alert'] = np.where(df['engine_status'] == "Abnormal", 1, 0)

# Replace null values with random numbers
oil_value_min, oil_value_max = 0, 4
fuel_liters_min, fuel_liters_max = 0, 60
df['oil_value'].fillna(np.random.uniform(oil_value_min, oil_value_max), inplace=True)
df['fuel_liters'].fillna(np.random.uniform(fuel_liters_min, fuel_liters_max), inplace=True)

# Calculate rate of change for 'battery_current'
df.sort_values(['thing_id', 'date_insertion'], inplace=True)
df['battery_current_change'] = df.groupby('thing_id')['power_supply_voltage'].diff()

df = df[["thing_id", "date_insertion", "speed", "total_km", "engine_status", "power_supply_voltage", "oil_value", "fuel_liters", "battery_current_change", "daily_avg_voltage"]]





# Define thresholds
oil_value_fail_threshold = 1
oil_value_about_to_fail_threshold = 2

# Create 'car_age' column
df['car_age'] = np.where(np.random.rand(len(df)) < 0.6, "old", "new")

# Create 'last_oil_change' column
conditions = [
    (np.random.rand(len(df)) < 0.25),
    (np.random.rand(len(df)) >= 0.25) & (np.random.rand(len(df)) < 0.5),
    (np.random.rand(len(df)) >= 0.5) & (np.random.rand(len(df)) < 0.75)
]
choices = ["new", "50km", "80km"]
df['last_oil_change'] = np.select(conditions, choices, default="old")

# Create 'status' column
conditions = [
    (df['car_age'] == "old") & (df['last_oil_change'] == "old"),
    (df['car_age'] == "old") & (df['last_oil_change'] == "80km"),
    (df['car_age'] == "old") & ((df['last_oil_change'] == "new") | (df['last_oil_change'] == "50km")),
    (df['car_age'] == "new") & ((df['last_oil_change'] == "new") | (df['last_oil_change'] == "50km") | (df['last_oil_change'] == "80km"))
]
choices = ["fail", "about to fail", "normal", "normal"]
df['status'] = np.select(conditions, choices, default="about to fail")





# Replace 'status', 'car_age', and 'last_oil_change' with numeric values
df['status'] = df['status'].map({"normal": 0, "about to fail": 1, "fail": 2})
df['car_age'] = df['car_age'].map({"old": 0, "new": 1})
df['last_oil_change'] = df['last_oil_change'].map({"new": 0, "50km": 1, "80km": 2, "old": 3})

# Select features and label
features_pd = df.drop('status', axis=1)
label_pd = df[['status']]

# Remove rows with any null value
features_pd = features_pd.dropna()
label_pd = label_pd.loc[features_pd.index]

# Remove 'thing_id' and 'date' from features
features_pd = features_pd.drop(['thing_id', 'date_insertion', 'speed', 'total_km', 'battery_current_change', 'daily_avg_voltage'], axis=1)

# Convert pandas DataFrame to NumPy array
features_array = features_pd.values
label_array = label_pd.values

# Convert labels to one-hot encoded format
label_array = to_categorical(label_array, num_classes=3)

# Scale features to range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features_array)

# Reshape to 3D array (batch_size, timesteps, input_dim)
scaled_features = np.reshape(scaled_features, (scaled_features.shape[0], 1, scaled_features.shape[1]))


# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(scaled_features.shape[1], scaled_features.shape[2])))
model.add(Dense(3,activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam')# Fit the model


# Split the data into a training set and a test set

# Split the data into a training set and a test set
features_train, features_test, label_train, label_test = train_test_split(scaled_features, label_array, test_size=0.2, random_state=42)

# Train the model on the training set
history = model.fit(features_train, label_train, epochs=50, batch_size=72, validation_split=0.2, shuffle=False)


# Plot history
# Check the keys in the history object
print(history.history.keys())

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# Make predictions on the test set
predictions = model.predict(features_test)

# Convert predicted probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Make predictions on the test set
predictions = model.predict(features_test)

# Convert predicted probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Convert true labels to class labels
true_classes = np.argmax(label_test, axis=1)

# Calculate the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()