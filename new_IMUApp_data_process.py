import pandas as pd
import numpy as np

# Read the csv file
df = pd.read_csv('20240521_angle_test/Gyroscope.csv')

# Define the sampling rate (30 Hz)
sampling_rate = 30

# Calculate the time difference between consecutive samples (in seconds)
dt = 1 / sampling_rate

# Convert angular velocity from rad/s to deg/s
df['Gyroscope x (deg/s)'] = df['Gyroscope x (rad/s)'] * 180 / np.pi
df['Gyroscope y (deg/s)'] = df['Gyroscope y (rad/s)'] * 180 / np.pi
df['Gyroscope z (deg/s)'] = df['Gyroscope z (rad/s)'] * 180 / np.pi

# Calculate cumulative angle (assuming 30Hz sampling rate)
df['X-axis Angle'] = (df['Gyroscope x (deg/s)'] * dt).cumsum()
df['Y-axis Angle'] = (df['Gyroscope y (deg/s)'] * dt).cumsum()
df['Z-axis Angle'] = (df['Gyroscope z (deg/s)'] * dt).cumsum()

# Add the cumulative angle columns to the original DataFrame
df = df[['Time (s)', 'Gyroscope x (rad/s)', 'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)',
         'X-axis Angle', 'Y-axis Angle', 'Z-axis Angle']]

# Save the updated DataFrame to a new csv file
df.to_csv('20240521_angle_test.csv', index=False)