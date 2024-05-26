import pandas as pd
import numpy as np

# Load the Excel file
file_path = './20240526_test.xls'

# Read the Accelerometer and Gyroscope sheets
accelerometer_df = pd.read_excel(file_path, sheet_name='Accelerometer', engine='xlrd')
gyroscope_df = pd.read_excel(file_path, sheet_name='Gyroscope', engine='xlrd')

# 合併兩個DataFrame，並Drop the 'Time (s)' column from the Gyroscope sheet
combined_df = pd.concat([accelerometer_df, gyroscope_df.drop(columns=['Time (s)'])], axis=1, ignore_index=False)

# Rename the columns
combined_df = combined_df.rename(columns={'Acceleration x (m/s^2)': 'X-axis Acceleration', 
                                          'Acceleration y (m/s^2)': 'Y-axis Acceleration',
                                          'Acceleration z (m/s^2)': 'Z-axis Acceleration',
                                          'Gyroscope x (rad/s)': 'X-axis Angular Velocity',
                                          'Gyroscope y (rad/s)': 'Y-axis Angular Velocity',
                                          'Gyroscope z (rad/s)': 'Z-axis Angular Velocity'})

# Define the sampling rate (30 Hz)
sampling_rate = 30

# Calculate the time difference between consecutive samples (in seconds)
dt = 1 / sampling_rate

# Convert angular velocity from rad/s to deg/s
combined_df['Gyroscope x (deg/s)'] = combined_df['X-axis Angular Velocity'] * 180 / np.pi
combined_df['Gyroscope y (deg/s)'] = combined_df['Y-axis Angular Velocity'] * 180 / np.pi
combined_df['Gyroscope z (deg/s)'] = combined_df['Z-axis Angular Velocity'] * 180 / np.pi

# Calculate cumulative angle (assuming 30Hz sampling rate)
combined_df['X-axis Angle'] = (combined_df['Gyroscope x (deg/s)'] * dt).cumsum()
combined_df['Y-axis Angle'] = (combined_df['Gyroscope y (deg/s)'] * dt).cumsum()
combined_df['Z-axis Angle'] = (combined_df['Gyroscope z (deg/s)'] * dt).cumsum()

combined_df.drop(columns=['Gyroscope x (deg/s)', 'Gyroscope y (deg/s)', 'Gyroscope z (deg/s)'])

# Add the cumulative angle columns to the original DataFrame
combined_df = combined_df[['Time (s)', 'X-axis Acceleration', 'Y-axis Acceleration', 'Z-axis Acceleration',
                            'X-axis Angular Velocity', 'Y-axis Angular Velocity', 'Z-axis Angular Velocity',
                            'X-axis Angle', 'Y-axis Angle', 'Z-axis Angle']]

# 保存合併後的數據到csv文件
output_csv_path = './20240526_test_combined_data.csv'
combined_df.to_csv(output_csv_path, index=False)

print("Data combined and saved to combined_data.csv successfully.")