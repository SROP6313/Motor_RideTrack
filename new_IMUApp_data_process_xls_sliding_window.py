import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.spatial.transform import Rotation as R

def process_imu_data(input_file, output_file, app_time_error):
    # Load the Excel file
    xls = pd.ExcelFile(input_file)
    
    # Read the Metadata Time sheet
    metadata_time_df = pd.read_excel(xls, sheet_name='Metadata Time')
    
    # Extract the START value from the "system time text" column
    start_time_str = metadata_time_df.loc[0, 'system time text']
    
    # Remove the time zone information and parse the START time into a datetime object
    start_time_str = start_time_str.split(' UTC')[0]
    start_datetime = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
    
    # Subtract some seconds from the start time (The APP's error: slower than real time + 1s)
    start_datetime = start_datetime - timedelta(seconds=app_time_error)

    # Read the Accelerometer and Gyroscope sheets
    accelerometer_df = pd.read_excel(xls, sheet_name='Accelerometer', engine='xlrd')
    gyroscope_df = pd.read_excel(xls, sheet_name='Gyroscope', engine='xlrd')

    # Adjust the length of the dataframes to be the same
    min_length = min(len(accelerometer_df), len(gyroscope_df))
    accelerometer_df = accelerometer_df.iloc[:min_length]
    gyroscope_df = gyroscope_df.iloc[:min_length]
    
    # Combine the two DataFrames and drop the 'Time (s)' column from the Gyroscope sheet
    Reverse_Axis_Data = pd.concat([accelerometer_df, gyroscope_df.drop(columns=['Time (s)'])], axis=1, ignore_index=False)
    
    # Rename the columns
    Reverse_Axis_Data = Reverse_Axis_Data.rename(columns={
        'Acceleration x (m/s^2)': 'X-axis Acceleration', 
        'Acceleration y (m/s^2)': 'Y-axis Acceleration',
        'Acceleration z (m/s^2)': 'Z-axis Acceleration',
        'Gyroscope x (rad/s)': 'X-axis Angular Velocity',
        'Gyroscope y (rad/s)': 'Y-axis Angular Velocity',
        'Gyroscope z (rad/s)': 'Z-axis Angular Velocity'
    })
    
    # Define the sampling rate (30 Hz)
    sampling_rate = 30
    
    # Calculate the time difference between consecutive samples (in seconds)
    dt = 1 / sampling_rate
    
    # Convert angular velocity from rad/s to deg/s
    Reverse_Axis_Data['Gyroscope x (deg/s)'] = Reverse_Axis_Data['X-axis Angular Velocity'] * 180 / np.pi
    Reverse_Axis_Data['Gyroscope y (deg/s)'] = Reverse_Axis_Data['Y-axis Angular Velocity'] * 180 / np.pi
    Reverse_Axis_Data['Gyroscope z (deg/s)'] = Reverse_Axis_Data['Z-axis Angular Velocity'] * 180 / np.pi
    
    # Initialize arrays to store the calculated Euler angles
    num_samples = len(Reverse_Axis_Data)
    roll = np.zeros(num_samples)
    pitch = np.zeros(num_samples)
    yaw = np.zeros(num_samples)
    
    # Define the window size (4 seconds * 30 Hz = 120 samples)
    window_size = 4 * sampling_rate
    
    # Iterate through each sample to calculate the Euler angles
    for i in range(num_samples):
        # Calculate the start of the window
        window_start = max(0, i - window_size + 1)
        
        # Initialize the rotation matrix for this window
        r = R.from_euler('xyz', [0, 0, 0], degrees=True)
        
        # Calculate rotation within the window
        for j in range(window_start, i + 1):
            r = r * R.from_euler('xyz', [Reverse_Axis_Data['Gyroscope x (deg/s)'][j] * dt,
                                         Reverse_Axis_Data['Gyroscope y (deg/s)'][j] * dt,
                                         Reverse_Axis_Data['Gyroscope z (deg/s)'][j] * dt], degrees=True)
        
        # Extract the Euler angles from the rotation matrix
        roll[i], pitch[i], yaw[i] = r.as_euler('xyz', degrees=True)
    
    # Add the calculated Euler angles to the DataFrame
    Reverse_Axis_Data['Pitch (deg)'] = pitch
    Reverse_Axis_Data['Roll (deg)'] = roll
    Reverse_Axis_Data['Yaw (deg)'] = yaw
    
    # Drop temporary columns used for calculations
    Reverse_Axis_Data = Reverse_Axis_Data.drop(columns=['Gyroscope x (deg/s)', 'Gyroscope y (deg/s)', 'Gyroscope z (deg/s)'])
    
    # Calculate the Absolute Time and format it
    Reverse_Axis_Data['Absolute Time'] = Reverse_Axis_Data['Time (s)'].apply(
        lambda x: (start_datetime + timedelta(seconds=x)).strftime('%Y-%m-%d %H:%M:%S.%f') + '+08:00'
    )
    
    # Reorganize the DataFrame to include relevant columns
    Reverse_Axis_Data = Reverse_Axis_Data[[
        'X-axis Angular Velocity', 'Y-axis Angular Velocity', 'Z-axis Angular Velocity',
        'X-axis Acceleration', 'Y-axis Acceleration', 'Z-axis Acceleration',           
        'Pitch (deg)', 'Roll (deg)', 'Yaw (deg)', 'Absolute Time'
    ]]
    
    # Save the processed data to a CSV file
    Reverse_Axis_Data.to_csv(output_file, index=False)

# Usage
process_imu_data('./20240611_data/20240611_eric_2.xls', './20240611_data/20240611_eric_2.csv', app_time_error=4)

print("Data processed and saved to output CSV file successfully.")
