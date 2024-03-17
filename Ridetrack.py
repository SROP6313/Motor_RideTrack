import time
import math
from joblib import dump, load
from typing import Tuple, Optional, Union, Dict, List
import warnings
import numpy as np 
import pandas as pd
import pickle
import seaborn as sns
from collections import Counter
from statistics import mode
from tqdm import tqdm, trange
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn import tree, svm, metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier



# 方便在觀察圖片時不會被warning訊息擋住
warnings.filterwarnings("ignore")


class SensorFusion:

    def Introduction(self):
        """
        Function: Introduction to how to use this class and its methods.

        Note: This function will print out the guide to console.
        """
        intro = """

       ╔═╗ ╔╦═══╦╗ ╔╦╗ ╔╗╔═══╦═══╗╔╗╔═╦═══╦═══╦═══╦═══╗╔╗
       ║║╚╗║║╔═╗║║ ║║║ ║║║╔══╣╔══╝║║║╔╩╗╔╗╠╗╔╗║╔═╗║╔═╗╠╝║
       ║╔╗╚╝║║ ╚╣╚═╝║║ ║║║╚══╣╚══╗║╚╝╝ ║║║║║║║╠╝╔╝╠╝╔╝╠╗║
       ║║╚╗║║║ ╔╣╔═╗║║ ║║║╔══╣╔══╝║╔╗║ ║║║║║║║║ ║╔╬═╝╔╝║║
       ║║ ║║║╚═╝║║ ║║╚═╝║║╚══╣╚══╗║║║╚╦╝╚╝╠╝╚╝║ ║║║║╚═╦╝╚╗
       ╚╝ ╚═╩═══╩╝ ╚╩═══╝╚═══╩═══╝╚╝╚═╩═══╩═══╝ ╚╝╚═══╩══╝
                  ╔═══╗  ╔╗  ╔════╗       ╔╗
                  ║╔═╗║  ║║  ║╔╗╔╗║       ║║
                  ║╚═╝╠╦═╝╠══╬╝║║╚╬═╦══╦══╣║╔╗
                  ║╔╗╔╬╣╔╗║║═╣ ║║ ║╔╣╔╗║╔═╣╚╝╝
                  ║║║╚╣║╚╝║║═╣ ║║ ║║║╔╗║╚═╣╔╗╗
                  ╚╝╚═╩╩══╩══╝ ╚╝ ╚╝╚╝╚╩══╩╝╚╝
              ╔═══╗             ╔═══╗
              ║╔═╗║             ║╔══╝
              ║╚══╦══╦═╗╔══╦══╦═╣╚══╦╗╔╦══╦╦══╦═╗
              ╚══╗║║═╣╔╗╣══╣╔╗║╔╣╔══╣║║║══╬╣╔╗║╔╗╗
              ║╚═╝║║═╣║║╠══║╚╝║║║║  ║╚╝╠══║║╚╝║║║║
              ╚═══╩══╩╝╚╩══╩══╩╝╚╝  ╚══╩══╩╩══╩╝╚╝

        歡迎使用 RideTrack SensorFusion 功能！
        
        這個Class包含以下功能：
        
        1. Axis_Process: 用於處理來自車載 IMU 的數據。
           用法：Axis_Process(data_path, save_path)
        
        2. ECU_Reverse: 用於處理來自車載 ECU 的數據。
           用法：ECU_Reverse(data_path, save_path)
        
        3. Data_Merge: 用於合併兩個 CSV 檔案到一個檔案。
           用法：Data_Merge(ecu_data_path, axis_data_path, save_path)
        
        4. calibrate_angles: 用於校正角度數據。
           用法：calibrate_angles(dataset, save_path)

        5. calibrate_imu: 用於校正IMU數據。
           用法：calibrate_imu(dataset, k, save_path)

        6. normalize_data: 用於正規化指定特性。
           用法：normalize_data(dataset, feature, method, save_path)

        7. apply_kalman_filter: 應用卡爾曼濾波器到一個資料集。
           用法：apply_kalman_filter(dataset, features, q_noise, r_noise, save_path)

        8. apply_pca: 應用PCA。
            用法：apply_pca(df, n_components, save_model)

        9. get_feature_weights: 使用PCA獲得特徵權重。
            用法：get_feature_weights(df, pca_path)

        10. feature_importance: 使用隨機森林、XGBoost選擇重要特徵。
            用法：def feature_importance(self, X, y, encoder=None):
        
        """
        print(intro)


 
    # 儲存檔案使用
    @staticmethod
    def _save_dataframe(df, path):
        """
        Function: Save dataframe to csv.

        Parameters:
            df: The dataframe to be saved.
            path: The path to save the dataframe.
        """
        df.to_csv(path, index=False)

    # 計算執行時間
    @staticmethod
    def _print_execution_time(start_time):
        """
        Function: Print the execution time from start_time to now.

        Parameters:
            start_time: The start time of execution.
        """
        # Compute and print the execution time
        execution_time = time.time() - start_time
        hours, rem = divmod(execution_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Execution time: {hours} hours {minutes} minutes {seconds} seconds")


    # 處理IMU資料
    def Axis_Process(self, data_path: str, save_path: str) -> None:
        """
        Function: Used for processing data from a car-mounted Axis.

        Parameters:
            Data_Path: Path of the TXT file containing the data from the car-mounted device.
            Data_Save_Path: Path of the CSV file to save the processed data.

        Python Libraries:
            pandas: Used for handling CSV data.
            numpy: Used for performing scientific computing.
            tqdm: Used for displaying progress bars.
        """
        
        start_time = time.time()  # Start time

        Axis_Raw_Data = pd.read_csv(data_path, header=None)
        Reverse_Axis_Data_Feature = ["Absolute Time", "X-axis Angular Velocity", "Y-axis Angular Velocity", "Z-axis Angular Velocity", "X-axis Acceleration", "Y-axis Acceleration", "Z-axis Acceleration", "X-axis Angle", "Y-axis Angle", "Z-axis Angle"]
        Axis_Raw_Data = np.array(Axis_Raw_Data)
        row_lengh, column_lengh = Axis_Raw_Data.shape
        Axis_Raw_Data = Axis_Raw_Data.reshape(int(column_lengh/len(Reverse_Axis_Data_Feature)),len(Reverse_Axis_Data_Feature))
        
        
        Reverse_Axis_Data = []
        for x in range(int(column_lengh/len(Reverse_Axis_Data_Feature))):
            Reverse_Axis_Data.append(x)
        Reverse_Axis_Data = pd.DataFrame(columns = Reverse_Axis_Data_Feature ,index=Reverse_Axis_Data)
        
        
        print("\nReading 3-axis data in part1 (1/2)")
        for row in tqdm(range(int(column_lengh/len(Reverse_Axis_Data_Feature)))):
            for column in range(len(Reverse_Axis_Data_Feature)):    
                Reverse_Axis_Data.iloc[row][column] = Axis_Raw_Data[row][column]
        
        
        print("\nReading sampling time in part2 (2/2)")
        for row in tqdm(range (len(Reverse_Axis_Data)-1)):
            Reverse_Axis_Data['Absolute Time'][row] = Reverse_Axis_Data['Absolute Time'][row][2:len(Reverse_Axis_Data['Absolute Time'][row])]
            Reverse_Axis_Data['Absolute Time'].iloc[row] = pd.to_datetime(Reverse_Axis_Data['Absolute Time'].iloc[row],unit='ms',utc=True).tz_convert('Asia/Taipei') 
            Reverse_Axis_Data['Z-axis Angle'][row] = Reverse_Axis_Data['Z-axis Angle'][row][1:len(Reverse_Axis_Data['Z-axis Angle'][row])-1]
        
        Reverse_Axis_Data['Z-axis Angle'][(len(Reverse_Axis_Data)-1)] = Reverse_Axis_Data['Z-axis Angle'][(len(Reverse_Axis_Data)-1)][1:len(Reverse_Axis_Data['Z-axis Angle'][(len(Reverse_Axis_Data)-1)])-2]

        if save_path:
            try:
                self._save_dataframe(Reverse_Axis_Data, save_path)   
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")      

        self._print_execution_time(start_time)
        
        return Reverse_Axis_Data

    # 處理ECU資料，這段解碼有簽保密條款，公開時可以把這個Function拿掉
    def ECU_Reverse(self, data_path: str, save_path: Optional[str] = None) -> None:
        """
        Function: Used for processing data from a car-mounted ECU.

        Parameters:
            Data_Path: Path of the TXT file containing the data from the car-mounted device.
            Data_Save_Path: Path of the CSV file to save the processed data.

        Python Libraries:
            pandas: Used for handling CSV data.
            numpy: Used for performing scientific computing.
            tqdm: Used for displaying progress bars.
        """

        start_time = time.time()  # Start time

        ECU_Raw_Data = pd.read_csv(data_path, header=None)
        ECU_Raw_Data = ECU_Raw_Data.drop(ECU_Raw_Data.index[0:2])
        ECU_Raw_Data_0F = ECU_Raw_Data[ECU_Raw_Data.index%2 == 0 ]
        ECU_Raw_Data_0E = ECU_Raw_Data[ECU_Raw_Data.index%2 == 1 ]

        Reverse_ECU_Data_Feature = ["ECU Absolute Time", "Atmospheric Pressure", "Inclination Switch", "Fault Code Count", "Ignition Coil Current Diagnosis", "Fault Light Mileage", "Engine Operating Time", "Ignition Advance Angle", "Idling Correction Ignition Angle", "Fuel Injection Prohibition Mode", "Injection Mode", "Bypass Delay Correction", "ABV Opening", "ABV Idling Correction", "ABV Learning Value",  "Lambda Setting", "Air-Fuel Ratio Rich", "Closed Loop Control", "Air Flow", "Throttle Valve Air Flow", "Intake Manifold Pressure", "Intake Manifold Front Pressure", "MFF_AD_ADD_MMV_REL", "MFF_AD_FAC_MMV_REL", "MFF_AD_ADD_MMV", "MFF_AD_FAC_MMV", "Fuel Injection Quantity", "MFF_WUP_COR", "Ignition Mode", "Engine RPM", "Engine RPM Limit", "Idling Target RPM", "Fuel Injection Start Angle", "Fuel Pump State", "Engine State", "Engine Temperature", "Water Temperature PWM", "Ignition Magnetization Time", "Fuel Injection Time", "Closed Loop Fuel Correction","Intake Temperature", "Combustion Chamber Intake Temperature", "TPS Opening", "TPS Idling Learning Value", "Battery Voltage", "O2 Voltage", "Vehicle Speed", "TPS Voltage", "Seat Switch State"]
        Reverse_ECU_Data = []

        for row in range(min(len(ECU_Raw_Data_0E),len(ECU_Raw_Data_0F))):
            Reverse_ECU_Data.append(row)
            
        Reverse_ECU_Data = pd.DataFrame(columns = Reverse_ECU_Data_Feature ,index=Reverse_ECU_Data)


        print("\n【Reverse Engineering Restores ECU Data Part 1 (1/2)】")
        for row in tqdm(range(min(len(ECU_Raw_Data_0E),len(ECU_Raw_Data_0F)))):
            Reverse_ECU_Data['ECU Absolute Time'].iloc[row] = pd.to_datetime(ECU_Raw_Data_0E[2].iloc[row], unit='s',utc=True).tz_convert('Asia/Taipei')
            Reverse_ECU_Data['Intake Temperature'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][22:24]
            Reverse_ECU_Data['Combustion Chamber Intake Temperature'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][24:26]
            Reverse_ECU_Data['TPS Opening'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][26:30]
            Reverse_ECU_Data['TPS Idling Learning Value'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][30:34]
            Reverse_ECU_Data['Battery Voltage'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][34:36]
            Reverse_ECU_Data['O2 Voltage'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][36:40]
            Reverse_ECU_Data['Vehicle Speed'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][40:42]
            Reverse_ECU_Data['TPS Voltage'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][42:46]
            Reverse_ECU_Data['Seat Switch State'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][46:48]
            Reverse_ECU_Data['Inclination Switch'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][26:30]
            Reverse_ECU_Data['Fault Code Count'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][30:32]
            Reverse_ECU_Data['Ignition Coil Current Diagnosis'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][32:36]
            Reverse_ECU_Data['Fault Light Mileage'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][36:40]
            Reverse_ECU_Data['Engine Operating Time'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][40:44]
            Reverse_ECU_Data['Ignition Advance Angle'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][44:46]
            Reverse_ECU_Data['Idling Correction Ignition Angle'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][46:48]
            Reverse_ECU_Data['Fuel Injection Prohibition Mode'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][48:50]
            Reverse_ECU_Data['Injection Mode'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][50:52]   
            Reverse_ECU_Data['Bypass Delay Correction'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][52:54]
            Reverse_ECU_Data['ABV Opening'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][54:58]
            Reverse_ECU_Data['ABV Idling Correction'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][58:60]
            Reverse_ECU_Data['ABV Learning Value'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][60:62]
            Reverse_ECU_Data['Lambda Setting'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][62:64]
            Reverse_ECU_Data['Air-Fuel Ratio Rich'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][64:66]
            Reverse_ECU_Data['Closed Loop Control'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][66:68]
            Reverse_ECU_Data['Air Flow'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][68:72]
            Reverse_ECU_Data['Throttle Valve Air Flow'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][72:76]
            Reverse_ECU_Data['Intake Manifold Pressure'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][76:80]
            Reverse_ECU_Data['Intake Manifold Front Pressure'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][80:84]   
            Reverse_ECU_Data['MFF_AD_ADD_MMV_REL'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][84:88]
            Reverse_ECU_Data['MFF_AD_FAC_MMV_REL'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][88:92]
            Reverse_ECU_Data['MFF_AD_ADD_MMV'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][92:96]
            Reverse_ECU_Data['MFF_AD_FAC_MMV'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][96:100]    
            Reverse_ECU_Data['Fuel Injection Quantity'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][100:104]
            Reverse_ECU_Data['MFF_WUP_COR'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][104:106]
            Reverse_ECU_Data['Ignition Mode'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][106:108]
            Reverse_ECU_Data['Engine RPM'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][108:112]
            Reverse_ECU_Data['Engine RPM Limit'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][112:116]
            Reverse_ECU_Data['Idling Target RPM'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][116:120]
            Reverse_ECU_Data['Fuel Injection Start Angle'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][120:124]
            Reverse_ECU_Data['Fuel Pump State'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][124:126]
            Reverse_ECU_Data['Engine State'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][126:128]
            Reverse_ECU_Data['Engine Temperature'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][128:130]
            Reverse_ECU_Data['Water Temperature PWM'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][130:132]
            Reverse_ECU_Data['Ignition Magnetization Time'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][132:136]
            Reverse_ECU_Data['Fuel Injection Time'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][136:140]
            Reverse_ECU_Data['Closed Loop Fuel Correction'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][140:144]
            Reverse_ECU_Data['Atmospheric Pressure'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][22:26]
    
        print("\n【Reverse Engineering Restores ECU Data Part 2 (2/2)】")

        for row in tqdm(range(min(len(ECU_Raw_Data_0E),len(ECU_Raw_Data_0F)))): 
            Reverse_ECU_Data['ECU Absolute Time'].iloc[row] = Reverse_ECU_Data['ECU Absolute Time'].iloc[row]
            Reverse_ECU_Data['Atmospheric Pressure'].iloc[row] = int(Reverse_ECU_Data['Atmospheric Pressure'].iloc[row],16)
            Reverse_ECU_Data['Inclination Switch'].iloc[row] = int(Reverse_ECU_Data['Inclination Switch'].iloc[row],16)*0.004887107
            Reverse_ECU_Data['Fault Code Count'].iloc[row] = int(Reverse_ECU_Data['Fault Code Count'].iloc[row],16)
            Reverse_ECU_Data['Ignition Coil Current Diagnosis'].iloc[row] = int(Reverse_ECU_Data['Ignition Coil Current Diagnosis'].iloc[row],16)*0.004882796      
            Reverse_ECU_Data['Fault Light Mileage'].iloc[row] = int(Reverse_ECU_Data['Fault Light Mileage'].iloc[row],16)
            Reverse_ECU_Data['Engine Operating Time'].iloc[row] = int(Reverse_ECU_Data['Engine Operating Time'].iloc[row],16)*0.083333333  
            Reverse_ECU_Data['Ignition Advance Angle'].iloc[row] = (int(Reverse_ECU_Data['Ignition Advance Angle'].iloc[row],16)*0.468745098)-30
            Reverse_ECU_Data['Idling Correction Ignition Angle'].iloc[row] =  (int(Reverse_ECU_Data['Idling Correction Ignition Angle'].iloc[row],16)*0.468745098)-30 
            Reverse_ECU_Data['Fuel Injection Prohibition Mode'].iloc[row] =  int(Reverse_ECU_Data['Fuel Injection Prohibition Mode'].iloc[row],16)
            Reverse_ECU_Data['Injection Mode'].iloc[row] =  int(Reverse_ECU_Data['Injection Mode'].iloc[row],16)
            Reverse_ECU_Data['Bypass Delay Correction'].iloc[row] =  (int(Reverse_ECU_Data['Bypass Delay Correction'].iloc[row],16)*0.1)-12.8
            Reverse_ECU_Data['ABV Opening'].iloc[row] =  (int(Reverse_ECU_Data['ABV Opening'].iloc[row],16)*0.46875)
            Reverse_ECU_Data['ABV Idling Correction'].iloc[row] =  (int(Reverse_ECU_Data['ABV Idling Correction'].iloc[row],16)*0.937490196)-120
            Reverse_ECU_Data['ABV Learning Value'].iloc[row] =  (int(Reverse_ECU_Data['ABV Learning Value'].iloc[row],16)*0.937490196)-120
            Reverse_ECU_Data['Lambda Setting'].iloc[row] =  (int(Reverse_ECU_Data['Lambda Setting'].iloc[row],16)*0.003905882)+0.5
            Reverse_ECU_Data['Air-Fuel Ratio Rich'].iloc[row] =  int(Reverse_ECU_Data['Air-Fuel Ratio Rich'].iloc[row],16)                                                                   
            Reverse_ECU_Data['Closed Loop Control'].iloc[row] =  int(Reverse_ECU_Data['Closed Loop Control'].iloc[row],16)                                      
            Reverse_ECU_Data['Air Flow'].iloc[row] =  (int(Reverse_ECU_Data['Air Flow'].iloc[row],16)*0.015624994)    
            Reverse_ECU_Data['Throttle Valve Air Flow'].iloc[row] =  (int(Reverse_ECU_Data['Throttle Valve Air Flow'].iloc[row],16)*0.015624994)    
            Reverse_ECU_Data['Intake Manifold Pressure'].iloc[row] =  int(Reverse_ECU_Data['Intake Manifold Pressure'].iloc[row],16)
            Reverse_ECU_Data['Intake Manifold Front Pressure'].iloc[row] =  int(Reverse_ECU_Data['Intake Manifold Front Pressure'].iloc[row],16)                                                                          
            Reverse_ECU_Data['MFF_AD_ADD_MMV_REL'].iloc[row] =  (int(Reverse_ECU_Data['MFF_AD_ADD_MMV_REL'].iloc[row],16)*0.003906249)-128                                         
            Reverse_ECU_Data['MFF_AD_FAC_MMV_REL'].iloc[row] =  (int(Reverse_ECU_Data['MFF_AD_FAC_MMV_REL'].iloc[row],16)*0.000976562)-32
            Reverse_ECU_Data['MFF_AD_ADD_MMV'].iloc[row] =  (int(Reverse_ECU_Data['MFF_AD_ADD_MMV'].iloc[row],16)*0.003906249)-128                                         
            Reverse_ECU_Data['MFF_AD_FAC_MMV'].iloc[row] =  (int(Reverse_ECU_Data['MFF_AD_FAC_MMV'].iloc[row],16)*0.000976562)-32                                                                                 
            Reverse_ECU_Data['Fuel Injection Quantity'].iloc[row] =  (int(Reverse_ECU_Data['Fuel Injection Quantity'].iloc[row],16)*0.003906249)                                                                                                                           
            Reverse_ECU_Data['MFF_WUP_COR'].iloc[row] =  (int(Reverse_ECU_Data['MFF_WUP_COR'].iloc[row],16)*0.003905882)                                    
            Reverse_ECU_Data['Ignition Mode'].iloc[row] =  int(Reverse_ECU_Data['Ignition Mode'].iloc[row],16)  
            Reverse_ECU_Data['Engine RPM'].iloc[row] =  int(Reverse_ECU_Data['Engine RPM'].iloc[row],16)
            Reverse_ECU_Data['Engine RPM Limit'].iloc[row] =  int(Reverse_ECU_Data['Engine RPM Limit'].iloc[row],16)                                                                    
            Reverse_ECU_Data['Idling Target RPM'].iloc[row] =  int(Reverse_ECU_Data['Idling Target RPM'].iloc[row],16)-32768
            Reverse_ECU_Data['Fuel Injection Start Angle'].iloc[row] =  (int(Reverse_ECU_Data['Fuel Injection Start Angle'].iloc[row],16)*0.46875)-180                                                                           
            Reverse_ECU_Data['Fuel Pump State'].iloc[row] =  int(Reverse_ECU_Data['Fuel Pump State'].iloc[row],16)
            Reverse_ECU_Data['Engine State'].iloc[row] =  int(Reverse_ECU_Data['Engine State'].iloc[row],16)                                                                            
            Reverse_ECU_Data['Engine Temperature'].iloc[row] =  int(Reverse_ECU_Data['Engine Temperature'].iloc[row],16)-40                                                                                                                                             
            Reverse_ECU_Data['Water Temperature PWM'].iloc[row] =  (int(Reverse_ECU_Data['Water Temperature PWM'].iloc[row],16)*0.390588235)                                                                                
            Reverse_ECU_Data['Ignition Magnetization Time'].iloc[row] =  (int(Reverse_ECU_Data['Ignition Magnetization Time'].iloc[row],16)*0.004)                                                                                    
            Reverse_ECU_Data['Fuel Injection Time'].iloc[row] =  (int(Reverse_ECU_Data['Fuel Injection Time'].iloc[row],16)*0.004)                                                                                                               
            Reverse_ECU_Data['Closed Loop Fuel Correction'].iloc[row] =  (int(Reverse_ECU_Data['Closed Loop Fuel Correction'].iloc[row],16)*0.000976428)-32                                                                                                                                  
            Reverse_ECU_Data['Intake Temperature'].iloc[row] =  int(Reverse_ECU_Data['Intake Temperature'].iloc[row],16)-40                                        
            Reverse_ECU_Data['Combustion Chamber Intake Temperature'].iloc[row] = int(Reverse_ECU_Data['Combustion Chamber Intake Temperature'].iloc[row],16)-40                                                                                                                                                                                                     
            Reverse_ECU_Data['TPS Opening'].iloc[row] = (int(Reverse_ECU_Data['TPS Opening'].iloc[row],16)*0.001953124)                                        
            Reverse_ECU_Data['TPS Idling Learning Value'].iloc[row] = (int(Reverse_ECU_Data['TPS Idling Learning Value'].iloc[row],16)*0.004882796)                                                                                
            Reverse_ECU_Data['Battery Voltage'].iloc[row] = (int(Reverse_ECU_Data['Battery Voltage'].iloc[row],16)*0.062498039)+4                                                                                                                       
            Reverse_ECU_Data['O2 Voltage'].iloc[row] = (int(Reverse_ECU_Data['O2 Voltage'].iloc[row],16)*0.004882796)                                      
            #Reverse_ECU_Data['Vehicle Speed'].iloc[row] = (int(Reverse_ECU_Data['Vehicle Speed'].iloc[row],16)*0.594417404)  
            Reverse_ECU_Data['Vehicle Speed'].iloc[row] = (Reverse_ECU_Data['Engine RPM'].iloc[row]*60*434*3.14)/10000000
            Reverse_ECU_Data['TPS Voltage'].iloc[row] = (int(Reverse_ECU_Data['TPS Voltage'].iloc[row],16)*0.004882796)                                       
            Reverse_ECU_Data['Seat Switch State'].iloc[row] = int(Reverse_ECU_Data['Seat Switch State'].iloc[row],16)                                       
        
        if save_path:
            try:
                self._save_dataframe(Reverse_ECU_Data, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")            
        self._print_execution_time(start_time)

        return Reverse_ECU_Data

    # 合併資料
    def Data_Merge(self, ecu_data_path: str, axis_data_path: str, save_path: Optional[str] = None) -> None:
        """

        Function: used to merge two CSV files into one file.

        Parameters:
            
            ECU_Data_Path: the file path of the CSV file containing ECU data.

            Axis_Data_Path: the file path of the CSV file containing instrument data.

            Merge_Data_Path: the file path where the merged file will be stored.

        Libraries:

            pandas: used for CSV data processing.

            numpy: used for scientific computing.

            tqdm: used for displaying progress bar.

        """

        start_time = time.time()  # Start time

        ECU_Raw_Data = pd.read_csv(ecu_data_path)
        #ECU_Raw_Data = ECU_Raw_Data.drop('Unnamed: 0',axis=1)

        Axis_Raw_Data = pd.read_csv(axis_data_path)
        #Axis_Raw_Data = Axis_Raw_Data.drop('Unnamed: 0',axis=1)

    
        Merge_Data_No_Feature = ['No']
        Merge_Data_No = []
        for row in range(len(ECU_Raw_Data['ECU Absolute Time'])):
            Merge_Data_No.append(row)

        Merge_Data_No = pd.DataFrame(columns = Merge_Data_No_Feature ,index=Merge_Data_No)


        print ("\n【Data Engineering Megre Data Part 1 (1/2)】")
 
        for row in tqdm(range (len(ECU_Raw_Data['ECU Absolute Time'])-1)):
            Merge_Data_No['No'].iloc[row] = (Axis_Raw_Data['Absolute Time'] < ECU_Raw_Data['ECU Absolute Time'][row]).sum()

        Merge_Data_No = Merge_Data_No.fillna(0)


        Merge_Data_Number_Feature = ['Number']
        Merge_Data_Number = []
        for row in range(len(ECU_Raw_Data['ECU Absolute Time'])):
            Merge_Data_Number.append(row)

        Merge_Data_Number = pd.DataFrame(columns = Merge_Data_Number_Feature ,index=Merge_Data_Number)


        print ("\n【Data Engineering Megre Data Part 2 (2/2)】")

        for row in tqdm(range (len(ECU_Raw_Data['ECU Absolute Time'])-1)):
            Merge_Data_Number['Number'].iloc[row] = (Axis_Raw_Data['Absolute Time'] < ECU_Raw_Data['ECU Absolute Time'][row+1]).sum() - (Axis_Raw_Data['Absolute Time'] < ECU_Raw_Data['ECU Absolute Time'][row]).sum()

        Merge_Data_Number = Merge_Data_Number.fillna(0)

        Merge_ECU_Data_Feature = ["ECU Absolute Time", "Atmospheric Pressure", "Inclination Switch", "Fault Code Count", "Ignition Coil Current Diagnosis", "Fault Light Mileage", "Engine Operating Time", "Ignition Advance Angle", "Idling Correction Ignition Angle", "Fuel Injection Prohibition Mode", "Injection Mode", "Bypass Delay Correction", "ABV Opening", "ABV Idling Correction", "ABV Learning Value",  "Lambda Setting", "Air-Fuel Ratio Rich", "Closed Loop Control", "Air Flow", "Throttle Valve Air Flow", "Intake Manifold Pressure", "Intake Manifold Front Pressure", "MFF_AD_ADD_MMV_REL", "MFF_AD_FAC_MMV_REL", "MFF_AD_ADD_MMV", "MFF_AD_FAC_MMV", "Fuel Injection Quantity", "MFF_WUP_COR", "Ignition Mode", "Engine RPM", "Engine RPM Limit", "Idling Target RPM", "Fuel Injection Start Angle", "Fuel Pump State", "Engine State", "Engine Temperature", "Water Temperature PWM", "Ignition Magnetization Time", "Fuel Injection Time", "Closed Loop Fuel Correction", "Intake Temperature", "Combustion Chamber Intake Temperature", "TPS Opening", "TPS Idling Learning Value", "Battery Voltage", "O2 Voltage", "Vehicle Speed", "TPS Voltage", "Seat Switch State"]

        Merge_ECU_Data = []

    
        lenght = len(Axis_Raw_Data) - Merge_Data_No["No"].iloc[0]

        for row in range(lenght):
            Merge_ECU_Data.append(row)

        Merge_ECU_Data = pd.DataFrame(columns = Merge_ECU_Data_Feature ,index=Merge_ECU_Data)

        count=0
        for row in range(len(ECU_Raw_Data)):
            for column in range (int(Merge_Data_Number["Number"].iloc[row])):
                Merge_ECU_Data.iloc[count] = ECU_Raw_Data.iloc[row]
                count = count + 1 
        
        Merge_ECU_Data  = Merge_ECU_Data.reset_index(drop=True)


        Merge_ECU_Data = Merge_ECU_Data.dropna(axis=0)

        last = len(ECU_Raw_Data)-1
        Max = len(Axis_Raw_Data) -  (ECU_Raw_Data['ECU Absolute Time'][last] < Axis_Raw_Data['Absolute Time']).sum()

        Merge_Axis_Data = Axis_Raw_Data.iloc[Merge_Data_No['No'].iloc[0]:Max]
        Merge_Axis_Data = Merge_Axis_Data.reset_index(drop=True)
        Merge_Data = pd.concat([Merge_ECU_Data, Merge_Axis_Data], axis=1)

        if save_path:
            try:
                self._save_dataframe(Merge_Data, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")
        self._print_execution_time(start_time)

        return Merge_Data


    # 校正角度使用
    def calibrate_angles(self, dataset: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Function: Used for calibrating angle data.

        Parameters:
            dataset: DataFrame containing the angle data.
            save_path: Path of the CSV file to save the calibrated data.

        Python Libraries:
            pandas: Used for handling CSV data.
            numpy: Used for performing scientific computing.
        """
    
        start_time = time.time()  # Start time

        # Copy the dataset to prevent modifying the original one
        calibrated_data = dataset.copy()

        # Convert DataFrame to numpy array for efficiency
        angles_array = dataset[['X-axis Angle', 'Y-axis Angle', 'Z-axis Angle']].to_numpy()
        calibrated_angles_array = angles_array.copy()

        # Define the initial angles
        initial_angles = np.radians(angles_array[0, :])  # Convert to radians

        # Define the rotation matrix
        rotation_matrix = self.get_rotation_matrix(initial_angles)
        inv_rotation_matrix = np.linalg.inv(rotation_matrix)

        # Apply the inverse rotation matrix to each set of angles
        for i in tqdm(range(len(angles_array))):
            # Convert angles to radians
            angles = np.radians(angles_array[i, :])

            # Apply the inverse rotation matrix
            new_angles = np.dot(inv_rotation_matrix, angles)

            # Convert back to degrees and update the calibrated data
            calibrated_angles_array[i, :] = np.degrees(new_angles)

        # Update the DataFrame
        calibrated_data[['X-axis Angle', 'Y-axis Angle', 'Z-axis Angle']] = calibrated_angles_array

        if save_path:
            try:
                self._save_dataframe(calibrated_data, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")

        self._print_execution_time(start_time)

        return calibrated_data
    

    
    # 校正角度呼叫副程式
    def get_rotation_matrix(self, angles):
        """
        Function: Compute the rotation matrix.

        Parameters:
            angles: A numpy array containing the x, y, z angles in radians.
        """

        rotation_x = np.array([[1, 0, 0],
                               [0, np.cos(angles[0]), -np.sin(angles[0])],
                               [0, np.sin(angles[0]), np.cos(angles[0])]])

        rotation_y = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                               [0, 1, 0],
                               [-np.sin(angles[1]), 0, np.cos(angles[1])]])

        rotation_z = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                               [np.sin(angles[2]), np.cos(angles[2]), 0],
                               [0, 0, 1]])

        rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

        return rotation_matrix



    # 校正加速度角速度使用
    def calibrate_imu(self, dataset: pd.DataFrame, k: int, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Function: Used for calibrating IMU data.

        Parameters:
            dataset: DataFrame containing the IMU data.
            k: Number of initial samples to use for calibration.
            save_path: Path of the CSV file to save the calibrated data.

        Python Libraries:
            pandas: Used for handling CSV data.
            numpy: Used for performing scientific computing.
        """
    
        start_time = time.time()  # Start time

        features = ['X-axis Angular Velocity', 'Y-axis Angular Velocity', 'Z-axis Angular Velocity', 
                    'X-axis Acceleration', 'Y-axis Acceleration', 'Z-axis Acceleration']
    
        # Copy the dataset to prevent modifying the original one
        calibrated_data = dataset.copy()  
    
        for feature in features:
            # Compute the mean of the first k samples
            mean_value = dataset[feature][:k].mean()
        
            # Subtract the mean from the entire column
            calibrated_data[feature] -= mean_value

        if save_path:
            try:
                self._save_dataframe(calibrated_data, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")

        self._print_execution_time(start_time)

        return calibrated_data


    def normalize_data(self, dataset: pd.DataFrame, feature: Union[str, List[str]], method: str = "minmax", save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Function: Normalize specified feature in dataset.

        Parameters:
            dataset: The dataframe containing the data to normalize.
            feature: The column(s) in the dataframe to normalize.
            method: The normalization method to use. Options are "minmax", "standard", "robust".
            save_path: Path to save normalized data. If None, data will not be saved.

        Returns:
            normalized_df: The dataframe after normalization.
        """
        start_time = time.time()  # Start time

        # 定義一個字典來映射方法名稱到相應的類
        methods = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "robust": RobustScaler
        }

        # 檢查指定的方法是否存在
        if method not in methods:
            raise ValueError(f"Invalid method. Expected one of: {list(methods.keys())}")

        # 創建相應的物件
        scaler = methods[method]()

        # 對指定特徵進行正規化
        normalized_data = scaler.fit_transform(dataset[feature])

        # 將正規化後的資料轉換為DataFrame
        normalized_df = pd.DataFrame(normalized_data, columns=feature)

        # 將DataFrame保存為CSV檔案
        if save_path:
            try:
                self._save_dataframe(normalized_df, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")

        self._print_execution_time(start_time)

        return normalized_df


    def initialize_kalman_filter(self, dim, q_noise=0.0001, r_noise=0.001):
        """Initializes a Kalman filter."""
    
        kf = KalmanFilter(dim_x=dim, dim_z=dim)
        kf.F = np.eye(dim)
        kf.H = np.eye(dim)
        kf.Q = np.eye(dim) * q_noise
        kf.R = np.eye(dim) * r_noise
        kf.x = np.zeros((dim, 1))
        kf.P = np.eye(dim)

        return kf

    def apply_kalman_filter(self, dataset: pd.DataFrame, features: Union[str, List[str]], q_noise: float = 0.0001, r_noise: float = 0.001, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Apply Kalman filter to a dataset.

        Parameters:
            dataset: DataFrame containing the data.
            features: Features to apply the filter on.
            q_noise: Noise in the system.
            r_noise: Measurement noise.
            save_path: Path of the CSV file to save the filtered data.
        """
    
        start_time = time.time()

        # Initialize the Kalman filter
        kf = self.initialize_kalman_filter(len(features), q_noise, r_noise)

        # Convert DataFrame to numpy array for efficiency
        data_array = dataset[features].to_numpy()
        filtered_data_array = np.zeros_like(data_array)

        # Apply the Kalman filter
        for i in tqdm(range(data_array.shape[0])):
            measurement = data_array[i, :].reshape(-1, 1)

            # Predict the next state
            kf.predict()

            # Update the state
            kf.update(measurement)

            # Save the filtered result
            filtered_data_array[i, :] = kf.x[:, 0]

        # Convert the filtered data back to DataFrame and update the original dataset
        filtered_df = pd.DataFrame(filtered_data_array, columns=features)
        for feature in features:
            dataset[feature] = filtered_df[feature]

        if save_path:
            try:
                self._save_dataframe(dataset, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")
        self._print_execution_time(start_time)

        return dataset

    def apply_pca(self, df: pd.DataFrame, n_components: Optional[int] = None, save_model: Optional[str] = None) -> Tuple[pd.DataFrame, PCA]:
        
        """
        Function: Apply PCA on a dataframe and optionally save the model.
        
        Parameters: 
            df: DataFrame. The dataset to apply PCA.
            n_components: int or None. The number of components to keep. 
                          If None, keep components that explain 95% of the variance.
            model_path: str. The path to save the PCA model.
        
        Returns: 
            df_pca: DataFrame. The transformed dataset.
            pca: PCA object. The PCA model used for transformation.
        """
        start_time = time.time()  # Start time
        # Determine the number of components
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(df)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= 0.95) + 1
        print(f'適合降至{n_components}維度')
        # Apply PCA
        pca = PCA(n_components=n_components)
        df_pca = pca.fit_transform(df)

        # Save the PCA model
        if save_model:
            dump(pca, save_model)

        self._print_execution_time(start_time)
        return df_pca, pca

    def get_feature_weights(self, df: pd.DataFrame, pca_path: str) -> pd.DataFrame:
        """
        Function: Calculate and print the weight of each feature based on the PCA model.

        Parameters: 
            df: DataFrame. The original dataset.
            pca_path: str. The path of the PCA model used for transformation.

        Returns: 
            feature_weights_df: DataFrame. Sorted weights of the features.
        """

        start_time = time.time()  # Start time
        # Load the PCA model
        pca = load(pca_path)

        # Multiply the components by the explained variance ratio
        weighted_components = pca.components_.T * pca.explained_variance_ratio_

        # Get the absolute sum of weights for each original feature
        feature_weights = np.sum(np.abs(weighted_components), axis=1)

        # Create a DataFrame for better visualization
        feature_weights_df = pd.DataFrame({
            'Feature': df.columns,
            'Weight': feature_weights
        })

        # Sort by weight
        feature_weights_df = feature_weights_df.sort_values(by='Weight', ascending=False)
        
        self._print_execution_time(start_time)
        return feature_weights_df

#, pca_path: Optional[str] = None
    def feature_importance(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], 
                           encoder: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:

        if encoder is None:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
        else:
            y = encoder.transform(y)
        
        mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X, y)
        importance_rf = pd.DataFrame({
            'Feature_Name': X.columns,
            'RF_Importance': rf.feature_importances_
        }).sort_values(by='RF_Importance', ascending=False)

        xgb = XGBClassifier()
        xgb.fit(X, y)
        importance_xgb = pd.DataFrame({
            'Feature_Name': X.columns,
            'XGB_Importance': xgb.feature_importances_
        }).sort_values(by='XGB_Importance', ascending=False)

        pca = PCA(n_components=X.shape[1])
        pca.fit(X)
        weighted_components = pca.components_.T * pca.explained_variance_ratio_
        feature_weights = np.sum(np.abs(weighted_components), axis=1)
        importance_pca = pd.DataFrame({
            'Feature_Name': X.columns,
            'PCA_Importance': feature_weights / np.sum(feature_weights)  # 正規化以使總和為1
        }).sort_values(by='PCA_Importance', ascending=False)

        return importance_rf, importance_xgb, importance_pca, mapping





class AutoTag:

    def Introduction(self):
        """
        Function: Introduction to how to use this class and its methods.

        Note: This function will print out the guide to console.
        """
        intro = """

       ╔═╗ ╔╦═══╦╗ ╔╦╗ ╔╗╔═══╦═══╗╔╗╔═╦═══╦═══╦═══╦═══╗╔╗
       ║║╚╗║║╔═╗║║ ║║║ ║║║╔══╣╔══╝║║║╔╩╗╔╗╠╗╔╗║╔═╗║╔═╗╠╝║
       ║╔╗╚╝║║ ╚╣╚═╝║║ ║║║╚══╣╚══╗║╚╝╝ ║║║║║║║╠╝╔╝╠╝╔╝╠╗║
       ║║╚╗║║║ ╔╣╔═╗║║ ║║║╔══╣╔══╝║╔╗║ ║║║║║║║║ ║╔╬═╝╔╝║║
       ║║ ║║║╚═╝║║ ║║╚═╝║║╚══╣╚══╗║║║╚╦╝╚╝╠╝╚╝║ ║║║║╚═╦╝╚╗
       ╚╝ ╚═╩═══╩╝ ╚╩═══╝╚═══╩═══╝╚╝╚═╩═══╩═══╝ ╚╝╚═══╩══╝
                  ╔═══╗  ╔╗  ╔════╗       ╔╗
                  ║╔═╗║  ║║  ║╔╗╔╗║       ║║
                  ║╚═╝╠╦═╝╠══╬╝║║╚╬═╦══╦══╣║╔╗
                  ║╔╗╔╬╣╔╗║║═╣ ║║ ║╔╣╔╗║╔═╣╚╝╝
                  ║║║╚╣║╚╝║║═╣ ║║ ║║║╔╗║╚═╣╔╗╗
                  ╚╝╚═╩╩══╩══╝ ╚╝ ╚╝╚╝╚╩══╩╝╚╝
                    ╭━━━╮  ╭╮  ╭━━━━╮
                    ┃╭━╮┃ ╭╯╰╮ ┃╭╮╭╮┃
                    ┃┃ ┃┣╮┣╮╭╋━┻┫┃┃┣┻━┳━━╮
                    ┃╰━╯┃┃┃┃┃┃╭╮┃┃┃┃╭╮┃╭╮┃
                    ┃╭━╮┃╰╯┃╰┫╰╯┃┃┃┃╭╮┃╰╯┃
                    ╰╯ ╰┻━━┻━┻━━╯╰╯╰╯╰┻━╮┃
                                      ╭━╯┃
                                      ╰━━╯

        歡迎使用 RideTrack AutoTag 功能！
        
        這個Class包含以下功能：
        
        1. cluster_data: 自動標記底層動作(分群)。
           用法：cluster_data(self, dataset, feature, method="kmeans", n_clusters=3, model_path="model.pkl", save_path=None)
        
        2. predict_cluster: 重置實驗使用。
           用法：predict_cluster(self, dataset, feature, model_path, save_path=None)
        
        3. determine_optimal_clusters: 尋找最佳分群數。
           用法：determine_optimal_clusters(self, dataset, max_k, save_path=None):
        
        
        """
        print(intro)


    
    # 儲存檔案使用
    @staticmethod
    def _save_dataframe(df, path):
        """
        Function: Save dataframe to csv.

        Parameters:
            df: The dataframe to be saved.
            path: The path to save the dataframe.
        """
        df.to_csv(path, index=False)

    # 計算執行時間
    @staticmethod
    def _print_execution_time(start_time):
        """
        Function: Print the execution time from start_time to now.

        Parameters:
            start_time: The start time of execution.
        """
        # Compute and print the execution time
        execution_time = time.time() - start_time
        hours, rem = divmod(execution_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Execution time: {hours} hours {minutes} minutes {seconds} seconds")


    #分群使用
    def cluster_data(self, dataset: pd.DataFrame, feature: Union[str, List[str]], method: str = "kmeans", n_clusters: int = 11, model_path: str = "model.pkl", save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Function: Perform clustering on specified feature in dataset.

        Parameters:
            dataset: The dataframe containing the data to cluster.
            feature: The column(s) in the dataframe to cluster.
            method: The clustering method to use. Options are "kmeans", "agglomerative", "dbscan".
            n_clusters: The number of clusters to form.
            model_path: Path to save clustering model.
            save_path: Path to save clustered data. If None, data will not be saved.

        Returns:
            dataset: The dataframe after clustering.
        """
        start_time = time.time()  # Start time

        # 定義一個字典來映射方法名稱到相應的類
        methods = {
            "kmeans": KMeans,
            "agglomerative": AgglomerativeClustering,
            # DBSCAN 是基於密度的分群算法，並不需要給定分群數量
            # "dbscan": DBSCAN
        }

        # 檢查指定的方法是否存在
        if method not in methods:
            raise ValueError(f"Invalid method. Expected one of: {list(methods.keys())}")

        # 創建相應的物件
        if method == "dbscan":
            model = methods[method]()
        else:
            model = methods[method](n_clusters=n_clusters)

        # 訓練模型並進行分群
        model.fit(dataset[feature])

        # 分群結果
        dataset['Action Element'] = model.labels_

        # 儲存分群模型
        if model_path:
            dump(model, model_path)

        # 儲存分群完資料成 CSV 檔案
        if save_path:
            try:
                self._save_dataframe(dataset, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")
        self._print_execution_time(start_time)

        return dataset
    

    # 利用載入訓練好分群模型進行預測
    def predict_cluster(self, dataset: pd.DataFrame, feature: Union[str, List[str]], model_path: str, save_path: Optional[str] = None) -> pd.DataFrame:
        
        """
        Function: Predict cluster for specified feature in dataset using pre-trained model.

        Parameters:
            dataset: The dataframe containing the data to predict.
            feature: The column(s) in the dataframe to predict.
            model_path: Path to pre-trained clustering model.
            save_path: Path to save predicted data. If None, data will not be saved.

        Returns:
            dataset: The dataframe after prediction.
        """
        """
        Function: Predict cluster for specified feature in dataset using pre-trained model.

        Parameters:
            dataset: The dataframe containing the data to predict.
            feature: The column(s) in the dataframe to predict.
            model_path: Path to pre-trained clustering model.
            save_path: Path to save predicted data. If None, data will not be saved.

        Returns:
            dataset: The dataframe after prediction.
        """
        
        start_time = time.time()  # Start time

        # 載入模型
        model = load(model_path)

        # 預測
        dataset['Action Element'] = model.predict(dataset[feature])
        
        # 儲存預測後的資料成 CSV 檔案
        if save_path:
            try:
                self._save_dataframe(dataset, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")
            self._print_execution_time(start_time)

        return dataset


    # 計算分群最佳群數  副程式
    def evaluate_clustering(self, k, dataset):
        """
        Evaluate clustering performance.
        """
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(dataset)
        scores = {
            'Silhouette Score': silhouette_score(dataset, labels),
            'Calinski-Harabasz Index': calinski_harabasz_score(dataset, labels),
            'Davies-Bouldin Index': davies_bouldin_score(dataset, labels),
            'Distortion': kmeans.inertia_
        }
        return scores

    def determine_optimal_clusters(self, dataset: pd.DataFrame, max_k: int, save_path: Optional[str] = None) -> Tuple[pd.DataFrame, dict, set]:
        """
        Determine the optimal number of clusters using various evaluation metrics.
        """
        start_time = time.time()
        half_max_k = max_k // 2
        evaluation_methods = ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index', 'Distortion']
        
        # Initialize the scores list for each evaluation method
        scores = {method: [] for method in evaluation_methods}

        for k in tqdm(range(2, max_k+1)):
            new_scores = self.evaluate_clustering(k, dataset)
            for method in evaluation_methods:
                scores[method].append(new_scores[method])

        top_k = {}
        for method in evaluation_methods[:3]:  # Only the first 3 methods aim to be maximized
            top_k[method] = np.argsort(scores[method])[-half_max_k:] + 2  # +2 because k starts from 2
        top_k['Distortion'] = np.argsort(scores['Distortion'])[:half_max_k] + 2  # Distortion should be minimized

        # Calculate the intersection of the top half max_k clusters for the first 3 evaluation metrics
        intersection = set(top_k[evaluation_methods[0]])
        for method in evaluation_methods[1:3]:  # Exclude the 'Distortion' method
            intersection.intersection_update(top_k[method])

        df_scores = pd.DataFrame(scores, index=range(2, max_k+1))
        self._plot_scores(df_scores, save_path)  # Moved plotting to a separate method for clarity

        for method, ks in top_k.items():
            print(f"根據 {method}，前 {half_max_k} 個建議的分群數量分別為 {ks}")

        print(f"根據前三個評分標準推薦的分群數交集為 {intersection}")

        self._print_execution_time(start_time)
        return df_scores, top_k, intersection
    
    # 副程式
    def _plot_scores(self, df_scores, save_path):
        """
        Plot the scores for each clustering evaluation method.
        """
        plt.figure(figsize=(12, 10))
        for i, method in enumerate(df_scores.columns, 1):
            plt.subplot(2, 2, i)
            plt.plot(df_scores.index, df_scores[method], marker='o')
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel(method)
            plt.title(method)
            plt.xticks(df_scores.index)

        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path)
            except Exception as e:
                print(f"Failed to save figure to {save_path}: {e}")


#########################################################################################

# pst主要是呼叫最下面class ppm，參考Github與網址說明在最下方class ppm

class DrivePSTs:



    def Introduction(self):
        """
        Function: Introduction to how to use this class and its methods.

        Note: This function will print out the guide to console.
        """
        intro = """

       ╔═╗ ╔╦═══╦╗ ╔╦╗ ╔╗╔═══╦═══╗╔╗╔═╦═══╦═══╦═══╦═══╗╔╗
       ║║╚╗║║╔═╗║║ ║║║ ║║║╔══╣╔══╝║║║╔╩╗╔╗╠╗╔╗║╔═╗║╔═╗╠╝║
       ║╔╗╚╝║║ ╚╣╚═╝║║ ║║║╚══╣╚══╗║╚╝╝ ║║║║║║║╠╝╔╝╠╝╔╝╠╗║
       ║║╚╗║║║ ╔╣╔═╗║║ ║║║╔══╣╔══╝║╔╗║ ║║║║║║║║ ║╔╬═╝╔╝║║
       ║║ ║║║╚═╝║║ ║║╚═╝║║╚══╣╚══╗║║║╚╦╝╚╝╠╝╚╝║ ║║║║╚═╦╝╚╗
       ╚╝ ╚═╩═══╩╝ ╚╩═══╝╚═══╩═══╝╚╝╚═╩═══╩═══╝ ╚╝╚═══╩══╝
                  ╔═══╗  ╔╗  ╔════╗       ╔╗
                  ║╔═╗║  ║║  ║╔╗╔╗║       ║║
                  ║╚═╝╠╦═╝╠══╬╝║║╚╬═╦══╦══╣║╔╗
                  ║╔╗╔╬╣╔╗║║═╣ ║║ ║╔╣╔╗║╔═╣╚╝╝
                  ║║║╚╣║╚╝║║═╣ ║║ ║║║╔╗║╚═╣╔╗╗
                  ╚╝╚═╩╩══╩══╝ ╚╝ ╚╝╚╝╚╩══╩╝╚╝
                ╭━━━╮        ╭━━━┳━━━┳━━━━╮
                ╰╮╭╮┃        ┃╭━╮┃╭━╮┃╭╮╭╮┃
                 ┃┃┃┣━┳┳╮╭┳━━┫╰━╯┃╰━━╋╯┃┃┣┻━╮
                 ┃┃┃┃╭╋┫╰╯┃┃━┫╭━━┻━━╮┃ ┃┃┃━━┫
                ╭╯╰╯┃┃┃┣╮╭┫┃━┫┃  ┃╰━╯┃ ┃┃┣━━┃
                ╰━━━┻╯╰╯╰╯╰━━┻╯  ╰━━━╯ ╰╯╰━━╯

        歡迎使用 RideTrack DrivePSTs 功能！
        
        這個Class包含以下功能：
        
        1. train_vomm: VoMM/PST模型訓練(可在加動作)。
           用法：train_vomm(self, train_data, l, k, save_model=None)
        
        2. test_vomm: 使用訓練完模型預測(含簡易過濾噪聲)。
           用法：test_vomm(self, data_set, frequency, save_path=None)
        
        3. compute_accuracy: 計算各駕駛行為準確度。
           用法：compute_accuracy(self, dataset, frequency, save_path=None):

        4. 用法：calculate_action_prediction_counts: 各駕駛行為混淆矩陣。
           用法：calculate_action_prediction_counts(self, test_label, test_predict, draw_plot=False)
        
        
        """
        print(intro)


    # 儲存檔案使用
    @staticmethod
    def _save_dataframe(df, path):
        """
        Function: Save dataframe to csv.

        Parameters:
            df: The dataframe to be saved.
            path: The path to save the dataframe.
        """
        df.to_csv(path, index=False)

    # 計算執行時間
    @staticmethod
    def _print_execution_time(start_time):
        """
        Function: Print the execution time from start_time to now.

        Parameters:
            start_time: The start time of execution.
        """
        # Compute and print the execution time
        execution_time = time.time() - start_time
        hours, rem = divmod(execution_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Execution time: {hours} hours {minutes} minutes {seconds} seconds")

        
    def load_vlmm_models(self, model_file):
        with open(model_file, 'rb') as f:
            self.models = pickle.load(f)

    def train_vomm(self, train_data, l, k, save_model=None):
        start_time = time.time()
        actions = ['Go Straight', 'Idle', 'Turn Left', 'Turn Right', 'Two-Stage Left', 'U-turn']

        self.models = []
        for action in actions:
            data = train_data.loc[train_data['Action'] == action, 'Action Element'].astype(int).tolist()
            model = ppm()
            model.fit(data, d=l, alphabet_size=k)
            self.models.append(model)

        if save_model:
            with open(save_model, 'wb') as f:
                pickle.dump(self.models, f)

        self._print_execution_time(start_time)

    def test_vomm(self, data_set, frequency, save_path=None):
        action_element_list = data_set['Action Element'].values.tolist()
        start_time = time.time()
        predictions = []
        actions = ['Go Straight', 'Idle', 'Turn Left', 'Turn Right', 'Two-Stage Left', 'U-turn']

        for num in tqdm(range(len(action_element_list))):
            max_score = float('-inf')
            selected_model = None

            for model, action in zip(self.models, actions):
                scores = [math.exp(model.logpdf(action_element_list[max(0, num-i):num])) ** (1/i) for i in range(6, 30)]
                model_score = max(scores)

                if model_score > max_score:
                    max_score = model_score
                    selected_model = action

            predictions.append(selected_model)

        data_set['Predict'] = predictions
        data_set['Filter_Predict'] = self.filter_actions(data_set['Predict'], frequency)

        if save_path:
            self._save_dataframe(data_set, save_path)

        self._print_execution_time(start_time)
        return data_set

################################################################################################################################

    # 這段我是把Transition當作一個模型訓練，目前成效很低，可能原因有兩個
    # 第一個：Transition有很多種，例如:左轉接直線、直線接右轉、直走接待轉...等等 ，這裡是全部都當作同一種Transition訓練所以準確度低
    # 第二個：當初訓練時會把每個動作，前面加上一小段Transition當作訓練，但這種方法並沒有因為把Transition全部移走了，所以其他動作再銜接處預測能力低

    def train_vomm_transition(self, train_data, l, k, save_model=None):
        start_time = time.time()
        actions = ['Go Straight', 'Idle', 'Turn Left', 'Turn Right', 'Two-Stage Left', 'U-turn', 'Transition']

        self.models = []
        for action in actions:
            data = train_data.loc[train_data['Action'] == action, 'Action Element'].astype(int).tolist()
            model = ppm()
            model.fit(data, d=l, alphabet_size=k)
            self.models.append(model)

        if save_model:
            with open(save_model, 'wb') as f:
                pickle.dump(self.models, f)

        self._print_execution_time(start_time)

    def test_vomm_transition(self, data_set, frequency, save_path=None):
        action_element_list = data_set['Action Element'].values.tolist()
        start_time = time.time()
        predictions = []
        actions = ['Go Straight', 'Idle','Turn Left', 'Turn Right', 'Two-Stage Left', 'U-turn', 'Transition']

        for num in tqdm(range(len(action_element_list))):
            max_score = float('-inf')
            selected_model = None

            for model, action in zip(self.models, actions):
                scores = [math.exp(model.logpdf(action_element_list[max(0, num-i):num])) ** (1/i) for i in range(6, 30)]
                model_score = max(scores)

                if model_score > max_score:
                    max_score = model_score
                    selected_model = action

            predictions.append(selected_model)

        data_set['Predict'] = predictions
        data_set['Filter_Predict'] = self.filter_actions(data_set['Predict'], frequency)

        if save_path:
            self._save_dataframe(data_set, save_path)

        self._print_execution_time(start_time)
        return data_set
    
#############################################################################################################################################

    # @staticmethod
    # def _mode_of_data(window):
    #     return mode(window)



    @staticmethod
    def _mode_of_data(window):
        count = Counter(window)
        return count.most_common(1)[0][0]


        
    def filter_actions(self, dataset, frequency):
        filtered_data = []
        previous_action = None
        for i, action in enumerate(dataset):
            if action != previous_action:
                window = dataset[i:i+(2*frequency)]
                if self._mode_of_data(window) == action:
                    pass
                else:
                    if self._mode_of_data(window + dataset[i-frequency:i+frequency]) != action:
                        action = self._mode_of_data(window)
                    else:
                        action = previous_action
            filtered_data.append(action)
            previous_action = action
        return filtered_data

    def compute_accuracy(self, dataset, frequency, save_path=None):
        dataset['Filter_Predict'] = self.filter_actions(dataset['Predict'], frequency)
        filtered_data = dataset[['Action', 'Predict', 'Filter_Predict']].dropna()

        result_df1 = self._compute_total_accuracy(filtered_data, ['Predict', 'Filter_Predict'])
        result_df2 = self._compute_label_accuracy(filtered_data, ['Predict', 'Filter_Predict'])
        result_df2['Accuracy (Total)'] = result_df1['Accuracy (Total)']

        if save_path:
            self._save_dataframe(result_df2, save_path)    

        return result_df2

    @staticmethod
    def _compute_total_accuracy(data, columns):
        match_percentages = []
        for column in columns:
            count = (data['Action'] == data[column]).sum()
            match_percentage = (count / len(data)) * 100
            match_percentages.append(match_percentage)
        return pd.DataFrame({'RideTrack': columns, 'Accuracy (Total)': match_percentages})

    @staticmethod
    def _compute_label_accuracy(data, columns):
        class_labels = data['Action'].unique()
        match_percentages = []
        for column in columns:
            column_match_percentages = []
            for label in class_labels:
                total = (data['Action'] == label).sum()
                count = ((data['Action'] == label) & (data[column] == label)).sum()
                match_percentage = (count / total) * 100 if total != 0 else 0
                column_match_percentages.append(match_percentage)
            match_percentages.append(column_match_percentages)
        return pd.DataFrame(match_percentages, columns=class_labels, index=columns).reset_index()




    def calculate_action_prediction_counts(self, test_label, test_predict, draw_plot=False):
        conf_matrix = confusion_matrix(test_label, test_predict)
        
        labels = np.unique(np.concatenate((test_label, test_predict)))
        #labels = np.unique(test_label)
        columns = [f'Predicted: {label}' for label in labels]

        result_df = pd.DataFrame(columns=['Action'] + columns + ['Accuracy'])

        for i, action in enumerate(labels):
            true_label_count = conf_matrix[i, i]
            total_counts = conf_matrix[i, :].sum()
            accuracy = true_label_count / total_counts * 100
            result_df.loc[i] = [action, *conf_matrix[i], accuracy]

        overall_correct = np.diag(conf_matrix).sum()
        overall_total = conf_matrix.sum()
        overall_accuracy = (overall_correct / overall_total) * 100

        print(result_df.to_markdown(index=False))
    
        if draw_plot:
            self._draw_accuracy_plot(result_df, overall_accuracy)       

        return result_df

    def _draw_accuracy_plot(self, result_df, overall_accuracy):
        actions = result_df['Action'].tolist()
        accuracies = result_df['Accuracy'].tolist() 

        fig, ax1 = plt.subplots(figsize=(10,6))

        palette = sns.color_palette("husl", len(actions))

        sns.barplot(x='Action', y='Accuracy', data=result_df, ax=ax1, palette=palette)

        ax1.axhline(overall_accuracy, color='red', linestyle='--')
        ax1.text(len(actions)-0.5, overall_accuracy + 1, f'Accuracy (Total): {overall_accuracy:.2f}', color='black', ha='right', fontsize=16)

        ax1.set_title('Accuracy analysis of different behaviors', fontsize=20, pad=12)
        ax1.set_xlabel('Behavior', fontsize=20, labelpad=10)
        ax1.set_ylabel('Accuracy', fontsize=20, labelpad=10)

        ax1.set_ylim(0, 100)
        plt.xticks(fontsize=14)
        for label in ax1.get_xticklabels():
            label.set_rotation(0)

        plt.show()



class ComparisonTargets:



    def Introduction(self):
        """
        Function: Introduction to how to use this class and its methods.

        Note: This function will print out the guide to console.
        """
        intro = """

       ╔═╗ ╔╦═══╦╗ ╔╦╗ ╔╗╔═══╦═══╗╔╗╔═╦═══╦═══╦═══╦═══╗╔╗
       ║║╚╗║║╔═╗║║ ║║║ ║║║╔══╣╔══╝║║║╔╩╗╔╗╠╗╔╗║╔═╗║╔═╗╠╝║
       ║╔╗╚╝║║ ╚╣╚═╝║║ ║║║╚══╣╚══╗║╚╝╝ ║║║║║║║╠╝╔╝╠╝╔╝╠╗║
       ║║╚╗║║║ ╔╣╔═╗║║ ║║║╔══╣╔══╝║╔╗║ ║║║║║║║║ ║╔╬═╝╔╝║║
       ║║ ║║║╚═╝║║ ║║╚═╝║║╚══╣╚══╗║║║╚╦╝╚╝╠╝╚╝║ ║║║║╚═╦╝╚╗
       ╚╝ ╚═╩═══╩╝ ╚╩═══╝╚═══╩═══╝╚╝╚═╩═══╩═══╝ ╚╝╚═══╩══╝
                  ╔═══╗  ╔╗  ╔════╗       ╔╗
                  ║╔═╗║  ║║  ║╔╗╔╗║       ║║
                  ║╚═╝╠╦═╝╠══╬╝║║╚╬═╦══╦══╣║╔╗
                  ║╔╗╔╬╣╔╗║║═╣ ║║ ║╔╣╔╗║╔═╣╚╝╝
                  ║║║╚╣║╚╝║║═╣ ║║ ║║║╔╗║╚═╣╔╗╗
                  ╚╝╚═╩╩══╩══╝ ╚╝ ╚╝╚╝╚╩══╩╝╚╝
        ╭━━━╮                      ╭━━━━╮         ╭╮
        ┃╭━╮┃                      ┃╭╮╭╮┃        ╭╯╰╮
        ┃┃ ╰╋━━┳╮╭┳━━┳━━┳━┳┳━━┳━━┳━╋╯┃┃┣┻━┳━┳━━┳━┻╮╭╋━━╮
        ┃┃ ╭┫╭╮┃╰╯┃╭╮┃╭╮┃╭╋┫━━┫╭╮┃╭╮╮┃┃┃╭╮┃╭┫╭╮┃┃━┫┃┃━━┫
        ┃╰━╯┃╰╯┃┃┃┃╰╯┃╭╮┃┃┃┣━━┃╰╯┃┃┃┃┃┃┃╭╮┃┃┃╰╯┃┃━┫╰╋━━┃
        ╰━━━┻━━┻┻┻┫╭━┻╯╰┻╯╰┻━━┻━━┻╯╰╯╰╯╰╯╰┻╯╰━╮┣━━┻━┻━━╯
                  ┃┃                        ╭━╯┃
                  ╰╯                        ╰━━╯

        歡迎使用 RideTrack ComparisonTargets 功能！
        
        這個Class包含以下功能：
        
        1. compare_ml_models: 傳統機器學習演算法(SnapShot)。
           用法：compare_ml_models(self, train_dataset, test_dataset, feature, save_path=None)
        
        2. compare_m2_models: 傳統機器學習演算法(Window)。
           用法：compare_m2_models(self, train_dataset, test_dataset, feature, window_size, save_path=None)
        
        3. compare_m3_models: DeepConvLSTM   ( Python 3.6 )
           用法：compute_accuracy(self, dataset, frequency, save_path=None)
       
        """
        print(intro)


    # 儲存檔案使用
    @staticmethod
    def _save_dataframe(df, path):
        """
        Function: Save dataframe to csv.

        Parameters:
            df: The dataframe to be saved.
            path: The path to save the dataframe.
        """
        df.to_csv(path, index=False)

    # 計算執行時間
    @staticmethod
    def _print_execution_time(start_time):
        """
        Function: Print the execution time from start_time to now.

        Parameters:
            start_time: The start time of execution.
        """
        # Compute and print the execution time
        execution_time = time.time() - start_time
        hours, rem = divmod(execution_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Execution time: {hours} hours {minutes} minutes {seconds} seconds")





    def __init__(self):
        self.models = [
            ('Support Vector Machines', svm.SVC()), 
            ('Nearest Neighbors', KNeighborsClassifier(n_neighbors=6)), 
            ('Decision Trees', tree.DecisionTreeClassifier()), 
            ('Forests of randomized trees', RandomForestClassifier(n_estimators=10)), 
            ('Neural Network models', MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), random_state=42, activation='relu')),
            ('GaussianProcess', GaussianProcessClassifier())
        ]
        
        self.EPOCH = 10
        self.BATCH_SIZE = 16
        self.LSTM_UNITS = 32
        self.CNN_FILTERS = 3
        self.LEARNING_RATE = 0.001
        self.PATIENCE = 20
        self.SEED = 0
        self.DROPOUT = 0.1

    def _train_and_evaluate(self, model, train_dataset, test_dataset, feature):
        model_name, model_instance = model
        model_instance.fit(train_dataset[feature], train_dataset['Action'])
        test_predict = model_instance.predict(test_dataset[feature])
        acc = metrics.accuracy_score(test_dataset['Action'], test_predict)
        f1 = f1_score(test_dataset['Action'], test_predict, average='weighted')
        recall = recall_score(test_dataset['Action'], test_predict, average='weighted')
        confusion_mat = confusion_matrix(test_dataset['Action'], test_predict)
        return model_name, acc, f1, recall, confusion_mat

    def compare_ml_models(self, train_dataset, test_dataset, feature, save_path=None):
        start_time = time.time()
        results = []
        confusion_matrices = {}
        for model in self.models:
            model_name, acc, f1, recall, confusion_mat = self._train_and_evaluate(model, train_dataset, test_dataset, feature)
            results.append((model_name, acc, f1, recall))
            confusion_matrices[model_name] = confusion_mat

        results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1_Score', 'Recall'])
        if save_path:
            self._save_dataframe(results_df, save_path)

        self._print_execution_time(start_time)
        return results_df, confusion_matrices

    def create_windows(self, data, feature, window_size):
        windows = []
        labels = []
        for i in range(window_size, len(data)):
            windows.append(data[feature][i-window_size:i].values)
            labels.append(data['Action'].iloc[i])
        return np.array(windows), np.array(labels)

    def compare_m2_models(self, train_dataset, test_dataset, feature, window_size, save_path=None):
        start_time = time.time()
        train_data, train_labels = self.create_windows(train_dataset, feature, window_size)
        test_data, test_labels = self.create_windows(test_dataset, feature, window_size)

        train_data = train_data.reshape((train_data.shape[0], -1))
        test_data = test_data.reshape((test_data.shape[0], -1))
 
        results = []
        confusion_matrices = {}
        for model in self.models:
            model_name, model_instance = model
            model_instance.fit(train_data, train_labels)
            test_predict = model_instance.predict(test_data)
            acc = metrics.accuracy_score(test_labels, test_predict)
            f1 = f1_score(test_labels, test_predict, average='weighted')
            recall = recall_score(test_labels, test_predict, average='weighted')
            confusion_mat = confusion_matrix(test_labels, test_predict)
            results.append((model_name, acc, f1, recall))
            confusion_matrices[model_name] = confusion_mat

        results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1_Score', 'Recall'])
        if save_path:
            self._save_dataframe(results_df, save_path)

        self._print_execution_time(start_time)
        return results_df, confusion_matrices

################################################################################################################################

    # 參考Github:https://github.com/isukrit/encodingHumanActivity 
    # 之前Jason家澤有幫忙跑過可以詢問他，他回想起來可能會比其他學弟直接摸索來的快

    def model(self, x_train, num_labels, LSTM_units, dropout, num_conv_filters, batch_size):
        """
        Baseline model with a CNN layer and LSTM RNN layer.
        Inputs:
        - x_train: required for creating input shape for RNN layer in Keras
        - num_labels: number of output classes (int)
        - LSTM_units: number of RNN units (int)
        - dropout: dropout rate (float)
        - num_conv_filters: number of CNN filters (int)
        - batch_size: number of samples to be processed in each batch
        Returns
        - model: A Keras model
        """
        cnn_inputs = Input(batch_shape=(batch_size, x_train.shape[1], x_train.shape[2], 1), name='rnn_inputs')
        cnn_layer = Conv2D(num_conv_filters, kernel_size = (1, x_train.shape[2]), strides=(1, 1), padding='valid', data_format="channels_last")
        cnn_out = cnn_layer(cnn_inputs)

        sq_layer = Lambda(lambda x: K.squeeze(x, axis = 2))
        sq_layer_out = sq_layer(cnn_out)

        rnn_layer = LSTM(LSTM_units, return_sequences=False, name='lstm') #return_state=True
        rnn_layer_output = rnn_layer(sq_layer_out)

        dropout_layer = Dropout(rate = dropout)
        dropout_layer_output = dropout_layer(rnn_layer_output)

        dense_layer = Dense(num_labels, activation = 'softmax')
        dense_layer_output = dense_layer(dropout_layer_output)

        model = Model(inputs=cnn_inputs, outputs=dense_layer_output)

        print (model.summary())

        return model

    def compare_m3_models(self, train_data ,model_path=None, save_path=None):
        tmp = np.load(train_data, allow_pickle=True)
        X = tmp['X']
        X = np.squeeze(X, axis = 1)
        y_one_hot = tmp['y']
        folds = tmp['folds']

        print (y_one_hot.shape)

        NUM_LABELS = y_one_hot.shape[1]

        avg_acc = []
        avg_recall = []
        avg_f1 = []
        early_stopping_epoch_list = []
        y = np.argmax(y_one_hot, axis=1)

        results = {
            'Seed': [],
            'DataFile': [],
            'Fold': [],
            'EarlyStoppingEpoch': [],
            'AllTrainableCount': [],
            'Accuracy': [],
            'MAE': [],
            'Recall': [],
            'F1': []
        }


        for i in range(0, len(folds)):
            train_idx = folds[i][0]
            test_idx = folds[i][1]

            X_train, y_train, y_train_one_hot = X[train_idx], y[train_idx], y_one_hot[train_idx]
            X_test, y_test, y_test_one_hot = X[test_idx], y[test_idx], y_one_hot[test_idx]

            X_train_ = np.expand_dims(X_train, axis = 3)
            X_test_ = np.expand_dims(X_test, axis = 3)
            
            train_trailing_samples =  X_train_.shape[0]%self.BATCH_SIZE
            test_trailing_samples =  X_test_.shape[0]%self.BATCH_SIZE

            if train_trailing_samples!= 0:
                X_train_ = X_train_[0:-train_trailing_samples]
                y_train_one_hot = y_train_one_hot[0:-train_trailing_samples]
                y_train = y_train[0:-train_trailing_samples]
            if test_trailing_samples!= 0:
                X_test_ = X_test_[0:-test_trailing_samples]
                y_test_one_hot = y_test_one_hot[0:-test_trailing_samples]
                y_test = y_test[0:-test_trailing_samples]

            print (y_train.shape, y_test.shape)   

            rnn_model = self.model(x_train = X_train_, num_labels = NUM_LABELS, LSTM_units = self.LSTM_UNITS, \
                dropout = self.DROPOUT, num_conv_filters = self.CNN_FILTERS, batch_size = self.BATCH_SIZE)
            
            model_filename = model_path + 'model_baseline_'+ str(i) + '.h5'
            callbacks = [self.ModelCheckpoint(filepath=model_filename, monitor = 'val_acc', save_weights_only=True, save_best_only=True), self.EarlyStopping(monitor='val_acc', patience=self.PATIENCE)]#, LearningRateScheduler()]

            opt = self.optimizers.Adam(clipnorm=1.)

            rnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            history = rnn_model.fit(X_train_, y_train_one_hot, epochs=self.EPOCH, batch_size=self.BATCH_SIZE, verbose=1, callbacks=callbacks, validation_data=(X_test_, y_test_one_hot))

            early_stopping_epoch = callbacks[1].stopped_epoch - self.PATIENCE + 1 
            print('Early stopping epoch: ' + str(early_stopping_epoch))
            early_stopping_epoch_list.append(early_stopping_epoch)

            if early_stopping_epoch < 0:
                early_stopping_epoch = -100

            # Evaluate model and predict data on TEST 
            print("******Evaluating TEST set*********")
            rnn_model.load_weights(model_filename)
            y_test_predict = rnn_model.predict(X_test_, batch_size = BATCH_SIZE)
            y_test_predict = np.array(y_test_predict)
            y_test_predict = np.argmax(y_test_predict, axis=1)

            all_trainable_count = int(np.sum([K.count_params(p) for p in set(rnn_model.trainable_weights)]))
            
            acc_fold = accuracy_score(y_test, y_test_predict)
            recall_fold = recall_score(y_test, y_test_predict, average='macro')
            f1_fold  = f1_score(y_test, y_test_predict, average='macro')
            
            MAE = metrics.mean_absolute_error(y_test, y_test_predict, sample_weight=None, multioutput='uniform_average')

            results['Fold'].append(i)
            results['EarlyStoppingEpoch'].append(early_stopping_epoch)
            results['AllTrainableCount'].append(all_trainable_count)
            results['Accuracy'].append(acc_fold)
            results['MAE'].append(MAE)
            results['Recall'].append(recall_fold)
            results['F1'].append(f1_fold)
        

        results['Accuracy'].append(np.mean(results['Accuracy']))
        results['Recall'].append(np.mean(results['Recall']))
        results['F1'].append(np.mean(results['F1']))

        return results




class else_:


    def Introduction(self):
        """
        Function: Introduction to how to use this class and its methods.

        Note: This function will print out the guide to console.
        """
        intro = """

       ╔═╗ ╔╦═══╦╗ ╔╦╗ ╔╗╔═══╦═══╗╔╗╔═╦═══╦═══╦═══╦═══╗╔╗
       ║║╚╗║║╔═╗║║ ║║║ ║║║╔══╣╔══╝║║║╔╩╗╔╗╠╗╔╗║╔═╗║╔═╗╠╝║
       ║╔╗╚╝║║ ╚╣╚═╝║║ ║║║╚══╣╚══╗║╚╝╝ ║║║║║║║╠╝╔╝╠╝╔╝╠╗║
       ║║╚╗║║║ ╔╣╔═╗║║ ║║║╔══╣╔══╝║╔╗║ ║║║║║║║║ ║╔╬═╝╔╝║║
       ║║ ║║║╚═╝║║ ║║╚═╝║║╚══╣╚══╗║║║╚╦╝╚╝╠╝╚╝║ ║║║║╚═╦╝╚╗
       ╚╝ ╚═╩═══╩╝ ╚╩═══╝╚═══╩═══╝╚╝╚═╩═══╩═══╝ ╚╝╚═══╩══╝
                  ╔═══╗  ╔╗  ╔════╗       ╔╗
                  ║╔═╗║  ║║  ║╔╗╔╗║       ║║
                  ║╚═╝╠╦═╝╠══╬╝║║╚╬═╦══╦══╣║╔╗
                  ║╔╗╔╬╣╔╗║║═╣ ║║ ║╔╣╔╗║╔═╣╚╝╝
                  ║║║╚╣║╚╝║║═╣ ║║ ║║║╔╗║╚═╣╔╗╗
                  ╚╝╚═╩╩══╩══╝ ╚╝ ╚╝╚╝╚╩══╩╝╚╝
                          ╭━━━┳╮
                          ┃╭━━┫┃
                          ┃╰━━┫┃╭━━┳━━╮
                          ┃╭━━┫┃┃━━┫┃━┫
                          ┃╰━━┫╰╋━━┃┃━┫
                          ╰━━━┻━┻━━┻━━╯

        歡迎使用 RideTrack Else 功能！
        
        這個Class包含以下功能：
        
        1. Tradition_Category: Equal Width Bucketing (輸入分位數)。
           用法：Tradition_Category(self, DataSet, Quantiles, Feature, Save)
        
        2. Tradition_Category_Value: Equal Width Bucketing (輸入數值)。
           用法：Tradition_Category_Value(self, DataSet, Quantiles_Value, Feature, Save)
        
        3. Tradition_Encoding: 依分類標記SnapShot。
           用法：Tradition_Encoding(self, DataSet, Feature, Save)

        4. 用法：Tradition_Find_Top_K: 找 Top K 佔資料百分比。
           用法：Tradition_Find_Top_K(self, DataSet, Target_Percentage)

        5. 用法：Plot_Action_Cluter: 繪製底層動作所組成之高階行為。
           用法：Plot_Action_Cluter(self, DataSet, Action1, Action2, Feature, Cluster, Length, Save)

        6. 用法：Plot_Action_Track: 繪製駕駛行為軌跡。
           用法：Plot_Action_Track(self, DataSet, Step_Column_Name, Slice_size, Save)

        7. Calculating_Time: 影片中標記換算時間使用。
           用法：Calculating_Time(self, Video_Ecu_Time, Video_Mark_Time, Real_Ecu_Time)

        """
        print(intro)


    # 傳統閥值分類並儲存檔案
    def Tradition_Category(self, DataSet, Quantiles, Feature, Save):

        Category=[]
        for feature in Feature:
            Variable_Category = f"{feature}_Category"
            Category.append(Variable_Category)
            Variable_thresholds = DataSet[feature].quantile(Quantiles).tolist()

            DataSet[Variable_Category] = pd.cut(DataSet[feature], bins=Variable_thresholds, labels=False)
    
        DataSet[Category] = DataSet[Category].fillna(0)

        if Save:
            DataSet.to_csv(f'Traditional_Threshold_{len(Quantiles)-1}^{len(Feature)}_groups.csv', index=False)
        
        return DataSet
    
    # 傳統閥值分類並儲存檔案
    def Tradition_Category_Value(self, DataSet, Quantiles_Value, Feature, Save):

        Category=[]
        for index, feature in enumerate(Feature):
            Variable_Category = f"{feature}_Category"
            Category.append(Variable_Category)
            Variable_thresholds = Quantiles_Value[index]

            DataSet[Variable_Category] = pd.cut(DataSet[feature], bins=Variable_thresholds, labels=False)
            # DataSet[Variable_Category] = DataSet[Variable_Category].fillna(0)
    
        DataSet[Category] = DataSet[Category].fillna(0)
        # DataSet['Action Element'] = DataSet['Action Element'].fillna(0)

        if Save:
            DataSet.to_csv(f'Traditional_Threshold_{len(Quantiles_Value)-1}^{len(Feature)}_groups.csv', index=False)
        
        return DataSet

    # Tradition_Encoding 呼叫使用轉成十進制
    def convert_to_decimal(self, Number, Base):
        decimal = 0
        power = 0
        while Number > 0:
            digit = Number % 10
            decimal += digit * (Base ** power)
            Number //= 10
            power += 1
        return decimal

    # 傳統閥值編碼並儲存檔案
    def Tradition_Encoding(self, DataSet, Feature, Save):
        for num in range(len(DataSet)):
            combined_string = ''.join(DataSet[Feature].iloc[num].astype(int).astype(str))
            converted_list = [int(char) for char in combined_string]
            max_value = max(converted_list)

            base = int(max_value) + 1
            decimal_number = self.convert_to_decimal(int(combined_string), base)
            DataSet['Action Element'].iloc[num] = decimal_number


        # 獲取當前日期和時間
        now = datetime.now()
        date = now.strftime('%Y%m%d')  # 格式化日期為YYYYMMDD
        time = now.strftime('%H%M')  # 格式化時間為HHMM

        if Save:
            DataSet.to_csv(f'{date}_Tradition_Encoding_{time}.csv', index=False)

        # 計算各類別的數量
        category_counts = DataSet['Action Element'].value_counts()

        # 繪製直方圖
        plt.bar(category_counts.index, category_counts.values)

        # 設定標題和軸標籤
        plt.title('Category Counts')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.savefig(f'{date}_plot_Category_Counts_{time}.png', dpi=300, bbox_inches='tight')
        plt.show()  # 若不需要顯示圖形，請取消註解此行
        plt.close()

        return DataSet   
 
    # 輸入資料含蓋量，取出前K個類別
    def Tradition_Find_Top_K(self, DataSet, Target_Percentage):
        # 計算各類別的數量
        category_counts = DataSet['Action Element'].value_counts()

        # 繪製直方圖
        plt.bar(category_counts.index, category_counts.values)

        # 設定標題和軸標籤
        plt.title('Category Counts')
        plt.xlabel('Category')
        plt.ylabel('Count')

        # 顯示圖形
        plt.show()


        # 根據數量由高到低排序
        category_counts = category_counts.sort_values(ascending=False)

        # 計算數量百分比
        category_percentages = category_counts / len(DataSet) * 100

        # 計算累積百分比
        category_cumulative_percentages = category_percentages.cumsum()

        # 建立 DataFrame
        category_stats = pd.DataFrame({'Count': category_counts, 'Percentage': category_percentages, 'Cumulative Percentage': category_cumulative_percentages})

        # 找到累積百分比達到目標百分比的資料
        filtered_data = category_stats[category_stats['Cumulative Percentage'] <= Target_Percentage]

        print(f'Data with cumulative percentage up to {Target_Percentage}%:')
        print(f'{filtered_data}\nTop K：{len(filtered_data)}')

        return category_stats

    # 繪製動作元素所組成之動作
    def Plot_Action_Cluter(self, DataSet, Action1, Action2, Feature, Cluster, Length, Save):    
        colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'cyan', 4: 'yellow', 5: 'magenta', 6: 'black', 7: 'white', 8: 'orange', 9: 'purple', 10: 'brown'}

        DataSet_Action1 = DataSet[DataSet['Action'] == Action1][:Length]
        DataSet_Action2 = DataSet[DataSet['Action'] == Action2][:Length]
    
        # 設定 x 軸長度
        DataSet_Action1_Length  = np.arange(len(DataSet_Action1))
        DataSet_Action2_Length = np.arange(len(DataSet_Action2))


        for i in range(Cluster):
            plt.scatter([], [], c=colors[i], label=f"Action Element {i}")

        # 點
        plt.scatter(DataSet_Action1_Length,  DataSet_Action1[Feature], c=[colors[x] for x in DataSet_Action1 ['Action Element']], zorder=2)

        # 線 
        plt.plot(DataSet_Action1_Length, DataSet_Action1[Feature][DataSet_Action1['Action'] == Action1], c='LightBlue' , label=Action1, linewidth=10, zorder=1)    

        plt.scatter(DataSet_Action2_Length, DataSet_Action2[Feature], c=[colors[x] for x in DataSet_Action2['Action Element']], zorder=2)

        plt.plot(DataSet_Action2_Length, DataSet_Action2[Feature][DataSet_Action2['Action'] == Action2], c='LightGreen' , label=Action2, linewidth=10, zorder=1)

        plt.legend(loc='best',bbox_to_anchor=(1.55, 1))
        plt.title(f'{Feature}\n {Action1} vs {Action2}')

        plt.xlabel('Time Step')
        plt.ylabel('Value')
        

        if Save:
            plt.savefig(f'{Feature} {Action1} vs {Action2}（best）.png', bbox_inches='tight')

        plt.show()

    # 繪製駕駛行為軌跡
    def Plot_Action_Track(self, DataSet, Step_Column_Name, Slice_size, Save):

        start_time = time.time()

        DataSet = DataSet.fillna('Unlabeled')

        # 計算需要切割的次數
        num_slices = math.ceil(len(DataSet) / Slice_size)

        # 動態計算切割範圍
        slices = []
        for i in range(num_slices):
            start = i * Slice_size
            end = min((i + 1) * Slice_size, len(DataSet))
            slices.append((start, end))

        # 設定顏色映射
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'cyan'] 

        # 繪製散點圖
        for i, (start, end) in enumerate(slices):
            # 切割資料
            Test_Data_slice = DataSet.iloc[start:end]
            Test_Data_slice.reset_index(drop=True, inplace=True)

            # 繪製第一張圖 (Test_Data_slice[Step_Column_Name])
            plt.figure(figsize=(12, 8))
            for j, condition in enumerate(['Go Straight', 'Idle', 'Turn Right', 'Turn Left', 'Two-Stage Left', 'U-turn', 'Unlabeled']):
                condition_points = [idx for idx, val in enumerate(Test_Data_slice[Step_Column_Name]) if val == condition]
                plt.scatter(Test_Data_slice.index[condition_points], Test_Data_slice['Z-axis Angular Velocity'][condition_points],
                            color=colors[j], label=condition, alpha=0.5)

            plt.title(f'{i} Slice - {Step_Column_Name}')
            plt.xlabel('Time Step')
            plt.ylabel('Z-axis Angular Velocity')
            plt.xticks(range(0, len(Test_Data_slice)+1, 500))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            if Save:
                plt.savefig(f'plot_{i+1}_step.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

            # 繪製第二張圖 (Test_Data_slice['Action'])
            plt.figure(figsize=(12, 8))
            for j, condition in enumerate(['Go Straight', 'Idle', 'Turn Right', 'Turn Left', 'Two-Stage Left', 'U-turn', 'Unlabeled']):
                condition_points = [idx for idx, val in enumerate(Test_Data_slice['Action']) if val == condition]
                plt.scatter(Test_Data_slice.index[condition_points], Test_Data_slice['Z-axis Angular Velocity'][condition_points],
                            color=colors[j], label=condition, alpha=0.5)

            plt.title(f'{i} Slice - Action')
            plt.xlabel('Time Step')
            plt.ylabel('Z-axis Angular Velocity')
            plt.xticks(range(0, len(Test_Data_slice)+1, 500))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            if Save:            
                plt.savefig(f'plot_{i+1}_action.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        # 計算執行時間
        end_time = time.time()
        execution_time = end_time - start_time
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = int(execution_time % 60)
        print(f"繪製動作序列所花費時間：{hours}小時{minutes}分鐘{seconds}秒")

        return

    # 影片中標記換算時間使用
    def Calculating_Time(self, Video_Ecu_Time, Video_Mark_Time, Real_Ecu_Time):


        Video_Ecu_Time = Video_Ecu_Time.split(':', 4)
        Hours_1   = int(Video_Ecu_Time[0])
        Minutes_1 = int(Video_Ecu_Time[1])
        Seconds_1 = int(Video_Ecu_Time[2])
        Frames_1  = int(Video_Ecu_Time[3])

        Video_Mark_Time = Video_Mark_Time.split(':', 4)
        Hours_2   = int(Video_Mark_Time[0])
        Minutes_2 = int(Video_Mark_Time[1])
        Seconds_2 = int(Video_Mark_Time[2])
        Frames_2  = int(Video_Mark_Time[3])

        Diff_Hours   = Hours_2   - Hours_1
        Diff_Minutes = Minutes_2 - Minutes_1
        Diff_Seconds = Seconds_2 - Seconds_1

        if Frames_2 >= Frames_1:
            Diff_Frames = Frames_2 - Frames_1
        else:
            Diff_Seconds = Diff_Seconds - 1 
            Diff_Frames = Frames_2 - Frames_1 + 25
    

        Real_Ecu_Time = datetime.strptime(Real_Ecu_Time, "%H:%M:%S")
        Real_Ecu_Time = Real_Ecu_Time.strftime('%H:%M:%S')
        Real_Ecu_Time = datetime.strptime(Real_Ecu_Time, "%H:%M:%S")

        real_diff_time = timedelta(hours=Diff_Hours, minutes=Diff_Minutes, seconds=Diff_Seconds)#, milliseconds=Diff_Frames*40
        real_mark_time = Real_Ecu_Time + real_diff_time

        real_mark_time = real_mark_time.strftime('%H:%M:%S')


        real_mark_time = str(real_mark_time)
        real_mark_time = real_mark_time.split(':', 3)
        Hours   = int(real_mark_time[0])
        Minutes = int(real_mark_time[1])
        Seconds = float(real_mark_time[2])


        real_time_error = 0.68
        frames_time_error = Diff_Frames*0.04

        Seconds = Seconds + real_time_error + frames_time_error

        real_mark_time = str(Hours)+':'+str(Minutes)+':'+str(Seconds)
    
        return real_mark_time

    # 簡易平滑資料使用
    def Convolve(self, Data_Set, File_Name, Data, Window_Size, Save):


        """

        Function: This function performs smoothing on the input feature data by replacing the original data with the average value within a window of size Window_Size. It plots the data of turning left and right in different colors on the same graph, and saves the result as a file.
    
        Parameters:

            Data_Set: pandas DataFrame, contains the feature and label data of the dataset

            File_Name: string, used to name the saved image file

            Data: numpy array, the numerical values of feature data

            Window_Size: integer, the size of the smoothing window


        """        
        Smoothed_Data = np.convolve(Data, np.ones(Window_Size)/Window_Size, mode='same')
        Data = Smoothed_Data
        
        x1 = np.arange(len(Data[Data_Set['Action']== 'left']))
        x2 = np.arange(len(Data[Data_Set['Action']== 'right']))

        plt.figure()
        plt.plot(x1, Data[Data_Set['Action']== 'left'] , c='r' , label='Turn left')
        plt.plot(x2, Data[Data_Set['Action'] == 'right'], c='g', label='Turn right')
        plt.legend(loc='lower right')
        
        if Save:
            plt.savefig(File_Name+'_Window_Size_'+str(Window_Size)+'.png')
        
        return 

    def Data_Smoothing(self, DataSet, Feature, Method, Slice_Size):
        if Method == 'Kalman':
            kf = KalmanFilter(dim_x=len(Feature), dim_z=len(Feature))
            kf.F = np.eye(len(Feature))
            kf.H = np.eye(len(Feature))
            kf.Q = np.eye(len(Feature)) * 0.0001
            kf.R = np.eye(len(Feature)) * 0.001
            kf.x = np.zeros((len(Feature), 1))
            kf.P = np.eye(len(Feature))
            Data = np.array(DataSet[Feature].values)
            filtered_data = np.zeros_like(Data)

            for i in range(Data.shape[0]):
                measurement = Data[i, :].reshape(len(Feature), 1)
                kf.predict()
                kf.update(measurement)
                filtered_data[i, :] = kf.x[:, 0]

            for i, column in enumerate(Feature):
                DataSet[column] = filtered_data[:, i]


            num_slices = math.ceil(len(DataSet) / Slice_Size)
            slices = []
            for i in range(num_slices):
                start = i * Slice_Size
                end = min((i + 1) * Slice_Size, len(DataSet))
                slices.append((start, end))

            for i, (start, end) in enumerate(slices):
                plt.figure()
                Test_Data_slice = DataSet.iloc[start:end]
                Test_Data_slice.reset_index(drop=True, inplace=True)
                Feature_Value = Test_Data_slice[Feature]
                plt.title(f'{i} Slice - {Method}')
                plt.xlabel('Time Step')
                plt.ylabel(f'{Feature}')
                plt.plot(range(len(Test_Data_slice)), Feature_Value, label=f'{Feature}')

            return DataSet

        if Method == 'Window_Average':
            Window_Size = 30
            Smoothed_Data = np.convolve(DataSet[Feature], np.ones(Window_Size)/Window_Size, mode='same')
            DataSet[Feature] = Smoothed_Data


            num_slices = math.ceil(len(DataSet) / Slice_Size)
            slices = []
            for i in range(num_slices):
                start = i * Slice_Size
                end = min((i + 1) * Slice_Size, len(DataSet))
                slices.append((start, end))

            for i, (start, end) in enumerate(slices):
                plt.figure()
                Test_Data_slice = DataSet.iloc[start:end]
                Test_Data_slice.reset_index(drop=True, inplace=True)
                Feature_Value = Test_Data_slice[Feature]
                plt.title(f'{i} Slice - {Method}')
                plt.xlabel('Time Step')
                plt.ylabel(f'{Feature}')
                plt.plot(range(len(Test_Data_slice)), Feature_Value, label=f'{Feature}')

            return Smoothed_Data
        
        if Method == 'None':
            num_slices = math.ceil(len(DataSet) / Slice_Size)
            slices = []
            for i in range(num_slices):
                start = i * Slice_Size
                end = min((i + 1) * Slice_Size, len(DataSet))
                slices.append((start, end))

            for i, (start, end) in enumerate(slices):
                plt.figure()
                Test_Data_slice = DataSet.iloc[start:end]
                Test_Data_slice.reset_index(drop=True, inplace=True)
                Feature_Value = Test_Data_slice[Feature]
                plt.title(f'{i} Slice - {Method}')
                plt.xlabel('Time Step')
                plt.ylabel(f'{Feature}')
                plt.plot(range(len(Test_Data_slice)), Feature_Value, label=f'{Feature}')

            return DataSet
        

################################################################################################################################

# 參考網址: https://github.com/rpgomez/vomm

class ppm:
    """ This class implements the "predict by partial match" algorithm. """

    def __init__(self):
        """ Not much to do here. """

    def generate_fast_lookup(self):
        """Takes the pdf_dict dictionary and computes a faster lookup
        dictionary mapping suffix s to its longer contexts xs.  I need
        this to speed up computing the probability of logpdf for an
        observed sequence
        """

        # I want to create a fast look up of context s -> xs
        # So scoring a sequence is faster.

        context_by_length  = dict([(k,[]) for k in range(self.d+1) ])

        for x in self.pdf_dict.keys():
            context_by_length[len(x)].append(x)

        # Now lets generate a dictionary look up context s -> possible context xs.
        self.context_child = {}

        for k in range(self.d):
            for x in context_by_length[k]:
                self.context_child[x] = [ y for y in context_by_length[k+1] if y[1:] == x ]

        for x in context_by_length[self.d]:
            self.context_child[x] = []

    def fit(self,training_data, d=4, alphabet_size = None):
        """
        This is the method to call to fit the model to the data.
        training_data should be a sequence of symbols represented by
        integers 0 <= x < alphabet size.

        d specifies the largest context sequence we're willing to consider.

        alphabet_size specifies the number of distinct symbols that
        can be possibly observed. If not specified, the alphabet_size
        will be inferred from the training data.
        """

        if alphabet_size == None:
            alphabet_size = max(training_data) + 1

        self.alphabet_size = alphabet_size
        self.d = d

        counts = self.count_occurrences(tuple(training_data),d=self.d,
                                   alphabet_size = self.alphabet_size)

        self.pdf_dict = self.compute_ppm_probability(counts)

        self.logpdf_dict = dict([(x,np.log(self.pdf_dict[x])) for x in self.pdf_dict.keys()])

        # For faster look up  when computing logpdf(observed data).
        self.generate_fast_lookup()

        return

    def logpdf(self,observed_data):
        """Call this method after using fitting the model to compute the log of
        the probability of an observed sequence of data.

        observed_data should be a sequence of symbols represented by
        integers 0 <= x < alphabet_size. """

        temp = tuple(observed_data)
        # start with the null context and work my way forward.

        logprob = 0.0
        for t in range(len(temp)):
            chunk = temp[max(t-self.d,0):t]
            sigma = temp[t]
            context = self.find_largest_context(chunk,self.context_child,self.d)
            logprob += self.logpdf_dict[context][sigma]

        return logprob

    def generate_data(self,prefix=None, length=200):
        """Generates data from the fitted model.

        The length parameter determines how many symbols to generate.

        prefix is an optional sequence of symbols to be appended to,
        in other words, the prefix sequence is treated as a set of
        symbols that were previously "generated" that are going to be
        appended by an additional "length" number of symbols.

        The default value of None indicates that no such prefix
        exists. We're going to be generating symbols starting from the
        null context.

        It returns the generated data as an array of symbols
        represented as integers 0 <=x < alphabet_size.

        """

        if prefix != None:
            new_data = np.zeros(len(prefix) + length,dtype=int)
            new_data[:len(prefix)] = prefix
            start = len(prefix)
        else:
            new_data = np.zeros(length,dtype=int)
            start = 0

        for t in range(start,len(new_data)):
            chunk = tuple(new_data[max(t-self.d,0):t])
            context = self.find_largest_context(chunk,self.context_child,self.d)
            new_symbol = np.random.choice(self.alphabet_size,p=self.pdf_dict[context])
            new_data[t] = new_symbol

        return new_data[start:]

    def __str__(self):
        """ Implements a string representation to return the parameters of this model. """

        return "\n".join(["alphabet size: %d" % self.alphabet_size,
                          "context length d: %d" % self.d,
                          "Size of model: %d" % len(self.pdf_dict),
                          "Frequency threshold: %f" % self.freq_threshold,
                          "Meaning threshold: %f" % self.meaning_threshold,
                          "Kullback-Leibler threshold: %f" % self.kl_threshold])

    def find_contexts(self, training_data, d= 4):
        """
        Takes a sequence of observed symbols and finds all contexts of
        length at most d.

        training_data represents the sequence of observed symbols as a
        tuple of integers x where 0 <= x < alphabet size.

        Contexts of length k are represented as tuples of integers of length k.

        This function returns the observed contexts as a set.
        """

        contexts = set()

        N = len(training_data)

        for k in range(1,d+1):
            contexts = contexts.union([training_data[t:t+k] for t in range(N-k+1)])

        return contexts

    def count_occurrences(self, training_data, d=4, alphabet_size = None):
        """
        Counts the number of occurrences of s\sigma where s is a context
        and \sigma is the symbol that immediately follows s in the
        training data.

        training_data represents the sequence of observed symbols as a
        tuple of integers x where 0 <= x < alphabet size.

        d determines the longest context to consider.

        alphabet_size determines the number of possible observable
        distinct symbols. If not set, the function determines it from the
        training data.

        The function returns the counts as a dictionary with key context s
        and value the counts array.

        """

        contexts = self.find_contexts(training_data, d = d)

        if alphabet_size == None:
            alphabet_size = max(training_data) + 1

        counts = dict([(x, np.zeros(alphabet_size,dtype=int)) for x in contexts])

        # Include the null context as well.
        counts[()] = np.bincount(training_data,minlength = alphabet_size)

        N = len(training_data)
        for k in range(1,d+1):
            for t in range(N-k):
                s = training_data[t:t+k]
                sigma = training_data[t+k]
                counts[s][sigma]  += 1

        return counts

    def compute_ppm_probability(self, counts):
        d = max([len(x) for x in counts.keys()])
        alphabet_size = counts[()].shape[0]
        pdf = dict([(x, np.zeros(alphabet_size, dtype=np.float64)) for x in counts.keys()])

        byk = [[] for k in range(d + 1)]
        for x in counts.keys():
            byk[len(x)].append(x)

        pdf[()] = (counts[()] + 1.0) / (counts[()].sum() + alphabet_size)

        for k in range(1, d + 1):
            for x in byk[k]:
                sigma_observed = np.argwhere(counts[x] > 0).reshape(-1)
                alphabet_obs_size = len(sigma_observed)
                sigma_escaped = np.argwhere(counts[x] == 0).reshape(-1)
                denominator = alphabet_obs_size + counts[x].sum()
                x_1 = x[1:]

                if alphabet_obs_size > 0:
                    escape_factor = alphabet_obs_size * 1.0 / denominator
                else:
                    escape_factor = 1.0

                pdf[x][sigma_observed] = counts[x][sigma_observed] * 1.0 / denominator

                if len(sigma_escaped) > 0:
                    pdf[x][sigma_escaped] = escape_factor * pdf[x_1][sigma_escaped] / pdf[x_1][sigma_escaped].sum(axis=0)

                pdf[x] = pdf[x] / pdf[x].sum()

        return pdf

    def find_largest_context(self, chunk,fast_lookup_table,d):
        """Find the largest context that matches the observed chunk of symbols
        and returns it.

        chunk is a sequence of symbols represented as a tuple of integers 0 <= x < alphabet size.

        fast_lookup_table is a dictionary with key context s and value the
        list of contexts which are of the form xs.

        d is the size of the largest context in the set of contexts.
        """

        if len(chunk) == 0:
            return ()

        current_context = ()
        end = len(chunk)
        start = end

        while chunk[start:end] == current_context:
            start -= 1
            if start < 0 or start < end - d:
                break

            if chunk[start:end] in fast_lookup_table[current_context]:
                current_context = chunk[start:end]
            else:
                break

        return current_context