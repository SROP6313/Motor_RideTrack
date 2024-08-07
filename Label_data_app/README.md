## Label Data App .EXE file Download
[Download here](https://github.com/SROP6313/Motor_RideTrack/releases)

## Label Data App Python Environment Setup
#### Use Anaconda .yml file
```bash
conda env create -f label_data_app.yml
```

## Usage (for v3.0)
#### 1. Just excute the EXE file or in terminal run
```
python Label_data_app_v3.py
```
#### 2. Enter the time when the video start recording (ex. 2024-01-25 12:49:53)
![image](https://github.com/SROP6313/Motor_RideTrack/assets/103128273/e53ac4d7-a0a9-4fc0-b4a3-ae5bfcb8984d)

#### 3. Select video file
![image](https://github.com/SROP6313/Motor_RideTrack/assets/103128273/d83e7ab9-76a6-4485-afd2-24301284c4d4)

#### 4. Select IMU csv file
![image](https://github.com/SROP6313/Motor_RideTrack/assets/103128273/04eafe1d-9d85-4beb-bbb8-97938b3087d5)

#### 5. Click `▶` button to play the video you selected
![image](https://github.com/user-attachments/assets/866b0df0-2a4c-4276-9c65-e898529d2688)

#### 6. Mark the behavior at any time you want
* Click anyone of the behavior buttons: `Go Straight`, `Idle`, `Turn Left`, `Turn Righ`, `Hook Turn`, `U-turn`.
* The video will automatically pause. Just click `▶` to continue.
#### 7. End the behavior at any time you want
* Click the `End` button to end the behavior you clicked.
* The video will automatically pause. Just click `▶` to continue.
* The marked action should be saved in the csv file.
#### 8. Save and exit :+1:
