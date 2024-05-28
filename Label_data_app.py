# Make sure use the "label_data_app" conda environment

import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime, timedelta

def convert_time(time_str):
    return pd.to_datetime(time_str, format='mixed')
    # return pd.to_datetime(time_str, format='%Y-%m-%d %H:%M:%S.%f%z')  # Or this option

# Load CSV file
csv_file_path = './30Hz20240125_eric.csv'  # The PATH of the IMU data needed to be labeled !!!!!!!!
df = pd.read_csv(csv_file_path)

# Convert 'Absolute Time' column to datetime
df['Absolute Time'] = df['Absolute Time'].apply(convert_time)

class VideoPlayer:
    def __init__(self, root, video_path, start_time):
        self.root = root
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.start_time = datetime.fromisoformat(start_time)
        self.current_frame_time = self.start_time
        self.paused = True
        self.action_start_time = None
        self.current_action = None

        self.create_ui()
        self.update()

    def create_ui(self):
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.progress_label = tk.Label(self.root, text="Time: 00:00:00")
        self.progress_label.pack()

        self.progress = ttk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_progress_change)
        self.progress.pack(fill=tk.X)

        self.play_button = tk.Button(self.root, text="Play", command=self.play)
        self.play_button.pack(side=tk.LEFT)

        self.pause_button = tk.Button(self.root, text="Pause", command=self.pause)
        self.pause_button.pack(side=tk.LEFT)

        self.action_buttons = []
        actions = ["Go Straight", "Idle", "Turn Left", "Turn Right", "Two-Stage Left", "U-turn"]
        for action in actions:
            btn = tk.Button(self.root, text=action, command=lambda a=action: self.mark_action(a))
            btn.pack(side=tk.LEFT)
            self.action_buttons.append(btn)

        self.end_button = tk.Button(self.root, text="End", command=self.end_action, state=tk.DISABLED)
        self.end_button.pack(side=tk.LEFT)

        self.save_and_exit_button = tk.Button(self.root, text="Save and Exit", command=self.save_and_exit)
        self.save_and_exit_button.pack(side=tk.LEFT)

    def update(self):
        if not self.paused:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame_time += timedelta(seconds=1/self.cap.get(cv2.CAP_PROP_FPS))
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

                    elapsed_time = (self.current_frame_time - self.start_time).total_seconds()
                    self.progress_label.config(text=f"Time: {str(self.current_frame_time)}")
                    self.progress.set(elapsed_time / self.total_duration * 100)

            self.root.after(10, self.update)

    def play(self):
        self.paused = False
        self.update()

    def pause(self):
        self.paused = True

    def mark_action(self, action):
        self.current_action = action
        self.action_start_time = self.current_frame_time
        self.end_button.config(state=tk.NORMAL)
        self.pause()

    def end_action(self):
        self.action_end_time = self.current_frame_time
        self.update_csv()
        self.end_button.config(state=tk.DISABLED)

    def update_csv(self):
        start_time = self.action_start_time
        end_time = self.action_end_time
        action = self.current_action
        
        print(f"Marking rows from {start_time} to {end_time} as {action}")
        
        mask = (df['Absolute Time'] >= start_time) & (df['Absolute Time'] <= end_time)
        num_rows = mask.sum()
        print(f"{num_rows} rows will be marked")
        
        df.loc[mask, 'Action'] = action
        df.to_csv(csv_file_path, index=False)
        
        print("CSV file saved successfully")

    def on_progress_change(self, value):
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(value) / 100 * self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.current_frame_time = self.start_time + timedelta(seconds=float(value) / 100 * self.total_duration)

    @property
    def total_duration(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)

    def save_and_exit(self):
        df.to_csv(csv_file_path, index=False)
        self.root.destroy()

def main():
    root = tk.Tk()
    root.title("KDD RideTrack Label System App v1.0")

    # Load video file
    video_path = filedialog.askopenfilename()
    start_time = "2024-01-25 12:49:53.056000+08:00"

    player = VideoPlayer(root, video_path, start_time)
    root.mainloop()

if __name__ == "__main__":
    main()