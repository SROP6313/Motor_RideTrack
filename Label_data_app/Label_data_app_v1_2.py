# Make sure use the "label_data_app" conda environment

import tkinter as tk
from tkinter import filedialog, ttk
import tkinter.messagebox as messagebox
import cv2
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime, timedelta

def convert_time(time_str):
    return pd.to_datetime(time_str, format='mixed')
    # return pd.to_datetime(time_str, format='%Y-%m-%d %H:%M:%S.%f%z')  # Or this option

class VideoPlayer:
    def __init__(self, root, video_path, csv_file_path, start_time):
        self.root = root
        self.video_path = video_path
        self.csv_file_path = csv_file_path
        self.cap = cv2.VideoCapture(video_path)
        self.start_time = datetime.fromisoformat(start_time)
        self.current_frame_time = self.start_time
        self.paused = True
        self.action_start_time = None
        self.current_action = None

        self.df = pd.read_csv(self.csv_file_path)
        # Convert 'Absolute Time' column to datetime
        self.df['Absolute Time'] = self.df['Absolute Time'].apply(convert_time)

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
        actions = ["Go Straight", "Idle", "Turn Left", "Turn Right", "Hook Turn", "U-turn"]
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
        self.pause()

    def update_csv(self):
        start_time = self.action_start_time
        end_time = self.action_end_time
        action = self.current_action
        
        print(f"Marking rows from {start_time} to {end_time} as {action}")
        
        mask = (self.df['Absolute Time'] >= start_time) & (self.df['Absolute Time'] <= end_time)
        num_rows = mask.sum()
        print(f"{num_rows} rows will be marked")
        
        self.df.loc[mask, 'Action'] = action
        self.df.to_csv(self.csv_file_path, index=False)
        
        print("CSV file saved successfully")

    def on_progress_change(self, value):
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(value) / 100 * self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.current_frame_time = self.start_time + timedelta(seconds=float(value) / 100 * self.total_duration)

    @property
    def total_duration(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)

    def save_and_exit(self):
        self.df.to_csv(self.csv_file_path, index=False)
        self.root.destroy()

def main():
    root = tk.Tk()
    root.title("KDD RideTrack Label System App v1.2")

    # Prompt user to enter start time
    start_time = None
    while start_time is None:
        start_time_str = tk.simpledialog.askstring("Start Time", "Please enter the time when THE VIDEO STARTS (YYYY-MM-DD HH:MM:SS).")
        if start_time_str is None:
            break
        else:
            try:
                start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                start_time = start_time.isoformat(timespec='milliseconds') + "+08:00"
            except ValueError:
                messagebox.showerror("Invalid Format", f"Invalid start time format: {start_time_str}\nPlease try again.")
                start_time = None

    if start_time != None:
        # Load video file
        video_path = filedialog.askopenfilename(title='Select Video File')
        csv_file_path = filedialog.askopenfilename(title='Select IMU CSV File')

        player = VideoPlayer(root, video_path, csv_file_path, start_time)
        root.mainloop()

if __name__ == "__main__":
    main()