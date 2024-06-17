# Make sure use the "label_data_app" conda environment

import tkinter as tk
from tkinter import filedialog, ttk, IntVar
import tkinter.messagebox as messagebox
import cv2
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

def convert_time(time_str):
    return pd.to_datetime(time_str, format='mixed')

def resize_frame(frame, width=720, height=540):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

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
        self.playback_speed = 1.0
        self.plot_type = IntVar()
        self.plot_type.set(0)  # 默認為 "Angle"

        self.df = pd.read_csv(self.csv_file_path)
        self.df['Absolute Time'] = self.df['Absolute Time'].apply(convert_time)

        self.create_plot()
        self.create_ui()
        self.update()

    def create_ui(self):
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.video_label = tk.Label(left_frame)
        self.video_label.pack(side=tk.TOP)

        self.progress_label = tk.Label(left_frame, text="Time: 00:00:00")
        self.progress_label.pack(side=tk.TOP)

        self.progress = ttk.Scale(left_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_progress_change)
        self.progress.pack(fill=tk.X, side=tk.TOP)

        button_frame = tk.Frame(left_frame)
        button_frame.pack(side=tk.TOP)

        top_row_frame = tk.Frame(button_frame)
        top_row_frame.pack(side=tk.TOP)
        self.play_button = tk.Button(top_row_frame, text="Play", command=self.play)
        self.play_button.pack(side=tk.LEFT)
        self.pause_button = tk.Button(top_row_frame, text="Pause", command=self.pause)
        self.pause_button.pack(side=tk.LEFT)

        speed_frame = tk.Frame(button_frame)
        speed_frame.pack(side=tk.TOP)
        tk.Label(speed_frame, text="Playback Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="1.0x")
        self.speed_scale = ttk.Scale(speed_frame, from_=0, to=2, orient=tk.HORIZONTAL, command=self.update_speed)
        self.speed_scale.set(0)  # Default to 1.0x (index 0)
        self.speed_scale.pack(side=tk.LEFT)
        tk.Label(speed_frame, textvariable=self.speed_var).pack(side=tk.LEFT)

        middle_row_frame = tk.Frame(button_frame)
        middle_row_frame.pack(side=tk.TOP)
        self.action_buttons = []
        actions = ["Go Straight", "Idle", "Turn Left", "Turn Right", "Two-Stage Left", "U-turn"]
        for action in actions:
            btn = tk.Button(middle_row_frame, text=action, command=lambda a=action: self.mark_action(a))
            btn.pack(side=tk.LEFT)
            self.action_buttons.append(btn)

        self.end_button = tk.Button(middle_row_frame, text="End", command=self.end_action, state=tk.DISABLED)
        self.end_button.pack(side=tk.LEFT)

        bottom_row_frame = tk.Frame(button_frame)
        bottom_row_frame.pack(side=tk.TOP)
        self.save_and_exit_button = tk.Button(bottom_row_frame, text="Save and Exit", command=self.save_and_exit)
        self.save_and_exit_button.pack(side=tk.LEFT)
        
        plot_type_frame = tk.Frame(button_frame)
        plot_type_frame.pack(side=tk.TOP, pady=10)
        tk.Radiobutton(plot_type_frame, text="Angle", variable=self.plot_type, value=0, command=self.update_plot).pack(side=tk.LEFT)
        tk.Radiobutton(plot_type_frame, text="Acceleration", variable=self.plot_type, value=1, command=self.update_plot).pack(side=tk.LEFT)
        tk.Radiobutton(plot_type_frame, text="Angular Velocity", variable=self.plot_type, value=2, command=self.update_plot).pack(side=tk.LEFT)

        self.plot_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def update_speed(self, event):
        speed_options = [1.0, 2.0, 4.0]
        index = int(float(self.speed_scale.get()))
        self.playback_speed = speed_options[index]
        self.speed_var.set(f"{self.playback_speed}x")

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.line_x, = self.ax.plot([], [], lw=1, label='X')
        self.line_y, = self.ax.plot([], [], lw=1, label='Y')
        self.line_z, = self.ax.plot([], [], lw=1, label='Z')
        self.ax.set_xlabel('Time')
        self.ax.legend()

    def update(self):
        if not self.paused:
            if self.cap.isOpened():
                for _ in range(int(self.playback_speed)):
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                if ret:
                    self.current_frame_time += timedelta(seconds=1/self.cap.get(cv2.CAP_PROP_FPS) * self.playback_speed)
                    frame = resize_frame(frame)  # Resize the frame
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

                    elapsed_time = (self.current_frame_time - self.start_time).total_seconds()
                    self.progress_label.config(text=f"Time: {str(self.current_frame_time)}")
                    self.progress.set(elapsed_time / self.total_duration * 100)

                    self.update_plot()

            self.root.after(10, self.update)

    def play(self):
        self.paused = False
        self.update()
        self.play_button.config(state=tk.DISABLED)

    def pause(self):
        self.paused = True
        self.play_button.config(state=tk.NORMAL)

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

    def update_plot(self):
        curr_time = self.current_frame_time
        mask = (self.df['Absolute Time'] <= curr_time)
        times = (self.df.loc[mask, 'Absolute Time'] - self.start_time) / pd.Timedelta(seconds=1)

        plot_type = self.plot_type.get()
        if plot_type == 0:  # Angle
            x_data = self.df.loc[mask, 'Pitch (deg)']
            y_data = self.df.loc[mask, 'Roll (deg)']
            z_data = self.df.loc[mask, 'Yaw (deg)']
            self.ax.set_ylabel('Angle (deg)')
        elif plot_type == 1:  # Acceleration
            x_data = self.df.loc[mask, 'X-axis Acceleration']
            y_data = self.df.loc[mask, 'Y-axis Acceleration']
            z_data = self.df.loc[mask, 'Z-axis Acceleration']
            self.ax.set_ylabel('Acceleration (m/s^2)')
        else:  # Angular Velocity
            x_data = self.df.loc[mask, 'X-axis Angular Velocity']
            y_data = self.df.loc[mask, 'Y-axis Angular Velocity']
            z_data = self.df.loc[mask, 'Z-axis Angular Velocity']
            self.ax.set_ylabel('Angular Velocity (deg/s)')

        self.line_x.set_data(times, x_data)
        self.line_y.set_data(times, y_data)
        self.line_z.set_data(times, z_data)

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.plot_canvas.draw()

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
    root.title("KDD RideTrack Label System App v2.1")

    start_time = None
    while start_time is None:
        start_time_str = tk.simpledialog.askstring("Start Time", "Please enter the time when THE VIDEO STARTS RECORDING (YYYY-MM-DD HH:MM:SS).")
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
        video_path = filedialog.askopenfilename(title='Select Video File')
        csv_file_path = filedialog.askopenfilename(title='Select IMU CSV File')

        player = VideoPlayer(root, video_path, csv_file_path, start_time)
        root.state('zoomed')
        root.mainloop()

if __name__ == "__main__":
    main()