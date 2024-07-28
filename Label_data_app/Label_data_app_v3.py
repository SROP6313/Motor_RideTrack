# Make sure use the "label_data_app" conda environment

import tkinter as tk
from tkinter import filedialog, ttk, IntVar, BooleanVar
import tkinter.messagebox as messagebox
import cv2
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import os
from tqdm import tqdm
import sys

def convert_time(time_str):
    return pd.to_datetime(time_str, format='mixed')

def create_low_quality_video(input_path, output_path, width=720, height=540):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    for _ in tqdm(range(total_frames), desc="Converting to low quality", unit="frames"):
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        out.write(resized_frame)

    cap.release()
    out.release()

class VideoPlayer:
    def __init__(self, root, video_path, csv_file_path, start_time):
        self.root = root
        self.original_video_path = video_path
        self.csv_file_path = csv_file_path
        self.start_time = datetime.fromisoformat(start_time)
        self.current_frame_time = self.start_time
        self.paused = True
        self.action_start_time = None
        self.current_action = None
        self.playback_speed = 1.0
        self.plot_type = IntVar()
        self.plot_type.set(0)  # 默認為 "Angle"
        self.show_background = BooleanVar()
        self.show_background.set(False)  # 默認不顯示背景區塊
        self.show_plot = BooleanVar()
        self.show_plot.set(True)  # Default to showing the plot
        sys.stdout = self  # Redirect print to this object

        self.df = pd.read_csv(self.csv_file_path)
        self.df['Absolute Time'] = self.df['Absolute Time'].apply(convert_time)

        if 'Action' not in self.df.columns:
            self.df['Action'] = ''  # Add empty 'Action' column if it doesn't exist

        self.window_size = timedelta(seconds=60)  # Default window size of 60 seconds
        self.window_start = self.start_time

        self.create_message_display()
        self.create_low_quality_video()
        self.cap = cv2.VideoCapture(self.low_quality_video_path)

        self.create_plot()
        self.create_ui()
        self.update()

    def create_low_quality_video(self):
        base, ext = os.path.splitext(self.original_video_path)
        self.low_quality_video_path = f"{base}_low_quality.mp4"
        
        if not self.original_video_path.endswith("_low_quality.mp4"):
            print("Creating low quality video...")
            create_low_quality_video(self.original_video_path, self.low_quality_video_path)
            print("Low quality video created successfully. Using it now.")
        else:
            print("Input video is already a low quality version. Using it directly.")
            self.low_quality_video_path = self.original_video_path

    def create_message_display(self):
        self.message_display = tk.Text(self.root, height=6, width=50)
        self.message_display.pack(side=tk.BOTTOM, fill=tk.X)
        self.message_display.config(state=tk.DISABLED)  # Make it read-only

    def create_ui(self):
        def play_pause_button_style(button):
            button.config(
                width=12, # 設定按鈕寬度
                height=1, # 設定按鈕高度
                bg='white', # 設定dad按鈕背景顏色
                fg='blue', # 設定按鈕文字顏色
                font=('Helvetica', 15, 'bold') # 設定按鈕文字字體和大小
            )

        def forward_backward_button_style(button):
            button.config(
                width=6,
                height=1,
                bg='white',
                fg='black',
                font=('Arial', 15, 'normal')
            )

        def action_button_style(button):
            button.config(
                width=9,
                height=1,
                bg='gold',
                fg='black',
                font=('Arial', 12, 'bold')
            )

        def end_button_style(button):
            button.config(
                width=5,
                height=1,
                bg='white',
                fg='red',
                font=('Arial', 12, 'bold')
            )

        def exit_button_style(button):
            button.config(
                width=15,
                height=1,
                bg='white',
                fg='black',
                font=('Arial', 12, 'bold')
            )

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
        self.play_button = tk.Button(top_row_frame, text="►", command=self.play)
        play_pause_button_style(self.play_button)
        self.play_button.pack(side=tk.LEFT)
        self.pause_button = tk.Button(top_row_frame, text="▌▌", command=self.pause)
        play_pause_button_style(self.pause_button)
        self.pause_button.pack(side=tk.LEFT)
        self.backward_button = tk.Button(top_row_frame, text="◀◀ 1s", command=self.backward_one_second)
        forward_backward_button_style(self.backward_button)
        self.backward_button.pack(side=tk.LEFT)
        self.forward_button = tk.Button(top_row_frame, text="1s ▶▶", command=self.forward_one_second)
        forward_backward_button_style(self.forward_button)
        self.forward_button.pack(side=tk.LEFT)        

        speed_frame = tk.Frame(button_frame)
        speed_frame.pack(side=tk.TOP, pady=5)
        tk.Label(speed_frame, text="Playback Speed: ", fg="black", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="1.0x")
        self.speed_scale = ttk.Scale(speed_frame, from_=0, to=2, orient=tk.HORIZONTAL, command=self.update_speed)
        self.speed_scale.set(0)  # Default to 1.0x (index 0)
        self.speed_scale.pack(side=tk.LEFT)
        tk.Label(speed_frame, textvariable=self.speed_var, fg="black", font=("Arial", 10, "bold")).pack(side=tk.LEFT)

        middle_row_frame = tk.Frame(button_frame)
        middle_row_frame.pack(side=tk.TOP)
        self.action_buttons = []
        actions = ["Go Straight", "Idle", "Turn Left", "Turn Right", "Hook Turn", "U-turn"]
        for action in actions:
            btn = tk.Button(middle_row_frame, text=action, command=lambda a=action: self.mark_action(a))
            action_button_style(btn)
            btn.pack(side=tk.LEFT)
            self.action_buttons.append(btn)

        self.end_button = tk.Button(middle_row_frame, text="End", command=self.end_action, state=tk.DISABLED)
        end_button_style(self.end_button)
        self.end_button.pack(side=tk.LEFT)

        bottom_row_frame = tk.Frame(button_frame)
        bottom_row_frame.pack(side=tk.TOP)
        self.save_and_exit_button = tk.Button(bottom_row_frame, text="Save and Exit", command=self.save_and_exit)
        exit_button_style(self.save_and_exit_button)
        self.save_and_exit_button.pack(side=tk.LEFT)
        
        # Add checkbox for plot visibility
        plot_frame = tk.Frame(button_frame, bg="oldlace")
        plot_frame.pack(side=tk.TOP, pady=15)
        tk.Checkbutton(
            plot_frame, 
            text="Show IMU Data Plot (Functions below will cause the app SLOWER !!)", 
            variable=self.show_plot, 
            command=self.toggle_plot_visibility,
            bg="lightyellow", fg="red", font=("Verdana", 12, "bold")
        ).pack(side=tk.TOP)

        plot_type_frame = tk.Frame(plot_frame)
        plot_type_frame.pack(side=tk.TOP, pady=5)
        tk.Radiobutton(plot_type_frame, text="Angle", font=("Arial", 12), bg="oldlace", variable=self.plot_type, value=0, command=self.update_plot).pack(side=tk.LEFT)
        tk.Radiobutton(plot_type_frame, text="Acceleration", font=("Arial", 12), bg="oldlace", variable=self.plot_type, value=1, command=self.update_plot).pack(side=tk.LEFT)
        tk.Radiobutton(plot_type_frame, text="Angular Velocity", font=("Arial", 12), bg="oldlace", variable=self.plot_type, value=2, command=self.update_plot).pack(side=tk.LEFT)

        # Add checkbox for background blocks
        background_frame = tk.Frame(plot_frame)
        background_frame.pack(side=tk.TOP, pady=5)
        tk.Checkbutton(background_frame, text="Show Action Label", bg="mistyrose", fg="black", font=("Arial", 12), variable=self.show_background, command=self.update_plot).pack(side=tk.LEFT)

        # Add sliders for window control
        window_control_frame = tk.Frame(plot_frame)
        window_control_frame.pack(side=tk.TOP, pady=5)

        tk.Label(window_control_frame, text="Data Plot Range (s):", fg="black", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.window_size_slider = ttk.Scale(window_control_frame, from_=10, to=300, orient=tk.HORIZONTAL, command=self.update_window_size)
        self.window_size_slider.set(60)  # Default to 60 seconds
        self.window_size_slider.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.plot_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def update_speed(self, event):
        speed_options = [1.0, 2.0, 4.0]
        index = int(float(self.speed_scale.get()))
        self.playback_speed = speed_options[index]
        self.speed_var.set(f"{self.playback_speed}x")

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        if self.show_plot.get():
            self.plot_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)    
                
        self.line_x, = self.ax.plot([], [], lw=1, label='X')
        self.line_y, = self.ax.plot([], [], lw=1, label='Y')
        self.line_z, = self.ax.plot([], [], lw=1, label='Z')
        self.ax.set_xlabel('Time')
        self.ax.legend()

    def forward_one_second(self):
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        new_frame = min(current_frame + int(fps), int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.current_frame_time += timedelta(seconds=1)
        self.update_frame()

    def backward_one_second(self):
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        new_frame = max(current_frame - int(fps), 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.current_frame_time -= timedelta(seconds=1)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            elapsed_time = (self.current_frame_time - self.start_time).total_seconds()
            self.progress_label.config(text=f"Time: {str(self.current_frame_time)}")
            self.progress.set(elapsed_time / self.total_duration * 100)

            self.update_plot()
    
    def update(self):
        if not self.paused:
            if self.cap.isOpened():
                for _ in range(int(self.playback_speed)):
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                if ret:
                    self.current_frame_time += timedelta(seconds=1/self.cap.get(cv2.CAP_PROP_FPS) * self.playback_speed)
                    self.update_frame()

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
        self.print_to_message_display(f"Start marking {action}.")  # Notify action start

    def end_action(self):
        self.action_end_time = self.current_frame_time
        self.update_csv()
        self.end_button.config(state=tk.DISABLED)
        self.pause()

    def toggle_plot_visibility(self):
        if self.show_plot.get():
            self.plot_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        else:
            self.plot_canvas.get_tk_widget().pack_forget()

    def update_window_size(self, value):
        self.window_size = timedelta(seconds=float(value))
        self.update_plot()

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
        if not self.show_plot.get():
            return

        self.ax.clear()
        window_end = self.current_frame_time
        window_start = window_end - self.window_size

        mask = (self.df['Absolute Time'] > window_start) & (self.df['Absolute Time'] <= window_end)
        df_window = self.df.loc[mask]

        if df_window.empty:
            return

        times = (df_window['Absolute Time'] - window_start) / pd.Timedelta(seconds=1)

        plot_type = self.plot_type.get()
        if plot_type == 0:  # Angle
            x_data = df_window['Pitch (deg)']
            y_data = df_window['Roll (deg)']
            z_data = df_window['Yaw (deg)']
            self.ax.set_ylabel('Angle (deg)')
        elif plot_type == 1:  # Acceleration
            x_data = df_window['X-axis Acceleration']
            y_data = df_window['Y-axis Acceleration']
            z_data = df_window['Z-axis Acceleration']
            self.ax.set_ylabel('Acceleration (m/s^2)')
        else:  # Angular Velocity
            x_data = df_window['X-axis Angular Velocity']
            y_data = df_window['Y-axis Angular Velocity']
            z_data = df_window['Z-axis Angular Velocity']
            self.ax.set_ylabel('Angular Velocity (deg/s)')

        self.ax.plot(times, x_data, lw=1, label='X')
        self.ax.plot(times, y_data, lw=1, label='Y')
        self.ax.plot(times, z_data, lw=1, label='Z')

        if self.show_background.get():
            self.add_background_blocks(times, df_window)

        self.ax.set_xlabel('Time (s)')
        self.ax.set_xlim(0, self.window_size.total_seconds())
        self.ax.legend()
        self.ax.relim()
        self.ax.autoscale_view(True, False, True)  # Only autoscale y-axis
        self.plot_canvas.draw()

    def add_background_blocks(self, times, df_window):
        colors = {
            'Go Straight': 'lightblue',
            'Idle': 'lightgrey',
            'Turn Left': 'lightgreen',
            'Turn Right': 'coral',
            'Hook Turn': 'yellow',
            'U-turn': 'violet'
        }
        actions = list(colors.keys())
        added_labels = set()
        prev_action = None
        start = 0
        for i, time in enumerate(times):
            current_action = df_window.iloc[i]['Action']
            if current_action != prev_action:
                if prev_action in actions:
                    if prev_action not in added_labels:
                        self.ax.axvspan(start, time, color=colors[prev_action], alpha=0.3, label=f'{prev_action}')
                        added_labels.add(prev_action)
                    else:
                        self.ax.axvspan(start, time, color=colors[prev_action], alpha=0.3)
                start = time
            prev_action = current_action

    def print_to_message_display(self, message):
        self.message_display.config(state=tk.NORMAL)  # Enable editing
        self.message_display.insert(tk.END, message + "\n")  # Insert message
        self.message_display.config(state=tk.DISABLED)  # Disable editing
        self.message_display.see(tk.END)  # Scroll to the end

    def write(self, message):
        self.print_to_message_display(message.strip())

    def flush(self):
        pass

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
    root.title("KDD RideTrack Label System App v2.3")

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