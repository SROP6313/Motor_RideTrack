import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Load the provided CSV file
file_path = '30Hz20240125_eric.csv'
data = pd.read_csv(file_path)
data = data[1415:3629]  # 取ECU開始紀錄時間的之後2分鐘(1415:3629)
# data = data[0:int(len(data)/4)]

IMU_feature = ['X-axis Acceleration', 'Y-axis Acceleration', 'Z-axis Acceleration', 'X-axis Angle', 'Y-axis Angle', 'Z-axis Angle', 'X-axis Angular Velocity', 'Y-axis Angular Velocity', 'Z-axis Angular Velocity']
X = range(len(data))
Y1 = data[IMU_feature[3]]
Y2 = data[IMU_feature[5]]
# Y3 = data[IMU_feature[2]]

fig, ax = plt.subplots(figsize=(18, 12))

DISPLAY_MODE = 'Line'  # 定義是折線圖還是散佈圖

if DISPLAY_MODE == 'Line':
    line1, = ax.plot(X, Y1, label='X-axis Angle')
    line2, = ax.plot(X, Y2, label='Z-axis Angle')

elif DISPLAY_MODE == 'Scatter':
    scatter1 = ax.scatter(X, Y1, s=2, label='X-axis Acceleration')
    scatter2 = ax.scatter(X, Y2, s=2, label='Y-axis Acceleration')

ax.legend()  # add legend for multiple scatters

def line_animate(i):
    line1.set_data(X[:i], Y1[:i])
    line2.set_data(X[:i], Y2[:i])
    return line1, line2

def scatter_animate(i):
    scatter1.set_offsets(np.c_[X[:i], Y1[:i]])
    scatter2.set_offsets(np.c_[X[:i], Y2[:i]])
    return scatter1, scatter2

if DISPLAY_MODE == 'Line':
    ani = animation.FuncAnimation(fig, line_animate, frames=len(X), blit=False, interval=47)  # interval為顯示2筆資料之間的間隔(ms)，可能有誤差

elif DISPLAY_MODE == 'Scatter':
    ani = animation.FuncAnimation(fig, scatter_animate, frames=len(X), blit=False, interval=47)

ani.save('./IMU_animation.mp4', writer='ffmpeg', fps=18)  # 儲存影片

plt.title("IMU Data Animation")
plt.xlabel("timestamp")
plt.ylabel("value")
# plt.ylim(-15, 15)

plt.show()