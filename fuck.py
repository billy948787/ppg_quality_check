import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. 讀取資料
# 注意：CSV 標頭可能有空格，將其去除
df = pd.read_csv('/Users/lijiye/coding/zig_bt_client/Raw_data.csv')
df.columns = df.columns.str.strip()

# 取得 raw signal
raw_signal = df['data'].values
fs = 100.0  # 近似採樣率 100 Hz

# 2. 設計帶通濾波器 (0.5 Hz 到 5.0 Hz)
lowcut = 0.5
highcut = 5.0
nyq = 0.5 * fs # 奈奎斯特頻率
low = lowcut / nyq
high = highcut / nyq

# 4階 Butterworth 濾波器
b, a = butter(4, [low, high], btype='band')

# 3. 雙向濾波 (避免相位偏移)
filtered_signal = filtfilt(b, a, raw_signal)

# 4. 畫圖 (取前1000筆，大約10秒的資料來觀察脈動)
samples_to_plot = 1000 
t = np.arange(samples_to_plot) / fs

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, raw_signal[:samples_to_plot], color='blue')
plt.title('Raw PPG Signal (First 10s)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal[:samples_to_plot], color='red')
plt.title('Filtered PPG Signal (0.5 - 5 Hz)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()