import numpy as np
from numpy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

def fft_filter(data_x, data_y, high=10000):
    
    spectre = fft(data_y)
    
    frequencies = fftfreq(len(data_x), np.mean(np.diff(data_x)))

    amplitudes = np.abs(spectre)

    mean = np.mean(amplitudes)

    min_height = 5 * mean # минимальная высота пика по частоте, по условию задачи должна быть где то 2 * mean

    # но доступ к файлу закрыт, а используемые файлы с лабы 3.00 очень зашумленные

    peaks, peaks_dict = find_peaks(amplitudes, height=min_height)

    f_peaks = []
    for i in peaks:
        if amplitudes[i] >= 2 * min_height:
            f_peaks.append(i)

    for i in range(len(frequencies)):
        if i not in f_peaks:
            spectre[i] = 0
        elif frequencies[i] > 0:
            print(f"{frequencies[i]} Gz")
            
    
    return data_x, ifft(spectre).real

path = input("Enter path to data file: ")

pd.options.display.max_rows = 10000

csv_data = pd.read_csv(path, skiprows=25, names=["Time", "Signal"], usecols=[0, 1])
signal_x = [float(elem[0]) for elem in csv_data.values]
signal_y = [float(elem[1]) for elem in csv_data.values]

plt.figure(figsize=(10, 8))

ulm = 1.2 * max(signal_y)
dlm = 1.2 * min(signal_y)

plt.subplot(2, 1, 1)
plt.ylim(dlm, ulm)
plt.plot(signal_x, signal_y, label="Сигнал")
plt.xlabel("Время, с")
plt.ylabel("Напряжение, В")
plt.title("Оригинальный сигнал")
plt.legend()
plt.grid()

fft_signal_x, fft_signal_y = fft_filter(signal_x, signal_y)

plt.subplot(2, 1, 2)
plt.ylim(dlm, ulm)
plt.plot(fft_signal_x, fft_signal_y, label="Сигнал")
plt.xlabel("Время, с")
plt.ylabel("Напряжение, В")
plt.title("FFT фильтрация")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
