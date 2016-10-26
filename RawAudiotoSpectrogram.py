import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

f, t, Sxx = signal.spectrogram(audio_array, hz)

plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()