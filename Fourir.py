import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play
from IPython.display import Audio
from IPython.display import display
import numpy as np
from scipy.io import wavfile

#load audio file
fs, data = wavfile.read("son.wav")
#apply fft
fft_data=np.fft.fft(data)
# Plot magnitude of the FFT
plt.plot(np.abs(fft_data))
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.title('FFT Magnitude')
plt.show()

# Apply inverse FFT
ifft_data = np.fft.ifft(fft_data).real.astype(int)
# generate a random signal
x = np.random.randn(10000)

# take the fft of the signal
X = np.fft.fft(x)

# take the ifft of the frequency domain signal
y = np.fft.ifft(X)

# plot the original signal, the frequency domain signal, and the reconstructed signal
plt.subplot(311)
plt.plot(x)
plt.title('Original Signal')
plt.subplot(312)
plt.plot(np.abs(X))
plt.title('Frequency Domain Signal')
plt.subplot(313)
plt.plot(y)
plt.title('Reconstructed Signal')
plt.tight_layout()
plt.show()

# Play original audio
from IPython.display import audio_file
display(Audio(data, rate=fs))
display(audio_file)

# Play reconstructed audio
display(Audio(ifft_data, rate=fs))
# load audio file
audio_file = AudioSegment.from_file("son.WAV", format="WAV")
audio_file.export(format='wav').display()

# play audio file
play(audio_file)