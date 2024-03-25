# --
import wave
import numpy as np
import matplotlib.pyplot as plt

## COSTANTI
t_sec = 0.1 # intervallo di tempo da osservare
w_size = 100 # dimensioni di campioni permessi nella window

# apro il file
file = 'dataset/wav/Vivian Roost - To the Sky.wav'
audio_wav = wave.open(file, 'rb')
signal = audio_wav.readframes(-1) # ricavo tutti i frames del file aperto
signal = np.frombuffer(signal, dtype=np.int16) # trasformo i frame in frame di interi (?)

# calcolo il numero di campioni
sample_rate = audio_wav.getframerate() # mi salvo l'intervallo di campionamento
sample_freq = 1/sample_rate
n_sample = int(sample_rate * t_sec) # calcolo il numero di campioni in base alla finestra che imposto in 't_sec'

# definisco inizio e fine del campionamento
star_sample = n_sample - w_size
end_sample = n_sample + w_size

audio_segment = signal[star_sample:end_sample] #ottendo i segmenti del file audio da ... a ...

## CALCOLO DELLA TRASFORMATA DI FOURIER
fft_res = np.fft.fft(audio_segment)

frequencies = np.fft.fftfreq(len(fft_res), sample_freq)
positive_frequencies = frequencies[frequencies >= 0] # prendo le frequenze solo positive
# positive_frequencies = np.logspace(np.log10(positive_frequencies[0]), np.log10(positive_frequencies[-1]), len(positive_frequencies)) # prendo la scala come logaritmica

amplitudes = np.abs(fft_res)
amplitudes_dB = 20 * np.log10(amplitudes)
amplitudes_dB = amplitudes_dB[:len(positive_frequencies)]

print('Frequenze: ' + str(positive_frequencies))
print('Ampiezze: ' + str(amplitudes_dB))

plt.plot(positive_frequencies, amplitudes_dB)
plt.title('Spettro di frequenza')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Ampiezza')
plt.show()