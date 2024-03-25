# TODO
# • preparare un dataset di circa 20 canzoni di vario genere [√ preparato per 10 canzoni]
# • provare ad eseguire un'estrazione di informazioni (√bpm, danceability, √pitch, √loudness, (√)energy, ...) dalle entità del dataset
# • definire IO delle associazioni delle canzoni ad un cerco colore/simbolo/cosa in modo che possa poi essere usato per un map con l'estrazione delle info

# INTERESTING EXTRACTION: https://nitratine.net/blog/post/finding-emotion-in-music-with-python/
# in questo caso si tratta dei dati che si possono ottenere con le api di Spotify
# "danceability" : 0.735,
# "energy" : 0.578,
# "key" : 5,
# "loudness" : -11.840,
# "mode" : 0,
# "speechiness" : 0.0461,
# "acousticness" : 0.514,
# "instrumentalness" : 0.0902,
# "liveness" : 0.159,
# "valence" : 0.624,
# "tempo" : 98.002,


import librosa # per caricare il file
import pyloudnorm # per la Loudness in RMS

import essentia # FONTE: https://hpac.cs.umu.se/teaching/sem-mus-17/Reports/Paranjape.pdf
import essentia.standard as es# FONTE: https://hpac.cs.umu.se/teaching/sem-mus-17/Reports/Paranjape.pdf
import essentia.streaming

import statsmodels.api as sm
from scipy.signal import find_peaks
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math as m

'''
# AUDIO FILES
audio_file = 'dataset/wav/ASAP Rocky - RIOT.wav' # HIP HOP
audio_file = 'dataset/wav/Burial - Archangel.wav' # EMOTIONAL ELECTRO
audio_file = 'dataset/wav/Dolly Parton - Jolene.wav' # COUNTRY POP
audio_file = 'dataset/wav/Jaska - Run away.wav' # AFRO HOUSE
audio_file = 'dataset/wav/Jay-Z - The Story Of OJ.wav' # RAP
audio_file = 'dataset/wav/Kanye West - Jail.wav' # HIP HOP
audio_file = 'dataset/wav/Lana Del Rey - Groupie Love ft. A$AP Rocky.wav' # POP
audio_file = 'dataset/wav/Nada - Amore Disperato.wav' # POP
audio_file = 'dataset/wav/Pink Floyd - The Great Gig In The Sky.wav' # ALTERNATIVE ROCK
audio_file = 'dataset/wav/Tyler, The Creator - IFHY.wav' # RAP
audio_file = 'dataset/wav/Vivian Roost - To the Sky.wav' # CLASSICAL
audio_file = 'dataset/wav/Astor Piazzolla - Libertango.wav' # TANGO
audio_file = 'dataset/wav/Peggy Gou - It Goes Like (Nanana).wav' # HOUSE
audio_file = 'dataset/wav/BUNT. - Clouds (ft. Nate Traveller).wav' # ELETTRONICA
audio_file = 'dataset/wav/manu chao - bongo bong.wav' # LATINA
audio_file = 'dataset/wav/Armand Van Helden - Wings (I Won t Let You Down).wav' # HOUSE
audio_file = 'dataset/wav/triangular_wave_base_1100_Hz.wav' # TEST onda triangolare a 1200 Hz 
audio_file = 'dataset/wav/Daft Punk - Giorgio By Moroder.wav' # FUNKY ELETTRONICA
audio_file = 'dataset/wav/Pharrell Williams - Happy.wav' # POP ALLEGRA
'''

data_set = [
    'ASAP Rocky - RIOT',
    'Burial - Archangel',
    'Dolly Parton - Jolene',
    'Jaska - Run away',
    'Jay-Z - The Story Of OJ',
    'Kanye West - Jail',
    'Lana Del Rey - Groupie Love ft. A$AP Rocky',
    'Nada - Amore Disperato',
    'Pink Floyd - The Great Gig In The Sky',
    'Tyler, The Creator - IFHY',
    'Vivian Roost - To the Sky',
    'Astor Piazzolla - Libertango',
    'Peggy Gou - It Goes Like (Nanana)',
    'BUNT. - Clouds (ft. Nate Traveller)',
    'manu chao - bongo bong',
    'Armand Van Helden - Wings (I Won t Let You Down)',
    'Daft Punk - Giorgio By Moroder',
    'Pharrell Williams - Happy',
    'Sons Of The East - Into The Sun']

MAX_VAL = 8000
MIN_VAL = 200
SELECTED_DATA = 18

# Band Pass Filter: https://swharden.com/blog/2020-09-23-signal-filtering-in-python/
def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def coefficent(pitch, energy, bpm):
    return (energy/bpm)-(energy/pitch);

# Mapping dei colori: https://matplotlib.org/stable/gallery/color/colormap_reference.html
def calculate_color(pitch, energy, bpm):
    cf = coefficent(pitch, energy, bpm)

    normalized_value = ((cf - MIN_VAL) / (MAX_VAL - MIN_VAL))
    '''
    mapping dei valori ai colori
    ---> output in RGBa mappando i valori (da 0 a 1) su il tipo PLASMA che va da un minimo di blu scuro ad un massimo di giallo
    '''
    h = plt.cm.jet(normalized_value) 
    # h = plt.cm.rainbow(normalized_value)
    # h = plt.cm.gnuplot2(normalized_value)

    r, g, b = tuple(int(rgba * 255) for rgba in h[:3]) # ----> conversione in RGB
    output_string = 'rgb(' + str(r) + ', ' + str(g) + ', ' + str(b) + ')'
    
    print('---> Normalized CF: ' + str(normalized_value))
    print('---> Map: ' + str(h))
    print('---> Output color: ' + output_string + '\n')


    return output_string


def image_generator(pitch, energy, bpm):
    img_length = 1080
    img_hight = 240
    image = Image.new('RGB', (img_length, img_hight), calculate_color(pitch, energy, bpm))
    path = './img_jet_map/img-' + title + '.png'
    image.save(path)
    # image.show()

def spectrogram_generator(audio_file, samples, sr):
    # ESTRAGGO E VISUALIZZAO LO SPETTROGRAMMA
    spctrm_graph = librosa.feature.melspectrogram(y=samples, sr=sr)
    spctrm_dB = librosa.power_to_db(spctrm_graph, ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(spctrm_dB, x_axis='time', y_axis='mel', sr=sr, fmax=16000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.title(audio_file)
    plt.show()

# =============================================================================================================================
for i in range(0, 1):

    title = data_set[i]
    audio_file = 'dataset/wav/' + title + '.wav'

    # Carico il file audio utilizzando la libreria "librosa"
    audio_samples, sampling_rate = librosa.load(audio_file) # output 1: **NP array** contenenete i campioni del file wav. Output 2: freq. di campionamento
    loader = es.EasyLoader(filename=audio_file) # FONTE https://essentia.upf.edu/essentia_python_tutorial.html
    audio_signal = loader()

    # estraggo informazioni generali ed utili
    T = 1/sampling_rate # periodo T di campionamento
    N = len(audio_samples) # Numero di sample estratti
    t = N/sampling_rate # lunghezza del segnale in secondi

    min = int(t/60) # calcolo dei minuti
    sec = t%60 # calcolo dei secondi

    print("\nFile: " + title)
    print("\nDuration: " + str(min) + " min e " + str(int(sec)) + " sec")
    # =============================================================================================================================
    '''
    utitlizzo il segnale filtrato solo per il rilevamento del pitch per evitare che la presenza 
    di kicks e low-end in generale influenzi quanto viene misurato
    '''
    filtered_audio_samples = bandpass(audio_samples, [100, 10000], sampling_rate)

    # =============================================================================================================================
    spectrogram, phase = librosa.magphase(librosa.stft(audio_samples)) # estraggo lo spettrogramma S - viene usata una Short Time Fourier Transformation

    # Calcolo dei BPM tramite libreria
    bpm, numero_frame_rilevato_beat = librosa.beat.beat_track(y=audio_samples, sr=sampling_rate) # output 1: valore calcolato di BPM. Output 2: array contenente tutti i punti dove un beat è rilevato 

    print(f"BPM: {bpm:.2f}")
    # =============================================================================================================================

    # =============================================================================================================================
    # Calcolo di Loudness (RMS - Root Mean Square: used for the overall amplitude of a signal)
    # RMS measures the average power of the signal, providing a representation of its overall loudness level. This value is often expressed in decibels relative to a reference level.
    rms_spectrogram = librosa.feature.rms(S=spectrogram) # faccio il calcolo sui campioni per tutto il file
    mean_rms_spectrogram = rms_spectrogram.mean()
    print(f"\nRMS: {mean_rms_spectrogram:.4f}")

    # Calcolo di Loudness in LUFS (converto RMS in LUF)
    meter = pyloudnorm.Meter(sampling_rate)
    lufs = meter.integrated_loudness(audio_samples)
    print(f"LUFS: {lufs:.4f}")

    # FONTE: https://essentia.upf.edu/reference/streaming_Energy.html
    # Calcolo dell'energia del segnale
    energy_extractor = es.Energy() # estrattore dell'energia
    energy = energy_extractor(audio_signal) # calcolo effettivo dell'energia fatto sul file audio
    print(f"Signal Energy: + {energy}")
    # =============================================================================================================================

    # =============================================================================================================================
    # Estrazione del Pitch: pitch refers to the specific frequency of a sound wave, which is measured in Hertz (Hz)
    # Fonte: https://scicoding.com/pitchdetection/
    autocorrelazione = sm.tsa.acf(filtered_audio_samples)
    peaks = find_peaks(autocorrelazione)[0] # trova il picco (di frequenze (?)) nell'autocorrelazione
    lag = peaks[0] # prendo solo il primo picco come componente per il pitch
    pitch = sampling_rate / lag # Trasformo il valore di picco ricavato prima in frequenza
    print(f"\nPitch: {pitch:.2f} Hz")
    # =============================================================================================================================

    # =============================================================================================================================
    # Estrazione della KEY della traccia
    # FONTR: https://medium.com/@oluyaled/detecting-musical-key-from-audio-using-chroma-feature-in-python-72850c0ae4b1
    chroma_rappresentation = librosa.feature.chroma_stft(y=audio_samples, sr=sampling_rate)
    # calcolo della media della Chroma Feature
    mean_chroma = np.mean(chroma_rappresentation, axis=1)
    # mapping dei risultati con le note permesse
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # cerco il valore della Key
    estimated_key_index = np.argmax(mean_chroma) # indice nell'array
    estimated_key = chroma_to_key[estimated_key_index] # mapping tra l'indice rilevato e le note effettive

    print("Key: " + estimated_key)
    # =============================================================================================================================

    # =============================================================================================================================
    # FONTE: https://mtg.github.io/essentia.js/docs/api/Essentia.html#Danceability

    # Calcolo/Estrazione di "Danceability" !! NON PER IL FILE AUDIO DI TEST - Segnale Triancolare - !!
    danceability = es.Danceability() #  è uno degli estrattori di "essentia.standard"  per calcolare la danceability di una traccia
    danceability_value, dfa_array = danceability(audio_signal) # https://essentia.upf.edu/reference/streaming_Danceability.html --> spiegato ciò che ritorna essentia.standard.Danceability
    print(f"\nDanceability: + {danceability_value:.4f}") # valore di danceability da 0 a 3
    # =============================================================================================================================

    coef = coefficent(pitch, energy, bpm)
    print(f"Coefficent: + {coef:.2f}")
    print("----------------------------------------------------------------\n")
            

    # =============================================================================================================================

    '''
    spectrogram_generator(audio_file, audio_samples, sampling_rate)
    '''
    image_generator(pitch, energy, bpm)

    print("================================================================\n")

    # GIà LA COMBINAZIONE DEGLI ELEMENTI ESTRATTI MI PERMETTE DI DARE UNA VALUTAZIONE DELLA TRACCIA ANALIZZATA