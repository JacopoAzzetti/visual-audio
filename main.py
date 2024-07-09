'''
INTERESTING EXTRACTION: https://nitratine.net/blog/post/finding-emotion-in-music-with-python/
in questo caso si tratta dei dati che si possono ottenere con le api di Spotify
"danceability" : 0.735,
"energy" : 0.578,
"key" : 5,
"loudness" : -11.840,
"mode" : 0,
"speechiness" : 0.0461,
"acousticness" : 0.514,
"instrumentalness" : 0.0902,
"liveness" : 0.159,
"valence" : 0.624,
"tempo" : 98.002,

LaTeX: https://it.overleaf.com/project/661688150c1f877285fcccfd
'''

# TODO: conda create -n visual-audio python=3.8 
# installare poi tutte le dipendenze per far funzionare il programma

import os
import librosa # per caricare il file
import pyloudnorm # per la Loudness in RMS
import inquirer # per la scelta dell'input

import essentia # FONTE: https://hpac.cs.umu.se/teaching/sem-mus-17/Reports/Paranjape.pdf
import essentia.standard as es# FONTE: https://hpac.cs.umu.se/teaching/sem-mus-17/Reports/Paranjape.pdf
import essentia.streaming

import statsmodels.api as sm
from scipy.signal import find_peaks
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import math as m

# Music player
import tkinter as tk
from tkinter import ttk
from pygame import mixer

# Speach Recognition
import webrtcvad

# Separatore traccie con modelli pre-addestrati
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import AudioFile as OpenFileDemucs
import soundfile as sf

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

# dataset iniziale ---> AUMENTARE
data_set = [
    'ASAP Rocky - RIOT',
    'Burial - Archangel',
    'Dolly Parton - Jolene',
    'Jay-Z - The Story Of OJ',
    'Jaska - Kimono',
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
    'Sons Of The East - Into The Sun',
    'The Rippingtons - Aruba!',
    'Nocturne, Op 9, No 2',
    'Snow Patrol - Chasing Cars',
    'In The Hall Of The Mountain King',
    'Artemas - i like the way you kiss me',
    'Juno Love (Santiago Garcia Remix)',
    'ASAP Rocky - RIOT (splitted)',
    'Andrea Bianchini - allnightallday',
    'juno love + into the sun']

# Valori max e min del coefficente (da verificare con diverse canzoni) per tarare la normalizzazione del coefficente (deve essere tra 0 e 1)
MAX_VAL = 15000
MIN_VAL = 0

MAX_VAL_BUCKET = 40
MIN_VAL_BUCKET = 0

# MUSIC PLAYER
class MusicPlayerApp:
    def __init__(self, window, file_dir):
        self.window = window
        self.file_dir = file_dir
        self.window.title(title)
        self.window.geometry("1100x350")

        self.setup_ui()

        # Inizializzazione di mixer di pygame (altrimenti non va)
        mixer.init()

    def setup_ui(self):
        self.play_button = ttk.Button(self.window, text="Play", command=self.play_music)
        self.stop_button = ttk.Button(self.window, text="Pause", command=self.stop_music)
        # https://www.pythontutorial.net/tkinter/tkinter-progressbar/
        self.progress_bar = ttk.Progressbar(self.window, orient="horizontal", length=1080, mode="determinate")

        # Caricamento e visualizzazione dell'immagine
        image = Image.open('./' + self.file_dir + '/' + title + '.png')  # path dell'immagine analizzata
        # image = image.resize((1080, 40))
        photo = ImageTk.PhotoImage(image)
        self.image_label = tk.Label(self.window, image=photo)
        self.image_label.image = photo

        self.play_button.pack(pady=10)
        self.stop_button.pack(pady=5)
        self.progress_bar.pack()
        self.image_label.pack()

    def play_music(self):
        mixer.music.load(file_path)  # Inserisci il percorso del file audio desiderato
        mixer.music.play()
        self.update_progress()

    def stop_music(self):
        mixer.music.stop()
        self.update_progress()
    
    def update_progress(self):
        total_length = mixer.Sound.get_length(mixer.Sound(file_path))
        # posizione corrente della canzone con conversione della lungezza in secondi
        current_position = mixer.music.get_pos() / 1000

        progress = (current_position / total_length) * 100

        self.progress_bar["value"] = progress

        # aggiorna ogni 100 millisecondi 
        self.window.after(100, self.update_progress)


class VisualAudioImage:
    def __init__(self, title, audio, sr, bpm, output_dir):
        self.title = title
        self.bpm = bpm
        self.sampling_rate = sr
        self.audio = audio
        self.out_dir = output_dir

        os.makedirs(output_dir, exist_ok=True) # creo la directory se non c'è già

        samples_in_bucket = int(sr)
        self.num_bucket = int(audio.size/samples_in_bucket)

        self.coef_per_bucket = []
        self.pitch_per_bucket_values = []
        self.energy_per_bucket_values = []

        # divido l'array principale dei campioni in num_bucket buckets usando .split di numpy
        self.buckets = [audio[i:i+samples_in_bucket] for i in range(0, len(audio), samples_in_bucket)]

        self.image_generation()


    def calculate_pitch_per_bucket(self):
        
        # calcolo del pitch per bucket
        for bucket in self.buckets:
            pitch_iteration = pitch_detection(bucket, self.sampling_rate)
            self.pitch_per_bucket_values.append(pitch_iteration) # aggiungo il risultato PER ORA ad un array dedicato al risultato solo per il pitch

    def calculate_energy_per_bucket(self):
       
        # calcolo dell'energia del segnale per bucket
        for bucket in self.buckets:
            energy_iteration = energy_detection(bucket, 1)
            self.energy_per_bucket_values.append(energy_iteration)

    def calculate_coef_per_bucket(self):    
        for i in range(0, len(self.energy_per_bucket_values)):
            cf_tmp = coefficent(self.pitch_per_bucket_values[i], self.energy_per_bucket_values[i], self.bpm, 1);
            self.coef_per_bucket.append(cf_tmp)

        # Per l'analisi per bucket, il risultato del coefficente è molto minore a quello generale per tutte le tracce, così devo diminuire i livelli massimi e minimi per la normalizzazione
        # normalizzo tutti i valori del coefficente per impostarli da 1 a 0
        for i in range(len(self.coef_per_bucket)):
            self.coef_per_bucket[i] = 1-((self.coef_per_bucket[i] - MIN_VAL_BUCKET) / (MAX_VAL_BUCKET - MIN_VAL_BUCKET))

    def image_generation(self):
        self.calculate_pitch_per_bucket()
        self.calculate_energy_per_bucket()
        self.calculate_coef_per_bucket()

        # larghezza = len(coef_per_bucket)
        larghezza = 3*len(self.coef_per_bucket)
        altezza = 240

        larghezza_colonne = int(larghezza / len(self.coef_per_bucket))-1 # diminuisco la larghezza delle colonne per fare in modo che ci stia tutta l'immagine nei 1080 pixel (problemi di approssimazione)

        counter_colonne = 0

        immagine = Image.new('RGB', (larghezza, altezza), 'white')

        for x in range(len(self.coef_per_bucket)):
            color = plt.cm.Spectral(self.coef_per_bucket[x])[0]
            color_rgb = tuple(int(float(rgba) * 255) for rgba in color[:3])
            
            for i in range(larghezza_colonne+1):
                pos = x+(counter_colonne*larghezza_colonne)+i
                if pos < larghezza:
                    for y in range(altezza):
                        immagine.putpixel((pos, y), color_rgb)
            
            counter_colonne += 1

        # img_fitered = immagine.filter(ImageFilter.BLUR)
        img_fitered = immagine

        img_fitered = img_fitered.resize((1080, 240))

        # Salva l'immagine
        img_fitered.save('./' + self.out_dir + '/' + self.title + '.png')

# Band Pass Filter: https://swharden.com/blog/2020-09-23-signal-filtering-in-python/
def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

# calcolo del coefficente che vado poi ad utilizzare per mappare il colore da associare all'immagine
def coefficent(pitch, energy, bpm, duration):
    return ((energy/pitch)+((energy/bpm)*duration));

# Mapping dei colori: https://matplotlib.org/stable/gallery/color/colormap_reference.html
def calculate_color(pitch, energy, bpm, duration):
    cf = coefficent(pitch, energy, bpm, duration)

    normalized_value = 1-((cf - MIN_VAL) / (MAX_VAL - MIN_VAL))
    '''
    mapping dei valori ai colori
    ---> output in RGBa mappando i valori (da 0 a 1)
    '''
    # h = plt.cm.jet(normalized_value) 
    # h = plt.cm.rainbow(normalized_value) # ---> è la scala colori che (a mio parere) fa mappa al meglio coefficente <--> colore
    h = plt.cm.Spectral(normalized_value)[0]
    r, g, b = tuple(int(rgba * 255) for rgba in h[:3]) # ----> conversione in RGB
    output_string = 'rgb(' + str(r) + ', ' + str(g) + ', ' + str(b) + ')'

    return output_string

'''
Estrazione del Pitch: 
Pitch is the subjective perception of a sound wave by the individual person, which cannot be directly measured. 
However, this does not necessarily mean that people will not agree on which notes are higher and lower.
Pitch refers to the specific frequency of a sound wave, which is measured in Hertz (Hz)

Fonti: 
    - https://scicoding.com/pitchdetection/
    - https://en.wikipedia.org/wiki/Pitch_(music)
'''

def pitch_detection(samples, sampling_rate):
    # aggiungendo la finestra di scorrimento sul segnale dato in input, è possibile lavorare anche con dei sottoinsiemi che ho definito "bucket"
    autocorrelation_window = 0.1  # Lunghezza della finestra in secondi
    window_length = int(autocorrelation_window * sampling_rate)
    # calcolo dell'autocorrelazione con successivo rilevamento dei picchi e calcolo del pitch
    autocorrelation = sm.tsa.acf(samples, nlags=window_length, missing='conservative')
    peaks = find_peaks(autocorrelation)[0]
    # ci sono casi in cui non riesce a trovare nessun picco, quindi in quel caso ritorno un pitch minimo
    if len(peaks) != 0:
        lag = peaks[0]
        pitch = sampling_rate / lag
        return pitch
    
    return 1

def energy_detection(audio_signal, time):
    energy_extractor = es.Energy() # estrattore dell'energia
    energy = energy_extractor(audio_signal) # calcolo effettivo dell'energia fatto sul file audio
    return energy/time

def image_generator(pitch, energy, bpm, duration):
    # dimensioni dell'immagine da generare
    img_length = 1080
    img_hight = 240
    # creo l'immagine con il colore rgb calcolato dalla funzione appostita
    image = Image.new('RGB', (img_length, img_hight), calculate_color(pitch, energy, bpm, duration))
    path = './img_spectral_map/img-timeenrgy-' + title + '.png'
    image.save(path)

def spectrogram_generator(file_path, samples, sr):
    # ESTRAGGO E VISUALIZZAO LO SPETTROGRAMMA
    spctrm_graph = librosa.feature.melspectrogram(y=samples, sr=sr)
    spctrm_dB = librosa.power_to_db(spctrm_graph, ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(spctrm_dB, x_axis='time', y_axis='mel', sr=sr, fmax=16000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.title(file_path)
    plt.show()

def separazione_traccie(file_path):
    output_dir = './separated_tracks/'
    os.makedirs(output_dir, exist_ok=True) # creo la directory se non c'è già

    model = pretrained.get_model('mdx')
    
    # ds.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", "track with space.mp3"])
    f = OpenFileDemucs(file_path)
    wav_audio = f.read(streams=0)
    sr = f.samplerate()


    # aggiunta di dimensioni batch per poter usare apply_model
    if wav_audio.dim() == 1:
        wav_audio = wav_audio.unsqueeze(0)  # Aggiungi una dimensione per il canale
    
    if wav_audio.dim() == 2:
        wav_audio = wav_audio.unsqueeze(0)  # Aggiungi una dimensione batch

    # separazionde delle traccie audio usando la CPU come calcolatore
    separazione = apply_model(model, wav_audio, device='cpu')[0]
    
    for i, source in enumerate(separazione):
        track_name = model.sources[i]
        
        output_file = os.path.join(output_dir, f"{track_name}.wav")

        source = source.cpu().numpy()
        
        sf.write(output_file, source.T, sr)

    print("Separazione traccie conclusa!")

'''
MAIN PROGRAM:

Faccio un ciclo for per poter eseguire l'analisi di tutti i file audio inseriti nel dataset
'''
# ripulisco il terminale prima di inizare
print(chr(27) + "[2J")
# chiedo che canzone si vuole analizzare
# https://stackoverflow.com/questions/37565793/how-to-let-the-user-select-an-input-from-a-finite-list
question = [
  inquirer.List('song',
                message="Which song would you like to analyze?",
                choices=data_set,
            ),
]

# uso la risposta come input per la scelta della canzone
answer = inquirer.prompt(question)
title = answer['song']

file_path = 'dataset/wav/' + title + '.wav'

# pulisco il terminale dalla risposta e stampo subito il titolo scelto
print(chr(27) + "[2J")
print("\nFile: " + title)

# Carico il file audio utilizzando la libreria "librosa"
audio_samples, sampling_rate = librosa.load(file_path) # output 1: nparray contenenete i campioni del file wav. Output 2: freq. di campionamento
# FONTE https://essentia.upf.edu/essentia_python_tutorial.html 
# --> si tratta di un altro tipo di loading per poter poi calcolare l'energia del segnale
loader = es.EasyLoader(filename=file_path)
audio_signal = loader()

# estraggo informazioni generali ed utili
T = 1/sampling_rate # periodo T di campionamento
N = len(audio_samples) # Numero di sample estratti
t = N/sampling_rate # lunghezza del segnale in secondi

min = int(t/60) # calcolo dei minuti
sec = t%60 # calcolo dei secondi

print("\nDuration: " + str(min) + " min e " + str(int(sec)) + " sec")
# =============================================================================================================================
'''
utitlizzo il segnale filtrato solo per il rilevamento del pitch per evitare che la presenza 
di kicks e low-end (sotto i 100 Hz) in generale influenzi quanto viene misurato
'''
filtered_audio_samples = bandpass(audio_samples, [100, 10000], sampling_rate)

# =============================================================================================================================
spectrogram, phase = librosa.magphase(librosa.stft(audio_samples)) # estraggo lo spettrogramma S - viene usata una Short Time Fourier Transformation

# Calcolo dei BPM tramite libreria
bpm, numero_frame_rilevato_beat = librosa.beat.beat_track(y=audio_samples, sr=sampling_rate) # output 1: valore calcolato di BPM. Output 2: array contenente tutti i punti dove un beat è rilevato 

print('BPM: ', int(bpm[0]))
# =============================================================================================================================

# =============================================================================================================================
# Calcolo di Loudness (RMS - Root Mean Square: used for the overall amplitude of a signal)
# RMS measures the average power of the signal, providing a representation of its overall loudness level. This value is often expressed in decibels relative to a reference level.
rms_spectrogram = librosa.feature.rms(S=spectrogram) # faccio il calcolo sui campioni per tutto il file
mean_rms_spectrogram = rms_spectrogram.mean()
print(f'\nRMS: {mean_rms_spectrogram:.4f}')

# Calcolo di Loudness in LUFS (converto RMS in LUF)
meter = pyloudnorm.Meter(sampling_rate)
lufs = meter.integrated_loudness(audio_samples)
print(f'LUFS: {lufs:.4f}')

# FONTE: https://essentia.upf.edu/reference/streaming_Energy.html
# Calcolo dell'energia del segnale
energy = energy_detection(audio_signal, t)
print(f'Signal Energy: + {energy}')
# =============================================================================================================================

print('Inizio la separazione delle tracce...')
print('[l\'operazione può dichiedere alcuni minuti]')
separazione_traccie(file_path=file_path)

# =============================================================================================================================
# Pitch rilevato sull'intero file audio --> frequenza specifica del file audio (guardando lo spettrogramma si può intendere come la frequenza più presente)
pitch = pitch_detection(filtered_audio_samples, sampling_rate)
print(f'\nPitch: {pitch:.2f} Hz')

audio_image = VisualAudioImage(title=title, audio=audio_samples, sr=sampling_rate, bpm=bpm, output_dir='timefunction-img/' + title)

drums, sr_drums = librosa.load('./separated_tracks/drums.wav')
drums_image = VisualAudioImage(title='(drums) ' + title, audio=drums, sr=sr_drums, bpm=bpm, output_dir='timefunction-img/' + title)

vocals, sr_vocals = librosa.load('./separated_tracks/vocals.wav')
vocals_image = VisualAudioImage(title='(vocals) ' + title, audio=vocals, sr=sr_vocals, bpm=bpm, output_dir='timefunction-img/' + title)

bass, sr_bass = librosa.load('./separated_tracks/bass.wav')
bass_image = VisualAudioImage(title='(bass) ' + title, audio=bass, sr=sr_bass, bpm=bpm, output_dir='timefunction-img/' + title)

other, sr_other = librosa.load('./separated_tracks/other.wav')
other_image = VisualAudioImage(title='(other) ' + title, audio=other, sr=sr_other, bpm=bpm, output_dir='timefunction-img/' + title)


# =============================================================================================================================

# =============================================================================================================================
# Estrazione della KEY della traccia
# FONTR: https://medium.com/@oluyaled/detecting-musical-key-from-audio-using-chroma-feature-in-python-72850c0ae4b1
# estrae un "chronogram" (ovvero un grafico dove si possono osservare le diverse note) dal file audio datogli in input
chroma_rappresentation = librosa.feature.chroma_stft(y=audio_samples, sr=sampling_rate)
# calcolo della media della Chroma Feature
mean_chroma = np.mean(chroma_rappresentation, axis=1)
# mapping dei risultati con le note
chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# cerco il valore della Key
estimated_key_index = np.argmax(mean_chroma) # indice nell'array
estimated_key = chroma_to_key[estimated_key_index] # mapping tra l'indice rilevato e le note effettive

print('Key: ' + estimated_key)
# =============================================================================================================================

# =============================================================================================================================
# FONTE: https://mtg.github.io/essentia.js/docs/api/Essentia.html#Danceability

# Calcolo/Estrazione di "Danceability"
danceability = es.Danceability() #  è uno degli estrattori di "essentia.standard"  per calcolare la danceability di una traccia
danceability_value, dfa_array = danceability(audio_signal) # https://essentia.upf.edu/reference/streaming_Danceability.html --> spiegato ciò che ritorna essentia.standard.Danceability
print(f'\nDanceability: + {danceability_value:.4f}') # valore di danceability da 0 a 3
# =============================================================================================================================
# Il coefficente mi serve per poter fare il mapping della canzone con il relativo colore
coef = float(coefficent(pitch, energy, bpm, t)[0])
print(f'Coefficent: + {coef:.2f}')
        

# =============================================================================================================================

'''
spectrogram_generator(title, filtered_audio_samples, sampling_rate)
'''
image_generator(pitch, energy, bpm, t)

print('================================================================\n')

'''
    Possibile fonte per il player: 
    https://medium.com/@desmondmutuma35/building-a-music-player-with-python-a-step-by-step-tutorial-7cedf4770e1
'''

window = tk.Tk()
app = MusicPlayerApp(window, file_dir='timefunction-img/' + title)

# Main event loop
window.mainloop()
