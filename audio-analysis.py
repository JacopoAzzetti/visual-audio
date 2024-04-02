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
'''

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
from PIL import Image, ImageFilter
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

# dataset iniziale ---> AUMENTARE
data_set = [
    'ASAP Rocky - RIOT',
    'Burial - Archangel',
    'Dolly Parton - Jolene',
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
    'Sons Of The East - Into The Sun',
    'The Rippingtons - Aruba!',
    'Nocturne, Op 9, No 2',
    'Snow Patrol - Chasing Cars',
    'In The Hall Of The Mountain King']

# Valori max e min del coefficente (da verificare con diverse canzoni) per tarare la normalizzazione del coefficente (deve essere tra 0 e 1)
MAX_VAL = 15000
MIN_VAL = 0

MAX_VAL_BUCKET = 40
MIN_VAL_BUCKET = 0

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
    h = plt.cm.Spectral(normalized_value)

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
    autocorrelation = sm.tsa.acf(samples, nlags=window_length, missing="conservative")
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
    path = './img_rainbow_map/3-img-timeenrgy-' + title + '.png'
    image.save(path)

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

'''
MAIN PROGRAM:

Faccio un ciclo for per poter eseguire l'analisi di tutti i file audio inseriti nel dataset
'''
# for i in range(9, 10):
for i in range(len(data_set)):

    title = data_set[i]
    audio_file = 'dataset/wav/' + title + '.wav'

    # Carico il file audio utilizzando la libreria "librosa"
    audio_samples, sampling_rate = librosa.load(audio_file) # output 1: nparray contenenete i campioni del file wav. Output 2: freq. di campionamento
    # FONTE https://essentia.upf.edu/essentia_python_tutorial.html 
    # --> si tratta di un altro tipo di loading per poter poi calcolare l'energia del segnale
    loader = es.EasyLoader(filename=audio_file)
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
    di kicks e low-end (sotto i 100 Hz) in generale influenzi quanto viene misurato
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
    energy = energy_detection(audio_signal, t)
    print(f"Signal Energy: + {energy}")
    # =============================================================================================================================

    # =============================================================================================================================
    # Pitch rilevato sull'intero file audio --> frequenza specifica del file audio (guardando lo spettrogramma si può intendere come la frequenza più presente)
    pitch = pitch_detection(filtered_audio_samples, sampling_rate)
    print(f"\nPitch: {pitch:.2f} Hz")

    # Calcolo del pitch in funzione del tempo --> divido il segnale in bucket contenenti un certo numero di campioni e viene eseguito il calcolo del pitch per ogni bucket
    samples_in_bucket = int(sampling_rate)
    num_bucket = int(audio_samples.size/samples_in_bucket)
    
    pitch_per_bucket = []
    energy_per_bucket = []
    coef_per_bucket = []
    # divido l'array principale dei campioni in num_bucket buckets usando .split di numpy
    buckets = [audio_samples[i:i+samples_in_bucket] for i in range(0, len(audio_samples), samples_in_bucket)]
    
    '''
    TODO:
        - verificare la correttezza del calcolo del pitch
        - aggiungere il calcolo dell'energia per bucket (occhio che non si calcola sugli audio_samples, ma sull'audio signal)
        - calolare per ogni bucket il coefficente
        - si potrebbe creare un qualcosa dove per ogni cella sono mappati pitch e energy per ogni bucket [[5458.36, 1102.5], [5458.36, 1160.52], [5304.74, 3150.0], ...]
    '''

    # calcolo del pitch per bucket
    for bucket in buckets:
        pitch_iteration = pitch_detection(bucket, sampling_rate)
        pitch_per_bucket.append(pitch_iteration) # aggiungo il risultato PER ORA ad un array dedicato al risultato solo per il pitch


    # calcolo dell'energia del segnale per bucket
    for bucket in buckets:
        energy_iteration = energy_detection(bucket, 1)
        energy_per_bucket.append(energy_iteration)
    
    for i in range(0, len(energy_per_bucket)):
        cf_tmp = coefficent(pitch_per_bucket[i], energy_per_bucket[i], bpm, 1);
        coef_per_bucket.append(cf_tmp)

    # MAX_VAL = np.max(coef_per_bucket)
    # MIN_VAL = np.min(coef_per_bucket)
    
    # Per l'analisi per bucket, il risultato del coefficente è molto minore a quello generale per tutte le tracce, così devo diminuire i livelli massimi e minimi per la normalizzazione
    # normalizzo tutti i valori del coefficente per impostarli da 1 a 0
    for i in range(0, len(coef_per_bucket)):
        coef_per_bucket[i] = 1-((coef_per_bucket[i] - MIN_VAL_BUCKET) / (MAX_VAL_BUCKET - MIN_VAL_BUCKET))
    '''
    test per vedere intnto la modifica del colore di un immagine infunzione del tempo
    '''
    # larghezza = len(coef_per_bucket)
    larghezza = 3*len(coef_per_bucket)
    altezza = 240

    larghezza_colonne = int(larghezza / len(coef_per_bucket))-1 # diminuisco la larghezza delle colonne per fare in modo che ci stia tutta l'immagine nei 1080 pixel (problemi di approssimazione)

    counter_colonne = 0

    immagine = Image.new("RGB", (larghezza, altezza), "white")

    # for x in range(len(coef_per_bucket)):
    #     color = plt.cm.Spectral(coef_per_bucket[x])
    #     color_rgb = tuple(int(rgba * 255) for rgba in color[:3])
        
    #     for y in range(altezza):
    #         immagine.putpixel((x, y), color_rgb)

    
    for x in range(len(coef_per_bucket)):
        color = plt.cm.Spectral(coef_per_bucket[x])
        color_rgb = tuple(int(rgba * 255) for rgba in color[:3])
        
        for i in range(larghezza_colonne+1):
            pos = x+(counter_colonne*larghezza_colonne)+i
            if pos < larghezza:
                for y in range(altezza):
                    immagine.putpixel((pos, y), color_rgb)
                    # print("Put Pixel X = ", (pos), ", Y = ", y)
                    # print("x = ", x, " colonna = ", pos, " i = ", i , "\n")
        
        counter_colonne += 1

    img_fitered = immagine.filter(ImageFilter.BLUR)

    # Salva l'immagine
    img_fitered.save('./timefunction-img/3-' + title + '.png')

        
    '''
    ENERGIA IN FUNZIONE DEL TEMPO --> DA VERIFICARNE LA FATTIBILITA'

    '''
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

    print("Key: " + estimated_key)
    # =============================================================================================================================

    # =============================================================================================================================
    # FONTE: https://mtg.github.io/essentia.js/docs/api/Essentia.html#Danceability

    # Calcolo/Estrazione di "Danceability"
    danceability = es.Danceability() #  è uno degli estrattori di "essentia.standard"  per calcolare la danceability di una traccia
    danceability_value, dfa_array = danceability(audio_signal) # https://essentia.upf.edu/reference/streaming_Danceability.html --> spiegato ciò che ritorna essentia.standard.Danceability
    print(f"\nDanceability: + {danceability_value:.4f}") # valore di danceability da 0 a 3
    # =============================================================================================================================
    # Il coefficente mi serve per poter fare il mapping della canzone con il relativo colore
    coef = coefficent(pitch, energy, bpm, t)
    print(f"Coefficent: + {coef:.2f}")
            

    # =============================================================================================================================

    '''
    spectrogram_generator(title, filtered_audio_samples, sampling_rate)
    '''
    image_generator(pitch, energy, bpm, t)

    print("================================================================\n")

    # GIà LA COMBINAZIONE DEGLI ELEMENTI ESTRATTI MI PERMETTE DI DARE UNA VALUTAZIONE DELLA TRACCIA ANALIZZATA