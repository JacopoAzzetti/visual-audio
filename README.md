# Visual Audio

_Una rappresentazione visuale di tracce audio_

## Creazione dell'ambiente virtuale

Per iniziare, è necessario che sia installato `conda` sulla propria macchina (https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

Una volta installato ed utilizzabile, è necessario creare l'ambiente virtuale già predisposto con tutte le dipendenze richieste dal progetto. Dalla cartella principale del progetto (`./visual-audio`), lanciare il seguente comando:

`conda env create -f tools/visual-audio-env.yml`

Una volta completata l'installazione come indicato, attivare l'ambiente virtuale attraverso il comando:

`conda activate visual-audio`

Se fosse necessario eliminare un ambiente virtuale creato, utilizzare il seguente comando:

`conda remove -n <ENV NAME> --all`

### N.B.:

A disposizione è stato messo anche anche il file "requirements.txt" in caso di installazione delle dipendenze tramite pip e non utilizzando un ambiente `conda`.<br>Quindi tramite il comando:

_(consigliabile sempre creare un ambiente virtuale prima)_<br>
`pip install -r requirements.txt`

È importante che venga installato anche la libreria `ffmpeg` che non viene installata tramite il file requirements _(a differenza di conda che lo installa subito alla creazione dell'environment)_.
Per fare questo:

- Linux: `sudo apt-get install ffmpeg`
- MacOS: `brew install ffmpeg`
<!-- - Windows: `choco install ffmpeg` -->

Potrebbe essere necessario anche installare la libreria `tkinter`:

- MacOS: `brew install python3-tk`
- Linux: `sudo apt-get install python3-tk`

## Esecuzione del programma

Una volta abilitato l'ambiente virtuale creato in precedenza, sarà possibile lanciare per la prima volta il programma, sempre dalla directory principale, usando il comando:

`python3 main.py`

La prima apertura con il nuovo ambiente può occupare qualche minuto. Successivamente verrà mostrata una scelta tra le canzoni presenti nel dataset:

```
Which song would you like to analyze?:

> ASAP Rocky - RIOT
Burial - Archangel
Dolly Parton - Jolene
Jay-Z - The Story Of OJ
Kanye West - Jail
Lana Del Rey - Groupie Love ft. A$AP Rocky
Nada - Amore Disperato
Pink Floyd - The Great Gig In The Sky
Tyler, The Creator - IFHY
Vivian Roost - To the Sky
Astor Piazzolla - Libertango
Peggy Gou - It Goes Like (Nanana)
BUNT. - Clouds (ft. Nate Traveller)
...
```

Scorrendo in alto ed in basso con le frecce della tastiera, è possibile scorrere tra le possibili canzoni. Premere invio per confermare la scelta.

Successivamente, in base alla durata ed al peso della canzone selezionata, ci vorrà qualche minuto perché la canzone venga analizzata e le relative separazioni tra i vari elementi portata a termine.

Alla fine verrà visualizzata l'**immagine risultante** dall'analisi ed un player con cui sarà possibile ascoltare la traccia e nel frattempo osservare lo scorrere del tempo rispetto all'immagine.<br>
Tutti gli altri output si troveranno:

- `./timefunction-img` contenente una sotto-cartella per ogni canzone analizzata. Al loro interno si trovano le immagini risultanti dall'analisi di ogni elemento (vocal, drums, bass, others) e la canzone complessiva

- nella cartella `./separated_tracks` sono contenute le traccie audio risultanti dalla separazione della canzone nei suoi elementi _(drums.wav, vocals.wav, ...)_

- `img_spectral_map` che contiene l'immagine NON in funzione del tempo, ma con il colore associato in media rispetto all'intera traccia

Sul terminale invece, sono mostrati i dati calcolati come:

- durata
- BPM
- RMS
- LUFS
- energia del segnale
- pitch calcolato sull'intera canzone
- chiave
- danceability
- coefficiente usato per l'associazione del colore _(tutto spiegato meglio nella documentazione)_

Un esempio:

```
File: ASAP Rocky - RIOT

Duration: 3 min e 10 sec
BPM:  123

RMS: 0.1841
LUFS: -10.3398
Signal Energy: + 4839.337588470828
[...]
Pitch: 816.67 Hz
[...]
Key: G

Danceability: + 1.1934
Coefficent: + 7498.92
================================================================
```

## Ampliare il Dataset

È possibile ampliare il dataset messo a disposizione aggiungendo prima i file .wav nella cartella `./dataset/wav` e successivamente apportare una modifica al codice `main.py`, ovvero aggiungere il titolo della canzone (nome del file inserito) nell'array `data_set` a riga 78.
