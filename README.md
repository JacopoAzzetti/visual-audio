# Visual Audio

_Una rappresentazione visuale di tracce audio_

## Creazione dell'ambiente virtuale

Per iniziare, è necessario che sia installato `conda` sulla propria macchina (https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

Una volta installato ed utilizzabile, è necessario creare l'ambiente virtuale già predisposto con tutte le dipendenze richieste dal progetto. Dalla cartella principale del progetto (`./visual-audio`), lanciare il seguente comando:

`conda env create -f tools/visual-audio-env.yml`

Una volta completata l'installazione come indicato, attivare l'ambiente virtuale attraverso il comando:

`conda activate visual-audio`

    Se fosse necessario eliminare un ambiente virtuale creato, utilizzare il seguente comando:

    conda remove -n <ENV NAME> --all

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

Alla fine verrà visualizzato l'esito dell'analisi ed un player con l'**immagine risultante** che sarà possibile seguire mentre il player farà ascoltare la canzone.
Tutti gli altri output si possono trovare:

- nella cartella `./timefunction-img`, è presente una cartella con il nome della canzone analizzata contenente le immagini risultanti dall'analisi di ogni elemento (vocal, drums, bass, others) e la canzone complessiva
- nella cartella `./separated_tracks` sono contenuti gli output audio risultanti dalla separazione della canzone nei suoi elementi

## Ampliare il Dataset

È possibile ampliare il dataset messo a disposizione aggiungendo prima i file .wav nella cartella `./dataset/wav` e successivamente apportare una modifica al codice `main.py`, ovvero aggiungere il titolo della canzone (nome del file inserito) nell'array `data_set` a riga 78.
