# Identificazione automatica della lingua delle schede di un museo

Un museo internazionale cataloga le opere in più lingue. Capire a mano se una descrizione è in italiano, inglese o tedesco è lento e soggetto a errori, soprattutto quando la collezione cresce.

Ho addestrato un classificatore di testi che riconosce la lingua di una scheda a partire dalle sole combinazioni di lettere. Sul test set raggiunge circa il **95% di accuratezza**. Tre frasi mai viste in addestramento - una per lingua - sono state classificate correttamente.

Il modello è volutamente semplice: bigrammi di caratteri e Multinomial Naive Bayes. Non usa dizionari né modelli linguistici preaddestrati, e resta leggibile e facile da integrare in un catalogo già esistente.

## Risultati

Split 80/20, `random_state=42`. Metriche pesate sul test set.

| Metrica | Valore |
| --- | --- |
| Accuratezza | ~0.95 |
| Precisione, richiamo, F1 | molto vicini all'accuratezza |

Richiamo per lingua:

| Lingua | Richiamo |
| --- | --- |
| Tedesco | 1.00 |
| Italiano | 0.95 |
| Inglese | 0.88 |

L'inglese è la classe più difficile: condivide più sequenze di lettere con le altre due. Italiano, inglese e tedesco restano comunque separabili perché i bigrammi tipici (`ch`, `gl`, `sch`, `th`) non coincidono.

## Come funziona

1. **Pulizia.** Testo in minuscolo, punteggiatura sostituita da spazi.
2. **Bigrammi di caratteri.** Ogni scheda diventa una sequenza di coppie di lettere (`"elmo"` → `el lm mo`).
3. **Bag of words.** `CountVectorizer` conta quanto spesso compare ciascun bigramma.
4. **Classificazione.** `MultinomialNB` assegna il codice lingua: `it`, `en` o `de`.

I bigrammi di caratteri funzionano bene sulle lingue vicine anche quando i testi sono brevi, come le didascalie di un catalogo. Una parola sola contiene già abbastanza segnale per orientare il modello.

## Dataset

294 descrizioni di oggetti di museo, bilanciate su tre lingue (98 per classe). Colonne: `Testo` e `Codice Lingua`.

Fonte: [museo_descrizioni.csv](https://raw.githubusercontent.com/Profession-AI/progetti-ml/refs/heads/main/Modello%20per%20l'identificazione%20della%20lingua%20dei%20testi%20di%20un%20museo/museo_descrizioni.csv).

Una copia locale è in `dataset.txt`. Lo script scarica il CSV originale a ogni esecuzione.

## Stack

Python, pandas, NumPy, NLTK (`ngrams`), scikit-learn (`CountVectorizer`, `MultinomialNB`).

## Esecuzione

```bash
pip install pandas numpy nltk scikit-learn
python identificazione_lingua.py
```

Lo script stampa la forma del dataset, le metriche sul test set, il classification report e la predizione su tre frasi di prova.

## Limiti

Il corpus è piccolo e tutto nello stesso dominio: schede di opere. Su testi di altro tipo, o su lingue non presenti in addestramento, le prestazioni possono calare. Il modello non distingue varietà della stessa lingua e non gestisce testi misti.

## Struttura

```
identificazione_lingua.py   pipeline completa: pulizia, training, metriche, frasi di prova
dataset.txt                 copia locale del dataset
```
