## Identificazione della lingua delle descrizioni del museo

from nltk import ngrams

def my_ngram_function(sentence, n):
    ngrams_sentence = ngrams(sentence, n)
    return ["".join(grams) for grams in ngrams_sentence]

# prova della funzione ngram
print(my_ngram_function('elmo romano', 2))


## Caricamento del dataset

import pandas as pd
import numpy as np

dataset = pd.read_csv("https://raw.githubusercontent.com/Profession-AI/progetti-ml/refs/heads/main/Modello%20per%20l'identificazione%20della%20lingua%20dei%20testi%20di%20un%20museo/museo_descrizioni.csv")
print(dataset.head())
print(len(dataset))
print(np.unique(dataset['Codice Lingua']))


## Pulizia: tutto minuscolo e tolgo la punteggiatura

import string

def pulisci_testo(sentence):
    sentence = sentence.lower()
    for c in string.punctuation:
        sentence = sentence.replace(c, " ")
    return sentence

testi = []
for sentence in dataset['Testo']:
    testi.append(pulisci_testo(sentence))

print(testi[0])


## Bigrammi di caratteri, poi bag of words

ng_dataset = [" ".join(my_ngram_function(sentence, 2)) for sentence in testi]
print(ng_dataset[0][:180])

labels = dataset['Codice Lingua']

from sklearn.model_selection import train_test_split

X_train_txt, X_test_txt, y_train, y_test = train_test_split(ng_dataset, labels, test_size=0.2, random_state=42)
print(len(X_train_txt))
print(len(X_test_txt))

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
X_train = count_vectorizer.fit_transform(X_train_txt).toarray()
X_test = count_vectorizer.transform(X_test_txt).toarray()
print(X_train.shape)


## Naive Bayes

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train,y_train)
print(model.score(X_test, y_test))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

y_pred = model.predict(X_test)
print('Accuratezza:', accuracy_score(y_test, y_pred))
print('Precisione:', precision_score(y_test, y_pred, average='weighted'))
print('Richiamo:', recall_score(y_test, y_pred, average='weighted'))
print('F1-score:', f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))


## Frasi di prova che non sono nel dataset

frase = 'Statua in bronzo di epoca romana'
frase_ng = " ".join(my_ngram_function(pulisci_testo(frase), 2))
print(frase)
print(model.predict(count_vectorizer.transform([frase_ng]).toarray()))

frase = 'Bronze statue from the roman period'
frase_ng = " ".join(my_ngram_function(pulisci_testo(frase), 2))
print(frase)
print(model.predict(count_vectorizer.transform([frase_ng]).toarray()))

frase = 'Goldene Münze aus dem Mittelalter'
frase_ng = " ".join(my_ngram_function(pulisci_testo(frase), 2))
print(frase)
print(model.predict(count_vectorizer.transform([frase_ng]).toarray()))

print("Commento:\nSul test set l'accuratezza è circa 0.95. Precisione, richiamo e F1-score assumono valori molto vicini.\nIl richiamo è pari a 1.00 per il tedesco, a 0.95 per l'italiano e a 0.88 per l'inglese, che risulta la classe con il risultato più basso.\nI bigrammi di caratteri sono efficaci perché italiano, inglese e tedesco presentano combinazioni di lettere diverse.\nLe tre frasi di prova, non presenti nel dataset, sono state classificate correttamente.\nIl dataset contiene tuttavia solo 294 testi, tutti relativi a oggetti del museo: su testi di altro tipo le prestazioni potrebbero diminuire.")
