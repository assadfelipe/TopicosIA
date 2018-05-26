import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

dataset = pd.read_csv("Tweets_Mg.csv")
print('O dataset lido tem', dataset['id'].count(), 'tuplas')


tweets = dataset['Text'].values
classes = dataset['Classificacao'].values

ngram = 0
while(True):
    resp = int(input('tecle 1 para o modelo unigram e 2 para o modelo bigram: '))
    if resp == 1 or resp == 2:
        ngram = resp
        break


if ngram==1:
    vectorizer = CountVectorizer(analyzer="word")
else:
    vectorizer = CountVectorizer(ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets)
modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)


testes = ['Esse governo esta no inicio, vamos ver o que vai dar',
         'Estou muito feliz com o governo de Minas esse ano',
         'O estado de Minas Gerais decretou calamidade financeira!!!',
         'A seguranca desse pais esta deixando a desejar',
         'O governador de Minas e do PT']

freq_testes = vectorizer.transform(testes)
print(modelo.predict(freq_testes))
resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)
print(metrics.accuracy_score(classes,resultados))


