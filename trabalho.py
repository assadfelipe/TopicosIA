import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics, preprocessing, model_selection, neighbors
from sklearn.model_selection import cross_val_predict
import csv
from sklearn import svm
import numpy as np


############################## DECLARACAO DE VARIAVEIS ####################################################
testes = []
gabarito = []
nota_gabarito = []
nota_resp = []

############################################################################################################



################################################### FUNCOES ###############################################
def compara(origem, destino):
    soma = 0
    for i in range(len(origem)):
        parcial = origem[i] - destino[i]
        parcial = parcial*parcial
        soma += parcial
    return ((soma ** 0.5)/len(origem))

###########################################################################################################
        

########################## LEITURA DO ARQUIVO PARA APRENDIZADO DE MAQUINA #################################
dataset = pd.read_csv("Tweets_Mg.csv")
print('O dataset lido tem', dataset['id'].count(), 'tuplas')

###########################################################################################################

############################################### BAG OF WORDS ##############################################
tweets = dataset['Text'].values
classes = dataset['Classificacao'].values

sample = []
for t in classes: 
    nota = 0
    if t == 'Positivo':
        nota = 3
    elif t == 'Neutro':
        nota = 2
    elif t == 'Negativo':
        nota = 1
    sample.append(nota)
        

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

clf = neighbors.KNeighborsClassifier()
clf.fit(freq_tweets,classes)

accuracy = res.score(testes, gabarito)

print(accuracy)

exit(1)
#############################################################################################################


#################################### Nayve Bayes ############################################################

with open('testes.csv', 'r') as ficheiro:
    reader = csv.reader(ficheiro)
    for linha in reader:
        testes.append(linha[2])
        gabarito.append(linha[9])

freq_testes = vectorizer.transform(testes)
resp = modelo.predict(freq_testes)
resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)
print('Acuracia do modelo: ',metrics.accuracy_score(classes,resultados))


accuracy = clf.score(testes, gabarito)

print(accuracy)

exit(1)

#print('\n\nMatriz de Confusao\n')
#print (pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True),'')


for i in range(len(gabarito)):
    if gabarito[i] == 'Positivo':
        nota_gabarito.append(3)
    elif gabarito[i] == 'Neutro':
        nota_gabarito.append(2)
    elif gabarito[i] == 'Negativo':
        nota_gabarito.append(1)

    if resp[i] == 'Positivo':
        nota_resp.append(3)
    elif resp[i] == 'Neutro':
        nota_resp.append(2)
    elif resp[i] == 'Negativo':
        nota_resp.append(1)
		
erro_quadratico = compara(nota_gabarito, nota_resp)
print('\n\nerro quadratico: ', erro_quadratico)

###############################################################################################################


############################################### KNN ###########################################################

df = dataset
df.drop(['Username'],1, inplace=True)
df.drop(['User Screen Name'],1, inplace=True)
df.drop(['User Location'],1, inplace=True)
df.drop(['id'],1, inplace=True)
df.drop(['Created At'],1, inplace=True)
df.drop(['Geo Coordinates.latitude'],1, inplace=True)
df.drop(['Geo Coordinates.longitude'],1, inplace=True)


x = np.array(df['Text'])
y = np.array(df['Classificacao'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)

print(accuracy)