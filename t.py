# Importing the libraries
import numpy as np
import re
import pickle 
import nltk
#import heapq
from nltk.corpus import stopwords
from sklearn.datasets import load_files
import pandas as pd
import csv
#from gensim.models import Word2Vec

#Data=pd.read_excel('Restaurant - Copy.xlsx')
Data = pd.read_csv('Restaurant - Copy.csv')

Data= Data[Data['Polarity'] != 'neutral']
Data= Data[Data['Polarity'] != 'conflict']
#z=Data['Polarity'].values.astype('str')

X=Data.iloc[:,1].values
y=Data.iloc[:,3].values


for data in range(len(y)):
    if(y[data]=='positive'):
        y[data]=1
    elif(y[data]=='negative'):
        y[data]=0


'''
#Storing as pickle Files
with open('x.pickle','wb') as f:
    pickle.dump(X,f)
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
#Umpickling the dataset
with open('x.pickle','rb') as f:
    X=pickle.load(f)
with open('y.pickle','rb') as f:
    y=pickle.load(f)

'''

with open('stopwords-bn.txt', 'r', encoding='utf8') as bn:
    bangla_stop_words = [line.strip() for line in bn]

corpus = []
for i in range(0, len(X)):
    Data = re.sub(r'\W', ' ', str(X[i]))
    #review = review.lower()
    Data = re.sub(r'^br$', ' ', Data)
    Data = re.sub(r'\s+br\s+',' ',Data)
    Data = re.sub(r'\s+[a-z]\s+', ' ',Data)
    Data = re.sub(r'^b\s+', '', Data)
    Data = ' '.join([word for word in Data.split() if word not in bangla_stop_words])  
    Data = re.sub(r'\s+', ' ', Data)
    Data = re.sub(r'\॥’...-!?:-:-‘’/‘‘’।,', ' ', Data)
    corpus.append(Data) 
    

'''#word2vec
sentences = nltk.sent_tokenize(corpus)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
X = Word2Vec(sentences, min_count=1)
'''
# Creating the BOW model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = None)
X = vectorizer.fit_transform(corpus).toarray()



# Creating the Tf-Idf model directly
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = None)
X = vectorizer.fit_transform(corpus).toarray()



# word level tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{2,}', max_features = 2000, min_df = 3, max_df = 0.6)
X = vectorizer.fit_transform(corpus).toarray()



# ngram level tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{2,}', ngram_range=(3,3), max_features = 2000, min_df = 3, max_df = 0.6)
X = vectorizer.fit_transform(corpus).toarray()
#token_pattern=r'\w{,}',  # tokenize only words of 1+ chars
#ngram_range=(3,3), #traigram
#analyzer='word', #word by word calculation

# characters level tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(3,3), max_features = 2000, min_df = 3, max_df = 0.6)
X = vectorizer.fit_transform(corpus).toarray()

#analyzer='char', #charecter by charecter calculation


# Splitting the dataset into the Training set and Test set
y=y.astype('int')

from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
   
# Training the classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

# Testing model performance
sent_pred = classifier.predict(text_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test, sent_pred)
print (cm)  
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(sent_test,sent_pred)
print(accuracy*100)

# svm
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
clf_output = clf.fit(text_train, sent_train)

accuracySVM=clf.score(text_test, sent_test)
print(accuracySVM*100)

#naive bayes
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(text_train, sent_train)
nb.score(text_test, sent_test)
'''
confusion matrix:
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
'''