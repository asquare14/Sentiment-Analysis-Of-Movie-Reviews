import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
from time import time
import re
import string
import os
from pprint import pprint
import collections
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
np.random.seed(37)
from nltk.tokenize import TweetTokenizer
import datetime
from scipy import stats
from nltk.util import ngrams
from sklearn.preprocessing import StandardScaler

from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

#import spacy
#nlp = spacy.load('en_core_web_sm',disable=['parser', 'tagger', 'ner'])
#from spacy.lang.en import English

class Pre_Processing:

    def read_file(self):
        df_train1 = pd.read_csv("train.csv", sep = ',')
        df_test = pd.read_csv("test.csv", sep = ',')
        #print(df_train1.head)
        return df_train1,df_test

    def drop_unecessary_columns(self,df_train1):
        df_train = df_train1.drop(['PhraseId','SentenceId'],axis=1)
        df_train.head()
        #print(df_train.head)
        return df_train

    def cleaning(self,df_train,s):
        s = str(s)
        s = s.lower()
        s = re.sub('\s\W',' ',s)
        s = re.sub('\W,\s',' ',s)
        s = re.sub(r'[^\w]', ' ', s)
        s = re.sub("\d+", "", s)
        s = re.sub('\s+',' ',s)
        s = re.sub('[!@#$_]', '', s)
        s = s.replace(",","")
        s = s.replace("[\w*"," ")
        s = re.sub(r'https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
        s = re.sub(r'\<a href', ' ', s)
        s = re.sub(r'&amp;', '', s) 
        s = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', s)
        s = re.sub(r'[^\x00-\x7f]',r'',s) 
        s = re.sub(r'<br />', ' ', s)
        s = re.sub(r'\'', ' ', s)
        return s

    def tokenizer(self,s): 
        return [w.text.lower() for w in nlp(s)]

    def stem_lemmatize(self,df_train,df_test):
        sentences = list(df_train.Phrase.values) + list(df_test.Phrase.values)
        sentences2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in sentences]
        #sentences2 = [[lem.lemmatize(word) for word in sentence.split(" ")] for sentence in sentences]
        for i in range(len(sentences2)):sentences2[i] = ' '.join(sentences2[i])
        #print(len(sentences2))
        return sentences2

    def tfidf(self,df_train,df_test,sentences2):
        tfidf = TfidfVectorizer(analyzer='word',
                                strip_accents = 'ascii', 
                                encoding='utf-8', 
                                ngram_range = (1,2), 
                                min_df = 3,
                                sublinear_tf = True)
        _ = tfidf.fit(sentences2)
        print(len(tfidf.get_feature_names()))
        train_phrases2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in list(df_train.Phrase.values)]
        for i in range(len(train_phrases2)):train_phrases2[i] = ' '.join(train_phrases2[i])
        train_df_flags = tfidf.transform(train_phrases2)
        test_phrases2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in list(df_test.Phrase.values)]
        for i in range(len(test_phrases2)):test_phrases2[i] = ' '.join(test_phrases2[i])
        test_df_flags = tfidf.transform(test_phrases2)
        return train_df_flags,test_df_flags



