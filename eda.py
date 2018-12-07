import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import seaborn as sns
import datetime as dt
import calendar
from scipy.stats import skew,kurtosis
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud, STOPWORDS

from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv("train.csv" , sep = ",")
df_test  = pd.read_csv("test.csv" , sep = ",")
#df_train.head()
print(len(df_train))
print(df_train.iloc[[0]])


def number_of_instances_of_each_sentiment():
    plt1 = df_train.groupby(["Sentiment"]).size()
    fig, ax = plt.subplots(figsize=(12,8))
    sns.barplot(plt1.keys(), plt1.values)
    plt.show()

def Mean(lst):
    return sum(lst) / len(lst)

def avg_len_phrase(): #against sentiment
    train_0_length = []
    train_1_length = []
    train_2_length = []
    train_3_length = []
    train_4_length = []

    train_0 = df_train[df_train['Sentiment'] == 0]
    train_0 = train_0['Phrase']



    for Phrase in train_0:
        train_0_length.append(len(Phrase))

    train_1 = df_train[df_train['Sentiment'] == 1]
    train_1 = train_1['Phrase']
    for Phrase in train_1:
        train_1_length.append(len(Phrase))


    train_2 = df_train[df_train['Sentiment'] == 2]
    train_2 = train_2['Phrase']
    for Phrase in train_2:
        train_2_length.append(len(Phrase))


    train_3 = df_train[df_train['Sentiment'] == 3]
    train_3 = train_3['Phrase']
    for Phrase in train_3:
        train_3_length.append(len(Phrase))


    train_4 = df_train[df_train['Sentiment'] == 4]
    train_4 = train_4['Phrase']
    for Phrase in train_4:
        train_4_length.append(len(Phrase))

    train_length_mean = [Mean(train_0_length) , Mean(train_1_length), Mean(train_2_length), Mean(train_3_length), Mean(train_4_length)]

    y = train_length_mean
    x = [0,1,2,3,4]

    plt.bar(x, y)
    plt.xlabel('Sentiment', fontsize=5)
    plt.ylabel('Average Length of Reviews', fontsize=5)
    plt.show()




#Word CLouds for different sentiments.

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=1800,
                      height=900
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(12,10))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


def wordcloud_helper(n):
    train_0 = df_train[df_train['Sentiment'] == n]
    train_0 = train_0['Phrase']
    print("Words with Sentiment 0")
    wordcloud_draw(train_0,'white')

from sklearn.feature_extraction.text import CountVectorizer

def most_common_words():
    cvector = CountVectorizer(min_df = 0.0, max_df = 1.0, ngram_range=(1,2))
    cvector.fit(df_train.Phrase)
    print(len(cvector.get_feature_names()))
    Sentiment_matrix = cvector.transform(df_train['Phrase'])
    Sentiment_words = Sentiment_matrix.sum(axis=0)
    Sentiment_words_freq = [(word, Sentiment_words[0, idx]) for word, idx in cvector.vocabulary_.items()]
    Sentiment_tf = pd.DataFrame(list(sorted(Sentiment_words_freq, key = lambda x: x[1], reverse=True)),columns=['Terms','Phrase'])
    index = np.arange(len(Sentiment_tf.Terms[:10]))
    plt.bar(index, Sentiment_tf.Phrase[:10])
    plt.xticks(index, Sentiment_tf.Terms, fontsize=10, rotation=30)
    plt.show()

if __name__ == '__main__':
    number_of_instances_of_each_sentiment()
    avg_len_phrase()
    wordcloud_helper()
    most_common_words()
