""" Medical Text Classification model using tfidf and SVM """

import numpy as np
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm
from sklearn.metrics import f1_score, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier

import os
import string
import re

import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

import nltk
import ssl

''' SSl Certificate'''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
words = set(nltk.corpus.words.words())

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()

#Removes Non-English words
def clean_sent(sent):
    return " ".join(w for w in nltk.wordpunct_tokenize(sent) \
     if w.lower() in words or not w.isalpha())


def freq_words(path):
    ''' Function to calculate frequency of words in a document.'''

    #  This variable is used to hold the list of dictionaries that contains term frequency of each document.
    FreqDict_list = []

    #  Loop for reading a document file and doing string manipulation.
    for file in os.listdir(path):
        with open (path+file,'r',encoding='utf8') as f:
            text = f.readlines()
        str1=""
        for line in text:
            str1+=line.lower()

        # String Patterns -numbers and measurements are removed.
        delims = re.findall('[0-9]+', str1) + re.findall('[0-9]+[a-z]+', str1) + re.findall('[0-9]{1,}\.[0-9]{1,}[a-z]+', str1)
        # print(delims)
        if (delims != []):
            for delim in delims:
                str1 = str1.replace(delim, '')

        # String Patterns such as dates, email and measurements are removed.
        # delims = re.findall('\d{2}[/-:]\d{2}[/-:]\d{2,4}', str1) + re.findall("[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", str1) + re.findall('[0-9]{1,}\.[0-9]{1,}[a-z]+', str1) +  re.findall('[0-9]+[a-z]+', str1)
        # if (delims != []):
        #     for delim in delims:
        #         str1 = str1.replace(delim, ' ')

        #  Replacing string punctuation with a whitespace
        str2 = str1.translate(str1.maketrans(string.punctuation, ' '*len(string.punctuation)))

        #Calling cleansent function to remove meaningless words
        str2 = clean_sent(str2)


        #  Lemmatization of spacy tokenized tokens and Stopwords removed from token list.
        str3 = [token.lemma_ for token in nlp(str2)]
        td_list = [word for word in str3 if word not in stopwords.words('english')]

        '''Creating frequency of words dictionary for a document.'''
        freq_terms = {}
        for term in td_list:
            if term in freq_terms:
                freq_terms[term] += 1
            else:
                freq_terms[term] = 1

        '''Changing values of dictionary into its term frequency in the document.'''
        for term in freq_terms:
            freq_terms[term] = (freq_terms[term]/ len(freq_terms))

        FreqDict_list.append(freq_terms)

    '''Storing term frequency of all documents in a dataframe.'''
    df = pd.DataFrame(FreqDict_list)
    df['File'] = os.listdir(path)
    df.fillna(0, inplace=True)
    df.set_index('File', inplace=True)

    #Returning tf dataframe.
    return df

#Call freq_words function and passed CLINICAL DATA from directory.
path = "/Users/kirthanashri/PycharmProjects/DOCU CLINICAL CLASS/CLINICAL DATA/data/"
tf = freq_words(path)
print(tf.shape)

def tfidf(tf):
    ''' Function to calculate tfidf'''
    idf = tf.copy(deep=True)
    for c in idf.columns:
        idf[c] = np.log(tf.shape[0]/(1+idf.c[idf[c]!=0].count()))
    tfidf = idf.mul(tf, fill_value=0)
    return tfidf

#Call tfidf function.
tfidf = tfidf(tf)
print(tfidf)

#Reading trainlabels csv file.
df1 = pd.read_csv('/Users/kirthanashri/PycharmProjects/DOCU CLINICAL CLASS/trainLabels.csv')

#Renaming column names.
df1 = df1.rename(columns={'1001.txt':'File', 'Neurology':'med_domain'})

#Setting filenames as index and adding a record-'1001.txt'
df1.set_index('File', inplace=True)
df1.loc['1001.txt','med_domain'] = 'Neurology'

#Concatenating df1 and tfidf dataframes.
df = pd.concat([tfidf,df1], axis=1)
df = df.sort_index(axis=1)

#Dropping first 102 columns(Meaningless words)
df.drop(df.columns.to_list()[:102], axis=1,inplace=True)
'''Used to remove delims from this file - Which the above delims didnt detect'''
# print(df.columns.to_list()[:101])
df.drop('�', axis=1, inplace=True)

#Segregating training and testing datasets from df.
train_df = df[~df['med_domain'].isna()]
test_df = df[df['med_domain'].isna()]
print(train_df.shape, test_df.shape)

def FeatureSelection():
    '''
    return accuracy of model for different number of features

    For feature selection, two methods are used to analyse accuracy of the model.
    '''

    # FeatureScores- Univariate method.
    nf=500
    data = []
    while nf<=train_df.shape[1]:
        X = train_df.loc[:,train_df.columns!='med_domain']
        y = train_df['med_domain']

        #Selecting nf best features based on their feature score found using chisquare.
        bestfeatures = SelectKBest(score_func=chi2, k=nf)
        fit = bestfeatures.fit(X,y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['Specs','Score']

        featureScores = featureScores.nlargest(nf,'Score')
        featureScores['Specs'].to_list()

        X = np.asarray(train_df[featureScores['Specs'].to_list()])
        y = np.asarray(train_df['med_domain'])

        # Splitting dataset and training model.
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4)
        clf = svm.SVC(kernel='sigmoid')
        clf.fit(X_train,y_train)
        y_hat = clf.predict(X_test)

        # Finding accuracy of model using fscore and jaccardscore.
        a1 = f1_score(y_test, y_hat, average='weighted')
        a2 = jaccard_score(y_test, y_hat, labels= train_df['med_domain'].unique() , average='weighted')

        data.append([nf, a1, a2])
        nf+=500

    # Creating accuracy dataframe that shows accuracy of model for different number of features.
    acc_df = pd.DataFrame(data, columns=['Feature Nos', 'F1', 'Jaccord'])
    print('\n Univariate Method: \n',acc_df)

    # Feature Importance Method.
    nf=500
    data = []
    while nf<=train_df.shape[1]:
        X = train_df.loc[:,train_df.columns!='med_domain']
        y = train_df['med_domain']
        model = ExtraTreesClassifier()
        model.fit(X,y)

        feature_importances = pd.Series(model.feature_importances_, index=X.columns)

        # Selecting nf features based on their feature scores found using feature importance.
        X = train_df[feature_importances.nlargest(nf).index.to_list()]

        # Splitting dataset and training model.
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4)
        clf = svm.SVC(kernel='sigmoid')
        clf.fit(X_train,y_train)
        y_hat = clf.predict(X_test)

        # Finding accuracy of model using fscore and jaccardscore.
        a1 = f1_score(y_test, y_hat, average='weighted')
        a2 = jaccard_score(y_test, y_hat, labels= train_df['med_domain'].unique() , average='weighted')

        # data - List of lists containing FeatureNos, its f1score and jaccardscore.
        data.append([nf, a1, a2])
        nf+=500

    # Creating accuracy dataframe that shows accuracy of model for different number of features.
    acc_df = pd.DataFrame(data, columns=['Feature Nos', 'F1', 'Jaccord'])
    print('\n Feature Importance Method: \n',acc_df)


#Calling FeatureSelection function
FeatureSelection()
