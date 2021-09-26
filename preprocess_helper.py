import numpy as np
import pandas as pd
import logging

import itertools
import operator
import multiprocessing
from time import time

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.compose import ColumnTransformer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')

from gensim.models import Word2Vec

class preprocess_helper:
    sw = set(stopwords.words('english'))
    
    def __init__(self, filename, text_col = 'original_text', label_col = 'label'):
        self.df = pd.read_csv(filename)
        self.text_col = text_col
        self.label_col = label_col
        
        
    def get_df(self):
        return self.df
    
    def get_tfidf_transformer(self, target_col = "text"):
        return ColumnTransformer(
            [('tfidf', self.get_tfidf_vectorizer(), target_col)], remainder='passthrough'
        )

    def get_tfidf_vectorizer(self):
        ngram_range=(1,3)
        vectorizer = TfidfVectorizer(
            stop_words={'english'},
            lowercase=True,
            ngram_range=ngram_range,
            max_df=0.7,
            min_df=0.001
        )

        return vectorizer

    def accumulate(self, l):
        it = itertools.groupby(l, operator.itemgetter(1))
        for key, subiter in it:
           yield key, len([item[0] for item in subiter])
        
    def get_tagset(self):
        tagset = nltk.data.load('help/tagsets/upenn_tagset.pickle')
        return list(tagset.keys())
    
    def create_map(self, tag_counts):
        tag_count_map = dict.fromkeys(self.get_tagset(), 0)
        for k,v in tag_counts:
            if k in self.get_tagset():
                tag_count_map[k] = v
        
        return tag_count_map

    def tokenize_and_clean_text(self, text):
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if word.isalpha() and not word in self.sw]
        return " ".join(tokens_without_sw), len(text_tokens), len(text_tokens) - len(tokens_without_sw), self.create_map(self.accumulate(nltk.pos_tag(text_tokens)))
    
    def get_word2vec(self):
        w2v_model = Word2Vec(
            min_count=1,
            window=2,
            sample=6e-5,
            alpha=0.03,
            min_alpha=0.0007,
            negative=20,
            workers=4
        )
        t = time()
        sentences = [t.split(" ") for t in self.df["text"]]
        w2v_model.build_vocab(sentences, progress_per=10000)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
        return w2v_model


    def process_with_tokenization(self, sample_size = -1):
        
        if (sample_size > 0):
            self.df = self.df.sample(sample_size)
        
        logging.info("tokenizing and removing stopwords")
        self.df[["text", "original_length", "length_diff", "tag_count_map"]] = self.df.apply(
            lambda col: self.tokenize_and_clean_text(col[self.text_col].lower()), axis=1, result_type="expand"
        )
        
        print(self.df['tag_count_map'])
        self.df[self.get_tagset()] = self.df['tag_count_map'].apply(pd.Series)
        
        
        X = self.df.drop(columns=[self.text_col, self.label_col, "tag_count_map"])
        y = self.df[self.label_col]
        
        return X, y
    
    def train_test_split(self, X, y):

        random_seed = 42
        test_size = 0.2
        
        print(type(X))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        

        return X_train, X_test, y_train, y_test
    
    def split_text(text):
        return [t.split(" ") for t in text]
        
