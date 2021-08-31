import numpy as np
import pandas as pd
import logging

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.compose import ColumnTransformer

nltk.download('punkt')
nltk.download('stopwords')

class preprocess_helper:
    sw = set(stopwords.words('english'))
    
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        
    def get_df(self):
        return self.df
    
    def get_transformer(self, target_col = "text"):
        return ColumnTransformer(
            [('tfidf', self.get_vectorizer(), target_col)], remainder='passthrough'
        )

    def get_vectorizer(self):
        ngram_range=(1,3)
        vectorizer = TfidfVectorizer(stop_words=None, lowercase=False, #token_pattern = r'\b[a-z]{3,12}\b', 
                                         ngram_range=ngram_range)

        return vectorizer

    def tokenize_and_clean_text(self, text):
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if word.isalpha() and not word in self.sw]

        return " ".join(tokens_without_sw), len(text_tokens), len(tokens_without_sw)

    def process(self, text_col = 'original_text', label_col = 'label'):

        random_seed = 42
        test_size = 0.2
        
        logging.info("tokenizing and removing stopwords")
        self.df[["text", "original_length", "length"]] = self.df.apply(
            lambda col: self.tokenize_and_clean_text(col[text_col]), axis=1, result_type="expand"
        )
        
        self.df["length_diff"] = self.df["original_length"] - self.df["length"]
        
        X = self.df.drop(columns=[text_col, label_col])
        y = self.df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        
#         logging.info("transforming with TfidfVectorizer...")
#         X_train_trans = vectorizer.transform(X_train)
#         X_test_trans = vectorizer.transform(X_test)

        return X_train, X_test, y_train, y_test
