{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (<ipython-input-1-48d09f6b16fb>, line 33)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"<ipython-input-1-48d09f6b16fb>\"\u001b[1;36m, line \u001b[1;32m33\u001b[0m\n\u001b[1;33m    return X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)\u001b[0m\n\u001b[1;37m                                            ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "from nltk.tokenize import sent_tokenize, word_tokenize\n",
    "from nltk.corpus import stopwords\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "def to_dataframe(filename):\n",
    "    return pd.read_csv(filename)\n",
    "\n",
    "def get_vectorizer(target_series):\n",
    "\n",
    "    vectorizer = TfidfVectorizer(stop_words=None, lowercase=False, #token_pattern = r'\\b[a-z]{3,12}\\b', \n",
    "                                     min_df=min_df, ngram_range=ngram_range).fit(target_series)\n",
    "\n",
    "    return vectorizer\n",
    "\n",
    "def tokenize_without_sw(text):\n",
    "    text_tokens = word_tokenize(text)\n",
    "\n",
    "    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]\n",
    "\n",
    "    return tokens_without_sw\n",
    "\n",
    "def process(filename, text_column = 'original_text'):\n",
    "    clean_text_column = 'clean_text'\n",
    "    df = to_dataframe(filename)\n",
    "\n",
    "    df[text_cleaned_column] = df[text_column].apply(lambda x: tokenize_without_sw(x))\n",
    "    vectorizer = get_vectorizer(target_series)\n",
    "\n",
    "    df[clean_text_column] = vectorizer.transform(df[clean_text_column])\n",
    "\n",
    "    df['length'] = df[clean_text_column].str.split().str.len()\n",
    "\n",
    "\n",
    "    X=df[clean_text_column]\n",
    "    y=train_df['label']\n",
    "\n",
    "    return X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
