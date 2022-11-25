"""
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
"""

import re
import langid
import pymorphy2
from stop_words import get_stop_words

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import RegexpTokenizer
from collections import Counter

import pandas as pd
import os


class TextPreproc:

    def __init__(self, data_path='dataset/train.tsv'):

        """
        input data must be a Dataframe with columns:
            - data.column[0]: str
            - data.column[1]: int (0/1)
        """

        self.data = self.load_data(data_path)
        preprocd_folder = 'preprocd_{}'.format(data_path.split('/')[0])

        if os.path.exists(preprocd_folder):
            self.preprocd_data = self.load_data('preprocd_{}'.format(data_path))
            with open("{}/word_list.txt".format(preprocd_folder), "r", encoding="utf-8") as output:
                word_list = [tmp for tmp in output][0].split(' ')

        else:
            os.makedirs(preprocd_folder)

            self.vocab = Counter()
            self.preprocd_data = self.data.copy(deep=True)
            self.preprocd_data[self.preprocd_data.columns[0]] = \
                self.preproc_data(self.data[self.data.columns[0]])
            self.preprocd_data.to_csv('preprocd_{}'.format(data_path), sep='\t', index=False)

            word_list = sorted(self.vocab, key=self.vocab.get, reverse=True)
            with open("{}/word_list.txt".format(preprocd_folder), "w", encoding="utf-8") as output:
                output.write(" ".join(word_list))

        word_list = sorted(word_list)
        self.vocab_to_int = {word: idx + 1 for idx, word in enumerate(word_list)}
        self.int_to_vocab = {idx: word for word, idx in self.vocab_to_int.items()}

    def load_data(self, data_path):

        """
        0 - ok
        1 - fake
        """

        df = pd.read_csv(data_path, sep='\t')
        df.columns = [name.lower() for name in df.columns]
        return df

    def preproc_data(self, text_corpus):

        '''
            text_corpus - list of messages, each message have str type
            out - list of preprocessed messages
        '''

        tokenizer = RegexpTokenizer(r'\w+')
        morph = pymorphy2.MorphAnalyzer()
        sent_tokenizer = PunktSentenceTokenizer()
        lemmatizer = WordNetLemmatizer()

        langid.set_languages(['en', 'ru'])
        stop_words_en = set().union(get_stop_words('en'), stopwords.words('english'))
        stop_words_ru = set().union(get_stop_words('ru'), stopwords.words('russian'))
        stop_words = list(set().union(stop_words_en, stop_words_ru))
        stop_words.sort()

        out = []
        for i, text in enumerate(text_corpus):

            text = re.sub(r'[^\w\s]+|[\d]+', r'', text).strip()
            sent_list = [sent for sent in sent_tokenizer.tokenize(text)]
            tokens = [word for sent in sent_list for word in tokenizer.tokenize(sent.lower())]
            lemmed_tokens = []

            for token in tokens:

                if langid.classify(token)[0] == 'en':
                    lemmed_tokens.append(lemmatizer.lemmatize(token))

                elif langid.classify(token)[0] == 'ru':
                    lemmed_tokens.append(morph.parse(token)[0].normal_form)

            clean_text = [token for token in lemmed_tokens if token not in stop_words]
            self.vocab.update(clean_text)
            out.append(" ".join(clean_text))

        return out

    def get_initial_data(self):
        return self.data

    def get_preprocd_data(self):
        return self.preprocd_data

    def get_vocab(self):
        return self.vocab_to_int


if __name__ == "__main__":
    model = TextPreproc()
    print(model.get_preprocd_data())
    #print(model.get_vocab())
