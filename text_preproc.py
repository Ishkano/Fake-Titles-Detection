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

    def __init__(self, data_path):
        """
        Initializing preprocessed data and vocabulary of words made on it

        input data must be a Dataframe with columns:
            - data.column[0]: str
            - data.column[1]: int (0/1)
        """

        preproc_folder_path = 'preproc_{}'.format(data_path.split('/')[0])
        if not os.path.exists(preproc_folder_path):
            os.makedirs(preproc_folder_path)

        if os.path.exists('preproc_{}'.format(data_path)):
            self.text, self.target = self.load_data('preproc_{}'.format(data_path))
            with open("{}/word_list.txt".format(preproc_folder_path), "r", encoding="utf-8") as output:
                word_list = [tmp for tmp in output][0].split(' ')
        else:
            self.text, self.target = self.load_data(data_path)
            self.text, self.vocab = self.preproc_text(self.text)

            pd.DataFrame({"text": self.text, "target": self.target}).to_csv('preproc_{}'.format(data_path),
                                                                            sep='\t',
                                                                            index=False)

            word_list = sorted(self.vocab, key=self.vocab.get, reverse=True)
            with open("{}/word_list.txt".format(preproc_folder_path), "w", encoding="utf-8") as output:
                output.write(" ".join(word_list))

        self.vocab = {word: idx for idx, word in enumerate(sorted(word_list))}

    def preproc(self, data_path):
        """
        func to preprocess any data given after initialization

        input data must be a Dataframe with columns:
            - data.column[0]: str
            - data.column[1]: int (0/1)
        """
        if os.path.exists('preproc_{}'.format(data_path)):
            text, target = self.load_data('preproc_{}'.format(data_path))
        else:
            text, target = self.load_data(data_path)
            text, vocab = self.preproc_text(text)

            pd.DataFrame({"text": text, "target": target}).to_csv('preproc_{}'.format(data_path),
                                                                  sep='\t',
                                                                  index=False)
        return text, target

    @staticmethod
    def load_data(data_path, encoding='utf_8'):
        """
        loading X and y data from file containing 2 columns:
            -data.column[0]: str
            -data.column[1]: int (0/1)
        """

        data_type = data_path.split('.')[-1]
        if data_type == "csv":
            df = pd.read_csv(data_path, encoding=encoding)
        elif data_type == "tsv":
            df = pd.read_csv(data_path, sep='\t', encoding=encoding)
        else:
            raise TypeError('data must be one of the following types: csv, tsv')

        return list(df[df.columns[0]]), list(df[df.columns[-1]])

    @staticmethod
    def preproc_text(text_corpus):

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

        vocab = Counter()

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
            vocab.update(clean_text)
            out.append(" ".join(clean_text))

        return out, vocab

    def get_initial_data(self):
        return self.text, self.target

    def get_vocab(self):
        return self.vocab


class BagOfWordsVectorizer:

    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text_corpus):
        """vectorization of the text using bag of words method"""

        out = []
        for i, text in enumerate(text_corpus):
            vec = [0] * len(self.vocab)
            for word in text.split(' '):
                try:
                    vec[self.vocab[word]] += 1
                except:
                    pass
            out.append(vec)

        return out


if __name__ == "__main__":
    preproc_model = TextPreproc(data_path='kinopoisk_data/train.csv')

    # print(preproc_model.get_initial_data()[0][:3], preproc_model.get_initial_data()[1][:3])
    # print(preproc_model.get_vocab())

    print(preproc_model.preproc(data_path='kinopoisk_data/test.csv'))

    # vocab = preproc_model.get_vocab()
    # data = preproc_model.get_initial_data()[0][:3]
    # print(data, [sum(BagOfWordsVectorizer(vocab).encode(data)[i]) for i in range(3)])
