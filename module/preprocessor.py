import pandas as pd
import warnings
import numpy as np


class Preprocessor:

    def __init__(self):
        self.__train_df = None
        self.__train_labels_df = None
        self.X_train = None
        self.Y = {
                'toxic': None,
                'severe_toxic': None,
                'obscene': None,
                'threat': None,
                'insult': None,
                'identity_hate': None,
                }
        self.__test_df = None
        self.X_test = None
        self.test_id = None

    def process_test_file(self, test_file):
        self.__test_df = pd.read_csv(test_file)
        self.__test_df.name = 'test_df'
        self.__check_null_in_data(self.__test_df)
        self.__test_df = self.__test_df.astype({'comment_text': 'string'})
        self.X_test = self.__test_df['comment_text']
        self.test_id = self.__test_df['id']
        self.X_test = self.__remove_nonalnum(self.X_test)
        return self.test_id, self.X_test

    def process_train_files(self, train_file):
        self.__train_df = pd.read_csv(train_file)
        self.__train_df.name = 'train_df'
        self.__check_null_in_data(self.__train_df)
        final_test_df = self.__convert_data_types(self.__train_df)
        self.__split_X_Y(final_test_df)
        self.X_train = self.__remove_nonalnum(self.X_train)
        return self.X_train, self.Y

    def __split_X_Y(self, df):
        self.X_train = df['comment_text']
        self.Y['toxic'] = np.array(df['toxic']).reshape(-1)
        self.Y['severe_toxic'] = np.array(df['severe_toxic']).reshape(-1)
        self.Y['obscene'] = np.array(df['obscene']).reshape(-1)
        self.Y['threat'] = np.array(df['threat']).reshape(-1)
        self.Y['insult'] = np.array(df['insult']).reshape(-1)
        self.Y['identity_hate'] = np.array(df['identity_hate']).reshape(-1)

    def __check_null_in_data(self, df):
        for col, res in enumerate(df.isnull().any()):
            if res:
                warning_msg = 'Dataframe {} column {} has null values'.format(df.name, col)
                warnings.warn(warning_msg)

    def __convert_data_types(self, df):
        convert_dict = {'comment_text': 'string',
                        'toxic': 'category',
                        'severe_toxic': 'category',
                        'obscene': 'category',
                        'threat': 'category',
                        'insult': 'category',
                        'identity_hate': 'category',
                        }
        return df.astype(convert_dict)

    def __remove_nonalnum(self, X):
        corpus = []
        for x in X:
            x = x.lower()
            temp = ''
            for c in x:
                if c.isalnum() or c == ' ':
                    temp += c
            corpus.append(temp)
        return np.array(corpus)
