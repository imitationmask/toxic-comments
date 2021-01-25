import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from module.predictor import Predictor
from module.trainer import Trainer


class NBTrainer(Trainer):

    def __init__(self, **kwargs):
        self.vectorizer = None
        self.vectorizer_name = ''
        self.classifiers = {
            'toxic': None,
            'severe_toxic': None,
            'obscene': None,
            'threat': None,
            'insult': None,
            'identity_hate': None,
        }

    def train(self, X, Y, test_size=None, **kwargs):
        self.classifiers['toxic'] = self.train_single_y(X, Y, 'toxic', test_size, **kwargs)
        self.classifiers['severe_toxic'] = self.train_single_y(X, Y, 'severe_toxic', test_size, **kwargs)
        self.classifiers['obscene'] = self.train_single_y(X, Y, 'obscene', test_size, **kwargs)
        self.classifiers['threat'] = self.train_single_y(X, Y, 'threat', test_size, **kwargs)
        self.classifiers['insult'] = self.train_single_y(X, Y, 'insult', test_size, **kwargs)
        self.classifiers['identity_hate'] = self.train_single_y(X, Y, 'identity_hate', test_size, **kwargs)
        return self.vectorizer, self.classifiers

    def train_single_y(self, X, Y, y_column, test_size=None, **kwargs):
        y = Y[y_column]
        random_state, vectorizer_name, ngram_range, stop_words, classifier_name = 0, 'CountVectorizer', (1, 1), None, 'MultinomialNB'
        for name, value in kwargs.items():
            if name == 'random_state':
                random_state = value
            elif name == 'vectorizer_name':
                vectorizer_name = value
            elif name == 'ngram_range':
                ngram_range = value
            elif name == 'stop_words':
                stop_words = value
            elif name == 'classifier_name':
                classifier_name = value

        self.vectorizer_name = vectorizer_name
        if vectorizer_name == 'CountVectorizer':
            self.vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
        elif vectorizer_name == 'TfidfVectorizer':
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop_words)
        else:
            raise Exception('training {}: undefined vectorizer {}'.format(y_column, vectorizer_name))

        if classifier_name == 'MultinomialNB':
            classifier = MultinomialNB()
        if test_size is None:
            X_train_count = self.vectorizer.fit_transform(X)
            classifier.fit(X_train_count, y)
        else:
            X_train, X_validation, y_train, y_validation = self.train_test_split(X, y, test_size, random_state)
            X_train_count, X_validation_count = self.fit(X_train, X_validation, self.vectorizer)
            self.predict(classifier, y_column, X_train_count, y_train, X_validation_count, y_validation)

        return classifier

    def train_test_split(self, X, y, test_size, random_state):
        X_train, X_validation, y_train, y_validation = tts(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_validation, y_train, y_validation

    def fit(self, X_train, X_validation, vectorizer):
        X_train_count = vectorizer.fit_transform(X_train)
        X_validation_count = vectorizer.transform(X_validation)
        return X_train_count, X_validation_count

    def predict(self, classifier, y_column, X_train_count, y_train, X_validation_count, y_validation):
        classifier.fit(X_train_count, y_train)
        prediction = classifier.predict(X_validation_count)
        print("MultinomialNB with vectorizer {} predicts column {} of training accuracy: {}".
              format(self.vectorizer_name, y_column, accuracy_score(y_validation, prediction)))

        return prediction


class NBPredictor(Predictor):

    def __init__(self, vectorizer, classifiers):
        self.vectorizer = vectorizer
        self.classifiers = classifiers
        self.predictions = {
            'toxic': None,
            'severe_toxic': None,
            'obscene': None,
            'threat': None,
            'insult': None,
            'identity_hate': None,
        }

    def predict(self, X_test):
        for column in self.predictions:
            self.predictions[column] = self.predict_singe_y(X_test, column)
        return self.predictions

    def predict_singe_y(self, X_test, y_column):
        X_test_count = self.vectorizer.transform(X_test)
        prediction = self.classifiers[y_column].predict(X_test_count)
        return prediction

    def save_prediction(self, test_id, predictions):
        df = pd.DataFrame({"id": test_id,
                           "toxic": predictions['toxic'],
                           "severe_toxic": predictions['severe_toxic'],
                           "obscene": predictions['obscene'],
                           "threat": predictions['threat'],
                           "insult": predictions['insult'],
                           "identity_hate": predictions['identity_hate'],
                           })
        df.to_csv('result/submission.csv', index=False, header=True)