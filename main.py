import ast
from module.preprocessor import Preprocessor
from module.model.naive_bayes import NBTrainer, NBPredictor
from config import create_config


def run_naive_bayes():
    hyper_params = create_config('naive_bayes')
    train_file = hyper_params['preprocessing']['train_file']
    test_file = hyper_params['preprocessing']['test_file']
    test_size = hyper_params['training']['test_size']
    random_state = hyper_params['training']['random_state']
    vectorizer_name = hyper_params['training']['vectorizer']
    ngram_range = ast.literal_eval(hyper_params['training']['ngram_range'])
    stop_words = hyper_params['training']['stop_words']
    classifier_name = hyper_params['training']['classifier']

    preprocessor = Preprocessor()
    X_train, Y = preprocessor.process_train_files(train_file)
    test_id, X_test = preprocessor.process_test_file(test_file)

    nbtrainer = NBTrainer()
    nbtrainer.train_single_y(X_train, Y, 'toxic', test_size=test_size, random_state=random_state,
                             vectorizer_name=vectorizer_name, ngram_range=ngram_range, stop_words=stop_words,
                             classifier_name=classifier_name)

    nbtrainer.train_single_y(X_train, Y, 'severe_toxic', test_size=test_size)
    nbtrainer.train_single_y(X_train, Y, 'obscene', test_size=test_size)
    nbtrainer.train_single_y(X_train, Y, 'threat', test_size=test_size)
    nbtrainer.train_single_y(X_train, Y, 'insult', test_size=test_size)
    nbtrainer.train_single_y(X_train, Y, 'identity_hate', test_size=test_size)

    vectorizer, classifiers = nbtrainer.train(X_train, Y)
    nbpredictor = NBPredictor(vectorizer, classifiers)
    predictions = nbpredictor.predict(X_test)
    nbpredictor.save_prediction(test_id, predictions)


if __name__ == '__main__':
    run_naive_bayes()