import pickle
import re
import random

import numpy

random.seed(42)
numpy.random.seed(42)

import pandas
from nltk.stem import WordNetLemmatizer

import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer

def preprocessor(recipe):
    items = []
    for ingredient_str in recipe:
        ingredient = []
        for word in ingredient_str.split():
            word = re.sub('[^A-Za-z]', ' ', word).strip()
            word = WordNetLemmatizer().lemmatize(word)
            ingredient.append(word)
        for item in ingredient:
            items.append(item)
        items.append('x'.join(ingredient))
    return ' '.join(items)


from sklearn.base import TransformerMixin
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

import xgboost
from xgboost import XGBClassifier


def get_model():
    et_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words='english',
                                 ngram_range = ( 1 , 1 ),analyzer="word", 
                                 max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False,
                                preprocessor=preprocessor)),
            ('densetransformer', DenseTransformer()),
            ('classifier', XGBClassifier(silent=False, nthread=8, n_estimators=40, max_depth=3))
        ])
    return et_pipeline


def get_train_dataset():
    train = pandas.read_json('data/cooking_train.json')

    X_train = train['ingredients']
    y_train = train['cuisine']
    return X_train, y_train


def fit(et_pipeline):
    X_train, y_train = get_train_dataset()
    et_pipeline.fit(
        X_train[:100], y_train[:100],
        classifier__eval_set=(X_train[100:200], y_train[100:200])
    )


def answer(et_pipeline):
    test = pandas.read_json("data/cooking_test.json")

    ids = test['id']
    X_test = test['ingredients']

    y_test = et_pipeline.predict(X_test)

    df = pandas.DataFrame({"Id": ids, "cuisine": y_test})
    df.to_csv("data/my_submission.csv", index=False)


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.todense()


def main():
    print("get model")
    model = get_model()
    print("get dataset")
    X_train, y_train = get_train_dataset()
    print("train test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    print("fit")
    model.fit(
        numpy.array(X_train), numpy.array(y_train),
    )
    print("score")
    score = model.score(X_test, y_test)
    print(score)


main()
