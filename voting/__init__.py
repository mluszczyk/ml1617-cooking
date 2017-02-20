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

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

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


def get_model():
    lr = LogisticRegression(C=5)
    linearsvc = LinearSVC(C=.5)
    et = ExtraTreesClassifier(n_estimators=50)

    et_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words='english',
                                 ngram_range = ( 1 , 1 ),analyzer="word", 
                                 max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False,
                                preprocessor=preprocessor)),
            ('classifier', VotingClassifier(estimators=[
                ('lr', lr),
                ('linearsvc', linearsvc),
                ('et', et),
            ]))
        ])
    et_pipeline.set_params(classifier__weights=[2, 2, 1])
    return et_pipeline

