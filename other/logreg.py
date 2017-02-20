import re
import random
import numpy

random.seed(42)
numpy.random.seed(42)

import pandas
from nltk.stem import WordNetLemmatizer

train = pandas.read_json('data/cooking_train.json')

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

recipes = train.ingredients[:6]

for r in recipes:
    print(preprocessor(r))
    print()


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


X_train = train['ingredients']
y_train = train['cuisine']

et_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False,
                            preprocessor=preprocessor)),
        ('classifier', LogisticRegression(C=5)) 
    ])


score = cross_val_score(et_pipeline, X_train, y_train)
print(score)
print(score.mean())

