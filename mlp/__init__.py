'''
Based on https://github.com/ogencoglu/WhatsCooking.

Original header:
Author         : Oguzhan Gencoglu
Contact        : oguzhan.gencoglu@topdatascience.com, oguzhan.gencoglu@tut.fi
Description    : 17th Place out of 1388 teams (top 2%) Submission for Kaggle What's Cooking Competition
'''

import json
import sys
import subprocess
import pickle
import numpy as np
import random

random.seed(42)
np.random.seed(42)

import keras.backend
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from nltk.stem.wordnet import WordNetLemmatizer
import re
import itertools
import os.path
from datetime import datetime
import pandas
from sklearn.model_selection import train_test_split


n_ensemble = 10


def k_to_one_hot(k_hot_vector):
    # This function converts k-hot target vector to one-hot target matrix
    
    classes = np.unique(k_hot_vector)
    one_hot_matrix = []
    
    for i in np.arange(len(classes)):
        row = (k_hot_vector == classes[i]).astype(int, copy = False)
        if len(one_hot_matrix) == 0:
            one_hot_matrix = row
        else:
            one_hot_matrix = np.vstack((one_hot_matrix, row))
            
    return classes, one_hot_matrix.conj().transpose()
    
    
def create_submission(test_ids, guess):
    # create submission in proper format
    
    sub = np.transpose(np.vstack((test_ids, guess)))
    sub = np.vstack((['id', 'cuisine'], sub))
    sub_file_name = 'submission_' + str(datetime.now())[0:16] +'.csv'
    sub_file_name = sub_file_name.replace(' ', '_')
    sub_file_name = sub_file_name.replace(':', '-')
    np.savetxt(sub_file_name, sub, delimiter=",", fmt="%s")
    
    return None  
    

def remove_numbers(ing):
    # remove numbers from ingredients
    
    return [[re.sub("\d+", "", x) for x in y] for y in ing]

    
def remove_special_chars(ing):
    # remove certain special characters from ingredients
   
    ing = [[x.replace("-", " ") for x in y] for y in ing] 
    ing = [[x.replace("&", " ") for x in y] for y in ing] 
    ing = [[x.replace("'", " ") for x in y] for y in ing] 
    ing = [[x.replace("''", " ") for x in y] for y in ing] 
    ing = [[x.replace("%", " ") for x in y] for y in ing] 
    ing = [[x.replace("!", " ") for x in y] for y in ing] 
    ing = [[x.replace("(", " ") for x in y] for y in ing] 
    ing = [[x.replace(")", " ") for x in y] for y in ing] 
    ing = [[x.replace("/", " ") for x in y] for y in ing] 
    ing = [[x.replace("/", " ") for x in y] for y in ing] 
    ing = [[x.replace(",", " ") for x in y] for y in ing] 
    ing = [[x.replace(".", " ") for x in y] for y in ing] 
    ing = [[x.replace(u"\u2122", " ") for x in y] for y in ing] 
    ing = [[x.replace(u"\u00AE", " ") for x in y] for y in ing] 
    ing = [[x.replace(u"\u2019", " ") for x in y] for y in ing] 

    return ing
    
    
def make_lowercase(ing):
    # make all letters lowercase for all ingredients
    
    return [[x.lower() for x in y] for y in ing]
    
    
def remove_extra_whitespace(ing):
    # removes extra whitespaces
    
    return [[re.sub( '\s+', ' ', x).strip() for x in y] for y in ing] 
    
    
def stem_words(ing):
    # word stemming for ingredients
    
    lmtzr = WordNetLemmatizer()
    
    def word_by_word(strng):
        
        return " ".join(["".join(lmtzr.lemmatize(w)) for w in strng.split()])
    
    return [[word_by_word(x) for x in y] for y in ing] 
    
    
def remove_units(ing):
    # remove certain words from ingredients
    
    remove_list = ['g', 'lb', 's', 'n']
        
    def check_word(strng):
        
        s = strng.split()
        resw  = [word for word in s if word.lower() not in remove_list]
        
        return ' '.join(resw)

    return [[check_word(x) for x in y] for y in ing] 
    

def extract_feats(ingredients, uniques):
    # each ingredient + each word as feature
    
    feats_whole = np.zeros((len(ingredients), len(uniques)))
    for i in range(len(ingredients)):
        for j in ingredients[i]:
            feats_whole[i, uniques.index(j)] = 1
            
    new_uniques = []
    for m in uniques:
        new_uniques.append(m.split())
    new_uniques = list(set(list(itertools.chain.from_iterable(new_uniques))))
    
    feats_each = np.zeros((len(ingredients), len(new_uniques))).astype(np.uint8)
    for i in range(len(ingredients)):
        for j in ingredients[i]:
            for k in j.split():
                feats_each[i, new_uniques.index(k)] = 1
            
    return np.hstack((feats_whole, feats_each)).astype(bool)
    
    
def load_model(feature_num):
    # load neural net model architectiure
    
    mdl = Sequential()
    mdl.add(Dense(512, init='glorot_uniform', activation='relu', 
                                        input_shape=(feature_num,)))
    mdl.add(Dropout(0.5))
    mdl.add(Dense(128, init='glorot_uniform', activation='relu'))
    mdl.add(Dropout(0.5))
    mdl.add(Dense(20, activation='softmax'))
    mdl.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return mdl    


def prepare_data(train_ingredients, train_cuisines, test_ingredients):
    # preprocess training set
    print("\nPreprocessing...\n")  
    train_ingredients = make_lowercase(train_ingredients)
    train_ingredients = remove_numbers(train_ingredients)
    train_ingredients = remove_special_chars(train_ingredients)
    train_ingredients = remove_extra_whitespace(train_ingredients)
    train_ingredients = remove_units(train_ingredients)
    train_ingredients = stem_words(train_ingredients)
    
    # preprocess test set
    test_ingredients = make_lowercase(test_ingredients)
    test_ingredients = remove_numbers(test_ingredients)
    test_ingredients = remove_special_chars(test_ingredients)
    test_ingredients = remove_extra_whitespace(test_ingredients)
    test_ingredients = remove_units(test_ingredients)
    test_ingredients = stem_words(test_ingredients)
    
    # encode   
    print("Encoding...\n")  
    le = LabelEncoder()
    targets = le.fit_transform(train_cuisines)
    classes, targets = k_to_one_hot(targets)
    
    # extract features
    print("Feature extraction...\n") 
    uniques = list(set([item for sublist in train_ingredients + test_ingredients for item in sublist]))
    train_feats = extract_feats(train_ingredients, uniques)
    test_feats = extract_feats(test_ingredients, uniques)

    return train_feats, targets, test_feats, le


def clear_weights():
    for ens in range(n_ensemble):
        try:
            os.remove("mlp_data/model{}.hdf5".format(ens))
        except FileNotFoundError:
            pass

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def prepare_train(X_train, y_train, val=None):
    with open("mlp_data/mlp-processed-data.pickle", "wb") as f:
        pickle.dump((X_train, y_train, val), f)

    
def load_split_features():
    with open("mlp_data/mlp-processed-data.pickle", "rb") as f:
        return pickle.load(f)


def train_one(train_X, train_y, val, ens):
    nb_epoch = 30
    batch_size = 512
    keras.backend.clear_session()
    print("\n\tTraining...", ens)
    model = load_model(train_X.shape[1])

    history = model.fit(train_X, train_y, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=val)
    model_name = 'mlp_data/model' + str(ens) + '.hdf5'
    model.save_weights(model_name, overwrite=True)
    with open("mlp_data/history{}.json".format(ens), "w") as f:
        json.dump(history.history, f)


def predict_one(ens):
    with open("mlp_data/mlp_predict_input.pickle", "rb") as f:
        test_X = pickle.load(f)
    model = load_model(test_X.shape[1])
    model_name = 'mlp_data/model' + str(ens) + '.hdf5'
    model.load_weights(model_name)

    proba = model.predict_proba(test_X)
    with open("mlp_data/mlp_predict_output.pickle", "wb") as f:
        pickle.dump(proba, f)

def predict(test_X, label_transformer):
    preds = []
    with open("mlp_data/mlp_predict_input.pickle", "wb") as f:
        pickle.dump(test_X, f)

    for ens in range(n_ensemble):
        keras.backend.clear_session()
        print("\nSubmission", ens)
        subprocess.check_call([sys.executable, "-m", "mlp.predict", str(ens)])
        with open("mlp_data/mlp_predict_output.pickle", "rb") as f:
            proba = pickle.load(f)

        preds.append(proba)
    keras.backend.clear_session()
    preds = sum(np.log(preds))
    guess = np.argmax(preds, axis=1)
    return label_transformer.inverse_transform(guess)

