import re

import numpy
import random
numpy.random.seed(42)

import pandas as pd
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from nltk.stem import WordNetLemmatizer

from utils import preprocess_words, Encoder, augment_permutations


from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Embedding, SimpleRNN


def augment(X_train, y_train, aug_max=5):
    X_aug = []
    y_aug = []
    for ingredient_list, y in zip(X_train, y_train):
        ext = augment_permutations(ingredient_list, aug_max, random.shuffle)
        X_aug.extend(ext)
        y_aug.extend([y] * len(ext))

    return X_train + X_aug, y_train + y_aug


def get_keras_model():
    max_features = 3000
    model = Sequential()
    model.add(Embedding(max_features, 128, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))

    model.add(Dense(20, W_regularizer=l2(0.01)))

    model.add(Activation("softmax"))
    assert model.output_shape == (None, 20)

    model.compile("adam", metrics=['accuracy'], loss='categorical_crossentropy')
    return model


non_char_regex = re.compile('[^a-z]')
class InputTransformer:
    def __init__(self):
        self.encoder = Encoder()

    def transform(self, X_train, y_train, augment):
        X_train = list(X_train)
        y_train = list(y_train)
        print('before augmenting', len(X_train))
        if augment is not None:
            X_train, y_train = augment(X_train, y_train)

        print('after augmetning', len(X_train), len(y_train))

        def word_func(word):
            word = non_char_regex.sub('', word.lower())
            # word = WordNetLemmatizer().lemmatize(word)
            return self.encoder.transform(word) + 1

        X_train = [preprocess_words(ingredients, word_func) for ingredients in X_train]
        lengths = numpy.array(list(len(x) for x in X_train))
        print(lengths.min(), lengths.mean(), lengths.max(), lengths.std())

        X_train = sequence.pad_sequences(X_train, maxlen=90)

        print("ingredients")
        print(X_train[:3])

        label_transform = LabelBinarizer()
        y_train = label_transform.fit_transform(y_train)

        return X_train, y_train


def main():
    train = pd.read_json('data/cooking_train.json')
    X_train, X_test, y_train, y_test = train_test_split(train.ingredients, train.cuisine)

    transformer = InputTransformer()
    X_train, y_train = transformer.transform(X_train, y_train, augment)
    X_test, y_test = transformer.transform(X_test, y_test, None)

    keras_model = get_keras_model()
    keras_model.fit(
        X_train, y_train, validation_data=(X_test, y_test), nb_epoch=40,
        callbacks=[TensorBoard(),
                   ModelCheckpoint("model-checkpoint.h5", save_best_only=True, monitor='val_acc')]
    )
 

if __name__ == '__main__':
    main()
