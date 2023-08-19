import numpy as np
import pickle as pickle
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras import Model

from dga.morphological_features import morphol_features

import warnings
warnings.simplefilter("ignore")

#-----------------------------------


#[1 0 1 1 1 0] legit legit cryptolocker tinba simda
#exam =['kjmlkjynercs','gfihghgbidxl','swnfiepiyksn','euwdtwxijnrb','google','tmall','qq']


#lay tu dien
DICTIONARY = 'dga/dictionary.pkl'
max_features = 38
maxlen = 57


#-----------------------------------
Model_file = 'dga/lstm.h5'
def load_model(Modelfile):
    # model
    model = Sequential()  # create a basic neural network model
    model.add(Embedding(max_features, 128,
                        input_length=maxlen))  # adds an embedding layer(converts each character into a vector of 128 floats)
    model.add(LSTM(128))  # adds an LSTM layer
    model.add(Dropout(0.5, name='feature'))  # Dropout: prevent overtraining
    model.add(Dense(1))
    model.add(Activation('sigmoid'))  # squash the output of this layer between 0 and 1
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')

    # lay dac trung tu layer cuoi sau khi huan luyen lstm
    model.load_weights(Modelfile)
    model_output = model.get_layer("feature").output
    m = Model(inputs=model.input, outputs=model_output)
    return m

# Convert characters to int and pad
#domains = ['mytest','xkjhqoucnxizdefezeiguontwpn']

def features_combination(dom):
    with open(DICTIONARY, "rb") as f:
        dic = pickle.load(f)
    valid_chars = {x: idx + 1 for idx, x in enumerate(dic)}
    # print(valid_chars)
    # dac trung noi ham
    vec = [[valid_chars[y] for y in x] for x in dom]  # chuyen domain thanh vector
    vec = pad_sequences(vec, maxlen=maxlen)  # pad de co do dai bang nhau

    lstm_feat = load_model(Model_file).predict(vec)
    # print('lstm_feat:',lstm_feat)
    #print (lstm_feat)
    # dac trung hinh thai
    morphol_feat = morphol_features(dom)
    # print('morphol_feat:',morphol_feat)
    #print (morphol_feat)
    # ket hop dac trung
    X = np.concatenate((lstm_feat, morphol_feat), axis=1)
    return X

#domains = extract_data()
#print(domains)
#with open(DICTIONARY, "rb") as f:
#    dic = pickle.load(f)
#valid_chars = {x: idx + 1 for idx, x in enumerate(dic)}
# print(valid_chars)
# dac trung noi hamd
#vec = [[valid_chars[y] for y in x] for x in domains]  # chuyen domain thanh vector
#print(vec)
#vec = pad_sequences(vec, maxlen=maxlen)  # pad de co do dai bang nhau
#print(vec)
#lstm_feat = load_model(Model_file).predict(vec)
# print('lstm_feat:',lstm_feat)
#print (lstm_feat)
#dac trung hinh thai
#morphol_feat = morphological_features.morphol_features(domains)
#print(len(morphol_feat))
#print('morphol_feat:',morphol_feat)
#print (morphol_feat)
# ket hop dac trung
#X = np.concatenate((lstm_feat, morphol_feat), axis=1)
