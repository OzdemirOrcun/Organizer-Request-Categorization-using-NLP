import os
import re

import warnings
warnings.simplefilter("ignore", UserWarning)

import matplotlib
from matplotlib import pyplot as plt

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from string import punctuation
import pydot

from nltk.corpus import brown
from gensim.models import Word2Vec
import multiprocessing

#from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.externals import joblib

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras import regularizers
from keras.models import Model
from keras.models import Sequential

from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import Adadelta


from keras.callbacks import Callback
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils.vis_utils import plot_model

import scipy
from scipy.sparse import hstack
#from tqdm import tqdm

#from nltk.tokenize import word_tokenizer
def encode(training_dataset):
    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    y = training_dataset['AltTanim']
    y = labelEncoder.fit_transform(y)
    training_dataset['subID'] = y


data_for_bag = pd.read_excel('./tum_dataset_modified_enumareted_for_dictionary_after_morph.xlsx')
data_for_bag = data_for_bag.sample(frac=1, random_state=54020)
data_for_bag.columns = data_for_bag.columns.str.strip()
data_for_bag['BodyProcessed3'] = data_for_bag['BodyProcessed3'].astype(str)


data_for_to_be_vectorized = pd.read_excel('./tum_dataset_modified_enumareted_for_dictionary_after_morph.xlsx')
data_for_to_be_vectorized = data_for_to_be_vectorized.sample(frac=1, random_state=54020)
data_for_to_be_vectorized.columns = data_for_to_be_vectorized.columns.str.strip()
data_for_to_be_vectorized['BodyProcessed3'] = data_for_to_be_vectorized['BodyProcessed3'].astype(str)
sentences = data_for_to_be_vectorized['BodyProcessed3'].astype(str)

with open('turkceveingilizceStopWordsandMore.txt', 'r') as f:
    turkceVeIngilizceStopWords = f.readlines()

df_list_score = []

"""Train test split operation"""
x_train, x_test, y_train, y_test = train_test_split(data_for_to_be_vectorized['BodyProcessed3'],
                                                        data_for_to_be_vectorized['subID'],
                                                        test_size=0.2,
                                                        random_state=54020,
                                                        stratify=data_for_to_be_vectorized['subID'])
#Optimizer
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

"""this can be outside of the function"""
max_words = 80000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data_for_to_be_vectorized['BodyProcessed3'])
train_tokenized_sequence = tokenizer.texts_to_sequences(x_train)
test_tokenized_sequence = tokenizer.texts_to_sequences(x_test)
max_length = 35
word_index = tokenizer.word_index
padded_train_tokenized_sequence = pad_sequences(train_tokenized_sequence, maxlen=max_length)
padded_test_tokenized_sequence = pad_sequences(test_tokenized_sequence, maxlen=max_length)
""""""
####

#pd.DataFrame(y_test).to_csv('./y_true.csv', index=False, encoding='utf-8')

def applyRNN(x_train, x_test, y_train, y_test, epoch, batch_size):


    embedding_dim = 400
    embedding_matrix = np.random.random((max_words,embedding_dim))
    input = Input(shape=(max_length,))
    x = Embedding(input_dim=max_words,output_dim=embedding_dim,input_length=max_length,weights=[embedding_matrix],trainable=True)(input)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100,return_sequences=True))(x)
    average_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    concataneted = concatenate([average_pool,max_pool])

    ####ASLA ACTIVATION FUNCTIONU RELU YAPMA#########
    output = Dense(5,activation='sigmoid')(concataneted)

    rnn_model = Model(inputs=input, outputs=output)
    rnn_model.compile(loss = 'sparse_categorical_crossentropy', optimizer= adadelta,metrics=['accuracy'])



    rnn_history = rnn_model.fit(x= x_train,y=y_train,validation_data=(x_test,y_test),
                                batch_size=batch_size,
                                epochs=epoch,
                                verbose=1,
                                validation_split=0.10)

    print('\n')
    print('Cross_Validation Rate = 0.10')

    printAccuracyResults(model=rnn_model, history=rnn_history, type="CNN",
                         x_test=x_test, y_test=y_test)


def applyANN(x_train,x_test,y_train,y_test, batch_size, epochs,type,vector_type):
    n = 4
    print("Enümere edilmiş veri tfidf vectorizer ile {n} n-gram {type} bazında vektörleniyor.".format(n=n, type=type))

    with open('turkceveingilizceStopWordsandMore.txt', 'r') as f:
        turkceVeIngilizceStopWords = f.readlines()

    vectorizer = TfidfVectorizer(max_features=80000,
                                 min_df=450,
                                 max_df=0.70,
                                 analyzer=vector_type,
                                 stop_words=turkceVeIngilizceStopWords,
                                 ngram_range=(1, n),
                                 # range(1,n) opsiyonu değiştirilebilir, 1'den n e kadar mı yoksa sadece n mi diye.
                                 sublinear_tf=True,
                                 )

    vectorizer.fit(x_train)
    tfidf_matrix_train = vectorizer.transform(x_train)
    tfidf_matrix_test = vectorizer.transform(x_test)

    print("\n")
    print("Mesajlar tfidf vectorizer ile {type} halinde vektörize edildi.".format(type=type))
    print("\n")
    print("Vektörize edilen featurelar input layera aktarılmak üzere aktarılıyor.")

    tfidf_matrix_train_for_observation = vectorizer.fit_transform(x_train).todense()

    features = vectorizer.get_feature_names()
    TFIDF_dataset = pd.DataFrame(tfidf_matrix_train_for_observation, columns=features)

    print('\n')
    print('############')
    print('TFIDF Information for {type}'.format(type=type))
    print(TFIDF_dataset.info())
    print('############')
    print('\n')

    print(TFIDF_dataset.dtypes.__len__())

    annClassifier_for_wordtfidf = Sequential()
    # Adding Input layer and first hidden layer
    annClassifier_for_wordtfidf.add(
        Dense(32, kernel_initializer='uniform', activation='relu', input_shape=(TFIDF_dataset.dtypes.__len__(),)))
    # Adding Second hidden layer which can be multiplied
    annClassifier_for_wordtfidf.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    # Adding output layer
    annClassifier_for_wordtfidf.add((Dense(5, kernel_initializer='uniform', activation='sigmoid')))

    annClassifier_for_wordtfidf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = annClassifier_for_wordtfidf.fit(tfidf_matrix_train, y_train, batch_size=batch_size, epochs=epochs,
                                              validation_split=0.10)
    print('\n')
    print('Cross_Validation Rate = 0.10')

    print(history.history.keys())

    print(("\n"))
    print("Accuracy vs Epochs Plot of ANN")
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy vs Epochs Plot of ANN')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    print("\n")

    print("Loss vs Epochs Plot of ANN")

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs Epochs Plot of ANN')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print("\n")
    print("Accuracy Score of ANN")
    score = annClassifier_for_wordtfidf.evaluate(tfidf_matrix_test, y_test, batch_size=150)
    print(score)
    writeScores(type='ANN', df_list=df_list_score, score=score[1])

def applyCNN(x_train,x_test,y_train,y_test,batch_size,epochs):

        embedding_dim = 400
        embedding_matrix = np.random.random((max_words, embedding_dim))
        filter_sizes = [2, 3, 5]
        num_filters = 256
        drop = 0.3


        

        inputs = Input(shape=(max_length,), dtype='int32')
        embedding = Embedding(input_dim=max_words,
                              output_dim=embedding_dim,
                              weights=[embedding_matrix],
                              input_length=max_length,
                              trainable=True)(inputs)

        reshape = Reshape((max_length, embedding_dim, 1))(embedding)
        conv_0 = Conv2D(num_filters,
                        kernel_size=(filter_sizes[0], embedding_dim),
                        padding='valid', kernel_initializer='normal',
                        activation='sigmoid')(reshape)

        conv_1 = Conv2D(num_filters,
                        kernel_size=(filter_sizes[1], embedding_dim),
                        padding='valid', kernel_initializer='normal',
                        activation='sigmoid')(reshape)
        conv_2 = Conv2D(num_filters,
                        kernel_size=(filter_sizes[2], embedding_dim),
                        padding='valid', kernel_initializer='normal',
                        activation='sigmoid')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(max_length - filter_sizes[0] + 1, 1),
                              strides=(1, 1), padding='valid')(conv_0)

        maxpool_1 = MaxPool2D(pool_size=(max_length - filter_sizes[1] + 1, 1),
                              strides=(1, 1), padding='valid')(conv_1)

        maxpool_2 = MaxPool2D(pool_size=(max_length - filter_sizes[2] + 1, 1),
                              strides=(1, 1), padding='valid')(conv_2)
        concatenated_tensor = Concatenate(axis=1)(
            [maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        output = Dense(units=5, activation='sigmoid')(dropout)

        cnnClassifier = Model(inputs=inputs, outputs=output)
        #adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        cnnClassifier.compile(optimizer=adadelta, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        batch_size = batch_size
        epochs = epochs

        history = cnnClassifier.fit(x=x_train,
                                              y=y_train,
                                              validation_data=(x_test, y_test),
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              verbose=1)
        print('\n')
        print('Cross_Validation Rate = 0.10')

        printAccuracyResults(model=cnnClassifier, history=history, type="CNN",x_test=x_test,y_test=y_test)


def applyRNNandCNN(x_train,x_test,y_train,y_test,batch_size,epochs):
    embedding_dim = 400
    embedding_matrix = np.random.random((max_words,embedding_dim))
    input = Input(shape=(max_length,))
    x = Embedding(input_dim=max_words,output_dim=embedding_dim,input_length=max_length,weights=[embedding_matrix],trainable=True)(input)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100,return_sequences=True))(x)
    x = Conv1D(32, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    output = Dense(5, activation= 'sigmoid')(conc)

    rnnPlusCnnModel = Model(inputs = input,outputs = output)
    rnnPlusCnnModel.compile(loss="sparse_categorical_crossentropy"
                            ,optimizer=adadelta, metrics=['accuracy'])
    history = rnnPlusCnnModel.fit(x=x_train,
                                              y=y_train,
                                              validation_data=(x_test, y_test),
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              verbose=1)
    print('\n')
    print('Cross_Validation Rate = 0.10')

    printAccuracyResults(model=rnnPlusCnnModel, history=history, type="CNN+RNN",x_test=x_test,y_test=y_test)

def applyPreTrainedWord2Vec(batch_size,epoch,x_train,y_train,x_test,y_test):
    from keras.initializers import Constant
    from gensim.models import KeyedVectors
    word_vectors = KeyedVectors.load_word2vec_format('trmodel', binary=True)

    filename = 'turkcePreTrainedWord2Vec.txt'
    word_vectors.save_word2vec_format(filename, binary=False)

    embeddings_index = {}
    f = open(os.path.join('','turkcePreTrainedWord2Vec.txt'),encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddings_index[word] = coefs
    f.close()

    num_words = len(word_index) + 1
    embedding_dim = 400
    embedding_matrix = np.zeros((num_words,embedding_dim))

    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    input = Input(shape=(max_length,))
    x = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_length, embeddings_initializer=Constant(embedding_matrix),
                  trainable=True)(input)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    average_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    concataneted = concatenate([average_pool, max_pool])

    ####ASLA ACTIVATION FUNCTIONU RELU YAPMA#########
    output = Dense(5, activation='sigmoid')(concataneted)

    rnn_model = Model(inputs=input, outputs=output)
    rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

    rnn_history = rnn_model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
                                batch_size=batch_size,
                                epochs=epoch,
                                verbose=1,
                                validation_split=0.10)

    print('\n')
    print('Cross_Validation Rate = 0.10')

    printAccuracyResults(model=rnn_model,history=rnn_history,type="PreTrainedRNN",x_test=x_test,y_test=y_test)

def applyKendiPreTrainedModelim(batch_size,epoch,x_train,y_train,x_test,y_test):
    embedding_dim = 300

    w2v = Word2Vec(sentences=sentences,size = embedding_dim,min_count=5,negative=15,iter=10)
    word_vectors = w2v.wv

    filename = 'kisiselPreTrainedTurkce.txt'
    word_vectors.save_word2vec_format(filename, binary=False)

    embeddings_index = {}
    f = open(os.path.join('', 'kisiselPreTrainedTurkce.txt'), encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddings_index[word] = coefs
    f.close()

    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    input = Input(shape=(max_length,))
    x = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_length,
                  weights=[embedding_matrix],
                  trainable=True)(input)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    average_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    concataneted = concatenate([average_pool, max_pool])

    ####ASLA ACTIVATION FUNCTIONU RELU YAPMA#########
    output = Dense(5, activation='sigmoid')(concataneted)

    rnn_model = Model(inputs=input, outputs=output)
    rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

    rnn_history = rnn_model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
                                batch_size=batch_size,
                                epochs=epoch,
                                verbose=1,
                                validation_split=0.10)

    print('\n')
    print('Cross_Validation Rate = 0.10')

    printAccuracyResults(model = rnn_model,history=rnn_history,type="OrcunW2VRNN",x_test=x_test,y_test=y_test)



def writeScores(type,df_list,score):
    scores = []
    scores.append(score)
    df_list.append(getScoresDataFrame(type = type,scores=scores))
    col = 0
    fileWriter = pd.ExcelWriter('./scores/DL/scores_DL_{type}.xlsx'.format(type = type))
    for dataframe in df_list:
        dataframe.to_excel(fileWriter,sheet_name='Validation',startrow=0,startcol=col)
        col = col + len(dataframe.index) + 1
    fileWriter.save()


def getScoresDataFrame(scores,type):
    scores_df = pd.DataFrame(scores,columns = ['Accuracy Score {type}'.format(type = type)]).astype('float64')
    #print(scores_df)
    return scores_df


def printAccuracyResults(history,model,type,x_test,y_test):
    print(history.history.keys())

    print(("\n"))
    print("Accuracy vs Epochs Plot of {type}".format(type = type))
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy vs Epochs Plot of {type}'.format(type = type))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    print("\n")

    print("Loss vs Epochs Plot of {type}".format(type = type))

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs Epochs Plot of {type}'.format(type = type))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print("\n")
    print("Accuracy Score of {type}".format(type = type))
    score = model.evaluate(x_test, y_test, batch_size=150)
    print(score)
    writeScores(type=type, df_list=df_list_score, score=score[1])


def run(type,epoch,batch_size,vector_type):
    if(vector_type == 'None' or vector_type == 'none'):
        if (type == 'RNN'):
            applyRNN(x_train=padded_train_tokenized_sequence,
                     x_test=padded_test_tokenized_sequence, y_train=y_train, y_test=y_test,
                     epoch=epoch, batch_size=batch_size)
        elif(type == 'PreTrainedRNN'):
            applyPreTrainedWord2Vec(x_train=padded_train_tokenized_sequence, x_test=padded_test_tokenized_sequence,
                                    y_train=y_train, y_test=y_test,
                                    batch_size=batch_size, epoch=epoch)
        elif (type == 'CNN'):
            applyCNN(x_train=padded_train_tokenized_sequence, x_test=padded_test_tokenized_sequence, y_train=y_train,
                     y_test=y_test, batch_size=batch_size, epochs=epoch)
        elif(type == 'RNN+CNN' or type == 'CNN+RNN'):
            applyRNNandCNN(x_train= padded_train_tokenized_sequence,x_test = padded_test_tokenized_sequence,y_train=y_train,y_test=y_test,
                           batch_size = batch_size, epochs = epoch)
        elif(type == 'OrcunW2VRNN'):
            applyKendiPreTrainedModelim(batch_size=batch_size,epoch=epoch,x_train=padded_train_tokenized_sequence,y_train=y_train,x_test=padded_test_tokenized_sequence,y_test=y_test)
    elif(vector_type == 'word' or vector_type == 'char'):
        if(type == 'ANN'):
            applyANN(x_train=x_train, x_test=x_test, y_train=y_train,
                 y_test=y_test, batch_size=batch_size, epochs=epoch,vector_type = vector_type,type = type)
    else:
        print("Invalid demand.")


run(type="ANN",epoch=15, batch_size=150,vector_type="char")
