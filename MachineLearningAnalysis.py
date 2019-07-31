# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:50:20 2019

@author: orcun.ozdemir
"""

import os
import re

import warnings

warnings.simplefilter("ignore", UserWarning)
from matplotlib import pyplot as plt

import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np
from string import punctuation
from sklearn_evaluation import plot

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.externals import joblib

from gensim.corpora import Dictionary
import scipy

from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import Word2Vec

from scipy.sparse import hstack
from tqdm import tqdm

from wordcloud import WordCloud

# from nltk.tokenize import word_tokenizer



training_dataset = pd.read_excel('tum_dataset_modified_enumareted_for_dictionary_after_morph.xlsx')
training_dataset = training_dataset.iloc[:, 1:]



data_for_bag = pd.read_excel('./tum_dataset_modified_enumareted_for_dictionary_after_morph.xlsx')
data_for_bag = data_for_bag.sample(frac=1, random_state=0)
data_for_bag.columns = data_for_bag.columns.str.strip()


message = data_for_bag['BodyProcessed3']

"""
frequency_dictionary = gensim.corpora.Dictionary(message)
count = 0
for k, v in frequency_dictionary.iteritems():
    print(k,v)
    count += 1
    if count > 10:
        break
"""



data_for_bag['BodyProcessed3'] = data_for_bag['BodyProcessed3'].astype(str)


with open('turkceveingilizceStopWordsandMore.txt', 'r') as f:
    turkceVeIngilizceStopWords = f.readlines()




"""
for row in data_for_bag.head(10).iterrows():
    print(row[1]['subID'], row[1]['Mesaj:']) 
"""

"""Train test split operation"""
x_train, x_test, y_train, y_test = train_test_split(data_for_bag['BodyProcessed3'],
                                                    data_for_bag['subID'],
                                                    test_size=0.2,
                                                    random_state=0,
                                                    )

#pd.DataFrame(y_test).to_csv('./y_true.csv', index=False, encoding='utf-8')

def encode(training_dataset):
    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    y = training_dataset['AltTanim']
    y = labelEncoder.fit_transform(y)
    training_dataset['subID'] = y


#########################
def vectorized_n_gram(n, x_train, x_test, y_train, y_test, type):
    """Bag of Words based on word ngrams"""
    ## Şimdilik char_wb için tfidf bow implement edilmedi.

    ############################################################

    babaTest = pd.read_excel("TOP Konular mail içerikleri_test.xlsx")
    babaTest = babaTest.astype(str)
    x_baba_test = babaTest['Body']

    encode(babaTest)
    y_baba_test = babaTest['subID']

    #############################################################



    if (type == 'char_wb'):
        vectorizer_word = TfidfVectorizer(max_features=80000,
                                          min_df=300,
                                          max_df=0.90,
                                          analyzer='word',
                                          stop_words=turkceVeIngilizceStopWords,
                                          ngram_range=(1, n),
                                          # range(1,n) opsiyonu değiştirilebilir, 1'den n e kadar mı yoksa sadece n mi diye.
                                          sublinear_tf=True,
                                          )
        vectorizer_word.fit(x_train)
        tfidf_matrix_word_train = vectorizer_word.transform(x_train)
        tfidf_matrix_word_test = vectorizer_word.transform(x_test)



        ######################
        transformed_word_x_baba_test = vectorizer_word.transform(x_baba_test)
        ######################

        vectorizer_char = TfidfVectorizer(max_features=80000,
                                          min_df=300,
                                          max_df=0.90,
                                          analyzer='char',
                                          stop_words=turkceVeIngilizceStopWords,
                                          ngram_range=(1, n),
                                          # range(1,n) opsiyonu değiştirilebilir, 1'den n e kadar mı yoksa sadece n mi diye.
                                          sublinear_tf=True,
                                          )

        vectorizer_char.fit(x_train)
        tfidf_matrix_char_train = vectorizer_char.transform(x_train)
        tfidf_matrix_char_test = vectorizer_char.transform(x_test)

        ######################
        transformed_char_x_baba_test = vectorizer_char.transform(x_baba_test)
        #######################

        tfidf_matrix_word_char_train = hstack((tfidf_matrix_word_train, tfidf_matrix_char_train))  # x_train
        tfidf_matrix_word_char_test = hstack((tfidf_matrix_word_test, tfidf_matrix_char_test))  # x_test


        ##########################
        x_baba_test_stack = hstack(((transformed_word_x_baba_test,transformed_char_x_baba_test)))

        observeBagOfWords(n=n, x_train=x_train, type=type)
        print('\n')

        #xgBoostParameterTuning(n=n, x_train=tfidf_matrix_word_char_train, x_test=tfidf_matrix_word_char_test,y_train=y_train, y_test=y_test, type=type)


        applyTraditionalMachineLearningModels(n=n, x_train=tfidf_matrix_word_char_train,
                                             x_test=tfidf_matrix_word_char_test, y_train=y_train, y_test=y_test,
                                                type=type)


    elif (type == 'char' or type == 'word'):
        vectorizer = TfidfVectorizer(max_features=80000,
                                     min_df=300,
                                     max_df=0.90,
                                     analyzer=type,
                                     stop_words=turkceVeIngilizceStopWords,
                                     ngram_range=(1, n),
                                     # range(1,n) opsiyonu değiştirilebilir, 1'den n e kadar mı yoksa sadece n mi diye.
                                     sublinear_tf=True,
                                     )

        vectorizer.fit(x_train)
        tfidf_matrix_train = vectorizer.transform(x_train)
        tfidf_matrix_test = vectorizer.transform(x_test)


        tfidf_matrix_train_for_observation = vectorizer.fit_transform(x_train).todense()

        features = vectorizer.get_feature_names()
        TFIDF_dataset = pd.DataFrame(tfidf_matrix_train_for_observation, columns=features)

        ###TFIDF HISTOGRAM

        tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
        tfidf.columns = ['tfidf']
        #tfidf.hist(bins=25, figsize=(15, 7))

        #################################
        transformed_x_baba_test = vectorizer.transform(x_baba_test)
        #################################
        """
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=50,random_state=54020)
        svd_tfidf = svd.fit_transform(tfidf_matrix_train)

        from sklearn.manifold import TSNE
        tsne_model = TSNE(n_components=2,verbose=1,random_state=54020,n_iter=500)
        tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
        tsne_tfidf_df = pd.DataFrame(tsne_tfidf)
        tsne_tfidf_df.columns = ['x', 'y']
        tsne_tfidf_df['AltTanim'] = training_dataset['AltTanim']
        tsne_tfidf_df['Mesaj:'] = training_dataset['Mesaj:']

        groups = tsne_tfidf_df.groupby('AltTanim')
        fig,ax = plt.subplots(figsize =(15,10))
        ax.margins(0.05)
        for name, group in groups:
            ax.plot(group.x,group.y,marker = 'o',linestyle='',label= name)
        ax.legend()
        plt.show()
        """

        print('\n')
        print('############')
        print('TFIDF Information for {type}'.format(type=type))
        print(TFIDF_dataset.info())
        print('############')
        print('\n')

        #observeBagOfWords(n=n, x_train=x_train, type=type)
        print('\n')
        print('################################')
        print('Applying Traditional Machine Learning Models')
        print('\n')

        from sklearn.decomposition import LatentDirichletAllocation

        lda = LatentDirichletAllocation(n_components=4, max_iter=5, learning_method='online', learning_offset=50.,
                                        random_state=0).fit(tfidf_matrix_train)

        """
        print('\n')
        print("######################### LDA FOR TFIDF ############################")
        display_topics(lda, vectorizer.get_feature_names(), 10)
        print('\n')
        """

        applyTraditionalMachineLearningModels(n=n, x_train=tfidf_matrix_train, x_test=tfidf_matrix_test,y_train=y_train, y_test=y_test, type=type)
        #xgBoostParameterTuning(n=n, x_train=tfidf_matrix_train, x_test=tfidf_matrix_test,y_train=y_train, y_test=y_test, type=type)

        #applyTest(x_test=tfidf_matrix_test,y_train=y_train,y_test=y_test,x_train=tfidf_matrix_train)

        ##lowest tfidf wordcloud
        #get_word_cloud(tfidf.sort_values(by=['tfidf'],ascending=True).head(100))
        ##highest tfidf wordcloud
        #get_word_cloud(tfidf.sort_values(by = ['tfidf'],ascending= False).head(40))

# vectorized_word_n_gram(n=3,x_train = x_train,x_test = x_test,y_train = y_train,y_test = y_test)
############################


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def observeBagOfWords(n, x_train, type):
    ################################## Bag of Words için CounterVectorizer yaratıldı. ######################
    if (type == 'word' or type == 'char'):
        bow_vectorizer = CountVectorizer(max_features=40000,
                                         min_df=250,
                                         max_df=0.8,
                                         analyzer=type,
                                         stop_words=turkceVeIngilizceStopWords,
                                         ngram_range=(1, n),
                                         # range(1,n) opsiyonu değiştirilebilir, 1'den n e kadar mı yoksa sadece n mi diye.
                                         )




        vectorized_matrix_train_for_bow = bow_vectorizer.fit_transform(x_train).todense()

        word_indexes = bow_vectorizer.vocabulary_

        BagOfWords = pd.DataFrame(vectorized_matrix_train_for_bow, columns=word_indexes)

        from sklearn.decomposition import LatentDirichletAllocation

        lda = LatentDirichletAllocation(n_components = 5,max_iter=5,learning_method='online',learning_offset=50.,random_state=0).fit(vectorized_matrix_train_for_bow)



        print('\n')
        print("######################## LDA FOR BAG OF WORDS ############################")
        display_topics(lda,bow_vectorizer.get_feature_names(),10)


        print('\n')
        print('############')
        print('Bag Of Words Information for {type}'.format(type=type))
        print(BagOfWords.info())
        print('############')
        print('\n')


def xgBoostParameterTuning(n, x_train, x_test, y_train, y_test, type=type):
    """XGBoost"""

    from xgboost import XGBClassifier
    xgboostclassifier = XGBClassifier(tree_method="exact", predictor="cpu_predictor",
                                      objective="multi:softmax")
    xgboostclassifier.fit(x_train, y_train)
    y_pred_xgboost = xgboostclassifier.predict(x_test)

    # Create parameter grid
    parameters = {"learning_rate": [0.1,0.05, 0.001],
                  "gamma": [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
                  "max_depth": [2, 4, 7, 10],
                  "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
                  "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
                  "reg_alpha": [0, 0.5, 1],
                  "reg_lambda": [1, 1.5, 2, 3, 4.5],
                  "min_child_weight": [1, 3, 5, 7],
                  "n_estimators": [100, 250, 500, 1000]}

    from sklearn.model_selection import RandomizedSearchCV
    # Create RandomizedSearchCV Object
    xgb_rscv = RandomizedSearchCV(xgboostclassifier, param_distributions=parameters, scoring="f1_micro",
                                  cv=10, verbose=3, random_state=40)

    model_xgboost = xgb_rscv.fit(x_train, y_train)

    # Model best estimators
    print("Learning Rate: ", model_xgboost.best_estimator_.get_params()["learning_rate"])
    print("Gamma: ", model_xgboost.best_estimator_.get_params()["gamma"])
    print("Max Depth: ", model_xgboost.best_estimator_.get_params()["max_depth"])
    print("Subsample: ", model_xgboost.best_estimator_.get_params()["subsample"])
    print("Max Features at Split: ", model_xgboost.best_estimator_.get_params()["colsample_bytree"])
    print("Alpha: ", model_xgboost.best_estimator_.get_params()["reg_alpha"])
    print("Lamda: ", model_xgboost.best_estimator_.get_params()["reg_lambda"])
    print("Minimum Sum of the Instance Weight Hessian to Make a Child: ",
          model_xgboost.best_estimator_.get_params()["min_child_weight"])
    print("Number of Trees: ", model_xgboost.best_estimator_.get_params()["n_estimators"])


# measure_accuracy(name="XGBoost",y_test=y_test,y_pred=y_pred_xgboost,type=type,n=n,list=dataframe_lists_for_scores)


def applyTest(x_train,y_train,x_test,y_test):

    """
    test_dataset = pd.read_excel('./other/TOP Konuları dışındaki mail içerikleri_test.xlsx')

    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    y = test_dataset['AltTanim']
    y = labelEncoder.fit_transform(y)
    test_dataset['subID'] = y
    #print(test_dataset['Mesaj:'])

    v = vectorizer.transform(test_dataset['Mesaj:'])
    """



    from sklearn.svm import SVC
    svm_classifier = SVC(kernel='linear', random_state=0, decision_function_shape='ovo')
    svm_classifier.fit(x_train, y_train)

    y_pred_svm_classifier = svm_classifier.predict(x_test)

    """
    test_dataset['Predicates'] = y_pred_svm_classifier
    
    
    print(test_dataset['Predicates'])
    print(y)

    pred = test_dataset['Predicates']
    sub = y
    
    """

    print("SVM ACCURACY SCORE FOR TOP KONULARI DIŞINDAKİ MAİL İÇERİKLERİ")
    print(accuracy_score(y_test,y_pred_svm_classifier))




############################
def applyTraditionalMachineLearningModels(n, x_train, x_test, y_train, y_test, type=type):
    dataframe_lists_for_scores = []

    """Logistic Regression"""
    lr_word = LogisticRegression(solver='sag', verbose=2, random_state=54020, max_iter=1000, n_jobs=25)
    lr_word.fit(x_train, y_train)

    y_pred_word_lr = lr_word.predict(x_test)


    measure_accuracy(name='Logistic Regression Classifier', y_test=y_test, y_pred=y_pred_word_lr, n=n, type=type,list=dataframe_lists_for_scores)

    """kNN Classification"""
    from sklearn.neighbors import KNeighborsClassifier
    knn_word = KNeighborsClassifier(n_neighbors=8, leaf_size=30, metric='minkowski', p=2, n_jobs=25,algorithm='auto')
    knn_word.fit(x_train, y_train)

    y_pred_word_knn = knn_word.predict(x_test)

    measure_accuracy(name='kNN Classifier', y_test=y_test, y_pred=y_pred_word_knn, n=n, type=type,list=dataframe_lists_for_scores)

    """Decision Tree Classification"""

    from sklearn.tree import DecisionTreeClassifier
    dt_word = DecisionTreeClassifier(criterion='entropy', random_state=54020, splitter='random')
    dt_word.fit(x_train, y_train)
    y_pred_word_dt = dt_word.predict(x_test)

    measure_accuracy(name='Decision Tree Classifier', y_test=y_test, y_pred=y_pred_word_dt, n=n, type=type,list=dataframe_lists_for_scores)

    """Random Forest Classification"""
    from sklearn.ensemble import RandomForestClassifier
    rf_word = RandomForestClassifier(n_estimators=400, criterion='gini', min_samples_split=30, random_state=54020, warm_start= True,
                                     n_jobs=25)

    rf_word.fit(x_train, y_train)
    y_pred_word_rf = rf_word.predict(x_test)
   # y_pred_word_rf = rf_word.predict(test_dataset['Mesaj:'])

    measure_accuracy(name='Random Forest Classifer', y_test=y_test, y_pred=y_pred_word_rf, n=n, type=type,list=dataframe_lists_for_scores)

    """XGBoost"""

    from xgboost import XGBClassifier
    xgboostclassifier = XGBClassifier(subsample=0.4, reg_lambda=1.5, reg_alpha=0, n_estimators=100, min_child_weight=1, max_depth=7, learning_rate=0.05, gamma=0.3, colsample_bytree=0.6)
    xgboostclassifier.fit(x_train,y_train)
    y_pred_xgboost = xgboostclassifier.predict(x_test)

    ##For 4 n gram word
    #xgboostclassifier = XGBClassifier(subsample=0.4, reg_lambda=1.5, reg_alpha=0, n_estimators=100, min_child_weight=1, max_depth=7, learning_rate=0.01, gamma=0.3, colsample_bytree=0.6)
    ##

    measure_accuracy(name="XGBoost",y_test=y_test,y_pred=y_pred_xgboost,type=type,n=n,list=dataframe_lists_for_scores)

    """
    GradientBoosting
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,
                                     max_depth=1, random_state=54020,warm_start=True)
    clf.fit(x_train, y_train)
    y_pred_clf = clf.predict(x_test)
    measure_accuracy(name="Gradient Boosting Classifier",y_test=y_test,y_pred=y_pred_clf,type=type,n=n,list=dataframe_lists_for_scores)
    """

    """ SupportVectorMachine """
    from sklearn.svm import SVC
    svm_classifier = SVC(kernel='linear',random_state=54020,decision_function_shape='ovo')
    svm_classifier.fit(x_train,y_train)
    y_pred_svm = svm_classifier.predict(x_test)


   # y_pred_svm = svm_classifier.predict(test_dataset['Mesaj:'])

    measure_accuracy(name='Support Vector Machine Classifier', y_test=y_test,y_pred=y_pred_svm,type=type,n = n,list=dataframe_lists_for_scores)


    """
    v = vectorizer.transform(['isim oy_pred_word_lrlsun hesabıma erisemiyorum']).toarray()
    clf.predict(v)
    """
#############################

def measure_accuracy(name, y_test, y_pred, n, type,list):


    print("Confusion Matrix of {name} based on {type} ({n})ngrams\n".format(name=name, type=type,
                                                                                                  n=n))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    print(cm)
    print('\n')
    print("Accuracy Score of {name} based on {type} ({n})ngrams\n".format(name=name, type=type,
                                                                                                n=n))
    accuracyScore = accuracy_score(y_test,y_pred)
    print(accuracyScore)
    print('\n')


    """
    print("Writing accuracy score to a csv file.")
    pd.DataFrame(accuracy_score, columns=['scores_{name}_{type}_{n}_gram'.format(name=name, n=n, type=type)]).to_csv(
        './scores/scores_{name}_{type}_{n}_gram.csv'.format(name=name, type=type, n=n))
    """




    from sklearn.metrics import f1_score
    fscore = f1_score(y_test, y_pred, average='weighted')

    print("F1 Score of {name} based on {type} ({n})ngrams\n".format(name=name, type=type,
                                                                          n=n))
    print(fscore)
    print('\n')


    #fscore is numnpy float64
    writeF1Scores(score=fscore,n = n, type = type, df_list=list,name=name)

    print("#################################\n")

    print('\n')

    writePredictionsToCSV(y_pred = y_pred,name=name,typ=type,n=n)




    #plot.roc(y_test,fscore)

def get_word_cloud(terms):
    text = terms.index
    text = ' '.join(list(text))
    wordcloud = WordCloud(max_font_size = 40,
                          background_color= 'white',
                          stopwords=turkceVeIngilizceStopWords,
                          colormap="Blues"
                          ).generate(text)
    plt.figure(figsize=(25,25))
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis("off")
    plt.show()

def writePredictionsToCSV(y_pred,name,typ,n):
    print("Writing predictions to csv files")
    pd.DataFrame(y_pred,columns = ['y_pred_{name}_{type}_{n}_gram'.format(name = name,n = n,type = typ)]).to_csv('./predictions/{name}_{type}_{n}_gram.csv'.format(name = name, type = typ, n= n))


def writeF1Scores(name,n,type,df_list,score):
    scores = []
    scores.append(score)
    df_list.append(getF1ScoresDataFrame(name = name,scores=scores))
    col = 0
    fileWriter = pd.ExcelWriter('./scores/ML/scores_ML_{type}_{n}_gram.xlsx'.format(n= n, type = type))
    for dataframe in df_list:
        dataframe.to_excel(fileWriter,sheet_name='Validation',startrow=0,startcol=col)
        col = col + len(dataframe.index) + 1
    fileWriter.save()


def getF1ScoresDataFrame(scores,name):
    scores_df = pd.DataFrame(scores,columns = ['F1 Score {name}'.format(name = name)]).astype('float64')
    #print(scores_df)
    return scores_df




