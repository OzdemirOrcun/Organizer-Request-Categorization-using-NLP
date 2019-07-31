#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import re
import warnings
import nltk
nltk.download('words')
from nltk.corpus import words
from tqdm import tqdm
from sklearn.utils import shuffle

from string import punctuation
import string

warnings.simplefilter("ignore", UserWarning)


def letter_lower(val):
    return val.apply(lambda x: x.lower())

def removeCSSFromData(train_dataset):


    css_lines = 'Üyelik Bilgileri   body  {background:#f0f0f0font:11px Verdana} a {color:#000} a:hover {color: #8e199d}      Takip No: '
    bodyData = train_dataset['Body']
    css_free_body = bodyData.replace(css_lines, '', regex=True, inplace=True)
    pd.concat([train_dataset, css_free_body], axis=1)
    train_dataset.rename(index=str, columns={"Body": "CSS_Free_Body"}, inplace=True)

    train_dataset = train_dataset.iloc[:, 1:]

    dependent_column = train_dataset['AltTanim']
    lowered_dependent_column = letter_lower(dependent_column)
    train_dataset = pd.concat([train_dataset, lowered_dependent_column], axis=1)
    cleansed_train_dataset = train_dataset.iloc[:, [0, 2, 3,  5, 6,7]]

    train_dataset = cleansed_train_dataset
    train_dataset.head(10)

    css_free_body = train_dataset['CSS_Free_Body']
    train_dataset['Mesaj:'] = css_free_body
    train_dataset = train_dataset[train_dataset['Mesaj:'].notnull()]
    train_dataset['Mesaj:'] = train_dataset['Mesaj:'].astype(str)

    return train_dataset

def encode(training_dataset):
    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    y = training_dataset['AltTanim']
    y = labelEncoder.fit_transform(y)
    training_dataset['subID'] = y



def preprocess(train_dataset):

    stinky_words = ["chrome", "huawei", "safari", "mozilla", "firefox", "hello", "html", "mrb", "slm", "respect",
                    "integrity", "job", "opera", "ios", "iphone"]
    low_tfidf_words = ["mailadres", "mail", "iste", "ıd", "ol", "merhaba", "body", "ad", "adres"]

    from nltk.corpus import words
    l = words.words()
    EnglishDict = set()

    for item in l:
        EnglishDict.add(item)
    for stinky_item in stinky_words:
        EnglishDict.add(stinky_item)
    for low_tfidf_words_item in low_tfidf_words:
        EnglishDict.add(low_tfidf_words_item)


    for ind in tqdm(train_dataset.index[::]):

        body = train_dataset['Mesaj:'][ind].translate(ceviri_tablosu).lower()


        body = body.strip()
        tokens = body.split(' ')
        filtered_tokens = [token for token in tokens if token not in EnglishDict]
        single_doc = ' '.join(filtered_tokens)
        single_doc.join(filtered_tokens)
        train_dataset['Mesaj:'][ind] = single_doc



        if body != 'nan':
            a = re.sub(r"{.*}", ' ', train_dataset['Mesaj:'][ind] )
            b = re.sub(r"(&nbsp)|(&quot)", " ", a)
            c = re.sub(r"&#\d*", " ", b)
            d = re.sub(r"(&lt)|(&gt)|(&le)|(&ge)", " ", c)
            e = re.sub(r"[\w.-]+@[\w.-]+.", " mailAddresi ", d)
            f = re.sub(r"http\S+", " ", e)
            g = re.sub(r"www\S+", " ", f)
            h = re.sub("[.,‘’“”()<|;>+:&*?·•!-/-_\[\]]", " ", g)
            i = re.sub("\d", " ", h)
            k = re.sub("\s+", " ", i)
            l = re.sub(r"([a-z])\1+", r"\1", k)
            m = re.sub(r"v\S+", " ", l)
            n = re.sub(r" "," ",m)


           # print(train_dataset['Mesaj:'][ind])

#            ogelerineAyrilmisKelimeListesi = ml.ZemberekTool(zemberek_api).metinde_gecen_kokleri_bul(n)
#            train_dataset['Mesaj:'][ind] = ogelerineAyrilmisKelimeListesi
            train_dataset['Mesaj:'][ind] = n

        print(train_dataset['Mesaj:'][ind])

    for ind in tqdm(train_dataset.index[::]):
        try:
            body = train_dataset['Mesaj:'][ind]
            if body != 'nan':
                i = re.search(r'üyelik bilgileri(.*?)mesaj', body).group(1)
                j = re.sub(i, ' ', body)
                k = re.sub("\s+", " ", j)
                train_dataset['Mesaj:'][ind] = k
            print(train_dataset['Mesaj:'][ind])
        except:
            train_dataset['Mesaj:'][ind] = body

    ####train datasetin geneli yerine ['Mesaj:] yazdım.
    """
    train_dataset = train_dataset.applymap(lambda x: x.encode('unicode_escape').
                                          decode('utf-8') if isinstance(x, str) else x)
    """
    return train_dataset
    #train_dataset.to_excel("{name}_dataset_modified.xlsx".format(name = train_dataset))

def applyDigerToKonuDisindakiler(line):
    return "diğer"

def getNPercentofDf(n,df):
    x = round((n * len(df)) / 100)
    shuffle(df)
    df_modified = df.iloc[:x:]
    return df_modified

df_ic = pd.read_excel('TOP Konular mail içerikleri_train.xlsx')
df_dis = pd.read_excel('other/TOP Konuları dışındaki mail içerikleri_train.xlsx')

"""
from zemberek_python import main_libs as ml
zemberek_api = ml.zemberek_api(libjvmpath="/usr/lib/jvm/jdk-11/jre/lib/amd64/server/libjvm.so",
                            zemberekJarpath="./zemberek-tum-2.0.jar").zemberek()
"""

with open('turkceveingilizceStopWordsandMore.txt', 'r') as f:
    turkceVeIngilizceStopWordsandMore = f.readlines()



#df_ic = shuffle(df_ic)

df_ic = removeCSSFromData(train_dataset=df_ic)


kaynak = "ŞÇÖĞÜİI"
hedef = "şçöğüiı"

ceviri_tablosu = str.maketrans(kaynak,hedef)

df_dis = shuffle(df_dis)

df_dis['AltTanim'] = df_dis['AltTanim'].apply(applyDigerToKonuDisindakiler)

df_dis.to_excel('other/TOP dışı düzenlenmiş.xlsx')

df_dis = removeCSSFromData(df_dis)

df_modified_dis = getNPercentofDf(n = 30,df = df_dis)

df_modified_dis.to_excel('other/TOP dışı düzenlenmiş kısmi.xlsx')

df_ic = preprocess(df_ic)
df_modified_dis = preprocess(df_modified_dis)

preprocessed_df_tum = pd.concat([df_ic,df_modified_dis])

preprocessed_df_tum = shuffle(preprocessed_df_tum)


preprocessed_df_tum.to_excel("tum_dataset_modified_for_dictionary.xlsx",encoding="utf-8-sig")


encoded_dataset = pd.read_excel('tum_dataset_modified_for_dictionary.xlsx')
encoded_dataset = encoded_dataset.iloc[:, 1:]
encode(encoded_dataset)
encoded_dataset.to_excel('tum_dataset_modified_enumareted_for_dictionary.xlsx',encoding="utf8-sig")
