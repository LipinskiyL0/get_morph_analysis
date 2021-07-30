# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:31:07 2020

@author: Леонид
"""
import string
import pymorphy2
import nltk
from nltk.corpus import stopwords
import pandas as pd

def tokenize_me(file_text):
    #firstly let's apply nltk tokenization
    tokens = nltk.word_tokenize(file_text)
 
    #let's delete punctuation symbols
    tokens = [i for i in tokens if ( i not in string.punctuation )]
 
    #deleting stop_words
    # stop_words = stopwords.words('russian')
    # stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
    try:
        df=pd.read_csv('stop_words.csv')
        stop_words=list(df['stop words'])
    except:
        print("Ошибка чтения файла со стоп словами. Перечень стоп слов загружен по умолчанию ")
        stop_words=['что', 'это', 'так', 'вот', 'быть', 'будет', 'было', 'бывает', 'более' 'как',
                    'бы', 'на','от', 'На', 'и', 'у',  'с', 'же',
                    'в', '—', 'к', 'на', 'пока', 'было', 'даст', 'данной',
                    'данным', 'данном', 'данная', 'дает', 'дать', 'для', 'при', 'по',
                    'мой', 'моя','моей', 'они', 'их', 'мне', 'мою', 'своей', 'свою', 'свой','свои','ему',
                    'он','ей', 'него',
                    'которые','каком','какие', 'какой', 'некий', 'некой','кто-то', 'кому',
                    'которых', 'который','я', 'Я','мы', 'меня', 'мне', 'Мне', 
                    'есть', 'чему', 'из', 'за', 'т.д.', 'т. д.', '/', 'или', 'и/или',  
                    'тому', 'иному', 'все', 'всё',  'со', 
                    ]

    tokens = [i for i in tokens if ( i not in stop_words )]
 
    #cleaning words
    tokens = [i.replace("«", "").replace("»", "") for i in tokens]
 
    return tokens