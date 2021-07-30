# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:34:18 2021

@author: Leonid
"""
import string
import pymorphy2
import nltk
from nltk.corpus import stopwords
from tokenize_me import tokenize_me


def morph_anlysis_me(text):
        # c=c.iloc[:5000]
    #подготавливаем текст. Производим токенизацию (делим текст на слова)\
    #и лематизацию (переводим слова в нормальную форму)
    Result=[]
    for result in text:
        result=str(result)
        result=tokenize_me(result)
        mus=[]
        morph = pymorphy2.MorphAnalyzer()
        for r in result:
            #применяем несколько раз перевод в нормальную форму, т.к. столкнулся
            #с таким явлением, что ведущии переводит в ведущие, а ведущие переводит
            #в ведущий. Вместо того, что бы сразу перевести ведущии в ведущий. 
            y=morph.parse(r)[0]
            s=y.normal_form
            y=morph.parse(s)[0]
            s=y.normal_form
            y=morph.parse(s)[0]
            s=y.normal_form
                
            mus.append(s)
       
        #в результате токенизации и лематизации получаем список слов. далее
        #переводим слова в стоку
        result=mus
        
        mus=""
        for r in result:
            mus+=r+" "
        result=mus
        Result.append(result)
    return Result