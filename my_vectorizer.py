# -*- coding: utf-8 -*-
"""
создаем векторизатор. Класс реализующий разные варианты векторизации в обертке библиотеки sklearn
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

import pickle

class vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self,  method='bw', min_df=2, max_df=1.0, ngram_range= (1,1), LDA_n_components=10, LDA_n_lexems=10  ):
        """
        method - способ векторизации текста: bw - мешок слов, tfidf - tfidf
        LDA - латентное распределение Дирихле
        min_df - используется в методе bw, tfidf. Параметр определят больше скольки 
                    раз должно встречаться слово во всех корпусах, что бы 
                    оно было добавлено в словарь
        max_df - аналогично bw_min_df только в другую сторону
        LDA_n_components - количество тем в LDA
        LDA_n_lexems - количество лексем в теме
        self.model - основная модель
        self.model_bw - используется как этап расчета в LDA
        self.ngram_range - параметр ngram_range  в bw и tfidf
        
                    
        
        """
        self.method=method
        self.min_df=min_df
        self.max_df=max_df
        self.ngram_range=ngram_range
        self.LDA_n_components=LDA_n_components
        self.LDA_n_lexems=LDA_n_lexems
        return
    
    def fit(self, X, y = None):
        """
        Изучает правила преобразования на основе входных данных X.
        """
        
        if self.method=='tfidf':
            model = TfidfVectorizer(min_df=self.min_df, max_df=self.max_df, ngram_range=self.ngram_range)
            model = model.fit(X)
            
        elif self.method=='bw':
            model = CountVectorizer(min_df=self.min_df, max_df=self.max_df, ngram_range=self.ngram_range)
            model = model.fit(X)
            
            
        elif self.method=='LDA':
            self.model_bw = CountVectorizer(min_df=self.min_df, max_df=self.max_df)
            bag_of_words = self.model_bw.fit_transform(X)
            
            model = LatentDirichletAllocation(n_components=self.LDA_n_components, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
            model=model.fit(bag_of_words)

            
            
        else:
            print("Ошибочный метод")
            return
        
        self.model=model
        return self


    def transform(self, X):
        """
        Преобразует X в новый набор данных Xprime и возвращает его.
        """
        
        if self.method=='tfidf':
            
            values = self.model.transform(X)
            # Show the Model as a pandas DataFrame
            feature_names = self.model.get_feature_names()
            df_rez=pd.DataFrame(values.toarray(), columns = feature_names)
        elif self.method=='bw':
            
            bag_of_words =self.model.transform(X)
            
            feature_names = self.model.get_feature_names()
            df_rez=pd.DataFrame(bag_of_words.toarray(), columns = feature_names)
            
        elif self.method=='LDA':
            
            bag_of_words =self.model_bw.transform(X)
            feature_names = self.model_bw.get_feature_names()
            df_rez=self.model.transform(bag_of_words)
            df_rez=pd.DataFrame(df_rez, columns=['t'+str(i) for i in range(df_rez.shape[1])] )
            n=self.LDA_n_lexems
            if n>len(feature_names):
                n=len(feature_names)
            topics = dict()
            for idx, topic in enumerate(self.model.components_):
                 features = topic.argsort()[:-(n + 1): -1]
                 tokens = [feature_names[i] for i in features]
                 topics[idx] = tokens
                            
            self.topics=topics
        
        else:
            print("Ошибочная метрика")
            return
        return df_rez
if __name__ == '__main__':
    with open('rez_table_text.pkl', 'rb') as f:
        df  = pickle.load(f)
    name='Q18'
    ind=df[name+'_txt'].notnull()
    r=df[name].loc[ind]
    Result=list(df[name+'_txt'].loc[ind])   
    vect=vectorizer(method='tfidf', min_df=5, max_df=1.0, ngram_range=(1,2))
    vect=vect.fit(Result)
    df_rez=vect.transform(Result)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    