# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 00:39:23 2021
Класс многоуровней кластеризации методом k средних.
Если на вход поступает просто выборка, то возвращается метки кластеров как при обычной кластеризации
При этом используются методы fit и predict
При кластеризации текста часто самый большой кластер требуются разбить на подкластеры, а
потом из полученных кластеров снова требуется разбить на подкластеры. При этом 
используется метод fit и transform, т.к. этот метод вызывается конвеером, если 
текущий обработчик не последний в конвеере. Метод transform должен возвращать
один массив данных, по этому очередной уровень меток доклеивается в конец данных
Формат столбцов следующий: t1, t2, ...tn, level_0, level_1... Все что с t - входы, 
все что с level  - метки соответствующих уровней

@author: Leonid
"""

import nltk
from sklearn.cluster import KMeans

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import silhouette_samples, silhouette_score
from DivClass import DivClass as DE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

class KMeansClusters2(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters = 7, max_iter=300,  n_init=10, f_opt=False):
        """
        число k определяет количество кластеров
        модель model является реализацией Kmeans
        max_iter - максимальное количество итераций при поиске центройдов
        f_opt - оптимизация начальных центройдов. Если True то кластеризация с оптимизацией начальных 
                центройдов методом дифференциальной эволюции
         self.opt_model - сюда сохранаяется наилучшая модель найденная в ходе оптимизации центройдов
         self.fit_opt_model - критерий силуэта лучшей модели 
        """
        self.n_clusters = n_clusters
        self.max_iter=max_iter
        self.n_init=n_init
        # self.model=KMeans(n_clusters=n_clusters, max_iter=self.max_iter, n_init=n_init)
        self.f_opt=f_opt
        self.opt_model=False
        self.fit_opt_model=0
        
        
    def get_inputs(self, X):
        #Для многократной кластеризации отделяем во входной выборке входы от меток
        cols=list(X.columns)
        cols_in=[x for x in cols if ('level' in x)==False]
        return cols_in
    
    def get_outputs(self, X):
        #Для многократной кластеризации отделяем во входной выборке входы от меток
        cols=list(X.columns)
        cols_out=[x for x in cols if ('level' in x)==True]
        return cols_out
    
    def get_ind_last_claster(self, X):
        #Если последний столбец является метками, то возвращаем логическую маску
        #строк самого численного кластера по последнему столбцу. Иначе возвращаем
        #маску включающую в себя все строки
        cols=list(X.columns)
        if 'level' in cols[-1]:
            #является метками, значит мы должны отобрать самый многочисленный
            #кластер
            x=X[cols[-1]].value_counts()
            n=list(x.index)
            n_max=n[0]
            ind=X[cols[-1]]==n_max
        else:
            #не является меткой, значит данные вообще не размечены, значит
            #обучаем модель на всех данных
            ind=np.ones(len(X), dtype=bool)
        return ind
    
    def get_centroid_ind(self, df1):
        #функция находит индекс объекта, который ближе всех находится к центру 
        #кластера. Предполагается, что в функцию передается в переменной df1
        #единый кластер
        cols_in=self.get_inputs(df1)
        df2=df1[cols_in]
        centrs=df2.mean(axis=0)
        centrs=pd.DataFrame(centrs)
        centrs=centrs.T
        # dist=euclidean_distances(centrs, df2)
        dist=cosine_distances(centrs, df2)
        # centroid=df1.iloc[np.argmin(dist), :]
        # centroid=pd.DataFrame(centroid)
        # centroid=centroid.T
        return df1.index[np.argmin(dist)]
        
    def get_centroid_all_ind(self, df):
        #функция находит объекты ближайшие к центру кластера по каждому кластеру 
        #и возвращает их индексы
        flag=0
        cols_out=self.get_outputs(df)
        ind_centroid_all=[]
        col0=''
        for col in reversed(cols_out):
            ind=df[col].notnull()
            df1=df[ind]
            xx=list(df1[col].value_counts().index)
            xx.sort()
            
            for x in xx:
                if col0!='':
                    df_mus=df1[df1[col]==x]
                    if sum(df_mus[col0].isnull())==0:
                        #значит более глубокий уровень для этого кластера заполнен
                        #пропускаем
                        continue
                ind_centroid_all.append(self.get_centroid_ind(df1[df1[col]==x]))
            col0=col
                
        return ind_centroid_all
    
    def get_samples_ind(self, df, centroid_ind=None, n=10):
        #функция формирует примеры кластеров по следующему принципу:
        #первым примером берется центройд кластера. Если в переменную
        #centroid_ind - передали индексы центройдов кластеров, то оин и берутся
        #в качестве исходных данных. Иначе вызывается get_centroid_all_ind
        #и формируеются центройды. Сам центройд это первый пример в описании
        #кластера. Затем в описание кластера включаются примеры максимально 
        #удаленные от центройдав количестве n. Где n - параметр. Результат
        #возвращается по форме df
        if centroid_ind==None:
            centroid_ind=self.get_centroid_all_ind(df)
        cols_out=self.get_outputs(df)
        cols_in=self.get_inputs(df)
        X=df[cols_in]
        
        rez_ind=[]
        for i in centroid_ind:
            #отбираем индексы кластера к которому относится i-й центройд
            flag=0
            for col in cols_out:
                if df.loc[i,col]!=df.loc[i,col]:
                    continue
                
                if flag==0:
                    flag=1
                    ind=df[col]==df.loc[i,col]
                else:
                    ind=ind & (df[col]==df.loc[i,col])
                
            centrs=X.loc[i, :]
            centrs=pd.DataFrame(centrs)
            centrs=centrs.T
            X1=X[ind]
            dist=euclidean_distances(centrs, X1)
            dist=dist[0,:]
            cl_ind=np.argsort(-dist)
            if len(cl_ind>n):
                cl_ind=cl_ind[:n]
            #возвращаем сами примеры
            cl_ind=X1.index[cl_ind]
           
            #возвращаем их индексы
            rez_ind+=[i]
            rez_ind+=list(cl_ind)
        rez_ind=np.array(rez_ind)
        return rez_ind
                
       
    def fit(self, X, labels = None):
        # X1=self.get_inputs(X)
        # self.model=self.model.fit(X1)
        #1. Получаем название входных столбцов
        cols_in=self.get_inputs(X)
        #2. получаем индекс тех строк, по которым учим модель
        ind=self.get_ind_last_claster(X)
        
            
        self.model=KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, n_init=self.n_init)
        #3. Обучаем модель 
        if self.f_opt==False:
            self.model=self.model.fit(X.loc[ind, cols_in])
        else:
            #вызываем дифференциальную эволюцию. 
            self.X=X.loc[ind, cols_in].copy()
            de=DE()
            Min=X.loc[ind, cols_in].min(axis=0)
            Max=X.loc[ind, cols_in].max(axis=0)
            Min=list(Min)*self.n_clusters
            Max=list(Max)*self.n_clusters
            de.inicialization(FitFunct=self.er_centroid, Min=Min, Max=Max, n_ind=100)  
            de.opt(n_iter=5)     
            self.model=self.opt_model
            
                
        
        return self
    
    def predict(self, X):
        X1=self.transform(X)
        return X1
        
    
    def transform(self, X1):
        X=X1.copy()
        #1. проверяем является ли последний столбец метками
        cols=list(X.columns)
        
        #2. получаем индекс тех строк, по которым учим модель
        ind=self.get_ind_last_claster(X)
        
        #3. Вычисляем метки нового кластера
        cols_in=self.get_inputs(X)
        labels=self.model.predict(X.loc[ind, cols_in])
        
        #4. определяем максимальный уровень кластеризации 
        n=len(cols)-len(cols_in)
        X.loc[ind, 'level_'+str(n)]=labels
        
        return X
    
    def score_all(self, X, y=None):
        X=self.predict( X)
        cols_in=self.get_inputs(X)
        cols_out=self.get_outputs(X)
        
        silhouette_avg=[]
        for col in cols_out:
            ind=X[col].notnull()
            df_rez=X.loc[ind, cols_in]
            cluster_labels = X.loc[ind, col]
            silhouette_avg.append(silhouette_score(df_rez, cluster_labels))
        silhouette_avg=np.array(silhouette_avg)   
        self.silhouette_avg=silhouette_avg
        # print(self.n_clusters, self.silhouette_avg)
        return silhouette_avg
    def score(self, X, y=None):
        return self.score_all(X).mean()
    
    def er_centroid(self, centrs):
        #запускаем k-means из centrs как из начальных центройдов
        #и вычисляем успешность кластеризации
        #запоминаем лучший кластеризатор
        
       
        centrs1=centrs.reshape([self.n_clusters, self.X.shape[1]])
        model=KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter,init=centrs1, n_init=1 )
        model=model.fit(self.X)
        cluster_labels=model.predict(self.X)
        er=silhouette_score(self.X, cluster_labels)
        
        if self.opt_model==False:
            self.opt_model=model
            self.fit_opt_model=er
        elif self.fit_opt_model<er:
            self.opt_model=model
            self.fit_opt_model=er
        
        
        return er,

if __name__ == '__main__':
    X1=np.random.randn(100, 1)
    X2=np.random.randn(100, 1)
    
    Y1=np.random.randn(100, 1)+10
    Y2=np.random.randn(100, 1)+10
    X=np.concatenate([X1, X2], axis=1)    
    Y=np.concatenate([Y1, Y2], axis=1)
    
    X=np.concatenate([X, Y], axis=0)
    X=pd.DataFrame(X, columns=['t1', 't2'])
    model=KMeansClusters2(n_clusters=2,max_iter=1, f_opt=False)
    model=model.fit(X)
    X1=model.transform(X)
    ind_centroid=model.get_centroid_all_ind(X1)
    centroid=X1.loc[ind_centroid, :]
    sample=model.get_samples_ind(X1, n=10)
    sample=X1.loc[sample, :]
    plt.close('all')
    plt.figure()
    plt.plot(X1.loc[X1['level_0']==0, 't1'], X1.loc[X1['level_0']==0, 't2'], 'or')
    plt.plot(X1.loc[X1['level_0']==1, 't1'], X1.loc[X1['level_0']==1, 't2'], 'ob')
   
    plt.plot(sample.loc[X1['level_0']==0, 't1'], sample.loc[X1['level_0']==0, 't2'], 'ok')
    plt.plot(sample.loc[X1['level_0']==1, 't1'], sample.loc[X1['level_0']==1, 't2'], 'og')
    # plt.plot(centroid['t1'], centroid['t2'], 'ok')
   
    # model1=KMeansClusters2(n_clusters=2)
    # model1=model1.fit(X1)
    # X2=model1.transform(X1)
    
    # model2=KMeansClusters2(n_clusters=2)
    # model2=model2.fit(X2)
    # X3=model1.predict(X2)
    
    
    
    
    
