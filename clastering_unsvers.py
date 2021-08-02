# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 13:37:29 2021

В данном моделуе реализован функционал кластеризации ответов на открытые вопросы
Общая схема:
    -подключение к таблице quiz.poll_result_analysis
    -извлечение всех ответов на вопрос по id вопроса. (параметр question_id)
    -кластеризуем полученные ответы
    -результат записываем в таблицу ???

@author: Leonid
"""
from sklearn.pipeline import Pipeline
from my_clasterizator2 import KMeansClusters2
from my_vectorizer import vectorizer

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
 
from sklearn.model_selection import GridSearchCV
from my_split import my_cv_spliter
from sklearn.metrics import silhouette_samples, silhouette_score

from morph_anlysis_me import morph_anlysis_me
import pickle


class clastering_unsvers:
    def __init__(self):
        
        self.flag_load=False
        self.df=[]
        return
    def get_columns(self):
        #формируем полный перечень столбцов
        names=[(4, '' ), (5,'' )]
        for i in range(27):
            names.append((8, i+1))
            
        for i in range(27):
            names.append((10, i+1))
            names.append((11, i+1))
        
        for i in range(27):
            names.append((13, i+1))
            names.append((14, i+1))
        names.append((16, '')) 
        names.append((17, '')) 
        return names
    
    def get_columns_posetiv(self):
        #формируем список столбцов с позетивными высказываниями
        names=[ (5,'' )]
        for i in range(27):
            names.append((11, i+1))
        
        for i in range(27):
            names.append((14, i+1))
        
        names.append((17, '')) 
        return names
    def get_columns_negative(self):
        #формируем список столбцов с негативными высказываниями
        names=[(4, '' )]
        
        for i in range(27):
            names.append((8, i+1))
           
        for i in range(27):
            names.append((10, i+1))
           
        
        for i in range(27):
            names.append((13, i+1))
            
        names.append((16, '')) 
        
        return names
    def load_data(self):
        #загружаем данные.  Необходимо взять текстовые столбцы и развернуть их в 
        #единый список
        with open('Результаты опросов.pkl', 'rb') as f:
            self.df = pickle.load(f)
        names=self.get_columns()
        Result=[]
        for name in names:
            c1=self.df[name]
            
            ind=c1.notnull()
            Result=Result+(list(c1[ind]))
        self.Result=Result
        
       
        
        self.flag_load=True
        return True
    
    def load_data_pos_neg(self):
        #загружаем данные.  Необходимо взять текстовые столбцы и развернуть их в 
        #единый список. Отдельно связываются ответы позетивные и негативные
        with open('Результаты опросов.pkl', 'rb') as f:
            self.df = pickle.load(f)
        names_pos=self.get_columns_posetiv()
        flag=0
        for name in names_pos:
            cl=self.df[[('ID',''), ('Name',''),  ('FIO',''), ('Team ID',''),
                        ('Team','') ]+[name]]
            
            cl['columns']=[name]*len(cl)
            cl['qwe']=[name[0]]*len(cl)
            cl['team_qwe']=[name[1]]*len(cl)
            
            columns= list(cl.columns)
            columns=columns[-3:]+columns[:-3]
            
            cl=cl[columns]
            cl['unsvers']=cl[name]
            del cl[name]
            ind=cl['unsvers'].notnull()
            if flag==0:
                Result_pos=cl[ind]
                flag=1
            else:
                Result_pos=pd.concat([Result_pos, cl[ind]], axis=0)
        self.Result_pos=Result_pos
        Result_pos.columns=Result_pos.columns.droplevel(1)
        
        names_neg=self.get_columns_negative()
        flag=0
        for name in names_neg:
            cl=self.df[[('ID',''), ('Name',''),  ('FIO',''), ('Team ID',''),
                        ('Team','') ]+[name]]
            cl['columns']=[name]*len(cl)
            cl['qwe']=[name[0]]*len(cl)
            cl['team_qwe']=[name[1]]*len(cl)
            
            columns= list(cl.columns)
            columns=columns[-3:]+columns[:-3]
            cl=cl[columns]
            cl['unsvers']=cl[name]
            del cl[name]
            ind=cl['unsvers'].notnull()
            if flag==0:
                Result_neg=cl[ind]
                flag=1
            else:
                Result_neg=pd.concat([Result_neg, cl[ind]], axis=0)
        
        
        Result_neg.columns=Result_neg.columns.droplevel(1)
        self.Result_neg=Result_neg
        
        self.flag_load=True
        return True
    
    def load_data_by_columns(self):
        #загружаем данные.  Необходимо взять текстовые столбцы и развернуть их в 
        #единый список
        with open('Результаты опросов.pkl', 'rb') as f:
            self.df = pickle.load(f)
        names=self.get_columns()
        Result_by_columns={}
        for name in names:
            c1=self.df[name]
            
            ind=c1.notnull()
            Result_by_columns[name]=list(c1[ind])
        self.Result_by_columns=Result_by_columns
        self.flag_load=True
        return True
    
    def transform(self, Result):
        #Предполагается, что данные уже затянуты в переменную self.df (self.flag_load==True)
        #там собраны ответы по одному вопросуv
        if self.flag_load==False:
            return False
        try:
            if len(self.df)<10:
                return False
        except:
            return False
        
        
        Result0=Result
        Result=morph_anlysis_me( Result0)
        df_cl=pd.DataFrame({'original':Result0, 'morph':Result})
        pipe = Pipeline([("vect", vectorizer(method='tfidf',  ngram_range=(1,1) )),
                          ("c1", KMeansClusters2(n_clusters=10, f_opt=False, max_iter=300, n_init=100)),
                           
                         ])
        
        param_grid = {'vect__min_df': [0.01,   0.02, 0.03, 0.04             ],
                      'vect__max_df': [ 0.85, 0.9, 0.95, 0.98],
                      # 'vect__method':['tfidf', 'bw '],
                      # 'vect__ngram_range':[(1,1), (1,2)],
                      'c1__n_clusters': [6,7, 8,  10, 15, 20]
                       }
        grid = GridSearchCV(pipe, param_grid, return_train_score=True, cv=my_cv_spliter )
        
        grid=grid.fit(Result)
        print(grid.best_params_)
        
        pipe=grid.best_estimator_
        
        X11=pipe.predict(Result)
        score=pipe.score(Result)
        cols_in=pipe[-1].get_inputs(X11)
        
        key_words=[]
        X12=X11[cols_in].copy()
        for i in range(len(X12)):
            xx=X12.iloc[i, :].sort_values(ascending=False)
            xx=xx.sort_values(ascending=False)
            xx=xx[xx>0]
            xx1=list(xx.index)
            xx2=str(xx1)
            xx2=xx2.replace('[', '')
            xx2=xx2.replace(']', '')
            xx2=xx2.replace("'", '')
            key_words.append(xx2)
        
        
        df_cl['key_words']=key_words
        #прикрепляем метки кластеров
        df_cl['cl']=X11['level_0'].values
        
        #формируем центройды
        X11['centroid']=[0]*len(X11)
        ind_centroid_all=pipe[-1].get_centroid_all_ind(X11)
        X11.loc[ind_centroid_all, 'centroid']=[1]*len(ind_centroid_all)
        df_cl['centroid']=X11['centroid'].values
        del X11['centroid'] #удаляем этот столбец, что бы он не учитывался при расчете силуэта
        
        #расчитываем метрики кластеризации по каждому измерению, по каждому кластеру и общую
        
        df_cl['silhouette'] = silhouette_samples(X11[cols_in],X11['level_0'] )
        print(silhouette_score(X11[cols_in], X11['level_0']))
        
        c=df_cl.groupby(['cl'])['silhouette'].mean()
        c=c.reset_index()
        c.columns=['cl', 'silhouette_cl']
        
        df_cl=pd.merge(df_cl, c, how='left')
        df_cl['silhouette_all']=[df_cl['silhouette'].mean()]*len(df_cl)
        
        
        #рассчитываем объемы кластеров
        df_cl['count']=[1]*len(df_cl)
        c=df_cl.groupby(['cl'])['count'].sum()
        c=c.reset_index()
        c.columns=['cl', 'cl_num']
        df_cl=pd.merge(df_cl, c, how='left')
        print('объемы кластеров: \n',c)
        ind=self.get_samples(df_cl, n_str=20)
        df_cl['samples']=[0]*len(df_cl)
        df_cl.loc[ind, 'samples']=1
        model=pipe
        return df_cl,model,score
    
    def get_samples(self, df_cl,  n_str=20):
        flag=0
        cols_out='cl'
        
        xx=list(df_cl[cols_out].value_counts().index)
        xx.sort()
        flag=0
        for x in xx:
            df2=df_cl[df_cl[cols_out]==x].sample(n=n_str,  replace=True)
            if flag==0:
                df_rez=df2.copy()
                flag=1
            else:
                df_rez=pd.concat([df_rez, df2], axis=0)
        return list(df_rez.index)
    
    
        
    def save_data(self, add=False):
        #Параметр add говорит о том в каком режиме мы сохраняем данные
        #если add=False, то перезаписываем таблицу, иначе дозаписываем
        print('Сохранение данных {0}'.format(str(datetime.now())))
        
        # with open('Описание кластеров.pkl', 'wb') as f:
        #     pickle.dump(self.df_cl, f)
        # with open('модель.pkl', 'wb') as f:
        #     pickle.dump(self.model, f)
    
        
        
        print('Сохранение данных завершено {0}'.format(str(datetime.now())))
        return True
    def scheme_all(self):
        self.load_data()
        df_cl,model,score=self.transform(etl_cl.Result)
        rez={'df_cl':df_cl,'model':model, 'score':score }
        self.rez=rez
        print(score)
        with open('Описание кластеров_все.pkl', 'wb') as f:
            pickle.dump(self, f)
        df_cl.to_excel('Описание кластеров_все.xlsx')
        
    def scheme_pos_neg(self):
        self.load_data_pos_neg()
        df_cl,model,score=self.transform(list(self.Result_pos['unsvers']))
        self.Result_pos=self.Result_pos.reset_index()
        del self.Result_pos['index']
        
        df_cl=pd.concat([self.Result_pos, df_cl], axis=1)
        rez_poz={'df_cl':df_cl,'model':model, 'score':score }
        self.rez_poz=rez_poz
        print(score)
        
        df_cl,model,score=self.transform(self.Result_neg['unsvers'])
        self.Result_neg=self.Result_neg.reset_index()
        del self.Result_neg['index']
        df_cl=pd.concat([self.Result_neg, df_cl], axis=1)
        
        rez_neg={'df_cl':df_cl,'model':model, 'score':score }
        self.rez_neg=rez_neg
        print(score)
        with open('Описание кластеров_позетивные_негативные.pkl', 'wb') as f:
            pickle.dump(self, f)
        rez_poz['df_cl'].to_excel('Описание кластеров_позитивные.xlsx')
        rez_neg['df_cl'].to_excel('Описание кластеров_негативные.xlsx')
        return
        
    def scheme_by_columns(self):
        self.load_data_by_columns()
        names=self.get_columns()
        flag=0
        models={}
        for name in names:
            df_cl,model,score=self.transform(self.Result_by_columns[name])
            if flag==0:
                df_cl_rez=df_cl
                flag=1
            else:
                df_cl_rez=pd.concat([df_cl_rez, df_cl], axis=0) 
            models[name]=model    
            print(score)
        columns= list(df_cl_rez.columns)
        columns=[columns[-1]]+columns[:-1]
        self.df_cl_rez=df_cl_rez[columns].copy()
        self.models=models
        with open('Описание кластеров по колонкам.pkl', 'wb') as f:
            pickle.dump(self, f)
        df_cl_rez.to_excel('Описание кластеров по колонкам.xlsx')
        
if __name__=='__main__':
    etl_cl=clastering_unsvers()
    # etl_cl.scheme_all()
    etl_cl.scheme_pos_neg()
    # etl_cl.scheme_by_columns()
    # etl_cl.load_data()
    # df_cl,model,score=etl_cl.transform(etl_cl.Result_pos)
    # rez_poz={'df_cl':df_cl,'model':model, 'score':score }
    # etl_cl.rez_poz=rez_poz
    # print(score)
    
    # df_cl,model,score=etl_cl.transform(etl_cl.Result_neg)
    # rez_neg={'df_cl':df_cl,'model':model, 'score':score }
    # etl_cl.rez_neg=rez_neg
    # print(score)
    
    # etl_cl.save_data(add=True)
    
    

       
