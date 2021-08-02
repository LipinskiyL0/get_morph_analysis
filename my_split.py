# -*- coding: utf-8 -*-
"""
В кластеризации классическая перекрестная проверка не имеет смысла, т.к. 
нам необходим найти эффективное разбиение на кластеры на всей выборке, а не 
на ее части. 
многие кластеризаторы вообще не имеют predict
по этому, что бы для перекрестной проверки использовать GridSearchCV
нужно в нем подменить cv_spliter на тот, который будет каждый раз выдавать 
исходную выборку. Этот клас для этого и написан
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class my_cv_spliter(BaseEstimator):
    def __init__(self, cv = 5):
        
        return
    
    def get_n_splits(self, X=None, y=None, groups=1):
        
        return groups
    
    def split(self, X, y=None, groups=None):
        return [(np.arange(len(self)), None)]

if __name__ == '__main__':
    xx=my_cv_spliter()