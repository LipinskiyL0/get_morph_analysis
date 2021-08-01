# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:38:36 2021

@author: Leonid
"""

import pandas as pd
from morph_anlysis_me import morph_anlysis_me
import pickle 


if __name__ == '__main__':
    df=pd.read_excel('Результаты опросов_final.xlsx', header=[0,1])
    x0=list(df.columns.get_level_values(0))
    x1=list(df.columns.get_level_values(1))
    x11=[]
    for x in x1:
        if 'Unnamed' in str(x):
            x11.append('')
        else:
            x11.append(x)
    columns=pd.MultiIndex.from_arrays([x0, x11])
    columns.names=['вопрос', 'команда']
    df.columns=columns 
    
    with open('Результаты опросов.pkl', 'wb') as f:
        pickle.dump(df, f)
        
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
    for name in names:
        c1=df[name]
        
        ind=c1.notnull()
        Result=list(c1[ind])
        Result=morph_anlysis_me( Result)
        c2=c1.copy()
        c2[ind]=Result
        df[name]=c2
    df.to_excel('Результаты опросов_обработка.xlsx')
    with open('Результаты опросов_обработка.pkl', 'wb') as f:
        pickle.dump(df, f)
    