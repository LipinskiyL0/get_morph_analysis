# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 23:13:09 2019
Реализация генетического алгоритма
"""
import numpy as np       
    
def FitFunct(ind):
    return np.sum(ind)

class GAClassMy:
    def __init__(self):#конструктор класса
        self.pop=np.array([])#основная популяция
        self.fitness=np.array([]) #
        self.ofspring=np.array([])#популция потомков
        #настройки
        self.n_bits=0#длина бинарной строки
        self.p_mut=0.#вероятность мутации
        self.n_tur=0 #размер турнирной группы
        self.n_ind=0 #количество индивидов
        self.FitFunct=0
#==============================================================================        
    def inicialization(self, FitFunct, n_bits=100, p_mut=0.0051, n_tur=5, n_ind=30):
        #инициализация ГА
        if n_bits<1:
            return False
        if (p_mut<0)|(p_mut>1):
            return False
        if (n_tur<=0)|(n_tur>=n_ind):
            return False
        if (n_ind<=0):
            return False
        
        self.n_bits=n_bits 
        self.p_mut=p_mut
        self.n_tur=n_tur
        self.n_ind=n_ind
        self.FitFunct=FitFunct
        #генерируем популяцию
        self.pop= np.random.randint(0, 2, (n_ind, n_bits))
        #выделяем память под потомков
        self.ofspring=self.pop.copy()
        #создаем массив под родительские пары
        self.ind_per=np.zeros([self.n_ind, 2])
        self.ind_per=np.array(self.ind_per, dtype=int)
        return True
#==============================================================================        
    def opt(self, n_iter=10):
        #основной ход эволюции
        if self.pop.shape[0]==0:
            return False
        if self.n_bits<1:
            return False
        if (self.p_mut<0)|(self.p_mut>1):
            return False
        if (self.n_tur<=0)|(self.n_tur>=self.n_ind):
            return False
        if (self.n_ind<=0):
            return False
        if n_iter<=0:
            return False
        
        
        #рассчитываем пригодность
        self.fitness_pop=np.array(list(map(self.FitFunct, self.pop)))
        print("generation 0, min={0}, mean={1}, max={2}".format( np.min(self.fitness_pop),
              np.mean(self.fitness_pop), np.max(self.fitness_pop)))
        
        
        for i in range(1,n_iter):
            if self.selectionTour()==False:
                return False
            if self.crosTwoPoint()==False:
                return False
            if self.mutation() ==False:
                return False
            
            self.fitness_of=np.array(list(map(self.FitFunct, self.ofspring)))
            
            if np.max(self.fitness_of)<np.max(self.fitness_pop):
                #если в следующем поколении максимум меньше, передаем максимум
                #из предыдущего поколения, вставляем в случайное место
                index1=np.argmax(self.fitness_pop)
                index2=np.random.randint(0, self.n_ind, 1)
                self.ofspring[index2, :]=self.pop[index1, :].copy()
                self.fitness_of[index2]=self.fitness_pop[index1]
            
            #заменяем популяцию родителей, на популяцию потомков
            #заменяем только тех, кто лучше своих родителей
            self.pop=self.ofspring.copy()
            self.fitness_pop=self.fitness_of.copy()
            print("generation {3}, min={0}, mean={1}, max={2}".format( np.min(self.fitness_pop),
              np.mean(self.fitness_pop), np.max(self.fitness_pop), i))
        index=np.argmax(self.fitness_pop)    
                
        return self.pop[index, :], self.fitness_pop[index]
        
#==============================================================================
    def selectionTour(self):
        #турнирная селекция
        if self.pop.shape[0]==0:
            return False
        if self.n_bits<1:
            return False
        if (self.p_mut<0)|(self.p_mut>1):
            return False
        if (self.n_tur<=0)|(self.n_tur>=self.n_ind):
            return False
        if (self.n_ind<=0):
            return False
        #создаем родительские пары
        for i in range(self.n_ind):
            for j in range(2):
                tour=np.random.randint(0, self.n_ind, self.n_tur)
                ftour=self.fitness_pop[tour]
                index=np.argmax(ftour)
                self.ind_per[i, j]=tour[index]
        return True
#==============================================================================
    def crosTwoPoint(self):
        #двухточечное скрещивание
        if self.pop.shape[0]==0:
            return False
        if self.n_bits<1:
            return False
        if (self.p_mut<0)|(self.p_mut>1):
            return False
        if (self.n_tur<=0)|(self.n_tur>=self.n_ind):
            return False
        if (self.n_ind<=0):
            return False
        
        
        for i in range(self.n_ind):
            x=np.random.randint(0, self.n_bits-1, 2)
            x=x+1
            if x[0]>x[1]:
                x[0], x[1]= x[1], x[0]
            
            if np.random.random()<0.5:
                ind=self.pop[self.ind_per[i, 0], :].copy()
                ind[x[0]:x[1]]=self.pop[self.ind_per[i, 1], x[0]:x[1]]
            else:
                ind=self.pop[self.ind_per[i, 1], :].copy()
                ind[x[0]:x[1]]=self.pop[self.ind_per[i, 0], x[0]:x[1]]
            self.ofspring[i, :]=ind.copy()
            
        return True
#==============================================================================
    def mutation(self):
        #проводим мутацию с заданной вероятностью
        if self.pop.shape[0]==0:
            return False
        if self.n_bits<1:
            return False
        if (self.p_mut<0)|(self.p_mut>1):
            return False
        if (self.n_tur<=0)|(self.n_tur>=self.n_ind):
            return False
        if (self.n_ind<=0):
            return False
        mut=np.random.random([self.n_ind,self.n_bits])
        #находим значения меньше заданной вероятности и 
        #инвертируем соответствующие гены
        mask=mut<self.p_mut
        self.ofspring[mask]=1-self.ofspring[mask]
        return True
        
                 
               
            
    
                
        
        
        
        
        
#Ga=GAClassMy()
#Ga.inicialization(FitFunct)
#Ga.opt(60)        


