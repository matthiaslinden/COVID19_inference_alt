#!/usr/bin/env python3.7
#coding:utf-8

import pandas as pd

class Modelinfo(object):
    """ Assumptions:
        - all arrays with time-step-date refer to the same t_0 time
    """
    def __init__(self,startdate=None,length=0):
        self.startdate = startdate
        self.length = length
        self.weekoffset = 0
        self.startingweek = 0
        
        if startdate != None:
            self.weekoffset = self.startdate.weekday()
            self.startingweek = self.startdate.isocalendar()[1]
            
    def Series(self):
        return pd.date_range(self.startdate, periods=self.length, freq='D')
    
    def __str__(self):
        return "Model starting at %s %s week = %d running %d days, offset=%d"%(self.startdate,["mon","tue","wed","thu","fri","sat","sun"][self.weekoffset],self.startingweek,self.length,self.weekoffset)

class Dataset(object):
    def __init__(self,data,startdate=None):
        self.dataset = data
        self.startdate = startdate

class Datasets(object):
    def __init__(self):
        self.datasets = {}
        
    def ModelRange(self,model_vector,name):
        lower,upper = (0,0,0),(-1,-1,-1)
        if model_vector.ndim == 1:
            return model_vector[lower[0]:upper[0]]
        elif model_vector.ndim == 2:
            return model_vector[lower[0]:upper[0],lower[1]:upper[1]]
        elif model_vector.ndim == 3:
            return model
        else:
            print("Model vector of type %s"%(type(model_vector)))
            return model_vecor
        
    def AddDataSeries(self,name,data,startdate=None):
        self.datasets[name] = Dataset(data,startdate)
