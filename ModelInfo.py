#!/usr/bin/env python3.7
#coding:utf-8

import datetime
import pandas as pd

class ModelParameter(object):
    def __init__(self,name,p={}):
        self.name = name
        self.p = p
        
    def __str__(self):
        return "Param %s"%self.name

class SamplerParameters(ModelParameter):
    def __init__(self,draws=200,tune=200,cores=2,chains=2,init="advi+adapt_diag",max_treedepth=12,n_init=1000000):
        super(SamplerParameters,self).__init__("sampler")
        self.tune = tune
        self.draws = draws
        self.cores = cores
        self.chains = chains
        self.init = init
        self.n_init = n_init
        self.max_treedepth = max_treedepth
        
    def __str__(self):
        return "%s  %s:%d %d:%d %d:%d-%d"%(self.name,self.init,self.n_init,self.cores,self.chains,self.tune,self.draws,self.max_treedepth)

class Priors(ModelParameter):
    def __init__(self,name,p):
        super(Priors,self).__init__(name,p)
        
    def __getitem__(self,k):
        return self.p.get(k,(1,False,))+(self.name+"_"+k,)

class ModelParameters(object):
    """ Assumptions:
        - all arrays with time-step-date refer to the same t_0 time
    """
    def __init__(self,startdate=None,length=0,params=[]):
        self.startdate = startdate
        self.length = length
        self.weekoffset = 0
        self.startingweek = 0
                
        if startdate != None:
            self.weekoffset = self.startdate.weekday()
            self.startingweek = self.startdate.isocalendar()[1]
        
        self.p = {}
        self.ParseParams(params)

    def __getitem__(self,i):
        return self.p.get(i,None)
        
    def keys(self):
        return self.p.keys()
        
    def items(self):
        return self.p.items()
            
    def ModelTimeSeries(self):
        return pd.date_range(self.startdate, periods=self.length, freq='D')
    
    def ParseParams(self,params):
         for p in params:
             v = self.p.get(p.name,None)
             if v == None:
                 self.p[p.name] = p
    
    def __str__(self):
        s = "Model starting at %s %s week = %d running %d days, offset=%d\n"%(self.startdate,["mon","tue","wed","thu","fri","sat","sun"][self.weekoffset],self.startingweek,self.length,self.weekoffset)
        for k,v in self.p.items():
            s += "%s\n"%v
        return s

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

def main():
    p = []
    p.append( SamplerParameters(draws=400) )
    p.append( Priors("R_eff",{"initial":(3,True),"inital_s":(.3,False)}) )
        
    mp = ModelParameters(datetime.date(2020,2,12),120,p)
    print(mp)
    
    print(mp["R_eff"]["initial"])
    
if __name__=="__main__":
    main()
