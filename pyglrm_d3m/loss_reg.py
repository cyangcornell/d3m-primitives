import numpy as np
import julia

j = julia.Julia()
#j.eval("logging(DevNull, kind=:warn)")
j.using("LowRankModels")
j.using("DataFrames")

#__all__ = ['j', 'QuadLoss','L1Loss','HuberLoss','ZeroReg','NonNegOneReg','QuadReg','NonNegConstraint']

#Losses
def QuadLoss(scale=1.0, domain=j.RealDomain()):
    if not isinstance(scale, float):
        raise TypeError
    return j.QuadLoss(scale, domain)

def L1Loss(scale=1.0, domain=j.RealDomain()):
    if not isinstance(scale, float):
        raise TypeError
    return j.L1Loss(scale, domain)

def HuberLoss(scale=1.0, domain=j.RealDomain(), crossover=1.0):
    if not isinstance(scale, float) or not isinstance(crossover, float):
        raise TypeError
    return j.HuberLoss(scale, domain, crossover)


#Regularizers
def ZeroReg():
    return j.ZeroReg()

def NonNegOneReg(scale=1):
    return j.NonNegOneReg(scale)

def QuadReg(scale=1):
    return j.QuadReg(scale)

def NonNegConstraint():
    return j.NonNegConstraint()
