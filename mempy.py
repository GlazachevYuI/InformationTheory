import numpy as np
import abc
import scipy.linalg as lg
import matplotlib.pyplot as plt

#  описание классов данных и функционалов

def Pln(x,a): return x**a  
def ExpDecay(x,a): return np.exp(-x/a)

def fTheory(xdata=[],response=[],distr=[],params=[], kernel=()):
    Y=np.zeros(len(xdata))
    for j in range(min(len(params),len(distr))):
        Y+=np.array(distr[j]*kernel(xdata,params[j]))
    if sum(response)>0:
        response=np.array(response)/sum(response)
        Y=np.convolve(Y,response)[:len(xdata)]
    return Y

def InitDistribution(prm_start, prm_end, prm_number, distr_type='uniform'):
    match distr_type:
        case 'uniform': a=np.arange(prm_start, prm_end, (prm_end - prm_start)/prm_number)
        case 'log': a=np.geomspace(prm_start, prm_end, prm_number)
    p=np.ones(prm_number)/prm_number
    return p,a

def EstimateSD(Y): # оценка дисперсии из экспериментальных данных
    SD2=0.
    for i in range(1,len(Y)):
        SD2+=np.square(Y[i-1]-Y[i])
    return np.sqrt(SD2)

def Shennon(p):
    if p==0: return 0.
    else: return p*np.log(abs(p))
def Scilling(p):
    if p==0: return 0.
    else: return p-p*np.log(abs(p))
    
def ShowData(X,Y):
    plt.plot(X,Y)
    plt.show()


class f_Basic(abc.ABC):
    def __init__(self, xdata=[],ydata=[],distr=[],params=[]):
        self.X=xdata
        self.Y=ydata
        self.p=distr
        self.a=params
        self.Val=0.
        self.Grad=np.zeros(shape=(len(distr)))
    @abc.abstractmethod
    def Value(self): pass
      
#    @abc.abstractmethod
#    def Hessian(self): pass
    @abc.abstractmethod
    def GradJ(self,j): pass
    @abc.abstractmethod
    def HessJK(self,j,k):pass

    def Gradient(self):
        for j in range(len(self.p)): self.Grad[j]=self.GradJ(j)
        return self.Grad

class f_Chi2_1(f_Basic):
    def __init__(self, xdata=[],ydata=[],response=[], distr=[],params=[], kernel=()):
        super().__init__(xdata,ydata, distr, params)
        self.R=response
        self.SD=EstimateSD(self.Y)
        self.StD2=np.square(self.SD)
        self.kernel=kernel
        self.T=fTheory(self.X, self.R, self.p, self.a, self.kernel)
        self.Val=self.Value(self.T)
    def Value(self,T):
        self.Val=np.sum(np.square(self.Y-T))/(len(self.Y)*self.StD2)-1
        return self.Val
#    def Gradient(self):
#        for j in range(len(self.p)
#        return np.array(self.GradJ(J))
    def GradJ(self,j):
        return -2/(len(self.X)*self.StD2)*sum(np.array((self.Y-self.T)*self.kernel(self.X,self.a[j])))
                                              
    def HessJK(self,j,k):
        return 2/(len(self.X)*self.StD2)*sum(np.array(self.kernel(self.X,self.a[j])*self.kernel(self.X,a[k])))

class f_Entropy(f_Basic):
    def __init__(self, distr=[], params=[]):
        super().__init__(distr=distr, params=params)
        self.p=distr
        self.a=params
    def Value(self):
        self.Val=0.
        for j in range(len(p)):
            self.Val+=Scilling(self.p[j])
        return self.Val
#    def Gradient(self):
#        J=np.arange(len(self.p))
#        return np.array(GradJ(J))
    def GradJ(self,j):
        return -np.log(abs(self.p[j]))
    def HessJK(self,j,k):
        if j==k: return -1/self.p[j]
        else: return 0

class f_Total(f_Basic):
    def __init__(self, xdata=[],ydata=[],response=[], distr=[], params=[], kernel=(), lagrange=1.0):
        super().__init__(xdata,ydata, distr, params)
        self.R=response
        self.kernel=kernel
        
        self.u=np.concatenate((distr,[lagrange]))
        self.p=self.u[:len(distr)]
        self.lagrange=self.u[len(self.u)-1]
       
        self.Grad=np.zeros(shape=(len(self.u)))
        self.Hess=np.zeros(shape=(len(self.u),len(self.u)))
        
        self.Chi2=f_Chi2_1(self.X,self.Y, self.R, self.p, self.a, self.kernel)
        self.Entr=f_Entropy(distr=self.p, params=self.a)
        self.T=self.Chi2.T

        
    def Value(self,T):
        self.Val=-self.Entr.Value()+self.lagrange*self.Chi2.Value(T)        
        return self.Val
    def Gradient():
        J=np.arange(len(u))
        return np.array(GradJ(J))
    def GradJ(self,T,j):
        if j==len(self.u-1): return self.Chi2.Value(T)
        else: return -self.Entr.GradJ(j)+self.lagrange*self.Chi2.GradJ(T,j)
    def HessJK(self,T,j,k):
        return -self.Entr.HessJK(j,k)+self.lagrange*self/Chi2.HessJK(T,j,k)

    def Recalculate(self):
        self.Chi2.Val=self.Chi2.Value(self.T)
        self.Entr.Val=self.Entr.Value()
        self.Val=-self.Entr.Val+self.lagrange*self.Chi2.Val
        
        self.Chi2.Grad=self.Chi2.Gradient()
        self.Entr.Grad=self.Entr.Gradient()
        self.Grad[:len(self.u)-1]=-self.Entr.Grad+self.lagrange*self.Chi2.Grad
        self.Grad[len(self.u)-1]=self.Chi2.Val
        
        for j in range(len(self.p)):
            self.Hess[j,len(self.u)-1]=self.Chi2.Grad[j]
            self.Hess[len(self.u)-1,j]=self.Hess[j,len(self.u)-1]
            for k in range(j, len(self.p)):
                self.Hess[j,k]=-self.Entr.HessJK(j,k)+self.lagrange*self.Chi2.HessJK(j,k)
                self.Hess[k,j]=self.Hess[j,k]
    def StepOver(self):
        self.du=lg.solve(self.Hess,-self.Grad,assume_a='sym')
        self.u+=self.du
        self.T=fTheory(self.X, self.R, self.p, self.a, ExpDecay)
        self.Recalculate()

    
def PlotData():
    fig, (ax1,ax2,ax3)=plt.subplots(3,1)
    ax1.plot(X,Y,X,MaxEnt.T, label='Kinetics')
    ax1.legend()
    ax2.plot(MaxEnt.a, MaxEnt.p,'.', label='Distribution')
    ax2.set_xscale('log')
    ax2.legend()
    ax3.plot(X,(MaxEnt.Y-MaxEnt.T), label='Residuals')
    ax3.legend()
    plt.show()
            



X=np.arange(0,20,0.25)
response=[1,2,3,2,1]
Y=fTheory(X,response,[0.5,0.5],[2,5],ExpDecay)
sd=np.random.normal(0, 0.02, len(X))
Y+=sd

p,a=InitDistribution(0.5,10,20,distr_type='log')

MaxEnt=f_Total(X,Y,response,p,a,ExpDecay,1)
        
    
        






    






            
