import numpy as np
import abc
import matplotlib.pyplot as plt
import scipy.linalg as lalg

# описание функций
def Kernel(x,a):
    return np.exp(-x/a)
def Shennon(p):
    if p==0: a=0.
    else: a=p*np.log(abs(p))
    return a
    
def EstimateSD(Y): # оценка дисперсии из экспериментальных данных
    SD2=0.
    for i in range(1,len(Y)):
        SD2+=np.square(Y[i-1]-Y[i])
    return np.sqrt(SD2)

def InitDistribution(a0,a1,n): 
    a=np.geomspace(a0,a1,n)
    p=np.ones(n)/n
    return p, a
def CalculateTheory(X,Distr,Alpha):
    T=np.zeros(len(X))
    for i in range(len(X)):
        for j in range(len(Distr)):
            T[i]+=Distr[j]*Kernel(X[i],Alpha[j])
    return T

def AutoCorrelation(X):
    N=len(X)
    mu=np.mean(X)
    sd2=sum((X-mu)**2)
    C=np.zeros(len(X))
    for k in range(N):
        for i in range(N):
            C[k]+=(X[i]-mu)*(X[(i+k)%N]-mu)/sd2
    return C    

#  описание классов данных и функционалов
class TKinetics:  
    def __init__(self,PointNumber,TotalTime):
        self.PointNumber=PointNumber
        self.TotalTime=TotalTime
        self.TimeStep=TotalTime/PointNumber
        self.X = np.arange(0,self.TotalTime,self.TotalTime/self.PointNumber)
    def InitY(self,tau,p):
        self.Y = np.array(Kernel(self.X,tau))
    def InitR(self,Rwidth):
        self.R=np.array(1-(self.X[:int(Rwidth/self.TimeStep+1)]-Rwidth/2)**2/(Rwidth/2)**2)
    def Convolve(self,RR):
        __ss=sum(RR)
        for i in range(self.PointNumber,0,-1):
            for k in range(min(i,len(RR))):
                if k==0: self.Y[i-1]=self.Y[i-1]*RR[0]/__ss
                else:    self.Y[i-1]+=self.Y[i-1-k]*RR[k]/__ss
    def InitNoise(self, SD):
        self.SD=np.random.normal(0, SD, self.PointNumber)

class FBasic(abc.ABC):
    def __init__(self,p,a):
        self.p=p
        self.a=a
        self.Val=0.
        self.Grad=np.zeros(shape=(len(self.p)))
        self.Hess=np.zeros(shape=(len(self.p),len(self.p)))
    @abc.abstractmethod
    def Value(self): pass
    @abc.abstractmethod
    def Gradient(self): pass
    @abc.abstractmethod
    def Hessian(self): pass
    @abc.abstractmethod
    def GradJ(self,j): pass
    @abc.abstractmethod
    def HessJK(self,j,k):pass
    
    def Recalculate(self):
        self.Value()
        self.Gradient()
        self.Hessian()
        
class Scilling(FBasic):
    def __init__(self,p,a):
        super().__init__(p,a)
    def Value(self):
        for pp in list(self.p):
            self.Val+=pp-Shennon(pp)
        return self.Val
    def Gradient(self):
        self.Grad=-np.log(abs(self.p))
        return self.Grad
    def Hessian(self):
        self.Hess=np.diag(-1/self.p)
        return self.Hess
    def GradJ(self,j): return -np.log(abs(self.p[j]))
    def HessJK(self,j,k):
            if j==k: return -1/self.p[j]
            else: return 0.
        
#    def Recalulate(self):
#        self.Value()
#        self.Gradient()
#        self.Hessian()
    
class Chi2(FBasic):
    def __init__(self,p,a,X,Y,R):
        super().__init__(p,a)
        self.X=X
        self.Y=Y
        self.R=R
        self.T=np.zeros(shape=(len(self.X)))
        self.Res=np.zeros(shape=(len(self.X)))
        self.StD2=EstimateSD(self.Y)**2
    def Theory(self):
        for i in range(len(self.X)):
            for j in range(len(self.p)):
                self.T[i]+=self.p[j]*Kernel(self.X[i],self.a[j])
        if len(self.R)>0:
            self.T=np.convolve(self.T,self.R)
            self.T=self.T[:len(self.X)]
        return self.T
    def Residuals(self):
        self.Res = self.Y - self.T
        return self.Res
    def Value(self):
        self.Val=np.sum(np.square(self.Y-self.T))/(len(self.Y)*self.StD2)
        return self.Val
    def Gradient(self):
        for j in range(len(self.p)):
            for i in range(len(self.Res)):
                self.Grad[j]+=-2/(len(self.Res)*self.StD2)*self.Res[i]*Kernel(self.X[i],self.a[j])
        return self.Grad
    def Hessian(self):
        for j1 in range(len(self.p)):
            for j2 in range(j1,len(self.p)):
                for i in range(len(self.X)):
                    self.Hess[j1,j2]+=2/(len(self.X)*self.StD2)*Kernel(self.X[i],self.a[j1])*Kernel(self.X[i],self.a[j2])
                    self.Hess[j2,j1]=self.Hess[j1,j2]
        return self.Hess
    def GradJ(self,j):
        G=0.
        for i in range(len(self.Y)):
            G+=-2/(len(self.Y)*self.StD2)*(self.Y[i]-self.T[i])*Kernel(self.X[i],self.a[j])
        return G        
    def HessJK(self,j,k):
        H=0.
        for i in range(len(self.X)):
            H+=2/(len(self.X)*self.StD2)*Kernel(self.X[i],self.a[j])*Kernel(self.X[i],self.a[k])
        return H
    def Recalculate(self):
        self.Theory()
        self.Residuals()
        super().Recalculate()

class TotalFunctional(FBasic):
    def __init__(self,u,prm,X,Y,R):
        super().__init__(u,prm)
        self.vChi2=Chi2(u[:len(u)-1],prm,X,Y,R)
        self.vSci=Scilling(u[:len(u)-1],prm)
        self.lagr=u[len(u)-1]
        self.dp=np.zeros(len(u))
        
    def Value(self):
        self.Val=-self.vSci.Val+self.lagr*(self.vChi2.Val-1)
        return self.Val
    def Gradient(self):
        for j in range(len(self.p)): self.Grad[j]=self.GradJ(j)
#        self.Grad[:len(self.vChi2.Grad)]=-self.vSci.Grad+self.lagr*self.vChi2.Grad
#        self.Grad[len(self.vChi2.Grad)]=self.vChi2.Val-1
        return self.Grad
    def Hessian(self):
#        self.Hess=np.block([[-self.vSci.Hess+self.lagr*self.vChi2.Hess,self.vChi2.Grad.reshape(len(self.vChi2.Grad),1)],
#                            [self.vChi2.Grad,np.zeros(1)]])
        for j in range(len(self.p)):
            for k in range(j,len(self.p)):
                self.Hess[j,k]=self.HessJK(j,k)
                self.Hess[k,j]=self.Hess[j,k]
        return self.Hess
    def GradJ(self,j):
            if j< len(self.vSci.p): return -self.vSci.GradJ(j)+self.lagr*self.vChi2.GradJ(j)
            else: return self.vChi2.Val-1
    def HessJK(self,j,k):
            if (j<len(self.vSci.p))and (k<len(self.vSci.p)): return -self.vSci.HessJK(j,k)+self.lagr*self.vChi2.HessJK(j,k)
            if j==len(self.vSci.p) and (k<len(self.vSci.p)): return self.GradJ(k)
            if k==len(self.vSci.p) and (j<len(self.vSci.p)): return self.GradJ(j)
            if k==len(self.vSci.p) and (k==len(self.vSci.p)): return 0.

    def Recalculate(self):
        self.vChi2.Recalculate()
        self.vSci.Recalculate()
        super().Recalculate()
    def StepOver(self):
        self.dp=lalg.solve(self.Hess,-self.Grad,assume_a='sym')
        self.p+=self.dp
        self.vChi2.p=self.p[:len(self.p)-1]
        self.vSci.p=self.p[:len(self.p)-1]
        self.Recalculate()
  
    
# отладка функций        
def PlotData():
    fig, (ax1,ax2,ax3)=plt.subplots(3,1)
    ax1.plot(Kinetics.X,Kinetics.Y,Kinetics.X,FF.vChi2.T, label='Kinetics')
    ax1.legend()
    ax2.plot(Alpha,Distr,'.', label='Distribution')
    ax2.set_xscale('log')
    ax2.legend()
    ax3.plot(Kinetics.X,FF.vChi2.Res, label='Residuals')
    ax3.legend()
    plt.show()

def StepOver():
    global dp
    dp=lalg.solve(FF.Hess,-FF.Grad,assume_a='sym')
    global GlobalP
    GlobalP+=dp
    FF.vChi2.T=CalculateTheory(Kinetics.X,Distr,Alpha)
    
    print(f'Chi2={FF.vChi2.Value()}')

Kinetics=TKinetics(200,20)
Kinetics.InitY(5,1)
Kinetics.InitR(2)
Kinetics.Convolve(Kinetics.R)
Kinetics.InitNoise(0.02)
Kinetics.Y+=Kinetics.SD

n_distr=50
GlobalP=np.zeros(n_distr+1)
GlobalP[:n_distr], Alpha = InitDistribution(Kinetics.TimeStep,Kinetics.TotalTime, n_distr)

Distr=GlobalP[:n_distr]
GlobalP[n_distr]=1
Lagrange=GlobalP[n_distr]

Distr/=sum(Kinetics.Y)/(Distr @ Alpha) # нормировка на интеграл

FF=TotalFunctional(GlobalP,Alpha,Kinetics.X,Kinetics.Y,Kinetics.R)
FF.Recalculate()

PlotData()
