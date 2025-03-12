import numpy as np
import abc
from scipy import optimize
import scipy.linalg as lg
import matplotlib.pyplot as plt



def ExpDecay(x,tau):
    return np.exp(-x/tau)
def Pln(x,n): return x**n
def Cronecker(x,a):
    if x==a: return 1.
    else: return 0.


def GetValue(Name,xdata=[],ydata=[], tdata=[], pdata=[], adata=[], kernel=[]):
    val=0.
    match Name:
        case 'Chi2': val=np.sum(np.square(ydata-tdata))/len(ydata)
        case 'Scilling': val=np.sum(np.array(pdata-pdata*np.log(abs(pdata))))
    return val

def GetGradient(Name,xdata=[],ydata=[], tdata=[], pdata=[], adata=[], kernel=Cronecker):
    Grad=np.zeros(len(pdata))
    match Name:
        case 'Chi2':
            for j in range(len(Grad)):
                for i in range(len(ydata)):
                    Grad[j]+=-2/len(ydata)*(ydata[i]-tdata[i])*kernel(xdata[i],adata[j])
        case 'Scilling':
            Grad=np.array(-np.log(abs(pdata))) 
    return Grad

def GetHessian(Name, xdata=[],ydata=[], tdata=[], pdata=[], adata=[], kernel=Cronecker):
    match Name:
        case 'Chi2':
            Hess=np.zeros(shape=(len(pdata),len(pdata)))
            for j1 in range(len(pdata)):
                for j2 in range(j1,len(pdata)):
                    for i in range(len(xdata)):
                        Hess[j1,j2]+=2/len(xdata)*kernel(xdata[i],adata[j1])*kernel(xdata[i],adata[j2])
                        Hess[j2,j1]=Hess[j1,j2]   
        case 'Scilling': Hess=np.array(-1/pdata)
    return Hess
            
    
def EstimateDisp(Y): # оценка дисперсии, SD**2, из экспериментальных данных
    return sum(np.square(Y[1:]-Y[:len(Y)-1]))/(2*(len(Y)-1))


def InitDistribution(prm_start, prm_end, prm_number, prms_type='uniform'):
    match prms_type:
        case 'uniform': a=np.arange(prm_start, prm_end, (prm_end - prm_start)/prm_number)
        case 'log': a=np.geomspace(prm_start, prm_end, prm_number)
        
    p=np.ones(prm_number)/prm_number
    return p,a
def GetResponse(X, TimeStep, Rwidth, rtype='parabola'):
    return np.array(1-(X[:int(Rwidth/TimeStep+1)]-Rwidth/2)**2/(Rwidth/2)**2)   

# экспериментальные данные
class eData:
    def __init__(self, xdata=[],ydata=[],rdata=[]):
        self.X=np.array(xdata)
        self.Y=np.array(ydata)
        self.R=np.array(rdata)
    @property # число точек
    def M(self): return min(len(self.X),len(self.Y))
#теоретические данные-настройки
class tData:
    def __init__(self, coeffs=[], params=[], kernel=Cronecker):
        self.p=coeffs
        self.a=params
        self.kernel=kernel
    @property
    def N(self): return min(len(self.p),len(self.a))

eData_default=eData()
tData_default=tData()

def GetTheory(eData, tData):
    Y=np.zeros(len(eData.X))
    for j in range(tData.N):
        Y+=np.array(tData.p[j]*tData.kernel(eData.X,tData.a[j]))
    if sum(eData.R)>0:
        Y=np.convolve(Y,eData.R)[:len(eData.X)]/sum(eData.R)
    return Y

class MaxEntropy:
    def __init__(self, eData, tData, lagrange=0.):
        self.eData=eData
        self.tData=tData
        self.u=np.concatenate((np.array(self.tData.p),np.array([lagrange])))
        self.du=np.zeros(len(self.u))
        self.T=GetTheory(self.eData,self.tData)
        self.disp=EstimateDisp(self.eData.Y-self.T)

        self.Vals={'Chi2':0.,'Scilling':0.}
        self.Grads=self.Vals.copy()
        self.Hesses=self.Vals.copy()
        for Name in self.Vals.keys():
            self.Vals[Name]=GetValue(Name,self.X, self.Y, self.T, self.p, self.tData.a, self.tData.kernel)
            self.Grads[Name]=GetGradient(Name,self.X, self.Y, self.T, self.p, self.tData.a, self.tData.kernel)
            self.Hesses[Name]=GetHessian(Name,self.X, self.Y, self.T, self.p, self.tData.a, self.tData.kernel)

        if lagrange==0:
            self.lagr=2*self.Vals['Scilling']/(self.Vals['Chi2']/self.disp+self.Vals['Scilling'])
        else: self.lagr=lagrange
        
        self.Grad=np.zeros(len(self.u))
        self.Grad[:(len(self.Grad)-1)]=-self.Grads['Scilling']+self.lagr*self.Grads['Chi2']/self.disp
        self.Grad[len(self.Grad)-1]=self.Vals['Chi2']/self.disp-1

        self.Hessian=np.zeros(shape=(len(self.u),len(self.u)))
        self.Hessian[:(len(self.u)-1),:(len(self.u)-1)]=self.lagr*self.Hesses['Chi2']/self.disp
        for j in range(len(self.u)-1):
            self.Hessian[j,j]-=self.Hesses['Scilling'][j]
            self.Hessian[j,len(self.u)-1]=self.Grads['Chi2'][j]/self.disp
            self.Hessian[len(self.u)-1,j]=self.Hessian[j,len(self.u)-1]

#        self.fig2, self.axd2 =plt.subplot_mosaic([['Kinetics'],['Residuals'],['Distribution']],
#                                            figsize=(6,6), height_ratios=[2,1,2],
#                                            tight_layout=True, label='MaxEtr analisys')
        
    @property
    def p(self): return self.u[:self.tData.N]
    @p.setter
    def p(self,pp):
        for j in range(len(pp)):
            if pp[j] <= 0:
                self.u[j]/=10
            else:
                self.u[j]=pp[j]
        eData.p=self.p
    @property
    def lagr(self): return self.u[len(self.u)-1]
    @lagr.setter
    def lagr(self,lagrange): self.u[len(self.u)-1]=lagrange
    @property
    def dp(self): return self.du[:self.tData.N]
    @dp.setter
    def dp(self,dp):
        self.du[:self.tData.N]=dp
    @property
    def dlagr(self): return self.du[len(self.du)-1]
    @dlagr.setter
    def dlagr(self,dl): self.du[len(self.du)-1]=dl

    @property
    def X(self): return self.eData.X
    @property
    def Y(self): return self.eData.Y
    
    def GetTotalValue(self, Y=[], T=[], P=[],lagrange=0):
        if len(Y)==0: Y=self.Y
        if len(T)==0: T=self.T
        if len(P)==0: P=self.p
        if lagrange==0: lagrange=self.lagr
        return -GetValue('Scilling', pdata=P)+lagrange*(GetValue('Chi2',ydata=Y, tdata=T, pdata=P)/self.disp-1)
        
    def UpdateAll(self):
        self.T=GetTheory(self.eData,self.tData)
        self.Res=self.eData.Y-self.T
        self.disp=EstimateDisp(self.eData.Y-self.T)
        for Name in self.Vals.keys():
            self.Vals[Name]=GetValue(Name,self.eData.X, self.eData.Y, self.T, self.p, self.tData.a, self.tData.kernel)
            self.Grads[Name]=GetGradient(Name,self.eData.X, self.eData.Y, self.T, self.p, self.tData.a, self.tData.kernel)
        self.Grad[:(len(self.Grad)-1)]=-self.Grads['Scilling']+self.lagr*self.Grads['Chi2']/self.disp
        self.Grad[len(self.Grad)-1]=self.Vals['Chi2']/self.disp-1

    def UpdateChange(self,ddp,ddl):
        self.p+=ddp
        self.lagr+=ddl
        self.T=GetTheory(eData(self.X,self.Y,self.eData.R),tData(self.p,self.tData.a,self.tData.kernel))
        for Name in self.Vals.keys():
            self.Vals[Name]=GetValue(Name,self.X, self.Y, self.T, self.p, self.tData.a, self.tData.kernel)
            self.Grads[Name]=GetGradient(Name,self.X, self.Y, self.T, self.p, self.tData.a, self.tData.kernel)

        self.Grad[:(len(self.Grad)-1)]=-self.Grads['Scilling']+self.lagr*self.Grads['Chi2']/self.disp
        self.Grad[len(self.Grad)-1]=self.Vals['Chi2']/self.disp-1
        
        self.Hesses['Scilling']=GetHessian('Scilling',self.eData.X, self.eData.Y, self.T, self.p, self.tData.a, self.tData.kernel)
        for j in range(len(self.p)):
            self.Hessian[j,j]=-self.Hesses['Scilling'][j]+self.lagr*self.Hesses['Chi2'][j,j]
            self.Hessian[j,len(self.u)-1]=self.Grads['Chi2'][j]/self.disp
            self.Hessian[len(self.u)-1,j]=self.Hessian[j,len(self.u)-1]
             
            
    def StepOver(self):
        self.du=lg.solve(self.Hessian,-self.Grad,assume_a='sym')

    def ShowChange(self,dp):
        self.T=GetTheory(self.eData,tData(self.p+dp,self.tData.a,ExpDecay))
        res=Y-self.T
        sd=np.std(res)
        fig, (ax1,ax2,ax3)=plt.subplots(3,1)
        ax1.plot(X,Y, label='Experiment')
        ax1.plot(X,self.T, label='Theory')
        ax1.legend()
        ax2.plot(self.tData.a, self.p+dp,'.', label='Distribution')
        ax2.set_xscale('log')
        ax2.legend()
        ax3.plot(X,res, label=f'Residuals ({round(sd,4)})')
        ax3.legend()
        plt.show()

    def Show(self):
        for Name in list(self.axd2.keys()):
            self.axd2[Name].clear()
            match Name:
                case 'Kinetics':
                    self.axd2[Name].plot(self.X,self.Y,label='Experiment')
                    self.axd2[Name].plot(self.X,self.T, label='Theory')
                case 'Residuals':
                    self.axd2[Name].plot(self.X,(self.Y-self.T), label='Residuals')
                case 'Distribution':
                    self.axd2[Name].plot(self.tData.a, self.p,'.', label='Distribution')
                    self.axd2[Name].semilogx(self.tData.a, self.p)
            self.axd2[Name].legend()
        plt.show()
                    


#        self.ax1.clear()
#        self.ax2.clear()
#        self.ax3.clear()
        
#        self.ax1.plot(self.X,self.Y,label='Experiment')
#        self.ax1.plot(self.X,self.T, label='Theory')
#        self.ax1.legend()
#        self.ax2.plot(self.tData.a, self.p,'.', label='Distribution')
#        self.ax2.set_xscale('log')
#        self.ax2.legend()
#        self.ax3.plot(self.X,(self.Y-self.T), label='Residuals')
#        self.ax3.legend()
#        plt.show()
        
    def ShowAnalisys(self,n):
        chi2=np.zeros(n)
        sci=chi2.copy()
        total=chi2.copy()
        for i in range(n):
            pp=self.p+self.dp*i/10
            ll=self.lagr+self.dlagr*i/10
            t=GetTheory(self.eData,tData(pp,self.tData.a,ExpDecay))
            chi2[i]=GetValue('Chi2',X,Y,t,pp,self.tData.a,ExpDecay)/self.disp
            sci[i]=GetValue('Scilling',X,Y,t,pp,self.tData.a,ExpDecay)
            total[i]=-sci[i]+ll*(chi2[i]-1)
        plt.plot(chi2, label='Chi2')
        plt.plot(sci, label='Scilling')
        plt.plot(total, label='Total')
        plt.legend()
        plt.show()
        
    def MyGolden(self,lagr):
        dT=GetTheory(self.eData,tData(self.dp,self.tData.a,self.tData.kernel))
        grat=1-(np.sqrt(5.)-1)/2
        r=[0.,grat,1-grat,1.]
        def Func(k): return self.GetTotalValue(self.Y,self.T+k*dT,self.p+k*self.dp,lagr)
        f1=Func(r[1])
        f2=Func(r[2])
        while abs(r[3]-r[0])>0.05:
            if f1<f2:
                r[3]=r[2]
                r[2]=r[1]
                f2=f1
                r[1]=r[0]+grat*(r[3]-r[0])
                f1=Func(r[1])
            else:
                r[0]=r[1]
                r[1]=r[2]
                f1=f2
                r[2]=r[3]-(r[3]-r[0])*grat
                f2=Func(r[2])
        return (r[0]+r[3])/2, min(f1,f2)
    def Add_dp(self,p,dp):
        for j in range(len(p)):
            if dp[j] <= -p[j]: p[j]=p[j]/10
            else: p[j]+=dp[j]
        return p
        
def PlotData(myMEM):
    fig, (ax1,ax2,ax3)=plt.subplots(3,1)
    ax1.plot(meMEM.X,myMEM.Y,label='Experiment')
    ax1.plot(myMEM.X,myMEM.T, label='Theory')
    ax1.legend()
    ax2.plot(myMEM.tData.a, myMEM.p,'.', label='Distribution')
    ax2.set_xscale('log')
    ax2.legend()
    ax3.plot(myMEM.eData.X,(myMEM.eData.Y-myMEM.T), label='Residuals')
    ax3.legend()
    plt.show()
            



        
# проверка



#X=np.arange(0,25,0.25)
#R=[1,3,4,3,1]
#T=GetTheory(eData(X,[],R),tData([1],[5],ExpDecay))

#p,a= InitDistribution(0.25, 15, 40, prms_type='log') 

#Y=T+np.random.normal(0, 0.02, len(T))

# MEM=MaxEntropy(eData(X,Y,R),tData(p,a,ExpDecay),2)
