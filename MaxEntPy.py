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
    
def EstimateDisp(Y, n=1, mode='single', selected=[]): # оценка дисперсии, SD**2, из экспериментальных данных
    if len(Y)<=1: return 0
    r=list(range(len(Y)))
    dd=0.
    match mode:
        case 'single':
            return sum(np.square(Y[r[n:]+r[:n]]-Y))/(2*(len(Y)))
        case 'selected':
            for i in selected:
                dd+=sum(np.square(Y[r[i:]+r[:i]]-Y))/(2*(len(Y)))
            return dd/len(selected)       
        case 'full':          
            for i in r[1:]: dd+=sum(np.square(Y[r[i:]+r[:i]]-Y))/(2*(len(Y)))
            return dd/(len(r)-1)


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

        self.Vals={'Chi2':0.,'Scilling':0.,'Total':0.}
        self.Grads=self.Vals.copy()
        self.Hesses=self.Vals.copy()

        self.listVals=self.Vals.copy()
        self.listGrads=self.Vals.copy()
        for name in self.listVals.keys():
            self.listVals[name]=[]
            self.listGrads[name]=[]        
        self.iteration=0
        self.Status='Start'
        self.iH=np.arange(len(self.u))

#        for Name in self.Vals.keys():
#            self.Vals[Name]=GetValue(Name,self.X, self.Y, self.T, self.p, self.tData.a, self.tData.kernel)
#            self.Grads[Name]=GetGradient(Name,self.X, self.Y, self.T, self.p, self.tData.a, self.tData.kernel)
#            self.Hesses[Name]=GetHessian(Name,self.X, self.Y, self.T, self.p, self.tData.a, self.tData.kernel)#
#
#        if lagrange==0:
#            self.lagr=2*self.Vals['Scilling']/(self.Vals['Chi2']/self.disp+self.Vals['Scilling'])
#        else: self.lagr=lagrange
        
#        self.Grad=np.zeros(len(self.u))
#        self.Grad[:(len(self.Grad)-1)]=-self.Grads['Scilling']+self.lagr*self.Grads['Chi2']/self.disp
#        self.Grad[len(self.Grad)-1]=self.Vals['Chi2']/self.disp-1

        self.Hessian=np.zeros(shape=(len(self.u),len(self.u)))
#        self.Hessian[:(len(self.u)-1),:(len(self.u)-1)]=self.lagr*self.Hesses['Chi2']/self.disp
#        for j in range(len(self.u)-1):
#            self.Hessian[j,j]-=self.Hesses['Scilling'][j]
#            self.Hessian[j,len(self.u)-1]=self.Grads['Chi2'][j]/self.disp
#            self.Hessian[len(self.u)-1,j]=self.Hessian[j,len(self.u)-1]

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
    
    def GetVals(self,p,l, theory=[]):  # значение фуекционалов при заданных p, a, lagrange
        vals=self.Vals.copy()
        if len(theory)==0:
            T=np.transpose(self.C) @ p
        else:
            T=theory
        vals['Chi2']=np.sum(np.square(self.Y-T))/(len(self.Y)*self.disp)
        vals['Scilling']=np.sum(p-p*np.log(p))
        vals['Total']=-vals['Scilling']+l*vals['Chi2']
        return vals

    def GetGrads(self,p,lamda, theory=[]):
        grads=self.Grads.copy()
        if len(theory)==0:
            T=np.transpose(self.C) @ p
        else:
            T=theory
        grads['Chi2']=-2/(len(self.X)*self.disp)*(self.B-self.A @ p)
        grads['Scilling']=-np.log(p)
        grads['Total']=np.zeros(len(self.u))        
        grads['Total'][:len(p)]=-grads['Scilling']+lamda*grads['Chi2']
        grads['Total'][len(p)]=self.Vals['Chi2']-1               
        return grads

    def GetHessian(self,p,lamda, mode ='reset'):     
        if mode=='reset':
            self.Hessian=np.zeros(shape=(len(self.u),len(self.u)))

        self.Hessian[:len(p),:len(p)]=2*lamda/(len(self.X)*self.disp)*self.A
        self.Hesses['Scilling']=-np.array(1/p)
        iDiag=np.diag_indices_from(self.A)
        self.Hessian[iDiag]-=self.Hesses['Scilling']
        #2*lamda/(len(self.X)*self.disp)*self.A[iDiag]-self.Hesses['Scilling']

        self.Hessian[len(p),:len(p)]=self.Grads['Chi2']
        for j in range(len(self.p)):
#            self.Hessian[j,j]-=self.Hesses['Scilling'][j]
            self.Hessian[j,len(self.p)]=self.Hessian[len(self.p),j]
 
        
    def Reset(self, eData,tData,lagrange):
        self.eData=eData
        self.tData=tData
        self.u=np.zeros(len(tData.p)+1)
        self.du=self.u.copy()
        self.p=tData.p
        self.lagr=lagrange
        self.iW=np.arange(len(self.u))      # индексы в рабочем диапазоне
        self.iA=np.arange(len(self.p))      # индексы в  диапазоне A
        
        self.iH=self.iW.copy()
        self.iteration=0
        self.Status='Main'
        
        for name in self.listVals.keys():
            self.listVals[name].clear()
            self.listGrads[name].clear()
            
        self.C=np.zeros(shape=(len(self.p),len(self.eData.X)))  # матрица функций
        self.A=np.zeros(shape=(len(self.p),len(self.p)))     # характериситческая матрица
        for j in range(len(self.p)):
            self.C[j]=np.array(self.tData.kernel(self.eData.X,self.tData.a[j]))
            if sum(self.eData.R)>0:
                self.C[j]=np.convolve(self.C[j],self.eData.R)[:len(self.X)]/sum(self.eData.R)            
        self.A=self.C @ np.transpose(self.C)
        self.B=self.C @ self.eData.Y
        self.T=np.transpose(self.C) @ self.p

        self.Res=self.Y-self.T
        self.disp=EstimateDisp(self.Res,mode='selected', selected=[1,2,3])
#        self.disp=EstimateDisp(self.Res)
        
        self.Vals=self.GetVals(self.p,self.lagr,self.T)
        self.Grads=self.GetGrads(self.p,self.lagr,self.T)
        
        
        self.GetHessian(self.p,self.lagr)

        
    def Update(self,tData,lagrange):
        self.tData=tData
        self.p=tData.p
        self.lagr=lagrange
        self.iteration+=1
        
        self.T=np.transpose(self.C) @ self.p
        self.Res=self.Y-self.T
#        if self.iteration==5: self.disp=min(EstimateDisp(self.Res,mode='full'),self.disp)
    
        
        self.Vals=self.GetVals(self.p,self.lagr,self.T)
        self.Grads=self.GetGrads(self.p,self.lagr,self.T)

            
        lstGrads=self.Grads.copy()
        lstGrads['Chi2']=self.Grads['Chi2'] @ self.dp
        lstGrads['Scilling']=self.Grads['Scilling'] @ self.dp
        lstGrads['Total']=self.Grads['Total'] @ self.du
        
        for name in self.listVals.keys():
            self.listVals[name].append(self.Vals[name])
            self.listGrads[name].append(lstGrads[name])

#        if (self.Status=='Main') and (self.iteration>3):
#            if (self.listVals['Chi2'][self.iteration-2]-self.listVals['Chi2'][self.iteration-1])<0.1:
#                self.Status='Tune'
            
        self.GetHessian(self.p,self.lagr,'')          
            
    def StepOver(self):

        iZ=np.where(self.p<max(self.p)/10e7)[0] #  индексы где р очень мало, для приближенного решения и избегания сингулярности
        iW=np.delete(self.iH, iZ)               #  другая часть индексов, для точного решения
        self.iW=iW
        iZdiag=np.diag_indices_from(self.Hessian)

        iT=np.delete(self.iA,iZ)
        
        match self.Status:
#            case 'Start':
            case 'Main':
                self.du[iW]=lg.solve(self.Hessian[iW,:][:,iW],-self.Grads['Total'][iW],assume_a='sym')
                self.du[iZ]=-self.Grads['Total'][iZ]/self.Hessian[iZdiag][iZ]
                    
            case 'Tune':
                self.du[iT]=lg.solve(self.Hessian[iT,:][:,iT],-self.Grads['Total'][iT],assume_a='sym')                             
                self.du[iZ]=-self.Grads['Total'][iZ]/self.Hessian[iZdiag][iZ]
                self.dlagr=0.

        
#        self.du=lg.solve(self.Hessian,-self.Grads['Total'],assume_a='sym')

        mp=min(self.dp/self.p)
        if mp<-1:self.dp/=-mp

    def ShowChange(self,dp):
        self.T=GetTheory(self.eData,tData(self.p+dp,self.tData.a,ExpDecay))
        res=self.Y-self.T
        sd=np.std(res)
        fig, (ax1,ax2,ax3)=plt.subplots(3,1)
        ax1.plot(self.X,self.Y, label='Experiment')
        ax1.plot(self.X,self.T, label='Theory')
        ax1.legend()
        ax2.plot(self.tData.a, self.Add_dp(self.p,self.dp),'.', label='Distribution')
        ax2.set_xscale('log')
        ax2.legend()
        ax3.plot(self.X,res, label=f'Residuals ({round(sd,4)})')
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
                    
    def ShowAnalisys(self,n,lamda):
        chi2=np.zeros(n)
        sci=chi2.copy()
        total=chi2.copy()
        for i in range(n):
            pp=self.Add_dp(self.p,self.dp*i/10)
            #ll=self.lagr+self.dlagr*i/10
            vals=self.GetVals(pp,lamda)
            #t=GetTheory(self.eData,tData(pp,self.tData.a,ExpDecay))
            chi2[i]=vals['Chi2'] #GetValue('Chi2',X,Y,t,pp,self.tData.a,ExpDecay)/self.disp
            sci[i]=vals['Scilling'] #GetValue('Scilling',X,Y,t,pp,self.tData.a,ExpDecay)
            total[i]=vals['Total'] #-sci[i]+ll*(chi2[i]-1)
        plt.plot(chi2, label='Chi2')
        plt.plot(sci, label='Scilling')
        plt.plot(total, label='Total')
        plt.legend()
        plt.show()
        
    def MyGolden(self,name='Total'):
#        dT=GetTheory(self.eData,tData(self.dp,self.tData.a,self.tData.kernel))
        grat=1-(np.sqrt(5.)-1)/2
        r=[0.,grat,1-grat,1.]
#        lamda=self.lagr
        def Func(k):
            vals=self.GetVals(self.p+k*self.dp,self.lagr)
            grads=self.GetGrads(self.p+k*self.dp,self.lagr)
            lamda=sum(grads['Scilling'])/sum(grads['Chi2'])
            return -vals['Scilling']+lamda*vals['Chi2']
        f1=self.GetVals(self.Add_dp(self.p,r[1]*self.dp),self.lagr+self.dlagr)[name]
        f2=self.GetVals(self.Add_dp(self.p,r[2]*self.dp),self.lagr+self.dlagr)[name]
#        f1=Func(r[1])
#        f2=Func(r[2])
        while abs(r[3]-r[0])>0.05:
            if f1<f2:
                r[3]=r[2]
                r[2]=r[1]
                f2=f1
                r[1]=r[0]+grat*(r[3]-r[0])
                f1=self.GetVals(self.Add_dp(self.p,r[1]*self.dp),self.lagr+self.dlagr)[name]
                #f1=Func(r[1])
            else:
                r[0]=r[1]
                r[1]=r[2]
                f1=f2
                r[2]=r[3]-(r[3]-r[0])*grat
                f2 = self.GetVals(self.Add_dp(self.p,r[2]*self.dp),self.lagr+self.dlagr)[name]
                #f2=Func(r[2])
        return (r[0]+r[3])/2, (f1+f2)/2
    def StepTune(self):
        h=np.zeros(shape=(2,2))
        v=np.zeros(2)
        h[0,0]=self.dp @ self.Hessian[:len(self.p),:len(self.p)] @ self.dp 
        h[1,0]=self.Grads['Chi2'] @ self.dp
        h[0,1]=h[1,0]
        v[0]= self.Grads['Total'][:len(self.p)] @ self.dp
        v[1]= self.Vals['Chi2']-1
        return tuple(lg.solve(h,-v,assume_a='sym'))
    def TuningDisp():
        
        
    def Add_dp(self,p,dp):
        pp=p+dp
        for j in range(len(p)):
            if pp[j] <= 0: pp[j]=p[j]/10
        return pp
    def ShowVals(self):
        fig, axs = plt.subplots(2,1, figsize=(5,5), layout='constrained')
        for name in self.listVals.keys():
            axs[0].plot(self.listVals[name], label=name)
            axs[1].plot(self.listGrads[name], label=name)
        axs[0].set_title('Values')
        axs[0].legend()
        axs[1].set_title('Gradients')
        axs[1].legend()
        plt.show()
        
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
