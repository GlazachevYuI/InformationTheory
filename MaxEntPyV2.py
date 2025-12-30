import numpy as np
import abc
import scipy.linalg as lg
import matplotlib.pyplot as plt

class single_var:
    def __init__(self,coeffs=[], params=[], shennon=False, gradient_type='normal', kernel=lambda x , y: np.float64(x==y)):
        self.__kernel=kernel        
        self.N=len(coeffs)
        self.__v=np.array([coeffs,params])

        self.__index=slice(self.N)
        self.shennon=shennon
        
    @property
    def coeffs(self): return self.__v[0]
    @coeffs.setter
    def coeffs(self,p): self.__v[0]=p
    @property
    def params(self): return self.__v[1]
    @params.setter
    def params(self,p): self.__v[1]=p

    def theory_column(self,xdata,j):
        return np.array(self.__kernel(xdata, self.params[j]))

def mult_v(**sv):
    N=0
    for name in sv.keys():
        sv[name].__index=slice(N,N+sv[name].N)
        N+=sv[name].N
    return sv

class mult_vars: 
    def __init__(self,**uvar):  # uvar: name=single_var
        self.N=0
        self.sv=uvar
        for name in uvar.keys():
            uvar[name].__index=slice(self.N,self.N+uvar[name].N)
            if uvar[name].shennon: self.shennon=uvar[name]
            self.N+=uvar[name].N           
        self.du=np.zeros(self.N+1) # размер параметров + лагранж
        self.lagrange=1 #lagrange

    @property
    def p(self): return self.shennon.coeffs
    @p.setter
    def p(self,pp): self.shennon.coeffs=pp

        
class DataModelling:
    def __init__(self):
        self.__rawX=[]
        self.__rawY=[]
        self.__X=self.__rawX
        self.__Y=self.__rawY
        self.__R=[]
        self.IndexRange=[]
        self.Ranges={'XY':[],'R':[]}

        self.__fY=lambda x: x
        self.__kernel=lambda a,x: np.exp(-a/x)
    
        self.__p=[]
        self.__dp=[]
        self.__lnp=[]
        
        self.__a=[]
        self.__u=[1.0, 1.0]
        self.__du=[0,]
    
        self.__C=0
        self.__A=0
        
        self.functions=0
        self.N=0 # количесво всех коэффициентов без Лагранжа
        
        self.Vals={'Chi2':0.,'Shennon':0.,'Total':0.}
        self.Grads=self.Vals.copy()
        self.Hesses=self.Vals.copy()

        self.Settings={'Gradient':'normal','Chi2':'normal'}
        self.u=[]
    @property
    def rawX(self): return self.__rawX
    @property
    def rawY(self): return self.__rawY

    @property
    def X(self): return self.__X[self.Ranges['XY']]
    @property
    def Y(self): return self.__Y[self.Ranges['XY']]
    @property 
    def R(self): return self.__R[self.Ranges['R']]

    @property
    def p(self):
        if self.Settings['Gradient']=='log':
            return self.__p
        else:
            return self.__u[:-1]
    @p.setter
    def p(self,pp):        
        if self.Settings['Gradient']=='log':
            self.__p=pp
            self.__u[-1]=np.log(pp)
        else:
            self.__u[:-1]=pp
            self.__lnp=np.log(pp)

    @property 
    def dp(self):
        if self.Settings['Gradient']=='log':
            return self.__dp
        else:
            return self.__du[:-1]
    @dp.setter
    def dp(self,dpp):
        if self.Settings['Gradient']=='log':
            self.__dp=dpp
#            self.__du[-1]=dpp
        else:
            self.__du[:-1]=dpp
#            self.__dlnp=dpp
    
    @property 
    def lnp(self):
        if self.Settings['Gradient']=='log':
            return self.__u[:-1]
        else:
            return self.__lnp
    @lnp.setter
    def lnp(self,pp):
        if self.Settings['Gradient']=='log':
            self.__u[:-1]=pp
            self.__p=exp(pp)
        else:
            self.__lnp=pp
            self.__u[:-1]=exp(pp)
    
        
    @property 
    def a(self): return self.__a
    @a.setter
    def a(self,aa): self.__a=aa
    @property 
    def lagrange(self): return self.__u[-1]
    @lagrange.setter
    def lagrange(self,lagr): self.__u[-1]=lagr
                         
    @property
    def u(self): return self.__u
    @u.setter
    def u(self,uu):
        self.__u=uu
        if self.Settings['Gradient']=='log':
            self.__p=np.exp(uu[:-1])
        else:
            self.__lnp=np.log(uu[:-1])
        
    @property
    def du(self): return self.__du
    @du.setter
    def du(self,duu): self.__du=duu

    
    @property
    def kernel(self): return self.__kernel
    @property
    def fY(self): return self.__fY
    
# ввод исходных данных
    def RawData(self, rawX, rawY):
        self.__rawX=np.array(rawX)
        self.__rawY=np.array(rawY)
        self.__X=self.__rawX
        self.__Y=self.__rawY
        self.IndexRange=range(len(self.__X))
        self.Ranges['XY']=range(len(rawX))
# функция преобразования для Y       
    def functionY(self, fY=lambda y: y):
        self.__Y=fY(self.__rawY)
        self.__fY=fY
# аппаратная функция ( ответа) 
    def response(self, xdata=0,response_function=[]):
        if xdata==0:
            self.__R=np.array(response_function)
            self.Ranges['R']=range(len(response_function))
        else:
            self.__R=np.array(response_function(xdata))
            self.Ranges['R']=ranges(len(xdata))
# добавление фунций ядра
    def init_functions(self, **sv)
        self.N=0
        for name in sv.keys():
            sv[name].__index=slice(self.N,self.N+sv[name].N)
            if sv[name].shennon: self.shennon=sv[name]
            self.N+=sv[name].N
        self.functions=sv
    
# начальное значение функции распеределения (образа)
    def initial_image(self, number, step_type='uniform', range=[0,1], gradient_type='normal'):

        self.Settings['Gradient']=gradient_type
        self.u=united_vars(np.ones(number)/number,1.0,gradient_type)
        
#        self.__u=np.ones(number+1) # объединенная переменная P=lagrange(=1 при создании)
#        self.__du=np.zeros(number+1)
#        
#        self.lagrange=1.0
#        self.p=self.p/number
        match step_type:
            case 'uniform': self.a=np.arange(range[0], range[1], (range[1]-range[0])/number)
            case 'log': self.a=np.geomspace(range[0], range[1], number)

        self.__C=np.zeros(shape=(len(self.X),self.N))  # матрица функций C[i,j]=f(xi,pj]
        self.__A=np.zeros(shape=(self.N,self.N)     # характериситческая матрица

    def cacluate_theory(self,functions):
        T=np.zeros(len(self.X))       
        for name in functions.keys():
            T+=self.__CC[:,functions[name].__index] @ functions[name].coeffs
        return self.__fY(T)
    def create_matices(self, norma=1.0):
        jj=0
        for name in functions.key():
            for j in range(functions[name].N):
                self.__C[:,jj]=functions[name].theory_column(self.X-self.X[0],j))
                if sum(self.R)>0:
                    self.__C[:,jj]=np.convolve(self.__C[:,jj],self.R)[:len(self.X)]/sum(self.R)               
                jj+=1

#        for j in range(len(self.p)):
#            self.__C[:,j]=np.array(norma*self.__kernel(self.X-self.X[0],self.a[j]))
#            if sum(self.R)>0:
#                self.__C[:,j]=np.convolve(self.__C[:,j],self.R)[:len(self.X)]/sum(self.R)

        self.__A=np.transpose(self.__C) @ self.__C # характериситческая матрица

        self.T=self.calculate_theory(self.functions)    # теоретическия фунция
        
        self.Res=self.Y-self.T # невязка
        self.disp=EstimateDisp(self.Res, mode='selected', selected=[1,2,3]) # оценка дисперсии сигма-квадрат

        self.Grads['Total']=np.zeros(self.N+1))
        self.Hesses['Total']=np.zeros(shape=(self.N+1,self.N+1))
        
    def calculate_values(self, name='', u=[]):
            if len(u)==len(self.u):
                self.u=u
                self.Res=self.Y-self.calculate_theory(self.functions))
                
            self.Vals['Chi2']=np.sum(np.square(self.Res))/(len(self.X)*self.disp)              
            self.Vals['Shennon']=np.sum(self.shennon.coeffs*(1-np.log(self.shennon.coeffs)))
            self.Vals['Total']=-self.Vals['Shennon']+self.lagrange*(self.Vals['Chi2']-1)
                
    def calculate_gradients(self, name='', u=[]):
            if len(u)==len(self.u):
                self.u=u
                self.Res=self.Y-self.calculate_theory(self.functions))
            match self.Settings['Gradient']:
                case 'log':
                    self.Grads['Chi2']=-2/(len(self.X)*self.disp)*(np.transpose(self.__C) @ self.Res)*self.p
                    self.Grads['Shennon']=-np.log(self.shennon.coeffs)*self.shennnon.coeffs
                case _:
                    self.Grads['Chi2']=-2/(len(self.X)*self.disp)*(np.transpose(self.__C) @ self.Res)
                    self.Grads['Shennon']=-np.log(self.shennon.coeffs)
                    
            self.Grads['Total'][:len(self.p)]=-self.Grads['Shennon']+self.lagrange*self.Grads['Chi2']
            self.Grads['Total'][len(self.p)]=self.Vals['Chi2']-1
    def calculate_hessians(self,name='',u=[]):
            if len(u)==len(self.u):
                self.u=u
                self.Res=self.Y-self.__fY(self.__C @ self.p)
            iDiag=np.diag_indices_from(self.__A)
            
            match self.Settings['Gradient']:
                case 'log':
                    self.Hesses['Chi2']=2/(len(self.X)*self.disp)*self.__A*np.outer(self.p,self.p)
                    self.Hesses['Chi2'][iDiag]+=self.Grads['Chi2']
                    self.Hesses['Shennon']=-self.p*(1+self.lnp)
                case _:
                    self.Hesses['Chi2']=2/(len(self.X)*self.disp)*self.__A
                    self.Hesses['Shennon']=-1/self.p


            self.Hesses['Total'][:len(self.p),:len(self.p)]=self.lagrange*self.Hesses['Chi2']
            self.Hesses['Total'][iDiag]-=self.Hesses['Shennon']
            self.Hesses['Total'][len(self.p),:len(self.p)]=self.Grads['Chi2']
            self.Hesses['Total'][:len(self.p),len(self.p)]=self.Grads['Chi2']

    def update_functionals(self, gradient_type='normal', u=[]):

        if len(u)==len(self.u):  self.u=u
        
        self.T=self.__fY(self.__C @ self.p)    # теоретическия фунция
        self.Res=self.Y-self.T # невязка
#        self.disp=EstimateDisp(self.Res, mode='selected', selected=[1,2,3]) # оценка дисперсии сигма-квадрат

        self.calculate_values()
#        for Name in list(self.Vals.keys()):
#            self.calculate_values(name=Name)
#        for Name in list(self.Vals.keys()):
#           self.calculate_gradients(name=Name)
#        for Name in list(self.Vals.keys()):
#            self.calculate_hessians(name=Name)


#        self.Vals['Chi2']=np.sum(np.square(self.Res))/(len(self.Y)*self.disp)
#        self.Vals['Shennon']=np.sum(self.p-self.p*np.log(self.p))
#        self.Vals['Total']=-self.Vals['Shennon']+self.lagrange*(self.Vals['Chi2']-1)

        self.calculate_gradients()

#        self.Grads['Chi2']=-2/(len(self.X)*self.disp)*(np.transpose(self.__C) @ self.Res)
#        self.Grads['Shennon']=-np.log(self.p)
#        self.Grads['Total']=np.zeros(len(self.u))        
#        self.Grads['Total'][:len(self.p)]=-self.Grads['Shennon']+self.lagrange*self.Grads['Chi2']
#        self.Grads['Total'][len(self.p)]=self.Vals['Chi2']-1

        self.calculate_hessians()
#        self.Hesses['Chi2']=2/(len(self.X)*self.disp)*self.__A
#        self.Hesses['Shennon']=-1/self.p
#        self.Hesses['Total']=np.zeros(shape=(len(self.u),len(self.u)))
#        iDiag=np.diag_indices_from(self.__A)
#        self.Hesses['Total'][:len(self.p),:len(self.p)]=self.lagrange*self.Hesses['Chi2']
#        self.Hesses['Total'][iDiag]-=self.Hesses['Shennon']
#        self.Hesses['Total'][len(self.p),:len(self.p)]=self.Grads['Chi2']
#        self.Hesses['Total'][:len(self.p),len(self.p)]=self.Grads['Chi2']

    def test_derivatives(self, du=0.02):
        self.ValsCenter=self.Vals.copy()
        self.GradsCenter=self.Grads.copy()
        for nm in list(self.GradsCenter.keys()):
            self.GradsCenter[nm]=self.Grads[nm].copy()        
        self.HessesCenter=self.Hesses.copy()
        for nm in list(self.GradsCenter.keys()):
            self.HessesCenter[nm]=self.Hesses[nm].copy()
       
        uu=self.u.copy()
        
        self.update_functionals(u=uu*(1+du/2))
              
        self.ValsRight=self.Vals.copy()
        self.GradsRight=self.Grads.copy()
        for nm in list(self.GradsRight.keys()):
            self.GradsRight[nm]=self.Grads[nm].copy()        

        self.update_functionals(u=uu*(1-du/2))
        
        print('Gradients: численно: аналитически')
        for name in list(self.Vals.keys()):
            print(name,':',self.ValsRight[name]-self.Vals[name],':',self.GradsCenter[name] @ (uu*du)[:len(self.GradsCenter[name])])
        print('Hessians')
        for name in list(self.Vals.keys()):
            match name:
                case 'Shennon':
                    print(name,':',np.sum(np.square(self.GradsRight[name]-self.Grads[name])),':',
                          np.sum(np.square(self.HessesCenter[name] * (uu*du)[:len(self.GradsCenter[name])])))
                case _:
                    print(name,':',np.sum(np.square(self.GradsRight[name]-self.Grads[name])),':',
                          np.sum(np.square(self.HessesCenter[name] @ (uu*du)[:len(self.GradsCenter[name])])))
        
    def StepOver(self):
        if self.Settings['Gradient']=='log':
            self.du=lg.solve(self.Hesses['Total'],-self.Grads['Total'],assume_a='sym')
        else:
            iZ=np.where(self.p<max(self.p)/10e7)[0] #  индексы где р очень мало, для приближенного решения и избегания сингулярности
            iW=np.delete(np.arange(len(self.u)), iZ)               #  другая часть индексов, для точного решения
        
            iZdiag=np.diag_indices_from(self.Hesses['Total'])
            iT=np.delete(np.arange(len(self.p)),iZ)

            self.du[iW]=lg.solve(self.Hesses['Total'][iW,:][:,iW],-self.Grads['Total'][iW],assume_a='sym')
            self.du[iZ]=-self.Grads['Total'][iZ]/self.Hesses['Total'][iZdiag][iZ]

            iCheck=np.where(self.du<-self.u)  # проверка на p+dp <0 после итерации
            if len(self.du[iCheck])>0:
                mp=max(self.u[iCheck]/self.du[iCheck])
                self.du[:-1]*=-0.9*mp
        
#=======Вспомогательные функции=============
def EstimateDisp(Y, n=1, mode='single', selected=[]): # оценка дисперсии, SD**2, из экспериментальных данных
    if len(Y)<=1: return 0
    r=list(range(len(Y)))
    dd=0.
    match mode:
        case 'single':
            return sum(np.square(Y[n:]-Y[:(len(Y)-n)]))/(2*(len(Y)-n))
        case 'selected':
            for i in selected:
                dd+=sum(np.square(Y[i:]-Y[:(len(Y)-i)]))/(2*(len(Y)-i))
            return dd/len(selected)       
        case 'full':          
            for i in r[1:]: dd+=sum(np.square(Y[r[i:]+r[:i]]-Y))/(2*(len(Y)))
            return dd/(len(r)-1)

        




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
    def __init__(self, coeffs=[], params=[], entropy=False, response=[], kernel=1):
        self.p=coeffs
        self.a=params
        self.kernel=kernel
        self.entropy=entropy
        self.response=response
    @property
    def N(self): return min(len(self.p),len(self.a))
# ======= Описание основного класса =============================
class MaxEntropy:
    def __init__(self, eData, *tData, lagrange=1., weight=1., gradient_mode='normal'):

        self.X=eData.X
        self.Y=eData.Y
        self.R=eData.R
        
        self.u=[]   # глобальный вектор коэффициентов
        self.ip=[]  #индексы распределия р
        for tD in tData:
            #=== создаем спсиок индексов коэфициентов распределия (для энтропии)
            if tD.entropy: self.ip+=list(range(len(self.u),len(tD.p)+len(self.u))) 
            self.u=self.u+list(tD.p)
        self.u=np.array(self.u+[lagrange],dtype=float)
        
# === создаем матрицу базовых функий
        self.C=np.zeros(shape=(len(self.u)-1,self.X))
        j0=0
        for tD in tData:
            for j in range(tD.N):
                self.C[j0]=np.array(tD.kernel(self.X,tD.a[j]))
                if len(tD.response)>0:
                    self.C[j0]=np.convolve(self.C[j0],tD.response)[:len(self.X)]
                j0+=1
            
#===== характеристическая матрице и вспомогательный вектор
#        self.A=np.zeros(shape=(len(self.u)-1,len(self.u)-1))     
        self.A=self.C @ np.transpose(self.C)
        self.B=self.C @ self.Y
        
        self.T=np.transpose(self.C) @ self.u[:-1]
        self.Res=self.Y-self.T

#=== описание свойств класса===========
        @property
        def p(self): return self.u[self.ip]
        @p.setter
        def p(self,pp):       
#            jp=np.where(pp<=0) проверка <= ноль
#            self.u[jp]/=self.u[jp]           
            self.u[ip]=pp

        @property
        def lagr(self): return self.u[-1]
        @lagr.setter
        def lagr(self,lagrange): self.u[-1]=lagrange
        @property
        def dp(self): return self.du[self.ip]
        @dp.setter
        def dp(self,dp): self.du[self.ip]=dp
        @property
        def dlagr(self): return self.du[-1]
        @dlagr.setter
        def dlagr(self,dl): self.du[-1]=dl


# =====описание значений, градиентов и гессианов функционалов===========
        self.Vals={'Chi2':0.,'Entropy':0.,'Total':0.}
        self.Grads=self.Vals.copy()
        self.Hesses=self.Vals.copy()

        def GetVals(self, p,lamda, theory=[], disp=0, weight=1., gradient_mode='normal'):
            vals=self.Vals.copy()     
            if len(theory)==0: T=np.transpose(self.C) @ self.u[:-1]
            else: T=theory
            if disp==0: disp=self.disp
            vals['Chi2']=np.sum(np.square(self.Y-T)/weight)/(len(self.Y)*disp)
            vals['Entropy']=np.sum(p-p*np.log(p))
            vals['Total']=-vals['Entropy']+lamda*(vals['Chi2']-1)
            return vals

        def GetGrads(self,p,lamda, theory=[], disp=0, weight=1., gradient_mode='normal'):
            grads=self.Grads.copy()
            if len(theory)==0: T=np.transpose(self.C) @ self.u[:-1]
            else: T=theory
            if disp==0: disp=self.disp
            grads['Chi2']=-2/(len(self.X)*disp)*(self.B-self.A @ self.u[:-1])
            grads['Entropy']=-np.log(p)
            if gradiant_mode=='log':
                grads['Chi2'][ip]*=self.p
                grads['Entropy'][ip]*=self.p

#            grads['Total']=np.zeros(len(self.u))        
            grads['Total']=np.append(-grads['Entropy']+lamda*grads['Chi2'],self.Vals['Chi2']-1)
#            grads['Total'][len(p)]=self.Vals['Chi2']-1               
            return grads
        

