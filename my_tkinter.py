import tkinter as tk
from tkinter import *
from tkinter import ttk
import KineticsLib as Kin
import matplotlib.pyplot as plt
import numpy as np
import MaxEntPy as MEM
#from statsmodels.tsa.stattools import acf

# === Обработка нажатий кнопок ======

def ClickedReset():
    for Name in list(memData.keys()): memData[Name]=memVar[Name].get()
    for Name in list(memData.keys()): memVar[Name].set(memData[Name])

    global myMEM
    global setData
    global expData
    global iteration
    
    p,a = MEM.InitDistribution(memData['Start'],memData['End'],memData['Point number'], prms_type='log')
    setData.p=p
    setData.a=a
    setData.p=np.ones(len(p))*sum(expData.Y)/np.sum(myMEM.C)


    
    myMEM.Reset(expData,setData,memData['lagrange'])
    iteration=0
    Status['Analisys']=True
    print(myMEM.Vals)
    PlotData()
    
def quitclicked():
    plt.close()
    window.destroy()
    
def SimulateClicked():
    ApplyVarToData()
    ApplyDataToVar()
#    global Kinetics

    expData.X=np.arange(0,lstData['Total time'], lstData['Total time']/lstData['Point number'])
    simData.a=[lstData['Decay time']]
    simData.p=[1.]
    expData.R=MEM.GetResponse(expData.X,lstData['Total time']/lstData['Point number'],lstData['Pulse width'])
    expData.Y=MEM.GetTheory(expData,simData)
    expData.Y+=np.random.normal(0, lstData['Noise'], len(expData.X))                     

    Status['Simulated']=True
    Status['Analisys']=False

    PlotData()

def ClickedShowVals():
    myMEM.ShowVals()

        
def PlotData():

#    if fig is None: InitGraphSimulation()
    for Name in list(axd.keys()):
        axd[Name].clear()
        axd[Name].set_title(Name)
        axd[Name].title.set_size(10)
        match Name:
            case 'Kinetics':
                axd[Name].plot(expData.X,expData.Y,label='Kinetics')
                axd[Name].plot(expData.X[:len(expData.R)],expData.R,label='Impulse')
                if Status['Analisys']:
                    axd[Name].plot(myMEM.X,myMEM.T,label='Theory')
                axd[Name].legend()
            case 'Residuals':
                if Status['Analisys']:
                    axd[Name].plot(expData.X,expData.Y-myMEM.T, label='Residuals')
 #               else:
 #                   axd[Name].plot(expData.X,Kinetics.SD, label='Residuals')
#                axd[Name].legend()
#            case 'Autocorrelation':
#                axd[Name].plot(expData.X,Kin.AutoCorrelation(Kinetics.SD), label='Autocorrelation')
#                axd[Name].legend()
#            case 'Histogram':
#                axd[Name].hist(Kinetics.SD, bins=10)
            case 'Distribution':
                if Status['Analisys']:                
                    axd['Distribution'].plot(myMEM.tData.a,myMEM.p, linestyle='none')
                    axd['Distribution'].semilogx(myMEM.tData.a, myMEM.p)


    plt.show()


def ClickedStepOver():

    global myMEM
    global iteration
    
    myMEM.StepOver()
    k,f = myMEM.MyGolden(name='Chi2')
    k2,l2=myMEM.StepTune()

    print(myMEM.iteration,': ', end='')
    for name in myMEM.Vals.keys():
        print(name,':',round(myMEM.Vals[name],4), end=' ')
    print(f'  lagr: {round(myMEM.lagr,4)}, dlagr: {round(myMEM.dlagr,4) }, Golden: {round(k,4)}')

    myMEM.p=myMEM.p+k*myMEM.dp
#    vals=myMEM.GetVals(myMEM.p,myMEM.lagr)
#    if vals['Total']>f: myMEM.lagr+=myMEM.dlagr   

    #myMEM.lagr=sum(myMEM.Grads['Scilling'])/sum(myMEM.Grads['Chi2'])   #*(1+2**(-iteration))
#    if myMEM.Vals['Chi2']>1.1:
    #myMEM.lagr+=k*myMEM.dlagr

    myMEM.lagr+=myMEM.dlagr
    setData.p=myMEM.p
    myMEM.Update(setData,myMEM.lagr)


    PlotData()

    
def ApplyDataToVar():
    for Name in list(lstData.keys()): lstVar[Name].set(lstData[Name])
def ApplyVarToData():
    for Name in list(lstData.keys()): lstData[Name]=lstVar[Name].get()

# ============== основное тело программы ============
window = Tk()
window.title("Fluorescence decay analisys")
window.geometry('500x300')
window.resizable(False,False)

notebook=ttk.Notebook(window)
notebook.pack(expand=True, fill=BOTH)

Pages={'Simulation': ttk.Frame(notebook),
       'MaxEnt Analisys': ttk.Frame(notebook)}
for Name in Pages.keys():
    Pages[Name].pack(fill=BOTH, expand=True)
    notebook.add(Pages[Name], text=Name)
    for c in range(3): Pages[Name].columnconfigure(index=c, weight=1)
    for r in range(7): Pages[Name].rowconfigure(index=r,weight=1)


# =========== страница симуляции кинетик =========================
lstData={'Total time':100.0, 'Point number':200, 'Decay time':10.0,
         'Pulse width':2.,'Noise':0.04}
lstVar={'Total time':DoubleVar(), 'Point number':IntVar(), 'Decay time':DoubleVar(),
         'Pulse width':DoubleVar(),'Noise':DoubleVar()}

rw=1
for Name in list(lstData.keys()):
    lbl=Label(master=Pages['Simulation'], text=Name,font=("Arial",10))
    lbl.grid(column=0, row=rw)
    lstVar[Name].set(lstData[Name])    
    entry=Entry(master=Pages['Simulation'], textvariable=lstVar[Name], width=10)
    entry.grid(column=1, row=rw)
    rw+=1

btnSimulate=ttk.Button(Pages['Simulation'],text="Simulate", command=SimulateClicked)
btnSimulate.grid(column=0, row=6)

btnQ=ttk.Button(Pages['Simulation'],text="Quit",command=quitclicked)
btnQ.grid(column=1, row=6)

#================ Страница Макс этропии анализа ===================
memData={'Start':0.5,'End':100,'Point number':100,'lagrange':1.}
memVar={'Start':DoubleVar(),'End':DoubleVar(),'Point number':IntVar(),'lagrange':DoubleVar()}
rw=1
for Name in list(memData.keys()):
    lbl_mem=Label(master=Pages['MaxEnt Analisys'], text=Name,font=("Arial",10))
    lbl_mem.grid(column=0, row=rw)
    memVar[Name].set(memData[Name])    
    entry_mem=Entry(master=Pages['MaxEnt Analisys'], textvariable=memVar[Name], width=10)
    entry_mem.grid(column=1, row=rw)
    rw+=1

btnInit=ttk.Button(Pages['MaxEnt Analisys'],text='Reset', command=ClickedReset)
btnInit.grid(column=0,row=6)

btnStepOver=ttk.Button(Pages['MaxEnt Analisys'],text='Stepover', command=ClickedStepOver)
btnStepOver.grid(column=1,row=6)

btnShowVals=ttk.Button(Pages['MaxEnt Analisys'],text='Show vals/grads', command=ClickedShowVals)
btnShowVals.grid(column=2,row=6)


# ====================== описание графиков ======================

fig, axd = plt.subplot_mosaic([['Kinetics','Kinetics'],
                              ['Residuals','Residuals'],
#                              ['Autocorrelation','Histogram'],
                              ['Distribution','Distribution']],
                             figsize=(6,6),
                             height_ratios=[2,1,2],
#                             width_ratios=[2,1],
                              layout='constrained',
                            # tight_layout=True,
                            label='Fluorescece kinetics')


#fig, axd = plt.subplot_mosaic([['Kinetics','Kinetics'],
#                              ['Residuals','Residuals'],
#                              ['Autocorrelation','Histogram'],
#                              ['Distribution','Distribution']],
#                             figsize=(6,6),
#                             height_ratios=[2,1,1,2],
#                             width_ratios=[2,1],
#                              layout='constrained',
                            # tight_layout=True,
#                            label='Fluorescece kinetics')

Status={'Simulated':False,'Analisys':False}

# == вводим пустые переменные
iteration=0
expData=MEM.eData()  # экспериментаьны данные
simData=MEM.tData(kernel=MEM.ExpDecay) #  данны едля симуляции кинетики
setData=MEM.tData(kernel=MEM.ExpDecay) # ===данные настроек для MEM
myMEM=MEM.MaxEntropy(expData,simData,0) 

window.mainloop()

