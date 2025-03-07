import tkinter as tk
from tkinter import *
from tkinter import ttk
import KineticsLib as Kin
import matplotlib.pyplot as plt
import numpy as np
import MaxEntPy as MEM
#from statsmodels.tsa.stattools import acf

# === Обработка нажатий кнопок ======
global myMEM
def ClickedReset():
    for Name in list(memData.keys()): memData[Name]=memVar[Name].get()
    for Name in list(memData.keys()): memVar[Name].set(memData[Name])

    p,a = MEM.InitDistribution(memData['Start'],memData['End'],memData['Point number'], prms_type='log')
    
    myMEM=MEM.MaxEntropy(MEM.eData(Kinetics.X,Kinetics.Y,Kinetics.R),MEM.tData(p,a,MEM.ExpDecay),memData['lagrange'])
    Status['Analisys']=True
    PlotData()
    
def quitclicked():
    plt.close()
    window.destroy()
    
def SimulateClicked():
    ApplyVarToData()
    ApplyDataToVar()
    global Kinetics
    Kinetics=Kin.TKinetics(lstData['Point number'],lstData['Total time'])
    Kinetics.InitY(lstData['Decay time'],1)
    Kinetics.InitR(lstData['Pulse width'])
    Kinetics.Convolve(Kinetics.R)
    Kinetics.InitNoise(lstData['Noise'])
    Kinetics.Y*=1+Kinetics.SD
    Status['Simulated']=True
    Status['Analisys']=False
    PlotData()
        
def PlotData():

#    if fig is None: InitGraphSimulation()
    for Name in list(axd.keys()):
        axd[Name].clear()
        axd[Name].set_title(Name)
        axd[Name].title.set_size(10)
        match Name:
            case 'Kinetics':
                axd[Name].plot(Kinetics.X,Kinetics.Y,label='Kinetics')
                axd[Name].plot(Kinetics.X[:len(Kinetics.R)],Kinetics.R,label='Impulse')
                if Status['Analisys']:
                    axd[Name].plot(myMEM.X,myMEM.T,label='Theory')
                axd[Name].legend()
            case 'Residuals':
                axd[Name].plot(Kinetics.X,Kinetics.SD, label='Residuals')
#                axd[Name].legend()
            case 'Autocorrelation':
                axd[Name].plot(Kinetics.X,Kin.AutoCorrelation(Kinetics.SD), label='Autocorrelation')
#                axd[Name].legend()
            case 'Histogram':
                axd[Name].hist(Kinetics.SD, bins=10)
            case 'Distribution':
                if Status['Analisys']:                
                    axd['Distribution'].plot(myMEM.tData.a,myMEM.p,'.')
                    axd['Distribution'].semilogx(myMEM.tData.a, myMEM.p)


    plt.show()


def ClickedStepOver():
    return 0

    
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
lstData={'Total time':100.0, 'Point number':200, 'Decay time':20.0,
         'Pulse width':2.,'Noise':0.05}
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
memData={'Start':0.25,'End':100,'Point number':50,'lagrange':1.}
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

# ====================== описание графиков ======================

fig, axd = plt.subplot_mosaic([['Kinetics','Kinetics'],
                              ['Residuals','Residuals'],
                              ['Autocorrelation','Histogram'],
                              ['Distrubution','Distrubution']],
                             figsize=(6,6),
                             height_ratios=[2,1,1,2],
                             width_ratios=[2,1],
                              layout='constrained',
                            # tight_layout=True,
                            label='Fluorescece kinetics')

Status={'Simulated':False,'Analisys':False}

window.mainloop()

