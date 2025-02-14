import tkinter as tk
from tkinter import *
from tkinter import ttk
import KineticsLib as Kin
import matplotlib.pyplot as plt
import numpy as np
#from statsmodels.tsa.stattools import acf

def quitclicked():
    plt.close()
    window.destroy()
def SimulateClicked():
    ApplyVarToData()
    ApplyDataToVar()
    Kinetics=Kin.TKinetics(lstData['Point number'],lstData['Total time'])
    Kinetics.InitY(lstData['Decay time'],1)
    Kinetics.InitR(lstData['Pulse width'])
    Kinetics.Convolve(Kinetics.R)
    Kinetics.InitNoise(lstData['Noise'])
    Kinetics.Y*=1+Kinetics.SD

    for Name in list(axd.keys()):
        axd[Name].clear()
        match Name:
            case 'Kinetics':
                axd[Name].plot(Kinetics.X,Kinetics.Y,label='Kinetics')
                axd[Name].plot(Kinetics.X[:len(Kinetics.R)],Kinetics.R,label='Impulse')
            case 'Residuals':
                axd[Name].plot(Kinetics.X,Kinetics.SD, label='Residuals')
            case 'Autocorrelation':
                axd[Name].plot(Kinetics.X,Kin.AutoCorrelation(Kinetics.SD), label='Autocolleration')
            case 'Histogram':
                axd[Name].hist(Kinetics.SD, bins=10)
        axd[Name].legend(fontsize='small')
                               
#    ax1.cla()
#    ax2.cla()
#    ax3.cla()
#    ax1.plot(Kinetics.X,Kinetics.Y,label='Kinetics')
#    ax1.plot(Kinetics.X[:len(Kinetics.R)],Kinetics.R,label='Impulse')
#    ax1.legend()
#    ax2.plot(Kinetics.X,Kinetics.SD, label='Residuals')
#    ax2.legend()
#    ax2.legend()
#    ax3.plot(Kinetics.X,Kin.AutoCorrelation(Kinetics.SD), label='Autocolleration')
#    ax3.legend()

    plt.show()
    
def ApplyDataToVar():
    for Name in list(lstData.keys()): lstVar[Name].set(lstData[Name])
def ApplyVarToData():
    for Name in list(lstData.keys()): lstData[Name]=lstVar[Name].get()


window = Tk()
window.title("Fluorescence decay analisys")
window.geometry('400x250')
window.resizable(False,False)


#fig, (ax1,ax2,ax3) =plt.subplots(3,1,figsize=(7,7), height_ratios=[2,1,1])
#ax1.set_title('Fluorescence kinetics')
#ax1.set_xlabel('Time')
#ax1.set_ylabel('Signal')
#ax2.set_title('Residuals')
#ax2.set_xlabel('Time')
#ax3.set_title('Autocolleration')
#ax3.set_xlabel('Time')

fig, axd=plt.subplot_mosaic([['Kinetics','Kinetics'],
                              ['Residuals','Residuals'],
                              ['Autocorrelation','Histogram']],
                             figsize=(7,7),
                             height_ratios=[2,1,1],
                            width_ratios=[2,1],
                            label='Fluorescece kinetics')
#for Name in list(axd.keys():
#    axd[Name].set_title(Name)
   


for c in range(2): window.columnconfigure(index=c, weight=1)
for r in range(7): window.rowconfigure(index=r,weight=1)

lstData={'Total time':100.0, 'Point number':200, 'Decay time':20.0,
         'Pulse width':2.,'Noise':0.05}
lstVar={'Total time':DoubleVar(), 'Point number':IntVar(), 'Decay time':DoubleVar(),
         'Pulse width':DoubleVar(),'Noise':DoubleVar()}

rw=1
for Name in list(lstData.keys()):
    lbl=Label(text=Name,font=("Arial",10))
    lbl.grid(column=0, row=rw)
    lstVar[Name].set(lstData[Name])    
    entry=Entry(textvariable=lstVar[Name], width=10)
    entry.grid(column=1, row=rw)
    rw+=1

btnSimulate=ttk.Button(window,text="Simulate", command=SimulateClicked)
btnSimulate.grid(column=0, row=6)

btnQ=ttk.Button(window,text="Quit",command=quitclicked)
btnQ.grid(column=1, row=6)

window.mainloop()

