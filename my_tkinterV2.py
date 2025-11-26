import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from statsmodels.tsa.stattools import acf

import KineticsLib as Kin
import matplotlib.pyplot as plt
import numpy as np
import MaxEntPyV2 as MEM
                                       #from statsmodels.tsa.stattools import acf

# === Обработка нажатий кнопок ======

def ClickedReset():
    for Name in list(memData.keys()): memData[Name]=memVar[Name].get()
    for Name in list(memData.keys()): memVar[Name].set(memData[Name])

#    global myMEM
#    global setData
#    global expData
    global iteration

    if memData['log_grad']: gradient='log'        
    else: gradient='normal'
    

    myMEM.initial_image(memData['Point number'],step_type='log',range=[memData['Start'],memData['End']],gradient_type=gradient)
    myMEM.create_matices(norma=max(myMEM.Y))
    myMEM.update_functionals()

#    p,a = MEM.InitDistribution(memData['Start'],memData['End'],memData['Point number'], prms_type='log')
#    setData.p=p
#    setData.a=a
#    setData.p=np.ones(len(p))*sum(expData.Y)/np.sum(myMEM.C)

#    if fileVar['Post SQRT'].get():
#        setData.post_function=np.sqrt
#    else:
#        setData.post_function=MEM.UnitFunction
    
    
#    myMEM.Reset(expData,setData,memData['lagrange'])
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

    a=list(map(float,lstData['Decay time'].split(',')))
    coeffs=list(map(float,lstData['Amplitude'].split(',')))
    coeffs=coeffs/np.sum(coeffs)

    X=np.arange(0,lstData['Total time'], lstData['Total time']/lstData['Point number'])
    Y=np.array(myMEM.kernel(X,a[0])*coeffs[0])
    if len(coeffs)>1:
        for i in range(1, len(coeffs)):
            Y+=np.array(myMEM.kernel(X,a[i])*coeffs[i])

    X_for_R=X[:round(lstData['Point number']/lstData['Total time']*lstData['Pulse width'])]
                         
    R=np.array(1-((X_for_R-np.mean(X_for_R))/np.mean(X_for_R))**2) # перевернутая парабола, на границах 0, максимум = 1
#    R/=sum(R)
    if len(R)>0: Y=np.convolve(Y,R)[:len(X)]/sum(R)
    Y+=np.random.normal(0, lstData['Noise'], len(X))                                              

    myMEM.RawData(X,Y)
    myMEM.response(response_function=R)

#    simData.a=[lstData['Decay time']]
#    simData.p=[1.]
    
#    expData.R=MEM.GetResponse(expData.X,lstData['Total time']/lstData['Point number'],lstData['Pulse width'])
#    expData.Y=MEM.GetTheory(expData,simData)
#    expData.Y+=np.random.normal(0, lstData['Noise'], len(expData.X))                     

    Status['Simulated']=True
    Status['Analisys']=False
    Status['yLogScale']=False
    
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
                axd[Name].plot(myMEM.X,myMEM.Y,label='Kinetics')
                axd[Name].plot(myMEM.X[:len(myMEM.R)],myMEM.fY(myMEM.R),label='Impulse')
                if Status['Analisys']:
                    axd[Name].plot(myMEM.X,myMEM.T,label='Theory')
                if Status['yLogScale']:
                    axd[Name].semilogy(myMEM.X,myMEM.Y)
                axd[Name].legend()
            case 'Residuals':
                if Status['Analisys']:
                    axd[Name].plot(myMEM.X,myMEM.Res, label='Residuals')
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
                    axd['Distribution'].plot(myMEM.a,myMEM.p, linestyle='none')
                    axd['Distribution'].semilogx(myMEM.a, myMEM.p)


    plt.show()


def ClickedStepOver():

    global iteration

    myMEM.StepOver()

#    k,f = myMEM.MyGolden(name='Total', target='min')

#    print(myMEM.iteration,': ', end='')
    for name in myMEM.Vals.keys():
        print(name,':',round(myMEM.Vals[name],4), end=' ')
#    print(f' l: {round(myMEM.lagrange,4)}, dl: {round(myMEM.dlagr,4) }, Gold: {round(k,4)}', end=' ')
    print(f'disp: {round(np.sqrt(myMEM.disp),4)},sum(p):{round(np.sum(myMEM.p),4)}')

    myMEM.u+=myMEM.du

#    myMEM.p=myMEM.p+k*myMEM.dp
# 
#    myMEM.lagr+=myMEM.dlagr
#    setData.p=myMEM.p
 #   myMEM.Update(setData,myMEM.lagr)
    myMEM.update_functionals()
    PlotData()
    
def ClickedTuningDisp():
    deps=0.02

    
    myMEM.du=myMEM.TuningDisp(deps)

    k,f = myMEM.MyGolden(name='Chi2', dl=(0,1), eps=(deps,deps), target='min')

    myMEM.disp=myMEM.disp/(1+k*deps)

    myMEM.p=myMEM.p+myMEM.dp*k
    myMEM.lagr+=myMEM.dlagr*k
    myMEM.disp=myMEM.disp/(1+deps*k)    

    setData.p=myMEM.p
    myMEM.Update(setData,myMEM.lagr)
    
    for name in myMEM.Vals.keys():
        print(name,':',round(myMEM.Vals[name],4), end=' ')
    print(f'  lagr: {round(myMEM.lagr,4)}, dlagr: {round(myMEM.dlagr,4) }, Gold: {round(k,4)}')
    print(f'  lagr: {round(myMEM.lagr,4)}, dlagr: {round(myMEM.dlagr,4) }, Disp: {round(np.sqrt(myMEM.disp),4)}')
    
    PlotData()
    
def ApplyDataToVar():
    for Name in list(lstData.keys()): lstVar[Name].set(lstData[Name])
def ApplyVarToData():
    for Name in list(lstData.keys()): lstData[Name]=lstVar[Name].get()

def cmOpenFile():
    global FData
    file_path = filedialog.askopenfilename(title="Select a Text File",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")])

    if file_path:
        a=[]
        FData=0
        try:
            with open(file_path, 'r') as file:
                content = file.readlines()
        
            for lines in content:
                try:
                    b=list(map(float,lines.split()))
                    a+=[b]
                except: pass
            maxlen=0       
            for i in range(len(a)):
                maxlen=max(maxlen,len(a[i]))
            c=list(filter(lambda x: len(x)==maxlen, a))
            FData=np.array(c)
            
            myMEM.RawData(FData[:,0].copy(),FData[:,1].copy())
#            expData.X=FData[:,0].copy()
#            expData.Y=FData[:,1].copy()
#            expData.Y/=max(expData.Y)
#            if fileVar['Post SQRT'].get():
#                expData.Y=np.sqrt(extData.Y)

            myMEM.response(response_function=FData[:,2].copy())
#            expData.R=FData[:,2].copy()
#            expData.R/=max(expData.R)

            Status['Simulated']=True
            Status['Analisys']=False
            Status['yLogScale']=fileVar['LogScale'].get()

            fileVar['Left Index'].set(0)
            fileVar['Shift IRF'].set(0)
            fileVar['Width IRF'].set(len(FData))
            
            PlotData()

        except: print('Can not open:', file_path)
        
def cmRedrawFile():
    global FData
        
    i0=fileVar['Left Index'].get()
    irf_0=i0-fileVar['Shift IRF'].get()
    irf_1=min(irf_0+fileVar['Width IRF'].get(),len(FData)-irf_0)

    myMEM.IndexRange=range(i0, len(myMEM.rawX))
    myMEM.Ranges['XY']=range(i0, len(myMEM.rawX))
    #myMEM.X=myMEM.X[i0:].copy()
    #myMEM.Y=myMEM.Y[i0:].copy()
    #expData.Y/=max(expData.Y)
    
    if fileVar['Post SQRT'].get():
        myMEM.functionY(fY=lambda x: np.sqrt(x))
    else:
        myMEM.functionY()
    
    myMEM.Ranges['R']=range(irf_0,irf_1)
#    myMEM.R=FData[irf_0:irf_1,2].copy()
    #expData.R/=max(myMEM.R)

    Status['Simulated']=True

    Status['yLogScale']=fileVar['LogScale'].get()

    PlotData()

def cmResetDrawFile():
    Status['Simulated']=True
    Status['Analisys']=False
    Status['yLogScale']=fileVar['LogScale'].get()
    PlotData()    

def ClickedSDAnalisys():

    fig, axs = plt.subplots(2,1, figsize=(4,4), layout='constrained')
    n=len(myMEM.Y)//2
#    SD=myMEM.Y-myMEM.T
    axs[0].plot(myMEM.X[:n],acf(myMEM.Res,nlags=n-1), label='Autocorrelation')
    axs[0].legend()
    axs[1].hist(myMEM.Res, bins=10, label=str(round(np.std(myMEM.Res),4)))
    axs[1].legend()
    plt.show()        
def ClickedTestDerivatives():
    myMEM.test_derivatives()
        
# ============== основное тело программы ============
window = Tk()
window.title("Fluorescence decay analisys")
window.geometry('500x300')
window.resizable(False,False)

notebook=ttk.Notebook(window)
notebook.pack(expand=True, fill=BOTH)

Pages={'Simulation': ttk.Frame(notebook),
       'Open File': ttk.Frame(notebook),
       'MaxEnt Analisys': ttk.Frame(notebook)}
for Name in Pages.keys():
    Pages[Name].pack(fill=BOTH, expand=True)
    notebook.add(Pages[Name], text=Name)
    for c in range(3): Pages[Name].columnconfigure(index=c, weight=1)
    for r in range(8): Pages[Name].rowconfigure(index=r,weight=1)


# =========== страница симуляции кинетик =========================
lstData={'Total time':100.0, 'Point number':200,
         'Decay time':'10.0', 'Amplitude':'1.0',
         'Pulse width':2.,'Noise':0.04}
lstVar={'Total time':DoubleVar(), 'Point number':IntVar(),
        'Decay time':StringVar(), 'Amplitude':StringVar(),
         'Pulse width':DoubleVar(),'Noise':DoubleVar()}


for rw, Name in enumerate(list(lstData.keys())):
    lbl=Label(master=Pages['Simulation'], text=Name,font=("Arial",10))
    lbl.grid(column=0, row=rw)
    lstVar[Name].set(lstData[Name])    
    entry=Entry(master=Pages['Simulation'], textvariable=lstVar[Name], width=10)
    entry.grid(column=1, row=rw)
   

btnSimulate=ttk.Button(Pages['Simulation'],text="Simulate", command=SimulateClicked)
btnSimulate.grid(column=0, row=7)

btnQ=ttk.Button(Pages['Simulation'],text="Quit",command=quitclicked)
btnQ.grid(column=1, row=7)
# =========== страница открытия файла данных
fileData={'LogScale':False, 'Left Index':0, 'Width IRF':0, 'Shift IRF':0, 'Post SQRT':False}
fileVar={'LogScale':BooleanVar(),'Left Index':IntVar(), 'Width IRF':IntVar(),
         'Shift IRF':IntVar(),'Post SQRT':BooleanVar()}
        
for rw, Name in enumerate(list(fileData.keys())):
    lbl=Label(master=Pages['Open File'], text=Name,font=("Arial",10))
    lbl.grid(column=0, row=rw)
    fileVar[Name].set(fileData[Name])    
    if type(fileData[Name])==bool:
        entry=Checkbutton(master=Pages['Open File'], variable=fileVar[Name])
    else:
        entry=Entry(master=Pages['Open File'], textvariable=fileVar[Name], width=10)
    entry.grid(column=1, row=rw)


btnOpenFile=ttk.Button(Pages['Open File'],text="Open file",command=cmOpenFile,width=10)
btnOpenFile.grid(column=2, row=1)

btnResetDrawFile=ttk.Button(Pages['Open File'],text="Reset Analysis",command=cmResetDrawFile,width=15)
btnResetDrawFile.grid(column=2, row=4)

btnRedrawFile=ttk.Button(Pages['Open File'],text="Redraw file",command=cmRedrawFile,width=15)
btnRedrawFile.grid(column=2, row=3)

#================ Страница Макс этропии анализа ===================
memData={'Start':0.5,'End':100,'Point number':100,'lagrange':1.,'log_grad':False}
memVar={'Start':DoubleVar(),'End':DoubleVar(),'Point number':IntVar(),'lagrange':DoubleVar(),'log_grad':BooleanVar()}

for rw, Name in enumerate(list(memData.keys())):
    lbl_mem=Label(master=Pages['MaxEnt Analisys'], text=Name,font=("Arial",10))
    lbl_mem.grid(column=0, row=rw)
    memVar[Name].set(memData[Name])    
    if type(memData[Name])==bool:
        entry_mem=Checkbutton(master=Pages['MaxEnt Analisys'], variable=memVar[Name])
    else:
        entry_mem=Entry(master=Pages['MaxEnt Analisys'], textvariable=memVar[Name], width=15)

    entry_mem.grid(column=1, row=rw)
  
btnInit=ttk.Button(Pages['MaxEnt Analisys'],text='New', command=ClickedReset,width=15)
btnInit.grid(column=2,row=0)

btnStepOver=ttk.Button(Pages['MaxEnt Analisys'],text='Stepover', command=ClickedStepOver,width=15)
btnStepOver.grid(column=2,row=2)

btnShowVals=ttk.Button(Pages['MaxEnt Analisys'],text='Tune', command=ClickedTuningDisp,width=15)
btnShowVals.grid(column=2,row=3)

btnShowVals=ttk.Button(Pages['MaxEnt Analisys'],text='Show vals/grads', command=ClickedShowVals,width=15)
btnShowVals.grid(column=2,row=6)

btnShowVals=ttk.Button(Pages['MaxEnt Analisys'],text='SD analysis', command=ClickedSDAnalisys,width=15)
btnShowVals.grid(column=2,row=7)

btnShowVals=ttk.Button(Pages['MaxEnt Analisys'],text='Test derivatives', command=ClickedTestDerivatives,width=15)
btnShowVals.grid(column=2,row=8)


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

Status={'Simulated':False,'Analisys':False, 'yLogScale':False}

# == вводим пустые переменные
iteration=0

myMEM=MEM.DataModelling()

FData=0

window.mainloop()

