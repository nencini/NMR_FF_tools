import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from datetime import date
import os
import re
import yaml
import time
import MDAnalysis as mda
import fnmatch

#addad 29.9.2022
# 20.1.2023 removed saving of yaml
# imput now from saved yaml, 5.2.2023
def plot_T1_T2_noe(aminoAcids,plot_output):
    plt.rcParams["figure.figsize"] = [15.00, 12]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({'font.size': 20})

    fig, (ax1, ax2, ax3) = plt.subplots(3)


    #ax1.grid()
    ax1.set_ylabel("T1 [s]")
    ax1.set_xlabel("Residue")
    ax2.set_ylabel("T2 [s]")
    ax2.set_xlabel("Residue")
    ax3.set_ylabel("hetNOE")
    ax3.set_xlabel("Residue")
    max_T1=0
    max_T2=0
    min_T1=10000
    min_T2=10000
    max_noe=-10000
    min_noe=10000
    
    
    
    for i in range(len(aminoAcids["results"])):
       
        ax1.plot(i,aminoAcids["results"][i]["T1"],"o",color="blue")
        max_T1=max(max_T1,aminoAcids["results"][i]["T1"])
        min_T1=min(min_T1,aminoAcids["results"][i]["T1"])

        ax2.plot(i,aminoAcids["results"][i]["T2"],"o",color="blue")
        max_T2=max(max_T2,aminoAcids["results"][i]["T2"])
        min_T2=min(min_T2,aminoAcids["results"][i]["T2"])

        ax3.plot(i,aminoAcids["results"][i]["hetNOE"],"o",color="blue")
        max_noe=max(max_noe,aminoAcids["results"][i]["hetNOE"])
        min_noe=min(min_noe,aminoAcids["results"][i]["hetNOE"])
    ax1.set_ylim([min_T1-(max_T1-min_T1)/8,max_T1+(max_T1-min_T1)/8 ])
    ax2.set_ylim([min_T2-(max_T2-min_T2)/8,max_T2+(max_T2-min_T2)/8 ])
    ax3.set_ylim([min_noe-(max_noe-min_noe)/8,max_noe+(max_noe-min_noe)/8 ])

    plt.show()
    fig.savefig(plot_output)


    

#addad 29.9.2022
#imput now from saveed yaml, updated 5.2.2023
def PlotTimescales(system,merge,groupTimes,title="Title",xlabel="xlabel",ylim=None,ylim_weig=None,yscale="log",plot_output="weight.pdf"):
    
    biggest_corr_time=np.log10(system["info"]['05_biggest_corr_time_[s]'])+12
    smallest_corr_time=np.log10(system["info"]['04_smallest_corr_time_[s]'])+12
    N_exp_to_fit=system["info"]['03_N_exp_to_fit']
    
    step_exp=(biggest_corr_time-smallest_corr_time)/N_exp_to_fit
    Ctimes = 10 ** np.arange(smallest_corr_time, biggest_corr_time, step_exp)
    Ctimes = Ctimes * 0.001 * 10 ** (-9);
    Ctimes_list=[Ctimes]

    for i in system["results"]["Coeff"]:
        Ctimes_list.append(system["results"]["Coeff"][i])
        
        Ctimes=np.array(Ctimes_list)
        Ctimes=np.transpose(Ctimes)
    
    
    working_Ctimes=np.copy(Ctimes)
    
    #print(working_Ctimes)
    plt.rcParams["figure.figsize"] = [15.00, 7]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({'font.size': 20})


    fig, (ax1, ax2) = plt.subplots(2)

    ax1.title.set_text(title)
    ax1.set_ylim(Ctimes[0,0]/10,Ctimes[-1,0]*10)

    ax1.grid()
    
    
    if yscale=="log":
        ax1.set_yscale('log')
    ax1.set_ylabel("Timescale [s]")
    #ax1.set_xlabel(xlabel)
    #ax1.set_ylim([10**(-12.4), 10**(-6.8)])
    if not ylim==None:
        ax1.set_ylim(ylim[0],ylim[1])
    if not ylim_weig==None:
        ax2.set_ylim(ylim_weig[0],ylim_weig[1])
    else:
        ax2.set_ylim(0,1)
                
        
    ax2.grid()

    ax2.set_ylabel("Coefficient's weights")
    ax2.set_xlabel(xlabel)
    
    """Plot the timescales, user specifies the merge to be used.
    The merge works as follow: The code finds the first timescale with
    weight bigger bigger than 0 and merges with 'merge' subsequent timescales.
    The final result is plotted as a weighted average of the merged points."""
    
    colors=["blue","orange","green","red","purple","brown","ping","gray","olive","cyan"]
    
    for residue in range(1,working_Ctimes.shape[1]):
        timescale=0
        while timescale < working_Ctimes.shape[0]:
            #print("{} {} \n".format(i, j))
            if working_Ctimes[timescale,residue]>0:
                time_to_plot=working_Ctimes[timescale,0]
                total_weight=working_Ctimes[timescale,residue]
                if merge>1:
                    time_to_plot=0
                    total_weight=0
                    for i in range(0,merge):
                        try:
                            time_to_plot+=working_Ctimes[timescale+i,0]*working_Ctimes[timescale+i,residue]
                            total_weight+=working_Ctimes[timescale+i,residue]
                        except:
                            pass
                    time_to_plot/=total_weight
                                                       
                        
                if time_to_plot<groupTimes[0]:
                    ax1.plot(residue, time_to_plot, marker="o", markersize=5, markeredgecolor=colors[0], markerfacecolor=colors[0])
                    ax2.plot(residue, total_weight, marker="o", markersize=5, markeredgecolor=colors[0], markerfacecolor=colors[0])
                else:
                    for i in range(0,len(groupTimes)-1):
                        if time_to_plot>groupTimes[i] and time_to_plot<groupTimes[i+1]:
                            ax1.plot(residue, time_to_plot, marker="o", markersize=5, markeredgecolor=colors[i+1], markerfacecolor=colors[i+1])
                            ax2.plot(residue, total_weight, marker="o", markersize=5, markeredgecolor=colors[i+1], markerfacecolor=colors[i+1])
                        elif time_to_plot>groupTimes[-1]:
                            ax1.plot(residue, time_to_plot, marker="o", markersize=5, markeredgecolor=colors[len(groupTimes)+1], markerfacecolor=colors[len(groupTimes)+1])
                            ax2.plot(residue, total_weight, marker="o", markersize=5, markeredgecolor=colors[len(groupTimes)+1], markerfacecolor=colors[len(groupTimes)+1])
                
                timescale+=merge-1
            timescale+=1     
    fig.savefig(plot_output)
    plt.show()   




def plot_replicas(save_name,plotType,number,*replicas):
    plt.rcParams["figure.figsize"] = [15.00, 12]
    plt.rcParams["figure.autolayout"] = True

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    
    ax1.set_ylabel("T1 [s]")
    ax1.set_xlabel("Residue")
    ax2.set_ylabel("T2 [s]")
    ax2.set_xlabel("Residue")
    ax3.set_ylabel("hetNOE")
    ax3.set_xlabel("Residue")
    max_T1=0
    min_T1=100000
    max_T2=0
    min_T2=100000
    max_noe=-10000
    min_noe=10000
    col=["C0","C1","C2","C3","C4","C5","C6","C7"]
    colors=[]


    
    for colo in col:
        for nu in range(number):
            colors.append(colo)
            
    #colors=["blue","blue","blue","red","red","red","green","green","green","gray","gray","gray","brown","brown","brown"]
    averages={}

    for i in replicas[0]["results"]:
        averages[i]={}
        for j,replica in enumerate(replicas):
            #print(replica)
            if (j//number) not in averages[i]:
                averages[i][(j//number)]={}
                if "T1" not in averages[i][(j//number)]:
                    averages[i][(j//number)]["T1"]=[replica["results"][i]["T1"]]
                    averages[i][(j//number)]["T2"]=[replica["results"][i]["T2"]]
                    averages[i][(j//number)]["hetNOE"]=[replica["results"][i]["hetNOE"]]
            else:
                averages[i][(j//number)]["T1"].append(replica["results"][i]["T1"])
                averages[i][(j//number)]["T2"].append(replica["results"][i]["T2"])
                averages[i][(j//number)]["hetNOE"].append(replica["results"][i]["hetNOE"])
                
            if plotType=="all":
                ax1.plot(i+(0.1*(j//number)),replica["results"][i]["T1"],"o",color=colors[j])
                ax2.plot(i+(0.1*(j//number)),replica["results"][i]["T2"],"o",color=colors[j])
                ax3.plot(i+(0.1*(j//number)),replica["results"][i]["hetNOE"],"o",color=colors[j])
            max_T2=max(max_T2,replica["results"][i]["T2"])
            max_T1=max(max_T1,replica["results"][i]["T1"])
            min_T2=min(min_T2,replica["results"][i]["T2"])
            min_T1=min(min_T1,replica["results"][i]["T1"])
            max_noe=max(max_noe,replica["results"][i]["hetNOE"])
            min_noe=min(min_noe,replica["results"][i]["hetNOE"])
        if plotType=="average":
            for j,exp in enumerate(averages[i]):
                ax1.errorbar(i+0.1*j,np.average(averages[i][exp]["T1"]),np.std(averages[i][exp]["T1"],ddof=1)/np.sqrt(len(averages[i][exp]["T1"])),fmt='o',markersize=5,color=col[j])
                ax2.errorbar(i+0.1*j,np.average(averages[i][exp]["T2"]),np.std(averages[i][exp]["T2"],ddof=1)/np.sqrt(len(averages[i][exp]["T2"])),fmt='o',markersize=5,color=col[j])
                ax3.errorbar(i+0.1*j,np.average(averages[i][exp]["hetNOE"]),np.std(averages[i][exp]["hetNOE"],ddof=1)/np.sqrt(len(averages[i][exp]["hetNOE"])),fmt='o',markersize=5,color=col[j])
            
    ax1.set_ylim([0,max_T1+0.1 ])
    ax2.set_ylim([0,max_T2+0.1 ])
    ax3.set_ylim([min_noe-0.1,max_noe+0.1 ])
    
    ax1.set_ylim([min_T1-(max_T1-min_T1)/8,max_T1+(max_T1-min_T1)/8 ])
    ax2.set_ylim([min_T2-(max_T2-min_T2)/8,max_T2+(max_T2-min_T2)/8 ])
    ax3.set_ylim([min_noe-(max_noe-min_noe)/8,max_noe+(max_noe-min_noe)/8 ])
    
    
    
    plt.savefig(save_name)
    
    
    
def PlotTimescales_replicas(merge,groupTimes,title="Title",xlabel="xlabel",ylim=None,ylim_weig=None,yscale="log",*aminoAcidsReplicas):
    plt.rcParams["figure.figsize"] = [15.00, 7]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({'font.size': 20})

    
    
    fig, (ax1, ax2) = plt.subplots(2)

    ax1.title.set_text(title)
    
    
    
    ax1.grid()
    if yscale=="log":
        ax1.set_yscale('log')
    ax1.set_ylabel("Timescale [s]")
    ax1.set_xlabel(xlabel)
    
    
    #for residue in range(1,48):
    #    ax1.axvline(x = residue, color = '0.85', )
    
    
    ax2.grid()
    ax2.set_ylim(0,1)
    ax2.set_ylabel("Coefficient's weights")
    ax2.set_xlabel(xlabel)
    
    colors=["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    markers=["o","v","s","*","P","D","p","X"]
    for k,system in enumerate(aminoAcidsReplicas):
        
        biggest_corr_time=np.log10(system["info"]['05_biggest_corr_time_[s]'])+12
        smallest_corr_time=np.log10(system["info"]['04_smallest_corr_time_[s]'])+12
        N_exp_to_fit=system["info"]['03_N_exp_to_fit']

        step_exp=(biggest_corr_time-smallest_corr_time)/N_exp_to_fit
        Ctimes = 10 ** np.arange(smallest_corr_time, biggest_corr_time, step_exp)
        Ctimes = Ctimes * 0.001 * 10 ** (-9);
        Ctimes_list=[Ctimes]

        for i in system["results"]["Coeff"]:
            Ctimes_list.append(system["results"]["Coeff"][i])
        
            Ctimes=np.array(Ctimes_list)
            Ctimes=np.transpose(Ctimes)
        
        ax1.set_ylim(Ctimes[0,0]/10,Ctimes[-1,0]*10)

        working_Ctimes=np.copy(Ctimes)

        #ax1.set_ylim([10**(-12.4), 10**(-6.8)])
        if not ylim==None:
            ax1.set_ylim(ylim[0],ylim[1])
            
        if not ylim_weig==None:
            ax2.set_ylim(ylim_weig[0],ylim_weig[1])    
            

        """Plot the timescales, user specifies the merge to be used.
        The merge works as follow: The code finds the first timescale with
        weight bigger bigger than 0 and merges with 'merge' subsequent timescales.
        The final result is plotted as a weighted average of the merged points."""

        
        
        for residue in range(1,working_Ctimes.shape[1]):
            timescale=0
            while timescale < working_Ctimes.shape[0]:
                #print("{} {} \n".format(i, j))
                if working_Ctimes[timescale,residue]>0:
                    time_to_plot=working_Ctimes[timescale,0]
                    if merge>1:
                        time_to_plot=0
                        total_weight=0
                        for i in range(0,merge):
                            try:
                                time_to_plot+=working_Ctimes[timescale+i,0]*working_Ctimes[timescale+i,residue]
                                total_weight+=working_Ctimes[timescale+i,residue]
                            except:
                                pass
                        time_to_plot/=total_weight


                    if time_to_plot<groupTimes[0]:
                        ax1.plot(residue+k*0.15, time_to_plot, marker=markers[0], markersize=10, markeredgecolor=colors[k], markerfacecolor=colors[k])
                    else:
                        for i in range(0,len(groupTimes)-1):
                            if time_to_plot>groupTimes[i] and time_to_plot<groupTimes[i+1]:
                                ax1.plot(residue+k*0.15, time_to_plot, marker=markers[i+1], markersize=10, markeredgecolor=colors[k], markerfacecolor=colors[k])
                            elif time_to_plot>groupTimes[-1]:
                                ax1.plot(residue+k*0.15, time_to_plot, marker=markers[len(groupTimes)+1], markersize=10, markeredgecolor=colors[k], markerfacecolor=colors[k])

                    timescale+=merge-1
                timescale+=1






        for residue in range(1,working_Ctimes.shape[1]):
            timescale=0
            while timescale < working_Ctimes.shape[0]:
                #print("{} {} \n".format(i, j))
                if working_Ctimes[timescale,residue]>0:
                    time_to_plot=working_Ctimes[timescale,0]
                    if merge>1:
                        time_to_plot=0
                        total_weight=0
                        for i in range(1,merge):
                            try:
                                total_weight+=working_Ctimes[timescale,residue]
                                time_to_plot+=working_Ctimes[timescale,0]*working_Ctimes[timescale,residue]
                                working_Ctimes[timescale,residue]+=working_Ctimes[timescale+i,residue]

                            except:
                                pass
                        time_to_plot/=total_weight


                    if time_to_plot<groupTimes[0]:
                        ax2.plot(residue+k*0.15, working_Ctimes[timescale,residue], marker=markers[0], markersize=10, markeredgecolor=colors[k], markerfacecolor=colors[k])
                    else:
                        for i in range(0,len(groupTimes)-1):
                            if time_to_plot>groupTimes[i] and time_to_plot<groupTimes[i+1]:
                                ax2.plot(residue+k*0.15, working_Ctimes[timescale,residue], marker=markers[i+1], markersize=10, markeredgecolor=colors[k], markerfacecolor=colors[k])
                            elif time_to_plot>groupTimes[-1]:
                                ax2.plot(residue+k*0.15, working_Ctimes[timescale,residue], marker=markers[len(groupTimes)+1], markersize=10, markeredgecolor=colors[k], markerfacecolor=colors[k])
                    timescale+=merge-1
                timescale+=1

     
    
    plt.show()   



def PlotTimescales_replicas2(merge,shift,title,xlabel,ylim,yscale,units,labels,plot_output,*aminoAcidsReplicas):
    plt.rcParams["figure.figsize"] = [15.00, 7]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({'font.size': 20})

    
    
    fig, (ax1) = plt.subplots(1)

    ax1.title.set_text(title)
    
    uu=["[s]","[ms]","[us]","[ns]","[ps]","[fs]"]

    unit=uu[int(np.log10(units)/3)]

    
    ax1.grid()
    if yscale=="log":
        ax1.set_yscale('log')
    
    ax1.set_ylabel("Timescale "+unit)
    
    max_len=0
    for system in aminoAcidsReplicas:
        max_len=max(max_len,len(system["results"]["Coeff"]))
       

    
    ax1.set_xlim(0,max_len+int(max_len//4))
    ax1.set_xlabel(xlabel)
    
    colors=["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    markers=["o","v","s","*","P","D","p","X"]
    
    nn=labels
    for k,system in enumerate(aminoAcidsReplicas):
        
        biggest_corr_time=np.log10(system["info"]['05_biggest_corr_time_[s]'])+12
        smallest_corr_time=np.log10(system["info"]['04_smallest_corr_time_[s]'])+12
        N_exp_to_fit=system["info"]['03_N_exp_to_fit']

        step_exp=(biggest_corr_time-smallest_corr_time)/N_exp_to_fit
        Ctimes = 10 ** np.arange(smallest_corr_time, biggest_corr_time, step_exp)
        Ctimes = Ctimes * 0.001 * 10 ** (-9);
        Ctimes_list=[Ctimes]

        for i in system["results"]["Coeff"]:
            Ctimes_list.append(system["results"]["Coeff"][i])
        
            Ctimes=np.array(Ctimes_list)
            Ctimes=np.transpose(Ctimes)
        
        ax1.set_ylim(Ctimes[0,0]/10*units,Ctimes[-1,0]*10*units)

        working_Ctimes=np.copy(Ctimes)

        if not ylim==None:
            ax1.set_ylim(ylim[0],ylim[1]*units)
            
            

        """Plot the timescales, user specifies the merge to be used.
        The merge works as follow: The code finds the first timescale with
        weight bigger bigger than 0 and merges with 'merge' subsequent timescales.
        The final result is plotted as a weighted average of the merged points."""

        
        
        for residue in range(1,working_Ctimes.shape[1]):
            timescale=0
            while timescale < working_Ctimes.shape[0]:
                #print("{} {} \n".format(i, j))
                if working_Ctimes[timescale,residue]>0:
                    time_to_plot=working_Ctimes[timescale,0]
                    total_weight=working_Ctimes[timescale,residue]
                    if merge>1:
                        time_to_plot=0
                        total_weight=0
                        for i in range(0,merge):
                            try:
                                time_to_plot+=working_Ctimes[timescale+i,0]*working_Ctimes[timescale+i,residue]
                                total_weight+=working_Ctimes[timescale+i,residue]
                            except:
                                pass
                        time_to_plot/=total_weight
                    
                    ms=int(np.round(total_weight*20,0))


                    ax1.plot(residue+k*0.15*shift, time_to_plot*units, "o", markersize=ms, markeredgecolor=colors[k], markerfacecolor=colors[k])
                    
                    timescale+=merge-1
                timescale+=1






        
       
        ax1.plot(residue+k*0.15*0, 100**4, "o", markersize=20, color=colors[k], label=nn[k])
        
          
    ax1.legend()
      
    plt.savefig(plot_output)
    plt.show()
