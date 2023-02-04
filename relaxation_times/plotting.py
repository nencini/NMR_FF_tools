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
def plot_T1_T2_noe(aminoAcids,plot_output):
    plt.rcParams["figure.figsize"] = [15.00, 12]
    plt.rcParams["figure.autolayout"] = True

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
    max_noe=0
    min_noe=0
    
    
    
    for i in range(len(aminoAcids)):
       
        ax1.plot(i,aminoAcids[i].T1,"o",color="blue")
        max_T1=max(max_T1,aminoAcids[i].T1)

        ax2.plot(i,aminoAcids[i].T2,"o",color="blue")
        max_T2=max(max_T2,aminoAcids[i].T2)

        ax3.plot(i,aminoAcids[i].NOE,"o",color="blue")
        max_noe=max(max_noe,aminoAcids[i].NOE)
        min_noe=min(min_noe,aminoAcids[i].NOE)
    ax1.set_ylim([0,max_T1+0.1 ])
    ax2.set_ylim([0,max_T2+0.1 ])
    ax3.set_ylim([min_noe-0.1,max_noe+0.1 ])

    plt.show()
    fig.savefig(plot_output)


    

#addad 29.9.2022
def PlotTimescales(aminoAcids,merge,groupTimes,title="Title",xlabel="xlabel",ylim=None,ylim_weig=None,plot_output="weight.pdf"):
    
    step_exp=(aminoAcids[0].biggest_corr_time-aminoAcids[0].smallest_corr_time)/aminoAcids[0].N_exp_to_fit
    Ctimes = 10 ** np.arange(aminoAcids[0].smallest_corr_time, aminoAcids[0].biggest_corr_time, step_exp)
    Ctimes = Ctimes * 0.001 * 10 ** (-9);
    Ctimes_list=[Ctimes]

    for i in range(len(aminoAcids)):
        Ctimes_list.append(aminoAcids[i].Coeffs)
        Ctimes=np.array(Ctimes_list)
        Ctimes=np.transpose(Ctimes)
    
    
    working_Ctimes=np.copy(Ctimes)
    plt.rcParams["figure.figsize"] = [15.00, 7]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({'font.size': 20})


    fig, (ax1, ax2) = plt.subplots(2)

    ax1.title.set_text(title)
    ax1.set_ylim(Ctimes[0,0]/10,Ctimes[-1,0]*10)

    ax1.grid()
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

