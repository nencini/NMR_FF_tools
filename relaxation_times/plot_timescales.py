import yaml
import sys
import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os


def PlotTimescales(Ctimes,merge,name,title="Title",xlabel="xlabel"):
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
    ax1.set_xlabel(xlabel)
    #ax1.set_ylim([10**(-12.4), 10**(-6.8)])
    
    
    ax2.grid()
    ax2.set_ylim(0,1)
    ax2.set_ylabel("Coefficient's weights")
    ax2.set_xlabel(xlabel)

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
                    time_to_plot=working_Ctimes[timescale,0]*working_Ctimes[timescale,residue]
                    total_weight=working_Ctimes[timescale,residue]
                    for i in range(1,merge):
                        try:
                            time_to_plot+=working_Ctimes[timescale+i,0]*working_Ctimes[timescale+i,residue]
                            total_weight+=working_Ctimes[timescale+i,residue]
                            working_Ctimes[timescale,residue]+=working_Ctimes[timescale+i,residue]
                        except:
                            pass
                    time_to_plot/=total_weight
                                                       
                        
                if time_to_plot>10**(-9):
                    ax1.plot(residue, time_to_plot, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
                    ax2.plot(residue, working_Ctimes[timescale,residue], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
                    
                elif time_to_plot>10**(-11):
                    ax1.plot(residue, time_to_plot, marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue")
                    ax2.plot(residue, working_Ctimes[timescale,residue], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue")
                else:
                    ax1.plot(residue, time_to_plot, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
                    ax2.plot(residue, working_Ctimes[timescale,residue], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
                timescale+=merge-1
            timescale+=1
       
    

     
    plt.savefig("../figure_time_scales_protein/merging/"+name+"_"+str(merge)+".png")
    plt.show()   
    
    
    
def PlotTimescalesLinSlow(Ctimes,merge,name,title="Title",xlabel="xlabel"):
    working_Ctimes=np.copy(Ctimes)
    plt.rcParams["figure.figsize"] = [15.00, 7]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({'font.size': 20})


    fig, (ax1, ax2) = plt.subplots(2)

    ax1.title.set_text(title)
    ax1.set_ylim(10**(-10),Ctimes[-1,0]/1.9)

    ax1.grid()
    #ax1.set_yscale('log')
    ax1.set_ylabel("Timescale [s]")
    ax1.set_xlabel(xlabel)
    #ax1.set_ylim([10**(-12.4), 10**(-6.8)])

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
                            time_to_plot+=working_Ctimes[timescale,0]*working_Ctimes[timescale,residue]
                            total_weight+=working_Ctimes[timescale,residue]
                        except:
                            pass
                    time_to_plot/=total_weight
                                                       
                        
                if time_to_plot>10**(-8):
                    ax1.plot(residue, time_to_plot, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
                    
                elif time_to_plot>10**(-9)*5:
                    ax1.plot(residue, time_to_plot, marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue")
                elif time_to_plot>10**(-9):
                    ax1.plot(residue, time_to_plot, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
                elif time_to_plot>10**(-10):
                    ax1.plot(residue, time_to_plot, marker="o", markersize=5, markeredgecolor="black", markerfacecolor="black")
              
                timescale+=merge-1
            timescale+=1
       
    

    ax2.grid()
    ax2.set_ylim(0,1)
    ax2.set_ylabel("Coefficient's weights")
    ax2.set_xlabel(xlabel)


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
                    
                if time_to_plot>10**(-8):
                    ax2.plot(residue, working_Ctimes[timescale,residue], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
                elif time_to_plot>10**(-9)*5:
                    ax2.plot(residue, working_Ctimes[timescale,residue], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue")
                elif time_to_plot>10**(-9):
                    ax2.plot(residue, working_Ctimes[timescale,residue], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
                elif time_to_plot>10**(-10):
                    ax2.plot(residue, working_Ctimes[timescale,residue], marker="o", markersize=5, markeredgecolor="black", markerfacecolor="black")
              
                timescale+=merge-1
            timescale+=1

     
    plt.savefig("../figure_time_scales_protein/merging/"+name+"_"+str(merge)+"_slow_lin.png")
    plt.show()   
