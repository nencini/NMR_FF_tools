import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from datetime import date
import os
import re

gammaD=41.695*10**6; #r*s^(-1)*T^(-1)
gammaH=267.513*10**6;
gammaC=67.262*10**6;
gammaN=-27.166*10**6;




#added 31.5.2022

def ReadREADME(path,moleculeType):
    readme = path+ "/README.yaml"
    with open(readme) as yaml_file:
        readme = yaml.load(yaml_file, Loader=yaml.FullLoader)
    
    grofile=path+readme["FILES_FOR_ANALYSIS"]["RELAXATION_TIMES"][moleculeType]["gro"]["NAME"]
    xtcfile=path+readme["FILES_FOR_ANALYSIS"]["RELAXATION_TIMES"][moleculeType]["xtc"]["NAME"]
    tprfile=path+readme["FILES_FOR_ANALYSIS"]["RELAXATION_TIMES"][moleculeType]["tpr"]["NAME"]

    return grofile, xtcfile, tprfile

def CalculateCorrelationFunctions(path,begin,end,RM_avail,grofile,xtcfile,tprfile,atom1,atom2,moleculeType):
    if RM_avail:
        readme = path+ "/README.yaml"
        with open(readme) as yaml_file:
            readme = yaml.load(yaml_file, Loader=yaml.FullLoader)
        grofile=readme["FILES_FOR_ANALYSIS"]["RELAXATION_TIMES"][moleculeType]["gro"]["NAME"]
        xtcfile=readme["FILES_FOR_ANALYSIS"]["RELAXATION_TIMES"][moleculeType]["xtc"]["NAME"]
        tprfile=readme["FILES_FOR_ANALYSIS"]["RELAXATION_TIMES"][moleculeType]["tpr"]["NAME"]
    
    
    if RM_avail:
        new_folder=readme["FILES"]["xtc"]["NAME"][:-4] + "_" + str(int(begin/1000)) + "_" + str(int(end/1000)) + "_" + str(atom1) + "_" + str(atom2)
    else:
        new_folder="corr_func"+ "_"  +   grofile[:-4] + "_" + str(int(begin/1000)) + "_" + str(int(end/1000)) + "_" + str(atom1) + "_" + str(atom2)
    
    
    
    grofile=path+grofile
    xtcfile=path+xtcfile
    tprfile=path+tprfile
    
    ##### MAKE NDX FILE #####
    #grofile=path+readme["FILES"]["gro"]["NAME"]
    if RM_avail:
        readme["FILES_FOR_ANALYSIS"]["RELAXATION_TIMES"][moleculeType]["ndx_"+atom1+"_"+atom2]={}
        readme["FILES_FOR_ANALYSIS"]["RELAXATION_TIMES"][moleculeType]["ndx_"+atom1+"_"+atom2]["NAME"]="index_"+atom1+"_"+atom2+".ndx"
        output_ndx=path+readme["FILES_FOR_ANALYSIS"]["RELAXATION_TIMES"][moleculeType]["ndx_"+atom1+"_"+atom2]["NAME"]
    else:
        output_ndx="index_"+atom1+"_"+atom2+".ndx"
    
    if moleculeType=="Protein":    
        with open(grofile, 'rt') as gro_file:
            residue=""
            residues=0
            with open(output_ndx, 'w') as fo:
                for line in gro_file:
                    if 'Title' in line or len(line.split())==1 or len(line.split())==3:
                        pass
                    else:    
                    
                        if line.split()[1]==atom1:
                            residue=line.split()[0]
                            N=int(line.split()[2])
                        if line.split()[1]==atom2:
                            HN=int(line.split()[2])
                            if residue==line.split()[0]:
                                fo.write("[ {} ]\n {} {}\n".format(residue,N,HN))
                                residues+=1
                                
    else:
        with open(grofile, 'rt') as gro_file:
            residue=""
            residues=1
            with open(output_ndx, 'w') as fo:
                fo.write("[ {}_{} ] \n".format(atom1,atom2))
                for line in gro_file:
                    if 'Title' in line or len(line.split())==1 or len(line.split())==3:
                        pass
                    else:    
                    
                        if line.split()[1]==atom1:
                            residue=line.split()[0]
                            N=int(line.split()[2])
                        if line.split()[1]==atom2:
                            HN=int(line.split()[2])
                            if residue==line.split()[0]:
                                fo.write(" {} {}\n".format(N,HN))
                                
    #########################
    
    ##### GET CORRELATION FUNCTIONS #####
    #xtcfile=path+readme["FILES"]["xtc"]["NAME"]
    #tprfile=path+readme["FILES"]["tpr"]["NAME"]
    if end==-1:
        end=int(readme["FILES"]["xtc"]["LENGTH"])
    
    
    
    if os.path.isdir(new_folder):
        os.system("rm -r "+new_folder)
    os.system("mkdir " + new_folder)
    print("Number of corelation functions to calculate: {} \n".format(residues))
    for i in range(0,residues):
        print("Calculatin correlation function {}".format(i+1))
        os.system("echo " + str(i) + ' | gmx rotacf -f ' + xtcfile + ' -s ' + tprfile + '  -n ' + output_ndx + '  -o ' + new_folder + '/NHrotaCF_' + str(i) + ' -P 2 -d -e ' + str(end) + ' -b ' +str(begin))
    



class GetRelaxationData():
    def __init__(self,OP,smallest_corr_time, biggest_corr_time, N_exp_to_fit,analyze,magnetic_field,input_data,nuclei,title):
        self.OP=OP
        self.smallest_corr_time=smallest_corr_time
        self.biggest_corr_time=biggest_corr_time
        self.N_exp_to_fit=N_exp_to_fit
        self.magnetic_field=magnetic_field
        self.input_data=input_data
        self.nuclei=nuclei
        self.title=title
        
        self.org_corrF, self.times_out=self.read_data()
        
        #analyze only the part specified by the user
        analyze_until = round(len(self.org_corrF)*analyze)
        self.org_corrF=self.org_corrF[0:analyze_until]
        self.times_out=self.times_out[0:analyze_until]
        
        Teff, tau_eff_area, self.T1, self.T2, self.NOE = self.calc_relax_time()
        
        print("T1: {} T2: {} NOE: {}".format(self.T1, self.T2, self.NOE))

        
    



    def read_data(self):
        # for reading the correlation function data
        opf = open(self.input_data, 'r')
        lines = opf.readlines()
        data_times = []
        data_F = []
        for line in lines:
            if '#' in line:
                continue
            if '&' in line:
                continue
            if '@' in line:
                continue    
            if 'label' in line:
                continue
            if line == "":
                continue
            parts = line.split()
            if np.shape(parts)[0]==2:
                data_F.append(float(parts[1]))
                data_times.append(float(parts[0]))


        data_Fout = np.array(data_F)
        times_out = np.array(data_times)
        return data_Fout, times_out


    def calc_relax_time(self):
   
        # normalized correlation fuction
        NcorrF = (self.org_corrF - self.OP ** 2) / (1 - self.OP ** 2);

    
        # Create correlation times from the times and number of exponential specified by the user
        step_exp=(self.biggest_corr_time-self.smallest_corr_time)/self.N_exp_to_fit
        Ctimes = 10 ** np.arange(self.smallest_corr_time, self.biggest_corr_time, step_exp)

        # First, no forcing the plateou
        # create exponential functions and put them into a matrix, individual exponentials in columns
        #the lengthe of correlationd data to be used is specified by the user
        n = len(self.times_out)
        m = len(Ctimes)
        Cexp_mat = np.zeros((n, m))

        for i in range(0, n):
            for j in range(0, m):
                Cexp_mat[i, j] = np.exp(-self.times_out[i] / Ctimes[j])

        #least square solution
        Coeffs, res = optimize.nnls(Cexp_mat, NcorrF[0:n])

        # Effective correlation time from components, in units of sec

        Teff = sum(Coeffs * Ctimes * 0.001 * 10 ** (-9))

        # calculate t_eff from area
        dt = self.times_out[2] - self.times_out[1]
        pos = np.argmax(NcorrF[0:n] < 0);

        if pos > 0:
            tau_eff_area = sum(NcorrF[0:pos]) * dt * 0.001 * 10 ** (-9);
            conv = 1
        else:
            tau_eff_area = sum(NcorrF[0:n]) * dt * 0.001 * 10 ** (-9);
            conv = 0

   

        # changin the unit of time permanently
        Ctimes = Ctimes * 0.001 * 10 ** (-9);
        self.Coeffs=Coeffs
        self.Ctimes=Ctimes
	
        #Calculate the relaxation times for chosen nuclei
        R1, R2, NOE = choose_nuclei[self.nuclei](self.magnetic_field,Coeffs,Ctimes,self.OP) 


        
        #get the reconstucted correlation function
        self.rec_corrF=Cexp_mat.dot(Coeffs)
        self.plot_fit(self.rec_corrF)
        self.plot_exp_hist(Ctimes,Coeffs)

        return Teff, tau_eff_area, R1, R2, NOE


    def plot_fit(self, reconstruction):
        plt.figure(figsize=(15, 6))
        plt.rcParams.update({'font.size': 20})
        #plt.rcParams.update({'font.weight': "normal"})

        plt.plot(self.times_out,self.org_corrF,label="Original")
        plt.plot(self.times_out,reconstruction,label="Fit")
        plt.xlabel("Time [ps]")
        plt.ylabel("Autocorrelation function")
        plt.title(self.title)
        plt.legend()
        plt.show()


    def plot_exp_hist(self,Ctimes,Coeffs):
        plt.figure(figsize=(15, 6))
        plt.rcParams.update({'font.size': 20})        
        plt.plot(Ctimes,Coeffs)
        plt.xlabel("Time decay [s]")
        plt.ylabel("Coefficient")

def get_relaxation_D(magnetic_field,Coeffs,Ctimes,OP):
    omega = gammaD * magnetic_field
    
    #initiate spectral densities
    J0 = 0
    J1 = 0
    J2 = 0
    
    m = len(Ctimes)
    for i in range(0, m):
        w=0
        J0 = J0 + 2 * Coeffs[i] * Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

        w = omega
        J1 = J1 + 2 * Coeffs[i] * Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

        w = 2* omega
        J2 = J2 + 2 * Coeffs[i] * Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

    xksi=167000 # quadrupolar coupling constant [Hz]
    R1 = 3 * (xksi  * np.pi) ** 2 / 40.0 * (1 - OP ** 2) * (0 * J0 + 2 * J1 + 8 * J2)
    R2 = 3 * (xksi  * np.pi) ** 2 / 40.0 * (1 - OP ** 2) * (3 * J0 + 5 * J1 + 2 * J2)

    return 1/R1, 1/R2, 0


def get_relaxation_C(magnetic_field,Coeffs,Ctimes,OP):
    omega = gammaD * magnetic_field
    
    wc = gammaC * magnetic_field;
    wh = gammaH * magnetic_field;
        
    #initiate spectral densities
    J0 = 0
    J1 = 0
    J2 = 0
    Jw1 = 0

    m = len(Ctimes)
    for i in range(0, m):
        w = wh - wc
        J0 = J0 + 2 * Coeffs[i] * Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

        w = wc
        J1 = J1 + 2 * Coeffs[i] * Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

        w = wc + wh
        J2 = J2 + 2 * Coeffs[i] * Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

    # note! R1's are additive. Nh from the Ferreira2015 paper correctly omitted here
    R1 = (22000 * 2 * np.pi) ** 2 / 20.0 * (1 - OP ** 2) * (J0 + 3 * J1 + 6 * J2)


    return 1/R1, 0, 0


def get_relaxation_N(magnetic_field,Coeffs,Ctimes,OP):
    
    
    wh = gammaH * magnetic_field 
    wn = gammaN * magnetic_field 
    
    #initiate spectral densities
    J0 = 0
    JhMn = 0
    JhPn = 0
    Jh = 0
    Jn = 0

    m = len(Ctimes)
    for i in range(0, m):
        w = 0
      
        J0 = J0 + 2 * Coeffs[i] * Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])
        
        w = wh-wn;
        JhMn = JhMn + 2 * Coeffs[i]* Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

        w = wn;
        Jn = Jn + 2 * Coeffs[i]* Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])
        
        w = wh;
        Jh= Jh + 2 * Coeffs[i]* Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

        w = wn+wh;
        JhPn = JhPn + 2 * Coeffs[i]* Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])


    mu = 4 * np.pi * 10**(-7) #magnetic constant of vacuum permeability
    h_planck = 1.055 * 10**(-34); #reduced Planck constant
    rN = 0.101 * 10**(-9); # average cubic length of N-H bond
    d = 1 * (mu * gammaN * gammaH * h_planck) / (4 * np.pi * rN**3); # dipolar coupling constant

    #units were corrected by S.Ollila and E.Mantzari, removed 2*pi from R1 and R2
    R1 = (d**2 / 20) * (1 * JhMn + 3 * Jn + 6 * JhPn) + Jn * (wn * 160 * 10**(-6))**2 / 15   ; 
    R2 = 0.5 * (d**2 / 20) * (4 * J0 + 3 * Jn + 1 * JhMn + 6 * Jh + 6 * JhPn) + (wn * 160 * 10**(-6))**2 / 90 * (4 * J0 + 3 * Jn);
    NOE = 1 + (d**2 / 20) * (6 * JhPn - 1 * JhMn) * gammaH / (gammaN * R1);


    #print("T1: {}, T2: {}, NOE: {}".format(1/R1, 1/R2, NOE))
    
    
           
    return 1/R1, 1/R2, NOE
    


        
         
choose_nuclei = {
    "13C": get_relaxation_C,
    "2H": get_relaxation_D,
    "15N": get_relaxation_N
}


#addad 29.9.2022
def plot_T1_T2_noe(aminoAcids):
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



#addad 29.9.2022
def PlotTimescales(aminoAcids,merge,groupTimes,title="Title",xlabel="xlabel",ylim=None):
    
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
    ax1.set_xlabel(xlabel)
    #ax1.set_ylim([10**(-12.4), 10**(-6.8)])
    if not ylim==None:
        ax1.set_ylim(ylim[0],ylim[1])

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
                else:
                    for i in range(0,len(groupTimes)-1):
                        if time_to_plot>groupTimes[i] and time_to_plot<groupTimes[i+1]:
                            ax1.plot(residue, time_to_plot, marker="o", markersize=5, markeredgecolor=colors[i+1], markerfacecolor=colors[i+1])
                        elif time_to_plot>groupTimes[-1]:
                            ax1.plot(residue, time_to_plot, marker="o", markersize=5, markeredgecolor=colors[len(groupTimes)+1], markerfacecolor=colors[len(groupTimes)+1])
                
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
                    

                if time_to_plot<groupTimes[0]:
                    ax2.plot(residue, working_Ctimes[timescale,residue], marker="o", markersize=5, markeredgecolor=colors[0], markerfacecolor=colors[0])
                else:
                    for i in range(0,len(groupTimes)-1):
                        if time_to_plot>groupTimes[i] and time_to_plot<groupTimes[i+1]:
                            ax2.plot(residue, working_Ctimes[timescale,residue], marker="o", markersize=5, markeredgecolor=colors[i+1], markerfacecolor=colors[i+1])
                        elif time_to_plot>groupTimes[-1]:
                            ax2.plot(residue, working_Ctimes[timescale,residue], marker="o", markersize=5, markeredgecolor=colors[len(groupTimes)+1], markerfacecolor=colors[len(groupTimes)+1])
                timescale+=merge-1
            timescale+=1

     
    
    plt.show()   

#added 29.9.2022
def analyze_all_in_folder(OP,smallest_corr_time, biggest_corr_time, N_exp_to_fit,analyze,magnetic_field,folder_path,nuclei,output_name):
    aminoAcids={}
    for j,file in enumerate(os.listdir(folder_path)):
        x = re.findall("[0-9]", os.fsdecode(file))
        AA_index=""
        for i in x:
            AA_index+=i
        AA_index=int(AA_index)
        input_corr_file = folder_path+os.fsdecode(file)
        AA=GetRelaxationData(OP,smallest_corr_time, biggest_corr_time, N_exp_to_fit,analyze,magnetic_field,input_corr_file,nuclei,output_name+" "+str(AA_index))
        aminoAcids[AA_index]=AA
    return aminoAcids

#addad 31.5.2022
#executed if not imported

if __name__ == "__main__":
    # help message is automatically provided
    # type=string, action=store is default
    parser = OptionParser()
    parser.add_option('-r', '--readme',  dest='RM_avail',  help='Read informaton from README.yaml. \n Useful for analysis of multiple data sets. \n Default: True', default=True)
    parser.add_option('-g', '--gro',  dest='grofile',  help='gro file name', default="file.gro")
    parser.add_option('-x', '--traj', dest='xtcfile', help='xtc file name.', default="traj.xtc")
    parser.add_option('-s', '--tpr', dest='tprfile', help='tpr file name.', default="top.tpr")
    parser.add_option('-o', '--out',  dest='out_fname',  help='output (OPs mean&std) file name', default="Headgroup_Glycerol_OPs.dat")
    opts, args = parser.parse_args()
