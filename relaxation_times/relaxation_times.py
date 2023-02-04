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


# modified 2/11/2022
def CalculateCorrelationFunctions(path,begin,end,RM_avail,atom1,atom2,moleculeType,output_path,grofile=None,xtcfile=None,tprfile=None):
    """ Function to calculate Rotational Correlation functions from MD simulations.
    \n
    1) Creates index file
    2) Calculates RCF for the enteries in the index file.
    \n
    Takes following arguments:
      path - folder with gro, xtc, tpr, (README.yaml) files
      begin - where to start the RCF analysis, equivalent to -b in gromacs
              if begin==-1 and README.yaml exists, "EQILIBRATED" is used for the begining
      end - where to end the RCF analysis, equivalent to -e in gromacs
            if end==-1 and README.yaml exists, the whole trajectory is calculated
            if end==-1 and README.yaml DOES NOT exist, up to first 50 us are analyzed 
                                                       (should suffice for all of our cases)
      RM_avail - does README.yaml exist at "path" (True/False)
      atom1, atom 2 - name of the atoms used for analysis in the gro file
      moleculeType - Protein/"something_else" for index file creation purposes
                     Protein - creates separate groups in the index file for every atom1, atom2 pairs that are found
                     "something_else" - any name is allowed, 
                                        creates only 1 group that contains all atom1-atom2 pairs found
                                        Useful for lipids/suractants...
                                        RCF is calculated as an average from all the pairs found
      output_path - parent folder to store correlation function folders
    \n
    Optional arguments, mandatory when README.yaml not available:
      grofile -  default None, gro file in path
      xtcfile -  default None, xtc file in path
      tprfile -  default None, tpr file in path
      
    \n
    Output:
        Creates a folder at working directory with the name of gro file and saves correlation functions there.
        When README.yaml available, it saves the path of the correlations functions and date of analysis there
    """
    if RM_avail:
        readmeS = path+ "/README.yaml"
        with open(readmeS) as yaml_file:
            readme = yaml.load(yaml_file, Loader=yaml.FullLoader)
        grofile=readme["FILES_FOR_RELAXATION"]["gro"]["NAME"]
        xtcfile=readme["FILES_FOR_RELAXATION"]["xtc"]["NAME"]
        tprfile=readme["FILES_FOR_RELAXATION"]["tpr"]["NAME"]
    
        if end==-1:
            end=int(readme["FILES_FOR_RELAXATION"]["xtc"]["LENGTH"])
        if begin==-1:
            begin=int(readme["EQILIBRATED"])

        new_folder=output_path+readme["FILES"]["xtc"]["NAME"][:-4] + "_" + str(atom1) + "_" + str(atom2)
    else:
        new_folder=output_path+"corr_func"+ "_"  +   grofile[:-4] + "_" + str(int(begin/1000)) + "_" + str(int(end/1000)) + "_" + str(atom1) + "_" + str(atom2)
        if end==-1:
            end=50000000 # a dirty trick to deal with the lack of readme file, for the moment, will improve in the future
        if begin==-1:
            begin=0
        last_frame_should=42
    
    xtc=xtcfile
    tpr=tprfile
    grofile=path+grofile
    xtcfile=path+xtcfile
    tprfile=path+tprfile
    
    ##### MAKE NDX FILE #####
    if RM_avail:
        readme["FILES_FOR_RELAXATION"]["ndx_"+atom1+"_"+atom2]={}
        readme["FILES_FOR_RELAXATION"]["ndx_"+atom1+"_"+atom2]["NAME"]="index_"+atom1+"_"+atom2+".ndx"
        output_ndx=path+readme["FILES_FOR_RELAXATION"]["ndx_"+atom1+"_"+atom2]["NAME"]
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
    analyze=True
    if RM_avail:
        if not 'ANALYSIS' in readme:
            readme['ANALYSIS']={}

        if not 'CORRELATION_FUNCTIONS' in readme['ANALYSIS']:
            readme['ANALYSIS']['CORRELATION_FUNCTIONS']={}

        if not 'RELAXATION_TIMES' in readme['ANALYSIS']:
            readme['ANALYSIS']['RELAXATION_TIMES']={}
            
        if not moleculeType in readme['ANALYSIS']['RELAXATION_TIMES']:
            readme['ANALYSIS']['RELAXATION_TIMES'][moleculeType]={}
            
            
        if not moleculeType in readme['ANALYSIS']['CORRELATION_FUNCTIONS']:
            readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]={}
        
        anal_to_save="analysis0"
        if moleculeType=="Protein":
            for analysis in readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]:
                if readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][analysis]["PATH"]==new_folder:
                    anal_to_save=analysis
                else:
                    anal_to_save="analysis"+str(len(readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]))
        else:
            if "help" in readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]:
                for analysis in readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"]:
                    if readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"][analysis]["PATH"]==new_folder:
                        anal_to_save=analysis
                    else:
                        anal_to_save="analysis"+str(len(readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"]))


        if not anal_to_save in readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType] and moleculeType=="Protein":
            readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]={}
            readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["PATH"]=new_folder
        elif not moleculeType=="Protein":
            if "help" not in  readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]:
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"]={}
            if not anal_to_save in readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"]:
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"][anal_to_save]={}
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"][anal_to_save]["PATH"]=new_folder

        #check if the analysis was already performed
        file_adress = path+"/"+readme["FILES_FOR_RELAXATION"]["xtc"]["NAME"]
        timepre=os.path.getmtime(file_adress)
        file_mod = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timepre))
        if moleculeType=="Protein":
            if "FROM_XTC" in readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]:
                if (readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["FROM_XTC"]==file_mod 
                  and not "Problem" in str(readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["LENGTH"])):
                    if readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["BEGGIN"]==begin and readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["END"]==end: 
                        analyze=False
                        print("Correlation function for ",readme["FILES"]["tpr"]["NAME"][:-4]," already calculated.")
               
           

    if analyze:
        if RM_avail:
            last_frame_should=int(readme["FILES_FOR_RELAXATION"]["xtc"]["LENGTH"])-int(readme["FILES_FOR_RELAXATION"]["xtc"]["SAVING_FREQUENCY"])
        all_alright=True
        if os.path.isdir(new_folder):
            os.system("rm -r "+new_folder)
        os.system("mkdir " + new_folder)
        print("Number of corelation functions to calculate: {} \n".format(residues))
        for i in range(0,residues):
            print("Calculatin correlation function {}".format(i+1),end=", ")
            
            os.system("echo " + str(i) + ' | gmx rotacf -f ' + xtcfile + ' -s ' + tprfile + '  -n ' + output_ndx + '  -o ' + new_folder + '/NHrotaCF_' + str(i) + ' -P 2 -d -e ' + str(end) + ' -b ' +str(begin)+' 2> corr.log')
            groups=[]
            with open("corr.log", 'rt') as corr_log:
                for line in corr_log:
                    if "Reading frame" in line:
                        last_frame=int(float(line.split()[4]))
                    if "Last frame" in line:
                        last_frame=int(float(line.split()[4]))
                    if "Group" in line:
                        groups.append(line.split()[3])
                    if "Done with trajectory" in line:
                        last_frame=last_frame_should
            if RM_avail:
                if not last_frame==last_frame_should:
                    all_alright=False
            print(" last frame",last_frame)
            
            if RM_avail:
                if moleculeType=="Protein":
                    if i not in readme['ANALYSIS']['RELAXATION_TIMES'][moleculeType]:
                        readme['ANALYSIS']['RELAXATION_TIMES'][moleculeType][i]={}
                        readme['ANALYSIS']['RELAXATION_TIMES'][moleculeType][i]["RESIDUE"]=groups[i][0:len(groups[i])-1]
                else:
                    create_new=True
                    for j in readme['ANALYSIS']['RELAXATION_TIMES'][moleculeType]:
                        if readme['ANALYSIS']['RELAXATION_TIMES'][moleculeType][j]["RESIDUE"]==groups[i][0:len(groups[i])-1]:
                            create_new=False
                    if create_new:
                        new_index=len(readme['ANALYSIS']['RELAXATION_TIMES'][moleculeType])
                        readme['ANALYSIS']['RELAXATION_TIMES'][moleculeType][new_index]={}
                        readme['ANALYSIS']['RELAXATION_TIMES'][moleculeType][new_index]["RESIDUE"]=groups[i][0:len(groups[i])-1]
                        
        if all_alright and RM_avail:
            if moleculeType=="Protein":
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["LENGTH"]=last_frame
            else:
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"][anal_to_save]["LENGTH"]=last_frame
        elif RM_avail:
            if moleculeType=="Protein":
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["LENGTH"]="Problem at "+str(last_frame)
            else:
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"][anal_to_save]["LENGTH"]="Problem at "+str(last_frame)                
        try:
            os.system("rm corr.log")
        except:
            pass
        
        #directory = os.getcwd()

        correl={}
         
            
        if RM_avail:
            #readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["PATH"]=output_path

            today = str(date.today())
            
            if moleculeType=="Protein":
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["ANALYZED"]=today
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["FROM_XTC"]=file_mod
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["BEGGIN"]=begin
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["END"]=end
            else:
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"][anal_to_save]["ANALYZED"]=today
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"][anal_to_save]["FROM_XTC"]=file_mod
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"][anal_to_save]["BEGGIN"]=begin
                readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"][anal_to_save]["END"]=end
    
    
            with open(readmeS, 'w') as f:
                yaml.dump(readme,f, sort_keys=False)
            
             
        
        correl["xtc"]=xtc
        correl["tpr"]=tpr 
        correl["path"]=path
        correl["FROM_XTC"]=file_mod
        correl["BEGGIN"]=begin
        correl["END"]=end
        if moleculeType=="Protein":
            correl["LENGTH"]=readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType][anal_to_save]["LENGTH"]
            correl["BONDS"]={}
            
            for i in readme['ANALYSIS']['RELAXATION_TIMES'][moleculeType]:
                correl["BONDS"][i]=readme['ANALYSIS']['RELAXATION_TIMES'][moleculeType][i]["RESIDUE"]
                
        else:
            correl["LENGTH"]=readme['ANALYSIS']['CORRELATION_FUNCTIONS'][moleculeType]["help"][anal_to_save]["LENGTH"]
        correl["atom1"]=atom1
        correl["atom2"]=atom2
        correl["ANALYZED"]=today
    
        with open(new_folder+"/README_correl.yaml", 'w') as f:
            yaml.dump(correl,f, sort_keys=False)
    


def CorrelationFunctionsLipids(parent_folder_path,begin,end,RM_avail,moleculeType,output_path,systems,CH_bonds):
    """
    Get Correlation Functions for different bonds in lipids/other molecules.
    
    This function is calling CalculateCorrelationFunctions(...) function
    and later on reorganizes data.
    
    
    (folder_path,begin,end,RM_avail,moleculeType,output_path) 
    are inputs for CalculateCorrelationFunctions
        
        parent_folder_path - subfolders with xtc, tpr, README.yaml ...
        begin -              corresponds to -b in gmx
                             if -1, value of EQUILIBRATED from README.yeml is used
        end -                corresponds to -e in gmx
                             if -1, the whole trajectory is used
        RM_avail -           must be True
        moleculeType-        at the moment works only for 'All'
        output_path -        place to create a folder with results
        
    further inputs:
        systems    -         to specify if only subset of systems 
                             in parent_folder_path to be used
        CH_bonds   -         bonds in the system to calculate CF for
                             fails if different parts in the system
                             have the same name (let's say C1, H11 bond
                             exists in a lipid molecule as well as in
                             protein)
                             if you run into this problem, create a simulation
                             with only protein/lipids
                             this should be improved here in the future
        
    """
    for file in os.listdir(parent_folder_path):
        folder_path = parent_folder_path+os.fsdecode(file)+"/"
        for system in systems:
            if fnmatch.fnmatch(os.fsdecode(file), "*"+system+"*"):

                
                readmeS = folder_path+ "/README.yaml"
                with open(readmeS) as yaml_file:
                    readme = yaml.load(yaml_file, Loader=yaml.FullLoader)
                
                run_again=True
                main_name=readme["FILES"]["xtc"]["NAME"][:-4]
                if os.path.isfile(output_path+main_name+"/README_correl.yaml"):
                    
                    if end==-1:
                        e=int(readme["FILES_FOR_RELAXATION"]["xtc"]["LENGTH"])
                    else:
                        e=end
                    if begin==-1:
                        b=int(readme["EQILIBRATED"])
                    else:
                        b=begin
                        
                    with open(output_path+main_name+"/README_correl.yaml") as yaml_file:
                        readme_new = yaml.load(yaml_file, Loader=yaml.FullLoader)
                    
                 
                    
                    if (readme_new["xtc"]==folder_path+readme["FILES_FOR_RELAXATION"]["xtc"]["NAME"]
                        and readme_new["xtc_MODIFIED"]==readme["FILES_FOR_RELAXATION"]["xtc"]["MODIFIED"]
                        and readme_new["tpr"]==folder_path+readme["FILES_FOR_RELAXATION"]["tpr"]["NAME"]
                        and readme_new["BEGGIN"]==b and readme_new["END"]==e and readme_new["PROBLEMS"]==None):
                        run_again=False
                
          
                if run_again:
                    for atom1, atom2 in CH_bonds:
                        CalculateCorrelationFunctions(folder_path,begin,end,RM_avail,atom1,atom2,moleculeType,output_path)


                    

                    try:
                        os.system("mkdir "+output_path+main_name)
                    except:
                        pass


                    new_readme={}
                    new_readme["PROBLEMS"]={}
                    new_readme["BONDS"]={}
                    all_OK=True
                    for i,file2 in enumerate(os.listdir(output_path)):
                        correl_path=output_path+os.fsdecode(file2)+"/"
                        if main_name in os.fsdecode(file2) and main_name!=os.fsdecode(file2):

                            for k,bond in enumerate(CH_bonds):
                                bond_name=bond[0]+"_"+bond[1]
                                if bond_name in os.fsdecode(file2):
                                    os.system("mv "+correl_path+"/NHrotaCF_0.xvg "+output_path+main_name+"/NHrotaCF_"+str(k)+".xvg")

                                    with open(correl_path+"/README_correl.yaml") as yaml_file:
                                        old_readme = yaml.load(yaml_file, Loader=yaml.FullLoader)

                                    new_readme["xtc"]=old_readme["xtc"]
                                    new_readme["path"]=old_readme["path"]
                                    new_readme["xtc_MODIFIED"]=old_readme["FROM_XTC"]
                                    new_readme["tpr"]=old_readme["tpr"]
                                    new_readme["BEGGIN"]=old_readme["BEGGIN"]
                                    new_readme["END"]=old_readme["END"]
                                    new_readme["ANALYZED"]=old_readme["ANALYZED"]
                                    if "Problem" in str(old_readme["LENGTH"]):
                                        new_readme["PROBLEMS"][bond_name]=old_readme["LENGTH"]
                                    else:
                                        new_readme["BONDS"][k]=str(bond[0])+"_"+str(bond[1])

                                    os.system("rm -r "+correl_path)

                    if len(new_readme["PROBLEMS"])==0:
                        new_readme["PROBLEMS"]=None
                        path_to_readme=readme["ANALYSIS"]["CORRELATION_FUNCTIONS"]["All"]["help"]["analysis0"]["PATH"]
                        del(readme["ANALYSIS"]["CORRELATION_FUNCTIONS"]["All"]["help"])
                        alanysis=len(readme["ANALYSIS"]["CORRELATION_FUNCTIONS"]["All"])
                        readme["ANALYSIS"]["CORRELATION_FUNCTIONS"]["All"]["analysis"+str(alanysis)]={}
                        readme["ANALYSIS"]["CORRELATION_FUNCTIONS"]["All"]["analysis"+str(alanysis)]["PATH"]=path_to_readme
                        readme["ANALYSIS"]["CORRELATION_FUNCTIONS"]["All"]["analysis"+str(alanysis)]["LENGTH"]=old_readme["LENGTH"]
                        readme["ANALYSIS"]["CORRELATION_FUNCTIONS"]["All"]["analysis"+str(alanysis)]["ANALYZED"]=old_readme["ANALYZED"]
                        readme["ANALYSIS"]["CORRELATION_FUNCTIONS"]["All"]["analysis"+str(alanysis)]["FROM_XTC"]=old_readme["FROM_XTC"]
                        readme["ANALYSIS"]["CORRELATION_FUNCTIONS"]["All"]["analysis"+str(alanysis)]["BEGGIN"]=old_readme["BEGGIN"]
                        readme["ANALYSIS"]["CORRELATION_FUNCTIONS"]["All"]["analysis"+str(alanysis)]["END"]=old_readme["END"]


                    with open(output_path+main_name+"/README_correl.yaml", 'w') as f:
                        yaml.dump(new_readme,f, sort_keys=True)

                    with open(readmeS, 'w') as f:
                        yaml.dump(readme,f, sort_keys=False)


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
        self.analyze=analyze
        
        self.org_corrF, self.times_out=self.read_data()
        
        #analyze only the part specified by the user
        self.analyze_until = round(len(self.org_corrF)*analyze)
        self.org_corrF=self.org_corrF[0:self.analyze_until]
        self.times_out=self.times_out[0:self.analyze_until]
        
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
        T1, T2, NOE = choose_nuclei[self.nuclei](self.magnetic_field,Coeffs,Ctimes,self.OP) 


        
        #get the reconstucted correlation function
        self.rec_corrF=Cexp_mat.dot(Coeffs)
        self.plot_fit(self.rec_corrF)
        self.plot_exp_hist(Ctimes,Coeffs)

        return Teff, tau_eff_area, T1, T2, NOE


    def plot_fit(self, reconstruction):
        plt.figure(figsize=(15, 6))
        plt.rcParams.update({'font.size': 20})
        #plt.rcParams.update({'font.weight': "normal"})

        plt.plot(self.times_out,self.org_corrF,label="Original")
        plt.plot(self.times_out,reconstruction,label="Fit")
        plt.xlabel("Time [ps]")
        plt.ylabel("Autocorrelation function")
        lines=len(self.title)//40
        new_title=""
        for i in range(lines):
            new_title+=self.title[i*40:(i+1)*40]+" \n "
        new_title+=self.title[(i+1)*40:]   
        plt.title(new_title)
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




        
    

    







#added 18.10.2022s
def plot_replicas(*replicas):
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
    max_T2=0
    max_noe=0
    min_noe=0
    colors=["blue","red","green","gray","brown"]
    
    for j,replica in enumerate(replicas):
        for i in replica:
            ax1.plot(i,replica[i]["T1"],"o",color=colors[j])
            max_T1=max(max_T1,replica[i]["T1"])

            ax2.plot(i,replica[i]["T2"],"o",color=colors[j])
            max_T2=max(max_T2,replica[i]["T2"])

            ax3.plot(i,replica[i]["hetNOE"],"o",color=colors[j])
            max_noe=max(max_noe,replica[i]["hetNOE"])
            min_noe=min(min_noe,replica[i]["hetNOE"])

        ax1.set_ylim([0,max_T1+0.1 ])
        ax2.set_ylim([0,max_T2+0.1 ])
        ax3.set_ylim([min_noe-0.1,max_noe+0.1 ])



#added 29.9.2022
# 20.1.2023 add saving of yaml, includes now info on how the data was analyzed
def analyze_all_in_folder(OP,smallest_corr_time, biggest_corr_time, N_exp_to_fit,analyze,magnetic_field,folder_path,nuclei,output_path_relax,output_path_timescales,output_name=None):
    aminoAcids={}
    
    cr=False
    
    if os.path.isfile(folder_path+"README_correl.yaml"):
        with open(folder_path+"README_correl.yaml") as yaml_file:
                corr_readme = yaml.load(yaml_file, Loader=yaml.FullLoader)
        cr=True
    
    
    if cr and output_name==None:
        output_name=corr_readme["xtc"][:-4]
    
    for file in os.listdir(folder_path):
        if "README" not in os.fsdecode(file):
            x = re.findall("[0-9]", os.fsdecode(file))
            AA_index=""
            for i in x:
                AA_index+=i
            AA_index=int(AA_index)
            input_corr_file = folder_path+os.fsdecode(file)
            AA=GetRelaxationData(OP,smallest_corr_time, biggest_corr_time, N_exp_to_fit,analyze,magnetic_field,input_corr_file,nuclei,output_name+" "+str(AA_index))
            aminoAcids[AA_index]=AA
        else:
            print("correl function should exist")
            cr=True
            with open(folder_path+os.fsdecode(file)) as yaml_file:
                corr_readme = yaml.load(yaml_file, Loader=yaml.FullLoader)
    
    
    
    
    output_yaml_relax=output_path_relax+output_name+"_relax.yaml"
    output_yaml_timescales=output_path_timescales+output_name+"_timescales.yaml"
    
        
    relax_data={}
    relax_data["results"]={}
    relax_data["info"]={}
    relax_data["info"]["07_OP"]=OP
    relax_data["info"]["04_smallest_corr_time_[s]"]=10**(int(smallest_corr_time)-12)
    relax_data["info"]["05_biggest_corr_time_[s]"]=10**(int(biggest_corr_time)-12)
    relax_data["info"]["03_N_exp_to_fit"]=N_exp_to_fit
    relax_data["info"]["01_magnetic_field_[T]"]=magnetic_field
    relax_data["info"]["02_magnetic_field_[MHz]"]=float(np.round(magnetic_field /(2*np.pi/gammaH*10**6),2))
    relax_data["info"]["00_nuclei"]=nuclei
    relax_data["info"]["08_corr_func_length_[ps]"]=float(len(AA.org_corrF)*(AA.times_out[1]-AA.times_out[0]))/analyze
    relax_data["info"]["09_saving_frequency_[ps]"]=float((AA.times_out[1]-AA.times_out[0]))
    relax_data["info"]["06_analyze"]=analyze
    
    if cr:
        relax_data["info"]["10_xtc"]=corr_readme["xtc"]
   
    timescales={}
    timescales["Coeff"]={}
    for i in range(len(aminoAcids)):
        relax_data["results"][i]={}
        relax_data["results"][i]["T1"]=float(aminoAcids[i].T1)
        relax_data["results"][i]["T2"]=float(aminoAcids[i].T2)
        relax_data["results"][i]["hetNOE"]=float(aminoAcids[i].NOE)
        
        if cr:
            relax_data["results"][i]["res"]=corr_readme["BONDS"][i]
        
        timescales["Coeff"][i]=aminoAcids[i].Coeffs.tolist()
    timescales["Ctime"]=aminoAcids[i].Ctimes.tolist()    
    
    
    ### loads old spin relaxation yaml and checks if the analysis already exists
    try:
        with open(output_yaml_relax) as yaml_file:
            content = yaml.load(yaml_file, Loader=yaml.FullLoader)
        
        done=False
        for analysis in content:
            if content[analysis]["info"]==relax_data["info"]:
                content[analysis]["result"]=relax_data["results"]
                done=True
        if not done:
            content["analysis"+str(len(content))]=relax_data
    except:
        content={}
        content["analysis"+str(len(content))]=relax_data
        pass         
    
    with open(output_yaml_relax, 'w') as f:
        yaml.dump(content,f, sort_keys=True)
    
    ### loads old timescale yaml and checks if the analysis already exists
    try:
        with open(output_yaml_timescales) as yaml_file:
            content2 = yaml.load(yaml_file, Loader=yaml.FullLoader)
        
        
        done=False
        for analysis in content2:
            if content2[analysis]["info"]==relax_data["info"]:
                content2[analysis]["results"]=timescales
                done=True
        if not done:
            ann="analysis"+str(len(content2))
            content2[ann]={}
            content2[ann]["info"]=relax_data["info"]
            content2[ann]["results"]=timescales
    except:
        content2={}
        ann="analysis"+str(len(content2))
        content2[ann]={}
        content2[ann]["info"]=relax_data["info"]
        content2[ann]["results"]=timescales
        
    
    with open(output_yaml_timescales, 'w') as f:
        yaml.dump(content2,f, sort_keys=True)


    return aminoAcids



#added 12.1.2023

class AverageCorrelFunction():
    def __init__(self,name, output_path, *paths):
        
        
        self.name=name
        self.output_path=output_path
        self.loaded_data={}
        
        self.get_average()
        
    def get_average(self):
        
        try:
            os.system("mkdir "+self.output_path)
        except:
            pass
        
        mini=10**10
        for i,repeat in enumerate(paths):
            
            self.input_data=repeat+self.name
            org_corrF, times_out=self.read_data()
            self.loaded_data[repeat]=[times_out,org_corrF]
            if len(times_out)<mini:
                minName=repeat
            mini=min(mini,len(times_out))
        
        self.average=[]
        a=len(self.loaded_data[minName][0])
        for i in range(a):
            av=[]
            for repeat in paths:
                av.append(self.loaded_data[repeat][1][i])
            self.average.append(np.mean(av))
        

        
        to_save=np.zeros([len(self.average),2])
        for i in range(len(self.average)):
            to_save[i,0]=self.loaded_data[minName][0][i]
            to_save[i,1]=self.average[i]
        
        
        np.savetxt(self.output_path+"/"+self.name,to_save)
        

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
    
    
def Average_Correl_Function_All_residues(output_path, *paths):
    for j,file in enumerate(os.listdir(paths[0])):
        AverageCorrelFunction(file,output_path,*paths)
        



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
