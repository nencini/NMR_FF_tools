import sys
import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
import os
import fnmatch
import yaml
import gc
import math
import warnings
import re
import time
from datetime import date

def go_through_simulation(folder_path):
    """
    
    Function to gather information about simulation and create README.yaml
    
    \n
    1) goes throught the content of a folder and check for presence of 
       "xtc","edr","tpr","top","mdp","ndx","gro","cpt","log" files,
       their size od date of modification
              
    2) if "gro" not available but "xtc", "tpr" exist and compatible,go
       script dumps "gro" file
       
    3) extracts length and saving frequency of the simulation
       temperature, composition and version of gromacs used 
       for mdrun and grompp 
       
    
    Input: folder_path - folder to check the content of 
    Returns: none
    Output: creates/updates README.yaml at folder_path
    """    
    
    
    readme_file = folder_path+ "/README.yaml"
    today = str(date.today())
    
    
    if not os.path.isfile(readme_file):
        readme={}
    else:
        with open(readme_file) as yaml_file:
            readme = yaml.load(yaml_file, Loader=yaml.FullLoader)
     
     
    sim=check_for_latest_files(folder_path,readme)
    
    
    
    
    
    
    
    # TODO: improve this part, at the moment simulation is assumed to be equilibrated at the time 0
    #       however, if modified lated in README by hand to different value, it works well         
    if not 'EQILIBRATED' in sim:
        #sim['EQILIBRATED'] = input("Biding of {} eqilibrated after [ps] \n".format(recieved_self.name))
        sim['EQILIBRATED'] = "0"

    
    #get version of gromacs used for simulation run from log file
    log_file= folder_path + "/"+sim["FILES"]["log"]["NAME"]    
    try:
        with open(log_file, 'rb') as log_info:
            for line in log_info:
                if ':-) GROMACS - gmx mdrun,' in str(line):
                    sim["FILES"]["log"]['GMX_VERSION']=(re.sub("'",'',str(line.split()[5])[1:]))
                      
    except:
        pass
    
    #get version of gromacs used for tpr creation from tpr file
    topology_tpr= folder_path+ "/"+sim["FILES"]["tpr"]["NAME"]
    try:
        with open(topology_tpr, 'rb') as tpr_info:
            for line in tpr_info:
                if 'VERSION' in str(line):
                    sim["FILES"]["tpr"]['GMX_VERSION']=(str(line.split()[1])[2:8])                    
    except:
        pass
    
    #get T and composition from tpr
    if not 'TEMPERATUREf' in sim:
        #get temperature from tpr; taken from AddData.py by Anne Kiirikki
        
        file1 =  'temporary_tpr.txt'

        print("Exporting information with gmx dump")  
        
        
        
        try:
            os.system('echo System | gmx dump -s '+ topology_tpr + ' > '+file1)

            with open(file1, 'rt') as tpr_info:
                topology_line=False
                for line in tpr_info:
                    if 'ref-t' in line:
                        sim['TEMPERATURE']=float(line.split()[1])
                    if 'topology:' in line:
                        topology_line=True
                        new_entry=False
                        
                        sim["COMPOSITION"]={}
                    if topology_line:
                        if 'moltype' in line:
                            molecule_name=re.sub('"','',line.split()[3])
                        if '#molecules' in line:
                            if molecule_name in sim["COMPOSITION"]:
                                sim["COMPOSITION"][molecule_name]+=int(line.split()[2])
                            else:
                                sim["COMPOSITION"][molecule_name]=int(line.split()[2])
                                
                    
                    if 'bIntermolecularInteractions' in line:
                        topology_line=False
            #os.system('rm '+file1)
        except:
            print("Cannot read tpr and get temperature")
  
    
    #if composition not read from tpr, try to get it from top file
    if not 'COMPOSITION' in sim:
        sim["COMPOSITION"]={}
        
        try:
            top_file= sim["FILES"]["top"]["NAME"]
            with open(top_file,"r") as f:
                molecules_list = False
                for line in f.readlines():
                    if molecules_list:
                        if not line.startswith(";"):
                            items = line.split()
                            if len(items)==2:
                                if items[0] in sim["COMPOSITION"]:
                                    sim["COMPOSITION"][items[0]]+=items[1]
                                else:
                                    sim["COMPOSITION"][items[0]]=items[1]
                    elif line.startswith("[ molecules ]"):
                        molecules_list = True
        except:
            print("Cannot read top file and assign the composition")
            #sim["COMPOSITION"]="Encountered problems"
    
    
    


    with open(readme_file, 'w') as f:
        yaml.dump(sim,f, sort_keys=False)
        
        
    print("great success!!!")
    



def check_for_latest_files(folder_path,readme):
    files_to_consider=["xtc","edr","tpr","top","mdp","ndx","gro","cpt","log"]
    sim=readme
    if not "FILES" in sim:
        sim["FILES"]={}
    for fileU in files_to_consider:
        if not fileU in sim["FILES"]:
            sim["FILES"][fileU]={}
            
       
        

        
         
        for file in os.listdir(folder_path):
            if fnmatch.fnmatch(os.fsdecode(file), "*."+fileU):
                file_adress = folder_path+os.fsdecode(file)
                if not "NAME" in sim["FILES"][fileU] or sim["FILES"][fileU]["NAME"] == "none": 
                    sim["FILES"][fileU]["NAME"] = os.fsdecode(file)
                timepre=os.path.getmtime(file_adress)
                file_mod = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timepre))
                sim["FILES"][fileU]["SIZE"]=os.path.getsize(file_adress)/1000000
                sim["FILES"][fileU]["MODIFIED"] = file_mod
    
        try:
            print(sim["FILES"][fileU]["NAME"])
        except:
            sim["FILES"][fileU]["NAME"]= "none"
            sim["FILES"][fileU]["SIZE"]= "none"
            sim["FILES"][fileU]["MODIFIED"] = "none"
    
    # tries to create a gro file if that does not exist
    if  sim["FILES"]["gro"]["SIZE"]== "none" and not sim["FILES"]["xtc"]["SIZE"]== "none":
        try:
            os.system('echo System | gmx trjconv -f '+ folder_path+sim["FILES"]["xtc"]["NAME"] + ' -s ' + folder_path+sim["FILES"]["tpr"]["NAME"] + ' -b 0 -e 0 -o ' + folder_path+sim["FILES"]["xtc"]["NAME"][:-4] + '.gro' + " >& /dev/null")
            sim["FILES"]["gro"]["NAME"] = sim["FILES"]["xtc"]["NAME"][:-4] + '.gro'
            
            file_adress = folder_path + "/" + sim["FILES"]["gro"]["NAME"]
                
            timepre=os.path.getmtime(file_adress)
            file_mod = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timepre))
            sim["FILES"]["gro"]["SIZE"]=os.path.getsize(file_adress)/1000000
            sim["FILES"]["gro"]["MODIFIED"] = file_mod
            
        except:
            pass
    
    try:
        if not sim["FILES"]["xtc"]["NAME"]=="none":
            mol = mda.Universe(folder_path+sim["FILES"]["gro"]["NAME"],folder_path+sim["FILES"]["xtc"]["NAME"])
            Nframes=len(mol.trajectory)
            timestep = mol.trajectory.dt
            trj_length = Nframes * timestep
            begin_time=mol.trajectory.time
        
            sim["FILES"]["xtc"]['SAVING_FREQUENCY'] = timestep
            sim["FILES"]['xtc']['LENGTH'] = trj_length
            sim["FILES"]['xtc']['BEGIN'] = begin_time

    except Exception as e: 
        print(e)
        print("gro or xtc do not exist in the folder, or they do not match")
    
    print("Checking for new trajectories within defiened conditions is succesfully finished")

    
    return sim


#added 18.10.2022
def remove_water(folder_path,save_part,xtc=False):
    """
    
    Function to save only subpart of xtc, gro, tpr (to remove water)
    
    Fuction works only in README.yaml exists.
    Function should be run also if no modification to files is wanted.
    
    To fasten the correlation function calculations, it is beneficial
    to reduce the simulation size. Also, in some cases for the
    reasons of saving the space, only subselection of atoms is 
    saved in the trajectory.
    
    Script actions:
        1) script reduces tpr, xtc and also creates reduced gro file
           (this step is skipped if xtc=='Original')
        2) script writes info into README concerning which files
           to be used for correlation function calculations 
    
    
    Inputs: folder_path - contains README.yaml, xtc, tpr (and others)
            save_part   - which part of atoms should be saved to new files
                          only suports options in automatically generated index file
                          (this part should be improved in the future)
            xtc         - True/False/Original
                          True -     xtc, tpr in the folder correspond to each other
                                     and both of there files should be reduced to
                                     save_part
                          False -    xtc in the folder already contains only save_part
                                     whereas tpr contains all the atoms
                                     (happens for example when simulation tpr generated
                                     with compressed-x-grps in mdp file)
                          Original - xtc and tpr in the folder will be used for correlation
                                     function calculations, no new files are created by
                                     this fuction, function only writes info into README.yaml  
    
    """
    
    readme=folder_path+"/README.yaml"
    with open(readme) as yaml_file:
        content = yaml.load(yaml_file, Loader=yaml.FullLoader)
    
    
    if not "FILES_FOR_RELAXATION" in content:
        content["FILES_FOR_RELAXATION"]={}
    
    if xtc=="Original":
        for conversion in ["xtc","tpr","gro"]:
            if not conversion in content["FILES_FOR_RELAXATION"]:
                content["FILES_FOR_RELAXATION"][conversion]={}
            content["FILES_FOR_RELAXATION"][conversion]["NAME"]= content["FILES"][conversion]["NAME"]
            content["FILES_FOR_RELAXATION"][conversion]["SIZE"]= content["FILES"][conversion]["SIZE"]
            content["FILES_FOR_RELAXATION"][conversion]["MODIFIED"]= content["FILES"][conversion]["MODIFIED"]
    
    else:
    
        if not xtc:
            conversions={"xtc":"echo 'converting tpr and gro files'",
                         "tpr":"echo "+save_part+"|gmx convert-tpr -s " + folder_path+"/"+content["FILES"]["tpr"]["NAME"] + " -o " 
                                + folder_path+"/"+save_part+"_"+content["FILES"]["tpr"]["NAME"]+" -extend 10 -n "
                                + folder_path+"/"+content["FILES"]["ndx"]["NAME"] ,
                         "gro":"echo System| gmx trjconv -f " + folder_path+"/"+content["FILES"]["xtc"]["NAME"] + 
                                " -s " + folder_path+"/"+save_part+"_"+content["FILES"]["tpr"]["NAME"] + " -b " + content["EQILIBRATED"] 
                                + " -e " + content["EQILIBRATED"] + " -pbc mol -o " + folder_path+ "/"+save_part+"_" 
                                + content["FILES"]["gro"]["NAME"] }
        else:
            conversions={"xtc":"echo "+save_part+"| gmx trjconv -f " + folder_path+"/"+content["FILES"]["xtc"]["NAME"] + 
                               " -s " + folder_path+"/"+content["FILES"]["tpr"]["NAME"] + " -b " + content["EQILIBRATED"] 
                               + " -o " + folder_path+ "/"+save_part+"_" + content["FILES"]["xtc"]["NAME"] + " >& /dev/null",
                         "tpr":"echo "+save_part+"|gmx convert-tpr -s " + folder_path+"/"+content["FILES"]["tpr"]["NAME"] + " -o "
                                + folder_path+"/"+save_part+"_"+content["FILES"]["tpr"]["NAME"]+
                                " -extend 10 -n "+ folder_path+"/"+content["FILES"]["ndx"]["NAME"] + " >& /dev/null",
                         "gro":"echo System| gmx trjconv -f " + folder_path+"/"+save_part+"_"+content["FILES"]["xtc"]["NAME"] +      
                               " -s " + folder_path+"/"+save_part+"_"+content["FILES"]["tpr"]["NAME"] + " -b " + content["EQILIBRATED"]     
                               + " -e " + content["EQILIBRATED"] + " -pbc mol -o " + folder_path+ "/"
                               +save_part+"_" + content["FILES"]["gro"]["NAME"] + " >& /dev/null"}
    
           # conversions["xtc"]=("echo "+save_part+"| gmx trjconv -f " + folder_path+"/"+content["FILES"]["xtc"]["NAME"] + 
           #                    " -s " + folder_path+"/"+content["FILES"]["tpr"]["NAME"] + " -b " + content["EQILIBRATED"] 
           #                    + " -o " + folder_path+ "/"+save_part+"_" + content["FILES"]["xtc"]["NAME"] )
    
        check_xtc=False
        for conversion in conversions:
            if not conversion in content["FILES_FOR_RELAXATION"]:
                content["FILES_FOR_RELAXATION"][conversion]={}
                os.system(conversions[conversion])
                check_xtc=True
            elif not content["FILES_FOR_RELAXATION"][conversion]["FROM_ORIG"]==content["FILES"][conversion]["MODIFIED"] and conversion=="xtc":
                print("Tuleeko tassa ongelmia?",conversion)
                os.system(conversions[conversion])
                check_xtc=True
            #os.system(conversions[conversion])
        

            if conversion!="xtc" or xtc:
                content["FILES_FOR_RELAXATION"][conversion]["NAME"]=save_part+"_" + content["FILES"][conversion]["NAME"]
            else:
          
                content["FILES_FOR_RELAXATION"][conversion]["NAME"]= content["FILES"][conversion]["NAME"]
                #content["FILES"][conversion]["NAME"] = "none"
                #content["FILES"][conversion]["SIZE"] = "none"
                #content["FILES"][conversion]["MODIFIED"] = "none"
            
            
        
            try:
                file_adress = folder_path+"/"+content["FILES_FOR_RELAXATION"][conversion]["NAME"]
                timepre=os.path.getmtime(file_adress)
                file_mod = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timepre))
                content["FILES_FOR_RELAXATION"][conversion]["SIZE"]=os.path.getsize(file_adress)/1000000
                content["FILES_FOR_RELAXATION"][conversion]["MODIFIED"] = file_mod
                content["FILES_FOR_RELAXATION"][conversion]["FROM_ORIG"] = content["FILES"][conversion]["MODIFIED"]
            except:
                if conversion=="tpr":
                    os.system("echo "+save_part+"|gmx convert-tpr -s " + folder_path+"/"+content["FILES"]["tpr"]["NAME"] + " -o " 
                                + folder_path+"/"+save_part+"_"+content["FILES"]["tpr"]["NAME"]+" -n "
                                + folder_path+"/"+content["FILES"]["ndx"]["NAME"])
                    file_adress = folder_path+"/"+content["FILES_FOR_RELAXATION"][conversion]["NAME"]
                    timepre=os.path.getmtime(file_adress)
                    file_mod = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timepre))
                    content["FILES_FOR_RELAXATION"][conversion]["SIZE"]=os.path.getsize(file_adress)/1000000
                    content["FILES_FOR_RELAXATION"][conversion]["MODIFIED"] = file_mod
                    content["FILES_FOR_RELAXATION"][conversion]["FROM_ORIG"] = content["FILES"][conversion]["MODIFIED"]
                    
                
    
    check_xtc=True
    if check_xtc:
        mol = mda.Universe(folder_path+"/"+content["FILES_FOR_RELAXATION"]["gro"]["NAME"],
                           folder_path+"/"+content["FILES_FOR_RELAXATION"]["xtc"]["NAME"])

        Nframes=len(mol.trajectory)
        timestep = mol.trajectory.dt
        trj_length = Nframes * timestep
        begin_time=mol.trajectory.time

        content["FILES_FOR_RELAXATION"]["xtc"]['SAVING_FREQUENCY'] = timestep
        content["FILES_FOR_RELAXATION"]['xtc']['LENGTH'] = trj_length
        content["FILES_FOR_RELAXATION"]['xtc']['BEGIN'] = begin_time
    
    
    with open(readme, 'w') as f:
        yaml.dump(content,f, sort_keys=False)
        
        
        
#added 5.2.2023        
def what_analysis_done(system):
    """Prints info on different analysis in yaml files"""
    gather_info={}



    for i,analysis in enumerate(system):
        for inf in system[analysis]["info"]:
            try:
                if inf not in gather_info:
                    gather_info[inf]={}
                if system[analysis]["info"][inf] in gather_info[inf]:
                    gather_info[inf][system[analysis]["info"][inf]].append(analysis)
                else:
                    gather_info[inf][system[analysis]["info"][inf]]=[analysis]
            except:
                pass


    for inf in gather_info:
        for mf in gather_info[inf]:
            print(inf[3:],":",mf,", analyzed in: ",gather_info[inf][mf])
        print("*")

#added 5.2.2023
def load_yaml_files(output_path_relax,output_path_timescales):
    """Reads all the yaml files that exist in the folders.
    
    Spin relaxation times yamls are expected to end with: _relax.yaml
    Timescales yamls are expected to end with: _timescales.yaml
    
    This is automatically generated by the script.
    In this way, all the yaml files can be saved in the same directory.
    """
    #reads in spin relaxation times 
    relaxation_yamls={}
    for file in os.listdir(output_path_relax):
        if fnmatch.fnmatch(os.fsdecode(file), "*_relax.yaml"):
            with open(output_path_relax+os.fsdecode(file)) as yaml_file:
                relaxation_yamls[os.fsdecode(file)[:-11]] = yaml.load(yaml_file, Loader=yaml.FullLoader)

    print("Spin relaxation data exist for:\n")
    for system in relaxation_yamls:
        print("   *",system)            

    #reads in timescales
    timescales_yamls={}
    for file in os.listdir(output_path_timescales):
        if fnmatch.fnmatch(os.fsdecode(file), "*_timescales.yaml"):
            with open(output_path_timescales+os.fsdecode(file)) as yaml_file:
                timescales_yamls[os.fsdecode(file)[:-16]] = yaml.load(yaml_file, Loader=yaml.FullLoader)

    print("\n \nTimescales exist for:\n")
    for system in timescales_yamls:
        print("   *",system)
    
    return relaxation_yamls, timescales_yamls
