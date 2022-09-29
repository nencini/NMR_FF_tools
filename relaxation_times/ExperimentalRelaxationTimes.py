import numpy as np
import gc
import matplotlib.pyplot as plt
import math
from datetime import date
today = date.today()

gc.collect()



def t1_t2_relaxations(data_from_DynamicCenter,field,coeff,outputP,author,info,increments):
    with open(data_from_DynamicCenter,"r") as f:
        read_data=False
        peak_list=False
        peak=0
        results=False
        for line in f:
            if "integrals" in line and len(line.split())==2:
                read_data=True
            elif "results" in line and len(line.split())==2:
                results=True
                peak_dictionary={}
                peak_dictionary["peaks"]={}


            if "integral errors" in line:
                read_data=False

            if read_data:
                if "Mixing time [s]:" in line:
                    mixing_times=np.array(line.split()[3:])
                    mixing_times = mixing_times.astype('float64')
                elif "Peak name" in line or "SECTION:" in line:
                    pass
                else:
                    if peak==0:
                        intensities=np.array(line.split())
                        intensities = [intensities.astype('float64')]
                        peaks_intensities=intensities
                        line_length=len(line.split())
                    elif line_length==len(line.split()):
                        intensities=np.array(line.split())
                        intensities = [intensities.astype('float64')]
                        peaks_intensities=np.append(peaks_intensities,intensities,axis=0)
                    peak+=1
            elif results:
                if "Peak name" in line or "SECTION:" in line:
                    pass
                else:
                    peak_dictionary["peaks"][line.split()[0]]={}
                    peak_dictionary["peaks"][line.split()[0]]["ppm"]=[line.split()[1],line.split()[2]]

    for peak in range(0,peaks_intensities.shape[0]):
        peak_dictionary["peaks"][str(int(peaks_intensities[peak,0]))]["experiment"]=peaks_intensities[peak,1:]
            
        for_LS=np.array([[0, 0]])
        for i in range(1,peaks_intensities.shape[1]):
            if i<peaks_intensities.shape[1]-1:
                if peaks_intensities[peak,i]<peaks_intensities[peak,i+1]:
                    pass
                elif peaks_intensities[peak,i]==0:
                    break
                else:
                    for_LS=np.append(for_LS,[[mixing_times[i-1],np.log(peaks_intensities[peak,i])]],axis=0)
                       
            else:
                if peaks_intensities[peak,i]==0:
                    break
                else:
                    for_LS=np.append(for_LS,[[mixing_times[i-1],np.log(peaks_intensities[peak,i])]],axis=0)
        for_LS=np.delete(for_LS,0,0)
        curve_fit_coef = np.polyfit(for_LS[:,0], for_LS[:,1], 1, full=True)
            
        
    
        peak_dictionary["peaks"][str(int(peaks_intensities[peak,0]))]["fit"]=curve_fit_coef
        peak_dictionary["peaks"][str(int(peaks_intensities[peak,0]))]["points_used"]=len(for_LS)
    peak_dictionary["mixing_times"]=mixing_times
    peak_dictionary["INFO"]={}
    peak_dictionary["INFO"]["ANALYZED"]=today
    peak_dictionary["INFO"]["AUTHOR"]=author
    peak_dictionary["INFO"]["FIELD"]=field
    peak_dictionary["INFO"]["COEFFICIENTS"]=coeff
    peak_dictionary["INFO"]["OUTPUT_POINTS"]=outputP
    peak_dictionary["INFO"]["INFO"]=info
    
        
    return peak_dictionary, mixing_times
            

def print_results(*files):
    for file in files:
        relax_data=t1_t2_relaxations(file)
        
        curve_fit_x=np.arange(relax_data[1][0],relax_data[1][-1],0.01)
        for peak in relax_data[0]:
            curve_fit_y=np.exp(curve_fit_x*  relax_data[0][peak]["fit"][0][0])*np.exp(relax_data[0][peak]["fit"][0][1])
                
                
def compare_spectra(*files):
    most_peaks=0
    for i,file in enumerate(files):
        if len(file)>most_peaks:
            most_peaks=len(file)
            position=i
            
    reference=files[position]
    other_files=[]
    
    for i,file in enumerate(files):
        if not i==position:
            other_files.append(file)
    
    files=other_files
    
    relaxation_times={}
    for ref_peak in reference:
        relaxation_times[ref_peak]={}
        relaxation_times[ref_peak]["REFERENCE"]={}
        relaxation_times[ref_peak]["REFERENCE"]["ppm"]=reference[ref_peak]["ppm"]
        relaxation_times[ref_peak]["REFERENCE"]["T1"]=  -1/reference[ref_peak]["fit"][0][0]
        #relaxation_times[ref_peak]["REFERENCE"]["error"]=  -1/reference[ref_peak]["fit"][2][0]
        
    for k,file in enumerate(files):
        distances=np.zeros([len(reference),len(file)])
        for i,ref_peak in enumerate(reference):
            #print(float(reference[ref_peak]["ppm"][0]))
            for j,file_peak in enumerate(file):
                distances[i,j]=(math.dist([float(reference[ref_peak]["ppm"][0]), 
                                           float(reference[ref_peak]["ppm"][1])],
                                           [float(file[file_peak]["ppm"][0]),
                                            float(file[file_peak]["ppm"][1])
                                           ])) 
        
        
        if len(reference)>len(file) or len(reference)==len(file):
            reference_keys=list(reference)
            for i,file_peak in enumerate(file):
                key=np.where(distances[:,i]==min(distances[:,i]))[0][0]
                relaxation_times[reference_keys[key]]["file"+str(k)]={}
                relaxation_times[reference_keys[key]]["file"+str(k)]["ppm"]=file[file_peak]["ppm"]
                relaxation_times[reference_keys[key]]["file"+str(k)]["T1"]=-1/file[file_peak]["fit"][0][0]
        if len(reference)<len(file):  
            file_keys=list(file)
            for i,ref_peak in enumerate(reference):
                key=np.where(distances[i,:]==min(distances[i,:]))[0][0]
                relaxation_times[ref_peak]["file"+str(k)]={}
                relaxation_times[ref_peak]["file"+str(k)]["ppm"]=file[file_keys[key]]["ppm"]
                relaxation_times[ref_peak]["file"+str(k)]["T1"]=-1/file[file_keys[key]]["fit"][0][0]
            print("Linear prediction is the god")
            print(k)
            print("Length of reference: {}, length of file: {}".format(len(reference),len(file)))
        
        #print(distances)
    
    return relaxation_times
