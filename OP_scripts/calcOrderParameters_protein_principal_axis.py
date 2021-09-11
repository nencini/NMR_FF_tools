#!/usr/bin/env python
"""
 calculation of the angle for the order parameters
 from a MD trajectory
 useful for example for lipid bilayers
 meant for use with NMRlipids projects
------------------------------------------------------------
 Made by J.Melcr,  Last edit 2017/03/21
 Last edit by R. Nencini 2018/10/20
 edit by R.N 2021/02/09 to be suitable for protein analysis
 edit by R.N. 2021/03/09 to calculte OP with respect to 1st PA
 last edit: 2021/09/11
------------------------------------------------------------
 input: Order parameter definitions
        gro and xtc file (or equivalents)
 output: order parameters (3 textfiles)
--------------------------------------------------------
"""

# coding: utf-8

import MDAnalysis as mda
import numpy as np
import scipy.stats
import math
import os  #, sys
from optparse import OptionParser
import pandas as pd


bond_len_max=3.5  # in Angstroms, max distance between atoms for reasonable OP calculation (PBC and sanity check)
bond_len_max_sq=bond_len_max**2


class OrderParameter:
    """
    Class for storing&manipulating
    order parameter (OP) related metadata (definition, name, ...)
    and OP trajectories
    and methods to evaluate OPs.
    OP definition consist of:
       - name of the OP
       - residue name
       - involved atoms (exactly 2)
       + extra: mean, std.dev. & err. estimation (from Bayesian statistics)
                of the OP (when reading-in an already calculated result)
    """
    def __init__(self, name, chain, resID, resname, atom_A_name, atom_B_name, atom_A_id, atom_B_id, *args):
        """
        Initialization of an instance of this class.
        it doesn't matter which atom comes first,
        atom A or B, for OP calculation.
        """
        self.name = name             # name of the order parameter, a label
        self.chain = chain
        self.resID= resID            # residue id of Amino acid
        self.resname = resname       # name of residue atoms are in
        self.atAname = atom_A_name
        self.atBname = atom_B_name
        self.atAid = atom_A_id
        self.atBid = atom_B_id
        # variables for Bayesian statistics
        self.mean = None
        self.var = None
        self.confidence_int = None
        for field in self.__dict__:
            if not isinstance(field, str):
                raise UserWarning, "provided name >> {} << is not a string! \n \
                Unexpected behaviour might occur.".format(field)
            else:
                if not field.strip():
                    raise RuntimeError, "provided name >> {} << is empty! \n \
                    Cannot use empty names for atoms and OP definitions.".format(field)
        # extra optional arguments allow setting avg,std,errest values -- suitable for reading-in results of this script
        if len(args) == 0:
            self.avg = None     # average/mean value
            self.std = None     # standard deviation of the mean
            self.errest = None  # error estimate of the mean
        elif len(args) == 2:
            self.avg = args[0]
            self.std = args[1]
            self.errest = None
        elif len(args) == 3:
            self.avg = args[0]
            self.std = args[1]
            self.errest = args[2]
        else:
            raise UserWarning, "Number of optional positional arguments is {len}, not 3, 2 or 0. Args: {args}\nWrong file format?".format(len=len(args), args=args)
        self.traj = []  # for storing OPs


    def calc_OP(self, atoms,lipids):
        """
        calculates Order Parameter according to equation
        S = 1/2 * (3*cos(theta)^2 -1)
	
        done with respect to the first principal axis
        modyfied on the 3rd of Jun
        """
        #lipids= mol.select_atoms("resname POPC or resname DHPC or resname POPE or resname POPA")
        principal_axis1=lipids.principal_axes()[0]
        vec = atoms[1].position - atoms[0].position
        d_vec_2 = np.square(vec).sum()
        d_principal_2 = np.square(principal_axis1).sum()
        dot_vec_princ=np.dot(vec,principal_axis1)
        
        if d_vec_2>bond_len_max_sq:
            raise UserWarning, "Atomic distance for atoms \
 {at1} and {at2} in residue no. {resnr} is suspiciously \
 long: {d}!\nPBC removed???".format(at1=atoms[0].name, at2=atoms[1].name, resnr=atoms[0].resid, d=math.sqrt(d_vec_2))
 

        cos2 = dot_vec_princ**2/d_vec_2/d_principal_2
        S = 0.5*(3.0*cos2-1.0)
        return S


    def calc_angle(self, atoms, z_dim=45.0):
        """
        calculates the angle between the vector and z-axis in degrees
        no PBC check!
        assuming a sim-box-centred membrane --> it's centre ~ z_dim/2
        """
        vec = atoms[1].position - atoms[0].position
        d = math.sqrt(np.square(vec).sum())
        cos = vec[2]/d
        # values for the bottom leaflet are inverted so that 
        # they have the same nomenclature as the top leaflet
        cos *= math.copysign(1.0, atoms[0].position[2]-z_dim*0.5)
        try:
            angle = math.degrees(math.acos(cos))
        except ValueError:
            if abs(cos)>=1.0:
                print "Cosine is too large = {} --> truncating it to +/-1.0".format(cos)
                cos = math.copysign(1.0, cos)
                angle = math.degrees(math.acos(cos))
        return angle


    def get_avg_std_OP(self, alpha_confidence=0.95):
        """
        Provides average, variance and stddev of all OPs in self.traj
        """
        # convert to numpy array
        return scipy.stats.bayes_mvs(self.traj, alpha=alpha_confidence)


def read_trajs_calc_OPs(ordPars, top, trajs):
    """
    procedure that
    creates MDAnalysis (mda) Universe instance with topology top,
    reads in trajectories trajs and then
    goes through every frame and
    evaluates each Order Parameter "S" from the list of OPs ordPars.
    ordPars : list of OrderParameter class instances
       each item in this list describes an Order parameter to be calculated in the trajectory
    top : str
        filename of a top file (e.g. conf.gro)
    trajs : list of strings
        filenames of trajectories
    """
    # read-in topology and trajectory
    mol = mda.Universe(top, trajs)

    # make atom selections for each OP and store it as its attribute for later use with trajectory
    for op in ordPars.values():
        # selection = pairs of atoms, split-by residues
        if opts.selection=="average":
            """select exactly 2 atoms based on the atom IDs; 
            the atom.split splits into a list of <AtomGroup with 2 atoms>"""
            selection = mol.select_atoms(	"resname {rnm} and name {atA} {atB} and resid {rid}".format(
                                     rnm=op.resname, atA=op.atAname, atB=op.atBname, rid=op.resID)
                                    ).atoms.split("residue")

        if opts.selection=="split":
            """select exactly 2 atoms based on the atom IDs; 
            the atom.split is posibly not needed
            creates <AtomGroup with 2 atoms>"""
            selection = mol.select_atoms(	"bynum {atAid} {atBid}".format(
                                     atAid=op.atAid, atBid=op.atBid)
                                    ).atoms.split("residue")  
        
        
        for res in selection:

            # check if we have only 2 atoms (A & B) selected
            if res.n_atoms != 2:
                print res.resnames, res.resids
                for atom in res.atoms:
                    print atom.name, atom.id
                raise UserWarning, "Selection >> name {atA} {atB} << \
                contains {nat} atoms, but should contain exactly 2!".format(
                atA=op.atAname, atB=op.atBname, nat=res.n_atoms)
        op.selection = selection

    # go through trajectory frame-by-frame
    # and calculate each OP from the list of OPs
    # for each residue separately
    save_cumulative=0
    if opts.cumulative=="yes":
        for op in ordPars.values():
            with open(opts.out_fname+"/"+opts.out_fname+"_"+op.name+".cumulative","w") as f:            
                f.write(" ")	
    for frame in mol.trajectory:
        lipids= mol.select_atoms(opts.lipid_selection)
        save_cumulative=save_cumulative+1
        for op in ordPars.values():
            for residue in op.selection:
                if "vec" in op.name:
                    S = op.calc_angle(residue, z_dim=frame.dimensions[2])
                else:
                    S = op.calc_OP(residue,lipids)
                op.traj.append(S)
            if opts.cumulative=="yes" and save_cumulative%opts.save_cum==0 and save_cumulative>2:
                (op.mean, op.var, op.std) = op.get_avg_std_OP(alpha_confidence=alpha_confidence)
                op.avg = op.mean[0]
                op.confidence_int = op.mean[1]
                op.errest = max(abs(op.confidence_int-op.avg)) 
                with open(opts.out_fname+"/"+opts.out_fname+"_"+op.name+".cumulative","a") as f:            
                    f.write( "{:10.0f} {: 2.5f} \n".format(mol.trajectory.time, op.avg))
               


def parse_op_input(fname):
    """
    parses input file with Order Parameter definitions
    file format is as follows:
    OP_name    resname    atom1    atom2  +extra: OP_mean  OP_std
    (flexible cols)
    fname : string
        input file name
    returns : dictionary
        with OrderParameters class instances
    """
    ordPars = {}
    try:
        with open(fname,"r") as f:
            for line in f.readlines():
                if not line.startswith("#"):
                    items = line.split()
                    ordPars[items[0]] = OrderParameter(*items)
    except:
        raise RuntimeError, "Couldn't read input file >> {inpf} <<".format(inpf=fname)
    return ordPars



#%%

if __name__ == "__main__":
    # help message is automatically provided
    # type=string, action=store is default
    parser = OptionParser()
    parser.add_option('-i', '--inp',  dest='inp_fname',  help='input (OP definitions) file name, default OPs_definition.def, generated by this script. If you want to use your own OP definitions, the file has to have a different name and be specified under the flag "-i" \n \n The format is following \n OP name, chain, resid, resname, atom1, atom2, atom1 id, atom2 id \n \n For average option chain and atom ids are irrelevant but must be filled in', default="OPs_definition.def")
    parser.add_option('-t', '--top',  dest='top_fname',  help='topology (gro!!!) file name, default output.gro', default="output.gro")
    parser.add_option('-x', '--traj', dest='traj_fname', help='beginning of trajectory (xtc, dcd) files names , default traj.xtc.', default="traj.xtc")
    parser.add_option('-a', '--alpha',  dest='alpha',    help='confidence interval probability, default 0.95', default=0.95)
    parser.add_option('-o', '--out',  dest='out_fname',  help='output (OPs mean&std) file name, also used for folder name and cumulative averages, default OrderParameters', default="OrderParameters")
    parser.add_option('-l', '--lenght',  dest='mol_size',  help='number of atoms inside one chain, default 3502', default=3502)
    parser.add_option('-1', '--atm1',  dest='atom1',  help='atom 1 used in the OP definition, default "N"', default="N")
    parser.add_option('-2', '--atm2',  dest='atom2',  help='atom 2 used in the OP definition, default "HN"', default="HN")
    parser.add_option('-s', '--sel',  dest='selection',  help='average over chains or split? ("average"/"split"), default "split"', default="split")
    parser.add_option('-c', '--cul',  dest='cumulative',  help='should the cumulative average be calculated along the way ("yes"/"no"), default "yes"', default="yes")
    parser.add_option('-f', '--freq',  dest='save_cum',  help='frequency with which the cumulative average is saved, default 1 - every frame', default=1)
    parser.add_option('-b', '--beg',  dest='begin_protein',  help='atomID of the first protein atom, default 1', default=1)
    parser.add_option('-p', '--lipid',  dest='lipid_selection',  help='lipid composition for the PA bicelle orientation, default "resname DMPC" ', default="resname DMPC")
    opts, args = parser.parse_args()

    #size of one chain
    lenght = int(opts.mol_size) 
    begin_protein = int(opts.begin_protein)-1

    #assign names of the atoms for OP calculation
    atom1_name = opts.atom1
    atom2_name = opts.atom2




    #####################################################################################################

    #sanity warning 
    print "\n \n ################################# \n PLEASE READ \n #################################"
    print "\n \n This code is meant to calculate order parameters of protein back bone \n with respect to the first principal axis of specified lipid selection. \n \n In principle, it could work for other bonds in peptides but it was not tested. \n The user have to enter the number of atoms of his/her protein and the atom id of the first protein atom. \n The code then assumes that the whole protein is consecutive in the *.gro file, \n starting with atom id that user enterd. \n  ################################# \n !!! If this condition is not met the results do not make any sence. !!! \n ################################# \n \n At the moment, the 'average' option for calculating OPs works only in the case that the given protein reidue \n has always the same resid in all the chains in the provided topology *.gro file. \n \n Please execute with --help flag to see more information \n"
    confirm = raw_input("\n \n ################################# \n Are you sure you understood and want to continue? \n ################################# \n yes/no \n")
    if confirm=="yes":	
        #create the OH definition file
        ordPars = []
        with open(opts.top_fname,"r") as f:
            print "\n Creating Headgroup_Glycerol_OPs.def ..."
            for line in f.readlines():
                if not line.startswith("#"):

                    #reads the gro file
                    #by the try function it should select only the data with actual coordinate 
                    #because otherwise the assignment to integer and others fails
                    try:
                        resid= int(line[0]+line[1]+line[2]+line[3]+line[4])
                        resname = line[5]+line[6]+line[7]+line[8]
                        resname = resname.split()[0]
                        atom = line[9]+line[10]+line[11]+line[12]+line[13]+line[14]
                        atom = atom.split()[0]
                        atomid = int (line[15]+line[16]+line[17]+line[18]+line[19])
                        ordPars.append([resid, resname, atom, atomid])
      
            
                    except: 
                        print " "
        i=0
        try:
            os.remove("Headgroup_Glycerol_OPs.def")
        except:
            print "File has been already removed, everything's OK\n" 
        while i<len(ordPars)-2:
            atom1=0
            atom2=0
            resid= ordPars[i][0]
            resname = ordPars[i][1]
            while i<len(ordPars) and ordPars[i][0]==resid and ordPars[i][1]==resname :
                if ordPars[i][2]== atom1_name:
                    atom1=ordPars[i]
                if ordPars[i][2]== atom2_name:
                    atom2=ordPars[i]
                i+=1
            if atom1!=0 and atom2!=0:
            
                chain=int(math.ceil((atom1[3]-begin_protein)/lenght))

           
                if opts.selection=="split":    
                    with open("Headgroup_Glycerol_OPs.def","a") as f:
                        f.write( "{:8s} {:> 5d} {:> 5d} {:5s} {:5s} {:5s} {:> 5d} {:> 5d} \n".format(str(atom1[1])+str(atom1[0])+"_"+str(chain), chain, atom1[0], atom1[1], atom1[2], atom2[2], atom1[3], atom2[3]))
          
                if opts.selection=="average":
                    if chain==1:    
                        with open("Headgroup_Glycerol_OPs.def","a") as f:
                            f.write( "{:8s} {:> 5d} {:> 5d} {:5s} {:5s} {:5s} {:> 5d} {:> 5d} \n".format(str(atom1[1])+str(atom1[0])+"_"+str(chain), chain, atom1[0], atom1[1], atom1[2], atom2[2], atom1[3], atom2[3]))
          
    
    
    
    
    
        ####################################################################3
    
        try:
            os.mkdir("final_data")
        except:
            print "Folder final_data already exists, everything ok\n"
        try:
            os.mkdir(opts.out_fname)
        except:
            print "Folder already exists, everything is OK\n"


        # desired confience interval in Bayesian statistics
        alpha_confidence = opts.alpha

        # dictionary for storing of OrderParameter class instances (name-wise, of course)
        print "\nReading OP definitions ...\n"
        ordPars = parse_op_input(opts.inp_fname)
   
    

        # get all parts of trajectories
        trajs = []
        for file_name in os.listdir(os.getcwd()):
            if file_name.startswith(opts.traj_fname):
                trajs.append(file_name)

        # read trajectory and calculate all OPs
        print "Reading trajectories and calculating OPs ...\n"
        read_trajs_calc_OPs(ordPars, opts.top_fname, trajs)


        print "OP Name     mean    std    err.est.   confidence_interval {:2.0f}% (min, max)".format(alpha_confidence*100.0)
        print "--------------------------------------------------------------------"
        for op in ordPars.values():
            (op.mean, op.var, op.std) = op.get_avg_std_OP(alpha_confidence=alpha_confidence)
            op.avg = op.mean[0]
            op.confidence_int = op.mean[1]
            op.errest = max(abs(op.confidence_int-op.avg))
            print "{:10s} {: 2.4f} {: 2.4f} {: 2.4f}   {}".format(op.name, op.avg, op.std[0], op.errest, op.confidence_int)
        print "--------------------------------------------------------------------"


        try:
            with open("final_data/"+opts.out_fname+".dat","w") as f:
                f.write("# OP_name    chain    resid    resname    atom1    atom2    OP_mean   OP_stddev   OP_err.est. {:2.0f}% \n\
#--------------------------------------------------------------------------------------------\n".format(alpha_confidence*100.0) )
                for op in ordPars.values():
                    f.write( "{:>9s} {:>8s} {:>8s} {:>10s} {:>8s} {:>8s} {: 10.5f} {: 11.5f} {: 17.5f} \n".format(
                         op.name, op.chain, op.resID, op.resname, op.atAname, op.atBname,
                         op.avg, op.std[0], op.errest)
                       )
            print "\nOrderParameters written to >> {fname} <<".format(fname=opts.out_fname)
        except:
            print "ERROR: Problems writing main output file."


        #reorders the final output according to chain and residue number
        reorder= pd.DataFrame([], columns=list('ABCDEFGHI'))
        with open("final_data/"+opts.out_fname+".dat","r") as f:
            for line in f.readlines():
                if not line.startswith("#"):
                    reorder=reorder.append(pd.DataFrame([line.split()], columns=list('ABCDEFGHI')))

        reorder=reorder.astype({'I': 'float','G': 'float', 'H': 'float', 'B': 'int',  'C': 'int'})
        reorder=reorder.sort_values(by=['B', 'C'])
        if opts.selection=="average":
            reorder=reorder.replace({'B':1},"average")
        reorder=reorder.values.tolist()

        if opts.selection=="split":
            with open("final_data/"+opts.out_fname+".dat","w") as f:
                f.write("# OP_name    chain    resid    resname    atom1    atom2    OP_mean   OP_stddev   OP_err.est. % \n\
#--------------------------------------------------------------------------------------------\n")
                for i in range(0,len(reorder)):
                    f.write( "{:>9s} {:>8d} {:>8d} {:>10s} {:>8s} {:>8s} {: 10.5f} {: 11.5f} {: 17.5f} \n".format(
                    reorder[i][0], reorder[i][1], reorder[i][2], reorder[i][3], reorder[i][4], reorder[i][5],
                    reorder[i][6], reorder[i][7], reorder[i][8])
                    )
            
        if opts.selection=="average":
            with open("final_data/"+opts.out_fname+".dat","w") as f:
                f.write("# OP_name    chain    resid    resname    atom1    atom2    OP_mean   OP_stddev   OP_err.est. % \n\
#--------------------------------------------------------------------------------------------\n")
                for i in range(0,len(reorder)):
                    f.write( "{:>9s} {:>8s} {:>8d} {:>10s} {:>8s} {:>8s} {: 10.5f} {: 11.5f} {: 17.5f} \n".format(
                    reorder[i][0], reorder[i][1], reorder[i][2], reorder[i][3], reorder[i][4], reorder[i][5],
                    reorder[i][6], reorder[i][7], reorder[i][8])
                    )
    
    else:
        print "Don't worry. Take a deep brath. Stretch your body. Go for a run. And you can try again :-)."    
