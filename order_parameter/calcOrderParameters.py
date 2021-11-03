#!/usr/bin/env python
"""
 calculation of the angle for the order parameters
 from a MD trajectory
 useful for example for lipid bilayers
 meant for use with NMRlipids projects
------------------------------------------------------------
 Made by J.Melcr,  Last edit 2017/03/21
 Last edit by R. Nencini 2018/10/20
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
    def __init__(self, name, resname, atom_A_name, atom_B_name, *args):
        """
        Initialization of an instance of this class.
        it doesn't matter which atom comes first,
        atom A or B, for OP calculation.
        """
        self.name = name             # name of the order parameter, a label
        self.resname = resname       # name of residue atoms are in
        self.atAname = atom_A_name
        self.atBname = atom_B_name
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


    def calc_OP(self, atoms):
        """
        calculates Order Parameter according to equation
        S = 1/2 * (3*cos(theta)^2 -1)
        """
        vec = atoms[1].position - atoms[0].position
        d2 = np.square(vec).sum()
        if d2>bond_len_max_sq:
            raise UserWarning, "Atomic distance for atoms \
 {at1} and {at2} in residue no. {resnr} is suspiciously \
 long: {d}!\nPBC removed???".format(at1=atoms[0].name, at2=atoms[1].name, resnr=atoms[0].resid, d=math.sqrt(d2))
        if extra_align=="X":
            cos2 = vec[0]**2/d2
        if extra_align=="Y":
            cos2 = vec[1]**2/d2
        if extra_align=="Z":
            cos2 = vec[2]**2/d2
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
        if extra_select=="resid":
            selection = mol.select_atoms("resid {rnm} and name {atA} {atB}".format(
                                        rnm=op.resname, atA=op.atAname, atB=op.atBname)
                                        ).atoms.split("residue")
        if extra_select=="resname":
            selection = mol.select_atoms("resname {rnm} and name {atA} {atB}".format(
                                        rnm=op.resname, atA=op.atAname, atB=op.atBname)
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
    moje=0
    with open(opts.out_fname+".cumulative","w") as f:            
        f.write(" ")	
    for frame in mol.trajectory:
        moje=moje+1
        for op in ordPars.values():
            for residue in op.selection:
                if "vec" in op.name:
                    S = op.calc_angle(residue, z_dim=frame.dimensions[2])
                else:
                    S = op.calc_OP(residue)
                op.traj.append(S)
            if moje%100==0:
                (op.mean, op.var, op.std) = op.get_avg_std_OP(alpha_confidence=alpha_confidence)
                op.avg = op.mean[0]
                op.confidence_int = op.mean[1]
                op.errest = max(abs(op.confidence_int-op.avg)) 
                with open(opts.out_fname+".cumulative","a") as f:            
                    f.write( "{:10s} {: 2.5f} \n".format(op.name, op.avg))
               


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
    parser.add_option('-i', '--inp',  dest='inp_fname',  help='input (OP definitions) file name', default="Headgroup_Glycerol_OPs.def")
    parser.add_option('-t', '--top',  dest='top_fname',  help='topology (gro, pdb) file name', default="start_450mM.gro")
    parser.add_option('-x', '--traj', dest='traj_fname', help='beginning of trajectory (xtc, dcd) files names (will use all files beginning with this string).', default="converged_450mM")
    parser.add_option('-a', '--alpha',  dest='alpha',    help='confidence interval probability', default=0.95)
    parser.add_option('-o', '--out',  dest='out_fname',  help='output (OPs mean&std) file name', default="OPs_450mM.dat")
    parser.add_option('-l', '--align',  dest='align',  help='with respect to which axis is OP calculated, default Z', default="Z")
    parser.add_option('-s', '--sel',  dest='selection',  help='OP definitions uses resname (for lipid) or resid (for proteins), default resname', default="resname")

    opts, args = parser.parse_args()

    # desired confience interval in Bayesian statistics
    alpha_confidence = opts.alpha


    # OP is calculated with respect to which axis
    extra_align = opts.align
    extra_select = opts.selection


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
        with open(opts.out_fname,"w") as f:
            f.write("# OP_name    resname    atom1    atom2    OP_mean   OP_stddev   OP_err.est. {:2.0f}% \n\
#--------------------------------------------------------------------------------------------\n".format(alpha_confidence*100.0) )
            for op in ordPars.values():
                f.write( "{:10s} {:7s} {:5s} {:5s} {: 2.5f} {: 2.5f} {: 2.5f} \n".format(
                         op.name, op.resname, op.atAname, op.atBname,
                         op.avg, op.std[0], op.errest)
                       )
        print "\nOrderParameters written to >> {fname} <<".format(fname=opts.out_fname)
    except:
        print "ERROR: Problems writing main output file."


    # this single-line format may become soon deprecated, but
    # it is the format that is used in NMRlipids projects for processing through awk+gnuplot
    try:
        conc_formatted_line = "conc  {b1: 2.6f} {b1e: 2.6f}  {b2: 2.6f} {b2e: 2.6f}    {a1: 2.6f} {a1e: 2.6f}  {a2: 2.6f} {a2e: 2.6f}".format(
                              b1=ordPars['beta1'].avg, b2=ordPars['beta2'].avg,
                              a1=ordPars['alpha1'].avg, a2=ordPars['alpha2'].avg,
                              b1e=ordPars['beta1'].errest, b2e=ordPars['beta2'].errest,
                              a1e=ordPars['alpha1'].errest, a2e=ordPars['alpha2'].errest)
        print
        print "Single line format:\nconc  beta1 err  beta2 err  alpha1 err  alpha2 err"
        print conc_formatted_line
        
    except:
	print "ERROR: Problems writing the beta-alpha single line format file."
