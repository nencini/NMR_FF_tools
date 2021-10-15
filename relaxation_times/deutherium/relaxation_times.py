import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


class GetRelaxationData():
    def __init__(self,OP,smallest_corr_time, biggest_corr_time, N_exp_to_fit,analyze,magnetic_field,input_data):
        self.OP=OP
        self.smallest_corr_time=smallest_corr_time
        self.biggest_corr_time=biggest_corr_time
        self.N_exp_to_fit=N_exp_to_fit
        self.magnetic_field=magnetic_field
        self.input_data=input_data
        
        self.org_corrF, self.times_out=self.read_data()
        
        #analyze only the part specified by the user
        analyze_until = round(len(self.org_corrF)*analyze)
        self.org_corrF=self.org_corrF[0:analyze_until]
        self.times_out=self.times_out[0:analyze_until]
        
        Teff, tau_eff_area, R1, R2, rec_corrF = self.calc_relax_time()
        self.plot_fit(rec_corrF)
        print("R1: {} R2: {}".format(R1, R2))


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
            parts = line.split()
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

        # Constants for calculating R1

        wc = 2 * np.pi * 125.76 * 10 ** 6;
        wh = wc / 0.25;
        omega = 2 * np.pi * 6.536 * 10 ** 6 * self.magnetic_field; #Larmour frequency =deuterium gyromagtic ratio * magnatic field

        # changin the unit of time permanently
        Ctimes = Ctimes * 0.001 * 10 ** (-9);

        J0 = 0
        J1 = 0
        J2 = 0
        Jw1 = 0

        for i in range(0, m):
            w=0

            J0 = J0 + 2 * Coeffs[i] * Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

            w = omega
            J1 = J1 + 2 * Coeffs[i] * Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

            w = 2* omega
            J2 = J2 + 2 * Coeffs[i] * Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

        # R1=(2.1*10**9)*(J0+3*J1+6*J2)
        # note! R1's are additive. Nh from the Ferreira2015 paper correctly omitted here
        R1 = (167000  * np.pi) ** 3 / 40.0 * (1 - self.OP ** 2) * (0 * J0 + 2 * J1 + 8 * J2)
        R2 = (167000  * np.pi) ** 3 / 40.0 * (1 - self.OP ** 2) * (3 * J0 + 5 * J1 + 2 * J2)
    
        #get the reconstucted correlation function
        rec_corrF=Cexp_mat.dot(Coeffs)

        return Teff, tau_eff_area, R1, R2, rec_corrF


    def plot_fit(self, reconstruction):
        plt.figure(figsize=(15, 6))
        plt.rcParams.update({'font.size': 20})
        #plt.rcParams.update({'font.weight': "normal"})

        plt.plot(self.times_out,self.org_corrF,label="Original")
        plt.plot(self.times_out,reconstruction,label="Fit")
        plt.xlabel("Time [ps]")
        plt.ylabel("Autocorrelation function")
        plt.title(self.input_data)
        plt.legend()
        plt.show()
