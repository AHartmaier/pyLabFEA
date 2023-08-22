# Module pylabfea.data
"""Module pylabfea.data introduces the class ``Data`` for handling of data resulting
from virtual or physical mechanical tests in the pyLabFEA package. This class provides the
methods necessary for analyzing data. During this processing, all information is gathered
that is required to define a material, i.e., all parameters for elasticity, plasticity,
and microstructures are provided from the data. Materials are defined in
module pylabfea.material based on the analyzed data of this module.

uses NumPy, SciPy, MatPlotLib, Pandas

Version: 4.0 (2021-11-27)
Last Update: (24-04-2023)
Authors: Ronak Shoghi, Alexander Hartmaier, ICAMS/Ruhr University Bochum, Germany
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)"""

import pylabfea as FE
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class Data(object):
    """Define class for handling data from virtual mechanical tests in micromechanical
    simulations and data from physical mechanical tests on materials with various
    microstructures. Data is used to train machine learning flow rules in pyLabFEA.
    """

    def __init__(self, fname, mat_name="Simulanium", epl_crit=1.e-3, d_ep=5.e-4, epl_max=0.03, plot=False, Work_Hardening= True):
        self.mat_data=dict()
        self.mat_data['epc']=epl_crit
        self.mat_data['sdim']=6
        self.mat_data['Name']=mat_name
        self.mat_data['wh_data']=Work_Hardening
        self.mat_data['Ntext']=1
        self.mat_data['tx_name']='Random'
        self.mat_data['texture']=np.zeros(1)

        raw_data=self.read_data(fname)
        self.parse_data(raw_data, epl_crit = epl_crit, epl_max = epl_max, d_ep = d_ep)  # add data to mat_data
        if plot:
            self.plot_training_data()

    def key_parser(self, key):
        parameters=key.split('_')
        Keys_Parsed={"Stress_Type": parameters[0], "Load_Type": parameters[1], "Hash_Load": parameters[2],
                     "Hash_Orientation": parameters[3], "Texture_Type": parameters[4]}
        return Keys_Parsed

    def read_data(self, Data_File):
        data=json.load(open(Data_File))
        Final_Data=dict()

        for key, val in data.items():
            res=val['Results']
            Sigma=[res["S11"], res["S22"], res["S33"], res["S12"], res["S13"], res["S23"]]
            E_Total=[res["E11"], res["E22"], res["E33"], res["E12"], res["E13"], res["E23"]]
            E_Plastic=[res["Ep11"], res["Ep22"], res["Ep33"], res["Ep12"], res["Ep13"], res["Ep23"]]
            Load=val["Initial_Load"]
            Len_Sigma=len(Sigma[0])
            seq_full=np.zeros(Len_Sigma)
            peeq_full=np.zeros(Len_Sigma)
            peeq_plastic=np.zeros(Len_Sigma)
            Original_Stresses=np.zeros((Len_Sigma, 6))
            Original_Plastic_Strains=np.zeros((Len_Sigma, 6))
            Original_Total_Strains=np.zeros((Len_Sigma, 6))
            for i in range(Len_Sigma):
                Stress_6D=np.array([Sigma[0][i], Sigma[1][i], Sigma[2][i], Sigma[3][i], Sigma[4][i], Sigma[5][i]])
                Original_Stresses[i, :]=Stress_6D
                seq_full[i]=FE.sig_eq_j2(Stress_6D)

                E_Plastic_6D=np.array([E_Plastic[0][i], E_Plastic[1][i], E_Plastic[2][i],
                                       E_Plastic[3][i], E_Plastic[4][i], E_Plastic[5][i]])

                peeq_plastic[i]=FE.eps_eq(E_Plastic_6D)
                Original_Plastic_Strains[i, :]=E_Plastic_6D

                E_Total_6D=np.array([E_Total[0][i], E_Total[1][i], E_Total[2][i],
                                     E_Total[3][i], E_Total[4][i], E_Total[5][i]])

                peeq_full[i]=FE.eps_eq(E_Total_6D)
                Original_Total_Strains[i]=E_Total_6D

            Final_Data[key]={"SEQ": seq_full, "PEEQ": peeq_plastic, "TEEQ": peeq_full, "Load": Load,
                             "Stress": Original_Stresses,
                             "Plastic_Strain": Original_Plastic_Strains, "Total_Strain": Original_Total_Strains}

        self.SE_data={}
        for key in Final_Data:
            Stress=[Final_Data[key]["SEQ"]]
            Strain_Total=[Final_Data[key]["TEEQ"]]
            self.SE_data[key]={"Stress": Stress, "Strain": Strain_Total}

        self.SPE_data={}
        for key in Final_Data:
            Stress=[Final_Data[key]["SEQ"]]
            Strain_Plastic=[Final_Data[key]["PEEQ"]]
            self.SPE_data[key]={"Stress": Stress, "Strain": Strain_Plastic}

        self.Data_Visualization = {}
        for key in Final_Data:
            Stress=Final_Data[key]["Stress"]
            EQ_Stress=Final_Data[key]["SEQ"]
            EQ_Strain_Plastic=Final_Data[key]["PEEQ"]
            epl = []
            for i in range(len(EQ_Strain_Plastic)):
                # CHANGES: A Shift in the data in order to have 0 equivalant plastic strain at start of yielding
                temp_epl=(Final_Data[key]['Plastic_Strain'][i]) * (1 - (0.002 / (FE.eps_eq(Final_Data[key]['Total_Strain'][i]))))
                epl.append(temp_epl)
            Strain_Plastic= np.array(epl)
            #Strain_Plastic = [Final_Data[key]["Plastic_Strain"]]

            self.Data_Visualization[key]={"Stress": Stress, "Eq_Stress": EQ_Stress, "Eq_Strain": EQ_Strain_Plastic, "Strain": Strain_Plastic}

        return Final_Data

    def parse_data(self, db, epl_crit, epl_max, d_ep):
        """
        Estimate elastic properties and initial yield strength from data for each load case and form averages.

        Parameters
        ----------
        epl_max
        db : dict
            Database
        epl_crit : float
            Critical value for onset of yielding
        d_ep : float
            Intended interval between plastic strains

        Returns
        -------

        """
        Nlc=len(db.keys())
        E_av=0.
        nu_av=0.
        sy_av=0.
        peeq_max=0.
        sy_list=[]
        sig=[]
        epl=[]
        ept=[]
        ct=0
        sig_ideal = []
        for key, val in db.items():
            # estimate yield point for ideal plasticity consideration
            i_ideal = np.nonzero(val['PEEQ'] > 2.e-3)[0]
            if len(i_ideal) < 2:
                print(
                    f'Skipping data set {key} (No {ct}): No plastic range for ideal plasticity')
                Nlc-=1
                continue
            sig_ideal.append(val['Stress'][i_ideal[0]])

            # estimate yield point for load case
            iel=np.nonzero(val['PEEQ'] < epl_crit)[0]
            ipl=np.nonzero(np.logical_and(val['PEEQ'] >= epl_crit, val['PEEQ'] <= epl_max))[0]
            if len(iel) < 2:
                print(f'Skipping data set {key} (No {ct}): No elastic range: IEL: {iel}, vals: {len(val["PEEQ"])}')
                Nlc-=1
                continue
            if len(ipl) < 2:
                print(
                    f'Skipping data set {key} (No {ct}): No plastic range: IPL: {ipl}, vals: {len(val["PEEQ"])}; {epl_crit}, {epl_max}')
                Nlc-=1
                continue
            s0=val['SEQ'][iel[-1]]
            s1=val['SEQ'][ipl[0]]
            e0=val['PEEQ'][iel[-1]]
            e1=val['PEEQ'][ipl[0]]
            sy=s0 + (epl_crit - e0) * (s1 - s0) / (e1 - e0)
            sy_list.append(sy)
            sy_av+=sy / Nlc
            eps=val['PEEQ'][ipl[-1]]
            if eps > peeq_max:
                peeq_max=eps

            # for i in ipl[::]:
            for i in ipl:
                '''WARNING: select only values in intervals of d_epl for storing in data set !!!'''
                # CHANGES: A Shift in the data in order to have 0 equivalant plastic strain at start of yielding.
                sig.append(val['Stress'][i])
                temp_epl=(val['Plastic_Strain'][i]) * (1 - (0.002 / (FE.eps_eq(val['Total_Strain'][i]))))
                epl.append(temp_epl)
            ''' WARNING: This needs to be improved !!!'''
            ind=np.nonzero(np.logical_and(val['SEQ'] > 0.2 * sy, val['SEQ'] < 0.4 * sy))[0]
            seq=val['SEQ'][ind]
            eeq=FE.eps_eq(val['Total_Strain'][ind])
            E=np.average(seq / eeq)
            nu=0.3
            E_av+=E / Nlc
            nu_av+=nu / Nlc

            # get texture name
            ''' Warning: This should be read only once from metadata !!!'''
            Key_Translated=self.key_parser(key)
            self.mat_data['ms_type']=Key_Translated["Texture_Type"],  # unimodal texture type
            ct+=1

        self.mat_data['flow_stress']=np.array(sig)
        self.mat_data['plastic_strain']=np.array(epl)
        self.mat_data['peeq_max']=peeq_max
        self.mat_data['E_av']=E_av
        self.mat_data['nu_av']=nu_av
        self.mat_data['sy_av']=sy_av
        self.mat_data['Nlc']=Nlc
        self.mat_data['sy_list']=sy_list
        self.mat_data['sig_ideal'] = np.array(sig_ideal)
        print(f'\n###   Data set: {self.mat_data["Name"]}  ###')
        print(f'Type of microstructure: {Key_Translated["Texture_Type"]}')
        print('Estimated elastic constants: E=%5.2f GPa, nu=%4.2f' % (E_av / 1000, nu_av))
        print('Estimated yield strength: %5.2f MPa at PEEQ = %5.3f' % (sy_av, epl_crit))

    def plot_training_data(self):
        # Function for plotting stress strain curves from the database using total strain and plastic strain values.
        for data, xlabel, ylabel in [(self.SE_data, "Total Strain", "Stress"),
                                     (self.SPE_data, "Plastic Strain", "Stress")]:
            self.plot_data(data, xlabel, ylabel)

    def plot_data(self, data, xlabel, ylabel):
        for key in data:
            plt.scatter(data[key]["Strain"], data[key]["Stress"], s = 1)
            plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
            plt.xlabel(xlabel, fontsize = 14)
            plt.ylabel(ylabel, fontsize = 14)
        plt.show()

    def plot_set(self, db, mat_param):
        fontsize=18
        cmap=plt.cm.get_cmap('viridis', 10)
        plt.figure(figsize = (18, 7))
        plt.subplots_adjust(wspace = 0.2)
        plt.subplot(1, 2, 1)
        for i, key in enumerate(db.keys()):
            col=FE.polar_ang(np.array(db[key]['Load'])) / np.pi
            plt.plot(FE.eps_eq(np.array(db[key]['Main_Strains'])) * 100,
                     FE.seq_J2(np.array(db[key]['Main_Stresses'])), color = cmap(col))
        plt.xlabel(r'$\epsilon_{eq}^\mathrm{tot}$ (%)', fontsize = fontsize)
        plt.ylabel(r'$\sigma_{eq}$ (MPa)', fontsize = fontsize)
        plt.title('Equiv. total strain vs. equiv. J2 stress', fontsize = fontsize)
        plt.tick_params(axis = "x", labelsize = fontsize - 4)
        plt.tick_params(axis = "y", labelsize = fontsize - 4)

        syc_0=[]
        syc_1=[]
        for i, key in enumerate(db.keys()):
            plt.subplot(1, 2, 2)  # , projection='polar')
            syc=FE.s_cyl(np.array(db[key]["Parsered_Data"]["S_yld"]))
            plt.plot(np.array(db[key]["Parsered_Data"]["S_Cyl"])[db[key]["Parsered_Data"]["Plastic_data"], 1],
                     np.array(db[key]["Parsered_Data"]["S_Cyl"])[db[key]["Parsered_Data"]["Plastic_data"], 0], 'or')
            plt.plot(np.array(db[key]["Parsered_Data"]["S_Cyl"])[db[key]["Parsered_Data"]["Elastic_data"], 1],
                     np.array(db[key]["Parsered_Data"]["S_Cyl"])[db[key]["Parsered_Data"]["Elastic_data"], 0], 'ob')
            syc_0.append(syc[0][0])
            syc_1.append(syc[0][1])

        syc_1, syc_0=zip(*sorted(zip(syc_1, syc_0)))
        plt.plot(syc_1, syc_0, '-k')
        plt.plot([-np.pi, np.pi], [mat_param['sy_av'], mat_param['sy_av']], '--k')
        plt.legend(['raw data above yield point', 'raw data below yield point',
                    'interpolated yield strength', 'average yield strength'], loc = (1.04, 0.8),
                   fontsize = fontsize - 2)
        plt.title('Raw data ', fontsize = fontsize)
        plt.xlabel(r'$\theta$ (rad)', fontsize = fontsize)
        plt.ylabel(r'$\sigma_{eq}$ (MPa)', fontsize = fontsize)
        plt.tick_params(axis = "x", labelsize = fontsize - 4)
        plt.tick_params(axis = "y", labelsize = fontsize - 4)
        plt.show()

    # Check
    def plot_yield_locus(self, db, mat_data, active, scatter=False, data=None,
                         data_label=None, arrow=False, file=None, title=None,
                         fontsize=18):
        fig, ax=plt.subplots(subplot_kw = {'projection': 'polar'},
                             figsize = (15, 8))
        cmap=plt.cm.get_cmap('viridis', 10)
        # ms_max = np.amax(self.mat_param[active])
        Ndat=len(mat_data[active])
        v0=mat_data[active][0]
        scale=mat_data[active][-1] - v0
        if np.any(np.abs(scale) < 1.e-3):
            scale=1.
        sc=[]
        ppe=[]
        scy=[]
        for i in range(Ndat):
            if active == 'flow_stress':
                sc.append(FE.s_cyl(mat_data['flow_stress'][i]))
                ppe.append(FE.eps_eq(mat_data['plastic_strain'][i]))
                for j in range(len(ppe)):
                    if ppe[j] < 0.003:
                        scy.append(sc[j])
        label='Flow Stress'
        color='b'
        scy=np.array(scy)
        scy=np.array(scy)
        scy=np.array(scy)
        ax.scatter(scy[:, 1], scy[:, 0], marker = ".")
        plt.show()

