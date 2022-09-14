# Module pylabfea.data
"""Module pylabfea.data introduces the class ``Data`` for handling of data resulting
from virtual or physical mechanical tests in the pyLabFEA package. This class provides the
methods necessary for analyzing data. During this processing, all information is gathered
that is required to define a material, i.e., all parameters for elasticity, plasticity,
and microstructures are provided from the data. Materials are defined in
module pylabfea.material based on the analyzed data of this module.

uses NumPy, SciPy, MatPlotLib, Pandas

Version: 4.0 (2021-11-27)
Authors: Ronak Shoghi, Alexander Hartmaier, ICAMS/Ruhr University Bochum, Germany
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)"""

import pylabfea as FE
import json
import numpy as np
import matplotlib.pyplot as plt


class Data(object):
    """Define class for handling data from virtual mechanical tests in micromechanical
    simulations and data from physical mechanical tests on materials with various
    microstructures. Data is used to train machine learning flow rules in pyLabFEA.
    """

    def __init__(self, fname, mat_name="Simulanium", epl_crit=2.e-3, d_ep=5.e-4, epl_max=0.03, plot=False):
        self.mat_data = dict()
        self.mat_data['epc'] = epl_crit
        self.mat_data['sdim'] = 6
        self.mat_data['Name'] = mat_name
        self.mat_data['wh_data'] = True
        self.mat_data['Ntext'] = 1
        self.mat_data['tx_name'] = 'Random'
        self.mat_data['texture'] = np.zeros(1)

        raw_data = self.read_data(fname)
        self.parse_data(raw_data, epl_crit=epl_crit, epl_max=epl_max, d_ep=d_ep)  # add data to mat_data
        if plot:
            self.plot_set()

    def key_parser(self, key):
        parameters = key.split('_')
        Keys_Parsed = {"Stress_Type": parameters[0], "Load_Type": parameters[1], "Hash_Load": parameters[2],
                       "Hash_Orientation": parameters[3], "Texture_Type": parameters[5]}
        return Keys_Parsed

    def read_data(self, Data_File):
        data = json.load(open(Data_File))
        Final_Data = dict()

        for key, val in data.items():
            res = val['Results']
            Sigma = [res["S11"], res["S22"], res["S33"], res["S12"], res["S13"], res["S23"]]
            E_Total = [res["E11"], res["E22"], res["E33"], res["E12"], res["E13"], res["E23"]]
            E_Plastic = [res["Ep11"], res["Ep22"], res["Ep33"], res["Ep12"], res["Ep13"], res["Ep23"]]

            Load = val["Initial_Load"]

            Len_Sigma = len(Sigma[0])
            seq_full = np.zeros(Len_Sigma)
            peeq_full = np.zeros(Len_Sigma)
            Original_Stresses = np.zeros((Len_Sigma, 6))
            Original_Plastic_Strains = np.zeros((Len_Sigma, 6))
            Original_Total_Strains = np.zeros((Len_Sigma, 6))
            for i in range(Len_Sigma):
                Stress_6D = np.array([Sigma[0][i], Sigma[1][i], Sigma[2][i], Sigma[3][i], Sigma[4][i], Sigma[5][i]])
                Original_Stresses[i, :] = Stress_6D
                seq_full[i] = FE.sig_eq_j2(Stress_6D)

                E_Plastic_6D = np.array([E_Plastic[0][i], E_Plastic[1][i], E_Plastic[2][i],
                                         E_Plastic[3][i], E_Plastic[4][i], E_Plastic[5][i]])
                Original_Plastic_Strains[i, :] = E_Plastic_6D
                peeq_full[i] = FE.eps_eq(E_Plastic_6D)

                E_Total_6D = np.array([E_Total[0][i], E_Total[1][i], E_Total[2][i],
                                       E_Total[3][i], E_Total[4][i], E_Total[5][i]])
                Original_Total_Strains[i] = E_Total_6D
            Final_Data[key] = {"SEQ": seq_full, "PEEQ": peeq_full, "Load": Load,
                               "Stress": Original_Stresses,
                               "Plastic_Strain": Original_Plastic_Strains, "Total_Strain": Original_Total_Strains}

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
        Nlc = len(db.keys())
        E_av = 0.
        nu_av = 0.
        sy_av = 0.
        peeq_max = 0.
        sig = []
        epl = []
        ct = 0
        for key, val in db.items():
            # estimate yield point for load case
            iel = np.nonzero(val['PEEQ'] < epl_crit)[0]
            ipl = np.nonzero(np.logical_and(val['PEEQ'] >= epl_crit, val['PEEQ'] <= epl_max))[0]
            if len(iel) < 2:
                print(f'Skipping data set {key} (No {ct}): No elastic range: IEL: {iel}, vals: {len(val["PEEQ"])}')
                Nlc -= 1
                continue
            if len(ipl) < 2:
                print(f'Skipping data set {key} (No {ct}): No plastic range: IPL: {ipl}, vals: {len(val["PEEQ"])}; {epl_crit}, {epl_max}')
                Nlc -= 1
                continue
            s0 = val['SEQ'][iel[-1]]
            s1 = val['SEQ'][ipl[0]]
            e0 = val['PEEQ'][iel[-1]]
            e1 = val['PEEQ'][ipl[0]]
            sy = s0 + (epl_crit - e0) * (s1 - s0) / (e1 - e0)
            sy_av += sy / Nlc
            eps = val['PEEQ'][ipl[-1]]
            if eps > peeq_max:
                peeq_max = eps
            for i in ipl:
                '''WARNING: select only values in intervals of d_epl for storing in data set !!!'''
                sig.append(val['Stress'][i])
                epl.append(val['Plastic_Strain'][i])

            # estimate  elastic constants from stresses in range [0.1,0.4]sy
            ''' WARNING: This needs to be improved !!!'''
            ind = np.nonzero(np.logical_and(val['SEQ'] > 0.2 * sy, val['SEQ'] < 0.4 * sy))[0]
            seq = val['SEQ'][ind]
            eeq = FE.eps_eq(val['Total_Strain'][ind])
            E = np.average(seq / eeq)
            nu = 0.3
            E_av += E / Nlc
            nu_av += nu / Nlc

            # get texture name
            ''' Warning: This should be read only once from metadata !!!'''
            Key_Translated = self.key_parser(key)
            self.mat_data['ms_type'] = Key_Translated["Texture_Type"],  # unimodal texture type
            ct += 1

        self.mat_data['flow_stress'] = np.array(sig)
        self.mat_data['plastic_strain'] = np.array(epl)
        self.mat_data['peeq_max'] = peeq_max
        self.mat_data['E_av'] = E_av
        self.mat_data['nu_av'] = nu_av
        self.mat_data['sy_av'] = sy_av
        self.mat_data['Nlc'] = Nlc
        print(f'\n###   Data set: {self.mat_data["Name"]}  ###')
        print(f'Type of microstructure: {Key_Translated["Texture_Type"]}')
        print('Estimated elastic constants: E=%5.2f GPa, nu=%4.2f' % (E_av / 1000, nu_av))
        print('Estimated yield strength: %5.2f MPa at PEEQ = %5.3f' % (sy_av, epl_crit))

    def plot_set(self, db, mat_param):
        fontsize = 18
        cmap = plt.cm.get_cmap('viridis', 10)
        plt.figure(figsize=(18, 7))
        plt.subplots_adjust(wspace=0.2)
        plt.subplot(1, 2, 1)
        for i, key in enumerate(db.keys()):
            col = FE.polar_ang(np.array(db[key]['Load'])) / np.pi
            plt.plot(FE.eps_eq(np.array(db[key]['Main_Strains'])) * 100,
                     FE.seq_J2(np.array(db[key]['Main_Stresses'])), color=cmap(col))
        plt.xlabel(r'$\epsilon_{eq}^\mathrm{tot}$ (%)', fontsize=fontsize)
        plt.ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=fontsize)
        plt.title('Equiv. total strain vs. equiv. J2 stress', fontsize=fontsize)
        plt.tick_params(axis="x", labelsize=fontsize - 4)
        plt.tick_params(axis="y", labelsize=fontsize - 4)

        syc_0 = []
        syc_1 = []
        for i, key in enumerate(db.keys()):
            plt.subplot(1, 2, 2)  # , projection='polar')
            syc = FE.s_cyl(np.array(db[key]["Parsered_Data"]["S_yld"]))
            plt.plot(np.array(db[key]["Parsered_Data"]["S_Cyl"])[db[key]["Parsered_Data"]["Plastic_data"], 1],
                     np.array(db[key]["Parsered_Data"]["S_Cyl"])[db[key]["Parsered_Data"]["Plastic_data"], 0], 'or')
            plt.plot(np.array(db[key]["Parsered_Data"]["S_Cyl"])[db[key]["Parsered_Data"]["Elastic_data"], 1],
                     np.array(db[key]["Parsered_Data"]["S_Cyl"])[db[key]["Parsered_Data"]["Elastic_data"], 0], 'ob')
            syc_0.append(syc[0][0])
            syc_1.append(syc[0][1])

        syc_1, syc_0 = zip(*sorted(zip(syc_1, syc_0)))
        plt.plot(syc_1, syc_0, '-k')
        plt.plot([-np.pi, np.pi], [mat_param['sy_av'], mat_param['sy_av']], '--k')
        plt.legend(['raw data above yield point', 'raw data below yield point',
                    'interpolated yield strength', 'average yield strength'], loc=(1.04, 0.8),
                   fontsize=fontsize - 2)
        plt.title('Raw data ', fontsize=fontsize)
        plt.xlabel(r'$\theta$ (rad)', fontsize=fontsize)
        plt.ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=fontsize)
        plt.tick_params(axis="x", labelsize=fontsize - 4)
        plt.tick_params(axis="y", labelsize=fontsize - 4)
        plt.show()

    def plot_yield_locus(self, db, mat_param, active, scatter=False, data=None,
                         data_label=None, arrow=False, file=None, title=None,
                         fontsize=18):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                               figsize=(15, 8))
        cmap = plt.cm.get_cmap('viridis', 10)
        # ms_max = np.amax(self.mat_param[active])
        Ndat = len(mat_param[active])
        v0 = mat_param[active][0]
        scale = mat_param[active][-1] - v0
        if np.any(np.abs(scale) < 1.e-3):
            scale = 1.

        for i in range(Ndat):
            sc = []
            for set_ind, key in enumerate(db.keys()):
                val = mat_param[active][i]
                hc = (val - v0) * 10
                if active == 'work_hard':
                    sc.append(FE.s_cyl(mat_param['flow_stress'][set_ind, i, 0, :]))
                    label = 'PEEQ: ' + str(val.round(decimals=4))
                    color = cmap(hc)
                elif active == 'texture':
                    sc = FE.s_cyl(mat_param['flow_stress'][i, 0, :, :])
                    ind = np.argsort(sc[:, 1])  # sort dta points w.r.t. polar angle
                    sc = sc[ind]
                    label = mat_param['ms_type']
                    color = (hc, 0, 1 - hc)
                elif active == 'flow_stress':
                    sc = mat_param['flow_stress'][0, 0, :, :]
                    ax.plot(sc[:, 1], sc[:, 0], '.r', label='shear')
                    sc = mat_param['flow_stress'][0, 0, :, :]
                    label = 'Flow Stress'
                    color = 'b'
                else:
                    raise ValueError('Undefined value for active field in "plot_yield_locus"')
            Degree = []
            Radius = []
            for data in sc:
                Degree.append(data[0])
                Radius.append(data[1])
            Radius, Degree = zip(*sorted(zip(Radius, Degree)))

            if scatter:
                ax.plot(Radius, Degree, '.m', label=label)
            if data is not None:
                ax.plot(Radius, Degree, '.r', label=data_label)
            ax.plot(Radius, Degree, label=label, color=color)

        plt.legend(loc=(1.04, 0.7), fontsize=fontsize - 2)
        plt.tick_params(axis="x", labelsize=fontsize - 4)
        plt.tick_params(axis="y", labelsize=fontsize - 4)
        if arrow:
            dr = mat_param['sy_av']
            drh = 0.08 * dr
            plt.arrow(0, 0, 0, dr, head_width=0.05, width=0.004,
                      head_length=drh, color='r', length_includes_head=True)
            plt.text(-0.12, dr * 0.87, r'$\sigma_1$', color='r', fontsize=22)
            plt.arrow(2.0944, 0, 0, dr, head_width=0.05,
                      width=0.004, head_length=drh, color='r', length_includes_head=True)
            plt.text(2.26, dr * 0.92, r'$\sigma_2$', color='r', fontsize=22)
            plt.arrow(-2.0944, 0, 0, dr, head_width=0.05,
                      width=0.004, head_length=drh, color='r', length_includes_head=True)
            plt.text(-2.04, dr * 0.95, r'$\sigma_3$', color='r', fontsize=22)
        if title is not None:
            plt.title(title)
        if file is not None:
            plt.savefig(file + '.pdf', format='pdf', dpi=300)
        plt.show()
