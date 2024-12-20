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
import os.path

import pylabfea as FE
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import interpolate
from scipy.signal import savgol_filter


class Data(object):
    """Define class for handling data from virtual mechanical tests in micromechanical
    simulations and data from physical mechanical tests on materials with various
    microstructures. Data is used to train machine learning flow rules in pyLabFEA.

    Parameters
    ----------
    source: str or data array
        Either filename of datafile or array of initial yield stresses
    path_data: str
        Trunc of pathname for data files (optional, default: './')
    name: str
        Name of Dataset (optional, default: 'Dataset')
    mat_name : str
        Name of material (optional, default: 'Simulanium)
    sdim: int
        Dimensionality of stresses; if sdim = 3 only principal stresses are considered (optional, default: 6)
    epl_crit : float
        Critical plastic strain at which yield strength is defined(optional, default: 2.e-3)
    epl_start : float
        Start value of equiv. plastic strain at which data will be sampled(optional, default: 1.e-3)
    epl_max : float
        Maximum equiv. plastic strain up to which data is considered (optional, default: 0.05)
    plot : Boolean
        Plot data (optional, default: False)
    wh_data : Boolean
        Data for flow stresses and plastic strain tensors in work hardening regime exists (optional, default: True)
    mode : str
        Intermediate variable to control the operation mode between Ronak's version (='RS') and Jan's (='JS')
        Currently affects key_parser to decode the bc_keys and read_data where JS uses 32 for instead of 23 for the
        tensor components.

    Attributes
    ----------
    name: str
        Name of dataset
    sy_av: float
        Average initial yield stress
    E_av: float
        Average Young's modulus
    nu_av: float
        Average Poisson ratio
    mat_data: dictionary
        | Contains available data for ML yield function and for microstructural parameters
        | Items
        |    epc : critical value of equiv. plastic strain that defines the onset of plastic yielding
        |    ep_start : start value of equiv. plastic strain at which data is acquired
        |    ep_max : maximum value of equiv. plastic strain up to which data is considered
        |    lc_indices : array with start index for each load case
        |    peeq_max : maximum of equiv. plastic strain that occurred in data set, corrected for onset of plasticity at epc
        |    sdim : dimension of stress vector (must be 3 or 6)
        |    Name : material name
        |    Dataset : Name of dataset
        |    wh_data : indicate if strain hardening data exists
        |    Ntext : number of textures
        |    tx_name : name of texture
        |    tx_index : texture index
        |    texture : texture parameters
        |    flow_stress : array of flow stresses correlated to plastic strains
        |    plastic_strain : array of plastic strains corrected for onset of plasticity at epc
        |    E_av : avereage Young's modulus derived from data
        |    nu_av : average Poisson's ratio derived from data
        |    sy_av : average yield strength, i.e. flow stress at epc, obtained from data
        |    Nlc : number of load cases in data
        |    sy_list : ???
        |    sig_ideal : interpolated stress tensors at onset of plastic yielding (epc) for each load case

    """
    def __init__(self, source, path_data='./',
                 name='Dataset', mat_name="Simulanium",
                 sdim=6,
                 epl_crit=2.e-3,
                 epl_start=1.e-3, epl_max=0.03,
                 plot=False,
                 wh_data=True,
                 tx_data=False,
                 tx_descriptor='GSH_3',
                 mode='RS',
                 red_wh_data=False):
        if sdim!=3 and sdim!=6:
            raise ValueError('Value of sdim must be either 3 or 6')
        self.mat_data = dict()
        self.mat_data['epc'] = epl_crit
        self.mat_data['ep_start'] = epl_start
        self.mat_data['ep_max'] = epl_max
        self.mat_data['sdim'] = sdim
        self.mat_data['tdim'] = 0
        self.mat_data['Name'] = mat_name
        self.mat_data['Dataset'] = name
        self.mat_data['wh_data'] = wh_data
        self.mat_data['tx_data'] = tx_data  # JS: tx_data is a flag that tells if texture data should be used NOT if its present.
        self.mat_data['Ntext'] = 1
        self.mat_data['tx_name'] = 'Random'
        self.mat_data['tx_index'] = 0
        self.mat_data['texture'] = np.zeros(1)
        self.mat_data['tx_descriptor'] = tx_descriptor
        self.mat_data['tx_key'] = None
        self.mode = mode
        self.red_wh_data = red_wh_data

        if type(source) is str:
            # raw_data = self.read_data(path_data+source) # JS : this is not very robust better:
            raw_data = self.read_data(os.path.join(path_data, source))
            self.parse_data(raw_data, epl_crit, epl_start, epl_max)  # add data to mat_data
        else:
            raw_data = np.array(source)
            self.convert_data(raw_data)
        if plot:
            self.plot_training_data()

    def key_parser(self, key):
        # JS: Modified to teture version
        parameters = key.split('_')
        if self.mode == 'RS':
            Keys_Parsed = {"Stress_Type": parameters[0], "Load_Type": parameters[1], "Hash_Load": parameters[2],
                           "Hash_Orientation": parameters[3], "Texture_Type": parameters[4]}

        elif self.mode == 'JS':
            Keys_Parsed = {"Stress_Type": parameters[0], "Load_Type": parameters[1], "Hash_Load": parameters[2],
                           "Hash_Orientation": parameters[5], "Texture_Type": parameters[7], "N_Grains": parameters[3],
                           "Elements_Grain": parameters[4]}
        else:
            raise KeyError(f"Mode is: {self.mode}. Must be RS or JS")
        return Keys_Parsed

    def read_data(self, Data_File):
        # JS TODO: Store CYL data correctly
        data = json.load(open(Data_File))
        Final_Data = dict()

        for key, val in data.items():
            if key == 'Texture':
                self.mat_data['tx_name'] = val['name']
                try:
                    self.mat_data['tx_index'] = val['texture_index']
                except KeyError:
                    print("No texture_index found in this Data_Base.json -> Assign default value of 0")
                if not self.mat_data['tx_data']:
                    warnings.warn(f"WARNING: tx_data was set to false. I will just include qualitative texture info.")
                else:
                    if 'GSH' in self.mat_data['tx_descriptor']:
                        # JS: Use GSH coefficients
                        gsh_dim = int(self.mat_data['tx_descriptor'].split('_')[-1])
                        if gsh_dim in [3, 7, 12, 37]:
                            self.mat_data['texture'] = np.array(val['gsh_coeff_reconstructed_random'])[1:1 + gsh_dim]
                        else:
                            raise ValueError(f"GSH with {gsh_dim} not valid. Must be 3, 7, 12, ")
                    elif 'ADV' in self.mat_data['tx_descriptor']:
                        # JS: Use addressvector
                        adv_dim = int(self.mat_data['tx_descriptor'].split('_')[-1])
                        self.mat_data['texture'] = np.array(val[f'address_vector_{adv_dim}'])
                    elif self.mat_data['tx_descriptor'] == 'VF':
                        # JS: Use old descriptor baesd on volume fraction
                        raise NotImplementedError
                    self.mat_data['tdim'] = len(self.mat_data['texture'])
            else:
                res = val['Results']

                if 'cyl' in key:
                    # JS: The dict fields with 'cyl' in their key only have 'Stress' sig_ideal = db_dict[key][Results]
                    Final_Data[key] = {"Stress": res}
                else:
                    if self.mode == 'JS':
                        Sigma = [res["S11"], res["S22"], res["S33"], res["S12"], res["S13"], res["S32"]]
                        E_Total = [res["E11"], res["E22"], res["E33"], res["E12"], res["E13"], res["E32"]]
                        E_Plastic = [res["Ep11"], res["Ep22"], res["Ep33"], res["Ep12"], res["Ep13"], res["Ep32"]]
                    else:
                        Sigma = [res["S11"], res["S22"], res["S33"], res["S12"], res["S13"], res["S23"]]
                        E_Total = [res["E11"], res["E22"], res["E33"], res["E12"], res["E13"], res["E23"]]
                        E_Plastic = [res["Ep11"], res["Ep22"], res["Ep33"], res["Ep12"], res["Ep13"], res["Ep23"]]
                    # Load = val["Initial_Load"]
                    Len_Sigma = len(Sigma[0])
                    seq_full = np.zeros(Len_Sigma)
                    peeq_full = np.zeros(Len_Sigma)
                    peeq_plastic = np.zeros(Len_Sigma)
                    Original_Stresses = np.zeros((Len_Sigma, 6))
                    Original_Plastic_Strains = np.zeros((Len_Sigma, 6))
                    Original_Total_Strains = np.zeros((Len_Sigma, 6))
                    for i in range(Len_Sigma):
                        Stress_6D = np.array([Sigma[0][i], Sigma[1][i], Sigma[2][i], Sigma[3][i], Sigma[4][i], Sigma[5][i]])
                        Original_Stresses[i, :] = Stress_6D
                        seq_full[i] = FE.sig_eq_j2(Stress_6D)

                        E_Plastic_6D = np.array([E_Plastic[0][i], E_Plastic[1][i], E_Plastic[2][i],
                                                 E_Plastic[3][i], E_Plastic[4][i], E_Plastic[5][i]])

                        peeq_plastic[i] = FE.eps_eq(E_Plastic_6D)
                        Original_Plastic_Strains[i, :] = E_Plastic_6D

                        E_Total_6D = np.array([E_Total[0][i], E_Total[1][i], E_Total[2][i],
                                               E_Total[3][i], E_Total[4][i], E_Total[5][i]])

                        peeq_full[i] = FE.eps_eq(E_Total_6D)
                        Original_Total_Strains[i] = E_Total_6D

                    Final_Data[key] = {"SEQ": seq_full, "PEEQ": peeq_plastic, "TEEQ": peeq_full, #"Load": Load,
                                       "Stress": Original_Stresses,
                                       "Plastic_Strain": Original_Plastic_Strains, "Total_Strain": Original_Total_Strains}

        self.SE_data = {}
        for key in Final_Data:
            if 'cyl' in key:
                continue
            else:
                Stress = [Final_Data[key]["SEQ"]]
                Strain_Total = [Final_Data[key]["TEEQ"]]
                self.SE_data[key] = {"Stress": Stress, "Strain": Strain_Total}

        self.SPE_data = {}
        for key in Final_Data:
            if 'cyl' in key:
                continue
            else:
                Stress = [Final_Data[key]["SEQ"]]
                Strain_Plastic = [Final_Data[key]["PEEQ"]]
                self.SPE_data[key] = {"Stress": Stress, "Strain": Strain_Plastic}

        #For having also the elastic data and shift the 0 plastic strain to 0.02% to match the micromechanical data. Can be used in Stress-Strain Reconstruction.
        self.Data_Visualization = {}
        for key, dat in Final_Data.items():
            if 'cyl' not in key:
                Stress = dat["Stress"]
                EQ_Stress = dat["SEQ"]
                EQ_Strain_Plastic = dat["PEEQ"]
                epl = []
                for i in range(len(EQ_Strain_Plastic)):
                    # CHANGES: A Shift in the data in order to have 0 equivalent plastic strain at start of yielding
                    scale = np.maximum(EQ_Strain_Plastic[i], 1.e-10)
                    temp_epl = dat['Plastic_Strain'][i] * (1. - self.mat_data['epc'] / scale)
                    epl.append(temp_epl)
                Strain_Plastic = np.array(epl)
                Eq_Shifted_Strain = FE.eps_eq(Strain_Plastic)

                self.Data_Visualization[key] = {"Stress": Stress, "Eq_Stress": EQ_Stress, "Eq_Strain": EQ_Strain_Plastic,
                                                "Strain": Strain_Plastic, "Eq_Shifted_Strain": Eq_Shifted_Strain}

        return Final_Data

    def parse_data(self, db, epl_crit, epl_start, epl_max):
        """
        Read data and store in attribute 'mat_data'
        Estimate elastic properties and initial yield strength from data for each load case and form averages.

        Parameters
        ----------
        epl_max
        db : dict
            Database
        epl_crit : float
            Critical value for onset of yielding
        epl_start : float
            Start value of equic. plastic strain at which data acquisition will start
        epl_max : float
            Maximum equiv. strain up to which data is considered
        """
        def find_transition_index(stress, strain):
            """Calculates the index at which a significant transition in the total stress-strain relationship occurs.
            The function applies a Savitzky-Golay filter to smooth the stress data and then calculates the first
            and second derivatives of the smoothed stress with respect to strain. It identifies the transition index
            by finding the maximum absolute value in the second derivative, signaling a notable transition in the
            total stress-strain curve. This approach relies on the overall behavior of the stress-strain relationship
            rather than focusing on a specific plastic strain threshold (e.g., 0.002%).
            The function includes a conditional check to refine the identified transition index, correcting potential
            errors from anomalies in the second derivative calculations. This is achieved by adjusting the index based
            on a threshold condition related to a strain value, aiming for a more accurate determination of the
            transition point based on total stress-strain values.

            Parameters:
            ----------
            stress: list
                Array of stress values.
            strain: list
                Array of corresponding strain values.

            Returns:
            ----------
            transition_index: (int)
            The index within the stress or strain array where the significant transition occurs.
            """
            stress = np.array(stress)
            strain = np.array(strain)
            stress = stress.flatten()
            strain = strain.flatten()
            smoothed_stress = savgol_filter(stress, window_length = 30, polyorder = 3) # JS : Reduced windod from 51
            derivative_smoothed = np.gradient(smoothed_stress, strain)
            second_derivative = np.gradient(derivative_smoothed, strain)
            transition_index = np.argmax(np.abs(second_derivative))
            while val["TEEQ"][transition_index] > 0.0003:
                 transition_index = transition_index - 5
            return transition_index

        Nlc = len(db.keys()) # JS: includes cyl keys
        Ncyl = 0 # JS: counts cyl keys
        E_av = 0.
        nu_av = 0.
        sy_av = 0.
        peeq_max = 0.
        sy_list = []
        sig = []
        epl = []
        ct = 0
        sig_ideal = []
        lc_ind_list = np.zeros(Nlc + 1, dtype=int)
        elstrains = []
        elstress = []
        for key, val in db.items():
            if 'cyl' in key:
                # JS: CYL dicts contain only stress tensor at yield onset -> only append sig_ideal
                Ncyl += 1
                ct += 1
                sig_ideal.append(val['Stress'])
            else:
                # estimate yield point for ideal plasticity consideration
                # JS:   This might be a little inaccurate as it takes the stress state closest to epl_crit.
                #       For different loading directions, the yield onset is then not consistent
                i_ideal = np.nonzero(val['PEEQ'] > epl_crit)[0]
                if len(i_ideal) < 2:
                    print(
                        f'Skipping data set {key} (No {ct}): No plastic range for ideal plasticity')
                    Nlc -= 1
                    continue
                sig_ideal.append(val['Stress'][i_ideal[0]])

                # estimate yield point for load case
                iel = np.nonzero(val['PEEQ'] < epl_start)[0]
                ipl = np.nonzero(np.logical_and(val['PEEQ'] >= epl_start, val['PEEQ'] <= epl_max))[0]
                if len(iel) < 2:
                    print(f'Skipping data set {key} (No {ct}): No elastic range: IEL: {iel}, vals: {len(val["PEEQ"])}')
                    Nlc -= 1
                    continue
                if len(ipl) < 2:
                    print(
                        f'Skipping data set {key} (No {ct}): No plastic range: IPL: {ipl}, vals: {len(val["PEEQ"])}; {epl_start}, {epl_max}')
                    Nlc -= 1
                    continue
                s0 = val['SEQ'][iel[-1]]
                s1 = val['SEQ'][ipl[0]]
                e0 = val['PEEQ'][iel[-1]]
                e1 = val['PEEQ'][ipl[0]]
                s_start = s0 + (epl_start - e0) * (s1 - s0) / (e1 - e0)  # JS: here we interpolate linear between
                sy_list.append(s_start)
                sy_av += FE.seq_J2(val['Stress'][i_ideal[0]])
                eps = val['PEEQ'][ipl[-1]]
                if eps > peeq_max:
                    peeq_max = eps

                # for i in ipl[::]:
                if self.red_wh_data:
                    idx = [int(np.quantile(ipl, q)) for q in [0, 0.5, 1]]  # JS: Take only three values for wh data
                    print(len(idx))
                else:
                    idx = ipl
                for i in idx:
                    '''WARNING: select only values in intervals of d_epl for storing in data set !!!'''
                    # CHANGES: A Shift in the data in order to have 0 equivalent plastic strain at start of yielding.
                    sig.append(val['Stress'][i])
                    temp_epl = val['Plastic_Strain'][i] * (1. - epl_crit / FE.eps_eq(val['Plastic_Strain'][i]))
                    epl.append(temp_epl)
                nonzero_idcs = np.nonzero(lc_ind_list)
                if nonzero_idcs[0].size == 0:
                    idx_last = 0
                else:
                    idx_last = np.max(nonzero_idcs)
                lc_ind_list[ct] = len(idx) + lc_ind_list[idx_last] # JS : Changed ipl to idx, and [ct - 1] to np.max()

                ''' WARNING: This needs to be improved !!!''' # JS : This is just for calculating av. properties
                ind = np.nonzero(np.logical_and(val['SEQ'] > 0.2 * s_start, val['SEQ'] < 0.4 * s_start))[0]
                seq = val['SEQ'][ind]
                eeq = FE.eps_eq(val['Total_Strain'][ind])
                E = np.average(seq / eeq)
                nu = 0.3
                E_av += E
                nu_av += nu

                # get texture name
                ''' Warning: This should be read only once from metadata !!!'''
                Key_Translated = self.key_parser(key)
                self.mat_data['ms_type'] = Key_Translated["Texture_Type"]  # unimodal texture type
                if self.mat_data['tx_key'] and not self.mat_data['tx_key'] == Key_Translated["Hash_Orientation"]:
                    raise ValueError("Texture Keys differ in one Data Object!!")
                else:
                    self.mat_data['tx_key'] = Key_Translated["Hash_Orientation"]
                ct += 1
                it_stress = val["SEQ"]
                it_strain = val["TEEQ"]
                try:
                    it = find_transition_index(it_stress, it_strain)
                    elstrains.append(val['Total_Strain'][it])
                    elstress.append(val['Stress'][it])
                except ValueError:
                    print(f'Only {len(val["SEQ"])} points in SEQ --> reduce window_length manually!')
        E_av /= (Nlc-Ncyl)  # JS: subtract the number of CYL load cases as they contain no info on strains ...
        nu_av /= (Nlc-Ncyl)
        sy_av /= (Nlc-Ncyl)
        self.mat_data['flow_stress'] = np.array(sig)  # JS: Contain no CYL Data
        self.mat_data['plastic_strain'] = np.array(epl)
        self.mat_data['lc_indices'] = lc_ind_list
        self.mat_data['peeq_max'] = peeq_max - epl_crit
        self.mat_data['E_av'] = E_av
        self.mat_data['nu_av'] = nu_av
        self.mat_data['sy_av'] = sy_av
        self.mat_data['Nlc'] = Nlc  # JS: Contains also cyl load cases
        self.mat_data['Ncyl'] = Ncyl
        self.mat_data['sy_list'] = sy_list
        self.mat_data['sig_ideal'] = np.array(sig_ideal)  # JS: All my data is here if I use ideal plasticity
        self.mat_data['elstress'] = elstress
        self.mat_data['elstrains'] = elstrains
        print(f'\n###   Data set: {self.mat_data["Name"]}  ###')
        print(f'Type of microstructure: {Key_Translated["Texture_Type"]}')
        print('Estimated elastic constants: E=%5.2f GPa, nu=%4.2f' % (E_av / 1000, nu_av))
        print('Estimated yield strength: %5.2f MPa at PEEQ = %5.3f' % (sy_av, epl_start))

    def convert_data(self, sig):
        print("inside where it shoulld not be")
        """
        Convert data provided only for stress tensor at yield point into mat_data dictionary

        Parameters
        ----------
        sig : ndarray
            Stress tensors at yield onset
        """
        Nlc = len(sig)
        sdim = len(sig[0,:])
        if sdim != self.mat_data['sdim']:
            warnings.warn(
                'Warning: dimension of stress in data does not agree with parameter sdim. Use value from data.')
        self.mat_data['sig_ideal'] = sig
        self.mat_data['wh_data'] = False
        lc_ind_list = np.linspace(0, Nlc)
        self.mat_data['lc_indices'] = np.append(lc_ind_list, 0.)
        self.mat_data['E_av'] = 151220.  # WARNING: only valid for example
        self.mat_data['nu_av'] = 0.3  # WARNING: only valid for example
        self.mat_data['sy_av'] = np.mean(FE.sig_eq_j2(sig))
        self.mat_data['peeq_max'] = self.mat_data['epc']
        self.mat_data['Nlc'] = Nlc
        print(f'\n###   Data set: {self.mat_data["Name"]}  ###')
        print(f'Converted data for {Nlc} stress tensors at yield onset into material data.')

    def plot_training_data(self):
        # Function for plotting stress strain curves from the database using total strain and plastic strain values.
        for data, xlabel, ylabel in [(self.SE_data, "Total Strain", "Stress"),
                                     (self.SPE_data, "Plastic Strain", "Stress")]:
            self.plot_data(data, xlabel, ylabel)

    def plot_data(self, data, xlabel, ylabel):
        for key in data:
            plt.scatter(data[key]["Strain"], data[key]["Stress"], s=1)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
        plt.show()

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

    # Check
    def plot_yield_locus(self, db, mat_data, active, scatter=False, data=None,
                         data_label=None, arrow=False, file=None, title=None,
                         fontsize=18):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                               figsize=(15, 8))
        cmap = plt.cm.get_cmap('viridis', 10)
        # ms_max = np.amax(self.mat_param[active])
        Ndat = len(mat_data[active])
        v0 = mat_data[active][0]
        scale = mat_data[active][-1] - v0
        if np.any(np.abs(scale) < 1.e-3):
            scale = 1.
        sc = []
        ppe = []
        scy = []
        for i in range(Ndat):
            if active == 'flow_stress':
                sc.append(FE.s_cyl(mat_data['flow_stress'][i]))
                ppe.append(FE.eps_eq(mat_data['plastic_strain'][i]))
                for j in range(len(ppe)):
                    if ppe[j] < 0.003:
                        scy.append(sc[j])
        label = 'Flow Stress'
        color = 'b'
        scy = np.array(scy)
        scy = np.array(scy)
        scy = np.array(scy)
        ax.scatter(scy[:, 1], scy[:, 0], marker=".")
        plt.show()

    # def intersect(self, s_eq, e_eq, plot=True):
    #     """
    #     Calculates the elastic line from hom. stress and equiv. strain and determines intersection with stress-strain curve.
    #     Parameters
    #     ----------
    #     s_eq : array-like
    #         Array of equivalent stresses
    #     e_eq : array-like
    #         Array of equivalent total strains
    #     plot : bool, default=False
    #         whether the stress-strain curve + intersection should be plotted
    #
    #     Returns
    #     -------
    #     x : double
    #         value of equivalent strain at which elastic line intersects stress strain curve
    #     y : double
    #         value of equivalent stress at which elastic line intersects stress strain curve
    #     top : int
    #         index in s_eq and e_eq that is just above the intersection point
    #     bot : int
    #         index in s_eq and e_eq that is just below the intersection point
    #     """
    #     # Set up intersecting line with m at 0.002 strain
    #     m_1 = 0.0
    #     for j in range(10, 21, 5):
    #         m_1 += s_eq[j] / e_eq[j]
    #     m_1 /= 3
    #     c_1 = s_eq[15] - m_1 * (e_eq[15] + 0.002)
    #
    #     # find intersection interval
    #     s_eq = np.array(s_eq)
    #     e_eq = np.array(e_eq)
    #     s_l = m_1 * e_eq + c_1
    #
    #     diff = s_eq - s_l
    #     try:
    #         top = np.where(diff < 0)[0][0]
    #     except IndexError:
    #         print(
    #             "Plastic strain is too low. The line with slope E=%6.2f MPa is not intersecting the stress-strain curve.\n"
    #             "To solve this issue, increase load and rerun simulation." % m_1)
    #         raise
    #     bot = top - 1
    #
    #     # define line in this interval
    #     m_2 = (s_eq[top] - s_eq[bot]) / (e_eq[top] - e_eq[bot])
    #     c_2 = s_eq[top] - m_2 * e_eq[top]
    #
    #     # find intersetion point
    #     x = (c_2 - c_1) / (m_1 - m_2)
    #     y = m_1 * x + c_1
    #
    #     if plot:
    #         fig = plt.figure(figsize=(10, 6), dpi=200)
    #         ax = fig.add_subplot()
    #         ax.plot(e_eq, s_eq, label='CPFEM', marker='x')
    #         ax.plot(e_eq, s_l, marker='.', color='k')
    #         ax.plot(x, y, marker='x', color='r')
    #         ax.legend(loc="lower right")
    #         ax.set_title('Stress-Strain Curve', pad=20)
    #         ax.set_ylim(0, np.max(s_eq))
    #         plt.xlabel('$E_{vM}$')
    #         plt.ylabel('$S_{vM}$ [MPa]')
    #
    #     return x, y, top, bot
    #
    #
    # def calc_yield_point(self, result_file, plot=False):
    #     """
    #     Function that calculates the homogenized yield onset at 0.2 % equivalent plastic strain from a result dict object.
    #     Parameters
    #     ----------
    #     result_file : dict
    #         Dictionary containing the homogenized CP results
    #     plot : bool, default=False
    #         whether the stress-strain curve + intersection should be plotted
    #
    #     Returns
    #     -------
    #     syld : array-like
    #         homogenized yield onset at 0.2 % plastic strain
    #     """
    #     # Assumes a the 'Results' dict of Data_Base.json
    #     s_eq = result_file['S']
    #     e_eq = result_file['E']
    #
    #     # TODO: JS: change away from pd dataframe
    #     stresses = pd.DataFrame(result_file, columns=['S11', 'S22', 'S33', 'S32', 'S13', 'S12'])
    #
    #     # Interpolate between equivalent stresses and scale up
    #     try:
    #         x, y, top, bot = self.intersect(s_eq, e_eq, plot=plot)
    #     except IndexError:
    #         raise
    #     if s_eq[top] - s_eq[bot] <= 1e-5:
    #         print('WARNING: stress difference too small for interpolation of {res}'.format(res=result_file))
    #         print(y-s_eq[bot])
    #         s_yld = stresses.iloc[bot].values
    #         # print(stresses.iloc[bot].values)
    #         # print('openp seq: {}'.format(s_eq[bot]))
    #         # print('pylab seq: {}'.format(FE.seq_J2(stresses.iloc[bot].values)))
    #     else:
    #         s_yld = stresses.iloc[bot].values + (stresses.iloc[top].values - stresses.iloc[bot].values) * (y - s_eq[bot]) \
    #                 / (s_eq[top] - s_eq[bot])
    #     return s_yld
