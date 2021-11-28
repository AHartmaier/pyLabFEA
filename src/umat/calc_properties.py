''' Python wrapper to perform Abaqus calculation of stress-strain curves
obtained for ML flow rule under different loading conditions on a 
one-element-model

Requires Abaqus installation
uses NumPy and abaqus API

Version: 1.0 (2021-11-27)
Authors: Abhishek Biswas, Mahesh R.G. Prasad, Alexander Hartmaier, ICAMS/Ruhr University Bochum, Germany
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)

November 2021
'''

import numpy as np
import os
import json
from datetime import date
from odbAccess import *
from abaqusConstants import *
from odbMaterial import *
from odbSection import *

ml_name = str(sys.argv[1])
path = './'
print('Processing material data defined for ',ml_name)

def fclear():
    fname =['.odb','.msg','.dat','.com','.prt','.sim','.sta','.log','.lck']
    for i in fname:
      try:
        os.remove(abq_job+i)
      except:pass
    return None

def write_res():
    odb=openOdb(abq_job+'.odb',readOnly=True)
    frameRepository = odb.steps[odb.steps.keys()[-1]].frames
    nframes = len(frameRepository)
    ubc = np.zeros(3)
    ind = list(range(5))
    if nframes>10:
        ind += list(range(5,nframes-1,5))
    with open(path_r+f_name,'a') as f:
        for i in ind:
            s  = frameRepository[i].fieldOutputs['S'].getSubset(CENTROID).values[0].data # stress
            le = frameRepository[i].fieldOutputs['LE'].getSubset(CENTROID).values[0].data # log strain
            ep = np.zeros(6) # plastic strain
            for j, sel in enumerate(sdv_names):
                ep[j] = frameRepository[i].fieldOutputs[sel].getSubset(CENTROID).values[0].data
            ubc[0] = xload
            ubc[1] = yload
            ubc[2] = zload
            a = np.append(s, le)
            a = np.append(a, ep)
            a = np.append(a, ubc)
            a.tofile(f, sep=';', format='%12.5e')
            f.write('\n')
    return None

# Initialization
# string constants
sig_names = ['S11','S22','S33','S23','S13','S12']
epl_names = ['Ep11','Ep22','Ep33','Ep23','Ep13','Ep12']
eps_names = ['E11','E22','E33','E23','E13','E12']
eel_names = ['Ee11','Ee22','Ee33','Ee23','Ee13','Ee12']
ubc_names = ['StrainX', 'StrainY', 'StrainZ', 'StrainYZ','StrainXZ','StrainXY']
sbc_names = ['StressX','StressY','StressZ','StressYZ','StressXZ','StressXY']
ubc_names = ['ux', 'uy', 'uz']
fbc_names = ['fx', 'fy', 'fz']
sdv_names = ['SDV1', 'SDV2', 'SDV3', 'SDV4', 'SDV5', 'SDV6']

# read metadata of Support Vector file
with open('models/abq_'+ml_name+'-svm_meta.json') as f:
    meta = json.load(f)

name  = meta['Model']['Names']
param = meta['Model']['Parameters']
i = name.index('Ndata')
Ndata = param[i]
                
# Abaqus paramaters
ncpu=1
fac = 0.01*0.04 # scaling factor for boundary conditions
ang = np.radians(np.linspace(0,90,num=3))  # list of angles for load cases
abq_job = 'femBlock'  # Abaqus .inp file
abq_umat = 'ml_umat' # umat
f_name = 'abq_'+ml_name+'-res.csv'  # result file
meta_fname = 'abq_'+ml_name+'-res_meta.json'  # metadata file
path_m = 'models/'
path_r = 'results/'

# paramaters for metadata
sep = ';' # CSV separator
names = sep.join(sig_names) + sep + sep.join(eps_names) + sep + \
        sep.join(epl_names) + sep + sep.join(ubc_names) # string for header line
today = str(date.today())  # date
owner = os.getlogin()  # username
sys = os.uname()  # system information

# Create metadata
meta = {
  "Info" : {
      "Owner"       : owner,
      "Institution" : "ICAMS, Ruhr University Bochum, Germany", 
      "Date"        : today,
      "Description" : "Mechanical data from ML umat",
      "Method"      : "FEA",
      "System": {
          "sysname"  : sys[0],
          "nodename" : sys[1],
          "release"  : sys[2],
          "version"  : sys[3],
          "machine"  : sys[4]
      }
  },
  "Model"       : {
    "Creator"   : "Abaqus",
    "Version"   : "6.14-2",
    "Input"     : abq_job,
    "Material"  : abq_umat,
    "Path"      : "./"
  },
  "Parameters"  : {
      "Creator" : 'pylabfea',
      "Version" : "3.4",
      "Repository" : "https://github.com/AHartmaier/pyLabFEA.git",
      "Type"    : 'CSV',
      "File"    : ml_name+'-svm.csv',
      "Path"    : path_m,
      "Meta"    : ml_name+'-svm_meta.json'
  },
  "Data" : {
      "Class"    : 'Flow_Stress',
      "Type"     : 'CSV',
      "File"     : f_name,
      "Path"     : path_r,
      "Format"   : names,
      "Units" : { 
          'Stress' : 'MPa',
          'Strain' : 'None',
          'Disp'   : 'mm',
          'Force'  : 'N'},
      "Separator": sep,
      "Header"   : 0
  },
}
with open(path_r+meta_fname,'w') as fp:
    json.dump(meta, fp, indent=2)

# initialize Abaqus files 
fclear()
with open(path_r+f_name,'w') as f:
    f.write(names+'\n')

# define load cases
lc = [[1., 0., 0.], [0., 1., 0.], [1., 1., 0.], [-1., 1., 0.], \
      [0., 0., 1.], [0., 1., 1.], [1., 0., 1.], [0., -1., 1.],[1., 0., -1.]]
    
# Loop to start Abaqus jobs
for jj in lc:
    hh = 1./np.linalg.norm(jj)
    xload = jj[0]*hh*fac
    yload = jj[1]*hh*fac
    zload = jj[2]*hh*fac

    print(xload)
    print(yload)
    print(zload)
 
    with open(abq_job+'.inp','r') as f:
      data = f.readlines()
      
    # pass information about SVM file
    data[87] = '*User Material, constants='+str(Ndata)+'\n'
    data[88] = '*include, input=models/abq_'+ml_name+'-svm.csv\n'
 
    if not xload==0:
      data[113] = 'Set-15,1,1, '+str(xload)+'\n'
    else:
      data[113] = '***Set-15,1,1, '+str(xload)+'\n'

    if not yload==0:
      data[114] = 'Set-14,2,2, '+str(yload)+'\n'
    else:
      data[114] = '***Set-14,2,2, '+str(yload)+'\n'
 
    if not zload==0:
      data[115] = 'Set-13,3,3, '+str(zload)+'\n'
    else:
      data[115] = '***Set-13,3,3, '+str(zload)+'\n'
 
    with open(abq_job+'.inp','w') as f:
      f.writelines(data)
 
    os.system('abq6142 job={0} user={1} cpus={2} int'.format(abq_job,abq_umat,ncpu)) 
    write_res()
    fclear()
