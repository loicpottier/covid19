import os
import matplotlib
# sous bash de windows, ajouter ca:
if os.uname().nodename == 'pcloic':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import re
import sys
import numpy as np

file = sys.argv[1]
#file ='france_rea_jour_55/jour_55_err_4.1356_R0_6.21_dR0_4.7_R01_0.75_pvoy_0.16_dpvoy_23_debi_3_duri_7_mor_0.11185_dc_0_nc_151281_lm_14938_carte'
f = open(file,'r')
s = f.read()
f.close()
carte = eval(s)
carte = np.array(carte)
#carte = carte / np.max(carte)
carte = np.log(carte+1)
im = plt.imshow(carte,interpolation='none')
plt.title(file.replace('_pvoy','\n_pvoy'),fontdict = {'size':6})
im.set_cmap('inferno')
plt.savefig(file + '.pdf')
#plt.show(False)

#imfr = plt.imread('carte_de_france.png')
#im = plt.imshow(imfr,interpolation='none')
#plt.show(False)
    
