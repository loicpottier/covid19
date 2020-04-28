import matplotlib
# sous bash de windows, ajouter ca:
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import sys
import numpy as np

file = sys.argv[1]
f = open(file,'r')
s = f.read()
f.close()
carte = eval(s)
carte = np.array(carte)
#carte = carte / np.max(carte)
carte = np.log(carte+1)
im = plt.imshow(carte,interpolation='none')
plt.title(file,fontdict = {'size':6})
im.set_cmap('inferno')
plt.savefig(file + '.pdf')
#plt.show(False)

#imfr = plt.imread('carte_de_france.png')
#im = plt.imshow(imfr,interpolation='none')
#plt.show(False)
    
