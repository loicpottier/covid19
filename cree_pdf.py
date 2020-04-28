import matplotlib
# sous bash de windows, ajouter ca:
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import sys

lfpdf = []
lfpy = [sys.argv[1]]
#lfpy = ["france/jour_50_err_473.4286_R0_3.30_R01_1.48_R02_1.40_dR02_4_pvoy_0.11_debi_6_duri_6_mor_0.05409.py"]
for file in lfpy:
    f = open(file,'r')
    s = f.read()
    f.close()
    exec(s)
    fig = plt.figure(1, figsize=(10, 6))
    plt.clf()
    plt.title(pays + ' jour '+str(jour)+' depuis le '+datedebut+'\n'
              +file[14:-3].replace('_pvoy','_\npvoy').replace('_',' '))
    plt.plot([j for j,m,c,i,g,mf in hist if mf != -1],
             [mf for j,m,c,i,g,mf in hist if mf != -1],'o')
    plt.plot([j for j,m,c,i,g,mf in hist if m != 0],
             [m for j,m,c,i,g,mf in hist if m != 0])
    plt.savefig(file + '.pdf')
    #plt.show(False)

    
