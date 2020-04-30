# trace à partir des fichiers previsions
import os
import matplotlib
# sous bash de windows, ajouter ca:
if os.uname().nodename == 'pcloic':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import re
import sys
import math

try:
    pays = sys.argv[1] #'france' #'france_rea' #'italie'
except:
    pays = 'france_rea'
    
try:
    nbr_min_simulations = int(sys.argv[2])
except:
    nbr_min_simulations = 20
    
lf = os.listdir(pays)

print(len(lf), "fichiers")

# alpes maritimes
def dep06():
    dep06_morts = [2,2,4,6,7,8,8,9,12,12,15,17,20,23,29,32,35,38,43,47]
    fig = plt.figure(1)
    plt.clf()
    plt.plot([x for x in range(len(dep06_morts))],dep06_morts)
    plt.savefig('dep06_morts.pdf')
    plt.show()

#dep06()

######################################################################
# analyse des simulations

ln = [f[:-3]  for f in lf if f[-3:] == '.py']

def convert(s):
    m = re.compile('jour_(?P<jour>[0-9]+)_'
                   + 'err_(?P<err>[0-9\.]+)_'
                   + 'R0_(?P<R0>[0-9\.]+)_'
                   + 'dR0_(?P<dR0>[0-9\.]+)_'
                   + 'R01_(?P<R01>[0-9\.]+)_'
                   + 'pvoy_(?P<pvoy>[0-9\.]+)_'
                   + 'dpvoy_(?P<dpvoy>[0-9\.]+)_'
                   + 'debi_(?P<debi>[0-9]+)_'
                   + 'duri_(?P<duri>[0-9]+)_'
                   + 'mor_(?P<mor>[0-9\.]+)_'
                   + 'dc_(?P<dc>[-0-9]+)_'
                   + 'nc_(?P<nc>[0-9]+)_'
                   + 'lm_(?P<lm>[0-9]+)').search(s)
    return([#int(m.group('jour')),
        float(m.group('err')),
        float(m.group('R0')),
        float(m.group('dR0')),
        float(m.group('R01')),
        float(m.group('pvoy')),
        float(m.group('dpvoy')),
        int(m.group('debi')),
        int(m.group('duri')),
        float(m.group('mor')),
        #int(m.group('dc')),
        int(m.group('nc')),
        int(m.group('lm')),
    ])

m = re.compile('jour_(?P<jour>[0-9]+)_').search(ln[0])
jour = int(m.group('jour'))

#convert('jour_49_err_1.4747_R0_5.17_dR0_3.7_R01_0.99_pvoy_0.16_dpvoy_97_debi_2_duri_9_mor_0.10510_dc_0_nc_263709_lm_17041')

# tracé des meilleures courbes de simulation

lno = sorted(ln, key = lambda x: convert(x)[0])
nm = 50

def trace_tout():
    fig = plt.figure(1, figsize=(10, 6))
    plt.clf()
    for i,file in enumerate(lno[:nm][::-1]):
        print(i,file)
        params = convert(file)
        f = open(pays + '/' + file  + '.py','r')
        s = f.read()
        f.close()
        hist = []
        dl = {'hist':[],'jour':0}
        exec(s,{},dl)
        hist = dl['hist']
        jour = dl['jour']
        if i == len(lno[:nm])-1:
            plt.plot([j for j,m,c,i,g,mf in hist if mf != -1],
                     [mf for j,m,c,i,g,mf in hist if mf != -1],'o',
                     color = 'midnightblue')
            plt.title(pays + ', jour ' + str(jour) + ', ' + str(min(len(lno),nm))
                      + ' meilleures simulations (la meilleure est en gras)')
        lw = 1
        if  i == len(lno[:nm])-1: lw = 3
        plt.plot([j for j,m,c,i,g,mf in hist if m != 0],
                 [m for j,m,c,i,g,mf in hist if m != 0], linewidth = lw)
        j,m = [(j,m) for j,m,c,i,g,mf in hist if m != 0][-1]
        plt.text(j,m,str("%.1f" % params[0]),fontdict = {'size':6})
    plt.savefig(pays + '/' + '_tout_jour_' + str(jour) + '.pdf')
    #plt.show(False)

trace_tout()
print('trace_tout ok')

#indices
err = 0
R0 = 1
dR0 = 2
R01 = 3
pvoy = 4
dpvoy = 5
debi = 6
duri = 7
mor = 8
nc = 9
lm = 10
nom = ['err','R0','dR0','R01','pvoy','dpvoy','debi','duri','mor','nc','lm']

datas = [convert(s) for s in ln]

# on ne garde que les simulations avec des erreurs faibles
# datas = [x for x in datas if x[err] < 60]

######################################################################
# si on veut seulement l'erreur moyenne
# pour les donnees avec les memes parametres avec 20 simulations au moins

def moyenne(l):
    if l != []:
        return(sum(l)/len(l))
    else:
        return(0)

def ecartype(l):
    m = moyenne(l)
    var = moyenne([(x-m)**2 for x in l])
    return(math.sqrt(var))

params = [eval(x) for x in list(set([str(x[1:-2]) for x in datas]))]
ld = [[x for x in datas if x[1:-2] == p] for p in params]
ld20 = [x for x in ld if len(x) >= nbr_min_simulations]
le = [[moyenne([x[0] for x in lp])]
      +lp[0][1:-2]
      + [moyenne([x[-2] for x in lp]),
         moyenne([x[-1] for x in lp])]
      for lp in ld20]
le = sorted(le, key = lambda x : x[0])
try:
    meilleure = le[0]
    print('meilleure erreur sur ' + str(nbr_min_simulations)
          + ' essais:',meilleure)
    meilleure2 = le[1]
    print('2 eme meilleure erreur sur ' + str(nbr_min_simulations)
          + ' essais:',meilleure2)

    meilleures20 = [x for x in datas if x[1:-2] == meilleure[1:-2]]
    fichiers20 = sorted([x for x in ln if convert(x)[1:-2] == meilleure[1:-2]],
                    key = lambda x: convert(x)[0])
except:
    pass

def joli(param):
    err,R0,dR0,R01,pvoy,dpvoy,debi,duri,mor,nc,lm = param
    return("[err %.2f, R0 %.1f, dR0 %.1f, R01 %.2f, pvoy %.2f, dpvoy %.0f, debi %d, duri %d, mor %.4f, nc %d, lm %d]" % (err,R0,dR0,R01,pvoy,dpvoy,int(debi),int(duri),mor,int(nc),int(lm)))

def trace_20meilleures():
    fig = plt.figure(1, figsize=(10, 6))
    plt.clf()
    for i,file in enumerate(fichiers20[::-1]):
        print(i,file)
        params = convert(file)
        f = open(pays + '/' + file  + '.py','r')
        s = f.read()
        f.close()
        hist = []
        dl = {'hist':[],'jour':0}
        exec(s,{},dl)
        hist = dl['hist']
        jour = dl['jour']
        if i == 0:
            plt.plot([j for j,m,c,i,g,mf in hist if mf != -1],
                     [mf for j,m,c,i,g,mf in hist if mf != -1],'o',
                     color = 'midnightblue')
            plt.title(pays + ', jour ' + str(jour) + ', meilleure moyenne sur '
                      + str(len(fichiers20))
                      + ' simulations (la meilleure est en gras)\n'
                      + joli(meilleure)
                      # + '\n' + joli(meilleure2)
            )
        lw = 1
        if  i == len(fichiers20)-1: lw = 3
        plt.plot([j for j,m,c,i,g,mf in hist if m != 0],
                 [m for j,m,c,i,g,mf in hist if m != 0], linewidth = lw)
        j,m = [(j,m) for j,m,c,i,g,mf in hist if m != 0][-1]
        plt.text(j,m,str("%.1f" % params[0]),fontdict = {'size':6})
    plt.savefig(pays + '/' + '_' + str(nbr_min_simulations) + 'meilleures_jour_'
                + str(jour) + '.pdf')
    #plt.show(False)

try:
    trace_20meilleures()
    print('trace_' + str(nbr_min_simulations) + 'meilleures ok')
except:
    pass

######################################################################
# valeurs moyennes des paramètres
import numpy as np

try:
    erreur_max = float(sys.argv[3])
except:
    erreur_max = 20

datas = [x for x in datas if x[0] < erreur_max]
datas = sorted(datas, key = lambda x : x[0])

datas = np.array(datas)

######################################################################
# valeurs moyennes des paramètres en fonction de l'erreur
fig = plt.figure(figsize=(8,8))
plt.clf()

def lissage(l,d):
    l1 = []
    for i in range(len(l)):
        l1.append(moyenne(l[i:i+d]))
    return(l1)

#https://stats.stackexchange.com/questions/219810/r-squared-and-higher-order-polynomial-regression
# rend l'intersection avec l'axe x=0, l'erreur quadratique
# et le coefficient de détermination
def regression2(lx,ly):
    a,b,c = np.polyfit(lx,ly,2)
    e = 0
    ey = 0
    y = moyenne(ly)
    for i,x in enumerate(lx):
        e += (a*x**2+b*x+c - ly[i])**2
        ey += (ly[i] - y)**2
    r2 = 1 - e/ey
    e = math.sqrt(e/len(lx))
    return((c,e,r2))

def regression1(lx,ly):
    a,b = np.polyfit(lx,ly,1)
    e = 0
    ey = 0
    y = moyenne(ly)
    for i,x in enumerate(lx):
        e += (a*x+b - ly[i])**2
        ey += (ly[i] - y)**2
    r2 = 1 - e/ey
    e = math.sqrt(e/len(lx))
    return((b,e,r2))

lreg = [((0,0,0),(0,0,0))]*len(nom)

for p,pn in enumerate(nom):
    if pn in ['R0','mor','R01','debi','duri','dR0','pvoy','dpvoy']:
        print(p,pn)
        lx = []
        ly = []
        ly1 = []
        for k in range(100):
            emax = k/100 * erreur_max
            lv = [x[p] for x in datas if x[err] < emax]
            if lv != []:
                lx.append(emax)
                npmoy = moyenne(lv)
                ly1.append(npmoy)
                if pn == 'mor': npmoy = 100 * npmoy
                if pn == 'R01' : npmoy = 10 * npmoy
                if pn == 'pvoy' : npmoy = 100 * npmoy
                if pn == 'dpvoy': npmoy = npmoy / 10
                ly.append(npmoy)
        ly = lissage(ly,5)
        text = pn
        if pn == 'mor': text = 'IFR(%)'
        if pn == 'R01': text = '10*R01'
        if pn == 'pvoy': text = '100*pvoy'
        if pn == 'dpvoy': text = 'dpvoy/10'
        plt.text(lx[-1],ly[-1],text)
        plt.plot(lx,ly)
        lreg[p] = (regression1(lx,ly1),regression2(lx,ly1))
    
plt.title(str(len(datas)) + " simulations with error  < " + str(erreur_max)
          + ", average of parameters function of max error")
plt.savefig(pays + '/' + '_params_err_max_jour_' + str(jour) + '.pdf')

def prv(pn,v):
    if type(v) is str: return(v)
    if pn in ['err','R0','dR0','debi','duri']: return("%.1f" % v)
    if pn in ['R01','pvoy']: return("%.2f" % v)
    if pn in ['mor']: return("%.3f" % v)
    if pn in ['dpvoy']:return("%.0f" % v)
    return("")

f = open(pays + '/_histogramme.csv','w')
ninterv = 8
f.write(str(len(datas)) + " sim.  err<" + str(erreur_max)
        + ";average;cri;limit1 (r21);cri1;limit2 (r22);cri2;common interval;final average\n")
for p,pn in enumerate(nom):
    l = datas[:,p]
    pmax = max(l)
    pmin = min(l)
    m = moyenne(l)
    e = ecartype(l)
    r1,r2 = lreg[p]
    o1,e1,r21 = r1
    o2,e2,r22 = r2
    a = max([m-e,o1-e1,o2-e2])
    b = min([m+e,o1+e1,o2+e2])
    lv = [m,m-e,m+e,o1,("%.1f" % r21),o1-e1,o1+e1,o2,("%.1f" % r22),o2-e2,o2+e2,a,b,(a+b)/2]
    print(lv)
    lv1 = [prv(pn,x) for x in lv]
    print(lv1)
    hist = []
    #print("--------- %s: moyenne %.2f, ecartype %.2f" % (pn,m,e))
    if "" not in lv1:
        f.write("%s;%s;%s-%s;%s (%s);%s-%s;%s (%s);%s-%s;%s-%s;%s\n" % (pn,lv1[0],lv1[1],lv1[2],lv1[3],lv1[4],lv1[5],lv1[6],lv1[7],lv1[8],lv1[9],lv1[10],lv1[11],lv1[12],lv1[13]))
        # à l'écran, pas dans le fichier:
    if False:
        for k in range(ninterv):
            l1 = [x for x in l
                  if x >= pmin + k * (pmax - pmin) / ninterv
                  and x < pmin + (k + 1)* (pmax - pmin) / ninterv]
            m1 = moyenne(l1)
            e1 = ecartype(l1)
            hist.append((m1,e1,100*len(l1)/len(l)))
            print("entre %.2f et %.2f:\t proportion %.1f, moyenne %.2f, ecartype %.2f" % (pmin + k * (pmax - pmin) / ninterv, pmin + (k + 1)* (pmax - pmin) / ninterv, 100*len(l1)/len(l),m1,e1))

f.close()

'''
######################################################################
# R0 > 5 et IFR
print("---------------------------------------------------------------------------")
print("R0 > 5:",len([x for x in datas if x[R0] > 5]) / len(datas))
print("R0 > 5 et IFR < 5%:",len([x for x in datas if x[R0] > 5 and x[mor] < 0.05])
      / len(datas))
print("R0 > 5 et IFR >= 5%:",len([x for x in datas if x[R0] > 5 and x[mor] >= 0.05])
      / len(datas))
print("---------------------------------------------------------------------------")

print("R0 > 5 et IFR < 3.3%:",len([x for x in datas if x[R0] > 5 and x[mor] < 0.033])
      / len(datas))
print("R0 > 5 et 3.3% <= IFR < 6.6%:",
      len([x for x in datas if x[R0] > 5 and x[mor] >= 0.033 and x[mor] < 0.066])
      / len(datas))
print("R0 > 5 et IFR >= 6.6%:",len([x for x in datas if x[R0] > 5 and x[mor] >= 0.066])
      / len(datas))
print("---------------------------------------------------------------------------")
print("avec R0 > 5:")
l5 = [x for x in datas if x[R0] > 5]
print("%d simulations sur %d avec erreur < %d: proportion %.2f"
      % (len(l5),len(datas),erreur_max,len(l5)/len(datas)))
for ee in range(1,4*20 + 1):
    e = ee/4
    l0 = [x for x in datas if x[err] < e]
    l1 = [x for x in l5 if x[err] < e]
    l2 = [x for x in l1 if x[mor] < 0.05]
    if len(l1) != 0:
        print("err < %.2f, %d sim (%.3f): IFR<5: %.2f, moy: %.3f, ect: %.3f, moy(toutR0): %.3f"
              % (e,len(l1),len(l1)/len(l5),len(l2)/len(l1),
                 moyenne([x[mor] for x in l1]),
                 ecartype([x[mor] for x in l1]),
                 moyenne([x[mor] for x in l0]),))

print("---------------------------------------------------------------------------")

######################################################################
# ACP (analyse en composantes principales)

# normalisation
datas1 = np.copy(datas)
for i in range(len(datas1[0])):
    m = max(datas1[:,i])
    datas1[:,i] = datas1[:,i]/m

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Affichage du nuage de points projection sur err, R0 et mor
def donnees():
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(datas)):
        x,y,z = datas[i,0], datas[i,1], datas[i,8] 
        ax.scatter(x,y,z)
        ax.text(x,y,z,str(int(datas[i][0]))) # err
    plt.title("Données initiales")
    plt.show()

memes = [[z for z in datas if z[R0] == datas[i,R0] and z[mor] == datas[i,mor]]
         for i in range(len(datas))]
#####################
print("Affichage du nuage de points projection sur R0 et mor")
fig = plt.figure(figsize=(8,8))
plt.clf()
ax = fig.add_subplot(111)
for i in range(len(datas)):
        x,y = datas[i,R0], datas[i,mor]
        meme = memes[i]
        errmin = min([z[err] for z in meme])
        ax.scatter(x,y,s = 40*(len(meme)))
        ax.text(x,y,str(int(errmin)), fontdict = {'size':8})
        plt.title(str(len(datas)) + " simulations d'erreur < " + str(erreur_max)
                  + ', R0 en abcisse, IFR en ordonnée\n taille des disques = nbre de simulations de mêmes paramètres,\nsur les disques: erreur min')
plt.savefig(pays + '/' + '_R0_mor_jour_' + str(jour) + '.pdf')
#####################
print("Affichage du nuage de points projection sur R01 et err")
fig = plt.figure(figsize=(8,8))
plt.clf()
ax = fig.add_subplot(111)
for i in range(len(datas)):
        x,y = datas[i,R01], datas[i,err]
        meme = memes[i]
        errmin = min([z[err] for z in meme])
        ax.scatter(x,y)
        ax.text(x,y,str(int(datas[i][R0])),
                fontdict = {'size':8}) # debi
        plt.title(str(len(datas)) + " simulations d'erreur < " + str(erreur_max)
                  + ', R01 en abcisse, erreur en ordonnée\n taille des disques = nbre de simulations de mêmes paramètres,\n sur les disques: R0')
plt.savefig(pays + '/' + '_R01_err_jour_' + str(jour) + '.pdf')
#####################
print("Affichage du nuage de points projection sur R0 et err")
fig = plt.figure(figsize=(8,8))
plt.clf()
ax = fig.add_subplot(111)
for i in range(len(datas)):
        x,y = datas[i,R0], datas[i,err]
        meme = memes[i]
        errmin = min([z[err] for z in meme])
        ax.scatter(x,y)
        ax.text(x,y,str(int(100*datas[i][mor])),
                fontdict = {'size':8}) # debi
        plt.title(str(len(datas)) + " simulations d'erreur < " + str(erreur_max)
                  + ', R0 en abcisse, erreur en ordonnée\n taille des disques = nbre de simulations de mêmes paramètres,\n sur les disques: IFR')
plt.savefig(pays + '/' + '_R0_err_jour_' + str(jour) + '.pdf')
#####################
print("Affichage du nuage de points projection sur mor et duri")
fig = plt.figure(figsize=(8,8))
plt.clf()
ax = fig.add_subplot(111)
for i in range(len(datas)):
        x,y = datas[i,mor], datas[i,duri]
        meme = meme = memes[i]
        errmin = min([z[err] for z in meme])
        ax.scatter(x,y,s = 40*(len(meme)))
        ax.text(x,y,str(int(datas[i][R0])),
                fontdict = {'size':8}) # debi
        plt.title(str(len(datas)) + " simulations d'erreur < " + str(erreur_max)
                  + ', IFR en abcisse, durée infectant en ordonnée\n taille des disques = nbre de simulations de mêmes paramètres,\n sur les disques: R0')
plt.savefig(pays + '/' + '_mor_duri_jour_' + str(jour) + '.pdf')
#####################
print("Affichage du nuage de points projection sur mor et debi")
fig = plt.figure(figsize=(8,8))
plt.clf()
ax = fig.add_subplot(111)
for i in range(len(datas)):
        x,y = datas[i,mor], datas[i,debi]
        meme = memes[i]
        errmin = min([z[err] for z in meme])
        ax.scatter(x,y,s = 40*(len(meme)))
        ax.text(x,y,str(int(datas[i][duri])),
                fontdict = {'size':8}) # debi
        plt.title(str(len(datas)) + " simulations d'erreur < " + str(erreur_max)
                  + ', IFR en abcisse, début infectant en ordonnée\n taille des disques = nbre de simulations de mêmes paramètres,\n sur les disques: duri')
plt.savefig(pays + '/' + '_mor_debi_jour_' + str(jour) + '.pdf')
#####################
print("Affichage du nuage de points projection sur mor et err")
fig = plt.figure(figsize=(8,8))
plt.clf()
ax = fig.add_subplot(111)
for i in range(len(datas)):
        x,y = datas[i,mor], datas[i,err]
        meme = memes[i]
        errmin = min([z[err] for z in meme])
        ax.scatter(x,y)
        ax.text(x,y,str(int(datas[i][R0])),
                fontdict = {'size':8}) # err
        plt.title(str(len(datas)) + " simulations d'erreur < " + str(erreur_max)
                  + ', IFR en abcisse, erreur en ordonnée\n taille des disques = nbre de simulations de mêmes paramètres,\n sur les disques: R0')
plt.savefig(pays + '/' + '_mor_err_jour_' + str(jour) + '.pdf')
#####################
print("Affichage du nuage de points projection sur duri et debi")
fig = plt.figure(figsize=(8,8))
plt.clf()
ax = fig.add_subplot(111)
for i in range(len(datas)):
        x,y = datas[i,duri], datas[i,debi]
        meme = memes[i]
        errmin = min([z[err] for z in meme])
        ax.scatter(x,y,s = 40*(len(meme)))
        ax.text(x,y,str(int(datas[i][R0])),
                fontdict = {'size':8}) # debi
        plt.title(str(len(datas)) + " simulations d'erreur < " + str(erreur_max)
                  + ', durée infectant en abcisse, début infectant en ordonnée\n taille des disques = nbre de simulations de mêmes paramètres,\n sur les disques: R0')
plt.savefig(pays + '/' + '_duri_debi_jour_' + str(jour) + '.pdf')
#####################
print("Affichage du nuage de points projection sur R0 et duri")
fig = plt.figure(figsize=(8,8))
plt.clf()
ax = fig.add_subplot(111)
for i in range(len(datas)):
        x,y = datas[i,R0], datas[i,duri]
        meme = memes[i]
        errmin = min([z[err] for z in meme])
        ax.scatter(x,y,s = 40*(len(meme)))
        ax.text(x,y,str(int(100*datas[i][mor])),
                fontdict = {'size':8}) # debi
        plt.title(str(len(datas)) + " simulations d'erreur < " + str(erreur_max)
                  + ', R0 en abcisse, durée infectant en ordonnée\n taille des disques = nbre de simulations de mêmes paramètres,\n sur les disques: IFR')
plt.savefig(pays + '/' + '_R0_duri_jour_' + str(jour) + '.pdf')

#plt.show()

#donnees()
'''

''' ca ralentit pas mal 
print("ACP")
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(datas1)

datas2 = pca.transform(datas1) # rotation sur les axes propres
print(datas2[:3,:])

# visu sur les 3 plus grandes val propres
fig = plt.figure(figsize=(8,8))
plt.clf()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(datas)):
        x,y,z = datas2[i,0], datas2[i,1], datas2[i,2]
        ax.scatter(x,y,z)
        ax.text(x,y,z,str(int(datas[i][0])),fontdict = {'size':6})

plt.savefig(pays + '/' + '_3d_jour_' + str(jour) + '.pdf')

#plt.show()

# visu sur les 2 plus grandes val propres
fig = plt.figure(figsize=(8,8))
plt.clf()
ax = fig.add_subplot(111)
for i in range(len(datas)):
        x,y = datas2[i,0], datas2[i,1]
        ax.scatter(x,y)
        ax.text(x,y,str(int(datas[i][0])),fontdict = {'size':6}) # err

plt.savefig(pays + '/' + '_2d_jour_' + str(jour) + '.pdf')
#plt.show()

# valeurs propres
pca.explained_variance_ratio_
# vecteurs propres
pca.components_

# plus petite valeur propre non nulle:
pca.explained_variance_ratio_[-2]
# son vecteur propre:
v1 = pca.components_[-2]
# c'est la normale a l'hyperplan qui approche le mieux les données...
# du coup on a une équation genre
# a1x1+...+anxn = constante

constantes = [sum([a*x[i] for i,a in enumerate(v1)]) for x in datas]
constante = sum(constantes)/len(constantes)

print(nom)
print('normale: ',1000*v1, 'constante:', 1000*constante)

def equation(x):
    return(sum([a*x[i] for i,a in enumerate(v1)]) - constante)

[equation(x) for x in datas]

# correlations
cov = np.corrcoef(np.transpose(datas))
cov1 = cov - np.identity(len(cov))

lcor =[]
for i in range(9):
    for j in range(i+1,9):
      lcor.append((nom[i],nom[j],cov1[i,j]))

lcor = sorted(lcor,key = lambda x:-abs(x[2]))
print("************************************************************\n",
      "Corrélations:")
[print(x) for x in lcor]

'''





    
    
