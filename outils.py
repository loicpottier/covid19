import urllib.request
import gzip
import json
import matplotlib.pyplot as plt
import os
import time
import math
import numpy as np
import sys
import copy
import random
from urlcache import *

DIRSYNTHESE = 'synthese/'
DIRSYNTHESEUK = 'syntheseUK/'

def mfloat(x):
    try:
        return(float(x))
    except:
        print('donnee pas un nombre:',x,'*******')
        return(-100)

# Obtenir l'heure et la date locale
now = time.localtime(time.time())
print(time.asctime(now)) # Afficher la date en format lisible
aujourdhui = (str(now.tm_year) + '-'
              + (str(now.tm_mon) if now.tm_mon > 9
                 else '0' + str(now.tm_mon))
              + '-'
              + (str(now.tm_mday) if now.tm_mday > 9
                 else '0' + str(now.tm_mday)))
def mmax(l):
    if l == []:
        return(1000000000000000000000000000000)
    else:
        return(max(l))

def moyenne(l):
    return(sum(l)/len(l))
           
def centrer(v):
    vm = np.mean(v)
    return(v - vm)

# lissage de degre d
def moyennen(l,d):
    s = 0
    n = len(l)
    c = [0]*n
    for i in range(n):
        c[i] = (1 + (n-1)/2 - abs((n-1)/2 - i))/(1 + (n-1)/2)
        c[i] = c[i]**d
        s = s + c[i]*l[i]
    return(s/sum(c))

# plus rapide 
def lissagecum(L,d):
    cumsum_vec = np.cumsum(np.concatenate(([L[0]]*(d-1),L,[L[-1]]*(d-1))))
    ma_vec = (cumsum_vec[d:] - cumsum_vec[:-d]) / d
    return(ma_vec[(d//2-1):-(d//2)])

# mieux aux bords (approximation linaire)
def lissagecum(L,d): # uniquement d impair
    if d ==1:
        return(L)
    cumsum_vec = np.cumsum(L)
    l = (cumsum_vec[d:] - cumsum_vec[:-d]) / d
    #print(l)
    dd = l[1] - l[0]
    ldeb = [l[0] - dd*(i+1) for i in range((d+1)//2)]
    #print(dd,ldeb)
    dd = l[-1] - l[-2]
    lfin = [l[-1] + dd*(i+1) for i in range((d)//2)]
    #print(dd,lfin)
    l = np.concatenate((ldeb[::-1],l,lfin))
    return(l)

import scipy.signal

# lissage sur d # d impair
def lissage(l,d,repete = 1):
    if len(l) <= 10: return(l)
    l = np.array(l)
    d2 = d//2
    for k in range(repete):
        # on complete comme si c'etait de periode d et lineairement
        l0 = np.concatenate([np.zeros(d),l, np.zeros(d)])
        l0[-d:] = l[-d:] + l[-1] - l[-1-d]
        l0[:d] = l[:d] + l[0] - l[d]
        #print(l0)
        #l1 = np.convolve(l0,np.ones(d)/d,mode='valid')
        l1 = scipy.signal.convolve(l0,np.ones(d)/d,mode='valid')
        l = l1[d2+1:-d2-1]
    return(l)
'''
plt.plot([1,3,5,2,4,6,3,5,7,4])
plt.plot(lissage([1,3,5,2,4,6,3,5,7,4],3))
plt.show()
'''
# on prend le voisinage
def derivee(l, largeur = 1): # largeur impair
    if len(l) <= 3: return(l)
    if largeur == 1:
        l1 = np.array(np.concatenate([l[:1],l]))
        ld = l1[1:] - l1[:-1]
        ld[0] = ld[1]
        return(ld)
    return(lissage(np.gradient(l, edge_order = 2),largeur))

# derivee centrée
def deriveebis(l, largeur = 1): # largeur impair
    if largeur == 1:
        return(np.gradient(l, edge_order = 2))
    return(lissage(np.gradient(l, edge_order = 2),largeur))

def integrate(l,initial = 0):
    i = []
    s = initial
    for x in l:
        s += x
        i.append(s)
    return(i)

mois = ['déc','janv','fév','mars','avril','mai','juin','juil','août','sept','oct','nov','déc']

def joli(jour):
    if type(jour) is str:
        j = jour[8:10]
        m = jour[5:7]
        return(j + ' ' + mois[int(m)])
    else:
        return(str(jour))

#tous les 7 jours
def axejours(ax,lj):
    n = len(lj)
    lk = [n-1 - 7*k for k in range(n//7+1)][::-1]
    if lk[0] < 0:
        lk = [0] + lk[1:]
    #print(lk)
    ljaxe = [joli(lj[k]) for k in lk]
    #print(ljaxe)
    plt.xticks(lk,ljaxe,rotation = 70,fontsize = 6)
    #plt.xticks(np.arange(0, 1, step=0.2))

def val(jours,l):
    d = dict(l)
    return([d[j] if j in d else None for j in jours])

def plotcourbes(courbes,titre='',xlabel = 0,fontcourbes = 8):
    lj = []
    for (courbe,nom,t) in courbes:
        lj = lj + [x[0] for x in courbe]
    lj = sorted(list(set(lj)))
    for (courbe,nom,t) in courbes:
        lv = val(lj,courbe)
        lkv = [(k,lv[k]) for (k,j) in enumerate(lj) if lv[k] != None]
        if lkv != []:
            k,v = lkv[-1]
        else:
            k,v = 0,0
        if t != '=':
            plt.plot(lv,t)
        else:
            plt.plot(lv,'-', linewidth = 2,color = 'b')
        if nom != '':
            if xlabel == 0:
                plt.text(k,lv[k] if lv[k] != None else 0,nom,
                         fontdict = {'size':fontcourbes})
            elif xlabel == 'random':
                lind = [k for (k,v) in enumerate(lv) if v != None]
                x = lind[random.randint(0,len(lind)-1)]
                plt.text(x,lv[x],nom,fontdict = {'size':fontcourbes})
            else:
                plt.text(xlabel,lv[xlabel] if lv[xlabel] != None else 0,
                         nom,fontdict = {'size':fontcourbes})
    ax = plt.gca()
    axejours(ax,lj)
    y0,y1 = ax.get_ylim()
    ax.set_ylim(min(0,y0),y1)


#plt.clf();plotcourbes([(zipper([jour_de_num[j] for j in range(16)],[j for j in range(16)]),'a','-')],'b');plt.show()

real = '='
prev = '--'
minmax = ':'

def trace(lcourbes,titre,fichier,xlabel = 0,dimensions = None, close = True,
          fontcourbes = 8):
    plt.clf()
    #plt.figure(figsize = dimensions)
    plotcourbes(lcourbes,xlabel=xlabel,fontcourbes = fontcourbes)
    plt.grid()
    plt.title(titre,fontdict = {'size':10})
    try:
        plt.savefig(fichier + '.pdf', dpi=300)
    except:
        print('problème pdf: '+ fichier)
    try:
        plt.savefig(fichier + '.png', dpi=300)
        #plt.savefig(fichier + '.jpg', dpi=600)
    except:
        print('problème png: '+ fichier)
    plt.show(False)
    if close:
        plt.close()

def zipper(lj,lv):
    if len(lv) != len(lj):
        print(len(lj),print(lv),
              '========> jours et valeurs pas de même longueur',
              lj[:5],lv[:5])
    return([(lj[k],lv[k]) for k in range(min(len(lv),len(lj)))])

def loadcsv(file, sep = ';', end = '\n'):
    f = open(file,'r')
    s = f.read()
    if '\r\n' in s:
        end = '\r\n'
    data = [x.split(sep) for x in s.split(end)]
    return(data)

def chargecsv(url, zip = False, sep = ';', end = '\n'):
    s = urltexte(url, zip = zip)
    if '\r\n' in s:
        end = '\r\n'
    data = [x.split(sep) for x in s.split(end)]
    return(data)

def chargejson(url, zip = True):
    texte = urltexte(url, zip = zip)
    j = json.loads(texte)
    return(j)

def charge(url):
    j = chargejson(url)
    try:
        data = j['content']['data']['covid_hospit']
    except:
        try:
            data = j['content']['data']['covid_hospit_clage10']
        except:
            try:
                data = j['content']['data']['covid_hospit_incid']
            except:
                try:
                    data = j['content']['data']['sursaud_corona_quot']
                except:
                    try:
                        data = j['content']['zonrefs'][0]['values']
                    except:
                        data = j['content']['data']['sp_pos_quot']
    try:
        titre = j['content']['indicateurs'][0]['c_lib_indicateur']
    except:
        titre = j['content']['indic']['c_lib_indicateur']
    return(data,titre)

######################################################################
# dates

def nextday(jour):
    a = int(jour[:4])
    m = int(jour[5:7])
    j = int(jour[8:10])
    a1 = a
    m1 = m
    j1 = j + 1
    if m in [1,3,5,7,8,10,12] and j == 31:
        if m == 12:
            a1 = a + 1
            m1 = 1
            j1 = 1
        else:
            m1 = m + 1
            j1 = 1
    if m in [4,6,9,11] and j == 30:
        m1 = m + 1
        j1 = 1
    if m == 2 and a % 4 == 0 and j == 29:
        m1 = 3
        j1 = 1
    if m == 2 and a % 4 != 0 and j == 28:
        m1 = 3
        j1 = 1
    mm = str(m1)
    jj = str(j1)
    if m1 < 10: mm = '0' + mm
    if j1 < 10: jj = '0' + jj
    return(str(a1) + '-' + mm + '-' + jj)

def addday(j,k):
    j1 = j
    for i in range(k):
        j1 = nextday(j1)
    return(j1)

jdebut = '2018-01-01'
memo_num_de_jour = {}
def num_de_jour(jour): # depuis le 1 janvier 2018: num 0
    if jour in memo_num_de_jour:
        return(memo_num_de_jour[jour])
    n = 0
    j = jdebut
    while j != jour:
        memo_num_de_jour[j] = n
        n += 1
        j = nextday(j)
    memo_num_de_jour[jour] = n
    return(n)

jaujourdhui = num_de_jour(aujourdhui)

jour_de_num = ['']*10000
j = jdebut
for n in range(len(jour_de_num)):
    jour_de_num[n] = j
    j = nextday(j)

    
def table0(t):
    ts = ('<div class="container-fluid"><table border = 0 >'
          + '\n'.join(['<tr>' + ''.join(['<td><div class="container-fluid">' + x + '</div></td>' for x in r]) + '</tr>' for r in t])
          + '</table></div>')
    return(ts)

def table1(t):
    ts = ('<div class="container-fluid"><table border = 1 >'
          + '\n'.join(['<tr>' + ''.join(['<td><div class="container-fluid">' + x + '</div></td>' for x in r]) + '</tr>' for r in t])
          + '</table></div>')
    return(ts)

def table(t):
    ts = ('<div class="container-fluid">'
          + '<table class="table table-hover table-sm">'
          #class="table table-striped">'
          +'<thead><tr>'
          + '\n'.join(['<th scope="col"><div class="container-fluid">' + x + '</div></th>' for x in t[0]])
          + '</tr></thead>'
          + '<tbody>'
          + '\n'.join(['<tr>'
                       + '<th scope="row">' + r[0] + '</th>'
                       + ''.join(['<td><div class="container-fluid">'
                                  + x + '</div></td>' for x in r[1:]])
                       + '</tr>' for r in t[1:]])
          + '</tbody>'
          + '</table>'
          + '</div>')
    return(ts)

# super table: https://examples.bootstrap-table.com/#view-source

def premieredonnee(l):
    if l == []:
        return(-1)
    else:
        return(l[0])

######################################################################
# super table bootstrap
# https://examples.bootstrap-table.com/#view-source
# exemple de data en json:
# https://examples.wenzhixin.net.cn/examples/bootstrap_table/data

# j = chargejson('https://examples.wenzhixin.net.cn/examples/bootstrap_table/data',zip = False)
# j: {'total': 800,
#     'totalNotFiltered': 800,
#     'rows': [{'id': 0, 'price': '$0', 'name': 'Item 0'},
#              ...
#             ]
#    }

intervalle_seriel = 4.11 # = math.log(3.296)/0.29 
# majoré par maxR
def r0(l, derive = 7,maxR = 10): #maxR = 3
    # l1: log de l
    l1 = [math.log(x) if x>0 else 0 for x in l]
    # dérivée de l1
    dl1 = derivee(l1, largeur = derive) #derivee(l1,largeur=7)
    # r0 instantané
    lr0 = [min(maxR,math.exp(c*intervalle_seriel)) for c in dl1]
    return(lr0)

# majoré par maxR
def r0bis(l, derive = 7,maxR = 3):
    # dérivée de l
    dl = derivee(l, largeur = derive)
    l1 = [dl[k]/x if x != 0 else 0 for (k,x) in enumerate(l)]
    # r0 instantané
    lr0 = [min(maxR,math.exp(min(c,100)*intervalle_seriel)) for c in l1]
    return(lr0)

def Reff(l, derive = 7,maxR = 3):
    # l1: log de l
    l1 = [math.log(x) if x>0 else 0 for x in l]
    # dérivée de l1
    dl1 = derivee(l1, largeur = derive) #derivee(l1,largeur=7)
    # r0 instantané
    lr0 = [min(maxR,math.exp(min(100,c*intervalle_seriel))) for c in dl1]
    return(lr0)

maxfprimef = 1e3 # ca pete le determinant sinon

def fprimef(l, derive = 7):
    # dérivée de l
    dl = derivee(l, largeur = derive)
    l1 = [dl[k]*x/maxfprimef for (k,x) in enumerate(l)]
    return(l1)

population_dep = {1:656955, 2:526050, 3:331315, 4:165197, 5:141756, 6:1079396, 7:326875, 8:265531, 9:152398, 10:309907, 11:372705, 12:278360, 13:2034469, 14:691453, 15:142811, 16:348180, 17:647080, 18:296404, 19:240336, 21:532886, 22:596186, 23:116270, 24:408393, 25:539449, 26:520560, 27:600687, 28:429425, 29:906554, 30:748468, 31:1400935, 32:190040, 33:1633440, 34:1176145, 35:1082073, 36:217139, 37:605380, 38:1264979, 39:257849, 40:411979, 41:327835, 42:764737, 43:226901, 44:1437137, 45:682890, 46:173166, 47:330336, 48:76286, 49:815881, 50:490669, 51:563823, 52:169250, 53:305365, 54:730398, 55:181641, 56:755566, 57:1035866, 58:199596, 59:2588988, 60:825077, 61:276903, 62:1452778, 63:660240, 64:683169, 65:226839, 66:479000, 67:1132607, 68:763204, 69:1876051, 70:233194, 71:547824, 72:560227, 73:432548, 74:828405, 75:2148271, 76:1243788, 77:1423607, 78:1448625, 79:372627, 80:569769, 81:387898, 82:262618, 83:1073836, 84:560997, 85:683187, 86:437398, 87:370774, 88:359520, 89:332096, 90:140145, 91:1319401, 92:1613762, 93:1670149, 94:1406041, 95:1248354}

population_france = sum([population_dep[x] for x in population_dep])
population_dep[0] = population_france

# les regions
f = open('regions.csv','r')
s = f.read()
f.close()

ls = [x.split('\t') for x in s.split('\n')]
regions = {}
depregion = {}
nomdep = {}
for x in ls:
    d = x[0]
    if d[1] not in 'ab':
        nomdep[int(d)] = x[1]
    r = x[3] #nom de region
    if r not in regions and r != 'Corse':
        regions[r] = [int(x[2])] # numero de region
    try:
        regions[r].append(int(x[0])) # le departement
        depregion[int(x[0])] = int(x[2]) # numero de region
    except:
        pass # la corse...

lregions = [(regions[r][0],regions[r][1:]) for r in regions] # numeros des regions et departements
regiondep = dict(lregions)

def derivee_indic(t,dec):
    return(np.array([derivee(d,largeur=dec) for d in t]))

def lissage_indic77(t):
    res = np.array([lissage(d,7) for d in t])
    return(res)

def lissage77(x):
    return(lissage(x,7))

def id(x):
    return(x)

# https://www.jchr.be/python/ecma-48.htm
def printback(s):
    s = s + ' '
    print("\x1b[" + str(len(s)) + "8D" + s, end = '', flush = True)

def extrapole_manquantes(ld, manquante = 0):
    # on extrapole linéairement les données manquantes (valeur 0)
    for j in range(len(ld)):
        if ld[j] == 0:
            suiv = 0
            jsuiv = j
            try:
                for u in range(j+1,len(ld)):
                    if ld[u] != 0:
                        suiv = ld[u]
                        jsuiv = u
                        raise NameError('ok')
            except:
                pass
            prev = 0
            jprev = j
            try:
                for u in range(j-1,-1,-1):
                    if ld[u] != 0:
                        prev = ld[u]
                        jprev = u
                        raise NameError('ok')
            except:
                pass
            ld[j] = prev + (j - jprev)/(jsuiv-jprev) * (suiv - prev)
            #print(j,jprev,jsuiv,suiv,prev,ld[j])
    return(ld)

extrapole_manquantes([1,2,0,0,5])
