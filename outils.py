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
from urlcache import *

def mfloat(x):
    try:
        return(float(x))
    except:
        print('donnee pas un nombre:',x,'*******')
        return(-100)

# Obtenir l'heure et la date locale
now = time.localtime(time.time())
print(time.asctime(now)) # Afficher la date en format lisible

def mmax(l):
    if l == []:
        return(1000000000000000000000000000000)
    else:
        return(max(l))

def moyenne(l):
    return(sum(l)/len(l))
           
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

# lissage sur d
def lissage(l,d,degre=0):
    dd = d//2
    # on complete comme si c'etait de periode d et lineairement
    l0 = list(l) + [0]*d
    l0[-d:] = [x + l[-1] - l[-1-d] for x in l[-d:]]
    l1 = []
    for i in range(len(l)):
        ld = l0[max(0,i-dd):i+dd+1]
        l1.append(moyennen(ld,degre))
    return(l1)

# on prend le voisinage
def derivee(l, largeur = 1): # largeur impair
    if largeur == 1:
        l1 = np.array(np.concatenate([l[:1],l]))
        ld = l1[1:] - l1[:-1]
        ld[0] = ld[1]
        return(ld)
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
    j = jour[8:10]
    m = jour[5:7]
    return(j + ' ' + mois[int(m)])

#tous les 7 jours
def axejours(lj):
    n = len(lj)
    lk = [n-1 - 7*k for k in range(n//7)][::-1]
    #print(lk)
    ljaxe = [joli(lj[k]) for k in lk]
    plt.xticks(lk,ljaxe,rotation = 70,fontsize = 8)
    #plt.xticks(np.arange(0, 1, step=0.2))

def val(jours,l):
    d = dict(l)
    return([d[j] if j in d else None for j in jours])

def plotcourbes(courbes,titre='',xlabel = 0):
    lj = []
    for (courbe,nom,t) in courbes:
        lj = lj + [x[0] for x in courbe]
    lj = sorted(list(set(lj)))
    #print(lj)
    axejours(lj)
    for (courbe,nom,t) in courbes:
        lv = val(lj,courbe)
        k,v = [(k,lv[k]) for (k,j) in enumerate(lj) if lv[k] != None][-1]
        if t != '=':
            plt.plot(lv,t)
        else:
            plt.plot(lv,'-', linewidth = 2)
        if nom != '':
            if xlabel == 0:
                plt.text(k,lv[k],nom,fontdict = {'size':8})
            else:
                plt.text(xlabel,lv[xlabel],nom,fontdict = {'size':8})

def trace(lcourbes,titre,fichier,xlabel = 0):
    plt.clf()
    plotcourbes(lcourbes,xlabel=xlabel)
    plt.grid()
    plt.title(titre,fontdict = {'size':10})
    plt.savefig(fichier + '.pdf', dpi=600)
    plt.savefig(fichier + '.png', dpi=600)
    plt.show(False)

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

memo_num_de_jour = {}
def num_de_jour(jour): # depuis le 1 janvier 2000: num 0
    if jour in memo_num_de_jour:
        return(memo_num_de_jour[jour])
    n = 0
    j = '2000-01-01'
    while j != jour:
        memo_num_de_jour[j] = n
        n += 1
        j = nextday(j)
    memo_num_de_jour[jour] = n
    return(n)

jour_de_num = ['']*10000
j = '2000-01-01'
for n in range(len(jour_de_num)):
    jour_de_num[n] = j
    j = nextday(j)

    
def table(t):
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
