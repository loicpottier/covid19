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
import re

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
aujourdhui = (str(now.tm_year) + '-'
              + (str(now.tm_mon) if now.tm_mon > 9
                 else '0' + str(now.tm_mon))
              + '-'
              + (str(now.tm_mday) if now.tm_mday > 9
                 else '0' + str(now.tm_mday)))
aujourdhuih = (str(now.tm_year) + '-'
               + (str(now.tm_mon) if now.tm_mon > 9
                  else '0' + str(now.tm_mon))
               + '-'
               + (str(now.tm_mday) if now.tm_mday > 9
                  else '0' + str(now.tm_mday))
               + 'T'
               + (str(now.tm_hour) if now.tm_hour > 9
                  else '0' + str(now.tm_hour))
               + ':'
               + (str(now.tm_min) if now.tm_min > 9
                  else '0' + str(now.tm_min))
               + ':'
               + (str(now.tm_sec) if now.tm_sec > 9
                  else '0' + str(now.tm_sec)))

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
def derivee(l, largeur = 1): # largeur impaire
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

nomsmois = ['janvier','février','mars','avril','mai','juin', 'juillet','août','septembre','octobre','novembre','décembre']
def joli2(jour):
    if type(jour) is str:
        j = jour[8:10]
        m = jour[5:7]
        a = jour[:4]
        return(j + ' ' + nomsmois[int(m) - 1] + ' ' + a)
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
        return(zipper(lj[:min(len(lj),len(lv))],
                      lv[:min(len(lj),len(lv))]))
    return([(lj[k],lv[k]) for k in range(min(len(lv),len(lj)))])

def loadcsv(file, sep = r';', end = '\n'):
    f = open(file,'r')
    s = f.read()
    if '\r\n' in s:
        end = '\r\n'
    data = [x.split(sep) for x in s.split(end)]
    return(data)

def chargecsv(url, zip = False, sep = r';', end = '\n'):
    s = urltexte(url, zip = zip)
    if '\r\n' in s:
        end = '\r\n'
    data = [re.split(sep,x) for x in s.split(end)]
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
######################################################################
# google charts
import hashlib

# new Date(year, month[, date[, hours[, minutes[, seconds[, milliseconds]]]]]);
def newDate(x):
    if 'T' not in x: #2021-12-08
        a,m,j = x.split('-')
        return('new Date(' + a + ',' + str(int(m)-1) + ',' + str(int(j)-1) + ')')
    else: #2021-12-08T19:24:52
        #print(x)
        x1,x2 = x.split('T')
        a,m,j = x1.split('-')
        h,mn,s = x2.split(':')
        #exit()
        return('new Date(' + a + ',' + str(int(m)-1) + ',' + str(int(j)-1)
               + ',' + h + ',' + mn + ',' + s
               + ')')
        #return('new Date(\'' + x + '\')')

def entier(x):
    try:
        a = int(float(x))
        if a < 50:
            a = float(x)
    except:
        try:
            a = newDate(x)
        except:
            a = x
    return(a)

def arronditable(t):
    return([[(int(10000*x)/10000 if type(x) == float else x) for x in y] for y in t])

def trace_chart(table,titre,
                width = 800, #900
                height = 600, #700
                options = None,
                area = False): # table commence par les noms des colonnes
    id = hashlib.md5(str(table).encode("utf-8")).hexdigest()
    #ligne1 = ["{label: '" + table[0][0] + "', type: 'string'}"] + ["{label: '" + x + "', type: 'number'}"
    #                                                               for x in table[0][1:]]
    ligne1 = table[0]
    table = [ligne1] + [[newDate(line[0]) if type(line[0]) is str else line[0] ]
                        + [entier(x) for x in line[1:]]
                        for line in table[1:]]
    table = arronditable(table)
    #print(ligne1)
    #print(table[100])
    options_base = '''
              curveType: 'function', //
              legend: { position: 'bottom' },'''
    if options == None:
        options = options_base
    else:
        options = options_base + options
    script = ('''
        var data = google.visualization.arrayToDataTable('''
              + str(table).replace("'null'",'null').replace("'new ",'new ').replace('"new ','new ').replace(")'",')').replace(')"',')').replace('],','],\n')
              + ''');

        data.setColumnProperty(0, 'type', 'date');

        var options = {
              title: '''
              + "'" + titre + "'," + options + '''
        };
        var chart = new google.visualization.'''
              + ('AreaChart' if area else 'LineChart')
              + '(document.getElementById('
              + "'" + id + "'"
              + '''));

        chart.draw(data, options);
    ''')
    div = '<div id="' + id + '" style="width: ' + str(width) + 'px; height: ' + str(height) + 'px"></div>'
    #print(id)
    return({'id':id, 'script': script,'div': div})

def plotcourbes_charts(courbes,titre = '',options = None, area = False):
    lj = []
    for (courbe,nom,t) in courbes:
        lj = lj + [x[0] for x in courbe]
    lj = sorted(list(set(lj))) # liste des jours en abcisse
    table = ([['jour'] + [nom for (courbe,nom,t) in courbes]]
             + [[j] + ['null' for k in range(len(courbes))]
                for j in lj])
    for (c,(courbe,nom,t)) in enumerate(courbes):
        lv = val(lj,courbe)
        for k in range(len(lj)):
            table[1+k][1+c] = lv[k] if lv[k] not in [np.nan,np.inf,None] else 'null'
    # enlever les colonnes de null, ca plante google charts
    nulles = []
    for c in range(len(courbes)):
        nulle = True
        for l in table[1:]:
            if l[c] != 'null':
                nulle = False
        if nulle:
            nulles.append(c)
    for c in nulles[::-1]:
        for l in range(len(table)):
            table[l] = table[l][:c] + table[l][c+1:]
    return(trace_chart(table,titre,options = options, area = area))

# rend un dico {'id':id, 'script': script,'div': div}
def trace_charts(dirprev, lcourbes,titre = '',options = None, area = False):
    res = plotcourbes_charts(lcourbes,titre = titre,options = options, area = area)
    f = open(dirprev + 'charts/' + res['id'] + '.html', 'w')
    f.write(''' 
<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
<script type="text/javascript">
      google.charts.load('current', {'packages':['corechart'], 'language': 'fr'});
      google.charts.setOnLoadCallback(drawChart);

      function drawChart() {
      ''' + res['script'] + '''
      }
 </script>''' + '\n' + res['div'])
    f.close()
    res['href'] = 'https://cp.lpmib.fr/medias/covid19/charts/' + res['id'] + '.html'
    return(res)

######################################################################
# html
debut1 = '''<!DOCTYPE html><meta charset="UTF-8">
<head>
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-165293312-1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-165293312-1');
</script>
<script src="https://code.jquery.com/jquery-3.4.1.slim.min.js" integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js" integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" crossorigin="anonymous"></script>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
<style>
.tab-content>.tab-pane {
  height: 1px;
  overflow: hidden;
  display: block;
 visibility: hidden;
}
.tab-content>.active {
  height: auto;
  overflow: auto;
  visibility: visible;
}
</style>
</head>
<body>
<div class="container-fluid">
 '''

fin1 = '''
</div>
</body>
'''

# heatmap, bof, la valeur c est que la chaleur de la case, j'en voudrais une autre
# pas fini
def heatmap_charts(heattable,titre = '', options = None):
    id = hashlib.md5(str(table).encode("utf-8")).hexdigest()
    script = '''
<script src="https://cdn.anychart.com/releases/8.9.0/js/anychart-core.min.js"></script>
<script src="https://cdn.anychart.com/releases/8.9.0/js/anychart-heatmap.min.js"></script>

var data = [''' + ','.join(['{x: "' + x + '", y: "' + y + '", heat: ' + str(v) + '}'
                            for(x,y,v) in heattable]) + '''
];

// create a chart and set the data
chart = anychart.heatMap(data);

// set the container id
chart.container("container");

// initiate drawing the chart
chart.draw();'''


def normalise_nom(x):
    for c in '.,éèêàâôûù;-, ':
        x = x.replace(c,'')
    return(x)

######################################################################
# navbar avec menus dropdown

def idify(s):
    for c in [' ','\t','\n',"'",'"','-','(',')']:
        s = s.replace(c,'_')
    return(s)

def navbar_dropdown_n(id,nom,ll):
    r = '''
<script src="https://code.jquery.com/jquery-3.5.0.js"></script>
<script type="text/javascript">
$(function(){
'''
    for (i,(nomliste,liste)) in enumerate(ll):
        for (nomitem,url) in liste:
            r += '''
    $("#''' + idify(id + str(i) + nomitem) + '''").click(function(e){
        e.preventDefault(); //To prevent the default anchor tag behaviour
        var url = this.href;
        $.get(url, function(data){ 
	    $("#lechartamontrer''' + idify(id) + '''").html(data);
	    });
    });'''
    r += '''
});'''
    r += '''
</script>
<nav class="navbar navbar-expand-lg navbar-light bg-light">
  <a class="navbar-brand" href="#">''' + nom + '''</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#''' + idify(id) + '''" aria-controls="''' + idify(id) + '''" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="''' + idify(id) + '''">
    <ul class="navbar-nav">'''
    for (i,(nomliste,liste)) in enumerate(ll):
        if len(liste) == 1:
            nomitem,url = liste[0]
            r += '''
       <li class="nav-item">
        <a id="''' + idify(id + str(i) + nomitem) + '''" class="nav-link" href="''' + url + '''">''' + nomliste + '''</a>
      </li>'''
        else:
            r += '''
      <li class="nav-item dropdown">
        <a class="nav-link dropdown-toggle" href="#" id="''' + idify(id + str(i)) + '''" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
          ''' + nomliste + '''
        </a>
        <div id="''' + idify(id + str(i)) + '''" class="dropdown-menu" aria-labelledby="''' + idify(id + str(i)) + '''">'''
            for (nomitem,url) in liste:
                r += '''<a id="''' + idify(id + str(i) + nomitem) + '''" class="dropdown-item" href="''' + url + '''">''' + nomitem + '''</a>'''
            r += '''
        </div>
      </li>'''
    r += '''
    </ul>
  </div>
</nav>
<div id="lechartamontrer''' + idify(id) + '''"></div>
'''
    return(r)

######################################################################
# navbar avec un seul menu dropdown

def navbar_dropdown_1(id,nom,nomdrop,liste):
    r = '''
<script src="https://code.jquery.com/jquery-3.5.0.js"></script>
<script type="text/javascript">
$(function(){
'''
    for (nomitem,url) in liste:
        r += '''
    $("#''' + idify(id + '0' + nomitem) + '''").click(function(e){
        e.preventDefault(); //To prevent the default anchor tag behaviour
        var url = this.href;
        $.get(url, function(data){ 
	    $("#lechartamontrer''' + idify(id) + '''").html(data);
	    });
    });'''
    r += '''
});'''
    r += '''
</script>
<nav class="navbar navbar-expand-lg navbar-light bg-light">
  <a class="navbar-brand" href="#">''' + nom + '''</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#''' + idify(id) + '''" aria-controls="''' + idify(id) + '''" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="''' + idify(id) + '''">
    <ul class="navbar-nav">'''
    r += '''
    <li class="nav-item dropdown">
      <a class="nav-link dropdown-toggle" href="#" id="''' + idify(id + '0') + '''" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
        ''' + nomdrop + '''
      </a>
      <div id="''' + idify(id + '0') + '''" class="dropdown-menu" aria-labelledby="''' + idify(id + '0') + '''">'''
    for (j,(nomitem,url)) in enumerate(liste):
        r += '''<a id="''' + idify(id + '0' + nomitem) + '''" class="dropdown-item" href="''' + url + '''">''' + nomitem + '''</a>'''
    r += '''
        </div>
      </li>'''
    r += '''
    </ul>
  </div>
</nav>
<div id="lechartamontrer''' + idify(id) + '''"></div>
'''
    return(r)

def navbar_dropdown(id,nom,ll):
    if all([len(l) == 1 for x,l in ll]):
        return(navbar_dropdown_1(id,nom,ll[0][1][0][0],
                                 [(x,l[0][1]) for x,l in ll]))
    else:
        return(navbar_dropdown_n(id,nom,ll))

'''
print(navbar_dropdown('idtest','Navbartest', [('france',[['hosp','test'],['rea','testrea'],['rea','testrea']]),
                                              ('UK',[['hosp','test2'],['rea','testrea2'],['rea','testrea2']])]))
'''

# rend une barre de tabs avec leurs contenus
def tabs(lt):
    t = str(time.time()).replace('.','')
    r = '<ul class="nav nav-tabs" id="myTab" role="tablist">'
    for (k,(nom,contenu)) in enumerate(lt):
        nom2 = normalise_nom(nom) + t
        r += ('<li class="nav-item"><a class="nav-link' + (' active' if k == 0 else '')
              + '" id="'
              + nom2 + '-tab" data-toggle="tab" href="#'
              + nom2 + '" role="tab" aria-controls="'
              + nom2 + '" aria-selected="true">'
              + nom + '</a></li>')
    r += '</ul>'
    r += '<div class="tab-content">'
    for (k,(nom,contenu)) in enumerate(lt):
        nom2 = normalise_nom(nom) + t
        r += ('<div id="'
              + nom2 + '" class="tab-pane fade show' + (' active' if k == 0 else '')
              + '">'
              + contenu
              + '</div>')
    r += '</div>'
    return(r)

def table2(l):
    r = '<table class="table">'
    for x in l:
        r += ('<tr valign=top>')
        for y in x:
            r += '<td valign = top>'
            r += y
            r += '</td>'
        r += '</tr>'
    r += '</table>'
    return(r)

def table2h(l):
    r = '<table class="table table-sm"><thead><tr>'
    x = l[0]
    for y in x:
        r += '<th scope ="col">'
        r += y
        r += '</th>'
    r += ('</tr></thead><tbody>')
    for x in l[1:]:
        r += ('<tr valign=top>')
        for y in x:
            r += '<td valign = top>'
            r += y
            r += '</td>'
        r += '</tr>'
    r += '</tbody></table>'
    return(r)
