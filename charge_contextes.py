from outils import *

######################################################################
# frequentations des lieux
# https://www.google.com/covid19/mobility/
# les commerces et espaces de loisirs, les magasins d'alimentation et pharmacies, les parcs, les arrêts de transports en commun, les lieux de travail et les lieux de résidence.
# pourcentages de changement par rapport à la valeur de référence
# pour chaque département

response = urllib.request.urlopen('https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip')
encoding = response.info().get_content_charset() or "utf-8"
data = response.read()      # a `bytes` object
f = open('Region_Mobility_Report_CSVs.zip','wb')
f.write(data)
f.close()
import zipfile
with zipfile.ZipFile('Region_Mobility_Report_CSVs.zip', 'r') as zip_ref:
    zip_ref.extractall('mobilite_google')

f = open('mobilite_google/2020_FR_Region_Mobility_Report.csv','r')
s = f.read()
ls = [x.split(',') for x in s.split('\n')]
if ';' in ls[0][0]: # des fois le séparateur change!!!!! fait chier google
    ls = [x.split(';') for x in s.split('\n')]
elif '\t' in ls[0][0]:
    ls = [x.split('\t') for x in s.split('\n')]

f.close()

datalieux = []
departements = []

for d in range(1,96):
    dep = str(d)
    if d < 10:
        dep = '0' + str(d)
    ld = [x[7:] for x in ls if len(x) >= 8 and x[5] == 'FR-'+dep]
    ld = sorted(ld,key = lambda x: x[0])
    ld = [[x[0]] + [int(y) if y != '' else -100 for y in x[1:]]
          for x in ld]
    # on extrapole les données manquantes (valeur -100)
    for j in range(len(ld)):
        for k in range(1,7):
            if ld[j][k] == -100:
                suiv = 0
                jsuiv = j
                try:
                    for u in range(j+1,len(ld)):
                        if ld[u][k] != -100:
                            suiv = ld[u][k]
                            jsuiv = u
                            raise NameError('ok')
                except:
                    pass
                prev = 0
                jprev = j
                try:
                    for u in range(0,j-1):
                        if ld[u][k] != -100:
                            prev = ld[u][k]
                            jprev = u
                            raise NameError('ok')
                except:
                    pass
                ld[j][k] = ((jsuiv - j)*suiv+ (j-jprev)*prev)/(jsuiv-jprev)
    if len(ld) == 0:
        pass #print(dep, 'données vides')
    else:
        datalieux.append(ld)
        departements.append(d)

# on vire les départements avec des jours qui manquent
lmax = mmax([len(x) for x in datalieux])
departements = [d for (k,d) in enumerate(departements)
                if len(datalieux[k]) == lmax]
datalieux = [x for x in datalieux if len(x) == lmax]

datamobilite = {'nom': 'mobilite',
                'titre': 'fréquentation des lieux',
                'dimensions': ['departements','jours','mobilites'],
                'departements': departements,
                'jours': [x[0] for x in datalieux[1]],
                'mobilites': ['commerces et espaces de loisir (dont restaurants et bars)',
                              "magasins d'alimentation et pharmacies",
                              'parcs',
                              'arrêts de transports en commun',
                              'travail',
                              'résidence']
                }

datamobilite['valeurs'] = np.array([[x[1:] for x in d]
                                    for d in datalieux])
# pourcentage de la valeur moyenne
datamobilite['valeurs'] = datamobilite['valeurs'] + 100

print('mobilité de google ok', datamobilite['jours'][-1])

######################################################################
# données meteo
# https://www.data.gouv.fr/fr/datasets/donnees-d-observation-des-principales-stations-meteorologiques/
# j'ai mis les départements à la louche
stations ={
    '07005':[27,60,76,80], #ABBEVILLE somme
    '07015':[8,59,62], #LILLE-LESQUIN
    '07020':[50], #PTE DE LA HAGUE
    '07027':[14], #CAEN-CARPIQUET
    '07037':[], #ROUEN-BOOS
    '07072':[2,51], #REIMS-PRUNAY
    '07110':[29], #BREST-GUIPAVAS
    '07117':[22], #PLOUMANAC'H
    '07130':[35,53,56], #RENNES-ST JACQUES
    '07139':[61,72], #ALENCON
    '07149':[28,75,77,78,91,92,93,94,95], #ORLY
    '07168':[10,89], #TROYES-BARBEREY
    '07181':[54,55,57,88], #NANCY-OCHEY
    '07190':[67,68], #STRASBOURG-ENTZHEIM
    '07207':[], #BELLE ILE-LE TALUT
    '07222':[44,49], #NANTES-BOUGUENAIS
    '07240':[37,41], #TOURS
    '07255':[18,45,58], #BOURGES
    '07280':[21,39,52], #DIJON-LONGVIC
    '07299':[25,70,90], #BALE-MULHOUSE
    '07314':[17,85], #PTE DE CHASSIRON ile d oleron
    '07335':[36,86], #POITIERS-BIARD
    '07434':[16,79,87], #LIMOGES-BELLEGARDE
    '07460':[15,19,23,63], #CLERMONT-FD
    '07471':[3], #LE PUY-LOUDES
    '07481':[1,38,42,69,71], #LYON-ST EXUPERY
    '07510':[24,33,47], #BORDEAUX-MERIGNAC
    '07535':[], #GOURDON
    '07558':[48], #MILLAU
    '07577':[7,26,30,43], #MONTELIMAR
    '07591':[4,5,73,74], #EMBRUN
    '07607':[40,64,65], #MONT-DE-MARSAN
    '07621':[32,31,81], #TARBES-OSSUN
    '07627':[9], #ST GIRONS pyrennees milieu
    '07630':[12,46,82], #TOULOUSE-BLAGNAC
    '07643':[11,34,66], #MONTPELLIER
    '07650':[13,84], #MARIGNANE
    '07661':[83], #CAP CEPET toulon
    '07690':[6], #NICE
    '07747':[], #PERPIGNAN
    '07761':[], #AJACCIO
    '07790':[] #BASTIA
}
def station_de_dep(d):
    return([s for s in stations if d in stations[s]][0])

# novembre:
# https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/synop.202011.csv.gz
data = []
for mois in range(1,36):
    if mois <= 12:
        annee = '2019'
    elif mois <= 24:
        annee = '2020'
        mois = mois - 12
    else:
        annee = '2021'
        mois = mois - 24
    print(annee,mois,end = ' ', flush=True)
    smois = str(mois)
    if mois <10:
        smois = '0' + smois
    try:
        d = chargecsv('https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/synop.'
                      + annee + smois + '.csv.gz', zip = True)
        if annee == '2019' and mois == 1:
            data = d
        else:
            data = data + d[1:]
    except:
        pass
    #print(smois,'fait')

param = data[0] # les bons des parametres meteo mesures
# explication ici: https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=90&id_rubrique=32

#nomsmeteo = ['pression','humidité','précipitations sur 24h','température','nébulosité', 'vent']
#imeteo = [param.index(x) for x in ['pres','u','rr24','t','n','ff']]

nomsmeteo = ['pression','humidité','précipitations sur 24','température', 'vent']
imeteo = [param.index(x) for x in ['pres','u','rr24','t','ff']]

#nomsmeteo = ['humidité']
#imeteo = [param.index(x) for x in ['u']]

# les donnees manquantes 'mq' sont remplacees par -100 et les autres converties en float
def valmesure(x):
    try:
        return(float(x))
    except:
        return(-100)

# tm[(d,j)] = liste des mesures durant le jour j du departement d,
# une mesure = liste des valeurs des parametres
tm = {}
for x in data[1:]:
    try:
        j = x[1]
        station = x[0]
        if station[0:2] == '07': # metropole
            deps = stations[station]
            jour = j[:4] + '-' + j[4:6] + '-' + j[6:8]
            for dep in deps:
                dx = [valmesure(x[p]) for p in imeteo]
                if (dep,jour) in tm:
                    tm[(dep,jour)].append(dx)
                else:
                    tm[(dep,jour)] = [dx]
    except:
        pass

deps = sorted(list(set([x[0] for x in tm])))
#deps.remove(81)
depsd = dict([(d,deps.index(d)) for d in deps])

jours = sorted(list(set([x[1] for x in tm])))
joursd = dict([(j,jours.index(j)) for j in jours])

t = np.zeros((len(deps),len(jours),len(imeteo))) - 100

for (d,j) in tm:
    if d in deps:
        lx = tm[(d,j)]
        lv = [[] for k in range(len(imeteo))]
        for x in lx:
            for k in range(len(imeteo)):
                if x[k] != -100:
                    lv[k].append(x[k])
        for k in range(len(imeteo)):
            if lv[k] == []:
                t[depsd[d],joursd[j],k] = -100
            else:
                t[depsd[d],joursd[j],k] = moyenne(lv[k])

# maintenant il faut extrapoler les valeurs manquantes...

def extrapole_manquantes(t,deps,jours,nomvaleurs):
    deppb = []
    mespb = []
    for d in range(len(deps)):
        for j in range(len(jours)):
            for k in range(len(nomvaleurs)):
                if t[d,j,k] == -100:
                    suiv = 0
                    jsuiv = j
                    try:
                        for u in range(j+1,len(jours)):
                            if t[d,u,k] != -100:
                                suiv = t[d,u,k]
                                jsuiv = u
                                raise NameError('ok')
                    except:
                        pass
                    prev = 0
                    jprev = j
                    try:
                        for u in range(0,j-1):
                            if t[d,u,k] != -100:
                                prev = t[d,u,k]
                                jprev = u
                                raise NameError('ok')
                        if deps[d] not in deppb:
                            deppb.append(deps[d])    
                    except:
                        pass
                    if jsuiv != j and jprev != j:
                        t[d,j,k] = ((jsuiv - j)*suiv+ (j-jprev)*prev)/(jsuiv-jprev)
                    else:
                        if deps[d] not in deppb:
                            deppb.append(deps[d])
                        if nomvaleurs[k] not in mespb:
                            mespb.append(nomvaleurs[k])
    deps1 = deps[:]
    for d in deppb:
        deps1.remove(d)
    t1 = np.array([t[deps.index(d)] for d in deps1])
    return(t1,deps1)

t1, deps1 = extrapole_manquantes(t,deps,jours,nomsmeteo)

datameteo = {'nom': 'meteo',
              'titre': 'mesures meteo (stations principales)',
              'dimensions': ['departements','jours','meteo'],
              'departements': deps1,
              'jours': jours,
              'meteo': nomsmeteo,
              'valeurs': t1
}

# on ajoute la dérivée de la température lissée
datameteo['meteo'].append('variations de température')
ctemp = datameteo['meteo'].index('température')
t = datameteo['valeurs']
ndeps,njours,ncont = np.shape(t)
t1 = np.zeros((ndeps,njours, ncont + 1))
t1[:,:,:-1] = t
for d in range(ndeps):
    dt = derivee(lissage(t1[d,:,ctemp],11))
    t1[d,:,ncont] = dt

datameteo['valeurs'] = t1

print('météo ok',jours[-1])

######################################################################
# vacances

zones = {
    "A": [3,15,43,63,7,26,38,73,74,1,42,69,25,39,70,90,21,58,71,89,24,33,40,47,64,19,23,87,16,17,79,86],
    "B":[22,29,35,56,18,28,36,37,41,45,2,54,55,57,88,8,10,51,52,67,68,2,60,80,59,62,14,50,61,27,76,44,49,53,72,85,4,5,13,84,6,83],
    "C":[77,93,94,75,78,91,92,95,11,30,34,48,66,9,12,31,32,46,65,81,82]}

# périodes de vacances selon les zones
# ne pas oublier de mettre a jour le debut et la fin dans le calcul de jours, plus bas
vacances = {
    "A": [('2019-12-21','2020-01-05'),
          ('2020-02-22','2020-03-08'),('2020-03-17','2020-05-11'),('2020-05-12','2020-07-03'),
          ('2020-07-04','2020-09-01'),('2020-10-17','2020-11-01'),('2020-11-16','2020-12-18'),
          ('2020-12-19','2021-01-03'),('2021-01-04','2021-01-19'),('2021-01-20','2021-02-05'),
          ('2021-02-06','2021-02-21'),
          ('2021-04-10','2021-04-25'),
          ('2021-05-13','2021-05-16'),('2021-05-17','2021-07-05'),('2021-07-07','2021-09-01')],

    "B": [('2019-12-21','2020-01-05'),
          ('2020-02-15','2020-03-01'),('2020-03-17','2020-05-11'),('2020-05-12','2020-07-03'),
          ('2020-07-04','2020-09-01'),('2020-10-17','2020-11-01'),('2020-11-16','2020-12-18'),
          ('2020-12-19','2021-01-03'),('2021-01-04','2021-01-19'),('2021-01-20','2021-02-19'),
          ('2021-02-20','2021-03-07'),
          ('2021-04-24','2021-05-09'),
          ('2021-05-13','2021-05-16'),('2021-05-17','2021-07-05'),('2021-07-07','2021-09-01')],

    "C": [('2019-12-21','2020-01-05'),
          ('2020-02-08','2020-02-24'),('2020-03-17','2020-05-11'),('2020-05-12','2020-07-03'),
          ('2020-07-04','2020-09-01'),('2020-10-17','2020-11-01'),('2020-11-16','2020-12-18'),
          ('2020-12-19','2021-01-03'),('2021-01-04','2021-01-19'),('2021-01-20','2021-02-12'),
          ('2021-02-13','2021-02-28'),
          ('2021-04-17','2021-05-02'),
          ('2021-05-13','2021-05-16'),('2021-05-17','2021-07-05'),('2021-07-07','2021-09-01')]
}
vvacances = [100,
             100,100,80,
             100,100,50,
             100,50,50, # pas de déconfinement le 20 janvier
             100, # apres fevrier plein effectif a l ecole
             100,
             100,0,100]

jours = [jour_de_num[k]
         for k in range(num_de_jour('2019-12-21'), # mettre a jour 
                        num_de_jour('2021-09-01')+1)] # mettre a jour

def joursvacances(d):
    z = [z for z in zones if d in zones[z]][0]
    t = np.array([0 for x in jours])
    n0 = num_de_jour(jours[0])
    for (k,(jd,jf)) in enumerate(vacances[z]):
        nd = num_de_jour(jd)
        nf = num_de_jour(jf)
        #print(nd,nf)
        for n in range(nd,nf+1):
            t[n-n0] = vvacances[k]
    return(t)

deps = [d for d in range(1,96) if d != 20]
datavacances = {'nom': 'vacances',
                'titre': 'vacances scolaires en France',
                'dimensions': ['departements','jours','vacances'],
                'departements': deps,
                'jours': jours,
                'vacances': ['vacances'],
                'valeurs': np.array([[[x] for x in joursvacances(d)] for d in deps])
}

print('vacances ok',jours[-1])

######################################################################
# données mobilité Apple:

# https://covid19.apple.com/mobility
# https://covid19-static.cdn-apple.com/covid19-mobility-data/2022HotfixDev15/v3/en-us/applemobilitytrends-2020-12-05.csv
# https://covid19-static.cdn-apple.com/covid19-mobility-data/2022HotfixDev15/v3/en-us/applemobilitytrends-2020-12-05.csv
#f = open('applemobilitytrends-2020-12-05.csv','r')
#s = f.read()
#f.close()

now = time.localtime(time.time())
def sdep(d):
    dep = str(d)
    if d < 10:
        dep = '0' + str(d)
    return(dep)

j = str(now.tm_year) + '-' + str(now.tm_mon) + '-' + sdep(now.tm_mday)

s = chargejson('https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json',zip = False)
basePath = s['basePath']
csvPath = s['regions']['en-us']['csvPath']
data = None
while data == None and j != '2020-11-01':
    try:
        #print(j)
        # attention cette url change régulièrement,
        # il faut inspecter le reseau dans le navigateur pour voir cette url...
        # ou charger le json ci-dessus! si son url ne change pas...
        # (trouvé dans le source javascript .js  avec le lien dans la page https://covid19.apple.com/mobility)
        #url = 'https://covid19-static.cdn-apple.com/covid19-mobility-data/2023HotfixDev9/v3/en-us/applemobilitytrends-' + j + '.csv'
        url = 'https://covid19-static.cdn-apple.com' + basePath + csvPath
        #print(url)
        data = chargecsv(url,zip = False,sep = ',')
        #print('data ok')
    except:
        j = jour_de_num[num_de_jour(j)-1]

jours = data[0][6:]
set([x[0] for x in data if 'France' in x]) #{'country/region', 'sub-region', 'city'}

vapple = ['driving', 'walking', 'transit'] 
lf = [x[:6] + [valmesure(y) for y in x[6:]]
      for x in data[1:]
      if len(x) > 5 and x[0] == 'sub-region' and x[5] == 'France' and x[2] in vapple]

# les 22 anciennes régions et leurs départements
lr = set([x[1] for x in lf])
depreg_apple = {"Provence-Alpes-Côte d'Azur Region": [4,6,13,5,83,84],
                'Alsace Region': [67,68],
                'Aquitaine Region': [24,33,40,47,64],
                'Auvergne Region': [3,15,43,63],
                'Brittany Region': [22,29,35,56],
                'Burgundy Region': [21,58,71,89],
                'Centre Region' : [18,28,36,37,41,45],
                'Champagne-Ardenne Region': [8,10,51,52],
                'Corsica Region': [],
                'Franche-Comté Region': [25,70,39,90],
                'Languedoc-Roussillon': [11,30,34,48,66],
                'Limousin Region': [19,23,87],
                'Lorraine Region': [54,55,57,88],
                'Lower Normandy Region': [14,50,61],
                'Midi-Pyrénées Region': [9,12,32,31,65,46,81,82],
                'Nord-Pas-de-Calais Region': [59,62],
                'Pays de la Loire Region': [44,49,53,72,85],
                'Picardy Region': [2,60,80],
                'Poitou-Charentes Region': [16,17,79,86],
                'Rhône-Alpes Region': [1,7,26,38,42,69,73,74],
                'Upper Normandy Region': [27,76],
                'Île-de-France Region': [91,92,75,93,77,94,95,78],
}
# rajouter les valeurs de transit pour les regions qui en manquent (Normandie, etc):
# on met la moyenne des autres régions
lrtransit = []
lrmanque = []
for r in depreg_apple:
    lx = [x for x in lf if x[1] == r]
    if len(lx) <= 2:
        lrmanque.append((r,lx[0]))
    else:
        x = [x for x in lx if x[2] == 'transit'][0]
        lrtransit.append(np.array(x[6:]))

mt = np.mean(lrtransit, axis = 0)

for (r,x) in lrmanque:
    lf = lf + [x[0:2] + ['transit'] + x[3:6] + list(mt)]

deps = []
for r in depreg_apple:
    deps = deps + depreg_apple[r]

deps = list(set(deps))
t = np.zeros((len(deps),len(jours),len(vapple))) - 100

for x in lf:
    rx = x[1]
    lvx = x[6:]
    vx = vapple.index(x[2])
    ld = depreg_apple[rx]
    for dep in ld:
        d = deps.index(dep)
        for j in range(len(jours)):
            t[d,j,vx] = x[6+j]

#plt.clf();plt.plot(np.transpose(t[0:5,:,0]));plt.show(False)

t1, deps1 = extrapole_manquantes(t,deps,jours,vapple)

dataapple = {'nom': 'apple',
             'titre': 'mobilité apple',
             'dimensions': ['departements','jours','apple'],
             'departements': deps1,
             'jours': jours,
             #['driving', 'walking', 'transit']
             'apple': ['en voiture','à pied', 'en transport en commun'], 
             'valeurs': t1
}

print('mobilité apple ok',jours[-1])

######################################################################
# données de comportements
# https://www.data.gouv.fr/fr/datasets/donnees-denquete-relatives-a-levolution-des-comportements-et-de-la-sante-mentale-pendant-lepidemie-de-covid-19-coviprev/


# les comportements
# https://www.data.gouv.fr/fr/datasets/r/425c285e-532e-46a5-bcaa-9ba6d191d8be
ls = chargecsv('https://www.data.gouv.fr/fr/datasets/r/425c285e-532e-46a5-bcaa-9ba6d191d8be')

# description des champs: https://www.data.gouv.fr/fr/datasets/r/226c2104-61cb-4bfb-8a38-a56d5386377e
data = [[[x[ls[0].index('semaine')],
          x[ls[0].index('hyg4mes')],
          x[ls[0].index('nbmoy4mesHyg')],
          x[ls[0].index('dist1m')],
          x[ls[0].index('portmasque')]]
         for x in ls[1:] if int(x[0]) == r]
        for (r,dr) in lregions]

vagues = [num_de_jour(x) for x in ['2020-05-13', '2020-05-18', '2020-05-27', '2020-06-08',
                                   '2020-06-22', '2020-07-06', '2020-07-20', '2020-08-24',
                                   '2020-09-21', '2020-10-19']]

jours = [jour_de_num[x] for x in range(vagues[0],vagues[-1]+1)]

def approx_lin(lx,ly,z):
    if z <= lx[0]:
        return(ly[0])
    if z >= lx[-1]:
        return(ly[-1])
    for i in range(len(lx)-1):
        if lx[i] <= z and z < lx[i+1]:
            return(ly[i] + (z-lx[i])/(lx[i+1]-lx[i]) * (ly[i+1]-ly[i]))

nparam = len(data[0][0])-1
deps = [x for x in depregion]
data1 = np.zeros((len(deps), vagues[-1] - vagues[0] + 1, nparam))


for (r,(reg,depr)) in enumerate(lregions):
    lval = [[float(v.replace(',' ,'.')) if v != '' else -100
             for v in x[1:]]
            for x in data[r][6:]]
    for k in range(nparam):
        ly = [v[k] for v in lval]
        for d in depr:
            for j in range(vagues[0],vagues[-1]+1):
                data1[deps.index(d),j-vagues[0],k] = approx_lin(vagues,ly,j)

datahygiene = {'nom': 'hygiene',
               'titre': "hygiène pendant l'épidémie",
               'dimensions': ['departements','jours','hygiene'],
               'departements': deps,
               'jours': jours,
               'hygiene': ['hyg4mes: Se laver régulièrement les mains ; Saluer sans serrer la main et arrêter les embrassades ; Tousser dans son coude ; Utiliser un mouchoir à usage unique',
                           'nbmoy4mesHyg: Nombre moyen et évolutions régionales des mesures d’hygiène systématiquement adoptées parmi les 4 mesures recommandées',
                           'dist1m: Adoption systématique de la distance d’un mètre',
                           'portmasque: port du masque'],
               'valeurs': np.array(data1)
}

print('hygiène sociale ok',jours[-1])

######################################################################
# requetes google trends
# pytrends
# modif de request: https://stackoverflow.com/questions/57218531/modulenotfounderror-no-module-named-pandas-io-for-json-normalize réponse 4 sinon pytrends merde
# doc: https://github.com/GeneralMills/pytrends

import pandas as pd
from pytrends.request import TrendReq
import time

departements = ['0'+str(x) for x in range(1,10)] + [str(x) for x in range(10,95) if x != 20]

pytrends = TrendReq(hl='fr-FR', tz=60)
# en cas de limite atteinte:
# pytrends = TrendReq(hl='fr-FR', tz=60, timeout=(10,25), proxies=['https://34.203.233.13:80',], retries=2, backoff_factor=0.1, requests_args={'verify':False})

def charge_trends_dep(d,key,keywords,data,delai0):
    delai = delai0
    ok = False
    while not ok:
        try:
            time.sleep(delai)
            pytrends.build_payload(
             kw_list=keywords,
             cat=0,
             timeframe='today 12-m',
             geo='FR-' + d,
             gprop='')
            t = pytrends.interest_over_time()
            ok = True
            s = t.to_csv()
            f = open('_trends.csv','w')
            f.write(s)
            f.close()
            t = [x.split(',') for x in s.split('\n')][1:-1]
            t = [(x[0],sum([int(y) for y in x[1:-1]])) for x in t]
            data[d] = t
            print('ok', end = ' ', flush = True)
        except:# si jamais on dépassé le nombre de requete google trends, genre 1400 en 4h
            delai = 60
            print('nombre de requete google trends apparemment dépassé: delai = '
                  + str(delai), flush = True)
    return(delai)

def charge_trends(keywords):
    delai = 0
    key = keywords[0].replace(' ','')[:10]
    # enlever False pour mettre a jour
    if False: #key in ['horaires','voyage','itinéraire']: # True si on veut les valeurs les plus récentes
        data = {}
        for d in departements:
            print(d, end = ' ', flush = True)
            delai = charge_trends_dep(d,key,keywords,data,delai)
        f = open('google_trends_' + key + '.py','w')
        f.write('data_' + key + ' = ' + str(data))
        f.close()
        print('fichier enregistré')
        
    else:
        print('chargement du fichier google trends_' + key, flush = True)
        localsParameter = {}
        exec('from google_trends_' + key + ' import data_' + key,
             {}, localsParameter)
        data = localsParameter['data_' + key]
        print('ok-----------', flush = True)
    jours1 = [x[0] for x in data['01']]
    jours = []
    nj0 = num_de_jour(aujourdhui)
    for j in jours1:
        nj = num_de_jour(j)
        for k in range(7):
            if nj+k <= nj0:
                jours.append(jour_de_num[nj+k])
    # ameliorer la derniere periode, tronquer au present 
    datav = np.zeros((len(departements),len(jours),1))
    for (d,dep) in enumerate(departements):
        for j in range(len(jours1)):
            datav[d,7*j:min(len(jours),7*j+7),0] = data[dep][j][1]
    return((key,jours,datav))

lkeys = [charge_trends(k)
         for k in [['Covid','Covid 19', 'Coronavirus', 'SARS Cov 2'],
                   ['test covid','test coronavirus','test PCR','test antigénique'],
                   ['pharmacie','médecin', 'sos médecin'],
                   ['horaires','horaire'],
                   ['voyage'],
                   ['itinéraire'],#https://trends.google.com/trends/explore?geo=FR&q=itin%C3%A9raire
         ]]

jours = sorted([lj for (k,lj,data) in lkeys], key = lambda x: num_de_jour(x[-1]))[-1]
#lkeys[0][1]

datav = np.zeros((len(departements),len(jours),len(lkeys)))
for (d,dep) in enumerate(departements):
    for j in range(len(jours)):
        datav[d,j,:len(lkeys)] = [data[d,min(j,len(lj)-1),0] for (k,lj,data) in lkeys]

datagoogletrends = {'nom': 'googletrends',
              'titre': 'requêtes google',
              'dimensions': ['departements','jours','googletrends'],
              'departements': [int(d) for d in departements],
              'jours': jours,
              'googletrends': ['recherche ' + k[0] + ' google' for k in lkeys],
              'valeurs': datav}

datagoogletrends_prev = {'nom': 'googletrends',
                        'titre': 'requêtes google',
                        'dimensions': ['departements','jours','googletrends'],
                        'departements': [int(d) for d in departements],
                        'jours': jours,
                        'googletrends': ['recherche ' + k[0] + ' google'
                                         for k in lkeys[3:]],
                        'valeurs': datav[:,:,3:]}

print('requetes google ok',jours[-1])

######################################################################
# eaux usées
# https://www.data.gouv.fr/fr/datasets/stations-de-traitement-des-eaux-usees-france-entiere/

######################################################################
# https://covidnet.fr/

######################################################################
# pauvreté par départements
# https://www.insee.fr/fr/statistiques/fichier/4507225/base-filosofi-2017_CSV.zip
f = open('pauvrete/cc_filosofi_2017_DEP.CSV','r')
s = f.read()
ls = [x.split(';') for x in s.split('\n')]
f.close()

lchamps_pauvrete = [('TP6017','Taux de pauvreté'),
                    ('MED17','Niveau de vie médian'),
                    ('D117','1er décile du niveau de vie'),
                    ('D917','9e décile du niveau de vie'),
                    ('RD17','9e décile/1er décile'),
                    ('TP60AGE117','Taux de pauvreté<30 ans'),
                    ('TP60AGE217','Taux de pauvreté 30-39 ans'),
                    ('TP60AGE317','Taux de pauvreté 40-49 ans'),
                    ('TP60AGE417','Taux de pauvreté 50-59 ans'),
                    ('TP60AGE517','Taux de pauvreté 60-74 ans'),
                    ('TP60AGE617','Taux de pauvreté >75 ans'),
                    ('NBMENFISC17','Nombre de ménages fiscaux'),
                    ('NBPERSMENFISC17','Nombre de personnes dans les ménages fiscaux'),
                    ]

valdep = [[int(x[ls[0].index('CODGEO')])]
          + [x[ls[0].index(c)] for (c,t) in lchamps_pauvrete]
          for x in ls[1:-1] if x[ls[0].index('CODGEO')][1] not in ['A','B']]

deps = [v[0] for v in valdep]
jours = [jour_de_num[k] for k in range(num_de_jour('2019-01-01'),num_de_jour('2022-01-01'))]
datav = np.zeros((len(deps),len(jours),len(lchamps_pauvrete) - 2))
for (d,v) in enumerate(valdep):
    datav[d,:,:] = ([float(v[1]), float(v[2])/(float(v[13])/float(v[12]))]
                    + [float(x) for x in v[3:12]])

datapauvrete = {'nom': 'taux de pauvreté',
                'titre': 'taux de pauvreté',
                'dimensions': ['departements','jours','pauvreté'],
                'departements': deps,
                'jours': jours,
                'pauvreté': [v[1] for v in lchamps_pauvrete[:-2]],
                'valeurs': datav}

print('taux de pauvreté chargé')

######################################################################
# population
print('on charge la population', flush = True)
deps = [d for d in population_dep]
datav = np.zeros((len(deps),len(jours),1))
for (d,v) in enumerate(population_dep):
    datav[d,:,0] = v

datapop = {'nom': 'population',
           'titre': 'population',
           'dimensions': ['departements','jours','population'],
           'departements': [d for d in population_dep],
           'jours': jours,
           'population': ['population'],
           'valeurs': datav}

print('population chargée')

######################################################################
# vaccination
# https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-personnes-vaccinees-contre-la-covid-19/
######################################################################
# égoûts
# https://www.data.gouv.fr/fr/datasets/reseau-de-collecte-des-eaux-usees/

######################################################################
#
contextes = [datamobilite, datameteo, datavacances, dataapple, datahygiene,
             datagoogletrends, datagoogletrends_prev,
             regions, datapauvrete,lchamps_pauvrete[:-2],datapop]

import pickle
f = open('contextes.pickle','wb')
pickle.dump(contextes,f)
f.close()

