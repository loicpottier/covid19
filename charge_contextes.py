from outils import *
ajourgoogletrends = False #True si mettre a jour les fichiers de requetes google
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

ls2020 = ls

f.close()
f = open('mobilite_google/2021_FR_Region_Mobility_Report.csv','r')
s = f.read()
ls = [x.split(',') for x in s.split('\n')]
if ';' in ls[0][0]: # des fois le séparateur change!!!!! fait chier google
    ls = [x.split(';') for x in s.split('\n')]
elif '\t' in ls[0][0]:
    ls = [x.split('\t') for x in s.split('\n')]

f.close()
ls2021 = ls

ls = ls2020 + ls2021[1:]
os.system('rm -rf mobilite_google Region_Mobility_Report_CSVs.zip')

datalieux = []
departements = []

for d in range(1,96):
    dep = str(d)
    if d < 10:
        dep = '0' + str(d)
    # ils ont ajouté un champ place_id le 15-02-2021
    ld = [x[8:] for x in ls if len(x) >= 8 and x[5] == 'FR-'+dep]
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
                        for u in range(j-1,-1,-1):
                            if t[d,u,k] != -100:
                                prev = t[d,u,k]
                                jprev = u
                                raise NameError('ok')
                        if deps[d] not in deppb:
                            deppb.append(deps[d])    
                    except:
                        pass
                    if jsuiv != j and jprev != j:
                        t[d,j,k] = prev + (j - jprev)/(jsuiv-jprev) * (suiv - prev)
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
          ('2021-02-06','2021-02-21'),('2021-02-22','2021-04-02'),
          ('2021-04-03','2021-04-25'),('2021-04-26','2021-05-02'),
          ('2021-05-13','2021-05-16'),('2021-05-17','2021-07-05'),('2021-07-07','2021-09-01')],

    "B": [('2019-12-21','2020-01-05'),
          ('2020-02-15','2020-03-01'),('2020-03-17','2020-05-11'),('2020-05-12','2020-07-03'),
          ('2020-07-04','2020-09-01'),('2020-10-17','2020-11-01'),('2020-11-16','2020-12-18'),
          ('2020-12-19','2021-01-03'),('2021-01-04','2021-01-19'),('2021-01-20','2021-02-19'),
          ('2021-02-20','2021-03-07'),('2021-03-08','2021-04-02'),
          ('2021-04-03','2021-04-25'),('2021-04-26','2021-05-02'),
          ('2021-05-13','2021-05-16'),('2021-05-17','2021-07-05'),('2021-07-07','2021-09-01')],

    "C": [('2019-12-21','2020-01-05'),
          ('2020-02-08','2020-02-24'),('2020-03-17','2020-05-11'),('2020-05-12','2020-07-03'),
          ('2020-07-04','2020-09-01'),('2020-10-17','2020-11-01'),('2020-11-16','2020-12-18'),
          ('2020-12-19','2021-01-03'),('2021-01-04','2021-01-19'),('2021-01-20','2021-02-12'),
          ('2021-02-13','2021-02-28'),('2021-03-01','2021-04-02'),
          ('2021-04-03','2021-04-25'),('2021-04-26','2021-05-02'),
          ('2021-05-13','2021-05-16'),('2021-05-17','2021-07-05'),('2021-07-07','2021-09-01')]
}
vvacances = [100,
             100,100,80,
             100,100,50, # confinement apres la toussaint
             100,50,50, # pas de déconfinement le 20 janvier
             100,50, # apres fevrier encore 1/2 effectif a l ecole, confinement 3
             100,80, # confinement + vacances communes aux 3 zones
             100,50,100] # pont de l'ascension, vacances d'été

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
# couvre feu, confinements
deps = datavacances['departements']
jours = datavacances['jours']

t = np.zeros((len(deps),len(jours),1))

def confinement(cfdeps,j0,j1,valconf):
    for dep in cfdeps:
        d = deps.index(dep)
        for j in range(num_de_jour(j0),num_de_jour(j1)+1):
            t[d,j-num_de_jour(jours[0]),0] = max(valconf,
                                                 t[d,j-num_de_jour(jours[0]),0])

# confinements

# 17 mars au 11 mai: tous les departements
confinement(deps,'2020-03-17','2020-05-11',100)
# 30-10-2020 au 27-11-2020 confinement
confinement(deps,'2020-10-30','2020-11-27',80)
# 28-11-2020 au 14-12-2020 confinement mais commerces ouverts
confinement(deps,'2020-11-28','2020-12-14',60)

######################################################################
# couvre-feu 21h-6h

# 17-10-2020 au 23-10-2020
confinement([75,77,78,91,92,93,94,95, # ile de france
               13,38,59,69,34,76,42,31],# 8 metropoles
            '2020-10-17','2020-10-22',20)
# 23-10-2020 au 30-10-2020, 21h à 6h
'''
jsont = chargejson('https://www.data.gouv.fr/fr/datasets/r/bfded8e2-e1d0-4601-8b81-0807f8dca65d',
                  zip = False)
# ou bien, si ca merde,
f = open('couvre-feu-20201023.geojson','r')
s = f.read()
f.close()
jsont = json.loads(s)
cf2 = [x['properties']['INSEE_DEP'] for x in jsont['features']]
cf2 = [int(x) for x in cf2 if x not in ['2A','2B']]
'''
cf2 = [1, 5, 6, 7, 8, 9, 10, 12, 13, 14, 21, 26, 30, 31, 34, 35, 37, 38, 39, 42, 43, 45, 48, 49, 51, 54, 59, 60, 62, 63, 64, 65, 66, 67, 69, 71, 73, 74, 75, 76, 77, 78, 81, 82, 83, 84, 87, 91, 92, 93, 94, 95]
confinement(cf2,'2020-10-23','2020-10-29',20)

######################################################################
# couvre-feu 20h-6h

confinement(deps,'2020-12-15','2021-01-15',25)
######################################################################
# couvre-feu 18h-6h

# Hautes-Alpes, Alpes-Maritimes, Ardennes, Doubs, Jura, Marne, Haute-Marne, Meurthe-et-Moselle, Meuse, Moselle, Nièvre, Haute-Saône, Saône-et-Loire, Vosges et Territoire de Belfort
confinement([5,6,8,25,39,51,52,54,55,57,58,70,71,88,90],
            '2021-01-02','2021-01-30',35)
# Hautes-Alpes, Alpes-Maritimes, Ardennes, Doubs, Jura, Marne, Haute-Marne, Meurthe-et-Moselle, Meuse, Haute-Saône, Vosges, Moselle, Territoire de Belfort, Nièvre, Saône-et-Loire, Bas-Rhin, Bouches-du-Rhône, Haut-Rhin, Allier, Vaucluse, Cher, Côte d'Or, Alpes de Haute-Provence, Drôme et Var
confinement([5,6,8,25,39,51,52,54,55,57,58,70,71,88,90,67,13,68,3,84,18,21,4,26,83],
            '2021-01-12','2021-01-30',35)
confinement(deps,'2021-01-16','2021-03-19',35)

######################################################################
#confinement le week-end

confinement([6,59],'2021-02-26','2021-03-19',20)
confinement([62],'2021-03-04','2021-03-19',20)

######################################################################
# confinement3
confinement([7,92,93,94,91,95,77,78,2,59,60,62,80,6,76,27],'2021-03-20','2021-04-05',80)
confinement([7,92,93,94,91,95,77,78,2,59,60,62,80,6,76,27,10,58,69],'2021-03-27','2021-04-27',80)
confinement(deps,'2021-04-06','2021-05-02',80)

######################################################################
# couvre-feu de 19h à 6h
confinement(deps,'2021-03-20','2021-05-02',30)

dataconfinement = {'nom': 'confinement',
                   'titre': 'confinement ou couvre-feu',
                   'dimensions': ['departements','jours','confinement'],
                   'departements': deps,
                   'jours': jours,
                   'confinement': ['confinement ou couvre-feu'],
                   'valeurs': t}

print('confinement/couvre-feu ok')

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

j = str(now.tm_year) + '-' + sdep(now.tm_mon) + '-' + sdep(now.tm_mday)

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
        print(j)
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
    # si on veut les valeurs les plus récentes
    # car on a un nombre de requetes limité par jour
    if ajourgoogletrends and key in ['horaires','voyage','itinéraire']: 
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
# vaccination
#https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-personnes-vaccinees-contre-la-covid-19-1/

def get(data,x,champ):
    try:
        return(x[data[0].index(champ)])
    except:
        return(x[data[0].index('"' + champ + '"')].replace('"',''))
      

departements = ['0' + str(x) for x in range(1,10)] + [str(x) for x in range(10,96) if x != 20]

# parfois le séparateur est la virgule!
# nombre quotidien de personnes ayant reçu au moins une dose, par date d’injection :
csv = chargecsv('https://www.data.gouv.fr/fr/datasets/r/4f39ec91-80d7-4602-befb-4b522804c0af',
                zip = False,
                sep = ';')[:-1] # enlever le [''] de la fin
#nombre quotidien de résidents en EHPAD ayant reçu au moins une dose ou deux doses, par date d’injection :
csvehpad = chargecsv('https://www.data.gouv.fr/fr/datasets/r/54dcd1af-a4cd-47c6-a8a2-45d444529a68',
                     zip = False,
                     sep = ';')[:-1] # enlever le [''] de la fin

print('csv chargés', flush = True)
print(csv[0])

dernierjour = min([num_de_jour(get(csv,csv[-1], 'jour')),
                   num_de_jour(get(csvehpad,csvehpad[-1], 'jour'))])

jours = [j for j in range(num_de_jour('2020-02-01'),dernierjour+1)]
vac = np.zeros((len(departements),len(jours),2))
for x in csv:
    dep = get(csv,x,'dep')
    if dep in departements:
        d = departements.index(dep)
        jj = get(csv,x,'jour')
        #print(x,jj)
        if jj[:4] in ['2020','2021']:
            j = num_de_jour(jj) - jours[0]
            v = get(csv,x,'n_dose1')
            vac[d,j,0] = v

for x in csvehpad:
    dep = get(csvehpad,x,'dep')
    if dep in departements:
        d = departements.index(dep)
        jj = get(csvehpad,x,'jour')
        #print(x,jj)
        if jj[:4] in ['2020','2021']:
            j = num_de_jour(jj) - jours[0]
            v1 = get(csvehpad,x,'res_vac_dose1')
            v2 = get(csvehpad,x,'res_vac_dose2')
            v1 = 0 if v1 == '' else int(v1)
            v2 = 0 if v2 == '' else int(v2)
            vac[d,j,1] = v1+v2

#plt.plot(np.sum(vac[:,:,0],axis=0)[-60:]);plt.show()
#plt.plot(np.sum(vac[:,:,1],axis=0)[-60:]);plt.show()

datavaccins = {'nom': 'vaccins',
               'titre': 'vaccins quotidien',
               'dimensions': ['departements', 'jours','vaccins'],
               'jours': [jour_de_num[j] for j in jours],
               'departements': [int(d) for d in departements],
               'vaccins' : ['vaccins','vaccins ehpad'],
               'valeurs': vac}

print('vaccins ok', jour_de_num[jours[-1]])

######################################################################
# variants
#https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-indicateurs-sur-les-variants/#_

departements = ['0' + str(x) for x in range(1,10)] + [str(x) for x in range(10,96) if x != 20]

csv = chargecsv('https://www.data.gouv.fr/fr/datasets/r/16f4fd03-797f-4616-bca9-78ff212d06e8',
                zip = False,
                sep = ';')[:-1] # enlever le [''] de la fin

print('csv variants chargé', flush = True)
print(csv[0])

jmax = num_de_jour('2020-02-01')
for x in csv:
    s = get(csv,x,'semaine')
    j = s[11:]
    if j[:4] in ['2020','2021']:
        jmax = max(jmax,num_de_jour(j))

jours = [j for j in range(num_de_jour('2020-02-01'),jmax+1)]
var = np.zeros((len(departements),len(jours),3))

'''
Semaine = Semaine glissante
cl_age90 = Classe d’âge
Nb_tests_PCR_TA_crible = Nombre de tests PCR criblés parmi les PCR positives
Prc_tests_PCR_TA_crible = % de tests PCR criblés parmi les PCR positives
Nb_susp_501Y_V1 = Nombre de tests avec suspicion de variant 20I/501Y.V1 (UK)
Prc_susp_501Y_V1 = % de tests avec suspicion de variant 20I/501Y.V1 (UK)
Nb_susp_501Y_V2_3 = Nombre de tests avec suspicion de variant 20H/501Y.V2 (ZA) ou 20J/501Y.V3 (BR)
Prc_susp_501Y_V2_3 = % de tests avec suspicion de variant 20H/501Y.V2 (ZA) ou 20J/501Y.V3 (BR)
Nb_susp_IND = Nombre de tests avec une détection de variant mais non identifiable
Prc_susp_IND = % de tests avec une détection de variant mais non identifiable
Nb_susp_ABS = Nombre de tests avec une absence de détection de variant
Prc_susp_ABS = % de tests avec une absence de détection de variant
'''

for x in csv:
    dep = get(csv,x,'dep')
    clage = get(csv,x,'cl_age90')
    if dep in departements and clage == '0':
        d = departements.index(dep)
        s = get(csv,x,'semaine')
        j1,j2 = s[:10],s[11:]
        #print(j1,j2,x)
        if j2[:4] in ['2020','2021']:
            j2 = num_de_jour(j2) - jours[0]
            var[d,j2,0] = float(get(csv,x,'Prc_susp_501Y_V1'))
            var[d,j2,1] = float(get(csv,x,'Prc_susp_501Y_V2_3'))
            var[d,j2,2] = float(get(csv,x,'Prc_susp_IND'))

#on complète car ca ne commence qu'au 18 février
# point du 28 janvier
enqueteflash_7_janvier = {'Alsace-Champagne-Ardenne-Lorraine': 1.2, #Grand Est
                          'Aquitaine-Limousin-Poitou-Charentes': 1.7, #Nouvelle Aquitaine
                          'Auvergne-Rhône-Alpes': 1.6,
                          'Bourgogne-Franche-Comté': 0.2,
                          'Bretagne': 0.8,
                          'Centre-Val de Loire': 3.5,
                          'Ile-de-France': 6.9,
                          'Languedoc-Roussillon-Midi-Pyrénées': 2.9, #Occitanie
                          'Nord-Pas-de-Calais-Picardie': 2.6, #Hautes de France
                          'Normandie': 1.2,
                          'Pays de la Loire': 1.3,
                          "Provence-Alpes-Côte d'Azur": 4.8}
#point du 11 février
enqueteflash_27_janvier = {'Alsace-Champagne-Ardenne-Lorraine': 23.7, #Grand Est
                           'Aquitaine-Limousin-Poitou-Charentes': 12, #Nouvelle Aquitaine
                           'Auvergne-Rhône-Alpes': 14,
                           'Bourgogne-Franche-Comté': 0.2, # on sait pas
                           'Bretagne': 37.3,
                           'Centre-Val de Loire': 28.7,
                           'Ile-de-France': 22.2,
                           'Languedoc-Roussillon-Midi-Pyrénées': 11.4, #Occitanie
                           'Nord-Pas-de-Calais-Picardie': 11.8, #Hauts de France
                           'Normandie': 12.4,
                           'Pays de la Loire': 15.8,
                           "Provence-Alpes-Côte d'Azur": 11.2}
depvariant = {}
for r in enqueteflash_7_janvier:
    for d in regions[r]:
        depvariant[d]={'2021-01-07': enqueteflash_7_janvier[r]}

for r in enqueteflash_27_janvier:
    for d in regions[r]:
        depvariant[d]['2021-01-27'] = enqueteflash_27_janvier[r]

departements = [int(d) for d in departements]

def interpole(x,a,b,fa,fb):
    return(fa + (fb-fa)/(b-a) * (x-a))

for d,dep in enumerate(departements):
    for jj in jours:
        j = jj - jours[0]
        if jj >= num_de_jour('2021-01-07') and jj < num_de_jour('2021-01-27'):
            var[d,j,0] = interpole(jj,
                                   num_de_jour('2021-01-07'),num_de_jour('2021-01-27'),
                                   depvariant[dep]['2021-01-07'],depvariant[dep]['2021-01-27'])
        if jj >= num_de_jour('2021-01-27') and jj < num_de_jour('2021-02-18'):
            var[d,j,0] = interpole(jj,
                                   num_de_jour('2021-01-27'),num_de_jour('2021-02-18'),
                                   depvariant[dep]['2021-01-27'],
                                   var[d,num_de_jour('2021-02-18')-jours[0],0])
'''
plt.plot(np.transpose(var[:,-100:,0]));plt.show()
'''
datavariants = {'nom': 'variants',
               'titre': 'variants quotidien',
               'dimensions': ['departements', 'jours','variants'],
               'jours': [jour_de_num[j] for j in jours],
                'departements': departements,
               'variants' : ['variant UK', 'variants ZA BR', 'variant inconnu'],
               'valeurs': var}

print('variants ok', jour_de_num[jours[-1]])

######################################################################
#
contextes = [datamobilite, datameteo, datavacances, dataconfinement, dataapple, datahygiene,
             datagoogletrends, datagoogletrends_prev,
             regions, 
             datavaccins, datavariants]

import pickle
f = open(DIRCOVID19 + 'contextes.pickle','wb')
pickle.dump(contextes,f)
f.close()

