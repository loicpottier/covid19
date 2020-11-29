# prévision des indicateurs covid à partir des données de mobilité de google par département et des données météo.

######################################################################
from outils import *

######################################################################
# frequentations des lieux
# https://www.google.com/covid19/mobility/
# les commerces et espaces de loisirs, les magasins d'alimentation et pharmacies, les parcs, les arrêts de transports en commun, les lieux de travail et les lieux de résidence.
# pourcentages de changement par rapport à la valeur de référence
# pour chaque département

print('download du fichier de mobilité de google')
response = urllib.request.urlopen('https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip')
encoding = response.info().get_content_charset() or "utf-8"
data = response.read()      # a `bytes` object
f = open('Region_Mobility_Report_CSVs.zip','wb')
f.write(data)
f.close()
import zipfile
with zipfile.ZipFile('Region_Mobility_Report_CSVs.zip', 'r') as zip_ref:
    zip_ref.extractall('mobilite_google')
print('fait')

f = open('mobilite_google/2020_FR_Region_Mobility_Report.csv','r')
s = f.read()
ls = [x.split(',') for x in s.split('\n')]

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
        print(dep, 'données vides')
    else:
        datalieux.append(ld)
        departements.append(d)

# on vire les départements avec des jours qui manquent
lmax = max([len(x) for x in datalieux])
departements = [d for (k,d) in enumerate(departements)
                if len(datalieux[k]) == lmax]
datalieux = [x for x in datalieux if len(x) == lmax]

datamobilite = {'nom': 'mobilite',
                'titre': 'fréquentation des lieux',
                'dimensions': ['departements','jours','mobilites'],
                'departements': departements,
                'jours': [x[0] for x in datalieux[1]],
                'mobilites': ['commerces et espaces de loisir',
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

print('chargement des données météo')
# novembre:
#https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/synop.202011.csv.gz
data = []
for mois in range(2,12):
    print(mois,end = ' ')
    smois = str(mois)
    if mois <10:
        smois = '0' + smois
    d = chargecsv('https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/synop.2020'+ smois + '.csv.gz', zip = True)
    if mois == 2:
        data = d
    else:
        data = data + d[1:]
    #print(smois,'fait')

param = data[0] # les bons des parametres meteo mesures
# explication ici: https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=90&id_rubrique=32

#nomsmeteo = ['pression','humidité','precip.24h','température','nébulosité', 'vent']
#imeteo = [param.index(x) for x in ['pres','u','rr24','t','n','ff']]

nomsmeteo = ['pression','humidité','precip.24h','température', 'vent']
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
deppb = []
mespb = []
for d in range(len(deps)):
    for j in range(len(jours)):
        for k in range(len(imeteo)):
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
                    if nomsmeteo[k] not in mespb:
                        mespb.append(nomsmeteo[k])

deps1 = deps[:]

for d in deppb:
    deps1.remove(d)

t1 = np.array([t[deps.index(d)] for d in deps1])
#np.min(t1[:,:,0])

datameteo = {'nom': 'meteo',
              'titre': 'mesures meteo (stations principales)',
              'dimensions': ['departements','jours','meteo'],
              'departements': deps1,
              'jours': jours,
              'meteo': nomsmeteo,
              'valeurs': t1
}
######################################################################
# vacances

zones = {
    "A": [3,15,43,63,7,26,38,73,74,1,42,69,25,39,70,90,21,58,71,89,24,33,40,47,64,19,23,87,16,17,79,86],
    "B":[22,29,35,56,18,28,36,37,41,45,2,54,55,57,88,8,10,51,52,67,68,2,60,80,59,62,14,50,61,27,76,44,49,53,72,85,4,5,13,84,6,83],
    "C":[77,93,94,75,78,91,92,95,11,30,34,48,66,9,12,31,32,46,65,81,82]}

vacances = {
    "A": [('2020-02-22','2020-03-08'),('2020-03-17','2020-05-11'),('2020-05-12','2020-07-03'),('2020-07-04','2020-09-01'),('2020-10-17','2020-11-01'),('2020-12-19','2021-01-03')],
    "B": [('2020-02-15','2020-03-01'),('2020-03-17','2020-05-11'),('2020-05-12','2020-07-03'),('2020-07-04','2020-09-01'),('2020-10-17','2020-11-01'),('2020-12-19','2021-01-03')],
    "C": [('2020-02-08','2020-02-24'),('2020-03-17','2020-05-11'),('2020-05-12','2020-07-03'),('2020-07-04','2020-09-01'),('2020-10-17','2020-11-01'),('2020-12-19','2021-01-03')]
}
vvacances = [100,100,80,100,100,100]

jours = [jour_de_num[k]
         for k in range(num_de_jour('2020-02-08'),
                        num_de_jour('2021-01-03')+1)]

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
######################################################################
# on ajoute les mesures meteo aux données utilisées pour prévoir.

# fusion de contextes
def fusion(data1,data2):
    nomsv1 = data1['dimensions'][-1]
    deps1 = data1['departements']
    jours1 = data1['jours']
    nomsv2 = data2['dimensions'][-1]
    deps2 = data2['departements']
    jours2 = data2['jours']
    deps = [d for d in deps1 if d in deps2]
    jours = [j for j in jours1 if j in jours2]
    noms = nomsv1 + '-' + nomsv2
    data = {'nom': data1['nom'] + '-' + data2['nom'],
            'titre': data1['titre'] + '-' + data2['titre'],
            'dimensions': ['departements','jours', noms],
            'departements': deps,
            'jours': jours,
            noms: data1[nomsv1] + data2[nomsv2]
    }
    t = np.zeros((len(deps),len(jours),len(data[noms])))
    for d in range(len(deps)):
        for j in range(len(jours)):
            for k in range(len(data1[nomsv1])):
                d1 = deps1.index(deps[d])
                j1 = jours1.index(jours[j])
                t[d,j,k] = data1['valeurs'][d1,j1,k]
            for k in range(len(data2[nomsv2])):
                d2 = deps2.index(deps[d])
                j2 = jours2.index(jours[j])
                t[d,j,len(data1[nomsv1]) + k] = data2['valeurs'][d2,j2,k]            
    data['valeurs'] = t
    return(data)

datacontexte = fusion(fusion(datamobilite,datameteo),datavacances)
datacontexte['contextes'] = datacontexte['mobilites-meteo-vacances']

#datacontexte = fusion(datameteo,datavacances)
#datacontexte['contextes'] = datacontexte['meteo-vacances']

#datacontexte = fusion(datamobilite,datavacances)
#datacontexte['contextes'] = datacontexte['mobilites-vacances']

#datacontexte = fusion(datamobilite,datameteo)
#datacontexte['contextes'] = datacontexte['mobilites-meteo']

#datacontexte = datamobilite
#datacontexte['contextes'] = datacontexte['mobilites']

#datacontexte = datameteo
#datacontexte['contextes'] = datacontexte['meteo']

#datacontexte = datavacances
#datacontexte['contextes'] = datacontexte['vacances']

# normalisation des valeurs des contextes: entre 0 et 100
import copy

v = copy.deepcopy(datacontexte['valeurs'])

for c in range(len(v[0,0])):
    maxv = np.max(v[:,:,c])
    minv = np.min(v[:,:,c])
    v[:,:,c] = 100 * (v[:,:,c] - minv) / (maxv - minv)

datacontexte['valeurs'] = v
######################################################################
def sdep(d):
    dep = str(d)
    if d < 10:
        dep = '0' + str(d)
    return(dep)

def url_dep(url,d):
    return(url.replace('DEPARTEMENT',sdep(d)))

def charge_data(url):
    t = [[]]*97
    for d in range(1,96):
        try:
            print(sdep(d) + ' ', end = '')
            data,titre = charge(url_dep(url,d))
            try:
                ls = url.split('indic=')
                incid = ls[-1]
                jours = sorted(list(set([x['jour'] for x in data])))
                try:
                    unjour = [(j,
                               [x[incid]
                                for x in data if x['jour'] == j 
                                and x['territory'] == 'dep@'+ sdep(d)][0])
                              for j in jours]
                except:
                    unjour = [(j,
                               [x['p']
                                for x in data if x['jour'] == j 
                                and x['territory'] == 'dep@'+ sdep(d)][0])
                              for j in jours]
            except:
                jours = sorted(list(set([x['date_de_passage'] for x in data])))
                unjour = [(j,
                           [x['nbre_pass_corona']
                            for x in data if x['date_de_passage'] == j 
                            and x['territory'] == 'dep@'+ sdep(d)][0])
                          for j in jours]
            t[d] = unjour
        except:
            print('problème avec', str(d))
    return(t)

print('download des données de santepubliquefrance (long, département par département)')

def charge_donnees(url,nom,titre):
    data = charge_data(url)
    maxjours = max([len(d) for d in data])
    deps = [d for (d,dep) in enumerate(data)
            if len(dep) == maxjours and max([abs(x[1]) for x in dep]) < 9999]
    datares = {'nom': nom,
               'titre': titre,
               'dimensions': ['departements', 'jours'],
               'jours': [x[0] for x in data[1]],
               'departements': deps,
               'valeurs': np.array([[x[1] for x in dep]
                                   for (d,dep) in enumerate(data)
                                   if d in deps])
               }
    return(datares)

datahospi = charge_donnees('https://geodes.santepubliquefrance.fr/GC_infosel.php?lang=fr&allindics=0&nivgeo=dep&view=map2&codgeo=DEPARTEMENT&dataset=covid_hospit_incid&indic=incid_hosp',
                           'hospitalisations',
                           'Nombre quotidien de nouvelles personnes hospitalisées avec diagnostic COVID-19 déclarées en 24h')

print('hospitalisations fait')

datarea = charge_donnees('https://geodes.santepubliquefrance.fr/GC_infosel.php?lang=fr&allindics=0&nivgeo=dep&view=map2&codgeo=DEPARTEMENT&dataset=covid_hospit_incid&indic=incid_rea',
                         'réanimations',
                         'Nombre quotidien de nouvelles admissions en réanimation (SR/SI/SC) avec diagnostic COVID-19 déclarées en 24h')

print('réanimations fait')

datadeces = charge_donnees('https://geodes.santepubliquefrance.fr/GC_infosel.php?lang=fr&allindics=0&nivgeo=dep&view=map2&codgeo=DEPARTEMENT&dataset=covid_hospit_incid&indic=incid_dc',
                           'décès',
                           'Nombre quotidien de nouveaux décès avec diagnostic COVID-19 déclarés en 24h')

print('décès fait')

dataurge = charge_donnees('https://geodes.santepubliquefrance.fr/GC_infosel.php?lang=fr&allindics=0&nivgeo=dep&view=map2&codgeo=DEPARTEMENT&dataset=sursaud_corona_quot&indic=nbre_pass_corona&filters=sursaud_cl_age_corona=0',
                          'urgences',
                          'Nombre de passages aux urgences pour suspicion de COVID-19 - Quotidien')

print('urgences fait')

######################################################################
# on ne garde que les departements communs aux donnees

deps = datahospi['departements'][:]
[[deps.remove(d) for d in deps if  d not in t['departements'] and d in deps]
 for t in [datarea, datadeces, dataurge, datacontexte]]

for t in [datahospi,datarea, datadeces, dataurge, datacontexte]:
    t['valeurs'] = np.array([t['valeurs'][t['departements'].index(d)] for d in deps])
    t['departements'] = deps
    
######################################################################
# prevision
######################################################################
# Lt tableau de valeurs par departements,jours,contexte
# Lj noms des jours
# Ht tableau de valeurs par departements,jours
# Hj noms des jours
# même dimension en départements, pas en jours

# dj décalage du début de Lt par rapport au début de Ht
# rend C telle que sum_d ||Lt[d].C - Ht[d]||^2 minimale
# et les sous-tableaux décalés de dj pour lesquels C est obtenu
def erreur(dj,Lj,Lt,Hj,Ht):
    e = -1
    ndeps,njL,nlieux = np.shape(Lt)
    ndeps,njH = np.shape(Ht)
    decal = njH - (njL + dj)
    L = Lt[:,max(0,-dj):min(njL,njH-dj),:]
    H = Ht[:,max(0,dj):min(njH,njL+dj)]
    LJ = Lj[max(0,-dj):min(njL,njH-dj)]
    HJ = Hj[max(0,dj):min(njH,njL+dj)]
    C = np.zeros((ndeps,nlieux))
    if abs(decal) < 30 and len(LJ) != 0 : # décalage max des fins des tableaux
        e = 0
        for d in range(ndeps):
            A = np.transpose(L[d]) @ L[d]
            B = np.transpose(L[d]) @ H[d]
            C[d] = np.linalg.inv(A) @ B
            E = L[d] @ C[d] - H[d]
            e = e + np.transpose(E) @ E
        e = math.sqrt(e/(ndeps * len(LJ)))
    else:
        pass
    return(e,LJ,L,HJ,H,C)

# rend le decalage des dates qui minimise l'erreur de prévision,
# le vecteur de prévision et les sous-tableaux qui donnent cette erreur min
# (decalage de la fin du contexte par rapport à la fin des données)
def fit(Lj,Lt,Hj,Ht):
    ndeps,njL,nlieux = np.shape(Lt)
    ndeps,njH = np.shape(Ht)
    erreurs = []
    emin = 10000000000000000000
    dmin = 0
    Cmin = None
    Lmin,Hmin,LJmin,HJmin = None,None,None,None
    for dj in range(-njL,njH):
        decal = njH - (njL + dj)
        try:
            e,LJ,L,HJ,H,C = erreur(dj,Lj,Lt,Hj,Ht)
            #print(dj,e)
            if e > 0:
                erreurs.append((decal,e))
                if e < emin:
                    emin = e
                    dmin = decal
                    Cmin = C
                    Lmin,Hmin,LJmin,HJmin = L,H,LJ,HJ
        except:
            print('probleme avec dj = ',dj)
    erreurs = np.array(erreurs)
    # erreur minimale, C coefs des lieux
    C = Cmin
    L,H,LJ,HJ = Lmin,Hmin,LJmin,HJmin
    decalage = 0
    #print(Lj[-1],Hj[-1],dmin, emin, njH,njL,C)
    try:
        while HJ[-1-decalage] != LJ[-1]:
            decalage += 1
    except:
        plt.clf()
        plt.plot([x[0] for x in erreurs],[x[1] for x in erreurs])
        plt.grid()
        plt.show(False)
    return(decalage,C,emin)

######################################################################
# prévision des données

def prevision(datacontexte,data):
    #print('prevision', data['nom'])
    A,Aj = datacontexte['valeurs'], datacontexte['jours']
    Bj = data['jours']
    B = np.array([lissage(dep,7) for dep in data['valeurs']])
    Bd = np.array([derivee(dep,largeur=2) for dep in B])
    decalage,C,e = fit(Aj,A,Bj,Bd)
    # si le decalage < 0, pas de prevision du futur...
    f = 0
    j = Aj[-1]
    while num_de_jour(j) > num_de_jour(Bj[-1]):
        j = jour_de_num[num_de_jour(j)-1]
        f = f + 1
    ldB = {} # prévision des derivees pour les derniers jours des contextes
    for j in range(decalage):
        db = np.zeros(np.shape(B[:,0]))
        db = db + np.array([A[d,j-f-decalage,:] @ C[d] for d in range(len(C))])
        nj = Aj[j-f-decalage]
        ldB[addday(nj,decalage-1)] = db
    bj = B[:,-1]
    j = Bj[-1]
    P = []
    Pj = []
    while j in ldB:
        P.append(bj)
        Pj.append(j)
        bj = bj + ldB[j]
        j = addday(j,1)
    prev = {'nom': data['nom'],
            'decalage': decalage,
            'coefficients': C,
            'erreur': e,
            'jours': Pj,
            'valeurs': np.array(P)}
    return(prev)

def previsions(datacontexte,data,n):
    #print('prevision', data['nom'])
    A,Aj = datacontexte['valeurs'], datacontexte['jours']
    Bj = data['jours']
    B = np.array([lissage(dep,7) for dep in data['valeurs']])
    Bd = np.array([derivee(dep,largeur=2) for dep in B])
    decalage,C,e = fit(Aj,A,Bj,Bd)
    # si le decalage < 0, pas de prevision du futur...
    f = 0
    j = Aj[-1]
    while num_de_jour(j) > num_de_jour(Bj[-1]):
        j = jour_de_num[num_de_jour(j)-1]
        f = f + 1
    ldB = {} # prévision des derivees pour les derniers jours des contextes
    for j in range(-n+1,decalage):
        db = np.zeros(np.shape(B[:,0]))
        db = db + np.array([A[d,j-f-decalage,:] @ C[d] for d in range(len(C))])
        nj = Aj[j-decalage]
        ldB[addday(nj,decalage-1)] = db
    lprev = []
    for jfin in range(1,n):
        if jfin % 10 == 1:
            bj = B[:,-jfin]
            j = Bj[-jfin]
            P = []
            Pj = []
            k = 0
            while j in ldB: #and k <= decalage:
                k += 1
                P.append(bj)
                Pj.append(j)
                bj = bj + ldB[j]
                j = addday(j,1)
            prev = {'nom': data['nom'],
                    'decalage': decalage,
                    'coefficients': C,
                    'erreur': e,
                    'jours': Pj,
                    'valeurs': np.array(P)}
            lprev.append(prev)
    return(lprev)

######################################################################
# prévisions variées

P = [prevision(datacontexte,data) for data in [dataurge,datahospi,datarea,datadeces]]

titre = 'prévisions France\nà partir des données de mobilité Google et des données météo\ndécalages:\n'
for p in P:
    titre += p['nom'] + ' ' + str(p['decalage']) + ' '

trace([(zipper(p['jours'], np.sum(p['valeurs'],axis = 1)),
        p['nom'],'o')
       for p in P]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0),7))[-120:],
        '','-')
       for t in [dataurge,datahospi,datarea,datadeces]],
      titre,
      'donnees/_prevision_par_mobilite')

# previsions a differentes dates
data = dataurge
lP = previsions(datacontexte,data,250)
trace([(zipper(p['jours'], np.sum(p['valeurs'],axis = 1)),
        '','-')
       for p in lP]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0),7)),
        '','=')
       for t in [data]],
      'urgences',
      '_previsions_' + data['nom'])

####################### dispersion des coefficients des différents départements
def normalisecoef(C):
    s = np.sum(np.abs(C), axis = 1)
    return(np.array([C[d]/s[d] for d in range(len(C))]))

# pour la prévision des urgences
p = P[0] # prevision urge
plt.clf()
for d in normalisecoef(p['coefficients']):
    plt.plot(100 * d,'.')

plt.plot(100 * np.mean(normalisecoef(p['coefficients']),axis=0),'-')
plt.xticks(range(len(datacontexte['contextes'])),
           [x[:7] for x in datacontexte['contextes']],
           rotation = 45,fontsize = 8)
plt.grid()
plt.savefig('donnees/_dispersion_contexte.pdf')
plt.savefig('donnees/_dispersion_contexte.png')
plt.show(False)

rapport = []
rapport.append(['France' ] + [p['nom'] for p in P])
rapport.append(['décalage'] + [str(p['decalage']) for p in P])
rapport.append(['erreur moyenne'] + ["%4.2f" % p['erreur'] for p in P])
for l in range(len(datacontexte['contextes'])):
  rapport.append([datacontexte['contextes'][l]]
                 + [str(int(100 * np.mean(normalisecoef(p['coefficients']),
                                          axis=0)[l]))
                    + ' (± '
                    + str(int(100 * np.std(normalisecoef(p['coefficients']),
                                           axis=0)[l]))
                    + ')'
                    for p in P])

rapport = table(rapport)

now = time.localtime(time.time())
rapport += time.asctime(now)

f = open('donnees/_rapport','w')
f.write(rapport)
f.close()

######################################################################
# mobilite google et température

trace([(zipper(datacontexte['jours'],
               np.array(lissage(np.sum(datacontexte['valeurs'][:,:,l], axis = 0),7))
                       / len(datacontexte['departements'])),
        datacontexte['contextes'][l],
        '-')
       for l in range(len(datacontexte['contextes']))],
      'mobilité google:\n % de fréquentation par rapport à la moyenne\nlissage sur 7 jours\ndonnées météo, vacances',
      'donnees/_mobilite_google')

trace([(zipper(datacontexte['jours'],
               np.array(lissage(datacontexte['valeurs'][dep06,:,l],7)))[-120:],
        datacontexte['contextes'][l],
        '-')
       for l in range(len(datacontexte['contextes']))],
      'mobilité google 06:\n % de fréquentation par rapport à la moyenne\nlissage sur 7 jours\ndonnées météo, vacances',
      'donnees/_mobilite06_google')

