import sys
from urlcache import *
from outils import *
import pickle
import matplotlib.animation as animation

######################################################################
# paramètres importants

#0.3 c est bien, mais peu de dépendances,
#0.25 les urgences pas terribles
#0.15 c'est trop peu, genre overfitting

#c'est pas mal avec:

nombredelissage7 = 3
mincorrelation = 0.3 
mindecalage = 1
calculcoefdep = False # False: même jeu de coef de prevision pour tous les departements
mindependances = 6
contextessanslissage = []

#moins bien : contextessanslissage = ['vacances','confinement', 'confinement+commerces', 'couvre-feu 21h-6h', 'couvre-feu 20h-6h', 'couvre-feu 18h-6h']


nombredelissage7 = 5
mincorrelation = 0.3 
mindecalage = 1
calculcoefdep = False
mindependances = 6
contextessanslissage = []

'''

evaluation sur les 90 derniers jours

07-02-2021

nombredelissage7 = 5
mincorrelation = 0.3 
mindecalage = 1
calculcoefdep = False
mindependances = 6
contextessanslissage = []
Erreurs médianes moyennes:
à 7 jours: 5%
à 14 jours: 11%
à 21 jours: 15%
à 28 jours: 17%
à 35 jours: 18%
à 42 jours: 19%
à 49 jours: 21%
à 56 jours: 27%
à 63 jours: 36%
à 70 jours: 46%
à 77 jours: 44%

06-02-2021
avec trends a jour

nombredelissage7 = 5
mincorrelation = 0.3 
mindecalage = 1
calculcoefdep = False
mindependances = 6
contextessanslissage = []

Erreurs médianes moyennes:
à 7 jours: 6%
à 14 jours: 12%
à 21 jours: 18%
à 28 jours: 19%
à 35 jours: 20%
à 42 jours: 20%
à 49 jours: 22%
à 56 jours: 26%
à 63 jours: 37%
à 70 jours: 42%
à 77 jours: 49%

06-02-2021

nombredelissage7 = 5
mincorrelation = 0.25 
mindecalage = 1
calculcoefdep = False
mindependances = 6
contextessanslissage = []

Erreurs médianes moyennes:
à 7 jours: 5%
à 14 jours: 10%
à 21 jours: 16%
à 28 jours: 19%
à 35 jours: 21%
à 42 jours: 24%
à 49 jours: 26%
à 56 jours: 31%
à 63 jours: 44%
à 70 jours: 37%
à 77 jours: 45%

05-02-2021
nombredelissage7 = 5
mincorrelation = 0.25 
mindecalage = 1
calculcoefdep = False
mindependances = 6
contextessanslissage = []

Erreurs médianes moyennes:
à 7 jours: 5%
à 14 jours: 9%
à 21 jours: 14%
à 28 jours: 18%
à 35 jours: 20%
à 42 jours: 23%
à 49 jours: 29%
à 56 jours: 38%
à 63 jours: 46%
à 70 jours: 42%
à 77 jours: 51%

nombredelissage7 = 6
mincorrelation = 0.3 
mindecalage = 1
calculcoefdep = False
mindependances = 6
contextessanslissage = []

Erreurs médianes moyennes:
à 7 jours: 5%
à 14 jours: 11%
à 21 jours: 15%
à 28 jours: 18%
à 35 jours: 21%
à 42 jours: 29%
à 49 jours: 47%
à 56 jours: 69%
à 63 jours: 91%
à 70 jours: 120%
à 77 jours: 115%

nombredelissage7 = 4
mincorrelation = 0.3 
mindecalage = 1
calculcoefdep = False
mindependances = 6
contextessanslissage = []

Erreurs médianes moyennes:
à 7 jours: 6%
à 14 jours: 13%
à 21 jours: 18%
à 28 jours: 21%
à 35 jours: 24%
à 42 jours: 26%
à 49 jours: 26%
à 56 jours: 23%
à 63 jours: 20%
à 70 jours: 19%
à 77 jours: 21%

le mieux: 
nombredelissage7 = 5
mincorrelation = 0.3 
mindecalage = 1
calculcoefdep = False
mindependances = 6
contextessanslissage = []

Erreurs médianes moyennes:
à 7 jours: 5%
à 14 jours: 10%
à 21 jours: 15%
à 28 jours: 18%
à 35 jours: 19%
à 42 jours: 21%
à 49 jours: 25%
à 56 jours: 35%
à 63 jours: 49%
à 70 jours: 51%
à 77 jours: 62%
'''
######################################################################
# chargement des donnees

nouveau = False # False: on charge le fichier local
nouveauprev = False # seulement les previsions, on met pas a jour les donnees
nouveaucoefs = False
inclusconfinement = True #False
touteslesdonnees = True

#coefs[ni('couvre-feu 18h-6h'),ni('hospitalisations')]

if len(sys.argv) > 1 and sys.argv[1] == 'nouveauprev':
    nouveauprev = True
    nouveaucoefs = False
    nouveau = False

if len(sys.argv) > 1 and sys.argv[1] == 'nouveaucoefs':
    nouveauprev = True
    nouveaucoefs = True
    nouveau = False

if len(sys.argv) > 1 and sys.argv[1] == 'nouveau':
    nouveauprev = True
    nouveaucoefs = True
    nouveau = True

if nouveau:
    efface_cache()
    print('cache effacé, on charge les dernières données disponibles')
    try:
        import charge_contextes
        print('contextes chargés')
        try:
            import charge_indicateurs
            print('indicateurs chargés')
        except:
            print('*************************************probleme indicateurs')
            raise
    except:
        print('*************************************probleme contexte')
        raise
else:
    print('on utilise le cache')

print('nouveau',nouveau)
print('nouveauprev',nouveauprev)
print('nouveaucoefs',nouveaucoefs)

f = open('contextes.pickle','rb')
contextes = pickle.load(f)
f.close()
print('fichier des contextes chargé')

datamobilite, datameteo, datavacances, dataconfinement, dataapple, datahygiene, datagoogletrends, datagoogletrends_prev, regions, datapauvrete, lchamps_pauvrete, datapop = contextes

f = open('indicateurs.pickle','rb')
indicateurs = pickle.load(f)
f.close()
print('fichier des indicateurs chargé')

# j'ai viré sosmedecin car que 43 departements le donnent

dataurge, datahospiurge, datareatot, datahospitot, datahospi, datarea, datadeces, datahospiage, dataposage, datapos, datatauxposage, datatauxpos, dataR, dataexcesdeces, datadeces17mai = indicateurs

contextes_non_temporels = [x[1] for x in lchamps_pauvrete] + ['population']

donnees_proportions = (['commerces et espaces de loisir (dont restaurants et bars)',
                        "magasins d'alimentation et pharmacies",
                        'parcs',
                        'arrêts de transports en commun',
                        'travail',
                        'résidence',
                        'en voiture',
                        'à pied',
                        'en transport en commun',
                        'recherche Covid google',
                        'recherche testcovid google',
                        'recherche pharmacie google',
                        'recherche horaires google',
                        'recherche voyage google',
                        'recherche itinéraire google',
                        'excesdeces', 'R', 'taux positifs', 'taux positifs 09', 'taux positifs 19',
                        'taux positifs 29', 'taux positifs 39', 'taux positifs 49',
                        'taux positifs 59',
                        'taux positifs 69', 'taux positifs 79', 'taux positifs 89',
                        'taux positifs 90']
                       + dataconfinement['confinement'])

######################################################################
# on ne garde que les departements communs aux donnees

ldatacont = [datamobilite,datavacances]
if touteslesdonnees:
    ldatacont += [datameteo, dataapple, datagoogletrends]

if inclusconfinement:
    ldatacont.append(dataconfinement)
    print('----------- on a inclus les données de confinement/couvre-feu')

ldataind = ([dataurge, datahospiurge,# datasosmedecin, #dataR,
             datareatot,
             datahospi, datarea, datadeces, datapos, datatauxpos, datahospitot] )
if touteslesdonnees:
    ldataind += ([datahospiage[age] for age in sorted([a for a in datahospiage])]
                 + [dataposage[age] for age in sorted([a for a in dataposage])
                    if age != '0']
                 + [datatauxposage[age] for age in sorted([a for a in datatauxposage])
                    if age != '0'])

ldata = ldatacont + ldataind

departements = ldata[0]['departements'][:]
[[departements.remove(d) for d in departements[:] if  d not in t['departements'] and d in departements]
 for t in ldata if t['departements'] != []]

for t in ldata:
    if t['departements'] != []:
        t['valeurs'] = np.array([t['valeurs'][t['departements'].index(d)] for d in departements])
        t['departements'] = departements

######################################################################
# matrice des données
# on extrapole les données météo dans le futur en prenant les données un an avant

# union des jours
# la creation du memo de num_de_jour peut prendre un peu temps
jdebut = '2020-02-24'
jfin = '2021-05-01'
jours = set([])
for data in ldata:
    lj = [j for j in data['jours']]
    jours = jours.union({j for j in lj if jdebut <= j and j <= jfin})

jours = [num_de_jour(j) for j in sorted(jours)]

nomscont = [x for data in ldatacont for x in data[data['dimensions'][-1]]]
nomsind = [data['nom'] for data in ldataind]

noms = nomscont + nomsind
donnees_meteo = ['pression', 'humidité', 'précipitations sur 24', 'température', 'vent', 'variations de température']

nnoms,ndeps,njours = (len(noms), len(departements), len(jours))

nomsprevus = datavacances['vacances'] + dataconfinement['confinement'] + datameteo['meteo']

present = aujourdhui

def fin_donnees(data,present):
    if data['nom'] not in nomsprevus:
        return(present)
    else:
        return(data['jours'][-1])

def creeM(present):
    M = np.zeros((nnoms,ndeps,njours))
    intervalle = [None]*nnoms
    kc= 0
    for data in ldatacont:
        lnoms = data[data['dimensions'][-1]]
        j0 = max(num_de_jour(data['jours'][0]),jours[0])
        j1 = min(num_de_jour(data['jours'][-1]),jours[-1])
        j1 = min(j1,num_de_jour(fin_donnees(data,present)))
        mj0 = jours.index(j0)
        mj1 = jours.index(j1)
        dj0 = data['jours'].index(jour_de_num[j0])
        dj1 = data['jours'].index(jour_de_num[j1])
        for (c,nom) in enumerate(lnoms):
            #print(nom)
            intervalle[kc] = (mj0,mj1+1)
            M[kc, :, mj0:mj1+1] = copy.deepcopy(data['valeurs'][:, dj0:dj1+1,c])
            if nom in donnees_meteo:
                M[kc,:, mj1+1:] = copy.deepcopy(data['valeurs'][:, dj1+1-365:dj1+1-365 + njours - (mj1+1),c])
                intervalle[kc] = (mj0,mj1+1 + njours - (mj1+1))
            kc += 1
    for data in ldataind:
        j0 = max(num_de_jour(data['jours'][0]),jours[0])
        j1 = min(num_de_jour(data['jours'][-1]),jours[-1])
        j1 = min(j1,num_de_jour(fin_donnees(data,present)))
        mj0 = jours.index(j0)
        mj1 = jours.index(j1)
        dj0 = data['jours'].index(jour_de_num[j0])
        dj1 = data['jours'].index(jour_de_num[j1])
        #print(data['nom'],dj0,dj1)
        intervalle[kc] = (mj0,mj1+1)
        M[kc, :, mj0:mj1+1] = copy.deepcopy(data['valeurs'][:, dj0:dj1+1])
        kc += 1
    return((M,intervalle))


M,intervalle = creeM(aujourdhui)

print('matrice créée',jour_de_num[jours[0]],jour_de_num[jours[-1]])

def ni(x):
    return(noms.index(x))

#plt.plot(np.sum(M[ni('urgences'),:,:], axis = 0));plt.show()
'''
plt.plot(np.sum(M[ni('urgences'),:,:], axis = 0))
plt.plot(np.mean(M[ni('couvre-feu 18h-6h'),:,:], axis = 0))
plt.show()
plt.plot(lissage(np.sum(dataurge['valeurs'],axis=0)[-60:],7),'-o')
plt.plot(np.sum(dataurge['valeurs'],axis=0)[-60:],'-o')
plt.show()

'''

jaujourdhui = num_de_jour(aujourdhui) - jours[0]

def normalise_data(M,intervalle,lissageind = nombredelissage7,
                   lissagecont = nombredelissage7,
                   sanslissage = contextessanslissage):
    for (k,v) in enumerate(M):
        if noms[k] in nomsind: # + contextes_a_lisser
            # lissage sur 7 jours
            x0,x1 = intervalle[k] 
            for d in range(ndeps):
                for nl in range(lissageind):
                    v[d,x0:x1] = lissage(v[d,x0:x1],7)
        if noms[k] in nomscont: # normaliser les contextes avec la moyenne à 100 et le min a 0
            vmax = np.max(v)
            vmin = np.min(v)
            vm = np.mean(v)
            v[:,:] = 100 * (v - vmin)/(vm - vmin)
            if noms[k] not in sanslissage:
                x0,x1 = intervalle[k] 
                for d in range(ndeps):
                    for nl in range(lissagecont):
                        v[d,x0:x1] = lissage(v[d,x0:x1],7)

# pour tracer les courbes
Mreel,intervallereel = creeM(aujourdhui)
normalise_data(Mreel,intervallereel,lissageind = 1,lissagecont = 1,
               sanslissage = ['vacances','confinement',
                              'confinement+commerces',
                              'couvre-feu 21h-6h',
                              'couvre-feu 20h-6h',
                              'couvre-feu 18h-6h'])

#dependances(ni('résidence'),coefs)
#M[ni('urgences'),0,:100]

normalise_data(M,intervalle)

#M[ni('urgences'),0,:100]

# matrice des données relativisées à la population des départements

utiliser_proportions = True

def proportionsM(M):
    MR = copy.deepcopy(M)
    for x in range(nnoms):
        if utiliser_proportions and  noms[x] not in donnees_proportions:
            for d in range(ndeps):
                # on ramene a si c'était la France
                MR[x,d,:] = MR[x,d,:] / population_dep[departements[d]] * population_france 
    return(MR)

MR = proportionsM(M)

######################################################################
# matrice des dérivées

def aderiver(x):
    return(noms[x] in nomsind and noms[x] not in ['R'])

# pour calculer les decalages
def deriveM(MR,intervalle):
    MRD = copy.deepcopy(MR) 
    for x in range(nnoms):
        if aderiver(x):
            x0,x1 = intervalle[x]
            MRD[x,:,x0:x1] = derivee_indic(MR[x,:,x0:x1],7)
    return(MRD)

MRD = deriveM(MR,intervalle)

#plt.plot(np.sum(M[ni('urgences'),:,:], axis = 0));plt.show()

#touteslesdonnees
#plt.plot(np.transpose(np.mean(M[:,:,:300],axis=1)));plt.show()

######################################################################
# décalages et corrélations

decmax = 50 # pour les calculs
decmaxaccepte = 40 # au dela, on vire

def correlate(x,y):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return(np.correlate(x,y,mode = 'valid'))
    else:
        return(np.correlate(x,y,mode = 'valid')/(nx*ny))

def correlation(MRD,intervalle,x,y):
    #x = ni('travail')
    #y = ni('urgences')
    x0,x1 = intervalle[x]
    y0,y1 = intervalle[y]
    z1 = max(x1,y1)
    z0 = min(x0,y0)
    # vx longueur z1 - z0 + decmax
    # vy longueur z1 - z0 + decmax
    # se terminent le meme jour
    vxm = np.mean(MRD[x,:,x0:x1])
    vym = np.mean(MRD[y,:,y0:y1])
    vx = np.concatenate([np.zeros((ndeps,max(0,x0-z0 + decmax))),
                         MRD[x,:,x0:x1] - vxm,
                         np.zeros((ndeps,z1-x1))], axis = 1)
    vy = np.concatenate([np.zeros((ndeps,max(0,y0-z0 + decmax))),
                         MRD[y,:,y0:y1] - vym,
                         np.zeros((ndeps,z1-y1))], axis = 1)
    vvx = vx.flatten()
    vvy = vy.flatten()[decmax:]
    lcsm = correlate(vvx,vvy)
    #print(lcsm)
    d = np.argmax(np.abs(lcsm))
    corr = lcsm[d]
    d = decmax - d
    if d == 0 or d > decmaxaccepte:
        d = 0
        corr = lcsm[-1]
    xx0,xx1 = intervalle[x]
    yy0,yy1 = intervalle[y]
    x0 = max(-d,0,xx0,yy0-d)
    x1 = min(njours,njours-d,xx1,yy1-d)
    y0 = max(d,0,yy0,xx0+d)
    y1 = min(njours,njours+d,yy1,xx1+d)
    return([d,corr,x0,x1,y0,y1])

#correlation(MRD,intervalle,ni('travail'),ni('urgences'))
#correlation(MRD,intervalle,ni('recherche horaires google'),ni('urgences'))
#correlation(MRD,intervalle,ni('c.feu 20h-6h'),ni('urgences'))
#correlation(MRD,intervalle,ni('couvre-feu 20h-6h'),ni('urgences'))
'''
plt.plot(np.sum(M[ni('urgences'),:,:], axis = 0))
plt.plot(np.mean(M[ni('couvre-feu 20h-6h'),:,:], axis = 0))
plt.show()
'''


# contient les [decalage, correlation, xjour0,xjour1, yjour0,yjour1]
def calcule_correlations(M,MR,MRD,intervalle):
    coefs = np.zeros((nnoms,nnoms,6))
    for x in range(nnoms):
        print(noms[x], end=' ', flush = True)
        for y in range(nnoms):
            if x != y:
                #print('.', end='', flush = True)
                coefs[x,y] = correlation(MRD,intervalle,x,y)
                if noms[x] == 'travail' and noms[y] == 'urgences':
                    d,corr,x0,x1,y0,y1 = coefs[x,y]
                    print('-------- travail - urgences: decalage',d, 'correlation', corr)
    return(coefs)

noms_exclus_dependances = [] #[x for x in noms  if 'positif' in x]
#dataconfinement['confinement']

# abs(corr) > 0.2 and d >= 1

def predecesseurs(y,coefs, mincorr = mincorrelation):
    ldep = []
    for x in range(nnoms):
        if noms[x] not in noms_exclus_dependances:
            [d,corr,x0,x1,y0,y1] = coefs[x,y]
            if abs(corr) > mincorr and d >= mindecalage:
                ldep.append((x,d,corr))
    return(ldep)

def dependances(y,coefs):
    if False:
        ld = sorted(predecesseurs(y,coefs,mincorr = 0.10),
                    key = lambda x: - abs(x[2]))
        return([x for (x,d,c) in ld][:3])
    mc = mincorrelation
    ld = [x for (x,d,c) in predecesseurs(y,coefs,mincorr = mc)]
    if False and noms[y] in nomscont:
        ld = [x for x in ld if noms[x] in nomscont]
    while len(ld) <= mindependances and mc > 0.1:
        mc = mc * 5/6
        ld = [x for (x,d,c) in predecesseurs(y,coefs,mincorr = mc)]
        if False and noms[y] in nomscont:
            ld = [x for x in ld if noms[x] in nomscont]
    #print(mc)
    return(ld)

def erreur(p,graphe):
    s = 0
    sc = 0
    for x in range(nnoms):
        for y in range(nnoms):
            d = graphe[x,y,0]
            c = abs(graphe[x,y,1])
            if d != -1 and noms[x] in nomscont and noms[y] in nomsind:
                s += c * ((p[y] - p[x]) - d)**2
                sc += c
    return(np.sqrt(s/(sc)))

# on essaie de rendre cohérents les décalages entre eux
# en fait, c'est pas terrible à utiliser ensuite pour la prévision, ca lisse tout.
def decalages_coherents(coefs):
    print('calcul des decalages coherents')
    graphe = np.zeros((nnoms,nnoms,2)) - 1 # -1 pour l'infini: pas de correlation
    for y in range(nnoms):
        for (x,d,c) in predecesseurs(y,coefs):
            graphe[x,y] = np.array([d,c])
    p = np.array([random.randint(0,100) for x in range(nnoms)])
    emax = erreur(p,graphe)
    rate = 0
    while rate < 1000:
        x = random.randint(0,nnoms-1)
        dx = random.randint(-10,10)
        p[x] += dx
        e1 = erreur(p,graphe)
        if e1 >= emax:
            p[x] -= dx
            rate += 1 # nombre de raté de suite
        else:
            emax = e1
            printback(("%2.3f" % emax) + ' ' + str(rate))
            rate = 0
    print('erreur de décalage moyenne:',emax)
    for x in range(nnoms):
        for y in range(nnoms):
            d,corr,x0,x1,y0,y1 = coefs[x,y]
            d1 =  p[y] - p[x]
            dd = d1-d # il faut adapter les plages au nouveau décalage
            # et la ca lisse tout, on perd la correlation, en fait.
            if dd >= 0:
                xx0,xx1,yy0,yy1 = x0,x1-dd,y0+dd,y1
            else:
                xx0,xx1,yy0,yy1 = x0-dd,x1,y0,y1+dd
            coefs[x,y] = [d1,corr,xx0,xx1,yy0,yy1]

if nouveaucoefs:
    print('calcul des décalages et corrélations')
    coefs = calcule_correlations(M,MR,MRD,intervalle)
    #decalages_coherents(coefs)
    f = open('coefs.pickle','wb')
    pickle.dump(coefs,f)
    f.close()

f = open('coefs.pickle','rb')
coefs = pickle.load(f)
f.close()
print('fichier des décalages et corrélations chargé')

if False:
    print('---- dépendances:')
    for x in range(nnoms):
        print(noms[x],': ', end = '', flush = True)
        for y in dependances(x,coefs):
            print(noms[y],', ', end = '', flush = True)
        print('')

#coefs[ni('recherche horaires google'),ni('urgences')]
#coefs[ni('urgences'),ni('réanimations')]
#coefs[ni('urgences'),ni('décès')]
#coefs[ni('R'),ni('hospitalisations')]
#coefs[ni('couvre-feu 20h-6h'),ni('urgences')]
'''
L = [(x,int(coefs[ni(x),ni('hospitalisations')][1]*100),
      int(coefs[ni(x),ni('hospitalisations')][0])) for x in noms]
for (x,c,d) in L:
    if abs(c) > 20:
        print(x,d,c)

résidence 0 -40
vacances 27 -38
recherche Covid google 23 -42
confinement 23 -35

résidence 0 -40
vacances 27 -38
recherche Covid google 23 -42

plt.plot(np.mean(MF[ni('recherche Covid google'),:,:],axis=0))
plt.plot(np.mean(MF[ni('hospitalisations'),:,:],axis=0))
plt.plot(np.mean(MF[ni('décès'),:,:],axis=0))
plt.show()
'''



#dependances(ni('réanimations'))
#[noms[x] for x in dependances(ni('urgences'),coefs)]
#dependances(ni('taux positifs'))
#dependances(ni('positifs'))
#dependances(ni('température'))
#dependances(ni('précipitations sur 24'))
#dependances(ni('travail'))
'''
y = ni('couvre-feu 21h-6h')
for x in [(int(coefs[x,y][0]),"%0.2f" % coefs[x,y][1],noms[x][:25]) for x in dependances(y)]: print(x)
plt.plot(np.mean(MF[ni('couvre-feu 21h-6h'),:,:], axis=0));plt.show()
plt.plot(np.mean(MF[ni('température'),:,:], axis=0));plt.show()
plt.plot(np.mean(MF[ni('urgences'),:,:], axis=0));plt.show()
plt.plot(np.mean(MF[ni('température'),:,:], axis=0));plt.show()


'''
######################################################################
# coefficients de prevision
# calculés sur la matrice MRD des derivees des valeurs relatives
# avec les décalages et les dépendances donnés par les corrélations

erreurcoefs = [0]

def calcule_coefficients_france(y,M,MR,MRD,intervalle,coefs):
    ldepend = dependances(y,coefs)
    if ldepend == []:
        print('pas de dependance pour',noms[y])
        return(None)
    # plage de jours communs aux dependances
    ymin = 0
    ymax = njours
    for x in ldepend:
        [d,corr,x0,x1,y0,y1] = coefs[x,y]
        ymin = max(ymin,int(y0))
        ymax = min(ymax,int(y1))
        if noms[x] not in nomsprevus:
            ymax = min(ymax,intervalle[x][1])
    if noms[y] == 'urgences':
        print('jours coefficients',ymin,ymax)
    #plt.plot(np.sum(MRD[x,:,:],axis=0));plt.show()
    duree = ymax - ymin
    if duree <= 0: return(None)
    # on applatit tout
    L = np.zeros((ndeps*duree,len(ldepend)))
    H = np.zeros((ndeps*duree))
    for (kx,x) in enumerate(ldepend):
        [d,corr,x0,x1,y0,y1] = coefs[x,y]
        for dep in range(ndeps):
            L[dep*duree:dep*duree+duree,kx] = MRD[x,dep,
                                                  int(x0 + ymin - y0):int(x1 - (y1 - ymax))]
    for dep in range(ndeps):
        H[dep*duree:dep*duree+duree] = MRD[y,dep,ymin:ymax]
    C = np.zeros(len(ldepend))
    A = np.transpose(L) @ L
    B = np.transpose(L) @ H
    if abs(np.linalg.det(A)) < 1e-14:
        print('determinant nul:', np.linalg.det(A), 'max: ', noms[y])
    else:
        C = np.linalg.inv(A) @ B
    e = np.linalg.norm(L @ C - H) / np.linalg.norm(H) * 100
    #print('erreur',str(e) + '% ',noms[y])
    erreurcoefs[0] += e
    return([C,ldepend,ymin,ymax]) # la plage de jours prevus

def calcule_coefficients_dep(y,M,MR,MRD,intervalle,coefs):
    ldepend = dependances(y,coefs)
    if ldepend == []:
        print('pas de dependance pour',noms[y])
        return(None)
    # plage de jours communs aux dependances
    ymin = 0
    ymax = njours
    for x in ldepend:
        [d,corr,x0,x1,y0,y1] = coefs[x,y]
        ymin = max(ymin,int(y0))
        ymax = min(ymax,int(y1))
        if noms[x] not in nomsprevus:
            ymax = min(ymax,intervalle[x][1])
    if noms[y] == 'urgences':
        print('jours coefficients',ymin,ymax)
    duree = ymax - ymin
    L = np.zeros((ndeps,duree,len(ldepend)))
    H = np.zeros((ndeps,duree))
    for (kx,x) in enumerate(ldepend):
        [d,corr,x0,x1,y0,y1] = coefs[x,y]
        for dep in range(ndeps):
            L[dep,:,kx] = MRD[x,dep,
                              int(x0 + ymin - y0):int(x1 - (y1 - ymax))]
    for dep in range(ndeps):
        H[dep,:] = MRD[y,dep,ymin:ymax]
    C = np.zeros((ndeps,len(ldepend)))
    for dep in range(ndeps):
        A = np.transpose(L[dep]) @ L[dep]
        if np.linalg.det(A) < 1e-14:
            L[dep] = L[dep] + np.random.random((duree,len(ldepend))) * 0.1
            A = np.transpose(L[dep]) @ L[dep]
        B = np.transpose(L[dep]) @ H[dep]
        if abs(np.linalg.det(A)) < 1e-14:
            print('determinant nul:', dep, len(ldepend),np.linalg.det(A), 'max: ',
                  np.max(L[dep],axis=0), noms[y])
        else:
            C[dep] = np.linalg.inv(A) @ B
    e,h = 0,0
    for dep in range(ndeps):
        D = (L[dep] @ C[dep] - H[dep])
        e += np.transpose(D) @ D
        h += np.transpose(H[dep]) @ H[dep]
    er = np.sqrt(e) / (0.0001 + np.sqrt(h)) * 100
    if y == ni('urgences'):
        print('erreur',str(er) + '% ',noms[y])
    erreurcoefs[0] += er
    return([C,ldepend,ymin,ymax]) # la plage de jours prevus

def calcule_coefficients(y,M,MR,MRD,intervalle,coefs):
    if calculcoefdep:
        return(calcule_coefficients_dep(y,M,MR,MRD,intervalle,coefs))
    else:
        return(calcule_coefficients_france(y,M,MR,MRD,intervalle,coefs))

erreurcoefs[0] = 0
coefficients = [calcule_coefficients(y,M,MR,MRD,intervalle,coefs) for y in range(nnoms)]
print('--- erreur moyenne des coefficients de prevision:',
      str(erreurcoefs[0]/nnoms)+'%')

######################################################################
# prevision du lendemain répétée

nomsmin0 = [] #nomsind

# prevoit un jour en fonction des precedents
def prevoit_data(MF,MRF,MRDF,intervalle,coefficients,coefs,
                 y,futur,depart = aujourdhui, passe = False):
    if passe:
        if noms[y] in nomsprevus:
            jour = intervalle[y][1] - 1 + futur # prévoir les vacances? mouhaha!
        else:
            jour = min(num_de_jour(depart) - jours[0], intervalle[y][1] - 1) + futur
    else:
        jour = num_de_jour(depart) - jours[0] + futur
    if (passe or jour >= intervalle[y][1]) and jour < njours:
        # calcul de MRDF
        if coefficients[y] != None:
            [C,ldepend,ymin,ymax] = coefficients[y]
            L = np.zeros((ndeps,len(ldepend)))
            for (kx,x) in enumerate(ldepend):
                [d,corr,x0,x1,y0,y1] = coefs[x,y]
                L[:,kx] = copy.deepcopy(MRDF[x,:, jour - int(d)])
            for dep in range(ndeps):
                if calculcoefdep:
                    MRDF[y,dep,jour] = L[dep] @ C[dep]
                else:
                    MRDF[y,dep,jour] = L[dep] @ C
        else:
            if njours - jour == 10:
                print(njours - jour, 'pas prévu',jour_de_num[jour + jours[0]],noms[y])
            if aderiver(y):
                MRDF[y,:,jour] = 0
            else:
                MRDF[y,:,jour-1]
        # calcul de MRF: on intègre
        if aderiver(y):
            for dep in range(ndeps):
                MRF[y,dep,jour] = MRF[y,dep,jour-1] +  MRDF[y,dep,jour]
                if noms[y] in nomsmin0:
                    MRF[y,dep,jour] = max(0,MRF[y,dep,jour])
        else:
            for dep in range(ndeps):
                MRF[y,dep,jour] = MRDF[y,dep,jour]
                if noms[y] in nomsmin0:
                    MRF[y,dep,jour] = max(0,MRF[y,dep,jour])
        # calcul de MF: on remet la valeur absolue
        if utiliser_proportions and noms[y] not in donnees_proportions:
            for dep in range(ndeps):
                MF[y,dep,jour] = MRF[y,dep,jour] * population_dep[departements[dep]] / population_france
        else:
            MF[y,:,jour] = copy.deepcopy(MRF[y,:,jour])
        if futur <= 2:
            f = np.mean if noms[y] in donnees_proportions else np.sum
            if False:
                print('----- jour',jour,noms[y],'\n',
                      [int(f(MRDF[y,:,jour-k])) for k in range(8)][::-1],
                      [int(f(MRF[y,:,jour-k])) for k in range(8)][::-1],
                      [int(f(MF[y,:,jour-k])) for k in range(8)][::-1],
                      flush = True)

'''
y = ni('commerces et espaces de loisir (dont restaurants et bars)')
[C,ldepend,ymin,ymax] = coefficients[y]
#['travail', 'température', 'recherche itinéraire google']
L = np.zeros((ndeps,len(ldepend)))
for (kx,x) in enumerate(ldepend):
  [d,corr,x0,x1,y0,y1] = coefs[x,y]
  L[:,kx] = copy.deepcopy(MRDF[x,:, jour - int(d)])

for dep in range(ndeps):
  MRDF[y,dep,jour] = L[dep] @ C


y = ni('arrêts de transports en commun')
y = ni('travail')
[(noms[x],coefs[x,y][0:2]) for x in dependances(y,coefs)]
plt.plot(np.mean(M[y,:,:],axis=0))
plt.plot(np.mean(MF[y,:,:],axis=0))
plt.grid();plt.show()
plt.plot(np.mean(M[ni('à pied'),:,:],axis=0))
y = ni('urgences')
[(noms[x],coefs[x,y][0:2]) for x in dependances(y,coefs)]
ln = ['commerces et espaces de loisir (dont restaurants et bars)', "magasins d'alimentation et pharmacies", 'parcs', 'arrêts de transports en commun', 'travail', 'résidence', 'en voiture', 'à pied', 'en transport en commun']
plt.plot(np.transpose([np.mean(MF[x,:,:],axis=0) for x in [noms.index(y) for y in ln]]))
plt.grid();plt.show()

'''
def prevoit_tout(M,MR,MRD,intervalle,coefficients,coefs,
                 maxfutur, depart = aujourdhui, maxdata = 1.5, passe = False):
    # le premier jour <= depart ou manque une valeur
    depart0 = jour_de_num[np.min([jours[0] + intervalle[x][1]-1 for x in range(nnoms)]
                                 + [num_de_jour(depart)])]
    maxfutur = maxfutur + num_de_jour(depart) - num_de_jour(depart0) 
    print('prevision de tout à partir du ',depart0)
    MF = copy.deepcopy(M)
    MRF = copy.deepcopy(MR)
    MRDF = copy.deepcopy(MRD)
    for futur in range(1,maxfutur+1):
        #print('\n-------------------------------- prev',str(futur))
        printback(str(futur))
        for y in range(nnoms):
            #print(str(y) + ' ', end ='',flush=True)
            prevoit_data(MF,MRF,MRDF,intervalle,coefficients,coefs,
                         y,futur,depart = depart0, passe = passe)
    print('prevision finie')
    for x in range(nnoms):
        for dep in range(ndeps):
            m = np.max(np.abs(M[x,dep,:]))
            MF[x,dep,:] = np.maximum(np.minimum(MF[x,dep,:], m * maxdata), - m * maxdata)
    return((MF,MRF,MRDF,depart0))

jourstext = [jour_de_num[j] for j in jours]

def courbe_prev(nom,passe=100, futur = 50, dep = None):
    x = ni(nom)
    x0,x1 = intervalle[x]
    f = np.mean if nom in donnees_proportions else np.sum
    lj = jourstext[int(x1)-passe:int(x1)+futur]
    if dep == None:
        lv = f(MF[x,:,int(x1)-passe:int(x1)+futur], axis = 0)
    else:
        lv = MF[x,departements.index(dep),int(x1)-passe:int(x1)+futur]
    return((lj,lv,passe))

def trace_previsions(MF,MRF,MRDF,lnoms, passe = 200,futur = 60, dep = None):
    lcourbes = []
    print('previsions',','.join(lnoms))
    for nom in lnoms:
        x = ni(nom)
        x0,x1 = intervalle[x]
        f = np.mean if nom in donnees_proportions else np.sum
        if dep == None:
            lcourbes += [(zipper(jourstext[int(x1)-passe:int(x1)],
                                 f(Mreel[x,:,int(x1)-passe:int(x1)], axis = 0)),
                          '',real),
                         (zipper(jourstext[int(x1)-passe:int(x1)+futur],
                                 f(MF[x,:,int(x1)-passe:int(x1)+futur], axis = 0)),
                          nom,prev),
                         ]
        else:
            lcourbes += [(zipper(jourstext[int(x1)-passe:int(x1)],
                                 Mreel[x,departements.index(dep),int(x1)-passe:int(x1)]),
                          '',real),
                         (zipper(jourstext[int(x1)-passe:int(x1)+futur],
                                 MF[x,departements.index(dep),int(x1)-passe:int(x1)+futur]),
                          nom,prev),
                         ]
        trace(lcourbes,
              'prévisions ' + ' '.join(lnoms) + ('' if dep == None else str(dep))
              + ' (' + jour_de_num[jours[0] + x1] + ')',
          DIRSYNTHESE + '_prevision_' + '_'.join(lnoms) + ('' if dep == None else str(dep)))

x0,x1 = intervalle[ni('urgences')]
deb = 85
trace([(zipper(jourstext[x0:x1],np.sum(M[ni('urgences'),:,x0:x1],axis=0)),
        '',real),
       (zipper(jourstext[x0+deb:x1],[(j-deb)*830/(x1-x0-deb) for j in range(deb,x1-x0)]),
        '',prev)],
      'urgences pour covid19 (lissées, 88 départ.)\n'
      + 'pente: +' + str(int(830/(x1-x0-deb)*30)) + ' par mois',
      '_tendance_min_urgences',close = False)


######################################################################
# graphiques des prévisions à 60 jours

dureeprev = 90
MF,MRF,MRDF,depart0 = prevoit_tout(M,MR,MRD,intervalle,coefficients,coefs,
                           dureeprev)

#tout
if False:
    for x in nomsind:
        print(x)
        trace_previsions(MF,MRF,MRDF,[x],passe=100, futur = dureeprev)

dureeprev = 60 #90

# prévisions variées
if nouveauprev:
    trace_previsions(MF,MRF,MRDF,['urgences'], passe=100, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['nouv hospitalisations'], passe=100, futur = dureeprev)
    #trace_previsions(MF,MRF,MRDF,['sosmedecin'], passe=100, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['réanimations'],passe=100, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['hospitalisations'],passe=100, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['hospitalisation urgences'],passe=100, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['nouv réanimations'],passe=100, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['nouv décès'],passe=100, futur = dureeprev)
    if touteslesdonnees:
        trace_previsions(MF,MRF,MRDF,['hospi 09', 'hospi 19', 'hospi 29', 'hospi 39', 'hospi 49',
                                      'hospi 59', 'hospi 69', 'hospi 79', 'hospi 89', 'hospi 90'],
                         passe=100, futur = dureeprev)
        trace_previsions(MF,MRF,MRDF,['positifs 09', 'positifs 19', 'positifs 29', 'positifs 39',
                                      'positifs 49', 'positifs 59', 'positifs 69', 'positifs 79',
                                      'positifs 89', 'positifs 90'],
                         passe=80, futur = dureeprev)
        trace_previsions(MF,MRF,MRDF,['taux positifs 09', 'taux positifs 19', 'taux positifs 29',
                                      'taux positifs 39', 'taux positifs 49', 'taux positifs 59',
                                      'taux positifs 69', 'taux positifs 79', 'taux positifs 89',
                                      'taux positifs 90'],
                         passe=80, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['positifs'],passe=80, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['taux positifs'],passe=80, futur = dureeprev)
    # alpes maritimes
    trace_previsions(MF,MRF,MRDF,['urgences'],passe=100, futur = dureeprev,dep = 6)
    trace_previsions(MF,MRF,MRDF,['réanimations'],passe=100, futur = dureeprev,dep = 6)
    #trace_previsions(MF,MRF,MRDF,['sosmedecin'],passe=100, futur = dureeprev,dep = 6)
    trace_previsions(MF,MRF,MRDF,['taux positifs'],passe=80, futur = dureeprev,dep = 6)

    # R
    luj,lu,passe = courbe_prev('urgences')
    lu = r0(lissage(lu,7))
    lhj,lh,passe = courbe_prev('hospitalisation urgences')
    lh = r0(lissage(lh,7))

    trace([(zipper(luj[:passe],
                   lu[:passe]),
            '',real),
           (zipper(luj[passe:],
                   lu[passe:]),
            'R urgences',prev),
           (zipper(lhj[:passe],
                   lh[:passe]),
            '',real),
           (zipper(lhj[passe:],
                   lh[passe:]),
            'R hospi urge',prev)
           ],
          'prévision du taux de reproduction R',
          DIRSYNTHESE + '_prevision_R_par_urgences_hospi')

    luj,lu,passe = courbe_prev('urgences', dep= 6)
    lu = r0(lissage(lu,7))
    lhj,lh,passe = courbe_prev('hospitalisation urgences', dep= 6)
    lh = r0(lissage(lh,7))

    trace([(zipper(luj[:passe],
                   lu[:passe]),
            '',real),
           (zipper(luj[passe:],
                   lu[passe:]),
            'R urgences',prev),
           (zipper(lhj[:passe],
                   lh[:passe]),
            '',real),
           (zipper(lhj[passe:],
                   lh[passe:]),
            'R hospi urge',prev)
           ],
          'prévision du taux de reproduction R 06',
          DIRSYNTHESE + '_prevision_R_par_urgences_hospi 06')

######################################################################
# contextes influents
lcontinf = ['commerces et espaces de loisir (dont restaurants et bars)',
            "magasins d'alimentation et pharmacies",
            'arrêts de transports en commun',
            'travail',
            'résidence',
            'vacances']

if touteslesdonnees:
    lcontinf = ['commerces et espaces de loisir (dont restaurants et bars)',
                "magasins d'alimentation et pharmacies",
                'arrêts de transports en commun',
                'travail',
                'résidence',
                'humidité',
                'température',
                'vacances',
                'à pied',
                'en transport en commun',
                'recherche voyage google']

if nouveauprev:
    passe = 60
    trace([(zipper(jourstext[jaujourdhui-passe:min(jaujourdhui+1,intervalle[x][1])],
                   np.mean(Mreel[x,:,jaujourdhui-passe:min(jaujourdhui+1,intervalle[x][1])],
                           axis = 0)
                   if noms[x] != 'vacances'
                   else np.mean(Mreel[x,:,jaujourdhui-passe:min(jaujourdhui+1,intervalle[x][1])],
                                axis = 0)),
            noms[x][:25],'-')
           for x in [ni(x) for x in lcontinf]],
          'contextes corrélés',
          DIRSYNTHESE + '_contextes_influents',
          xlabel = 'random',
          fontcourbes = 6)
######################################################################
# recherches google et variation des urgences
if nouveauprev and touteslesdonnees:
    decalage = int(coefs[ni('recherche voyage google'),ni('urgences')][0])
    x0,x1 = intervalle[ni('urgences')][0]+30, jaujourdhui + 1
    xrech = intervalle[ni('recherche horaires google')][1]
    xurg = intervalle[ni('urgences')][1]
    lv = np.mean([15 * (np.mean(M[ni(x),:,x0-decalage:xrech], axis = 0)
                        - np.mean(M[ni(x),:,x0-decalage:xrech]))
                  for x in ['recherche horaires google',
                            'recherche voyage google',
                            'recherche itinéraire google']],
                 axis = 0)
    lv = lissage(700 + 10*lv,7)
    trace([(zipper(jourstext[x0:xrech+decalage],
                   lv),
            'rech + ' + str(decalage) + ' jours','-'),
           (zipper(jourstext[x0:xurg],
                   (np.sum(MRD[ni('urgences'),:,x0:xurg],axis = 0)
                    -(np.mean(np.sum(MRD[ni('urgences'),:,x0:xurg],axis = 0))))),
            'variation des urgences','-')],
          'recherche google voyage/itinéraire/horaire\n et variation des urgences',
          DIRSYNTHESE + '_recherche google',
          xlabel = 'random')

######################################################################
# dispersion des coefficients

def trace_coefficients(data):
    C = copy.deepcopy(coefficients[ni(data)][0])
    if calculcoefdep:
        C = np.mean(C,axis=0)
    depend = coefficients[ni(data)][1]
    # on normalise par les normes des contextes
    for (ky,y) in enumerate(depend):
        y0,y1 = intervalle[y]
        ny = np.linalg.norm(MRD[y,:,y0:y1]) / (ndeps*(y1-y0))
        C[ky] = C[ky] * ny
    C2 = [(depend[i],C[i]) for i in range(len(C))]
    C2 = sorted(C2, key = lambda x: -x[1])
    C = np.array([x[1] for x in C2])
    depend = [x[0] for x in C2]
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("coefficients des dépendances dans l'optimisation quadratique ("
                 + data + ")",
                 fontdict = {'size':6} )
    plt.plot(C,'o')
    plt.xticks(range(len(depend)),
               [noms[x][:20] for x in depend],
               rotation = 60,horizontalalignment='right',
               rotation_mode = 'anchor',
               fontsize = 8)
    plt.grid()
    plt.savefig(DIRSYNTHESE + '_coefficients_' + data + '.pdf', dpi = 600)
    plt.savefig(DIRSYNTHESE + '_coefficients_' + data + '.png', dpi = 600)
    #plt.show()

if nouveauprev:
    trace_coefficients('urgences')
    trace_coefficients('réanimations')
    trace_coefficients('hospitalisations')
    trace_coefficients('nouv décès')
    trace_coefficients('taux positifs')

    
'''
plt.plot(np.mean(MRDF[ni('résidence'),:,:],axis=0))
plt.plot(np.mean(MR[ni('résidence'),:,23:340],axis=0))
plt.plot(np.mean(MR[ni('couvre-feu 18h-6h'),:,0:317],axis=0))
plt.show()
correlate(np.mean(M[ni('résidence'),:,23:340],axis=0)-np.mean(M[ni('résidence'),:,23:340]),
          np.mean(M[ni('couvre-feu 18h-6h'),:,0:317],axis=0)-np.mean(M[ni('couvre-feu 18h-6h'),:,0:317]))
plt.plot(np.mean(M[ni('résidence'),:,23:340],axis=0)-np.mean(M[ni('résidence'),:,23:340]))
plt.plot(np.mean(M[ni('couvre-feu 18h-6h'),:,0:317],axis=0)-np.mean(M[ni('couvre-feu 18h-6h'),:,0:317]))
plt.show()
x=ni('couvre-feu 18h-6h')
y=ni('résidence')


correlation(MRD,intervalle,ni('couvre-feu 18h-6h'),ni('résidence'))
coefs[ni('couvre-feu 18h-6h'),ni('résidence')]
plt.plot([-0.38720458,-0.38963232,-0.39206656,-0.3945065 ,-0.3969609, -0.39935782
,-0.40175589,-0.4041622,-0.40657806,-0.40900728,-0.41144851,-0.41389928
,-0.41636107,-0.41882756,-0.4212996,-0.42377773,-0.42626206,-0.42875308
,-0.43124773,-0.43373698,-0.43622417,-0.43871206,-0.44119061,-0.44178226
,-0.44233753,-0.44279159,-0.44307648,-0.44323968,-0.44323601,-0.44296799
,-0.44245308,-0.44172287,-0.440802,-0.438587,-0.4363348,-0.43402796
,-0.43171083,-0.42067223,-0.40957099,-0.39842336,-0.3872281,-0.37596351
,-0.36465134,-0.35334108,-0.34204043,-0.33071024,-0.31936502,-0.30792083
,-0.29642239,-0.28490712,-0.27338413]);plt.show()
'''
######################################################################
# animation des prévisions passées

def tronqueM(M,intervalle,fin):
    M0 = copy.deepcopy(M)
    intervalle0 = copy.deepcopy(intervalle)
    for x in range(len(M)):
        if noms[x] not in nomsprevus:
            intervalle0[x] = (intervalle[x][0],fin)
            M0[x,:,fin+1:] = 0
    return((M0,intervalle0))

def cree_prev_duree(duree = 60, futur = 60, pas = 1, limiterfutur = True):
    M0,intervalle0 = creeM(aujourdhui)
    normalise_data(M0,intervalle0)
    x1 = min([x1 for (x0,x1) in intervalle0])
    lprev = []
    futur0 = futur
    for j in range(0,duree,pas):
        xdep = x1 - 1 - j
        depart = jour_de_num[jours[0] + xdep]
        print('---- prévision',-j,xdep)
        M,intervalle = tronqueM(M0,intervalle0,xdep)
        MR = proportionsM(M)
        MRD = deriveM(MR,intervalle)
        #coefs = calcule_correlations([M,MR,MRD,intervalle])
        if True:
            coefficients = [calcule_coefficients(y,M,MR,MRD,intervalle,coefs)
                            for y in range(nnoms)]
            futur0 = futur
            if limiterfutur:
                futur0 = int(14 + (futur - 14)*(duree-j)/duree)
        MF,MRF,MRDF,depart0 = prevoit_tout(M,MR,MRD,intervalle,coefficients,coefs,
                                           futur0, depart = depart, passe = True)
        lprev.append((j,futur0,x1,xdep,MF,depart0,coefficients))
    return(lprev)

def prev_lin(v,j,dj): # prevoit de j+1 à j+dj a partir de la pente en j
    try:
        dv = derivee(v,7)
        pv = [max(0,v[j] + dv[j] * k) for k in range(1,dj+1)]
        return(pv)
    except:
        return([0]*dj)

def prev_quad(v,j,dj): # prevoit de j+1 à j+dj a partir des dérivées 1 et 2 en j
    try:
        dv = derivee(v,7)
        ddv = derivee(dv,7)
        a = ddv[j] / 2
        b = dv[j] - 2 * a * j
        c = v[j] - a * j**2 - b * j
        pv = [max(0, a * (j+k)**2 + b * (j+k) + c)
              for k in range(1,dj+1)]
        return(pv)
    except:
        return([0]*dj)

def courbes_prev_duree(lprev,x,duree = 60, passe = 100, futur = 60, pas = 1, maxerreur = 100):
    M,intervalle = creeM(aujourdhui)
    normalise_data(M,intervalle)
    x0,x1 = intervalle[x]
    f = np.mean if noms[x] in donnees_proportions else np.sum
    reel = zipper(jourstext[int(x1)-passe:int(x1)],
                  f(Mreel[x,:,int(x1)-passe:int(x1)], axis = 0))
    lcourbes = []
    lC = []
    erreurs = dict([(j,[maxerreur]*(duree//7))
                    for j,futur0,x1,xdep,MF,depart0,coefficients in lprev])
    erreurslin = dict([(j,[maxerreur]*(duree//7))
                       for j,futur0,x1,xdep,MF,depart0,coefficients in lprev])
    erreursquad = dict([(j,[maxerreur]*(duree//7))
                       for j,futur0,x1,xdep,MF,depart0,coefficients in lprev])
    for j,futur0,x1,xdep,MF,depart0,coefficients in lprev:
        lC.append(coefficients)
        jdep = int(x1)-passe
        p = f(MF[x,:,jdep:xdep+futur0], axis = 0)
        plin = np.concatenate([p[:xdep-jdep],prev_lin(p[:xdep-jdep],xdep-jdep-1,futur0)])
        pquad = np.concatenate([p[:xdep-jdep],prev_quad(p[:xdep-jdep],xdep-jdep-1,futur0)])
        r = f(Mreel[x,:,jdep:xdep+futur0], axis = 0)
        lj = jourstext[jdep:xdep+futur0]
        for k in range(futur//7):
            if 0 <= passe - j + 7*k < len(p):
                erreurs[j][k] = min(abs(100 * (p[passe - j + 7*k] - r[passe - j + 7*k])
                                        /(1+ r[passe - j + 7*k])),
                                    maxerreur)
                erreurslin[j][k] = min(abs(100 * (plin[passe - j + 7*k] - r[passe - j + 7*k])
                                           /(1+ r[passe - j + 7*k])),
                                       maxerreur)
                erreursquad[j][k] = min(abs(100 * (pquad[passe - j + 7*k] - r[passe - j + 7*k])
                                           /(1+ r[passe - j + 7*k])),
                                       maxerreur)
        #print('--- jour',j, erreurs[j])
        # on vire dès qu'il y a une trop grande variation
        dp = [i for i in range(len(p)-1) if abs(p[i+1]-p[i])>abs(p[i])/3]
        if dp != []:
            p = p[:dp[0]]
            lj = lj[:dp[0]]
        lcourbes.append(zipper(lj,p))
    return((reel,lcourbes,lC,erreurs,erreurslin,erreursquad))

'''
x = ni('urgences')
passe = 100
futur = 60
pas = 1
plt.plot(f(M[x,:,int(x1)-passe:int(x1)], axis = 0))
plt.plot(f(MF[x,:,jdep:xdep+futur], axis = 0))
plt.show()
trace([(reel,'','-'),(lcourbes[0],'','-')],'','_',close=False)

'''

def axejours(lj):
    n = len(lj)
    lk = [n-1 - 7*k for k in range(n//7+1)][::-1]
    ljaxe = [joli(lj[k]) for k in lk]
    plt.xticks(lk,ljaxe,rotation = 70,fontsize = 8)

def val(jours,l):
    d = dict(l)
    return([d[j] if j in d else None for j in jours])

def mmax(l):
    if len(l) == 0:
        return(0)
    else:
        return(max(l))

#https://matplotlib.org/3.1.0/gallery/color/named_colors.html
def anime_previsions(lprev,nom, duree = 60, passe = 100, futur = 60):
    print('animation',nom)
    x = ni(nom)
    reel, previsions,lC,erreurs,erreurslin,erreursquad = courbes_prev_duree(lprev,x,
                                                                            duree = duree,
                                                                            passe = passe,
                                                                            futur = futur)
    previsions = previsions[::-1]
    courbes = [reel] + previsions
    fig, ax = plt.subplots()
    lj = []
    for courbe in courbes:
        lj = lj + [x[0] for x in courbe]
    lj = sorted(list(set(lj))) # liste des jours concernés par les courbes
    axejours(lj)
    plt.grid()
    plt.title('prévisions passées: ' + nom)
    # courbe réel
    lv = val(lj,reel)
    plt.plot(lv,'-', linewidth = 2, color = 'y')
    prev, = plt.plot(lj,val(lj,previsions[0]), '-')
    def init():
        ax.set_xlim(0, len(lj))
        ax.set_ylim(0,mmax([mmax([x[1] for x in c]) for c in courbes]))
        return prev, # rend un nuplet à un seul élément (le type de retour de plt.plot)
    def update(frame): #frame = indice dans previsions
        prev.set_data(lj,val(lj,previsions[frame]))
        return prev,
    def update2(frame): # garde les courbes précédentes, splendide dégradé de couleurs!
        p = frame/duree
        prev2, = plt.plot(lj,val(lj,previsions[frame]), '-', #color = 'tab:orange')
                          color = (1.,0.5 - 0.5 * p**3, 0.5 * p**3))
        prev, = plt.plot(lj,val(lj,reel[:-(duree-frame)]),'-', linewidth = 2, color = 'b')
        return (prev,prev2)
    ani = animation.FuncAnimation(fig, update2, frames= np.array(list(range(len(previsions)))),
                        init_func=init, blit=False)
    plt.rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'
    DPI=180
    writer = animation.FFMpegWriter(fps=5, bitrate=10000)
    ani.save(DIRSYNTHESE + "previsions_" + nom + ".mp4", writer = writer, dpi=DPI) 

'''
for i in range(100):
    p = i/100
    plt.plot([i+x for x in  range(100)],
             color = (0.7 * p,0.5 - 0.5 * p**3, 0.5 * p**3))

plt.show()
'''
#lprev = cree_prev_duree(duree = 5,futur = 60)v
#anime_previsions(lprev,'urgences',duree = 5, futur = 60)
#lprev = cree_prev_duree(duree = 30,futur = 60)
#anime_previsions(lprev,'urgences',duree = 30, futur = 60)
#lprev = cree_prev_duree(duree = 30,futur = 60)
#anime_previsions(lprev,'urgences',duree = 30, futur = 60)
######################################################################
# qualité des previsions

def evalue(lnom,duree = 60,passe = 100, maxerreur = 100):
    lprev = cree_prev_duree(duree = duree,futur = duree, limiterfutur = False)
    erreurs = {}
    for nom in lnom:
        x = ni(nom)
        reel, previsions,lC,ex,elinx,equadx = courbes_prev_duree(lprev,x, duree = duree,
                                                                 passe = passe,
                                                                 futur = duree,
                                                                 maxerreur = maxerreur)
        e7 = [[] for k in range(duree//7)]
        elin7 = [[] for k in range(duree//7)]
        equad7 = [[] for k in range(duree//7)]
        for k in range(duree//7):
            for j in ex:
                if j > 7*k:
                    e7[k].append(ex[j][k])
                    elin7[k].append(elinx[j][k])
                    equad7[k].append(equadx[j][k])
        erreurs[nom] = (e7,elin7,equad7)
    return((erreurs,lprev))




