import sys
from urlcache import *
from outils import *
import pickle
import matplotlib.animation as animation
np.seterr('raise')
######################################################################
# paramètres importants

nombredelissage7 = 2
mincorrelation = 0.034 # coefficients donnant une erreur minimale (Rnouv hospi)
mindecalage = 1
contextessanslissage = []
declargeur = 0
prolonger_contextes = True #False #True
jdebut = '2020-02-24' 
erreurmax_pour_prevoir = 40  # en %
utiliser_Reff_pour_prevoir = True
erreurmaxReff = 5

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
        import charge_indicateurs
        print('indicateurs chargés')
        try:
            import charge_contextes
            print('contextes chargés')
        except:
            print('*************************************probleme contexte')
            raise
    except:
        print('*************************************probleme indicateurs')
        raise
else:
    print('on utilise le cache')

print('nouveau',nouveau)
print('nouveauprev',nouveauprev)
print('nouveaucoefs',nouveaucoefs)

f = open(DIRCOVID19 + 'contextes.pickle','rb')
contextes = pickle.load(f)
f.close()
print('fichier des contextes chargé')

datamobilite, datameteo, datavacances, dataconfinement, dataapple, datahygiene, datagoogletrends, datagoogletrends_prev, regions, datapauvrete, lchamps_pauvrete, datapop, datavaccins, datavariants = contextes

f = open(DIRCOVID19 + 'indicateurs.pickle','rb')
indicateurs = pickle.load(f)
f.close()
print('fichier des indicateurs chargé')

# j'ai viré sosmedecin car que 43 departements le donnent

dataurge, datahospiurge, datareatot, datahospitot, datahospi, datarea, datadeces, datahospiage, dataposage, datapos, datatauxposage, datatauxpos, dataexcesdeces, datadeces17mai = indicateurs

contextes_non_temporels = [x[1] for x in lchamps_pauvrete] + ['population']


# à normaliser avec la population des départements
donnees_extensives_dep = ['vaccins', 'vaccins ehpad', 'urgences', 'hospitalisation urgences',
                          'réanimations', 'nouv hospitalisations', 'nouv réanimations', 'nouv décès',
                          'positifs', 'hospitalisations', 'hospi 0', 'hospi 09', 'hospi 19',
                          'hospi 29', 'hospi 39', 'hospi 49', 'hospi 59', 'hospi 69', 'hospi 79',
                          'hospi 89', 'hospi 90', 'positifs 09', 'positifs 19', 'positifs 29',
                          'positifs 39', 'positifs 49', 'positifs 59', 'positifs 69', 'positifs 79',
                          'positifs 89', 'positifs 90']
######################################################################
# on ajoute des Reff

# Reff par département
def dataReffdep(data):
    d = {'nom': 'R' + data['nom'],
         'titre': 'R effectif: ' + data['nom'],
         'dimensions': ['departements', 'jours'],
         'jours': data['jours'],
         'departements': data['departements'],
         'valeurs': np.array([np.array(r0(lissage(data['valeurs'][dep,:],7,repete = 2),
                                          derive=7,maxR = 3))
                              for dep in range(len(data['departements']))])}
    return(d)

#Reff national
def dataReffnat(data):
    r0d = r0(lissage(np.sum(data['valeurs'][:,:], axis = 0),7,repete = 2),derive=7,maxR = 3)
    d = {'nom': 'R' + data['nom'],
         'titre': 'R effectif: ' + data['nom'],
         'dimensions': ['departements', 'jours'],
         'jours': data['jours'],
         'departements': data['departements'],
         'valeurs': np.array([r0d for dep in range(len(data['departements']))])}
    return(d)

######################################################################
# on ne garde que les departements communs aux donnees

ldatacont = [datamobilite,datavacances]
if touteslesdonnees:
    ldatacont += [datameteo, dataapple, datavaccins, datavariants,
                  datagoogletrends_prev]

if inclusconfinement:
    ldatacont.append(dataconfinement)
    print('----------- on a inclus les données de confinement/couvre-feu')

ldataind = [dataurge, datahospiurge, datareatot,
            datahospi, datarea, datadeces, datahospitot,
            datapos, datatauxpos]

ldataReff = [dataReffdep(d) for d in ldataind]

ldataind = ldataind + ldataReff # + [datapos, datatauxpos]

if False and touteslesdonnees:
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
jfin = '2021-09-01'
jours = set([jour_de_num[j] for j in range(num_de_jour(jdebut),num_de_jour(jfin)+1)])
#jours = set([])
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
        if noms[k] in nomsind: 
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

mult_prop = 1e8
def proportionsM(M):
    MR = copy.deepcopy(M)
    for x in range(nnoms):
        if utiliser_proportions and  noms[x] in donnees_extensives_dep:
            for d in range(ndeps):
                # on ramene a 100 #si c'était la France
                MR[x,d,:] = MR[x,d,:] / population_dep[departements[d]] * mult_prop #population_france 
    return(MR)

MR = proportionsM(M)

######################################################################
# matrice des dérivées

def aderiver(x):
    return(noms[x] in nomsind
           and noms[x] not in [data['nom'] for data in ldataReff])

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
    x1 = min(x1,jaujourdhui)
    y0,y1 = intervalle[y]
    y1 = min(y1,jaujourdhui)
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

#plt.plot(vvx[:6000]);plt.plot(vvy[:6000]);plt.show()
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

#dataconfinement['confinement']

# abs(corr) > 0.2 and d >= 1

noms_exclus_dependances = nomsind[:] #[] #[x for x in noms  if 'positif' in x and 'taux' not in x]

indicateurs_pour_prevoir = []

def predecesseurs(y,coefs, mincorr = mincorrelation):
    ldep = []
    for x in range(nnoms):
        if noms[x] not in noms_exclus_dependances:
            [d,corr,x0,x1,y0,y1] = coefs[x,y]
            if abs(corr) > mincorr and d >= mindecalage:
                ldep.append((x,d,corr))
    return(ldep)

def dependances(y,intervalle,coefs):
    mc = mincorrelation
    ld = [x for (x,d,c) in sorted(predecesseurs(y,coefs,mincorr = mc),
                                  key = lambda x: - abs(x[2]))
          if aderiver(x) or x != y]
    ld1 = []
    for x in ld:
        if aderiver(x):
            ld1.append(10000+x) # signifie qu'on depend de MR aussi
    ld = ld + ld1
    if aderiver(y):
        ld.append(y)
        y0,y1 = intervalle[y]
        coefs[y,y] = [0,1.,y0,y1,y0,y1]
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

if nouveaucoefs:
    print('calcul des décalages et corrélations')
    coefs = calcule_correlations(M,MR,MRD,intervalle)
    #decalages_coherents(coefs)
    f = open(DIRCOVID19 + 'coefs.pickle','wb')
    pickle.dump(coefs,f)
    f.close()

f = open(DIRCOVID19 + 'coefs.pickle','rb')
coefs = pickle.load(f)
f.close()
print('fichier des décalages et corrélations chargé')

######################################################################
# coefficients de prevision
# calculés sur la matrice MRD des derivees des valeurs relatives
# avec les décalages et les dépendances donnés par les corrélations

contextes_a_prolonger = ['commerces et espaces de loisir (dont restaurants et bars)',
                         "magasins d'alimentation et pharmacies", 'parcs',
                         'arrêts de transports en commun', 'travail', 'résidence',
                         'en voiture', 'à pied', 'en transport en commun',
                         'recherche Covid google', 'recherche testcovid google',
                         'recherche pharmacie google', 'recherche horaires google',
                         'recherche voyage google', 'recherche itinéraire google',
                         'confinement', 'confinement+commerces', 'couvre-feu 21h-6h',
                         'couvre-feu 20h-6h', 'couvre-feu 18h-6h']

erreurcoefs = [[]]


# plage de jours communs aux dependances
def plage_jours_communs(y,ldepend,intervalle,coefs):
    ymin = 0
    ymax = njours
    for x in ldepend:
        if x < 10000:
            [d,corr,x0,x1,y0,y1] = coefs[x,y]
            d,x0,x1,y0,y1 = d,x0,x1,y0,y1
            ymin = max(ymin,int(y0))
            ymax = min(ymax,int(y1))
            if noms[x] not in nomsprevus:
                ymax = min(ymax,intervalle[x][1])
    return((ymin,ymax))

def valeurs_pour_prevoir(y,ymin,ymax,coefs,MR,MRD,x,dep):
    [d,corr,x0,x1,y0,y1] = coefs[(x if x < 10000 else x - 10000),y]
    d,x0,x1,y0,y1 = d,x0,x1,y0,y1
    if x == y:
        v = MR[x,dep,int(x0 + ymin - y0):int(x1 - (y1 - ymax))]
    elif x < 10000:
        v = MRD[x,dep,int(x0 + ymin - y0):int(x1 - (y1 - ymax))]
    else:
        v = MR[x-10000,dep,int(x0 + ymin - y0):int(x1 - (y1 - ymax))]
    return(v)

def calcule_coefficients(y,M,MR,MRD,intervalle,coefs):
    ldepend = dependances(y,intervalle,coefs)
    if ldepend == []:
        if noms[y][0] == 'R':
            erreurcoefs[0].append(100)
        print('pas de dependance pour',noms[y])
        return(None)
    ymin,ymax = plage_jours_communs(y,ldepend,intervalle,coefs)
    if noms[y] == 'urgences':
        print('jours coefficients',ymin,ymax)
    duree = ymax - ymin
    if duree <= 0: return(None)
    depnulles = []
    for dep in range(ndeps):
        for (kx,x) in enumerate(ldepend):
            v = valeurs_pour_prevoir(y,ymin,ymax,coefs,MR,MRD,x,dep)
            if np.sum(np.abs(v)) == 0 and kx not in depnulles:
                depnulles.append(kx)
    ldepend = [ldepend[k] for k in range(len(ldepend)) if k not in depnulles]
    C = np.zeros((ndeps,len(ldepend)))
    e = 0
    for dep in range(ndeps):
        L = np.zeros((duree,len(ldepend)))
        H = np.zeros(duree)
        for (kx,x) in enumerate(ldepend):
            L[:,kx] = valeurs_pour_prevoir(y,ymin,ymax,coefs,MR,MRD,x,dep)
        H[:] = MRD[y,dep,ymin:ymax]
        A = np.transpose(L) @ L
        B = np.transpose(L) @ H
        try:
            detA = abs(np.linalg.det(A))
        except:
            detA = 1e-15
        if detA < 1e-14:
            if detA != 0:
                print('determinant nul:', np.linalg.det(A), 'max: ', noms[y])
            else:
                print(dep,'determinant nul',noms[y])
        else:
            C[dep] = np.linalg.inv(A) @ B
        try:
            ee = np.linalg.norm(L @ C[dep] - H) / np.linalg.norm(H) * 100
            #print(C[dep],ee)
            e += ee
        except:
            #print('probleme norm')
            e += 100
    e = e/ndeps
    if noms[y][0] == 'R':
        erreurcoefs[0].append(e)
    if not prolonger_contextes or noms[y] in nomsind:
        print('erreur',("%2.3f" % e) + '% pour',noms[y],'dépendances:',len(ldepend))
        #for x in ldepend:
        #    print('\t', noms[x])
        if (noms[y] in nomsind and e < erreurmax_pour_prevoir
            and noms[y] not in indicateurs_pour_prevoir):
            indicateurs_pour_prevoir.append(noms[y])
    return([C,ldepend,ymin,ymax]) # la plage de jours prevus


# erreur sur les Reff en fonction de mincorrelation
if False:
    lerreur = []
    mc0 = mincorrelation
    nk = 100
    for k in range(nk):
        mincorrelation = 0.030 + k*(0.040 - 0.030)/nk
        erreurcoefs[0] = []
        indicateurs_pour_prevoir = [x for x in nomsind if x not in noms_exclus_dependances]
        coefficients = [calcule_coefficients(y,M,MR,MRD,intervalle,coefs)
                        for y in range(nnoms)]
        e = np.mean(erreurcoefs[0])
        print('--- erreur moyenne des coefficients de prevision:', ("%2.1f" % e) +'%')
        print('indicateurs utilisés pour prévoir:')
        for x in indicateurs_pour_prevoir:
            print(x)
        print('############################',mincorrelation,e)
        lerreur.append((mincorrelation,e))

    print(lerreur)
    plt.plot([m for (m,e) in lerreur],[e for (m,e) in lerreur])
    plt.grid()
    plt.show()
    mincorrelation = mc0

#mincorrelation = 0.034 # c'est e minimum d 'erreur sur les Reff: 13.4%

erreurcoefs[0] = []
indicateurs_pour_prevoir = [x for x in nomsind if x not in noms_exclus_dependances]
coefficients = [calcule_coefficients(y,M,MR,MRD,intervalle,coefs)
                for y in range(nnoms)]
e = np.mean(erreurcoefs[0])
print('--- erreur moyenne des coefficients de prevision:', ("%2.1f" % e) +'%')
print('indicateurs utilisés pour prévoir:')
for x in indicateurs_pour_prevoir:
    print(x)

'''       
for x in indicateurs_pour_prevoir:
    if x in noms_exclus_dependances:
        noms_exclus_dependances.remove(x)

# on recommence
coefficients = [calcule_coefficients(y,M,MR,MRD,intervalle,coefs)
                for y in range(nnoms)]
print('--- erreur moyenne des coefficients de prevision:',
      ("%2.1f" % np.mean(erreurcoefs[0])) +'%')
print('indicateurs utilisés pour prévoir:')
for x in indicateurs_pour_prevoir:
    print(x)
'''

######################################################################
# prevision du lendemain répétée
# le jour 0 est jours[0] (le premier des donnees des matrices)

def prolonge_contexte(MF,MRF,MRDF,y,jour):
    MF[y,:,jour] = MF[y,:,jour-1]
    MRF[y,:,jour] = MRF[y,:,jour-1]
    MRDF[y,:,jour] = MRDF[y,:,jour-1]

nomsmin0 = nomsind
# prevoit un jour en fonction des precedents
def prevoit_data(MF,MRF,MRDF,intervalle,coefficients,coefs,
                 y,futur,jourdebut = aujourdhui, prevoitdupasse = False, prevoitcontextes = True):
    if prevoitdupasse:
        if noms[y] in nomsprevus:
            jour = intervalle[y][1] - 1 + futur # prévoir les vacances? mouhaha!
        else:
            jour = min(jourdebut, intervalle[y][1] - 1) + futur
    else:
        jour = jourdebut + futur
    if (prevoitdupasse and not prevoitcontextes and jour < intervalle[y][1] and noms[y] in nomscont):
        return(0)
    elif (prevoitdupasse and not prevoitcontextes and jour >= intervalle[y][1] and jour < njours
          and prolonger_contextes and noms[y] in contextes_a_prolonger):
        prolonge_contexte(MF,MRF,MRDF,y,jour)
    else:
        if (prevoitdupasse or jour >= intervalle[y][1]) and jour < njours:
            if prolonger_contextes and noms[y] in contextes_a_prolonger:
                prolonge_contexte(MF,MRF,MRDF,y,jour)
            # calcul de MRDF
            else:
                if coefficients[y] != None:
                    [C,ldepend,ymin,ymax] = coefficients[y]
                    L = np.zeros((ndeps,len(ldepend)))
                    for (kx,x) in enumerate(ldepend):
                        [d,corr,x0,x1,y0,y1] = coefs[(x if x < 10000 else x - 10000),y]
                        if x == y:
                            L[:,kx] = copy.deepcopy(MRF[x,:, jour - int(d)])
                        elif x < 10000:
                            L[:,kx] = copy.deepcopy(MRDF[x,:, jour - int(d)])
                        else:
                            L[:,kx] = copy.deepcopy(MRF[x - 10000,:, jour - int(d)])
                    for dep in range(ndeps):
                        MRDF[y,dep,jour] = L[dep] @ C[dep]
                else:
                    if njours - jour == 10:
                        print(njours - jour, 'pas prévu',jour_de_num[jour + jours[0]],
                              noms[y])
                    if aderiver(y):
                        MRDF[y,:,jour] = 0
                    else:
                        MRDF[y,:,jour] = MRDF[y,:,jour-1]
                # calcul de MRF: on intègre
                Ry = 'R' + noms[y]
                if utiliser_Reff_pour_prevoir and Ry in indicateurs_pour_prevoir:
                    #print('--',Ry)
                    # on prevoit avec R effectif
                    yR = noms.index(Ry)
                    for dep in range(ndeps):
                        # f'/ f:
                        dvlog = math.log(max(0.01,MRF[yR,dep,jour - 1])) / intervalle_seriel
                        #if noms[y] == 'positifs':
                        #    print(jour,dvlog,MRF[y,dep,jour - 1],Ry)
                        MRF[y,dep,jour] = MRF[y,dep,jour - 1] + dvlog * MRF[y,dep,jour - 1]
                elif aderiver(y):
                    for dep in range(ndeps):
                        MRF[y,dep,jour] = MRF[y,dep,jour-1] +  MRDF[y,dep,jour]
                        if noms[y] in nomsmin0:
                            MRF[y,dep,jour] = max(0,MRF[y,dep,jour])
                else:
                    for dep in range(ndeps):
                        MRF[y,dep,jour] = MRDF[y,dep,jour]
                        if noms[y] in nomsmin0:
                            MRF[y,dep,jour] = max(0,MRF[y,dep,jour])
                # calcul de MF
                if utiliser_proportions and noms[y] in donnees_extensives_dep:
                    # on remet la valeur extensive
                    for dep in range(ndeps):
                        MF[y,dep,jour] = (MRF[y,dep,jour] * population_dep[departements[dep]]
                                          / mult_prop) #population_france)
                else:
                    MF[y,:,jour] = copy.deepcopy(MRF[y,:,jour])
                if futur <= 2:
                    f = np.mean if noms[y] not in donnees_extensives_dep else np.sum
    return(jour)

def integre2(v):
    n = len(v)
    s = 0
    for x in range(n):
        s += v[x] / (1.2**x)
    return(s)

def fit_homot(v,v0):
    k = 1
    dmin = np.linalg.norm(v - v0)
    dk = 0.1
    for i in range(20):
        kp = k*(1+dk)
        dp = np.linalg.norm(kp*v - v0)
        km = k*(1-dk)
        dm = np.linalg.norm(km*v - v0)
        if dp < dmin:
            dmin = dp
            k = kp
        elif dm < dmin:
            dmin = dm
            k = km
        else:
            dk = dk/2
    return(k)
    
    

# prevision à partir du jour jourdebut
# dureefutur est le nombre de jours du futur a prevoir
# prevoitcontextes indique si doit prevoir aussi les contextes apres le jour jourdebut
def prevoit_tout(M,MR,MRD,intervalle,coefficients,coefs,
                 dureefutur, jourdebut = aujourdhui, maxdata = 1.5, prevoitdupasse = False,
                 prevoitcontextes = True):
    # le premier jour <= jourdebut ou manque une valeur
    jourdebut0 = np.min([intervalle[x][1]-1 for x in range(nnoms)] + [jourdebut])
    dureefutur = dureefutur + jourdebut - jourdebut0
    print('prevision de tout à partir du ',jour_de_num[jourdebut0 + jours[0]])
    MF = copy.deepcopy(M)
    MRF = copy.deepcopy(MR)
    MRDF = copy.deepcopy(MRD)
    jfins = [0]*nnoms
    for dureefutur1 in range(1,dureefutur+1):
        printback(str(dureefutur1))
        for y in range(nnoms):
            if (prevoitcontextes
                or noms[y] not in nomscont
                or (noms[y] in nomscont and
                    jourdebut0 + dureefutur1 >= intervalle[y][1])):
                jfin = prevoit_data(MF,MRF,MRDF,intervalle,coefficients,coefs,
                                    y,dureefutur1,
                                    jourdebut = jourdebut0, prevoitdupasse = prevoitdupasse,
                                    prevoitcontextes = prevoitcontextes)
                jfins[y] = min(max(jfins[y],jfin),njours-1)
    print('prevision finie')
    # calage aujourdhui
    if True: # methode 1
        for y in range(nnoms):
            if noms[y] in nomsind and aderiver(y):
                j1 = min(intervalle[y][1]-1,jaujourdhui)
                for dep in range(ndeps):
                    # aujuste l'intégrale à l'intégrale réelle
                    som = integre2(M[y,dep,:j1])
                    somF = integre2(MF[y,dep,:j1])
                    if somF != 0:
                        MF[y,dep,:]  = MF[y,dep,:] /somF * som
                    # ajuste à aujourdhui
                    vm = M[y,dep,j1]
                    vmF = MF[y,dep,j1]
                    if vm * vmF != 0.:
                        for j in range(njours):
                            MF[y,dep,j]  = MF[y,dep,j] * math.exp(math.log(abs(vm / vmF)) * j / j1)
                    # minimise l'erreur par homothétie
                    k = fit_homot(MF[y,dep,:j1], M[y,dep,:j1])
                    MF[y,dep,:] = MF[y,dep,:] * k
                    # réajuste à aujourdhui linéairement sur les 100 derniers jours             
                    j0 = j1 - 100
                    v1F = MF[y,dep,j1]
                    v1 = M[y,dep,j1]
                    for j in range(j0,njours):
                        MF[y,dep,j]  = MF[y,dep,j] + (v1 - v1F)*min((j - j0)/(j1 - j0),1)
    # bornes sup et inf
    if True:
        for x in range(nnoms):
            for dep in range(ndeps):
                m = np.max(np.abs(M[x,dep,:]))
                MF[x,dep,:] = np.maximum(np.minimum(MF[x,dep,:], m * maxdata), - m * maxdata)
    return((MF,MRF,MRDF,jourdebut0))

def tronqueM(M,intervalle,fin):
    M0 = copy.deepcopy(M)
    intervalle0 = copy.deepcopy(intervalle)
    for x in range(len(M)):
        if noms[x] not in nomsprevus:
            intervalle0[x] = (intervalle[x][0],min(intervalle[x][1],fin))
            M0[x,:,fin+1:] = 0
    return((M0,intervalle0))

def prevoit_tout_deb_pres_fin(M,MR,MRD,intervalle,coefficients,coefs,
                              jourdebut,jourpresent,jourfin,
                              recalculecoeffficients = False): #jourfin exclus
    global indicateurs_pour_prevoir
    jjdebut = jourdebut + 1 + 80 # si pas positifs, 40 suffit; a cause des decalages pour prevoir
    jjpresent = jourpresent
    jjfin = jourfin
    dureefutur = jjfin - jjdebut
    M0,intervalle0 = tronqueM(M,intervalle,jjpresent)
    MR0 = proportionsM(M0)
    MRD0 = deriveM(MR0,intervalle0)
    if recalculecoeffficients: #True si on veut recalculer les coefficients à partir seulement du passé
        indicateurs_pour_prevoir = [x for x in nomsind if x not in noms_exclus_dependances]
        coefficients = [calcule_coefficients(y,M0,MR0,MRD0,intervalle0,coefs)
                        for y in range(nnoms)]
    M2F,M2RF,M2RDF,jourdebut0 = prevoit_tout(M0,MR0,MRD0,intervalle0,coefficients,coefs,
                                             dureefutur, jourdebut = jjdebut,
                                             prevoitdupasse = True,
                                             prevoitcontextes = False)
    return(M0,M2F,M2RF,M2RDF,jourdebut0)

'''
MT,M1F,M1RF,M1RDF,jourdebut0 =  prevoit_tout_deb_pres_fin(M,MR,MRD,intervalle,coefficients,coefs,
                                                          100,
                                                          jaujourdhui,
                                                          njours)
plt.plot(np.sum(M1F[ni('réanimations'),:,:],axis=0))
plt.plot(np.sum(MT[ni('réanimations'),:,:],axis=0))
plt.grid()
plt.show()

'''

jourstext = [jour_de_num[j] for j in jours]

def erreur_prevision(Mreel,MF,nom,passe = 2000):
    x = ni(nom)
    x0,x1 = intervalle[x]
    f = np.mean if nom not in donnees_extensives_dep else np.sum
    vr = f(Mreel[x,:,int(x1)-passe:int(x1)], axis = 0)
    vp = f(MF[x,:,int(x1)-passe:int(x1)], axis = 0)
    e = np.linalg.norm(vr - vp) / np.linalg.norm(vr) * 100
    return(e)

def trace_previsions(Mreel,MF,MRF,MRDF,
                     lnoms, passe = 200,futur = 60, deps = None, nomdeps = None, erreur = None):
    correctionpop = 1
    if deps == None:
        correctionpop = population_france / sum([population_dep[d] for d in departements])
    lcourbes = []
    print('previsions',','.join(lnoms), nomdeps, futur)
    for nom in lnoms:
        x = ni(nom)
        x0,x1 = intervalle[x]
        f = np.mean if nom not in donnees_extensives_dep else np.sum
        lcourbes += [(zipper(jourstext[int(x1)-passe:int(x1)],
                             [v*correctionpop
                              for v in f(Mreel[x,:,int(x1)-passe:int(x1)], axis = 0)]),
                      '',real),
                     (zipper(jourstext[int(x1)-passe:int(x1)+futur],
                             [v*correctionpop
                              for v in f(MF[x,:,int(x1)-passe:int(x1)+futur], axis = 0)]),
                      nom,prev),
        ]
    fichier = '_prevision_' + '_'.join(lnoms) + ('' if deps == None else nomdeps)
    trace(lcourbes,
          'prévisions ' + ' '.join(lnoms) 
          + ('' if deps == None else ' ' + nomdeps)
          + ' (' + jour_de_num[jours[0] + x1-1] + ')'
          + ('' if erreur == None or deps != None else ((" erreur moyenne: %2.1f" % erreur) + "%")),
          DIRSYNTHESE + fichier)
    return(fichier)

# on exclus les previsions ou l erreur moyenne sur Reff est > errmax

def trace_previsions_region(MF,MRF,MRDF,
                            Ratracer, passe = 200,futur = 60, deps = None, nomdeps = None,
                            erreurmax = erreurmaxReff):
    if deps == None:
        Mreeld = Mreel
        MFd = MF
    else:
        deps = [dep for dep in deps if dep in departements]
        Mreeld = np.zeros((nnoms,len(deps),njours))
        for (d,dep) in enumerate(deps):
            Mreeld[:,d,:] = Mreel[:,departements.index(dep),:]
        MFd = np.zeros((nnoms,len(deps),njours))
        for (d,dep) in enumerate(deps):
            MFd[:,d,:] = MF[:,departements.index(dep),:]
    atracer = []
    for (nom,_) in Ratracer:
        erreur = erreur_prevision(Mreeld,MFd,nom,passe = passe)
        if erreur <= erreurmax:
            atracer.append((nom, erreur))
    atracer = sorted(atracer, key = lambda x : x[1])
    fichiersR = []
    fichiers = []
    for (nom,err) in atracer:
        f = trace_previsions(Mreeld,MFd,MRF,MRDF,
                             [nom], passe = passe,futur = futur, deps = deps,
                             nomdeps = nomdeps, erreur = err)
        fichiersR.append(f)
        f = trace_previsions(Mreeld,MFd,MRF,MRDF,
                             [nom[1:]], passe = passe,futur = futur, deps = deps,
                             nomdeps = nomdeps)
        fichiers.append(f)
    return(atracer,fichiersR,fichiers)
    
x0,x1 = intervalle[ni('urgences')]
deb = 86
fin = 350
haut = 812
trace([(zipper(jourstext[x0:x1],np.sum(M[ni('urgences'),:,x0:x1],axis=0)),
        '',real),
       (zipper(jourstext[x0+deb:x1],[(j-deb)*haut/(fin-x0-deb) for j in range(deb,x1-x0)]),
        '',prev)],
      'urgences pour covid19 (lissées, 88 départ.)\n'
      + 'pente: +' + str(int(haut/(fin-x0-deb)*30)) + ' par mois',
      DIRSYNTHESE + '_tendance_min_urgences',close = True)

######################################################################
# graphiques des prévisions à 60 jours

MT,MF,MRF,MRDF,jourdebut0 = prevoit_tout_deb_pres_fin(M,MR,MRD,intervalle,coefficients,coefs,
                                                          0,
                                                          jaujourdhui,
                                                          njours)

dureeprev = 60

passe = jaujourdhui - (num_de_jour('2020-09-01') - jours[0])
# prévisions variées

Ratracer = []
for nom in [data['nom'] for data in ldataReff]:
    errx = erreur_prevision(Mreel,MF,nom,passe = passe)
    Ratracer.append((nom,errx))

Ratracer = sorted(Ratracer,key = lambda x: x[1])
print('R à tracer')
print(Ratracer)
iledefrance = regions['Ile-de-France']

if nouveauprev:
    atracerfrance = trace_previsions_region(MF,MRF,MRDF,
                                            Ratracer, passe=passe, futur = dureeprev,
                                            erreurmax = 5)
    atracer06 = trace_previsions_region(MF,MRF,MRDF,
                                        Ratracer, passe=passe, futur = dureeprev,
                                        deps = [6], nomdeps = 'Alpes-Maritimes',
                                        erreurmax = 10)
    atraceriledefrance = trace_previsions_region(MF,MRF,MRDF,
                                                 Ratracer, passe=passe, futur = dureeprev,
                                                 deps = iledefrance, nomdeps = 'Ile de France',
                                                 erreurmax = 8)
    f = open(DIRCOVID19 + 'atracer.pickle','wb')
    pickle.dump((atracerfrance,atracer06,atraceriledefrance),f)
    f.close()

f = open(DIRCOVID19 + 'atracer.pickle','rb')
atracerfrance,atracer06,atraceriledefrance = pickle.load(f)
f.close()
print('fichier des tracés chargé')


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
          #xlabel = 'random',
          fontcourbes = 6)
######################################################################
# recherches google et variation des urgences
indicateurrech= 'réanimations'

if nouveauprev and touteslesdonnees:
    decalage = int(coefs[ni('recherche voyage google'),ni(indicateurrech)][0])
    x0,x1 = intervalle[ni(indicateurrech)][0]+30, jaujourdhui + 1
    xrech = intervalle[ni('recherche horaires google')][1]
    xurg = intervalle[ni(indicateurrech)][1]
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
                   (np.sum(MRD[ni(indicateurrech),:,x0:xurg],axis = 0)
                    -(np.mean(np.sum(MRD[ni(indicateurrech),:,x0:xurg],axis = 0))))),
            'variation des urgences','-')],
          'recherche google voyage/itinéraire/horaire\n et variation des ' + indicateurrech,
          DIRSYNTHESE + '_recherche google',
          xlabel = 'random'
    )

######################################################################
# dispersion des coefficients

def trace_coefficients(data):
    C = copy.deepcopy(coefficients[ni(data)][0])
    #C = np.mean(C,axis=0)
    depend = coefficients[ni(data)][1]
    for (ky,y) in enumerate(depend):
        cmax = np.max(np.abs(C[:,ky]))
        if cmax != 0:
            C[:,ky] = C[:,ky] / cmax
    Cm = np.mean(C,axis=0)
    C3 = [(depend[i],Cm[i],C[:,i]) for i in range(len(depend))]
    C3 = sorted(C3, key = lambda x: -x[1])
    C = np.array([c for (d,cm,c) in C3])
    Cm = np.array([cm for (d,cm,c) in C3])
    depend = [d for (d,cm,c) in C3]
    fig = plt.figure(figsize=(6,6))
    fig.suptitle("coefficients des dépendances \ndans l'optimisation quadratique ("
                 + data + ")",
                 fontdict = {'size':4} )
    plt.plot(C,'o')
    plt.plot(Cm,'-')
    plt.xticks(range(len(depend)),
               [(noms[x][:20] if x < 10000 else noms[x - 10000][:20] + '(val)') for x in depend],
               rotation = 60,horizontalalignment='right',
               rotation_mode = 'anchor',
               fontsize = 5)
    plt.grid()
    plt.savefig(DIRSYNTHESE + '_coefficients_' + data + '.pdf', dpi = 600)
    plt.savefig(DIRSYNTHESE + '_coefficients_' + data + '.png', dpi = 600)
    #plt.show()

#trace_coefficients('Rurgences')
    
if nouveauprev:
    for (nom,err) in atracerfrance[0]:
        print('coefficients', nom, "%2.3f" % err)
        trace_coefficients(nom)

######################################################################
# animation des prévisions passées
def cree_prev_duree(duree = 60, dureefutur = 60, pas = 1):
    x1 = min([x1 for (x0,x1) in intervalle])
    lprev = []
    dureefutur0 = dureefutur
    for j in range(0,duree,pas):
        xdep = x1 - 1 - j
        jjdebut = 40
        jjpresent = xdep
        jjfin = xdep + dureefutur + 1
        print(jjdebut,jjpresent,jjfin)
        MT,M1F,M1RF,M1RDF,jourdebut0 = prevoit_tout_deb_pres_fin(M,MR,MRD,intervalle,coefficients,coefs,
                                                                 jjdebut,jjpresent,jjfin)
        #plt.plot(np.sum(M1F[ni('urgences'),:,:],axis=0))
        #plt.plot(np.sum(M[ni('urgences'),:,:],axis=0))
        #plt.grid()
        #plt.show()
        lprev.append((j,dureefutur,x1,xdep,M1F,jourdebut0,coefficients))
    return(lprev)

'''
dureefutur = 60 #60
duree = 90 # 250 
passe = 100 # 260
pasanime = 21

lprev = cree_prev_duree(duree = duree,dureefutur = dureefutur,pas = pasanime)

'''

def prev_lin(v,j,dj,largeur = 7): # prevoit de j+1 à j+dj a partir de la pente en j
    try:
        dv = derivee(v,largeur)
        pv = [max(0,v[j] + dv[j] * k) for k in range(1,dj+1)]
        return(pv)
    except:
        return([0]*dj)

def prev_quad(v,j,dj,largeur = 7): # prevoit de j+1 à j+dj a partir des dérivées 1 et 2 en j
    try:
        dv = derivee(v,largeur)
        ddv = derivee(dv,largeur)
        a = ddv[j] / 2
        b = dv[j] - 2 * a * j
        c = v[j] - a * j**2 - b * j
        pv = [max(0, a * (j+k)**2 + b * (j+k) + c)
              for k in range(1,dj+1)]
        return(pv)
    except:
        return([0]*dj)

def courbes_prev_duree(lprev,x,duree = 60, passe = 100, dureefutur = 60, pas = 1, maxerreur = 100):
    M,intervalle = creeM(aujourdhui)
    normalise_data(M,intervalle)
    x0,x1 = intervalle[x]
    f = np.mean if noms[x] not in donnees_extensives_dep else np.sum
    correctionpop = population_france / sum([population_dep[d] for d in departements])
    reel = zipper(jourstext[int(x1)-passe:int(x1)],
                  [correctionpop * v
                   for v in f(Mreel[x,:,int(x1)-passe:int(x1)], axis = 0)])
    lcourbes = []
    lC = []
    erreurs = dict([(j,[])
                    for j,dureefutur0,x1,xdep,MF,jourdebut0,coefficients in lprev])
    erreurslin = dict([(j,[])
                       for j,dureefutur0,x1,xdep,MF,jourdebut0,coefficients in lprev])
    erreursquad = dict([(j,[])
                       for j,dureefutur0,x1,xdep,MF,jourdebut0,coefficients in lprev])
    for j,dureefutur0,x1p,xdep,MF,jourdebut0,coefficients in lprev:
        dfutur = min(dureefutur0,dureefutur)
        lC.append(coefficients)
        jdep = int(x1)-passe
        # on lisse la prevision
        for d in range(ndeps):
            for nl in range(2):
                MF[x,d,jdep:xdep+dfutur] = lissage(MF[x,d,jdep:xdep+dfutur],7)
        p = f(MF[x,:,jdep:xdep+dfutur], axis = 0)
        plin = np.concatenate([p[:xdep-jdep],
                               prev_lin(p[:xdep-jdep],xdep-jdep-1,dfutur,largeur=15)])
        pquad = np.concatenate([p[:xdep-jdep],
                                prev_quad(p[:xdep-jdep],xdep-jdep-1,dfutur,largeur=15)])
        r = f(Mreel[x,:,jdep:xdep+dfutur], axis = 0)
        lj = jourstext[jdep:xdep+dfutur]
        for k in range(dureefutur//7):
            je = xdep + 7*k - jdep
            if 0 <= je < len(p) and je + jdep < x1:
                try:
                    erreurs[j].append((k, min(abs(100 * (p[je] - r[je])/(0.001+ r[je])),
                                              maxerreur)))
                    erreurslin[j].append((k, min(abs(100 * (plin[je] - r[je])/(0.001+ r[je])),
                                                 maxerreur)))
                    erreursquad[j].append((k, min(abs(100 * (pquad[je] - r[je])/(0.001+ r[je])),
                                                  maxerreur)))
                except:
                    print('probleme erreur',k,j,je,np.shape(erreurs))
                    #print('--- jour',j, erreurs[j])
        # on vire dès qu'il y a une trop grande variation d'un jour à l'autre
        dp = [i for i in range(len(p)-1) if abs(p[i+1]-p[i])>abs(p[i])/3]
        if dp != []:
            p = p[:dp[0]]
            lj = lj[:dp[0]]
        lcourbes.append(zipper(lj,[correctionpop * v for v in p]))
    return((reel,lcourbes,lC,erreurs,erreurslin,erreursquad))

'''
x = ni('urgences')
passe = 100
dureefutur = 60
pas = 1
plt.plot(f(M[x,:,int(x1)-passe:int(x1)], axis = 0))
plt.plot(f(MF[x,:,jdep:xdep+dureefutur], axis = 0))
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
######################################################################
# moyennes des prévisions, ponderees (plus avenir, moins de poids)
def prevision_moyenne(lprev,x):
    correctionpop = population_france / sum([population_dep[d] for d in departements])
    v = np.zeros(njours)
    nv = np.zeros(njours)
    xdepmin = njours
    f = np.mean if noms[x] not in donnees_extensives_dep else np.sum
    for j,dureefutur0,x1p,xdep,MF,jourdebut0,coefficients in lprev:
        xdepmin = min(xdep,xdepmin)
        xfin = min(njours,xdep+dureefutur0)
        for j in range(xdep,xfin):
            poids = (xfin - j)**2
            v[j] += f(MF[x,:,j],axis=0) * poids
            nv[j] += poids
    for j in range(njours):
        if nv[j] != 0:
            v[j] = v[j] / nv[j]
    for k in range(3):
        v = lissage(v,7)
    erreur = (np.linalg.norm(v[xdepmin:jaujourdhui]
                             - f(M[x,:,xdepmin:jaujourdhui],axis=0))
              / np.linalg.norm(f(M[x,:,xdepmin:jaujourdhui],axis=0))) * 100
    return([correctionpop * x for x in v],xdepmin,erreur)

#https://matplotlib.org/3.1.0/gallery/color/named_colors.html
dureefuturanime = 90
def anime_previsions(lprev,nom, duree = 60, passe = 100, dureefutur = dureefuturanime,pas = 1):
    print('animation',nom)
    x = ni(nom)
    reel, previsions,lC,erreurs,erreurslin,erreursquad = courbes_prev_duree(lprev,x,
                                                                            duree = duree,
                                                                            passe = passe,
                                                                            dureefutur = dureefutur,
                                                                            pas = pas)
    f = np.mean if noms[x] not in donnees_extensives_dep else np.sum
    pm,debpm,erreurm = prevision_moyenne(lprev,x)
    prevmoy = zipper(jourstext[debpm:],pm[debpm:])
    previsions = previsions[::-1]
    courbes = [reel] + previsions
    fig, ax = plt.subplots()
    lj = []
    for courbe in courbes:
        lj = lj + [x[0] for x in courbe]
    lj = sorted(list(set(lj))) # liste des jours concernés par les courbes
    axejours(lj)
    plt.grid()
    plt.title('prévisions passées: ' + nom + ' (erreur moyenne: ' + str(int(erreurm)) + '%)',
              fontdict = {'size':10})
    # courbe réel
    plt.plot(val(lj,reel),'-', linewidth = 2, color = 'y')
    prev, = plt.plot(lj,val(lj,previsions[0]), '-')
    def init():
        ax.set_xlim(0, len(lj))
        ax.set_ylim(0,mmax([mmax([x[1] for x in c]) for c in courbes]))
        return prev, # rend un nuplet à un seul élément (le type de retour de plt.plot)
    def update(frame): #frame = indice dans previsions
        prev.set_data(lj,val(lj,previsions[frame]))
        return prev,
    def update2(frame): # garde les courbes précédentes, splendide dégradé de couleurs!
        p = pas*frame/duree
        prev2, = plt.plot(lj,val(lj,previsions[frame]), '-', #color = 'tab:orange')
                          color = (1.,0.5 - 0.5 * p**3, 0.5 * p**3))
        prev, = plt.plot(lj,val(lj,reel[:-(duree-pas*frame)]),'-', linewidth = 2, color = 'b')
        prev, = plt.plot(lj,val(lj,prevmoy[:-(duree-pas*frame)]),'--', linewidth = 2, color = 'r')
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
#lprev = cree_prev_duree(duree = 5,dureefutur = 60)v
#anime_previsions(lprev,'urgences',duree = 5, dureefutur = 60)
#lprev = cree_prev_duree(duree = 30,dureefutur = 60)
#anime_previsions(lprev,'urgences',duree = 30, dureefutur = 60)
#lprev = cree_prev_duree(duree = 30,dureefutur = 60)
#anime_previsions(lprev,'urgences',duree = 30, dureefutur = 60)
######################################################################
# qualité des previsions

serreur = 'moyenne'

def evalue(lprev,lnom,duree = 60, dureefutur = 60,passe = 100,
           maxerreur = 100, pas = 1):
    erreurs = {}
    for nom in lnom:
        x = ni(nom)
        # ex est ordonné par jour de debut de prévision decroissants
        reel, previsions,lC,ex,elinx,equadx = courbes_prev_duree(lprev,x, duree = duree,
                                                                 passe = passe,
                                                                 dureefutur = dureefutur,
                                                                 maxerreur = maxerreur, pas = pas)
        e7 = [[] for k in range(dureefutur//7)]
        elin7 = [[] for k in range(dureefutur//7)]
        equad7 = [[] for k in range(dureefutur//7)]
        for j in sorted(ex):
            exj = dict(ex[j])
            elinxj = dict(elinx[j])
            equadxj = dict(equadx[j])
            for k in sorted(exj):
                e7[k].append(exj[k])
                elin7[k].append(elinxj[k])
                equad7[k].append(equadxj[k])
        print(len(e7[0]),nom)
        erreurs[nom] = (e7,elin7,equad7)
    return((erreurs,lprev))



