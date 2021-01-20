import sys
from urlcache import *
from outils import *
import pickle

######################################################################
# chargement des données

nouveau = False # False: on charge le fichier local
nouveauprev = False # seulement les previsions, on met pas a jour les donnees
nouveaucoefs = False
inclusconfinement = True #False

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

dataurge, datahospiurge, datasosmedecin, datareatot, datahospitot, datahospi, datarea, datadeces, datahospiage, dataposage, datapos, datatauxposage, datatauxpos, dataR, dataexcesdeces, datadeces17mai = indicateurs

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

ldatacont = [datamobilite, datameteo, datavacances,
             dataapple,
             datagoogletrends]

if inclusconfinement:
    ldatacont.append(dataconfinement)
    print('----------- on a inclus les données de confinement/couvre-feu')

ldataind = ([dataurge, datahospiurge, datasosmedecin, #dataR,
             datareatot,
             datahospi, datarea, datadeces, datapos, datatauxpos, datahospitot] 
            + [datahospiage[age] for age in sorted([a for a in datahospiage])
               #if age != '0'
            ]
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
def fin_donnees(data):
    if data['nom'] not in nomsprevus:
        return(present)
    else:
        return(data['jours'][-1])

intervalle = [None]*nnoms

def creeM(present):
    M = np.zeros((nnoms,ndeps,njours))
    kc= 0
    for data in ldatacont:
        lnoms = data[data['dimensions'][-1]]
        j0 = max(num_de_jour(data['jours'][0]),jours[0])
        j1 = min(num_de_jour(data['jours'][-1]),jours[-1])
        j1 = min(j1,num_de_jour(fin_donnees(data)))
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
        mj0 = jours.index(j0)
        mj1 = jours.index(j1)
        dj0 = data['jours'].index(jour_de_num[j0])
        dj1 = data['jours'].index(jour_de_num[j1])
        #print(data['nom'],dj0,dj1)
        intervalle[kc] = (mj0,mj1+1)
        M[kc, :, mj0:mj1+1] = copy.deepcopy(data['valeurs'][:, dj0:dj1+1])
        kc += 1
    return(M)

M = creeM(aujourdhui)

print('matrice créée',jour_de_num[jours[0]],jour_de_num[jours[-1]])

def ni(x):
    return(noms.index(x))

#plt.plot(np.sum(M[ni('urgences'),:,:], axis = 0));plt.show()
'''
plt.plot(np.sum(M[ni('urgences'),:,:], axis = 0))
plt.plot(np.mean(M[ni('couvre-feu 18h-6h'),:,:], axis = 0))
plt.show()
'''

jaujourdhui = num_de_jour(aujourdhui) - jours[0]

def normalise_data(m):
    for (k,v) in enumerate(m):
        if noms[k] in nomscont: # normaliser les contextes autour de la moyenne et ecart type
            vm = np.mean(v)
            ve = np.std(v)
            if ve != 0:
                v[:,:] = (v - vm) / ve * 100
        else: # lisser les indicateurs sur 7 jours
            x0,x1 = intervalle[k]
            for d in range(ndeps):
                v[d,x0:x1] = lissage77(v[d,x0:x1])

normalise_data(M)

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
    return(noms[x] in nomsind and noms[x] != 'R')

# pour calculer les decalages
def deriveM(MR):
    MRD = copy.deepcopy(MR) 
    for x in range(nnoms):
        if aderiver(x):
            x0,x1 = intervalle[x]
            MRD[x,:,x0:x1] = derivee_indic(MR[x,:,x0:x1],7)
    return(MRD)

MRD = deriveM(MR)

#plt.plot(np.sum(M[ni('urgences'),:,:], axis = 0));plt.show()

#tout
#plt.plot(np.transpose(np.mean(M[:,:,:300],axis=1)));plt.show()

######################################################################
# décalages et corrélations

decmax = 50 # pour les calculs
decmaxaccepte = 40 # au dela, on vire

def premier_max_local(l): #et plus grand que l[0], sinon l[0]
    lmax = [] # les max locaux
    for i in range(1,len(l)-1):
        if l[i] > l[0] and l[i-1] <= l[i] and l[i] >= l[i+1]:
            lmax.append(i)
    if lmax != []:
        i = lmax[0] # premier max local
        if len(lmax) >= 2:
            i2 = lmax[1]
            if l[i2] > 1.3 * l[i]: # deuxième max local meilleur que le premier
                return(i2)
        return(i)
    else:
        return(0)

def correlate(x,y):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return(np.correlate(x,y,mode = 'valid'))
    else:
        return(np.correlate(x,y,mode = 'valid')/(nx*ny))

def correlation(MRD,x,y):
    #x = ni('travail')
    #y = ni('urgences')
    x0,x1 = intervalle[x]
    y0,y1 = intervalle[y]
    z1 = max(x1,y1)
    z0 = min(x0,y0)
    # vx longueur z1 - z0 + decmax
    # vy longueur z1 - z0
    # se terminent le meme jour
    vx = np.concatenate([np.zeros((ndeps,max(0,x0-z0 + decmax))),
                         MRD[x,:,x0:x1],
                         np.zeros((ndeps,z1-x1))], axis = 1)
    vy = np.concatenate([np.zeros((ndeps,max(0,y0-z0))),
                         MRD[y,:,y0:y1],
                         np.zeros((ndeps,z1-y1))], axis = 1)
    vxm = np.mean(vx,axis=1)
    vym = np.mean(vy,axis=1)
    lcs = [correlate(vx[dep]-vxm[dep],vy[dep]-vym[dep])
           for dep in range(ndeps)]
    lcsm = np.mean(lcs,axis=0)
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

#correlation(MRD,ni('travail'),ni('urgences'))
#correlation(MRD,ni('recherche horaires google'),ni('urgences'))
#correlation(MRD,ni('c.feu 20h-6h'),ni('urgences'))
#correlation(MRD,ni('couvre-feu 20h-6h'),ni('urgences'))
'''
plt.plot(np.sum(M[ni('urgences'),:,:], axis = 0))
plt.plot(np.mean(M[ni('couvre-feu 20h-6h'),:,:], axis = 0))
plt.show()
'''


# contient les [decalage, correlation, xjour0,xjour1, yjour0,yjour1]
def calcule_correlations():
    coefs = np.zeros((nnoms,nnoms,6))
    for x in range(nnoms):
        print(noms[x], end=' ', flush = True)
        for y in range(nnoms):
            if x != y:
                #print('.', end='', flush = True)
                coefs[x,y] = correlation(MRD,x,y)
                if noms[x] == 'travail' and noms[y] == 'urgences':
                    d,corr,x0,x1,y0,y1 = coefs[x,y]
                    print('-------- travail - urgences: decalage',d, 'correlation', corr)
    return(coefs)

if nouveaucoefs:
    print('calcul des décalages et corrélations')
    coefs = calcule_correlations()
    f = open('coefs.pickle','wb')
    pickle.dump(coefs,f)
    f.close()

f = open('coefs.pickle','rb')
coefs = pickle.load(f)
f.close()
print('fichier des décalages et corrélations chargé')

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

noms_exclus_dependances = [] #dataconfinement['confinement']

# abs(corr) > 0.2 and d >= 1
def dependances(y):
    lcont = []
    lind = []
    for x in range(nnoms):
        if x not in noms_exclus_dependances:
            [d,corr,x0,x1,y0,y1] = coefs[x,y]
            if noms[y] in nomsind:
                if abs(corr) > 0.2 and d >= 1:
                    lind.append(x)
            if noms[y] in nomscont:
                if abs(corr) > 0.2 and d >= 1:
                    lcont.append(x)
    return(lcont + lind)

#dependances(ni('réanimations'))
#dependances(ni('urgences'))
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
# pour les prévisions
# calculés sur la matrice MRD des derivees des valeurs relatives

def calcule_coefficients(y):
    ldepend = dependances(y)
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
    duree = ymax - ymin
    L = np.zeros((ndeps,duree,len(ldepend)))
    H = np.zeros((ndeps,duree))
    for (kx,x) in enumerate(ldepend):
        [d,corr,x0,x1,y0,y1] = coefs[x,y]
        L[:,:,kx] = MRD[x,:,int(x0 + ymin - y0):int(x1 - (y1 - ymax))]
    H[:,:] = MRD[y,:,ymin:ymax]
    Q = np.identity(duree)
    C = np.zeros((ndeps,len(ldepend)))
    for dep in range(ndeps):
        A = np.transpose(L[dep]) @ Q @ L[dep]
        B = np.transpose(L[dep]) @ Q @ H[dep]
        if abs(np.linalg.det(A)) < 1e-14:
            print(dep, 'determinant nul:', np.linalg.det(A), 'max: ',np.max(L[dep]), noms[y])
        else:
            C[dep] = np.linalg.inv(A) @ B
    return([C,ldepend,ymin,ymax]) # la plage de jours prevus

#calcule_coefficients(ni('urgences'))

coefficients = [calcule_coefficients(y) for y in range(nnoms)]

#[(ldepend,ymin,ymax) for [C,ldepend,ymin,ymax] in coefficients]
'''
C = coefficients[ni('urgences')][0]
depend = coefficients[ni('urgences')][1]
C1 = C[:] / np.max(np.abs(C),axis = 0)
plt.plot(np.transpose(C1),'o')
plt.grid()
plt.show()
Cm = np.mean(C1,axis=0)
for kx,x in enumerate(depend):
  print(noms[x], ':', int(100*Cm[kx]))

plt.plot([noms[x][:3] for x in depend],np.transpose(C1),'o');plt.show()
ax = plt.gca();im = ax.imshow(C1,cmap="RdYlGn_r");plt.show()
'''
######################################################################
# prevision du lendemain répétée

# prevoit un jour en fonction des precedents
def prevoit_data(MF,MRF,MRDF,y,futur,depart = aujourdhui, passe = False):
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
                MRDF[y,dep,jour] = L[dep] @ C[dep]
        else:
            print('pas prévu',jour_de_num[jour + jours[0]],noms[y])
            MRDF[y,:,jour] = 0 #MRDF[y,:,jour-1]
        # calcul de MRF: on intègre
        if aderiver(y):
            for dep in range(ndeps):
                MRF[y,dep,jour] = max(0,MRF[y,dep,jour-1] +  (MRDF[y,dep,jour] + MRDF[y,dep,jour-1]) / 2)
        else:
            for dep in range(ndeps):
                MRF[y,dep,jour] = max(0,MRDF[y,dep,jour])
        # calcul de MF: on remet la valeur absolue
        if utiliser_proportions and noms[y] not in donnees_proportions:
            for dep in range(ndeps):
                MF[y,dep,jour] = MRF[y,dep,jour] * population_dep[departements[dep]] / population_france
        else:
            MF[y,:,jour] = copy.deepcopy(MRF[y,:,jour])

def prevoit_tout(maxfutur, depart = aujourdhui, maxdata = 1.5, passe = False):
    # le premier jour ou manque une valeur
    depart0 = jour_de_num[np.min([jours[0] + intervalle[x][1]-1 for x in range(nnoms)]
                                 + [num_de_jour(depart)])]
    maxfutur = maxfutur + num_de_jour(depart) - num_de_jour(depart0) 
    print('prevision de tout à partir du ',depart0)
    MF = copy.deepcopy(M)
    MRF = copy.deepcopy(MR)
    MRDF = copy.deepcopy(MRD)
    for futur in range(1,maxfutur+1):
        print(str(futur) + ' ', end = '', flush = True)
        for y in range(nnoms):
            #print(str(y) + ' ', end ='',flush=True)
            prevoit_data(MF,MRF,MRDF,y,futur,depart = depart0, passe = passe)
    print('prevision finie')
    for x in range(nnoms):
        for dep in range(ndeps):
            m = np.max(np.abs(M[x,dep,:]))
            MF[x,dep,:] = np.maximum(np.minimum(MF[x,dep,:], m * maxdata), - m * maxdata)
    return((MF,MRF,MRDF))

jourstext = [jour_de_num[j] for j in jours]

DIRSYNTHESE2 = 'synthese2/'

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
            lcourbes += [(zipper(jourstext[int(x1)-passe:int(x1)+futur],
                                 f(MF[x,:,int(x1)-passe:int(x1)+futur], axis = 0)),
                          nom,prev),
                         (zipper(jourstext[int(x1)-passe:int(x1)],
                                 f(M[x,:,int(x1)-passe:int(x1)], axis = 0)),
                          '',real)]
        else:
            lcourbes += [(zipper(jourstext[int(x1)-passe:int(x1)+futur],
                                 MF[x,departements.index(dep),int(x1)-passe:int(x1)+futur]),
                          nom,prev),
                         (zipper(jourstext[int(x1)-passe:int(x1)],
                                 M[x,departements.index(dep),int(x1)-passe:int(x1)]),
                          '',real)]
        trace(lcourbes,
          'prévisions ' + ' '.join(lnoms) + ('' if dep == None else str(dep)),
          DIRSYNTHESE2 + '_prevision_' + '_'.join(lnoms) + ('' if dep == None else str(dep)))

######################################################################
# graphiques des prévisions à 60 jours

dureeprev = 90
MF,MRF,MRDF = prevoit_tout(dureeprev)

#tout
if False:
    for x in nomsind:
        print(x)
        trace_previsions(MF,MRF,MRDF,[x],passe=100, futur = dureeprev)

dureeprev = 90

# prévisions variées
if nouveauprev:
    trace_previsions(MF,MRF,MRDF,['urgences', 'nouv hospitalisations','sosmedecin'],passe=100, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['réanimations'],passe=100, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['hospitalisations'],passe=100, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['nouv réanimations', 'nouv décès'],passe=100, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['hospi 09', 'hospi 19', 'hospi 29', 'hospi 39', 'hospi 49', 'hospi 59', 'hospi 69', 'hospi 79', 'hospi 89', 'hospi 90'],passe=100, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['positifs'],passe=80, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['positifs 09', 'positifs 19', 'positifs 29', 'positifs 39', 'positifs 49', 'positifs 59', 'positifs 69', 'positifs 79', 'positifs 89', 'positifs 90'],passe=80, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['taux positifs'],passe=80, futur = dureeprev)
    trace_previsions(MF,MRF,MRDF,['taux positifs 09', 'taux positifs 19', 'taux positifs 29', 'taux positifs 39', 'taux positifs 49', 'taux positifs 59', 'taux positifs 69', 'taux positifs 79', 'taux positifs 89', 'taux positifs 90'],passe=80, futur = dureeprev)

    # alpes maritimes
    trace_previsions(MF,MRF,MRDF,['urgences','réanimations','sosmedecin'],passe=100, futur = dureeprev,dep = 6)
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
          DIRSYNTHESE2 + '_prevision_R_par_urgences_hospi')

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
          DIRSYNTHESE2 + '_prevision_R_par_urgences_hospi 06')

######################################################################
# contextes influents
if nouveauprev:
    passe = 60
    trace([(zipper(jourstext[jaujourdhui-passe:min(jaujourdhui+1,intervalle[x][1])],
                   lissage(np.mean(M[x,:,jaujourdhui-passe:min(jaujourdhui+1,intervalle[x][1])],
                                   axis = 0),7)
                   if noms[x] != 'vacances'
                   else np.mean(M[x,:,jaujourdhui-passe:min(jaujourdhui+1,intervalle[x][1])],
                                   axis = 0)),
            noms[x][:25],'-')
           for x in [ni(x) for x in ['commerces et espaces de loisir (dont restaurants et bars)',
                                     "magasins d'alimentation et pharmacies",
                                     'arrêts de transports en commun',
                                     'travail',
                                     'résidence',
                                     'humidité',
                                     'température',
                                     'vacances',
                                     'à pied',
                                     'en transport en commun',
                                     'recherche voyage google']]],
          'contextes corrélés',
          DIRSYNTHESE2 + '_contextes_influents',
          xlabel = 'random')
######################################################################
# recherches google et urgences
if nouveauprev:
    decalage = 17
    x0,x1 = intervalle[ni('urgences')][0]+30, jaujourdhui + 1
    xrech = intervalle[ni('recherche horaires google')][1]
    xurg = intervalle[ni('urgences')][1]

    lv = np.mean([np.mean(M[ni(x),:,x0-decalage:xrech], axis = 0)
                  for x in ['recherche horaires google',
                            'recherche voyage google',
                            'recherche itinéraire google']],
                 axis = 0)
    lv = lissage(200 + 10*lv,7)

    trace([(zipper(jourstext[x0:xrech+decalage],
                   lv),
            'rech + ' + str(decalage) + ' jours','-'),
           (zipper(jourstext[x0:xurg],
                   np.sum(M[ni('urgences'),:,x0:xurg],axis = 0)),
            'urgences','-')],
          'recherche google voyage/itinéraire/horaire',
          DIRSYNTHESE2 + '_recherche google',
          xlabel = 'random')

######################################################################
# qualité des previsions

def compare_prev(duree):
    pas = 7
    lprev = {}
    for j in range(1,duree,1): # prevision  il y a j jours
        depart = jour_de_num[num_de_jour(aujourdhui) - j]
        print('départ',depart)
        MF,MRF,MRDF = prevoit_tout(duree, depart = depart)
        lprev[j] = MF
    ldj = list(range(pas,duree,pas))
    erreursfr = np.zeros((nnoms,len(ldj),2))
    for x in range(nnoms):
        f = np.mean if noms[x] in donnees_proportions else np.sum
        for k,dj in enumerate(ldj): # previsions à dj jours
            edjrel = []
            for j in lprev: # prevision a partir du jour -j
                MF = lprev[j]
                if dj <= j:
                    jp = num_de_jour(aujourdhui) - j + dj - jours[0]
                    if jp < intervalle[x][1]:
                        e = np.abs((f(MF[x,:,jp]) - f(M[x,:,jp])) / (1 + f(M[x,:,jp])))
                        if False and noms[x] == 'urgences':
                            print(f(MF[x,:,jp]),f(M[x,:,jp]),x,jp)
                        edjrel.append(e)
            if edjrel != []:
                erreursfr[x,k] = [x,np.mean(edjrel)]
    erreursfr = np.array(sorted(erreursfr, key = lambda x : x[-1,-1]))
    for e in erreursfr:
        x = e[0,0]
        nom = noms[int(x)]
        if nom in ['urgences', 'hospitalisation urgences', 'sosmedecin', 'R', 'réanimations', 'nouv hospitalisations', 'nouv réanimations', 'nouv décès', 'positifs', 'taux positifs', 'hospitalisations', 'décès']:
            print(nom)
            for (k,dj) in enumerate(ldj):
                print('erreur moyenne à ' + str(dj) + ' jours: ' + ("%2.1f" % (100*e[k,1])) + '%')
    return(erreursfr)

#e = compare_prev(60) # 3600 previsions
