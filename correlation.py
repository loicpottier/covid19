import sys
from urlcache import *
from outils import *
import pickle

######################################################################
# chargement des données

nouveau = False # False: on charge le fichier local
nouveauprev = True # seulement les previsions, on met pas a jour les donnees

if len(sys.argv) > 1 and sys.argv[1] == 'nouveau':
    nouveau = True
    nouveauprev = True

if len(sys.argv) > 1 and sys.argv[1] == 'nouveauprev':
    nouveau = False
    nouveauprev = True

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

f = open('contextes.pickle','rb')
contextes = pickle.load(f)
f.close()
print('fichier des contextes chargé')

datamobilite, datameteo, datavacances, dataapple, datahygiene, datagoogletrends, datagoogletrends_prev, regions, datapauvrete, lchamps_pauvrete, datapop = contextes

f = open('indicateurs.pickle','rb')
indicateurs = pickle.load(f)
f.close()
print('fichier des indicateurs chargé')

dataurge, datahospiurge, datasosmedecin, datareatot, datahospitot, datadecestot, datahospi, datarea, datadeces, datahospiage, dataposage, datapos, datatauxposage, datatauxpos, dataR, dataexcesdeces, datadeces17mai = indicateurs

contextes_non_temporels = [x[1] for x in lchamps_pauvrete] + ['population'] 
######################################################################
# on ne garde que les departements communs aux donnees

ldatacont = [datamobilite, datameteo, datavacances, dataapple,
             datagoogletrends]

ldataind = ([dataurge, datahospiurge, datasosmedecin, dataR, datareatot,
             datahospi, datarea, datadeces, datapos, datatauxpos, datahospitot, datadecestot] 
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

present = aujourdhui
def fin_donnees(data):
    if data['nom'] not in ['vacances']:
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

donnees_proportions = ['commerces et espaces de loisir (dont restaurants et bars)',
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
                       'taux positifs 29', 'taux positifs 39', 'taux positifs 49', 'taux positifs 59',
                       'taux positifs 69', 'taux positifs 79', 'taux positifs 89', 'taux positifs 90']

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
# c'est les décalages qui prennent du temps

decmax = 40

# x et y indices de donnees dans m
# x est decalé dans le passé
def ldecalages(M,x,y):
    lc = [None]*(decmax+1)
    for d in range(decmax+1): # d décalage dans le passé
        #print(d)
        xx0,xx1 = intervalle[x]
        yy0,yy1 = intervalle[y]
        x0 = max(-d,0,xx0,yy0-d)
        x1 = min(njours,njours-d,xx1,yy1-d)
        y0 = max(d,0,yy0,xx0+d)
        y1 = min(njours,njours+d,yy1,xx1+d)
        corr = np.corrcoef(M[x,:,x0:x1].flatten(),
                           M[y,:,y0:y1].flatten())
        #print(d,x0,x1,y0,y1, intervalle[x],intervalle[y],njours,corr[0,1])
        #print(corr[0,1])
        lc[d] = [corr[0,1],x0,x1,y0,y1]
    #plt.show()
    return(lc)

#ldecalages(MRD,ni('travail'),ni('urgences'))
#ldecalages(MRD,ni('résidence'),ni('urgences'))

#plt.plot([x[0] for x in ldecalages(MRD,ni('travail'),ni('urgences'))])
#plt.plot([x[0] for x in ldecalages(MRD,ni('recherche horaires google'),ni('urgences'))])
#plt.grid();plt.show()

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

def best_corr(lc):
    lcc = lissage([c for [c,x0,x1,y0,y1] in lc],7)
    if abs(max(lcc)) > abs(min(lcc)):
        dmax = premier_max_local(lcc)
        if dmax == 0:
            cmax = lc[0][0]
        else:
            cmax = lcc[dmax]
        [c,x0,x1,y0,y1] = lc[dmax]
        return(([cmax,x0,x1,y0,y1],dmax))
    else:
        dmin = premier_max_local([-x for x in lcc])
        if dmin == 0:
            cmin = lc[0][0]
        else:
            cmin = lcc[dmin]
        [c,x0,x1,y0,y1] = lc[dmin]
        #print(decmin,dmin,lc[:5],lcc[:5])
        return(([cmin,x0,x1,y0,y1],dmin))

#best_corr(ldecalages(MRD,ni('travail'),ni('urgences')))
#best_corr(ldecalages(MRD,ni('urgences'),ni('travail')))

def correlation(M,x,y):
    lc = ldecalages(M,x,y)
    cjp,dp = best_corr(lc)
    return([dp] + cjp)

######################################################################
# nouvelle version plus rapide

def correlation(MRD,x,y):
    #x = ni('travail')
    #y = ni('urgences')
    x0,x1 = intervalle[x]
    y0,y1 = intervalle[y]
    z1 = max(x1,y1)
    z0 = min(x0,y0)
    vx = np.concatenate([np.zeros((ndeps,x0-z0 + decmax)),
                         MRD[x,:,x0:x1],
                         np.zeros((ndeps,z1-x1))], axis = 1)
    vy = np.concatenate([np.zeros((ndeps,y0-z0)),
                         MRD[y,:,y0:y1],
                         np.zeros((ndeps,z1-y1))], axis = 1)
    vxm = np.mean(vx,axis=1)
    vym = np.mean(vy,axis=1)
    lcs = [np.correlate(vx[dep]-vxm[dep],vy[dep]-vym[dep],mode = 'valid')
           /(np.linalg.norm(vx[dep]-vxm[dep])*np.linalg.norm(vy[dep]-vym[dep]))
           for dep in range(ndeps)]
    lcsm = np.mean(lcs,axis=0)
    d = np.argmax(np.abs(lcsm))
    corr = lcsm[d]
    d = decmax - d
    xx0,xx1 = intervalle[x]
    yy0,yy1 = intervalle[y]
    x0 = max(-d,0,xx0,yy0-d)
    x1 = min(njours,njours-d,xx1,yy1-d)
    y0 = max(d,0,yy0,xx0+d)
    y1 = min(njours,njours+d,yy1,xx1+d)
    return([d,corr,x0,x1,y0,y1])

#correlation(MRD,ni('travail'),ni('urgences'))
#correlation(MRD,ni('recherche horaires google'),ni('urgences'))

print('calcul des décalages et corrélations')
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

if nouveau:
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

def dependances(y):
    lcont = []
    lind = []
    for x in range(nnoms):
        [d,corr,x0,x1,y0,y1] = coefs[x,y]
        if noms[y] in nomsind:
            if abs(corr) > 0.2 and d >= 1:
                lind.append(x)
        if noms[y] in nomscont:
            if abs(corr) > 0.2 and d >= 1:
                lcont.append(x)
    return((lcont,lind))

#dependances(ni('réanimations'))
#dependances(ni('urgences'))
#dependances(ni('taux positifs'))
#dependances(ni('positifs'))
#dependances(ni('température'))
#dependances(ni('précipitations sur 24'))
#dependances(ni('travail'))
######################################################################
# coefficients de prevision
# pour les prévisions
# calculés sur la matrice MRD des derivees des valeurs relatives

def calcule_coefficients(y):
    lcont,lind = dependances(y)
    ldepend = lcont + lind
    if ldepend == []:
        print('probleme pas de dependance',noms[y])
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

######################################################################
# prevision du lendemain répétée

# prevoit un jour en fonction des precedents
def prevoit_data(MF,MRF,MRDF,y,futur,depart = aujourdhui):
    if noms[y] == 'vacances':
        jour = intervalle[y][1] - 1 + futur # prévoir les vacances? mouhaha!
    else:
        jour = min(num_de_jour(depart) - jours[0], intervalle[y][1] - 1) + futur
    if jour < njours:
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
            #print('pas prévu',jour_de_num[jour + jours[0]],noms[y])
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

def prevoit_tout(maxfutur, depart = aujourdhui, maxdata = 1.5):
    MF = copy.deepcopy(M)
    MRF = copy.deepcopy(MR)
    MRDF = copy.deepcopy(MRD)
    for futur in range(1,maxfutur+1):
        print(str(futur) + ' ', end = '', flush = True)
        for y in range(nnoms):
            #print(str(y) + ' ', end ='',flush=True)
            prevoit_data(MF,MRF,MRDF,y,futur,depart = depart)
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
