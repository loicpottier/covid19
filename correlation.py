import sys
from urlcache import *
from outils import *
import pickle

######################################################################
# chargement des données

nouveau = False # False: on charge le fichier local

if len(sys.argv) > 1 and sys.argv[1] == 'nouveau':
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

M = np.zeros((len(noms), len(departements), len(jours)))
nnoms,ndeps,njours = np.shape(M)

intervalle = [None]*nnoms
kc= 0
for data in ldatacont:
    lnoms = data[data['dimensions'][-1]]
    j0 = max(num_de_jour(data['jours'][0]),jours[0])
    j1 = min(num_de_jour(data['jours'][-1]),jours[-1])
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

MR = copy.deepcopy(M)
for x in range(nnoms):
    if utiliser_proportions and  noms[x] not in donnees_proportions:
        for d in range(ndeps):
            # on ramene a si c'était la France
            MR[x,d,:] = MR[x,d,:] / population_dep[departements[d]] * population_france 

######################################################################
# matrice des dérivées

def aderiver(x):
    return(noms[x] in nomsind and noms[x] != 'R')

# pour calculer les decalages
MRD = copy.deepcopy(MR) 

for x in range(nnoms):
    if aderiver(x):
        x0,x1 = intervalle[x]
        MRD[x,:,x0:x1] = derivee_indic(MR[x,:,x0:x1],7)

#plt.plot(np.sum(M[ni('urgences'),:,:], axis = 0));plt.show()

#tout
#plt.plot(np.transpose(np.mean(M[:,:,:300],axis=1)));plt.show()

######################################################################
# décalages et corrélations

decmax = 40

# x et y indices de donnees dans m
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

#correlation(MRD,ni('travail'),ni('urgences'))
#correlation(MRD,ni('recherche horaires google'),ni('urgences'))

# contient les [decalage, correlation, xjour0,xjour1, yjour0,yjour1]
if nouveau:
    coefs = np.zeros((nnoms,nnoms,6))
    for x in range(nnoms):
        print('\n' + noms[x], end='', flush = True)
        for y in range(nnoms):
            if x != y:
                print('.', end='', flush = True)
                coefs[x,y] = correlation(MRD,x,y)

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
            print(dep, 'determinant nul:', np.linalg.det(A), 'max: ',np.max(L[dep]))
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

def prevoit_tout(maxfutur, depart = aujourdhui):
    MF = copy.deepcopy(M)
    MRF = copy.deepcopy(MR)
    MRDF = copy.deepcopy(MRD)
    for futur in range(1,maxfutur+1):
        print(str(futur) + ' ', end = '', flush = True)
        for y in range(nnoms):
            #print(str(y) + ' ', end ='',flush=True)
            prevoit_data(MF,MRF,MRDF,y,futur,depart = depart)
    print('prevision finie')
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
# tracé d'une prévision
# depart: jour de départ de la prevision
# passe: jours du passé (à partir d'aujourdhui)
# futur: duree de la prevision

def trace_previsions_passe(lnoms, depart, passe = 100, futur = 60, plot = False):
    MF,MRF,MRDF = prevoit_tout(futur, depart = depart)
    if plot:
        lcourbes = []
        for nom in lnoms:
            x = ni(nom)
            x0,x1 = intervalle[x]
            xdep = num_de_jour(depart) - jours[0]
            f = np.mean if nom in donnees_proportions else np.sum
            lcourbes += [(zipper(jourstext[xdep:xdep+futur],
                                 f(MF[x,:,xdep:xdep+futur], axis = 0)),
                          nom,prev),
                         (zipper(jourstext[int(x1)-passe:int(x1)],
                                 f(M[x,:,int(x1)-passe:int(x1)], axis = 0)),
                          '',real)]
        plotcourbes(lcourbes)
        plt.show()
    return(MF)

#trace_previsions_passe(['urgences', 'hospitalisation urgences','sosmedecin'], depart = '2020-11-15', futur = 60, plot = True)

#trace_previsions_passe(['taux positifs 29'], depart = '2020-11-14', futur = 60, plot = True)

######################################################################
# graphiques des prévisions à 60 jours

MF,MRF,MRDF = prevoit_tout(60)

#tout
if False:
    for x in nomsind:
        print(x)
        trace_previsions(MF,MRF,MRDF,[x],passe=100, futur = 50)

trace_previsions(MF,MRF,MRDF,['urgences', 'nouv hospitalisations','sosmedecin'],passe=100, futur = 50)
trace_previsions(MF,MRF,MRDF,['réanimations'],passe=100, futur = 50)
trace_previsions(MF,MRF,MRDF,['hospitalisations'],passe=100, futur = 50)
trace_previsions(MF,MRF,MRDF,['nouv réanimations', 'nouv décès'],passe=100, futur = 50)
trace_previsions(MF,MRF,MRDF,['hospi 09', 'hospi 19', 'hospi 29', 'hospi 39', 'hospi 49', 'hospi 59', 'hospi 69', 'hospi 79', 'hospi 89', 'hospi 90'],passe=100, futur = 50)
trace_previsions(MF,MRF,MRDF,['positifs'],passe=80, futur = 50)
trace_previsions(MF,MRF,MRDF,['positifs 09', 'positifs 19', 'positifs 29', 'positifs 39', 'positifs 49', 'positifs 59', 'positifs 69', 'positifs 79', 'positifs 89', 'positifs 90'],passe=80, futur = 50)
trace_previsions(MF,MRF,MRDF,['taux positifs'],passe=80, futur = 50)
trace_previsions(MF,MRF,MRDF,['taux positifs 09', 'taux positifs 19', 'taux positifs 29', 'taux positifs 39', 'taux positifs 49', 'taux positifs 59', 'taux positifs 69', 'taux positifs 79', 'taux positifs 89', 'taux positifs 90'],passe=80, futur = 50)
trace_previsions(MF,MRF,MRDF,['urgences','réanimations'],passe=100, futur = 50,dep = 6)
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
# animation des prévisions passées
import matplotlib.animation as animation

def courbes_prev_duree(x, duree, passe = 100, futur = 60, pas = 1):
    x0,x1 = intervalle[x]
    lcourbes = []
    for j in range(0,duree,pas):
        print('prévision -' + str(j) + '\n')
        xdep = x1 - 1 - j
        depart = jour_de_num[jours[0] + xdep]
        MF,MRF,MRDF = prevoit_tout(futur, depart = depart)
        f = np.mean if noms[x] in donnees_proportions else np.sum
        lcourbes.append(zipper(jourstext[xdep:xdep+futur],
                               f(MF[x,:,xdep:xdep+futur], axis = 0)))
    reel = zipper(jourstext[int(x1)-passe:int(x1)],
                  f(M[x,:,int(x1)-passe:int(x1)], axis = 0))
    return((reel,lcourbes))

def axejours(lj):
    n = len(lj)
    lk = [n-1 - 7*k for k in range(n//7+1)][::-1]
    ljaxe = [joli(lj[k]) for k in lk]
    plt.xticks(lk,ljaxe,rotation = 70,fontsize = 8)

def val(jours,l):
    d = dict(l)
    return([d[j] if j in d else None for j in jours])

def anime_previsions(nom, duree = 90):
    x = ni(nom)
    reel, previsions = courbes_prev_duree(x, duree)
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
    plt.plot(lv,'-', linewidth = 2)
    prev, = plt.plot(lj,val(lj,previsions[0]), '-')
    def init():
        ax.set_xlim(0, len(lj))
        ax.set_ylim(-10,max([max([x[1] for x in c]) for c in courbes]))
        return prev,
    def update(frame):
        prev.set_data(lj,val(lj,previsions[frame]))
        return prev,
    ani = animation.FuncAnimation(fig, update, frames= np.array(list(range(len(previsions)))),
                        init_func=init, blit=True)
    plt.rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'
    DPI=180
    writer = animation.FFMpegWriter(fps=5, bitrate=10000)
    ani.save("synthese2/previsions_" + nom + ".mp4", writer = writer, dpi=DPI) 

if nouveau:
    anime_previsions('urgences')
    anime_previsions('réanimations')
    anime_previsions('hospitalisations')

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

'''
taux positifs
erreur moyenne à 7 jours: 0.6%
erreur moyenne à 14 jours: 1.0%
erreur moyenne à 21 jours: 1.2%
erreur moyenne à 28 jours: 1.3%
erreur moyenne à 35 jours: 1.3%
erreur moyenne à 42 jours: 0.7%
erreur moyenne à 49 jours: 0.7%
erreur moyenne à 56 jours: 0.3%
décès
erreur moyenne à 7 jours: 0.3%
erreur moyenne à 14 jours: 0.8%
erreur moyenne à 21 jours: 1.7%
erreur moyenne à 28 jours: 2.6%
erreur moyenne à 35 jours: 3.2%
erreur moyenne à 42 jours: 3.9%
erreur moyenne à 49 jours: 4.2%
erreur moyenne à 56 jours: 5.1%
hospitalisations
erreur moyenne à 7 jours: 0.5%
erreur moyenne à 14 jours: 0.8%
erreur moyenne à 21 jours: 1.9%
erreur moyenne à 28 jours: 3.8%
erreur moyenne à 35 jours: 5.8%
erreur moyenne à 42 jours: 8.9%
erreur moyenne à 49 jours: 13.0%
erreur moyenne à 56 jours: 16.5%
sosmedecin
erreur moyenne à 7 jours: 6.2%
erreur moyenne à 14 jours: 10.6%
erreur moyenne à 21 jours: 13.6%
erreur moyenne à 28 jours: 19.1%
erreur moyenne à 35 jours: 18.9%
erreur moyenne à 42 jours: 20.3%
erreur moyenne à 49 jours: 20.0%
erreur moyenne à 56 jours: 17.3%
urgences
erreur moyenne à 7 jours: 3.0%
erreur moyenne à 14 jours: 5.7%
erreur moyenne à 21 jours: 8.8%
erreur moyenne à 28 jours: 12.2%
erreur moyenne à 35 jours: 14.8%
erreur moyenne à 42 jours: 14.3%
erreur moyenne à 49 jours: 11.1%
erreur moyenne à 56 jours: 19.2%
positifs
erreur moyenne à 7 jours: 9.3%
erreur moyenne à 14 jours: 15.1%
erreur moyenne à 21 jours: 18.9%
erreur moyenne à 28 jours: 22.7%
erreur moyenne à 35 jours: 27.5%
erreur moyenne à 42 jours: 18.6%
erreur moyenne à 49 jours: 12.7%
erreur moyenne à 56 jours: 24.5%
nouv hospitalisations
erreur moyenne à 7 jours: 3.5%
erreur moyenne à 14 jours: 6.7%
erreur moyenne à 21 jours: 10.9%
erreur moyenne à 28 jours: 16.5%
erreur moyenne à 35 jours: 22.7%
erreur moyenne à 42 jours: 30.5%
erreur moyenne à 49 jours: 23.6%
erreur moyenne à 56 jours: 28.7%
réanimations
erreur moyenne à 7 jours: 0.6%
erreur moyenne à 14 jours: 1.0%
erreur moyenne à 21 jours: 4.2%
erreur moyenne à 28 jours: 9.0%
erreur moyenne à 35 jours: 15.1%
erreur moyenne à 42 jours: 21.2%
erreur moyenne à 49 jours: 28.2%
erreur moyenne à 56 jours: 30.7%
R
erreur moyenne à 7 jours: 2.7%
erreur moyenne à 14 jours: 3.3%
erreur moyenne à 21 jours: 3.7%
verreur moyenne à 28 jours: 4.0%
erreur moyenne à 35 jours: 5.6%
erreur moyenne à 42 jours: 5.4%
erreur moyenne à 49 jours: 15.1%
erreur moyenne à 56 jours: 41.9%
hospitalisation urgences
erreur moyenne à 7 jours: 2.7%
erreur moyenne à 14 jours: 5.5%
erreur moyenne à 21 jours: 8.8%
erreur moyenne à 28 jours: 13.1%
erreur moyenne à 35 jours: 17.7%
erreur moyenne à 42 jours: 24.3%
erreur moyenne à 49 jours: 30.3%
erreur moyenne à 56 jours: 57.2%
nouv réanimations
erreur moyenne à 7 jours: 3.2%
erreur moyenne à 14 jours: 7.5%
erreur moyenne à 21 jours: 13.0%
erreur moyenne à 28 jours: 13.4%
erreur moyenne à 35 jours: 21.0%
erreur moyenne à 42 jours: 29.3%
erreur moyenne à 49 jours: 38.2%
erreur moyenne à 56 jours: 66.8%
nouv décès
erreur moyenne à 7 jours: 4.3%
erreur moyenne à 14 jours: 4.7%
erreur moyenne à 21 jours: 10.9%
erreur moyenne à 28 jours: 20.0%
erreur moyenne à 35 jours: 31.4%
erreur moyenne à 42 jours: 40.8%
erreur moyenne à 49 jours: 53.7%
erreur moyenne à 56 jours: 83.0%
'''


