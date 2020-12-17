# a voir: https://github.com/CSSEGISandData/COVID-19
from outils import *

from charge_contextes import contextes, extrapole_manquantes, datamobilite, datameteo, datavacances, dataapple, datahygiene, regions

from charge_indicateurs import indicateurs, dataurge, datahospiurge, datareatot, datahospitot, datadecestot, datahospi, datarea, datadeces, datahospiage, dataposage, datapos, population_dep

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

datacontexte = fusion(fusion(fusion(datamobilite,datameteo),datavacances),dataapple)
datacontexte['contextes'] = datacontexte['mobilites-meteo-vacances-apple']
#datacontexte = fusion(fusion(datamobilite,datameteo),datavacances)
#datacontexte['contextes'] = datacontexte['mobilites-meteo-vacances']
#datacontexte = fusion(datamobilite,datameteo)
#datacontexte['contextes'] = datacontexte['mobilites-meteo']
#datacontexte = datameteo
#datacontexte['contextes'] = datacontexte['meteo']

# normalisation des valeurs des contextes: entre -100 et 100

v = copy.deepcopy(datacontexte['valeurs'])

for c in range(len(v[0,0])):
    maxv = np.max(v[:,:,c])
    minv = np.min(v[:,:,c])
    v[:,:,c] = 200 * (v[:,:,c] - minv) / (maxv - minv) - 100

datacontexte['valeurs'] = v

######################################################################
# on ne garde que les departements communs aux donnees

ldata = ([dataurge, datahospiurge, datareatot, datahospitot, datadecestot,
          datahospi, datarea, datadeces, datapos, datacontexte]
         + [dataposage[age] for age in sorted([a for a in dataposage])
            if age != '0'])

deps = datahospi['departements'][:]
[[deps.remove(d) for d in deps[:] if  d not in t['departements'] and d in deps]
 for t in ldata
 if t['departements'] != []]

for t in ldata:
    if t['departements'] != []:
        t['valeurs'] = np.array([t['valeurs'][t['departements'].index(d)] for d in deps])
        t['departements'] = deps

######################################################################
# regroupement par regions de donnees par departement:
# on prend la moyenne des valeurs des departements

def groupedataregionscontexte(data):
    regs = [x for x in regions]
    deps = [d for d in data['departements']]
    datareg = copy.deepcopy(data)
    datareg['regions'] = regs
    datareg['dimensions'] = [x if x != 'departements' else 'regions' for x in data['dimensions']]
    datareg['valeurs'] = np.array([np.mean([data['valeurs'][d,:,:]
                                            for d in range(len(deps))
                                            if deps[d] in regions[r][1:]], axis = 0)
                                   for r in regs])
    return(datareg)

datacontexteregions = groupedataregionscontexte(datacontexte)

def groupedataregionsdonnees(data):
    regs = [x for x in regions]
    deps = [d for d in data['departements']]
    datareg = copy.deepcopy(data)
    datareg['regions'] = regs
    datareg['dimensions'] = [x if x != 'departements' else 'regions' for x in data['dimensions']]
    datareg['valeurs'] = np.array([np.mean([data['valeurs'][d,:]
                                            for d in range(len(deps))
                                            if deps[d] in regions[r][1:]], axis = 0)
                                   for r in regs])
    return(datareg)

######################################################################
# prevision
######################################################################
# Lt tableau de valeurs par departements,jours,contexte
# Lj noms des jours
# Ht tableau de valeurs par departements,jours
# Hj noms des jours
# même dimension en départements, pas en jours

# dj décalage du début de Lt par rapport au début de Ht, positif si Lt commence apres Ht
# rend C telle que sum_d ||Lt[d].C - Ht[d]||^2 minimale
# et les sous-tableaux décalés de dj pour lesquels C est obtenu
def erreur(dj,Lj,Lt,Hj,Ht,decale_limite = 30):
    e = -1
    ndeps,njL,nlieux = np.shape(Lt)
    ndeps,njH = np.shape(Ht)
    decal = njH - (njL + dj) # decalage des fins de tableaux, positif si H fini apres L
    # intersection des deux tableaux Lt et Ht
    L = Lt[:,max(0,-dj):min(njL,njH-dj),:]
    H = Ht[:,max(0,dj):min(njH,njL+dj)]
    LJ = Lj[max(0,-dj):min(njL,njH-dj)]
    HJ = Hj[max(0,dj):min(njH,njL+dj)]
    C = np.zeros((ndeps,nlieux))
    ed = np.zeros(ndeps)
    vm = math.sqrt(np.sum(H**2)/(ndeps*len(LJ))) # valeur moyenne des données par jour et département
    # décalage max des jours des fins des tableaux
    decal_jour = num_de_jour(HJ[-1]) - num_de_jour(LJ[-1])
    #print('decalage des jours de fin:',decal_jour)
    # au moins 60 jours de données à comparer
    if abs(decal_jour) < decale_limite and len(LJ) > 60 :
        e = 0
        #print(dj,'ok')
        for d in range(ndeps):
            A = np.transpose(L[d]) @ L[d]
            B = np.transpose(L[d]) @ H[d]
            if abs(np.linalg.det(A)) < 1e-14 and abs(dj) < 50:
                print(d, 'determinant nul:', np.linalg.det(A), 'max: ',np.max(L[d]))
            C[d] = np.linalg.inv(A) @ B
            E = L[d] @ C[d] - H[d]
            edep = np.transpose(E) @ E
            ed[d] = math.sqrt(edep / len(LJ)) # erreur moyenne par jour du département
            e = e + edep
        e = math.sqrt(e/(ndeps * len(LJ))) # erreur moyenne par jour et département
    else:
        pass
    return(e,ed,vm,LJ,L,HJ,H,C)

# dernier_min local (de gauche à droite) inférieur à l'erreur de fin,
# sinon dernier_min local
# sinon dernier élément
# le[0] = decal,e,ed,vm,LJ,L,HJ,H,C

def dernier_min_local(le):
    lmin = []
    lmininf = []
    lmax = []
    efin = le[-1][1]
    cmax = -1e20
    xcmax = None
    eminglob = efin
    minglob = None
    for i in range(len(le)-1):
        e = le[i][1]
    for i in range(len(le)-2):
        e0 = le[i][1] # le[0] = decal,e,ed,vm,LJ,L,HJ,H,C
        e1 = le[i+1][1]
        e2 = le[i+2][1]
        if e1 < e0 and e1 < e2: #min local
            lmin.append(le[i+1])
            if e1 <= efin:
                lmininf.append(le[i+1])
                if e1 < eminglob:
                    eminglob = e1
                    minglob = le[i+1]
    for i in range(1,len(le)-3):
        e0 = (le[i+1][1] + le[i-1][1]) / 2 - le[i][1]
        e1 = (le[i+2][1] + le[i][1]) / 2 - le[i+1][1]
        e2 = (le[i+3][1] + le[i+1][1]) / 2 - le[i+2][1]
        if (e0 > 0 and e2 > 0
            and ((e1 > e0 and e1 > e2)
                 or (le[i+1][1] <= le[i+2][1] and le[i+1][1] <= le[i][1]))):
            lmax.append(le[i+1])
    lpentemax = []
    efin = le[-1][1]
    for i in range(1,len(le)-2):
        p0 = (efin - le[i-1][1]) / (len(le) - 1 - (i-1))
        p1 = (efin - le[i][1]) / (len(le) - 1 - i)
        p2 = (efin - le[i+1][1]) / (len(le) - 1 - (i+1))
        if p1 > p0 and p1 >  p2:
            lpentemax.append(le[i])
#    print('mins:',[(x[0],x[1]) for x in lmin], [(x[0],x[1]) for x in lmininf],
#          [(x[0],x[1]) for x in lmax])
    if minglob != None:
        #print("min global plus petit que l'erreur à 0",minglob[0],minglob[1])
        return(minglob)
    elif lmininf != []:
        #print("dernier min local plus petit que l'erreur à 0",lmininf[-1][0],lmininf[-1][1])
        return(lmininf[-1])
    elif lpentemax != []:
        #print('dernier pentemax local')
        return(lpentemax[-1])
    elif lmin != []:
        #print('dernier min local')
        return(lmin[-1])
    elif lmax != []:
        #print('dernier max de courbure local',[(x[0],x[1]) for x in lmax])
        return(lmax[-1])
    else:
        #print('pas de max de courbure local ni de min local')
        return(le[-1])

#dernier_min_local([(0,3),(0,1),(0,2),(0,4),(0,3),(0,5),(0,3),(0,6)])

# rend le decalage des dates qui minimise l'erreur de prévision,
# le vecteur de prévision et les sous-tableaux qui donnent cette erreur min
# (decalage de la fin du contexte par rapport à la fin des données)
def fit(Lj,Lt,Hj,Ht, decale_limite = 30):
    #print(len(Lj),len(Hj),np.shape(Lt),np.shape(Ht))
    ndeps,njL,nlieux = np.shape(Lt)
    ndeps,njH = np.shape(Ht)
    #print(njL, njH)
    erreurs = []
    for dj in range(-njL,njH):
        decal = njH - (njL + dj)
        try:
            e,ed,vm,LJ,L,HJ,H,C = erreur(dj,Lj,Lt,Hj,Ht, decale_limite = decale_limite)
            #print(dj,len(LJ),e)
            if e > 0 and num_de_jour(LJ[-1]) < num_de_jour(HJ[-1]):
                decal_jour_passe = num_de_jour(LJ[-1]) - num_de_jour(HJ[-1])
                erreurs.append((decal_jour_passe,e,ed,vm,LJ,L,HJ,H,C))
        except:
            pass #print('probleme avec dj = ',dj)
    decal,emin,edmin,vmmin,LJ,L,HJ,H,C = dernier_min_local(sorted(erreurs,key=lambda x:x[0]))
    #print('dernier min local:', decal,emin,vmmin,'derniers jours:',LJ[-1],HJ[-1],
    #      'premiers jours:',LJ[0],HJ[0],'durée:',len(LJ), len(HJ), njL, njH)
    decalage = num_de_jour(HJ[-1]) - num_de_jour(LJ[-1])
    plt.clf()
    plt.plot([x[0] for x in erreurs],[x[1] for x in erreurs],'o-')
    plt.title('erreur')
    plt.grid()
    plt.show(False)
    return(decalage,C,emin,edmin,vmmin,LJ,L,HJ,H)

######################################################################
# prévision avec les decalages variables
######################################################################
# correlation entre dérivée d'un indicateur et contexte

def decale_jour(j,d):
    return(jour_de_num[num_de_jour(j)+d])

def inter(l1,l2):
    return([x for x in l1 if x in l2])

def decale_data(tjours,tvaleurs,decalage): # décalage dans le passé
    jourst = [jour_de_num[j]
              for j in range(num_de_jour(tjours[0]),
                             num_de_jour(tjours[-1])+1)
              if (j - decalage >= num_de_jour(datacontexte['jours'][0])
                  and j - decalage <= num_de_jour(datacontexte['jours'][-1]))]
    tj1 = tjours.index(jourst[0])
    tv = tvaleurs[:,tj1:tj1 + len(jourst)]
    return(jourst,tv)

def derivee_indic(t):
    return(np.array([derivee(d) for d in t]))

def lissage_indic(t,d):
    t1 = copy.deepcopy(t)
    ndeps, nj = np.shape(t)
    for dep in range(ndeps):
        t1[dep,:] = lissage(t[dep,:], d)
    return(t1)

# coef de correlation
def correlation_dec(c,cjours,cvaleurs,ijours,ivaleurs,decalage):
    mcv = np.mean(cvaleurs[:,:,c]) # moyenne globale du contexte c
    jourst,ind = decale_data(ijours,ivaleurs,decalage)
    ind = derivee_indic(ind)
    # rapporter l'indicateur à la population du département
    for d in range(np.shape(ind)[0]):
        ind[d,:] = ind[d,:] / population_dep[datacontexte['departements'][d]]
    mind = np.mean(ind) # moyenne globale
    ind = ind - mind
    j1 = cjours.index(decale_jour(jourst[0],-decalage))
    j2 = cjours.index(decale_jour(jourst[-1],-decalage))
    cv = cvaleurs[:,:,c]
    cv = cv[:,j1:j2+1]
    cv = cv - mcv
    corr = (np.trace(np.transpose(cv) @ ind)
            / (np.sqrt(np.trace(np.transpose(cv) @ cv))
               *
               np.sqrt(np.trace(np.transpose(ind) @ ind))))
    # intervalles de jours corrélés
    return([corr,
            cjours[j1],cjours[j2],
            jourst[0],jourst[-1]])

def premier_max_local(l):
    lmax = []
    for i in range(1,len(l)-1):
        if l[i-1] <= l[i] and l[i] >= l[i+1]:
            lmax.append(i)
    if lmax != []:
        i = lmax[0]
        if len(lmax) >= 2:
            i2 = lmax[1]
            if l[i2] > 1.3 * l[i]:
                return(i2)
        return(i)
    else:
        cmax = max(l)
        dmax = l.index(cmax)
        return(dmax)

decmax = 30
dico_lissage = {}

ldataregions = [datahospiage[age] for age in sorted(datahospiage) if age != '0']

def correlation(x,y,datacontexte,ldata):
    decmin = 1
    cx = datacontexte['mobilites-meteo-vacances-apple'].index(x)
    t = [t for t in ldata if t['nom'] == y][0]
    try:
        tv = dico_lissage[y]
    except:
        tv = lissage_indic(t['valeurs'],7)
        dico_lissage[y] = tv
    lc = [correlation_dec(cx,
                          datacontexte['jours'],datacontexte['valeurs'],
                          t['jours'],tv,
                          d)
          for d in range(decmin,decmax)]
    lcc = lissage([c for [c,jcont1,jcont2,jind1,jind2] in lc],7)
    if abs(max(lcc)) > abs(min(lcc)):
        dmax = premier_max_local(lcc)
        cmax = lcc[dmax]
        [c,jcont1,jcont2,jind1,jind2] = lc[dmax]
        return(([cmax,jcont1,jcont2,jind1,jind2],decmin + dmax))
    else:
        dmin = premier_max_local([-x for x in lcc])
        cmin = lcc[dmin]
        [c,jcont1,jcont2,jind1,jind2] = lc[dmin]
        #print(decmin,dmin,lc[:5],lcc[:5])
        return(([cmin,jcont1,jcont2,jind1,jind2],decmin + dmin))

#----------------------------------------------------------------------
# le dico de toutes les corrélations
# on vire les deces totaux et les positifs par age:
# les correlations et decalages sont bizarres
coefs = {}
noms_indicateurs = ([t['nom'] for t in ldata #[dataurge] # ldata
                     if t != datacontexte and not '9' in t['nom']])

noms_contextes = datacontexte['mobilites-meteo-vacances-apple']

for y in noms_indicateurs:
    coefs[y] = {}
    for x in noms_contextes:
        coefs[y][x] = correlation(x,y,datacontexte,ldata)
        print(y,x[:15],coefs[y][x][1], ("%1.3f" % coefs[y][x][0][0]))

noms_indicateurs_regions = [t['nom'] for t in ldataregions]
noms_contextes_regions = datacontexteregions['mobilites-meteo-vacances-apple']

for y in noms_indicateurs_regions:
    coefs[y] = {}
    for x in noms_contextes:
        coefs[y][x] = correlation(x,y,datacontexteregions,ldataregions)
        print(y,x[:15],coefs[y][x][1], ("%1.3f" % coefs[y][x][0][0]))

# utilisé par synthes_donnees.py

f = open('_correlations.py','w')
f.write('noms_indicateurs =' + str(noms_indicateurs + noms_indicateurs_regions) + '\n'
        + 'noms_contextes =' + str(noms_contextes) + '\n'
        + 'coefs = ' + str(coefs))
f.close()

#----------------------------------------------------------------------
# visualisation des corrélations selon le decalage
x = 'température'#'vacances'#'résidence' #'parcs' #'résidence'
y = 'positifs' #'hospitalisations total' # 'positifs'
cx = datacontexte['mobilites-meteo-vacances-apple'].index(x)
t = [t for t in ldata if t['nom'] == y][0]
try:
    tv = dico_lissage[y]
except:
    tv = lissage_indic(t['valeurs'],7)
    dico_lissage[y] = tv

lc = [correlation_dec(cx,
                      datacontexte['jours'],datacontexte['valeurs'],
                      t['jours'],tv,
                      d)[0]
      for d in range(1,50)]

plt.clf()
plt.plot(range(1,len(lc)+1),lissage(lc,7))
plt.show(False)


#coefs['hospitalisations total']['travail']
#coefs['urgences']['travail']
#correlation('résidence','positifs')
coefs['positifs']['résidence']
######################################################################
# prévision avec les décalages donné par les corrélations

def fit2(datacontexte,indicateur,datavaleursderivee,datajours):
    linfo = []
    for (c,contexte) in enumerate(datacontexte['mobilites-meteo-vacances-apple']):
        [corr,jcont1,jcont2,jind1,jind2],decalage = coefs[indicateur][contexte]
        jc1 = datacontexte['jours'].index(jcont1)
        jc2 = datacontexte['jours'].index(jcont2)
        j1 = datajours.index(jind1)
        j2 = datajours.index(jind2)
        linfo.append((c,corr,jc1,jc2,j1,j2,decalage))
    tv = datacontexte['valeurs']
    ndeps,njours,ncont = np.shape(tv)
    j2min = min([j2 for (c,corr,jc1,jc2,j1,j2,decalage) in linfo])
    njmin = min([j2min - j1+1 for (c,corr,jc1,jc2,j1,j2,decalage) in linfo])
    #print(datajours[j2min],njmin)
    L = np.zeros((ndeps,njmin,ncont))
    H = np.zeros((ndeps,njmin))
    for d in range(ndeps):
        H[d,0:njmin] = datavaleursderivee[d,j2min-(njmin-1):j2min+1]
        for (c,corr,jc1,jc2,j1,j2,decalage) in linfo:
            L[d,0:njmin,c] = tv[d,jc2-(j2-j2min)-(njmin-1):jc2-(j2-j2min)+1,c]
    # pas sur que ca change grand chose...on enlève la moyenne, car les correlations
    # avaient ete calculees comme ca
    vm = np.mean(L,axis = 1)
    vm = vm - vm # si on ne centre pas les données
    for d in range(np.shape(L)[0]):
        for v in range(np.shape(L)[2]):
            L[d,:,v] = L[d,:,v] - vm[d,v]
    hm = np.mean(H, axis = 1)
    hm = hm - hm # si on ne centre pas les données
    for d in range(np.shape(H)[0]):
        H[d,:] = H[d,:] - hm[d]
    ndeps,njours,ncont = np.shape(L)
    C = np.zeros((ndeps,ncont))
    ed = np.zeros(ndeps)
    e = 0
    for d in range(ndeps):
        A = np.transpose(L[d]) @ L[d]
        B = np.transpose(L[d]) @ H[d]
        if abs(np.linalg.det(A)) < 1e-14 and abs(dj) < 50:
            print(d, 'determinant nul:', np.linalg.det(A), 'max: ',np.max(L[d]))
        C[d] = np.linalg.inv(A) @ B
        E = L[d] @ C[d] - H[d]
        edep = np.transpose(E) @ E
        ed[d] = math.sqrt(edep / njours) # erreur moyenne par jour du département
        e = e + edep
    e = math.sqrt(e/(ndeps * njours))
    # on a maintenant les coefs de prévision des départements
    return((C,j2min,njmin,e,ed,vm,hm,linfo))

######################################################################
# prévision des données
# depart : jour de départ de la prevision

# lissage de la dérivée avec dérivées des derniers jours
def lissage2(Bd,ldB, passe = 5):
    #print('lissage de la dérivée')
    l = sorted([(j,ldB[j]) for j in ldB], key = lambda x : x[0])
    #print('Bd:', np.sum(Bd[:,-passe:],axis=0))
    #print('ldB:',[np.sum(lj) for (j,lj) in l])
    t1 = np.transpose(np.array([ld for (j,ld) in l]))
    ndeps, nj = np.shape(t1)
    #print(np.shape(Bd[:,-passe:]), np.shape(t1))
    t = np.concatenate([Bd[:,-passe:],t1], axis = 1)
    for dep in range(ndeps):
        t2 = t[dep,:]
        for k in range(passe):
            if passe+k < len(t2):
                t2[passe+k] = moyenne(t2[k+1:passe+k+1])
        t[dep,:] = lissage(t2, passe)
    t = t[:,passe:]
    #print('prev:',[(l[k][0],np.sum(t[:,k])) for k in range(len(l))])
    return(dict([(l[k][0],t[:,k]) for k in range(len(l))]))

# prévision de la dérivée avec les décalages donnés par les corrélations
def previsiondcorr(datacontexte,data, decale_limite = 30, depart = None, fin = None, passe = 5):
    print('prevision', data['titre'])
    A,Aj = datacontexte['valeurs'], datacontexte['jours']
    Bj = data['jours']
    #print('Bj', Bj[-5:])
    B = np.array([lissage(dep,7) for dep in data['valeurs']])
    Bd = np.array([derivee(dep) for dep in B])
    if depart == None:
        depart = Bj[-1]
    n = num_de_jour(Bj[-1]) - num_de_jour(depart) + 1
    f = num_de_jour(Aj[-1]) - num_de_jour(Bj[-1])
    if f >= 0:
        Aj = Aj[:-f-1]
        A = A[:,:-f-1,:]
    ndeps,njours,ncont = np.shape(A)
    C,j2min,njmin,e,ed,vm,hm,linfo = fit2(datacontexte,data['nom'],Bd,Bj)
    # j2min dernier jour calculé de B
    #print('j2min:', j2min,'njmin:',njmin)
    ldB = {} # prévision des derivees pour les derniers jours des contextes
    corrmax = max([abs(corr) for (c,corr,jc1,jc2,j1,j2,decalage) in linfo])
    linfook = [(decalage,j2) for (c,corr,jc1,jc2,j1,j2,decalage) in linfo
               if abs(corr) > corrmax/2]
    decmin = min([d for (d,j2) in linfook])
    j2min = min([j2 for (c,corr,jc1,jc2,j1,j2,decalage) in linfo])
    print('decalage min=', decmin, 'n=', n, 'j2min=', j2min,'njmin=',njmin)
    L = np.zeros((ndeps,njmin,ncont))
    try:
        for d in range(ndeps):
            for (c,corr,jc1,jc2,j1,j2,decalage) in linfo:
                #print(c,corr,jc1,jc2,j1,j2,decalage)
                #print(njmin,jc2 -(j2-j2min)-(njmin-1),jc2 -(j2-j2min)+1,len(A[d,:,c]))
                L[d,0:njmin,c] = A[d,jc2 -(j2-j2min)-(njmin-1):jc2 -(j2-j2min)+1,c]
                #print('ok')
        for jp in range(-n+1, decmin):
            #print(jp)
            db = np.zeros(np.shape(B[:,0]))
            db = db + np.array([hm[d] + (L[d,jp - decmin,:] - vm[d,:]) @ C[d]
                                for d in range(len(C))])
            nj = Aj[jp - decmin]
            j1 = addday(nj,decmin-1)
            if num_de_jour(j1) >= num_de_jour(Bj[-n]):
                ldB[addday(nj,decmin-1)] = db
            #print(addday(nj,decmin-1))
    except:
        print('************************** erreur ')
        pass
    if len(ldB) > 0:
        # on lisse les dérivées
        #print('lissage')
        ldB = lissage2(Bd,ldB,passe = passe)
        #print('ok')
        #print([(j,np.sum(ldB[j])) for j in sorted([j for j in ldB])])
    bj = B[:,-n]
    bjp, bjm = bj[:] , bj[:] # + erreurs, - erreurs
    j = Bj[-n]
    P, Pp, Pm, Pj = [], [], [], []
    while j in ldB:
        P.append(bj); Pp.append(bjp); Pm.append(bjm); Pj.append(j)
        #print(j,np.sum(bj),np.sum(ldB[j]))
        bjj = bj + ldB[j]
        bjpj = bjp + ldB[j] + 2*ed/math.sqrt(ndeps)
        bjmj = bjm + ldB[j] - 2*ed/math.sqrt(ndeps)
        bj = np.array([max(x,0) for x in bjj])
        bjp = np.array([max(x,0) for x in bjpj])
        bjm = np.array([max(x,0) for x in bjmj])
        j = addday(j,1)
    P.append(bj); Pp.append(bjp); Pm.append(bjm); Pj.append(j)
    prev = {'nom': data['nom'],
            'decalage': decmin,
            'depart': depart,
            'coefficients': C,
            'erreur': e,
            'erreurs': ed,
            'valeur moyenne': vm,
            'tableaux': None,
            'jours': Pj,
            'valeurs': np.array(P),
            'valeursmax': np.array(Pp),
            'valeursmin': np.array(Pm),
            'derivees': sorted(ldB),
    }
    return(prev)


p = previsiondcorr(datacontexte,datahospi)

#p = previsiondcorr(datacontexte,datarea)

#p = previsiondcorr(datacontexte,datareatot)

#p = previsiondcorr(datacontexte, dataposage['29'])

# prévision de la dérivée
def previsiond1(datacontexte,data, decale_limite = 30, depart = None, fin = None, passe = 5):
    print('prevision', data['titre'])
    A,Aj = datacontexte['valeurs'], datacontexte['jours']
    Bj = data['jours']
    B = np.array([lissage(dep,7) for dep in data['valeurs']])
    Bd = np.array([derivee(dep) for dep in B])
    if depart == None:
        depart = Bj[-1]
    n = num_de_jour(Bj[-1]) - num_de_jour(depart) + 1
    f = num_de_jour(Aj[-1]) - num_de_jour(Bj[-1])
    if f >= 0:
        Aj = Aj[:-f-1]
        A = A[:,:-f-1,:]
    ndeps = len(A[0])
    decalage,C,e,ed,vm,LJ,L,HJ,H = fit(Aj,A,Bj,Bd, decale_limite = decale_limite)
    ldB = {} # prévision des derivees pour les derniers jours des contextes
    for j in range(-n+1,decalage):
        db = np.zeros(np.shape(B[:,0]))
        db = db + np.array([A[d,j-decalage,:] @ C[d] for d in range(len(C))])
        nj = Aj[j-decalage]
        ldB[addday(nj,decalage-1)] = db
    # on lisse les dérivées
    ldB = lissage2(Bd,ldB,passe = passe)
    bj = B[:,-n]
    bjp, bjm = bj[:] , bj[:] # + erreurs, - erreurs
    j = Bj[-n]
    P, Pp, Pm, Pj = [], [], [], []
    while j in ldB:
        P.append(bj); Pp.append(bjp); Pm.append(bjm); Pj.append(j)
        bjj = bj + ldB[j]
        bjpj = bjp + ldB[j] + 2*ed/math.sqrt(ndeps)
        bjmj = bjm + ldB[j] - 2*ed/math.sqrt(ndeps)
        bj = np.array([max(x,0) for x in bjj])
        bjp = np.array([max(x,0) for x in bjpj])
        bjm = np.array([max(x,0) for x in bjmj])
        j = addday(j,1)
    P.append(bj); Pp.append(bjp); Pm.append(bjm); Pj.append(j)
    prev = {'nom': data['nom'],
            'decalage': decalage,
            'depart': depart,
            'coefficients': C,
            'erreur': e,
            'erreurs': ed,
            'valeur moyenne': vm,
            'tableaux': (LJ,L,HJ,H),
            'jours': Pj,
            'valeurs': np.array(P),
            'valeursmax': np.array(Pp),
            'valeursmin': np.array(Pm),
    }
    return(prev)

def previsions_multiplesd(datacontexte,data,n):
    #print('prevision', data['nom'])
    A,Aj = datacontexte['valeurs'], datacontexte['jours']
    Bj = data['jours']
    B = np.array([lissage(dep,7) for dep in data['valeurs']])
    Bd = np.array([derivee(dep) for dep in B])
    f = num_de_jour(Aj[-1]) - num_de_jour(Bj[-1])
    if f >= 0:
        Aj = Aj[:-f-1]
        A = A[:,:-f-1,:]
    decalage,C,e,ed,vm,LJ,L,HJ,H = fit(Aj,A,Bj,Bd)
    ldB = {} # prévision des derivees pour les derniers jours des contextes
    for j in range(-n+1,decalage):
        db = np.zeros(np.shape(B[:,0]))
        db = db + np.array([A[d,j-decalage,:] @ C[d] for d in range(len(C))])
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
                    'erreurs': ed,
                    'valeur moyenne': vm,
                    'jours': Pj,
                    'valeurs': np.array(P),
                    'valeursmax': np.array(P),
                    'valeursmin': np.array(P),}
            lprev.append(prev)
    return(lprev)

def normalisecoef(C):
    s = np.sum(np.abs(C), axis = 1)
    return(np.array([C[d]/s[d] for d in range(len(C))]))

######################################################################
# prévisions variées

def lissecontexte(data,d):
    data1 = copy.deepcopy(data)
    t = data1['valeurs']
    ndeps, nj, nv = np.shape(t)
    for dep in range(ndeps):
        for v in range(nv):
            t[dep,:,v] = lissage(t[dep,:,v], d)
    return(data1)

datacontexte1 = datacontexte #lissecontexte(datacontexte,7)

previsiond = previsiondcorr #previsiond1

P = [previsiond(datacontexte1,data) for data in [dataurge,datahospi,datarea,datadeces]]

trace([(zipper(p['jours'], np.sum(p['valeurs'],axis = 1)),
        p['nom'],'=')
       for p in P if len(p['valeurs'] != 0)]
      +
      [(zipper(p['jours'], np.sum(p['valeursmax'],axis = 1)),
        '','=')
       for p in P if len(p['valeurs'] != 0)]
      +
      [(zipper(p['jours'], np.sum(p['valeursmin'],axis = 1)),
        '','=')
       for p in P if len(p['valeurs'] != 0)]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0),7))[-60:],
        '','--')
       for t in [dataurge,datahospi,datarea,datadeces]],
      'prévisions à partir des données de mobilité et des données météo\n',
      'donnees/_prevision_par_mobilite')

Preatot = [previsiondcorr(datacontexte1,data) for data in [datareatot]]

trace([(zipper(p['jours'], np.sum(p['valeurs'],axis = 1)),
        p['nom'],'=')
       for p in Preatot if len(p['valeurs'] != 0)]
      +
      [(zipper(p['jours'], np.sum(p['valeursmax'],axis = 1)),
        'max','=')
       for p in Preatot if len(p['valeurs'] != 0)]
      +
      [(zipper(p['jours'], np.sum(p['valeursmin'],axis = 1)),
        'min','=')
       for p in Preatot if len(p['valeurs'] != 0)]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0),7))[-60:],
        '','--')
       for t in [datareatot]],
      'prévision des patients en réanimation\n',
      'donnees/_prevision_reatot_par_mobilite')

# prevision au depart d'un jour donné
date_depart = '2020-09-15'
Purge1 = [previsiond(datacontexte1,data, depart = date_depart)
          for data in [dataurge]]

trace([(zipper(p['jours'], np.sum(p['valeurs'],axis = 1)),
        p['nom'],'=')
       for p in Purge1 if len(p['valeurs'] != 0)]
      +
      [(zipper(p['jours'], np.sum(p['valeursmax'],axis = 1)),
        'max','=')
       for p in Purge1 if len(p['valeurs'] != 0)]
      +
      [(zipper(p['jours'], np.sum(p['valeursmin'],axis = 1)),
        'min','=')
       for p in Purge1 if len(p['valeurs'] != 0)]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0),7))[-120:],
        '','--')
       for t in [dataurge]],
      'prévision des urgences à partir du ' + date_depart + '\n',
      'donnees/_prevision_urge_date_depart')

# prevision au depart d'un jour donné
date_depart = '2020-11-22'
Preatot1 = [previsiond(datacontexte1,data, depart = date_depart, passe = 20)
          for data in [datareatot]]

trace([(zipper(p['jours'], np.sum(p['valeurs'],axis = 1)),
        p['nom'],'=')
       for p in Preatot1 if len(p['valeurs'] != 0)]
      +
      [(zipper(p['jours'], np.sum(p['valeursmax'],axis = 1)),
        'max','=')
       for p in Preatot1 if len(p['valeurs'] != 0)]
      +
      [(zipper(p['jours'], np.sum(p['valeursmin'],axis = 1)),
        'min','=')
       for p in Preatot1 if len(p['valeurs'] != 0)]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0),7))[-120:],
        '','--')
       for t in [datareatot]],
      'prévision des réanimations à partir du ' + date_depart + '\n',
      'donnees/_prevision_reatot_date_depart')

# previsions a differentes dates
data = datareatot
lP = previsions_multiplesd(datacontexte1,data,50)
trace([(zipper(p['jours'], np.sum(p['valeurs'],axis = 1)),
        '','-')
       for p in lP if len(p['valeurs'] != 0)]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0),7)),
        '','=')
       for t in [data]],
      'prévision des patients en réanimation\n',
      '_previsions_mult_' + data['nom'])

# previsions a differentes dates
data = dataurge
lP = previsions_multiplesd(datacontexte1,data,250)
trace([(zipper(p['jours'], np.sum(p['valeurs'],axis = 1)),
        '','-')
       for p in lP if len(p['valeurs'] != 0)]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0),7)),
        '','=')
       for t in [data]],
      'urgences',
      '_previsions_' + data['nom'])



######################################################################
# prevision cas positifs
pp = previsiond(datacontexte1, datapos)

trace([(zipper(p['jours'], np.sum(p['valeurs'],axis = 1)),
        p['nom'],'=')
       for p in [pp] if len(p['valeurs'] != 0)]
      +
      [(zipper(p['jours'], np.sum(p['valeursmax'],axis = 1)),
        '','=')
       for p in [pp] if len(p['valeurs'] != 0)]
      +
      [(zipper(p['jours'], np.sum(p['valeursmin'],axis = 1)),
        '','=')
       for p in [pp] if len(p['valeurs'] != 0)]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0),7))[-60:],
        '','--')
       for t in [datapos]],
      'prévision des cas positifs\n',
      'donnees/_prevision_positifs')

'''
Page = [(age,previsiond(datacontexte1, dataposage[age]))
        for age in sorted(dataposage) if age != '0']

trace([(zipper(p['jours'], np.sum(p['valeurs'],axis = 1)),
        '','=')
       for (age,p) in Page if len(p['valeurs'] != 0)]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0),7))[-50:],
        age,'--')
       for (age,t) in [(a,dataposage[a]) for a in sorted(list(dataposage)) if a != '0']],
      "prévision des cas positifs par tranche d'âge",
      'donnees/_prevision_positifs_ages', xlabel = 2)
'''
######################################################################
# prevision des hospi par age (regions)
Phospiage = [(age,previsiondcorr(datacontexteregions, datahospiage[age]))
             for age in sorted(datahospiage) if age != '0']

trace([(zipper(p['jours'], np.sum(p['valeurs'],axis = 1)),
        '','=')
       for (age,p) in Phospiage if len(p['valeurs'] != 0)]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0),7))[-60:],
        age,'--')
       for (age,t) in [(a,datahospiage[a]) for a in sorted(datahospiage) if a != '0']],
      "prévision des hospitalisations par tranche d'âge",
      'donnees/_prevision_hospi_ages', xlabel = 20)

Phospiage2 = [(age,previsiondcorr(datacontexteregions, datahospiage[age]))
              for age in ['19']]

trace([(zipper(p['jours'], np.sum(p['valeurs'],axis = 1)),
        '','=')
       for (age,p) in Phospiage2 if len(p['valeurs'] != 0)]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0),7))[-60:],
        age,'--')
       for (age,t) in [(age,datahospiage[age]) for age in ['19']]],
      "prévision des hospitalisations par tranche d'âge",
      'donnees/_prevision_hospi_10-19', xlabel = 20)

####################### dispersion des coefficients des différents départements
# pour la prévision des urgences
p = P[0] # prevision urge
plt.clf()

for d in normalisecoef(p['coefficients']):
    plt.plot(100 * d,'.')

plt.plot(100 * np.mean(normalisecoef(p['coefficients']),axis=0),'-')
plt.xticks(range(len(datacontexte1['contextes'])),
           [x[:7] for x in datacontexte1['contextes']],
           rotation = 45,fontsize = 8)
plt.grid()
plt.title('dispersion des coefficients selon les départements\n (prévision des urgences)',
          fontdict = {'size':8})
plt.savefig('donnees/_dispersion_contexte.pdf', dpi = 600)
plt.savefig('donnees/_dispersion_contexte.png', dpi = 600)
plt.show(False)

Ptout = [pp] + P + Preatot

######################################################################
# prevision à partir de l'hygiene sociale

#datacontexte12 = fusion(datamobilite, datahygiene)
#datacontexte12['contextes'] = datacontexte12['mobilites-hygiene']

datacontexte12 = datahygiene
datacontexte12['contextes'] = datahygiene['hygiene']

v = copy.deepcopy(datacontexte12['valeurs'])

for c in range(len(v[0,0])):
    maxv = np.max(v[:,:,c])
    minv = np.min(v[:,:,c])
    v[:,:,c] = 100 * (v[:,:,c] - minv) / (maxv - minv)

datacontexte12['valeurs'] = v
plt.clf();plt.plot(np.sum(datacontexte12['valeurs'],axis=0));plt.show(False)

trace([(zipper(datacontexte12['jours'],
               np.array(lissage(np.sum(datacontexte12['valeurs'][:,:,l], axis = 0),7))
                       / len(datacontexte12['departements'])),
        datacontexte12['contextes'][l],
        '-')
       for l in range(len(datacontexte12['contextes']))]
      +
      [(zipper(t['jours'],lissage(np.sum(t['valeurs'], axis = 0)/35,7)),
        '','=')
       for t in [dataurge]],
      'contextes',
      'donnees/_contextes_hygiene')

Phygiene = [previsiond1(datacontexte12,dataurge, decale_limite = 500)]
#Phygiene = [previsiond1(datacontexte12,datarea, decale_limite = 100)]
#Phygiene = [previsiond1(datahygiene,datapos, decale_limite = 100)]

p = Phygiene[0]
print('decalage: ',p['decalage'])
plt.clf()
plt.plot(np.transpose(normalisecoef(p['coefficients'])),'o')
plt.grid()
plt.show(False)

trace([(zipper(datahygiene['jours'],
               lissage(np.mean(datahygiene['valeurs'][:,:,l], axis = 0),7)),
        datahygiene['hygiene'][l][:10],
        '-')
       for l in range(len(datahygiene['hygiene']))],
      'hygienes',
      'donnees/_hygiene_urgences')

######################################################################
# mobilite google et température

trace([(zipper(datacontexte1['jours'],
               np.array(lissage(np.sum(datacontexte1['valeurs'][:,:,l], axis = 0),7))
                       / len(datacontexte1['departements'])),
        datacontexte1['contextes'][l],
        '-')
       for l in range(len(datacontexte1['contextes']))],
      'contextes',
      'donnees/_contextes_google')
#>>> [datacontexte1['contextes'][l] for l in [0,1,4,6,9,11]]
#['commerces et espaces de loisir (dont restaurants et bars)', "magasins d'alimentation et pharmacies", 'parcs', 'arrêts de transports en commun', 'travail', 'résidence', 'pression', 'humidité', 'précipitations sur 24', 'température', 'vent', 'vacances', 'en voiture', 'à pied', 'en transport en commun','variations de température]
trace([(zipper(datacontexte1['jours'],
               np.array(lissage(np.sum(datacontexte1['valeurs'][:,:,l], axis = 0),7))
                       / len(datacontexte1['departements']))[-120:],
        datacontexte1['contextes'][l][:30],
        '-')
       for l in [0,12,13,14,1,4,5,7,8,9,11,15] if l < len(datacontexte1['contextes'])],
      'contextes les plus influents (pour toutes les données, lissés sur 7 jours)',
      'donnees/_contextes_influents', xlabel = 38)

######################################################################
# r0

intervalle_seriel = 4.11 # = math.log(3.296)/0.29 

def r0(l):
    # l1: log de l
    l1 = [math.log(x) if x>0 else 0 for x in l]
    # dérivée de l1
    dl1 = derivee(l1,largeur=7) 
    # r0 instantané
    lr0 = [min(4000,math.exp(c*intervalle_seriel)) for c in dl1]
    return(lr0)

trace([(zipper(dataurge['jours'],
               r0(lissage(np.sum(dataurge['valeurs'],axis = 0),7)))[-100:],
        'R urgences','--')]
      +
      [(zipper(dataurge['jours'],
               r0(lissage(np.sum(datahospiurge['valeurs'],axis = 0),7)))[-100:],
        'R hospitalisations urgences','--')],
      'R (taux de reproduction)',
      'donnees/_R')
