from correlation import *

recalcule_lprev = True
duree_erreurs_dpasse = 6*7
duree_erreurs_dpresent = 250
atracerfrance = atracerregions[0][1]
lnomspasR = [nom[1:] for (nom,err) in atracerfrance[0]]

######################################################################
# prévisions à partir du passé, en utilisant uniquement les données passées

if recalcule_lprev:
    dureeprev = 60
    Mp = []
    for s in range(42): # prevision il y a 6 semaines
        MT_1,MF_1,MRF_1,MRDF_1,jourdebut_1= prevoit_tout_deb_pres_fin(M,MR,MRD,intervalle,coefficients,coefs,
                                                                      0,
                                                                      jaujourdhui-s,
                                                                      jaujourdhui-s+dureeprev,
                                                                      recalculecoefficients = True)
        Mp.append(MF_1)
    correctionpop = population_france / sum([population_dep[d] for d in departements])
    for x in lnomspasR:
        x0,x1 = intervalle[ni(x)]
        trace([(zipper(jourstext[x1-s:x1-s+dureeprev],
                       [correctionpop * v
                        for v in np.sum(Mp[s][ni(x),:,x1-s:x1-s+dureeprev],axis=0)]),
                (str(s) if s%7 == 0 else ''),prev)
               for s in range(42)]
              + [(zipper(jourstext[x1-100:x1],
                         [correctionpop * v
                          for v in np.sum(M[ni(x),:,x1-100:x1],axis=0)]),
                '',real)],
              x + ": prévisions à 2 mois, depuis 42 jours.",
              DIRSYNTHESE + '_' + x + '_prev_passees',close = True)

####################################################################
# evaluation et video des prévisions

dureeprevfutur = 120
debutprevpasse = duree_erreurs_dpresent
passe = 260
pasanime = 7

if recalcule_lprev:
    lprev = cree_prev_duree(debutprevpasse = debutprevpasse,
                            dureeprevfutur = dureeprevfutur,
                            pas = pasanime,
                            recalculecoefficients = False)
    # pour éviter de tout recalculer à chaque test
    f = open(DIRCOVID19 + 'previsions_passees.pickle','wb')
    pickle.dump(lprev,f)
    f.close()

f = open(DIRCOVID19 + 'previsions_passees.pickle','rb')
lprev = pickle.load(f)
f.close()
print('fichier des previsions_passees chargé')

######################################################################
# erreurs de prévision, comparaisons

lnoms = [nom for (nom,err) in atracerfrance[0]] + lnomspasR

erreurmax = 50
trop = '<p class="text-muted"> >' + str(erreurmax) + '% </p>'
def couleur3(m,m1,m2):
    if m >= erreurmax:
        return('<div class="bg-dark"><p class="text-white">>' + str(erreurmax) + '%</p></div>')
    elif m <= m1 and m <= m2:
        return('<div class="bg-success"><p class="text-white">' + ("%.0f" % m) + '%</p></div>')
    elif m > m1 and m > m2:
        return('<div class="bg-danger"><p class="text-white">' + ("%.0f" % m) + '%</p></div>')
    else:
        return('<div class="bg-warning"><p class="text-white">' + ("%.0f" % m) + '%</p></div>')
def couleur(m,m1):
    if m >= erreurmax:
        return('<div class="bg-dark"><p class="text-white">>' + str(erreurmax) + '%</p></div>')
    elif m <= m1:
        return('<div class="bg-success"><p class="text-white">' + ("%.0f" % m) + '%</p></div>')
    else:
        return('<div class="bg-warning"><p class="text-white">' + ("%.0f" % m) + '%</p></div>')

def ferreur(le):
    f = np.mean # np.median
    try:
        return(f(le))
    except:
        return(100)

dureeerreur = debutprevpasse//2

def cree_html_erreurs(e,lprev,mode = 'dpresent'):
    sm = [[] for k in range(len(e[lnoms[0]][0]))]
    smlin = [[] for k in range(len(e[lnoms[0]][0]))]
    smq = [[] for k in range(len(e[lnoms[0]][0]))]
    tx = []
    terreur = []
    for x in lnomspasR:
        ek,elink,equadk = e[x]
        # ek[k] = erreurs à k semaines pour chaque prevision, de la + recente à la plus vieille
        t = [['erreur ' + serreur + ' à','présente méthode','prévision par la tangente']]
        terreurx = []
        for k in range(1,len(ek)):
            if k < dureeerreur//7 + 1:
                m = ferreur(ek[k][:dureeerreur//7 + 1 - k])
                sm[k].append(m)
                ml = ferreur(elink[k][:dureeerreur//7 + 1 - k])
                smlin[k].append(ml)
                mq = ferreur(equadk[k][:dureeerreur//7 + 1 - k])
                smq[k].append(mq)
                terreurx.append([7*k,m,ml])
                if m < erreurmax or ml < erreurmax:
                    t.append([str(7*k) + ' jours',
                              couleur(m,ml),
                              couleur(ml,m)])
                else:
                    t.append([str(7*k) + ' jours',' ',' '])
        tx.append((x,table(np.transpose(t))))
        terreur.append((x,terreurx))
    t = [[ ' à','présente méthode','prévision par la tangente']]
    for k in range(1,len(ek)):
        if k < dureeerreur//7 + 1:
            m = np.mean(sm[k])
            ml = np.mean(smlin[k])
            mq = np.mean(smq[k])
            if m < erreurmax or ml < erreurmax:
                t.append([str(7*k) + ' jours',
                          couleur(m,ml),
                          couleur(ml,m)])
            else:
                t.append([str(7*k) + ' jours',' ',' '])
    tmoy = table(np.transpose(t))

    f = open(DIRSYNTHESE + '_evaluation_' + mode + '.html', 'w')
    f.write('<a id="erreursmoyennes"></a>'
            + "<p> <br> <br></p><p><h5>Moyenne des erreurs  " + serreur + "s pour les indicateurs</h5><br>")
    f.write(tmoy)
    f.write("<p><h5> Erreurs  " + serreur + "s par indicateur</h5><br>")
    for x,t in tx:
        f.write("<p><b>" + x + "</b><br>"
                + t
                + '</p>')
    f.close()
    courbes = []
    for x,le in terreur:
        de = dict([(e[0],e[1]) for e in le])
        ve = [(de[j] if j in de else 0) for j in range(le[0][0], le[-1][0]+1)]
        ve = extrapole_manquantes(ve)
        courbes.append((zipper([j for j in range(le[0][0], le[-1][0]+1)],
                               lissage(ve,7)), x, ''))
    trace(courbes,
          "erreurs relatives moyennes (%) des prévisions à x jours\n"
          + ('données du présent' if mode == 'dpresent' else 'données du passé'),
          DIRSYNTHESE + '_erreurs_moyennes_' + mode,close = True)

######################################################################
# avec uniquement les données du passé

if recalcule_lprev:
    lprevdonneespassees = cree_prev_duree(debutprevpasse = debutprevpasse,dureeprevfutur = dureeprevfutur,pas = pasanime,
                                          recalculecoefficients = True)
    # pour éviter de tout recalculer à chaque test
    f = open(DIRCOVID19 + 'previsions_donnees_passees.pickle','wb')
    pickle.dump(lprevdonneespassees,f)
    f.close()

f = open(DIRCOVID19 + 'previsions_donnees_passees.pickle','rb')
lprevdonneespassees = pickle.load(f)
f.close()
print('fichier des previsions_donnees_passees chargé')

edp,lprevdp = evalue(lprevdonneespassees,lnoms,
                     duree_erreurs = duree_erreurs_dpasse,
                     dureeprevfutur = duree_erreurs_dpasse,
                     passe = passe,
                     maxerreur=100,
                     pas = pasanime)

cree_html_erreurs(edp,lprevdp,mode = 'dpasse')

######################################################################
# avec les données du présent

e,lprev = evalue(lprev,lnoms,
                 duree_erreurs = debutprevpasse,
                 dureeprevfutur = dureeprevfutur,
                 passe = passe,
                 maxerreur=100,
                 pas = pasanime)

cree_html_erreurs(e,lprev,mode = 'dpresent')

######################################################################
# animation des prévisions du passé

for nom in lnoms:
    anime_previsions(lprev,nom,duree = debutprevpasse,passe=passe, dureeprevfutur = dureeprevfuturanime, pas = pasanime)

