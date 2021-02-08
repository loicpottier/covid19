from correlation import *
######################################################################
# video des prévisions

futur = 60 #60
duree = 280 # 250 # num_de_jour(aujourdhui) - jours[0] - 14 # 90
passe = 270 # 260 # duree + 7 # 100
lprev = cree_prev_duree(duree = duree,futur = futur)
anime_previsions(lprev,'urgences',duree = duree,passe=passe, futur = futur)
#anime_previsions(lprev,'sosmedecin',duree = duree,passe=passe, futur = futur)
anime_previsions(lprev,'hospitalisation urgences',duree = duree,passe=passe, futur = futur)
anime_previsions(lprev,'réanimations',duree = duree,passe=passe, futur = futur)
anime_previsions(lprev,'nouv réanimations',duree = duree,passe=passe, futur = futur)
anime_previsions(lprev,'hospitalisations',duree = duree,passe=passe, futur = futur)
anime_previsions(lprev,'taux positifs',duree = duree,passe=passe, futur = futur)
anime_previsions(lprev,'positifs',duree = duree,passe=passe, futur = futur)
anime_previsions(lprev,'nouv décès',duree = duree,passe=passe, futur = futur)
anime_previsions(lprev,'nouv hospitalisations',duree = duree,passe=passe, futur = futur)

######################################################################
# erreurs de prévision, comparaisons

lnom = ['urgences', 'hospitalisation urgences',#'sosmedecin',
        'réanimations', 'hospitalisations',
        'nouv réanimations','nouv hospitalisations',
        'nouv décès', 'taux positifs', 'positifs']
e,lprev = evalue(lnom,
                 duree = 90,
                 maxerreur=1000)

f = open(DIRSYNTHESE + '_evaluation.html', 'w')

erreurmax = 50
trop = '<p class="text-muted"> >' + str(erreurmax) + '% </p>'
def couleur(m,m1,m2):
    if m >= erreurmax:
        return('<div class="bg-dark"><p class="text-white">>' + str(erreurmax) + '%</p></div>')
    elif m <= m1 and m <= m2:
        return('<div class="bg-success"><p class="text-white">' + ("%.0f" % m) + '%</p></div>')
    elif m > m1 and m > m2:
        return('<div class="bg-danger"><p class="text-white">' + ("%.0f" % m) + '%</p></div>')
    else:
        return('<div class="bg-warning"><p class="text-white">' + ("%.0f" % m) + '%</p></div>')

sm = [[] for k in range(len(e['urgences'][0]))]

for x in lnom:
    f.write("<p>" + x + ":<br>")
    ek,elink,equadk = e[x]
    t = [['erreur médiane à','présente méthode','prévision par la tangente',
          'prévision à l\'ordre 2']]
    for k in range(1,len(ek)):
        m = np.median(ek[k])
        sm[k].append(m)
        ml = np.median(elink[k])
        mq = np.median(equadk[k])
        if m < erreurmax or ml < erreurmax:
            t.append([str(7*k) + ' jours',
                      couleur(m,ml,mq),
                      couleur(ml,m,mq),
                      couleur(mq,ml,m)])
        else:
            t.append([str(7*k) + ' jours',' ',' ',' '])
    f.write(table(np.transpose(t)))
    f.write('</p>')

f.write("<p>Erreurs médianes moyennes:<br> "
        + '<br>'.join(["à " + str(7*(k+1)) + " jours: " + ("%.0f" % np.mean(x)) + "%"
                    for (k,x) in enumerate(sm[1:])])
        + "</p>")

f.close()
