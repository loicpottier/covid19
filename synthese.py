from correlation import *
import time
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytz import reference
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import colors

now = time.localtime(time.time())
######################################################################
# page de synthèse
# mets la synthèse sur cp.lpmib.fr:
# https://cp.lpmib.fr/medias/covid19/_synthese.html
######################################################################

debut1 = '''<!DOCTYPE html><meta charset="UTF-8">
<head>
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-165293312-1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-165293312-1');
</script>
<script src="https://code.jquery.com/jquery-3.4.1.slim.min.js" integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js" integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" crossorigin="anonymous"></script>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
<div class="container-fluid">
 '''

fin1 = '''
</div>
</body>
'''
def images2(f,x,y):
    f.write('<tr valign=top>')
    f.write('<td valign = top> '
            + '<img src="' + DIRSYNTHESE + '' + x + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">' + '</td>\n')
    f.write('<td valign = top> '
            + '<img src="' + DIRSYNTHESE + y + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">' + '</td>\n')
    f.write('</tr>\n')

def images(f,lim):
    f.write('<tr valign=top>')
    for im in lim:
        f.write('<td valign = top> '
                + '<img src="' + DIRSYNTHESE + im
                + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">'
                + '</td>\n')
    f.write('</tr>\n')

def image_texte(f,x,y):
    f.write('<tr valign=top>')
    f.write('<td valign = top> '
            + '<img src="' + DIRSYNTHESE + x + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">' + '</td>\n')
    f.write('<td valign = top> '
            + '<div class="container-fluid"><p>' + y + '</p></div></td>\n')
    f.write('</tr>\n')

def image(x):
    return('<img src="' + DIRSYNTHESE + x
           + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">')

def imagew(x,w):
    return('<img src="' + DIRSYNTHESE + x
           + '" width="' + str(w) + '%" alt="Responsive image">')

def images1(f,x):
    f.write('<tr valign=top>')
    f.write('<td valign = top> '
            + image(x) + '</td>\n')
    f.write('<td valign = top> </td>\n')
    f.write('</tr>\n')

def video(vid):
    return("<div class=\"embed-responsive embed-responsive-4by3\">"
           + "<iframe class=\"embed-responsive-item\" src=\""
           + DIRSYNTHESE
           + vid + "\" allowfullscreen></iframe>"
           + "</div>")

def videos(f,lv):
    f.write('<tr valign=top>')
    for vid in lv:
        f.write('<td valign = top>')
        f.write(video(vid))
        f.write('</td>')
    f.write('</tr>')

def normalise_nom(x):
    for c in '.,éèêàâôûù;-, ':
        x = x.replace(c,'')
    return(x)

def tabs(lt):
    t = str(time.time()).replace('.','')
    r = '<ul class="nav nav-tabs" id="myTab" role="tablist">'
    for (k,(nom,contenu)) in enumerate(lt):
        nom2 = normalise_nom(nom) + t
        r += ('<li class="nav-item"><a class="nav-link' + (' active' if k == 0 else '')
              + '" id="'
              + nom2 + '-tab" data-toggle="tab" href="#'
              + nom2 + '" role="tab" aria-controls="'
              + nom2 + '" aria-selected="true">'
              + nom + '</a></li>')
    r += '</ul>'
    r += '<div class="tab-content">'
    for (k,(nom,contenu)) in enumerate(lt):
        nom2 = normalise_nom(nom) + t
        r += ('<div id="'
              + nom2 + '" class="tab-pane fade show' + (' active' if k == 0 else '')
              + '">'
              + contenu
              + '</div>')
    r += '</div>'
    return(r)

def table2(l):
    r = '<table class="table">'
    for x in l:
        r += ('<tr valign=top>')
        for y in x:
            r += '<td valign = top>'
            r += y
            r += '</td>'
        r += '</tr>'
    r += '</table>'
    return(r)

######################################################################
# heatmap correlations

def heat_map(nomsx,nomsy,fichier, fontcase = 5, dim = (10,10)):
    t = []
    for x in nomsx:
        ligne = []
        for y in nomsy:
            dec,corr,x0,x1,y0,y1 = coefs[ni(x),ni(y)]
            if x == y:
                dec = 0
                corr = 1
            if ni(x) not in dependances(ni(y),intervalle,coefs): #abs(corr) <= mincorrelation:
                dec = 0
            try:
                ligne.append([x,y,corr,int(dec)])
            except:
                print('probleme',corr,dec)
                ligne.append([x,y,0,0])
        t.append(ligne)
    trech = [x for x in t if 'recherche' in x[0][0]]
    tind = [x for x in t if x[0][0] in nomsind]
    treste = [x for x in t if x not in trech + tind]
    t = sorted(treste,
               key = lambda x: -moyenne([y[2] for y in x
                                         if y[1] in nomsind and 'positifs' not in y[1]])) + trech + tind
    noms1 = [x[0][0][:25] for x in t]
    noms2 = [x[1][:25] for x in t[0]]
    tcorr = np.array([[y[2] for y in x] for x in t])
    tdec = np.array([[y[3] for y in x] for x in t])
    fig = plt.figure(figsize = dim)
    fig.suptitle('corrélations (+:rouge, -: vert) et décalages en jours (nombres),\n',
                 fontdict = {'size':4},y = 1. )
    ax = plt.gca()
    im = ax.imshow(tcorr,cmap="RdYlGn_r")
    images = [im]
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=-0.6, vmax=0.6)
    im.set_norm(norm)
    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
    ax.set_xticks(np.arange(len(noms2)))
    ax.set_yticks(np.arange(len(noms1)))
    ax.set_xticklabels(noms2,fontdict = {'size':6})
    ax.set_yticklabels(noms1,fontdict = {'size':6})
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")
    for i in range(len(noms1)):
        for j in range(len(noms2)):
            text = ax.text(j, i, tdec[i, j] if tdec[i,j] != 0 else '',
                           ha="center", va="center", color="k",fontdict = {'size':fontcase})
    ax.set_title("corrélations (couleurs) et décalages en jours (nombres) entre données",y = -0.9)
    plt.savefig(DIRSYNTHESE + fichier + '.png',dpi = 600)
    plt.savefig(DIRSYNTHESE + fichier + '.pdf',dpi = 600)
    print('heatmap créée:',fichier)
######################################################################
# heatmap transmission entre tranches d'âges
def normalise2(v):
    v = lissage(lissage(v,7),7)
    vm = np.max(v)
    return(v/vm)

j1 = num_de_jour('2020-09-01') - jours[0]
j2 = num_de_jour('2020-11-01') - jours[0]

hages = sorted([age for age in datahospiage if age != '0'])
nhages = len(hages)

coefs2 = np.zeros((nhages,nhages,2))
decmax = 40
decmaxaccepte = 30

for (x,age1) in enumerate(hages):
        v1 = normalise2(np.sum(datahospiage[age1]['valeurs'][:,:],
                               axis=0))
        v1 = derivee(v1,7)
        for (y,age2) in enumerate(hages):
            if age1 != age2:
                v2 = normalise2(np.sum(datahospiage[age2]['valeurs'][:,:],
                                       axis=0))
                v2 = derivee(v2,7)
                lcsm = correlate(v1,v2[decmax:])
                d = np.argmax(np.abs(lcsm))
                corr = lcsm[d]
                d = decmax - d
                if d == 0 or d > decmaxaccepte:
                    d = 0
                    corr = lcsm[-1]
                coefs2[x,y] = [d,corr]

# heatmap decalages
fontcase = 10
dim = (6,6)
t = []
for (kx,x) in enumerate(hages):
    ligne = []
    for (ky,y) in enumerate(hages):
        dec,corr = coefs2[kx,ky]
        if x == y:
            dec = 0
            corr = 1
        ligne.append([x,y,corr,int(dec)])
    t.append(ligne)

noms1 = ['hospitalisations ' + x[0][0][:25] for x in t]
noms2 = [x[1][:25] for x in t[0]]
tcorr = np.array([[y[2] for y in x] for x in t])
tdec = np.array([[y[3] for y in x] for x in t])
fig = plt.figure(figsize = dim)
fig.suptitle('corrélations (couleurs) et décalages en jours (nombres),\n',
             fontdict = {'size':4} )
ax = plt.gca()
im = ax.imshow(tcorr,cmap="YlGn")
images = [im]
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=0., vmax=1.)
im.set_norm(norm)
fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
ax.set_xticks(np.arange(len(noms2)))
ax.set_yticks(np.arange(len(noms1)))
ax.set_xticklabels(noms2,fontdict = {'size':8})
ax.set_yticklabels(noms1,fontdict = {'size':8})
ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)
plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
         rotation_mode="anchor")
for i in range(len(noms1)):
    for j in range(len(noms2)):
        text = ax.text(j, i, tdec[i, j]
                       if tdec[i,j] != 0 and tcorr[i,j] > 0.2 else '',
                       ha="center", va="center", color="k",
                       fontdict = {'size':fontcase})

fichier = 'hospitalisations_age_heat_corr_dec'
plt.savefig(DIRSYNTHESE + fichier + '.png',dpi = 600)
plt.savefig(DIRSYNTHESE + fichier + '.pdf',dpi = 600)
print('heatmap tranches d ages créée:',fichier)

######################################################################
synthese = '_synthese.html'
f = open(synthese, 'w', encoding = 'utf8')

f.write(debut1)

f.write('''
<nav class="navbar fixed-top navbar-expand-lg navbar-dark bg-success">
  <a class="navbar-brand" href="#">Covid19: la suite?</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavAltMarkup" aria-controls="navbarNavAltMarkup" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNavAltMarkup">
    <div class="navbar-nav">
      <a class="nav-item nav-link active" href="#previsions">Données et prévisions</a>
      <a class="nav-item nav-link" href="#alpesmaritimes">06</a>
      <a class="nav-item nav-link" href="#iledefrance">IDF</a>
      <a class="nav-item nav-link" href="#previsions3mois">Prévisions depuis 8 mois<span class="sr-only">(current)</span></a>
      <a class="nav-item nav-link" href="#methode">Méthode</a>
      <a class="nav-item nav-link" href="#correlations">Corrélations et décalages</a>
      <a class="nav-item nav-link" href="#coefficients">Coefficients d'optimisation</a>
      <a class="nav-item nav-link" href="#evaluation">Évaluation</a>
      <a class="nav-item nav-link" href="https://github.com/loicpottier/covid19">Code</a>
    </div>
  </div>
</nav>
''')

vspace = '<p> <br> <br></p>'*1 # ruse dégueue pour pas que la navbar se superpose

f.write(vspace + '<div class="container-fluid">'
        + '<p>' + time.asctime(now) + '</p>'
        + '<p>loic.pottier@gmail.com<a href="https://twitter.com/LocPottier1"><img src="https://help.twitter.com/content/dam/help-twitter/brand/logo.png" width = 50></a></p>')

texte_intro = table2([['''<img src = "https://images-na.ssl-images-amazon.com/images/I/41SWU0l77iL._AC_.jpg" width = 300>''',
                       '''
<h4>Données et prévisions des indicateurs principaux de l'épidémie de covid19 en France</h4>
<p>La méthode employée est mathématique.<br>
                       Elle produit des prévisions pour le nombre de patients en réanimation avec une <a href="#erreursmoyennes">erreur moyenne</a> de 10% à 1 mois, 20% à 2 mois, et 30% à 3 mois. Pour les autres indicateurs de l'épidémie l'erreur est  de 15% à 20% jusqu'à 1 mois. </p>
                       <p>Elle débute par le calcul de corrélations maximales entre les données des <em>indicateurs</em> de l'épidémie (hospitalisations, réanimations, nombre de cas, décès, etc)<br>
                       et celles, décalées arbitrairement dans le temps, des données de son <em>contexte </em> (vacances, météo, mobilité des personnes, couvre-feu, etc),<br>
pour chaque jour et chaque département.<br>
On en déduit des décalages temporels entre contextes et indicateurs.<br>
                       Puis, à partir de ces décalages, on calcule une approximation et une prévision linéaires par optimisation quadratique des taux de reproduction effectifs des indicateurs (R effectif, Reff, R0 ou R dans la littérature). Les approximations des indicateurs dont le taux de reproduction effectif est approximé avec une erreur moyenne supérieure à 5% sont rejetées.<br>
                       Enfin, pour les autres, on déduit des approximations et des prévisions des indicateurs correspondants de l'épidémie.<br>
                       Ceci est fait pour chaque département (pour les départements où les données complètes du jour sont présentes au moment du calcul, i.e. en général autour de 88).<br>
                       On présente ici les résultats synthétisés pour la France, pour les Alpes-Maritimes et pour l'Ile de France.</p>
On trouvera une <a href = \"#methode\">description détaillée</a> et une <a href = \"#evaluation\">évaluation sommaire</a> de la méthode ci-dessous.</p>
<p>Les seules hypothèses qui sont faites sont que les données du contexte  gardent dans l'avenir les valeurs qu'elles ont au jour présent, sauf pour la météo, où l'on reprend les valeurs de l'année passée au même moment, et pour les vacances scolaires, qui sont prévues de longue date.</p>''']])

f.write(texte_intro)


texte_prevision = '''<h4>Prévisions</h4>
<p>Courbe bleue: données réelles.<br>
Courbe orange: approximation et prévision (des données lissées sur 7 jours).</p>
<p>Les indicateurs sont rangés par erreur croissante sur leur Reff approximé par rapport aux Reff réels (les approximations sont calculées sur toute la période de l'épidémie).
Les indicateurs avec des erreurs sur Reff >5% ne sont pas indiqués. Les indicateurs extensifs de l'épidémie sont cumulés sur les départements étudiés, les autres (taux de positifs), sont moyennés. Les Reff associés sont la moyenne des Reff des départements.</p>
'''

f.write("<a id=\"previsions\"></a>" + texte_prevision)

f.write("<p>On pourra se faire une idée de la précision de ces prévisions avec "
        + '<a href="#previsionspassees">ces tracés.</a>'
        + "</p>")

def ecrit_previsions_region(atracerregion):
    atracer,fichiersR,fichiers = atracerregion
    f.write(table2([["<h5>Indicateurs</h5>",
                     "<h5>Reff</h5>"], 
                    [tabs([(nom[1:],image(fichiers[k] + '.png'))
                           for (k,(nom,err)) in enumerate(atracer)]),
                     tabs([(nom,image(fichiersR[k] + '.png'))
                           for (k,(nom,err)) in enumerate(atracer)]),
                    ]]
    ))

ecrit_previsions_region(atracerfrance)

def dernierjour(x):
    return(jour_de_num[jours[0] + intervalle[ni(x)][1]-1])

f.write('<h4>Indicateurs et contextes</h4>'
        + "<p>Les indicateurs de la covid19 utilisés sont des données quotidiennes (courbes en traits gras ci-dessus), dont:"
        + '<ul><li>urgences: nombre de passages aux urgences pour suspicion de COVID-19 - Quotidien'
        + " (jusqu'au " + dernierjour('urgences') + ")"
        + '<li>hospitalisations: nombre quotidien de nouvelles personnes hospitalisées avec diagnostic COVID-19 déclarées en 24h'
        + " (jusqu'au " + dernierjour('nouv hospitalisations') + ")"
        + '<li>réanimations: nombre quotidien de nouvelles admissions en réanimation (SR/SI/SC) avec diagnostic COVID-19 déclarées en 24h'
        + " (jusqu'au " + dernierjour('nouv réanimations') + ")"
        + '<li>décès: nombre quotidien de nouveaux décès avec diagnostic COVID-19 déclarés en 24h'
        + " (jusqu'au " + dernierjour('nouv décès') + ")"
#        + '<li>tests positifs: nombre quotidien de nouveaux tests positifs au COVID-19 en 24h'
#        + " (jusqu'au " + dernierjour('positifs') + ")"
#        + "<li>patients en réanimation, hospitalisés, par tranches d'âge"
#        + " (jusqu'au " + dernierjour('réanimations') + ")"
#        + "<li>personnes testées positives, taux de tests positifs, par tranches d'âge"
#        + " (jusqu'au " + dernierjour('taux positifs') + ")"
        + "</ul></p>"
        + "<p>Les données concernent chaque département (ou région) et chaque jour depuis mars 2020, et proviennent de <a href=\"https://www.data.gouv.fr/fr/pages/donnees-coronavirus\">www.data.gouv.fr</a></p>"
)
f.write("<p>Les prévisions (courbes en tirets ci-dessus) sont calculées à partir</p>"
        + "<ul><li>des données de <a href=\"https://www.google.com/covid19/mobility/\">mobilité de Google</a>"
        + " (jusqu'au " + dernierjour('travail') + ")"
        + "<li>des données de <a href=\"https://covid19.apple.com/mobility\">mobilité d'Apple</a>"
        + " (jusqu'au " + dernierjour('en transport en commun') + ")"
        + "<li>des données <a href=\"https://www.data.gouv.fr/fr/datasets/donnees-d-observation-des-principales-stations-meteorologiques/\">météo</a>"
        + " (jusqu'au " + dernierjour('température') + ", les données météo à venir sont déduites de l'année passée)"
        + "<li>des dates des vacances scolaires"
        + "<li>de données de recherche de mots sur Google (<a href=\"https://trends.google.fr/trends/explore?geo=FR&q=voyage\">Google Trends</a>)"
        + " (jusqu'au " + dernierjour('recherche voyage google') + ")"
        + "<li>des dates de confinement et de couvre-feu"
        + "<li>des vaccinations"
        + "<li>des taux des variants"
        + "</ul>"
        + "</p>")

######################################################################
# corrélations

voyage_google = '''<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/2431_RC04/embed_loader.js"></script>
  <script type="text/javascript">
    trends.embed.renderExploreWidget("TIMESERIES", {"comparisonItem":[{"keyword":"voyage","geo":"FR","time":"today 12-m"}],"category":0,"property":""}, {"exploreQuery":"geo=FR&q=voyage&date=today 12-m","guestPath":"https://trends.google.fr:443/trends/embed/"});
  </script>'''

f.write('<a id="methode">'
        + vspace + '</a><h3>Méthode</h3><br>'
        + table0([[
            "<p><h5>Prévisions</h5>"
            + "Les données concernent les indicateurs de l'épidémie (urgences, réanimations, décès, tests positifs, etc) et les contextes (données météo: température, pression, données de mobilité fournies par google: fréquentation des commerces et lieux de loisir, des lieux de travail, etc).<br>"
            + "Elles se présentent comme un tableau à 3 dimensions: une pour les noms des données, une pour les départements français, une pour les jours. Par exemple, on dispose du nombre de passage aux urgences hospitalières, du nombre de patients en réanimation, etc, pour soupçon de covid chaque jour depuis avril 2020, pour chaque département.<br>"
            + "En tout on utilise "
            + str(nnoms) + " jeux de données, qui concernent "
            + str(ndeps) + " départements et "
            + str(njours) + " jours.<br>"
            + "On commence par calculer les coefficients de corrélation entre deux jeux de données, pour tous les décalages temporels d'au plus 40 jours. Et on retient le décalage qui maximise la valeur absolue du coefficient de corrélation entre deux données, si celle-ci est supérieure à "
            + ("%1.2f" % mincorrelation) + ": par exemple, ce coefficient de corrélation est maximal entre le travail et les urgences si on décale les urgences "
            + str(int(coefs[ni('travail'),ni('urgences')][0]))
            +  " jours dans le passé.<br>"
            +  "Cela suggère une causalité possible: une hausse de la fréquentation des lieux de travail peut provoquer une hausse des urgences pour covid "
            + str(int(coefs[ni('travail'),ni('urgences')][0]))
            +  " jours plus tard.<br>"
            + "A présent on dispose de décalages de jours entre certaines des données. On dira qu'une donnée \(d\) dépend d'une donnée \(d'\) si on a obtenu, à l'étape précédente, un décalage \(\Delta_{d' d} > 0\) de \(d'\) à \(d\).<br>"
            + "On considère alors qu'une donnée \(d\) sur une période de temps \([j_0,j_1]\) va dépendre des valeurs des donnéees \(d_i\) dont elle dépend sur les périodes \([j_0 - \Delta_{d' d_i},j_1 - \Delta_{d' d_i}]\).<br>"
            + "Fixons un dépatement. Appelons \(A\) la matrice dont la colonne \(i\) est formée des valeurs de \(d_i\) dans ce département sur la période \([j_0 - \Delta_{d' d},j_1 - \Delta_{d' d}]\), et \(B\) le vecteur colonne formé des valeurs de \(d\) dans ce département sur la période \([j_0,j_1]\).<br>",
            "On aimerait trouver une famille de coefficients \(C = (c_i)\) telle que \(AC = B\). Mais il n'y a pas en général de solution à cette équation, car \(A\) a plus de lignes (les jours) que de colonnes (les données). On cherche alors à minimiser \(||AC-B||\). C'est un problème quadratique convexe, dont la solution s'obtient simplement avec \[C = (^tAA)^{-1}(^tAB)\] (si \(^tAA\) est inversible, ce qui est le cas en pratique).<br>"
            + "Avec \(C\) on peut prévoir une valeur pour la donnée \(d\) le jour \(j_1+1\), simplement en calculant \(A_1C\), où \(A_1\) est obtenue comme \(A\) mais avec les intervalles \([j_0 - \Delta_{d' d}+1,j_1 - \Delta_{d' d}+1]\). Et ainsi de suite.<br>"
            + "On prévoit alors toutes les données d'un jour en parallèle, puis le suivant, etc. Si une donnée n'est pas prévisible (car pas de dépendance, ou  \(^tAA\) non  inversible), on garde sa valeur précédente.<br>"
            + "On fait cela pour chaque département.<br>"
            + "Évidemment, si une prévision dépend de données déjà obtenues par prévision, on imagine bien qu'elle est moins fiable.</p>"
            + "<p><h5>Important</h5>"
            + "D'une part, dans les calculs de corrélation, les données qui concernent les indicateurs de l'épidémie sont utilisées sous la forme de leur dérivée discrète \(d(j+1)-d(j)\). Comme ce sont déjà des valeurs par jour, donc des dérivées, on obtient alors la dérivée seconde des données cumulées. On calcule alors des corrélations entre les données de contexte, que l'on peut voir comme des forces, et les dérivées secondes des indicateurs absolus de l'épidémie.<br>"
            + "Cette analogie linéaire entre force et accélération empruntée à la physique newtonienne s'avère féconde: en effet les corrélations sont négligeables si on ne considère pas les dérivées discrètes des indicateurs journaliers, alors qu'au contraire elles apparaissent avec celles-ci.<br>"
            + "D'autre part, pour prévoir les indicateurs de l'épidémie on commence par prévoir le taux de reproduction associé (voir plus loin), puis on en déduit les valeurs prévues des données."
            + "</p>"],
        ["<hr>","<hr>"],
        ['''<p><h5>Calcul du taux de reproduction effectif \(R_{\\rm eff}\)</h5>
Le taux de reproduction R est le nombre moyen de personnes que contamine un malade (en dessous de 1, l'épidémie régresse). Il est difficile de le déterminer, car il change chaque jour, et on ne connaît pas tous les malades et qui les contamine.<br> Pour l'approcher, on déterminera pour chaque indicateur de l'épidémie un taux de reproduction effectif \(R_{\\rm eff}\), en utilisant un intervalle sériel estimé à \(s = 4,11\) (c'est le nombre moyen de jours entre deux contaminations successives dans une chaîne de contamination).<br>
Si l'indicateur de l'épidémie choisi est donné chaque jour par une fonction \(f\), alors son \(R_{\\rm eff}\) vérifie: \[R_{\\rm eff} = e^{s \\frac {f'} f}\]  de sorte que \(f(x+s) = R_{\\rm eff} f(x)\). 
On obtient cette expression simplement en considérant \(f\) comme localement exponentielle.<br>
Notons que la dérivée discrète est en pratique calculée sur les valeurs lissées 2 fois sur 7 jours.</p>''',
         ""]]))


f.write("<a id=\"correlations\"></a>"
        + vspace + '<h3>Corrélations</h3>'
        "<p>Attention: corrélation ne veut pas dire causalité, mais bon, cela peut donner des idées.</p>")

f.write(table2([[tabs([('contexte-indicateur',image('_heat_correlation_contexte_indicateur.png')),
                       ('confinement,couvre-feu',
                        image('_heat_correlation_confinement_indicateur.png')),
                       ('complètes',image('_heat_correlation_tout.png'))])]]))

f.write('<a id="coefficients"></a>'
        + vspace + "<h3>Coefficients d'optimisation</h3>"
        "<p>Dépendances et coefficients d'optimisation de différents indicateurs.<br>"
        + "Contextes influents.</p>")

f.write(table2([[tabs([('contextes influents',image('_contextes_influents.png'))]
                      +[('recherche Google',image('_recherche google.png'))]
                      +[('plancher des urgences',image('_tendance_min_urgences.png'))]),
                 tabs([('coef. ' + nom,image('_coefficients_' + nom + '.png'))
                       for (nom,err) in atracerfrance[0]]),
]]))

f.write('<a id="evaluation"></a>'
        + vspace + "<h3>Évaluation de la méthode de prévision</h3>"
        + "<p>On effectue deux évaluations différentes: on compare avec d'autres méthodes, et on compare les prévisions obtenues à partir des jours passés avec la réalité, avec deux méthodes différentes.</p>")

f.write(vspace + "<h4>Comparaisons avec d'autres méthodes simples.</h4>"
        + "<p>On compare les prévisions avec une prévision linéaire par la tangente, et une prévision par approximation à l'ordre 2<br>"
        + "Les prévisions à partir du jour \(j\) n'utilisent que les données antérieures au jour \(j\), mais avec les coefficients de prévisions calculés sur l'ensemble des données jusqu'à aujourd'hui.<br>"
        + "Les méthodes simples donnent toujours de erreurs supérieures, et dépassent 50% d'erreur après 40 jours "
        + '<a href="#methode">(détails ici)</a>.<br>'
        + "Voici les erreurs de notre méthode:</br>")

f.write(table2([[imagew('_erreurs_moyennes.png',40)]]))


f.write("<a id=\"previsionspassees\"></a>"
        + "<h4>Prévisions passées depuis 6 semaines.</h4>"
        + "<p>Pour avoir une idée de la pertinence des prévisions précédentes, on calcule les prévisions obtenues à partir du passé, comparées à la réalité:</p>"
        +"<p>Courbe bleue: données réelles.<br>"
        + "Courbe pointillées: données approximées puis prévues à partir du jour \(j\) "
        + "en utilisant uniquement les données des jours précédant \( j\).</p>"
        + table2([[tabs([('réanimations',image('_réanimations_prev_passees.png'))]),
                   tabs([(nom[1:],image('_' + nom[1:] + '_prev_passees.png'))
                         for (nom,err) in atracerfrance[0]
                         if nom != 'Rréanimations'])]]))

f.write("<a id=\"previsions3mois\"></a>"
        + vspace + "<h4>Prévisions passées sur "+ str(dureefuturanime) + " jours, depuis 8 mois</h4>"
        + "<p>Courbe bleue: données réelles.<br>"
        + "Courbes orange à rose: données approximées puis prévues à partir du jour \(j\) "
        + "en utilisant les données des jours précédant \( j\), mais avec les coefficients de prévisions calculés sur l'ensemble des données jusqu'à aujourd'hui.<br>"
        + "Courbe rouge pointillée: moyenne des prévisions.</p>")


f.write(table2([[tabs([(nom[1:],video('previsions_' + nom[1:] + '.mp4'))
                       for (nom,err) in atracerfrance[0]]),
                 tabs([(nom,video('previsions_' + nom + '.mp4'))
                       for (nom,err) in atracerfrance[0]]),
                 ]]
               ))

heat_map(nomscont + nomsind, nomsind + nomscont,'_heat_correlation_tout',
         fontcase = 4,
         dim = (10,14))
heat_map(nomscont, nomsind,'_heat_correlation_contexte_indicateur',
         dim=(10,10))
if inclusconfinement:
    heat_map(dataconfinement['confinement'], nomsind,
             '_heat_correlation_confinement_indicateur',
             dim = (10,4))

######################################################################
f.write('<a id="alpesmaritimes"></a>'
        + vspace + "<h3>Dans les Alpes-Maritimes</h3>")

ecrit_previsions_region(atracer06)

######################################################################
f.write('<a id="iledefrance"></a>'
        + vspace + "<h3>En Ile de France</h3>")

ecrit_previsions_region(atraceriledefrance)

######################################################################
f.write("<a id=\"transmission\"></a>"
        + vspace + "<h4>Transmission entre classe d'âges</h4>"
        + "<p>Sur un an, les corrélations entre hospitalisations par classes d'âges suggèrent une contamination des jeunes adultes vers les plus vieux, par âges croissants, avec un décalage d'une dizaine de jours.</p>"
        + "<p>"+ imagew('hospitalisations_age_heat_corr_dec.png',40) + "</p>")

f.write("<a id=\"erreurs\"></a>"
        + "<h4>Détail des erreurs avec différentes méthodes.</h4>")
f1 = open(DIRSYNTHESE + '_evaluation.html', 'r')
f.write(f1.read())
f1.close()

f.write(fin1)
f.close()

os.system('scp _synthese.html lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19/_synthese.html')
os.system('tar -cvf groba.tgz ' + DIRSYNTHESE + '/*.png ' + DIRSYNTHESE + '/*.mp4')

os.system('scp groba.tgz lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19')
os.system("ssh lpmib@ssh-lpmib.alwaysdata.net 'cd testdjango2/testdjango2/medias/covid19 && tar -xvf groba.tgz'")


