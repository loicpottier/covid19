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
            if ni(x) not in dependances(ni(y),coefs): #abs(corr) <= mincorrelation:
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
    fig.suptitle('corrélations (couleurs) et décalages en jours (nombres),\n'
                 #+ 'les décalages des coefficients de corrélation\n'
                 #+ 'entre -' + ("%0.2f" % mincorrelation) + ' et '
                 #+ ("%0.2f" % mincorrelation) + ' ne sont pas pris en compte'
                 , fontdict = {'size':6} )
    ax = plt.gca()
    im = ax.imshow(tcorr,cmap="RdYlGn_r")
    images = [im]
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=-0.75, vmax=0.75)
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
      <a class="nav-item nav-link active" href="#previsions">Données et prévisions aujourd'hui</a>
      <a class="nav-item nav-link" href="#previsions3mois">Prévisions depuis 3 mois<span class="sr-only">(current)</span></a>
      <a class="nav-item nav-link" href="#methode">Méthode</a>
      <a class="nav-item nav-link" href="#correlations">Corrélations et décalages</a>
      <a class="nav-item nav-link" href="#coefficients">Coefficients d'optimisation</a>
      <a class="nav-item nav-link" href="#evaluation">Évaluation</a>
      <a class="nav-item nav-link" href="#alpesmaritimes">06</a>
      <a class="nav-item nav-link" href="https://github.com/loicpottier/covid19">Code</a>
    </div>
  </div>
</nav>
''')

vspace = '<p> <br> <br></p>'*1 # ruse dégueue pour pas que la navbar se superpose

f.write(vspace + '<div class="container-fluid">'
        #+ '<h3> Prévisions des indicateurs de la covid 19.</h3>'
        + '<p>' + time.asctime(now) + '</p>'
        + '<p>loic.pottier@gmail.com<a href="https://twitter.com/LocPottier1"><img src="https://help.twitter.com/content/dam/help-twitter/brand/logo.png" width = 50></a></p>')

f.write("<h4>Données et prévisions des indicateurs principaux de l'épidémie</h4>"
        + "<p>On suppose qu'après les vacances de février, les collèges et lycées fonctionnent encore en demi-effectif.</p>"
        + "<p>On considère que le couvre-feu de 18h à 6h sera levé le 1er mars 2021.</p>"
        +"<p>La méthode employée est fondée sur les corrélations maximales entre dérivées des données dont on déduit les décalages temporels, puis, à partir de ces décalages, sur la prévision linéaire par optimisation quadratique des dérivées des données, et enfin sur l'intégration de ces dérivées.<br>"
        + "Tout cela département par département (pour les "
        + str(ndeps) + " départements où les données complètes du jour sont présentes au moment du calcul). Les données météo à venir sont déduites de l'année passée. On n'utilise aucune modélisation.<br>"
        + "On trouvera une <a href = \"#methode\">description</a> et une <a href = \"#evaluation\">évaluation</a> de la méthode ci-dessous.</p>")


f.write("<a id=\"previsions\"></a>"
        + vspace + "<h4>Prévisions aujourd'hui</h4>"
        + "<p>Courbe bleue: données réelles.<br>"
        + "Courbe orange: prévision (des données lissées sur 7 jours).")

f.write(table2([[tabs([('urgences',image('_prevision_urgences.png')),
                       #('sosmédecin',image('_prevision_sosmedecin.png')),
                       ('hospitalisations urgences',image('_prevision_hospitalisation urgences.png')),
                       ('réanimations',image('_prevision_réanimations.png')),
                       ('hospitalisations',image('_prevision_hospitalisations.png'))]),
                 tabs([('nouvelles hospitalisations',image('_prevision_nouv hospitalisations.png')),
                       ('hospitalisations par âge',image('_prevision_hospi 09_hospi 19_hospi 29_hospi 39_hospi 49_hospi 59_hospi 69_hospi 79_hospi 89_hospi 90.png')),
                       ('nouvelles réanimations',image('_prevision_nouv réanimations.png')),
                       ('nouveaux décès',image('_prevision_nouv décès.png'))]),
                 tabs([('R',image('_prevision_R_par_urgences_hospi.png')),
                       ('taux positifs',image('_prevision_taux positifs.png')),
                       ('positifs',image('_prevision_positifs.png')),
                       ('taux positifs par âge',image('_prevision_taux positifs 09_taux positifs 19_taux positifs 29_taux positifs 39_taux positifs 49_taux positifs 59_taux positifs 69_taux positifs 79_taux positifs 89_taux positifs 90.png')),
                       ('positifs par âge',image('_prevision_positifs 09_positifs 19_positifs 29_positifs 39_positifs 49_positifs 59_positifs 69_positifs 79_positifs 89_positifs 90.png'))])]]))

f.write("<a id=\"previsions3mois\"></a>"
        + vspace + "<h4>Données et prévisions jour par jour depuis 3 mois</h4>"
        + "<p>Courbe bleue: données réelles.<br>"
        + "Courbe orange: données prévues à partir du jour \(j\) "
        + "en utilisant uniquement les données des jours précédant \( j\).</p>")

f.write(table2([[tabs([('urgences',video('previsions_urgences.mp4')),
                       #('sosmédecin',video('previsions_sosmedecin.mp4')),
                       ('hospitalisations urgences',video('previsions_hospitalisation urgences.mp4')),
                       ('nouvelles hospitalisations',video('previsions_nouv hospitalisations.mp4'))]),
                 tabs([('réanimations',video('previsions_réanimations.mp4')),
                       ('nouvelles réanimations',video('previsions_nouv réanimations.mp4')),
                       ('hospitalisations',video('previsions_hospitalisations.mp4')),
                       ]),
                 tabs([('taux positifs',video('previsions_taux positifs.mp4')),
                       ('positifs',video('previsions_positifs.mp4')),
                       ('nouv décès',video('previsions_nouv décès.mp4'))])]]))

def dernierjour(x):
    return(jour_de_num[jours[0] + intervalle[ni(x)][1]-1])

f.write('<p>Les indicateurs de la covid19 utilisés sont des données quotidiennes (courbes en traits gras ci-dessus), dont:'
        + '<ul><li>urgences: nombre de passages aux urgences pour suspicion de COVID-19 - Quotidien'
        + " (jusqu'au " + dernierjour('urgences') + ")"
        + '<li>hospitalisations: nombre quotidien de nouvelles personnes hospitalisées avec diagnostic COVID-19 déclarées en 24h'
        + " (jusqu'au " + dernierjour('nouv hospitalisations') + ")"
        + '<li>réanimations: nombre quotidien de nouvelles admissions en réanimation (SR/SI/SC) avec diagnostic COVID-19 déclarées en 24h'
        + " (jusqu'au " + dernierjour('nouv réanimations') + ")"
        + '<li>décès: nombre quotidien de nouveaux décès avec diagnostic COVID-19 déclarés en 24h'
        + " (jusqu'au " + dernierjour('nouv décès') + ")"
        + '<li>tests positifs: nombre quotidien de nouveaux tests positifs au COVID-19 en 24h'
        + " (jusqu'au " + dernierjour('positifs') + ")"
        + "<li>patients en réanimation, hospitalisés, par tranches d'âge"
        + " (jusqu'au " + dernierjour('réanimations') + ")"
        + "<li>personnes testées positives, taux de tests positifs, par tranches d'âge"
        + " (jusqu'au " + dernierjour('taux positifs') + ")"
        + "</ul></p>"
        + "<p>Les données concernent chaque département (ou région) et chaque jour depuis mars 2020, et proviennent de <a href=\"https://www.data.gouv.fr/fr/pages/donnees-coronavirus\">www.data.gouv.fr</a></p>"
)
f.write("<p>Les prévisions (courbes en tirets ci-dessus) sont calculées à partir</p>"
        + "<ul><li>des données de <a href=\"https://www.google.com/covid19/mobility/\">mobilité de Google</a>"
        + " (jusqu'au " + dernierjour('travail') + ")"
        + "<li>des données de <a href=\"https://covid19.apple.com/mobility\">mobilité d'Apple</a>"
        + " (jusqu'au " + dernierjour('en transport en commun') + ")"
        + "<li>des données <a href=\"https://www.data.gouv.fr/fr/datasets/donnees-d-observation-des-principales-stations-meteorologiques/\">météo</a>"
        + " (jusqu'au " + dernierjour('température') + ")"
        + "<li>des dates des vacances scolaires"
        + "<li>de données de recherche de mots sur Google (<a href=\"https://trends.google.fr/trends/explore?geo=FR&q=voyage\">Google Trends</a>)"
        + " (jusqu'au " + dernierjour('recherche voyage google') + ")"
        + "<li>des dates de confinement et de couvre-feu"
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
            "<p>Les données concernent les indicateurs de l'épidémie (urgences, réanimations, décès, tests positifs, etc) et les contextes (données météo: température, pression, données de mobilité fournies par google: fréquentation des commerces et lieux de loisir, des lieux de travail, etc).<br>"
            + "Elles se présentent comme un tableau à 3 dimensions: une pour les noms des données, une pour les départements français, une pour les jours. Par exemple, on dispose du nombre de passage aux urgences hospitalières, du nombre de patients en réanimation, etc, pour soupçon de covid chaque jour depuis avril 2020, pour chaque département.<br>"
            + "En tout "
            + str(nnoms) + " jeux de données pour "
            + str(ndeps) + " départements et "
            + str(njours) + " jours sont utilisés.<br>"
            + "On commence par calculer les coefficients de corrélation entre deux jeux de données, pour tous les décalages temporels d'au plus 40 jours. Et on retient le décalage qui maximise la valeur absolue du coefficient de corrélation entre deux données, si celle-ci est supérieure à "
            + ("%1.2f" % mincorrelation) + ": par exemple, ce coefficient de corrélation est maximal entre le travail et les urgences si on décale les urgences "
            + str(int(coefs[ni('travail'),ni('urgences')][0]))
            +  " jours dans le passé.<br>"
            +  "Cela suggère une causalité possible: une hausse de la fréquentation des lieux de travail peut provoquer une hausse des urgences pour covid "
            + str(int(coefs[ni('travail'),ni('urgences')][0]))
            +  " jours plus tard.<br>"
            + "A présent on dispose de décalages de jours entre certaines des données. On dira qu'une donnée \(d\) dépend d'une donnée \(d'\) si on a obtenu, à l'étape précédente, un décalage \(\Delta_{d' d} > 0\) de \(d'\) à \(d\).<br>"
            + "On considère alors qu'une donnée \(d\) sur une période de temps \([j_0,j_1]\) va dépendre des valeurs des donnéees \(d_i\) dont elle dépend sur les périodes \([j_0 - \Delta_{d' d},j_1 - \Delta_{d' d}]\).<br>"
            + "Appelons \(A\) la matrice dont la colonne \(i\) est formée des valeurs de \(d_i\) pour tous les départements sur la période \([j_0 - \Delta_{d' d},j_1 - \Delta_{d' d}]\), et \(B\) le vecteur colonne formé des valeurs de \(d\) pour tous les départements sur la période \([j_0,j_1]\).<br>",
            "On aimerait trouver une famille de coefficients \(C = (c_i)\) telle que \(AC = B\). Mais il n'y a pas en général de solution à cette équation, car \(A\) a plus de lignes que de colonnes. On cherche alors à minimiser \(||AC-B||\), c'est un problème convexe, dont la solution s'obtient avec \[C = (^tAA)^{-1}(^tAB)\] (si \(^tAA\) est inversible, ce qui est le cas en pratique).<br>"
            + "Avec \(C\) on peut prévoir une valeur pour la donnée \(d\) le jour \(j_1+1\), simplement en calculant \(A_1C\), où \(A_1\) est obtenue comme \(A\) mais avec les intervalles \([j_0 - \Delta_{d' d}+1,j_1 - \Delta_{d' d}+1]\). Et ainsi de suite.<br>"
            + "On prévoit alors toutes les données d'un jour en parallèle, puis le suivant, etc. Si une donnée n'est pas prévisible (car pas de dépendance, ou  \(^tAA\) non  inversible), on garde sa valeur précédente.<br>"
            + "Évidemment, si une prévision dépend de données déjà obtenues par prévision, on imagine qu'elle est moins fiable.</p>"
            + "<p><h5>Important</h5>Les données qui concernent les indicateurs de l'épidémie sont utilisées sous la forme de leur dérivée discrète \(d(j+1)-d(j)\). Comme ce sont déjà des valeurs par jour, donc des dérivées, on obtient alors la dérivée seconde des données cumulées. On calcule alors des corrélations entre les données de contexte, que l'on peut voir comme des forces, et les dérivées secondes des indicateurs absolus de l'épidémie.<br>"
            + "Cette analogie linéaire entre force et accélération empruntée à la physique newtonienne s'avère féconde: en effet les corrélations sont négligeables si on ne considère pas les dérivées discrètes des indicateurs journaliers, alors qu'au contraire elles apparaissent avec celles-ci.<br>"
            + "La contre-partie est qu'on obtient une prévision des variations des indicateurs, mais pour obtenir leur valeur, il suffit d'intégrer (i.e. cumuler)."
            + "</p>"]]))


f.write("<a id=\"correlations\"></a>"
        + vspace + '<h3>Corrélations</h3>'
        "<p>Attention: corrélation ne veut pas dire causalité, mais bon, cela peut donner des idées.</p>")

f.write(table2([[tabs([('contexte-indicateur',image('_heat_correlation_contexte_indicateur.png')),
                       ('confinement,couvre-feu',
                        image('_heat_correlation_confinement_indicateur.png')),
                       ('complètes',image('_heat_correlation_tout.png'))])]]))

f.write('<a id="coefficients"></a>'
        + vspace + "<h3>Coefficients d'optimisation</h3>"
        "<p>Dépendances et coefficients d'optimisation de différents indicateurs, ainsi que quelques contextes influents.</p>")

f.write(table2([[tabs([('coef. ' + x,image('_coefficients_' + x + '.png'))
                       for x in ['urgences','réanimations','hospitalisations',
                                 'nouv décès','taux positifs']]
                      + [('contextes influents',image('_contextes_influents.png'))]),
                 tabs([('recherche Google',image('_recherche google.png'))])]]))

f.write('<a id="evaluation"></a>'
        + vspace + "<h3>Évaluation de la méthode de prévision</h3>"
        + "<p>On compare les prévisions avec une prévision linéaire par la tangente, et une prévision par approximation à l'ordre 2, sur 90 jours<br>"
        + "Les prévisions à partir du jour \(j\) n'utilisent que les données antérieures au jour \(j\).<br>"
        + "En vert, la meilleure des 3 méthodes, pour l'erreur relative médiane, en orange la seconde et en rouge la dernière. Rien n'est indiqué si les 3 méthodes ont des erreurs supérieures à 50%.</p>")

f1 = open(DIRSYNTHESE + '_evaluation.html', 'r')
f.write(f1.read())
f1.close()

heat_map(nomscont + nomsind, nomsind + nomscont,'_heat_correlation_tout',
         fontcase = 4,
         dim = (10,14))
heat_map(nomscont, nomsind,'_heat_correlation_contexte_indicateur',
         dim=(10,10))
if inclusconfinement:
    heat_map(dataconfinement['confinement'], nomsind,
             '_heat_correlation_confinement_indicateur',
             dim = (10,4))

f.write('<a id="alpesmaritimes"></a>'
        + vspace + "<h3>Dans les Alpes-Maritimes</h5>")

f.write(table2([[tabs([('urgences',image('_prevision_urgences6.png'))]),
                 tabs([('réanimations',image('_prevision_réanimations6.png')),
                       #('sosmédecin',image('_prevision_sosmedecin6.png'))
                 ]),
                 tabs([('taux positifs',image('_prevision_taux positifs6.png')),
                       ('R',image('_prevision_R_par_urgences_hospi 06.png'))])]]))


f.write(fin1)
f.close()

os.system('scp _synthese.html lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19/_synthese.html')
os.system('tar -cvf groba.tgz ' + DIRSYNTHESE + '/*.png ' + DIRSYNTHESE + '/*.mp4')

os.system('scp groba.tgz lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19')
os.system("ssh lpmib@ssh-lpmib.alwaysdata.net 'cd testdjango2/testdjango2/medias/covid19 && tar -xvf groba.tgz'")


