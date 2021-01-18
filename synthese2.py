from correlation import *
import time
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytz import reference
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

now = time.localtime(time.time())
######################################################################
# page de synthèse
# mets la synthèse sur cp.lpmib.fr:
# https://cp.lpmib.fr/medias/covid19/_synthese2.html
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
            + '<img src="' + DIRSYNTHESE2 + '' + x + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">' + '</td>\n')
    f.write('<td valign = top> '
            + '<img src="' + DIRSYNTHESE2 + '' + y + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">' + '</td>\n')
    f.write('</tr>\n')
    
def images(f,lim):
    f.write('<tr valign=top>')
    for im in lim:
        f.write('<td valign = top> '
                + '<img src="' + DIRSYNTHESE2 + '' + im
                + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">'
                + '</td>\n')
    f.write('</tr>\n')
    
def image_texte(f,x,y):
    f.write('<tr valign=top>')
    f.write('<td valign = top> '
            + '<img src="' + DIRSYNTHESE2 + '' + x + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">' + '</td>\n')
    f.write('<td valign = top> '
            + '<div class="container-fluid"><p>' + y + '</p></div></td>\n')
    f.write('</tr>\n')

def images1(f,x):
    f.write('<tr valign=top>')
    f.write('<td valign = top> '
            + '<img src="' + DIRSYNTHESE2 + x + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">' + '</td>\n')
    f.write('<td valign = top> </td>\n')
    f.write('</tr>\n')

def videos(f,lv):
    f.write('<tr valign=top>')
    for vid in lv:
        f.write('<td valign = top>')
        f.write("<div class=\"embed-responsive embed-responsive-4by3\">")
        f.write("<iframe class=\"embed-responsive-item\" src=\"" + DIRSYNTHESE2
                + vid + "\" allowfullscreen></iframe>")
        f.write("</div>")
        f.write('</td>')
    f.write('</tr>')

######################################################################
synthese = '_synthese2.html'
f = open(synthese, 'w', encoding = 'utf8')

f.write(debut1)
f.write('<div class="container-fluid">'
        + '<h2> Prévisions des indicateurs de la covid 19.</h2>'
        + '<p>' + time.asctime(now) + '</p>'
        + '<p>loic.pottier@gmail.com'
        + '<a href="https://twitter.com/LocPottier1"><img src="https://help.twitter.com/content/dam/help-twitter/brand/logo.png" width = 50></a></p>'
        + '<p><a href="https://github.com/loicpottier/covid19">Code</a><br>')

f.write("<i>(Méthode et code refondus complètement le 14 janvier)</i></p>")
f.write("<h3>Données et prévisions des indicateurs principaux de l'épidémie</h3>")
f.write("<p>On suppose qu'après les vacances de février, les collèges et lycées fonctionnent encore en demi-effectif.</p>"
        +"<p>La méthode employée est fondée sur les corrélations maximales entre données dont on déduit les décalages temporels, et, à partir de ces décalages, sur la prévision linéaire par optimisation quadratique.<br>"
        + "Tout cela département par département. Les données méteo à venir sont déduites de l'année passée. On n'utilise aucune modélisation. Détails à venir.</p>")
f.write("<h4>Données et prévisions jour par jour depuis 3 mois</h4>"
        + "<p>Courbe bleue: données réelles. Courbe orange: données prévues à partir du jour \(j\) en utilisant uniquement les données des jours précédant \( j\).</p>" )

f.write('<table class="table">')
videos(f,['previsions_urgences.mp4',
          'previsions_réanimations.mp4',
          'previsions_hospitalisations.mp4'])
videos(f,['previsions_taux positifs.mp4','previsions_positifs.mp4','previsions_sosmedecin.mp4'])
f.write('</table>')

f.write("<h4>Prévisions aujourd'hui</h4>")
f.write('<table class="table">')
images(f,['_prevision_urgences_nouv hospitalisations_sosmedecin.png',
          '_prevision_réanimations.png',
          '_prevision_hospitalisations.png'])

f.write('</table>')

f.write('<table class="table">')
images(f,['_prevision_R_par_urgences_hospi.png',
           '_prevision_nouv réanimations_nouv décès.png',
           '_prevision_hospi 09_hospi 19_hospi 29_hospi 39_hospi 49_hospi 59_hospi 69_hospi 79_hospi 89_hospi 90.png'])
f.write('</table>')

f.write("<p>Comme pour les prévisions météo, on se fonde sur des données objectives, sur lequelles on applique uniquement des méthodes mathématiques élémentaires (niveau L2). En particulier, on n'utilise pas de modélisation.<br>"
        + "Il s'agit ici de dérivations, corrélations, convolutions et moindres carrés. Mais contrairement à la météo, aucune modification <i>a posteriori</i> n'est apportée: les résultats sont bruts.</p>" 
        + '<p>Les indicateurs de la covid19 utilisés sont des données quotidiennes (courbes en traits gras ci-dessous):'
        + '<ul><li>urgences: nombre de passages aux urgences pour suspicion de COVID-19 - Quotidien'
        + '<li>hospitalisations: nombre quotidien de nouvelles personnes hospitalisées avec diagnostic COVID-19 déclarées en 24h'
        + '<li>réanimations: nombre quotidien de nouvelles admissions en réanimation (SR/SI/SC) avec diagnostic COVID-19 déclarées en 24h'
        + '<li>décès: nombre quotidien de nouveaux décès avec diagnostic COVID-19 déclarés en 24h'
        + '<li>tests positifs: nombre quotidien de nouveaux tests positifs au COVID-19 en 24h'
        + "<li>patients en réanimation, hospitalisés, par tranches d'âge"
        + "<li>personnes testées positives, taux de tests positifs, par tranches d'âge"
        + "</ul></p>"
        + "<p>Les données concernent chaque département (ou région) et chaque jour depuis mars 2020, et proviennent de <a href=\"https://www.data.gouv.fr/fr/pages/donnees-coronavirus\">www.data.gouv.fr</a></p>"
)
f.write("<p>Les prévisions (courbes en tirets ci-dessous) sont calculées à partir</p>"
        + "<ul><li>des données de <a href=\"https://www.google.com/covid19/mobility/\">mobilité de Google</a>"
        + "<li>des données de <a href=\"https://covid19.apple.com/mobility\">mobilité d'Apple</a>"
        + "<li>des données <a href=\"https://www.data.gouv.fr/fr/datasets/donnees-d-observation-des-principales-stations-meteorologiques/\">météo</a>"
        + "<li>des dates des vacances scolaires"
        + "<li>de données de recherche de mots sur Google (<a href=\"https://trends.google.fr/trends/explore?geo=FR&q=voyage\">Google Trends</a>)"
        + "<li>sans déconfinement le 20 janvier 2021"
        + "</ul></p>"
        + "Voir plus bas pour des précisions sur la méthode employée.</p>")

######################################################################
# corrélations

voyage_google = '''<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/2431_RC04/embed_loader.js"></script>
  <script type="text/javascript">
    trends.embed.renderExploreWidget("TIMESERIES", {"comparisonItem":[{"keyword":"voyage","geo":"FR","time":"today 12-m"}],"category":0,"property":""}, {"exploreQuery":"geo=FR&q=voyage&date=today 12-m","guestPath":"https://trends.google.fr:443/trends/embed/"});
  </script>'''

f.write('<div>'
        + '<h4>Détail des prévisions.</h4>'
        + "<p>Chaque donnée du contexte (fréquentation des lieux de rencontre sociale, météo, vacances) est affectée d'un coefficient qui détermine son influence sur la valeur de l'indicateur de l'épidémie examiné (par exemple, le nombre de patients en réanimation). La méthode permettant de calculer ces coefficients est expliquée plus bas.<p>"
        + '</div>'
)

f.write('<h3>Méthodes</h3><br>')

f.write('<h3> Prévisions diverses</h3>')
f.write("<h4>Prévisions des cas positifs par tranche d'âge</h4>"
        + "<p>Résultats peu concluants: les corrélations sont très faibles et les décalages indétectables.</p>")
f.write('<table class="table">')
images2(f,'_prevision_positifs.png','_prevision_positifs 09_positifs 19_positifs 29_positifs 39_positifs 49_positifs 59_positifs 69_positifs 79_positifs 89_positifs 90.png')
images2(f,'_prevision_taux positifs.png','_prevision_taux positifs 09_taux positifs 19_taux positifs 29_taux positifs 39_taux positifs 49_taux positifs 59_taux positifs 69_taux positifs 79_taux positifs 89_taux positifs 90.png')
f.write('</table>')

f.write("<h4>Dans les Alpes-Maritimes</h4>")
f.write('<table class="table">')
images(f,['_prevision_urgences_réanimations_sosmedecin6.png',
          '_prevision_taux positifs6.png',
          '_prevision_R_par_urgences_hospi 06.png'])
f.write('</table>')

f.write(fin1)
f.close()

os.system('scp _synthese2.html lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19/_synthese.html')
os.system('tar -cvf groba.tgz ' + DIRSYNTHESE2 + '/*.png ' + DIRSYNTHESE2 + '/*.mp4')

os.system('scp groba.tgz lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19')
os.system("ssh lpmib@ssh-lpmib.alwaysdata.net 'cd testdjango2/testdjango2/medias/covid19 && tar -xvf groba.tgz'")


