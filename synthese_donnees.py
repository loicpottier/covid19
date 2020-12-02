import time
import os
now = time.localtime(time.time())
######################################################################
# page de synthèse
# mets la synthèse sur cp.lpmib.fr:
# https://cp.lpmib.fr/medias/covid19/_synthese.html
######################################################################

from pytz import reference

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
            + '<img src="donnees/' + x + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">' + '</td>\n')
    f.write('<td valign = top> '
            + '<img src="donnees/' + y + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">' + '</td>\n')
    f.write('</tr>\n')
def images1(f,x):
    f.write('<tr valign=top>')
    f.write('<td valign = top> '
            + '<img src="donnees/' + x + '"  class="img-fluid" alt="Responsive image">' + '</td>\n')
    f.write('<td valign = top> </td>\n')
    f.write('</tr>\n')

######################################################################
synthese = 'donnees/_synthese.html'
f = open(synthese, 'w', encoding = 'utf8')

f.write(debut1)
f.write('<div class="container-fluid">'
        + '<h2> Indicateurs covid 19 et prévisions </h2>'
        + '<p>' + time.asctime(now) + '</p>'
        + '<p>loic.pottier@gmail.com'
        + '<a href="https://twitter.com/LocPottier1"><img src="https://help.twitter.com/content/dam/help-twitter/brand/logo.png" width = 50></a></p>')

f.write('<h3> Prévisions</h3>'
        + '<p>Données quotidiennes:'
        + '<ul><li>urgences: nombre de passages aux urgences pour suspicion de COVID-19 - Quotidien'
        + '<li>hospitalisations: nombre quotidien de nouvelles personnes hospitalisées avec diagnostic COVID-19 déclarées en 24h'
        + '<li>réanimations: nombre quotidien de nouvelles admissions en réanimation (SR/SI/SC) avec diagnostic COVID-19 déclarées en 24h'
        + '<li>décès: nombre quotidien de nouveaux décès avec diagnostic COVID-19 déclarés en 24h'
        + '</ul></p>')
f.write('<p>Les prévisions sont calculées à partir des données de mobilité de Google, des données météo, et des dates des vacances. Voir plus bas pour des précisions.</p>')
f.write('<table class="table">')
images2(f,'_prevision_reatot_par_mobilite.png','_prevision_par_mobilite.png')
images2(f,'_prevision_positifs.png','_prevision_positifs_ages.png')
f.write('</table>')

ff = open('donnees/_rapport','r')
rapport = ff.read()
ff.close()

f.write('<div>'
        + '<h4>Détail des prévisions.</h4>'
        + "<p>Chaque donnée du contexte (fréquentation des lieux de rencontre sociale, météo, vacances) est affectée d'un coefficient qui détermine son influence sur l'indicateur de l'épidémie examiné (par exemple, le nombre de patients en réanimation). La méthode permettant de calculer ces coefficients est expliquée plus bas.<p>"
        + "<p>Dans ce tableau, les valeurs en gras sont celles dont l'intervalle de confiance ne contient pas strictement 0: le signe du coefficient est donc déterminé.<br>"
        + "S'il est positif, le contexte favorise l'indicateur de l'épidémie, s'il est négatif, il le freine. Plus le coefficient est grand en valeur absolue, plus cet effet est fort.</p>"
        + rapport + '</div>'
)

f.write('<table class="table">')
images2(f,'_dispersion_contexte.png','_contextes_influents.png')
f.write('</table>')

f.write("<h4>Prévisions des cas positifs par tranche d'âge</h4>"
        + "<p>Par tranches de 10 ans, par exemple 29 signifie 20-29 ans.</p>")
ff = open('donnees/_rapportpos','r')
rapportpos = ff.read()
ff.close()

f.write(rapportpos)

f.write('<h3>Méthodes</h3><br>')
f.write('<h5>Prévision.</h5>'
        + '<div>'
        + '<p>Les données de la covid19 proviennent de <a href= "https://geodes.santepubliquefrance.fr"> geodes.santepubliquefrance.fr</a> et de <a href= "https://www.data.gouv.fr/fr/datasets"> www.data.gouv.fr</a>, lissées sur 7 jours. '
        + 'Les données de mobilité proviennent de <a href="https://www.google.com/covid19/mobility/"> Google,</a>'
        + ' les données météorologiques proviennent de  <a href="https://www.data.gouv.fr/fr/datasets/donnees-d-observation-des-principales-stations-meteorologiques"> www.data.gouv.fr</a></p>'
         + "<p>On utilise les données de mobilité et météo pour prévoir les valeurs des principaux indicateurs de l'épidémie dans un proche avenir (de 5 à 20 jours). "
        + "La prédiction est linéaire, par optimisation quadratique et convolution.</p>"
        + "<p>Par exemple, pour un département, on dispose des données de mobilité des habitants chaque jour pour des types de lieux: "
        + "commerces et loisirs, parcs et nature, travail, résidence, etc. On dispose aussi des données de température et de pression atmosphérique. "
        + "Ces données sont mises dans une matrice A, dont les lignes sont les jours, et les colonnes sont les données de mobilité et météo (qu'on appellera \"contextes\").</p>"
        + "<p>Supposons qu'on veuille prévoir les passages aux urgences pour soupçon de covid19 chaque jour. "
        + "On connaît ces données jusqu'au présent. Il se trouve que la méthode fonctionne mieux avec la dérivée discrète des données à prévoir: on détermine alors pour chaque jour la différence du nombre de passage par rapport à la veille, et on stocke cette valeur pour chaque jour dans une matrice colonne B dont les lignes sont les jours.</p>"
        + "<p>Ce sont ces données que l'on cherche à prévoir, en fonction des données de contexte, en faisant l'hypothèse qu'il y a un intervalle de temps entre le moment où le contexte change et celui où cela se répercute sur l'épidémie visible: c'est le temps d'apparition des symptômes et de leur aggravation, pour le passage aux urgences. "
        + "On cherche alors un jeu de coefficients C, i.e. une matrice colonne avec autant de lignes que de contextes, telle que AC = B.</p>"
        + "<p>Comme le nombre de lignes de A (les jours) est bien supérieur au nombre de colonnes, il n'y en a généralement pas, alors on cherche à minimiser l'erreur, i.e. la norme de AC - B. "
        + "C'est un problème d'optimisation convexe, et la solution est celle où la différentielle est nulle, on l'obtient par C = (tA A)^-1 (tA B)</p>"
        + "<p>Comme on ne connaît pas le décalage de jours entre A et B, on teste tous les décalages (convolution) et on retient celui avec une erreur qui est le premier minimum local (par décalage croissant), le décalage étant le même pour tous les départements. "
        + "Puis on moyenne tous les C des départements pour obtenir un jeu de coefficients de prévision moyen. "
        + "C'est celui qui est donné dans le tableau ci-dessus.</p>"
        + "<p>Pour prévoir, par exemple les urgences, qui ont un décalage de 14 jours, il suffit de calculer AC pour les 14 jours du futur de B (avec les C de chaque département), à partir des 14 derniers jours de A, puis de cumuler AC à partir du dernier jour connu (le présent). "
        + "Ce sont les courbes en gras à la fin des courbes dans le premier graphique ci-dessus."
        + '</div>'
        + '</p>'
        + '<h5>Calcul de r0.</h5>'
        + '<div>'
        + "<p>Pour le calcul de r0 (ou R, ou encore \"taux de reproduction\"):<br>"
        + '<ul><li>r0 est déterminé avec un intervalle sériel s = 4,11:'
        + "<ul><li>r0 est le nombre moyen de personnes que contamine un malade (en dessous de 1, l'épidémie régresse et s'arrête rapidement),"
        + "<li>l'intervalle sériel est le nombre moyen de jours entre deux contaminés successifs dans une chaîne de contamination, il vaut 4 jours environ: A contamine B 4 jours après avoir été contaminé,"
        + "<li> si f est la fonction exponentielle étudiée, r0 = exp(s*f'/f), de sorte que f(x+s) = r0*f(x)"
        + '<li> les dérivées sont calculées sur 7 jours'
        + '</ul>'
        + "<li> les données de visite aux urgences, d'hospitalisation, de patients en réanimation, de décès et de tests sont des valeurs sur une journée."
        + '</ul></p>'
        + '</div>')

f.write('<h3> Données</h3>')
f.write('<table class="table">')
images2(f,'_r0_france.png','_indicateurs_covid_france.png')
images2(f,'_nouveaux_cas_france.png','_taux_pos_france.png')
images2(f,'_positifs_france.png','_total_positifs_france.png')
images2(f,'_r0_06.png','_indicateurs_covid_06.png')
f.write('</table><br>')
f.write(fin1)
f.close()

limages = ['_synthese.html',
           '_r0_france.png',
           '_indicateurs_covid_france.png',
           '_derivee_seconde_indicateurs_covid_france.png',
           '_r0_06.png',
           '_indicateurs_covid_06.png',
           '_derivee_seconde_indicateurs_covid_06.png',
           '_taux_pos_france.png',
           '_taux_pos_06.png',
           '_positifs_france.png',
           '_positifs_france_0-19.png',
           '_prevision_par_mobilite.png',
           '_mobilite_google.png',
           '_prevision06_par_mobilite.png',
           '_mobilite06_google.png',
           '_nouveaux_cas_france.png',
           '_total_positifs_france.png',
           '_dispersion_contexte.png',
           '_prevision_positifs.png',
           '_prevision_positifs_ages.png',
           '_prevision_reatot_par_mobilite.png',
           '_contextes_influents.png',
]
os.system('scp donnees/_synthese.html lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19')
os.system('tar -cvf groba.tgz '
          + ' '.join(['donnees/' + x for x in limages]))

os.system('scp groba.tgz lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19')
os.system("ssh lpmib@ssh-lpmib.alwaysdata.net 'cd testdjango2/testdjango2/medias/covid19 && tar -xvf groba.tgz'")
