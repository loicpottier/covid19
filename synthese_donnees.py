from _correlations import noms_indicateurs, noms_contextes, coefs
from outils import *
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
            + '<img src="donnees/' + x + '"  class="img-fluid rounded border border-secondary" alt="Responsive image">' + '</td>\n')
    #f.write('<td valign = top> </td>\n')
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
        + '<li>tests positifs: nombre quotidien de nouveaux tests positifs au COVID-19 en 24h'
        + '</ul></p>')
f.write("<p>Les prévisions (courbes en gras ci-dessous) sont calculées à partir des données de mobilité de Google et d'Apple, des données météo, et des dates des vacances. Voir plus bas pour des précisions.</p>")
f.write('<table class="table">')
images2(f,'_prevision_reatot_par_mobilite.png','_prevision_par_mobilite.png')
images2(f,'_prevision_positifs.png','_prevision_positifs_ages.png')
images2(f,'_prevision_hospi_ages.png','_R.png')
images2(f,'_prevision_urge_date_depart.png','_prevision_reatot_date_depart.png')
f.write('</table>')

######################################################################
# corrélations
f.write('<h3>Corrélations</h3>'
        + "<p>Le premier tableau détaille les corrélations entre données du contexte (fréquentation des lieux de rencontre sociale, météo, vacances) et indicateurs de la covid19 (urgences, réanimations, etc). En rouge et orange sont indiquées les corrélations positives: ce sont les contextes qui favorisent l'épidémie quand ils augmentent, en vert ce sont ceux qui freinent l'épidémie. Les âges sont donnés par tranches de 10 ans, par exemple 29 signifie 20-29 ans).</p>"
        + "<p>Le second donne les décalages entre un contexte et son effet sur chaque indicateur de l'épidémie.</p>")

f.write('<table class="table">')
images2(f,'_heatmap_correlations.png','_heatmap_decalages.png')
f.write('</table>')

#noms_indicateurs = noms_indicateurs[:5] + sorted(noms_indicateurs[5:],
#                                                 key = lambda x: int(x[-2:]))
noms_indicateurs = [x for x in noms_indicateurs 
                    if x not in ['décès total']]

table_correlation = []
table_correlation.append(['corrélations <em>(décalage en jours)</em>' ]
                         + [x.replace('hospitalisations','hospi.').replace('réanimations','réa.').replace('hospitalisation','hospi.').replace('réanimation','réa.')
                            if '9' not in x[-2:] else 'hos. ' + x[-2:]
                            for x in noms_indicateurs])
for cont in noms_contextes:
    ligne = [cont]
    for ind in noms_indicateurs:
        [corr,_,_,_,_],dec = coefs[ind][cont]
        if dec != 0 and abs(corr) > 0.02:
            if corr > 0:
                if abs(corr) > 0.3:
                    tm = ('<p class="font-weight-bold text-danger">'
                          + ('%d' % int(100*corr)) + '<em> (' + str(dec) + '' + ')</em></p>')
                else:
                    tm = ('<p class="font-weight-bold text-warning">'
                          + ('%d' % int(100*corr)) + '<em> (' + str(dec) + '' + ')</em></p>')
            else:
                tm = ('<p class="font-weight-bold text-success">'
                      + ('%d' % int(100*corr)) + '<em> (' + str(dec) + '' + ')</em></p>')
        else:
            tm = '<p class="text-muted">'+ '-' + '</p>'
        ligne.append(tm)
    table_correlation.append(ligne)

table_correlation = table(table_correlation)

#f.write("<p> Entre parenthèses est donné le décalage en jours entre un contexte et son effet sur l'indicateur concerné (c'est le décalage qui donne une corrélation maximale). Les corrélations trop faibles ne sont pas indiquées.</p>")

#f.write(table_correlation)

######################################################################
# heatmaps
# correlations
tcorr = []
for cont in noms_contextes:
    ligne = []
    for ind in noms_indicateurs:
        [corr,_,_,_,_],dec = coefs[ind][cont]
        ligne.append([cont,ind,int(100*corr)])
    tcorr.append(ligne)

tcorr = sorted(tcorr,key = lambda x: -max([y[2] for y in x]))
noms_contextes1 = [x[:20] for x in [x[0][0] for x in tcorr]]
noms_indicateurs1 = [x[:18] for x in noms_indicateurs]
tcorr = [[y[2] for y in x] for x in tcorr]
tcorr = np.array(tcorr)
fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(-tcorr,cmap="RdYlGn")
ax.set_xticks(np.arange(len(noms_indicateurs1)))
ax.set_yticks(np.arange(len(noms_contextes1)))
ax.set_xticklabels(noms_indicateurs1)
ax.set_yticklabels(noms_contextes1)
ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)
plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
         rotation_mode="anchor")
for i in range(len(noms_contextes1)):
    for j in range(len(noms_indicateurs1)):
        text = ax.text(j, i, tcorr[i, j],
                       ha="center", va="center", color="w")

ax.set_title("corrélations contextes/indicateurs",y = -0.1)
fig.tight_layout()
plt.savefig('donnees/_heatmap_correlations.png',dpi = 150)
plt.savefig('donnees/_heatmap_correlations.pdf',dpi = 150)
plt.show(False)

######################################################################
# decalages
tdeccorr = []
for cont in noms_contextes:
    ligne = []
    for ind in noms_indicateurs:
        [corr,_,_,_,_],dec = coefs[ind][cont]
        ligne.append([cont,ind,dec,corr])
    tdeccorr.append(ligne)


# indicateurs tries par décalage moyen croissant
indictries = sorted(noms_indicateurs,
                    key = lambda ind: np.mean(np.array([[y[2] for y in x if y[1] == ind][0]
                                                        for x in tdeccorr])))

tdeccorr = [sorted(cont,
                   key = lambda x: indictries.index(x[1]))
            for cont in tdeccorr]

tdeccorr = sorted(tdeccorr,
                  key = lambda cont: min([x[2] for x in cont]))

noms_contextes1 = [x[:20] for x in [x[0][0] for x in tdeccorr]]
noms_indicateurs1 = [x[:18] for x in indictries]
tdec = [[y[2] for y in x] for x in tdeccorr]
tdec = np.array(tdec)
tcorr = [[y[3] for y in x] for x in tdeccorr]
tcorr = np.array(tcorr)
fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(-tcorr,cmap="RdYlGn")
ax.set_xticks(np.arange(len(noms_indicateurs1)))
ax.set_yticks(np.arange(len(noms_contextes1)))
ax.set_xticklabels(noms_indicateurs1)
ax.set_yticklabels(noms_contextes1)
ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)
plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
         rotation_mode="anchor")
for i in range(len(noms_contextes1)):
    for j in range(len(noms_indicateurs1)):
        text = ax.text(j, i, tdec[i, j],
                       ha="center", va="center", color="w")

ax.set_title("décalages contextes/indicateurs",y = -0.1)
fig.tight_layout()
plt.savefig('donnees/_heatmap_decalages.png',dpi = 150)
plt.savefig('donnees/_heatmap_decalages.pdf',dpi = 150)
plt.show(False)

####

f.write('<div>'
        + '<h4>Détail des prévisions.</h4>'
        + "<p>Chaque donnée du contexte (fréquentation des lieux de rencontre sociale, météo, vacances) est affectée d'un coefficient qui détermine son influence sur la valeur de l'indicateur de l'épidémie examiné (par exemple, le nombre de patients en réanimation). La méthode permettant de calculer ces coefficients est expliquée plus bas.<p>"
        + '</div>'
)

f.write('<table class="table">')
images2(f,'_dispersion_contexte.png','_contextes_influents.png')
f.write('</table>')

f.write('<h3>Méthodes</h3><br>')
f.write('<h5>Prévision.</h5>'
        + '<div>'
        + '<div class="container"><div class="row"><div class="col-sm">'
        + '<p>Les données de la covid19 proviennent de <a href= "https://geodes.santepubliquefrance.fr"> geodes.santepubliquefrance.fr</a> et de <a href= "https://www.data.gouv.fr/fr/datasets"> www.data.gouv.fr</a>, lissées sur 7 jours. '
        + "Les données de mobilité proviennent de <a href=\"https://www.google.com/covid19/mobility/\"> Google</a> et <a href=\"https://covid19.apple.com/mobility\"> d'Apple</a>,"
        + ' les données météorologiques proviennent de  <a href="https://www.data.gouv.fr/fr/datasets/donnees-d-observation-des-principales-stations-meteorologiques"> www.data.gouv.fr</a></p>'
         + "<p>On utilise les données de mobilité et météo pour prévoir les valeurs des principaux indicateurs de l'épidémie dans un proche avenir (de 5 à 20 jours). "
        + "La prédiction est linéaire, par optimisation quadratique et convolution.</p>"
        + "<p>Par exemple, pour un département, on dispose des données de mobilité des habitants chaque jour pour des types de lieux: "
        + "commerces et loisirs, parcs et nature, travail, résidence, etc. On dispose aussi des données de température et de pression atmosphérique. "
        + "Ces données sont mises dans une matrice A, dont les lignes sont les jours, et les colonnes sont les données de mobilité et météo (qu'on appellera \"contextes\").</p>"
        + "<p>Supposons qu'on veuille prévoir les passages aux urgences pour soupçon de covid19 chaque jour. "
        + "On connaît ces données jusqu'au présent. Il se trouve que la méthode fonctionne mieux avec la dérivée des données à prévoir: d(j) - d(j-1) si d(j) est la donnée au jour j, et on stocke cette valeur pour chaque jour dans une matrice colonne B dont les lignes sont les jours.</p>"
        + "<p>Ce sont ces données que l'on cherche à prévoir, en fonction des données de contexte, en faisant l'hypothèse qu'il y a un intervalle de temps entre le moment où le contexte change et celui où cela se répercute sur l'épidémie visible: c'est le temps d'apparition des symptômes et de leur aggravation, pour le passage aux urgences.</p> "
        + "<p>On cherche alors un jeu de coefficients C, i.e. une matrice colonne avec autant de lignes que de contextes, telle que AC = B.</p>"
        + '</div>'
        + '<div class="col-sm">'
        + "<p>Comme le nombre de lignes de A (les jours) est bien supérieur au nombre de colonnes, il n'y en a généralement pas, alors on cherche à minimiser l'erreur, i.e. la norme de AC - B. "
        + "C'est un problème d'optimisation convexe, et la solution est celle où la différentielle est nulle, on l'obtient par C = (tA A)^-1 (tA B)</p>"
        + "<p>Comme on ne connaît pas le décalage de jours entre un contexte de A et l'effet qu'il a sur B, on teste tous les décalages et on retient celui où le coefficient de corrélation (cosinus de l'angle entre la colonne de A correspondant au contexte et B dans l'espace adéquat) est localement maximum en valeur absolue. "
        + "Le tableau ci-dessus indique les coefficients de corrélation et les décalages.</p>"
        + "<p>Pour prévoir, par exemple les urgences un jour j donné, on prend les valeurs de chaque contexte le jour du passé correspondant à son décalage d, qui est j - d, et le coefficient de C qui lui correspond, pour chaque département, puis on somme tout cela, et on cumule cette approximation de la dérivée à partir du présent jusqu'au jour j. "
        + "Ce sont les courbes en gras à la fin des courbes dans les premiers graphiques de cette page."
        + '</div></div></div>'
        + '</div>'
        + '</p>'
        + '<h5>Calcul du taux de reproduction R.</h5>'
        + '<div>'
        + "<p>Pour le calcul de :<br>"
        + '<ul><li>R est déterminé avec un intervalle sériel s = 4,11:'
        + "<ul><li>R est le nombre moyen de personnes que contamine un malade (en dessous de 1, l'épidémie régresse et s'arrête rapidement),"
        + "<li>l'intervalle sériel est le nombre moyen de jours entre deux contaminés successifs dans une chaîne de contamination, il vaut 4 jours environ: A contamine B 4 jours après avoir été contaminé,"
        + "<li> si f est la fonction exponentielle étudiée, R = exp(s*f'/f), de sorte que f(x+s) = R*f(x)"
        + '<li> les dérivées sont calculées sur 7 jours'
        + '</ul>'
        + "<li> les données de visite aux urgences, d'hospitalisation, de patients en réanimation, de décès et de tests sont des valeurs sur une journée."
        + '</ul></p>'
        + '</div>')

f.write('<h3> Prévisions variées</h3>')

f.write("<h4>Prévisions des urgences par l'hygiène sociale</h4>")
f.write('<p>À venir...</p>')
f.write("<h4>Prévisions des cas positifs par tranche d'âge</h4>"
        + "<p>Par tranches de 10 ans, par exemple 29 signifie 20-29 ans.</p>"
        + "<p>Résultats peu concluants: les corrélations sont faibles et les décalages fantaisistes.</p>")
f.write(fin1)
f.close()

limages = ['_synthese.html',
           '_prevision_par_mobilite.png',
           '_dispersion_contexte.png',
           '_prevision_positifs.png',
           '_prevision_positifs_ages.png',
           '_prevision_reatot_par_mobilite.png',
           '_contextes_influents.png',
           '_prevision_urge_date_depart.png',
           '_prevision_reatot_date_depart.png',
           '_prevision_hospi_ages.png',
           '_R.png',
           '_heatmap_correlations.png',
           '_heatmap_decalages.png',
]
os.system('scp donnees/_synthese.html lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19')
os.system('tar -cvf groba.tgz '
          + ' '.join(['donnees/' + x for x in limages]))

os.system('scp groba.tgz lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19')
os.system("ssh lpmib@ssh-lpmib.alwaysdata.net 'cd testdjango2/testdjango2/medias/covid19 && tar -xvf groba.tgz'")


