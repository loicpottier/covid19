# synthese d'un jour en France
# mets la synthèse sur cp.lpmib.fr:
# https://cp.lpmib.fr/medias/covid19/synthesis_last.html

import os
import matplotlib
# sous bash de windows, ajouter ca:
if os.uname().nodename == 'pcloic':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import re
import sys
import math

# la date dans ma zone
import datetime
now = datetime.datetime.now()
from pytz import reference
localtime = reference.LocalTimezone()
localtime.tzname(now) #Central European Summer Time
print("now =", now)
# dd/mm/YY H:M:S
dt_string = now.strftime("%A, %d %B %Y %H:%M:%S, " + localtime.tzname(now))
print("date and time =", dt_string)
today = dt_string
print("today = ",today)

try:
    jour = sys.argv[1]
    nsimulations = sys.argv[2]
    intervalles = sys.argv[3]
    resultats = sys.argv[4]
    meilleure_simul = sys.argv[5]
    carte = sys.argv[6]
except:
    jour = '61'
    nsimulations = '2263' 
    intervalles = 'france_rea_jour_61/_intervalles.csv'
    resultats = 'france_rea_jour_61/_resultats_jour_60.csv'
    meilleure_simul = 'france_rea_jour_61/jour_61_err_1.1421_R0_8.20_dR0_4.7_R01_1.03_pvoy_0.08_dpvoy_59_debi_4_duri_8_mor_0.00817_dc_0_nc_2972725_lm_18754_dv_270_ddv_2.py.png'
    carte = 'france_rea_jour_61/jour_61_err_1.1421_R0_8.20_dR0_4.7_R01_1.03_pvoy_0.08_dpvoy_59_debi_4_duri_8_mor_0.00817_dc_0_nc_2972725_lm_18754_dv_270_ddv_2_carte.png'

dir = intervalles.split('/')[0]
try:
    m = re.compile('err_(?P<err>[0-9\.]+)_').search(meilleure_simul)
    meilleure_erreur = m.group('err')
except:
    meilleure_erreur = ''

def table(t):
    ts = ('<table border = 1 >'
          + '\n'.join(['<tr>' + ''.join(['<td>' + x + '</td>' for x in r]) + '</tr>' for r in t])
          + '</table>')
    return(ts)


debut1 = '''<!DOCTYPE html>
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
<body>'''

fin1 = '''
</body>
'''
# doc bootstrap
# https://getbootstrap.com/docs/4.4/layout/overview/

synthese = dir + '/' + '_synthese.html'

f = open(synthese, 'w')
try:
    g = open(dir + '/' + '_histogramme.csv', 'r')
    ls = g.read().split('\n')
    print(ls)
    t = [s.split(';') for s in ls if s != '']
    g.close()
except:
    t = []
    
g = open(intervalles, 'r')
ls = g.read().split('\n')
print(ls)
ti = [s.split(';') for s in ls if s != '']
g.close()
f.write(debut1)
######################################################################
# titre
f.write('<div class="container-fluid">'
        + '<h3>Simulation of the covid19 epidemic in France, day ' + jour + ' since March 4</h3><br>'
        + today + ', beta version'
        + '<br>loic.pottier@gmail.com'
        + '<br>details (to be updated): <a href="https://hal.archives-ouvertes.fr/hal-02555718v1">pre-print hal-02555718v1</a><hr>')

f.write('<table>')

######################################################################
# intervalles
f.write('<tr><td valign=top> <h4>Intervals for random values of parameters</h4>'
        + '<br>' + nsimulations + ' simulations computed so far'
        + table(ti) + '</td>')

######################################################################
# estimations
f.write('<td valign=top> <h4>Estimations of parameters for zero error</h4>'
        + table(t) + '</td></tr>')

######################################################################
# meilleure simulation
im = '<img src="' + meilleure_simul.split('/')[1] + '" width = 700">'
print(im)
f.write('<tr valign=top><td> <h4>Simulation with lowest error (' + meilleure_erreur[:-3] + '%)</h4>'
        + '<br> Deaths in hospital, function of the number of days since March 4.'
        + '<br> In title, nc is the number of infected people at the end of the simulation.'
        + '<br>' + im + '</td>')
######################################################################
# limites de la simulation
im = '<img src="' + meilleure_simul.split('/')[1].replace('.png','_limite.png') + '" width = 700">'
print(im)
f.write('<td> <h4>Limits of this simulation, with end of lockdown on May 11</h4>'
        + '<br> Deaths in hospital, function of the number of days since March 4.'
        + '<br> After end of lockdown: <br> basic reproduction number R02 = sqrt(R01*R0/dR0),'
        + ' <br>ratio of travellers pvoy2 = (pvoy + pvoy/dpvoy)/2,'
        + ' <br> maximal distance of neighbors dv2 = dv as before lockdown.'
        + '<br>For example, if R0 = 7.48, then R02 = 1.28, pvoy2 = 0.03, dv2 = 464, i.e. 41.7 km.'
        + '<br>Day 200 is end of September, 300 is beginning of January 2021.'
        + '<br>' + im + '</td></tr>')

######################################################################
# carte de la simulation
ca = '<img src="' + carte.split('/')[1] + '">'
print(ca)
f.write('<tr><td valign=top> <h4>Map of epidemic for this simulation</h4><br>'
        + 'Each pixel is 4.5 km wide and contains 2500 people '
        + ca + '</td></tr>')


f.write('</table>'
        + '</div>')

######################################################################
# la doc
f.write('''
<div class="container-fluid"><h3>Method</h3>

<div class="container-fluid"><h4>Data</h4>

The data used are the number of patients in intensive care for covid19, from March 4 [1]. It is more regularly updated than the number of hospital deaths. It is closely linked to it: for covid19, the number of hospital deaths on a given day is 7.5% of the number of patients in intensive care on the same day, to within 1%. To study the epidemic, we can therefore reduce ourselves to studying the number of patients in intensive care each day.

The French population is modeled by 60 million individuals, placed in a square of 700 km aside. 
Each individual has a state:
<ul>
<li> 0 if he has never been in contact with the virus
<li>1 or more: this is the number of days since he was infected with the virus
<li>-1 if he died
</ul>
</div>

<div class="container-fluid"><h4>Parameters</h4>

They are separated into 2 groups:<br>

<h5>Variable parameters</h5>

<ul>
<li>    <b>R0</b>: the number of people contaminated by an infected person during the period in which they are contagious[5]
<li>     <b>dR0</b>: the divider of R0 during the social distancing period: from March 9 to March 16, i.e. R0 becomes R0 / dR0.
<li>     <b>R01</b>: R0 during lockdown, ie from March 17th.
<li>     <b>pvoy</b>: the rate of travelers (people who travel far during a day)
<li>     <b>dpvoy</b>: the divider of the rate of traveler during lockdown.
<li>     <b>debi</b>: the number of days before an infected person becomes contaminant.
<li>     <b>duri</b>: the duration during which an infected person is contaminant.
<li>     <b>IFR</b>: the mortality rate (Infection Fatality Ratio), ie the probability that an infected person has to die of disease.
<li>     <b>dv</b>: maximal distance of neighbors (in multiples of 0.09 km), who are the people likely to be met by a person during a day.
<li>     <b>ddv</b>: divisor of dv during lockdown.
</ul>

<h5>Fixed parameters</h5>

<ul>
<li>     <b>dinc</b>: the duration of incubation, from 2 to 12 days, on average 5, taken at random for each individual [3]
<li>     <b>dmal</b>: the duration of the disease, fixed at 20 days [3] [4]. An infected person dies with the IFR probability in the period between dinc + dmal / 2 and dinc + dmal.
</ul>

<h4>Simulation</h4>

The country is a square matrix of 7746 x 7746 people (it can be seen as a computer screen with resolution 7746 x 7746). Each pixel is 90 m wide and contains one person. The neighbors of a person are those at distance less than dv * 0.09 km (Manhattan distance). This determines the number of people likely to be contaminated by a non-traveling person.<br>

If she is a traveler, the whole population is likely to be contaminated by her.<br>


On day 1 we choose 3 people at random who become contaminated (states = 1,2,3),

then one calculates day after day the states of the 60 million people
<ul>
<li>     every day, a person will or will not contaminate people randomly among its geographic neighbors, or throughout the country, for travelers; for this we use a Poisson distribution with parameter R0.
<li>     every day, an infected person can die (Bernouilli's distribution of parameter IFR / (dmal / 2)), if he is in the second period of the disease (state between dinc + dmal / 2 and dinc + dmal).
</ul>
The day when the number of deaths exceeds that of reality on March 9 is considered to be March 9 for the simulation, which then stops at the end of the actual data.<br>


 At the end, the error is calculated[6] with the actual death numbers each day, reduced to a percentage of the actual death number on the last day.
</div>
</div>
''')
f.write('<div class="container-fluid"><h4> Past days</h4>')
for j in range(60,int(jour)+1):
    f.write('<a href="synthese_jour_' + str(j) + '.html"> Day ' + str(j) + '</a><br>')
f.write('</div>')

f.write('''
<div class="container-fluid">[1] <a href = "https://geodes.santepubliquefrance.fr/#c=indicator&f=0&i=covid_hospit.rea&s=2020-04-25&t=a01&view=map2" >geodes.santepubliquefrance.fr</a>
''')
f.write(fin1)
f.close()
os.system('cp ' + synthese + ' syntheses/synthese_jour_' + jour + '.html')
os.system('cp ' + synthese + ' syntheses/synthesis_last.html')
os.system('scp syntheses/synthesis_last.html lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19')
os.system('scp ' + synthese + ' syntheses/synthese_jour_'
          + jour + '.html lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19')
os.system('scp ' + meilleure_simul + ' lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19')
os.system('scp ' + meilleure_simul.replace('.png','_limite.png') + ' lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19')

os.system('scp ' + carte + ' lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19')
