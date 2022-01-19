from monde0 import *

######################################################################
# prévisions des pays

import time
t0 = time.time()
CPj = coefficients_prevision(M,jourdepartprevision = jaujourdhui + 1)
t1 = time.time()
P1 = prevoit_indicateursR(M,CPj,jourdepartprevision = jaujourdhui + 1)
t2 = time.time()
print('calcul de CPj:',"%2.3f" % (t1 - t0), '\n           P1:', "%2.3f" % (t2 - t1))

'''
p = 'France'
i = 'weekly_icu_admissions'
jdep,jfin,v = M[p][i]
jdep1,jfin1,v = P1[p][i]
deb = 200
fin = deb + 1000

for k in range(0,lissagesReff+1):
  plt.plot(r0(np.maximum(lissage(M[p][i][2][deb:fin],7,repete = k),0),
                               derive=7,maxR = 3))

plt.grid(); plt.show()

plt.plot(M[p][i][2][deb:fin])
plt.grid(); plt.show()


plt.plot([prevoit_jour(M,P1,CPj,p,i,j,jaujourdhui+1,trace=True) for j in range(jdep+deb,jdep+min(jdep+jfin1,fin))])
plt.plot(P1[p][i][2][deb:fin])
plt.plot(M[p][i][2][deb:fin])
plt.grid(); plt.show()


p = 'France'
i = 'Rweekly_icu_patients'
jdep,jfin,v = M[p][i]
jdep1,jfin1,v = P1[p][i]
deb = 540
fin = deb + 10
plt.plot([prevoit_jour(M,P1,CPj,p,i,j,jaujourdhui+1,trace=True) for j in range(jdep+deb,jdep+min(jdep+jfin1,fin))])
plt.plot(P1[p][i][2][deb:fin])
plt.plot(M[p][i][2][deb:fin])
plt.grid(); plt.show()

plt.plot(P1[p]['grocery_and_pharmacy_percent_change_from_baseline'][2])
plt.plot(M[p]['grocery_and_pharmacy_percent_change_from_baseline'][2])
plt.grid(); plt.show()

p = 'United States'
i = 'Rnew_cases'
jdep,jfin,v = M[p][i]
jdep1,jfin1,v = P1[p][i]
deb = 500
plt.plot(P1[p][i][2][deb:])
plt.plot(M[p][i][2][deb:])
plt.plot([prevoit_jour(M,P1,CPj,p,i,j,jaujourdhui+1) for j in range(jdep,jfin1+1)][deb:])
plt.grid(); plt.show()

'''

# profiling 
import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
P1 = prevoit_indicateursR(M,CPj,jourdepartprevision = jaujourdhui + 1)
pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream = s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())



'''

for i0 in correlations[p][i]:
  if i0 != 'date':
    plt.plot(P1[p][i0][2])
    plt.plot(M[p][i0][2])

plt.grid(); plt.show()

i0 = 'retail_and_recreation_percent_change_from_baseline'
plt.plot(P1[p][i0][2])
plt.plot(M[p][i0][2])
plt.grid(); plt.show()

'''

def affiche_infos(p):
    affiche_correlations(p)
    print('----------- erreurs de prévision ' + p + ' ----------------')
    for i in nomsindicateursRp(p):
        jdep,jfin,v = P1[p][i]
        C,dependances,e = CPj[p][i]
        print('prevision de', i + '.'*(30 - len(i)) , ("%3.0f" % (100*e)) + '%')
    print('----------------------------------------------------------------------')

#affiche_infos('United Kingdom')
#affiche_infos('United States')
affiche_infos('France')

DIRPREV = 'previsions_quotidiennes/'
debuttrace = 300

lescharts = dict([(p,{}) for p in paysok])

######################################################################
# prochains pics, déterminés par Reff <= 1

try:
    f = open(DIRPREV + 'pics.pickle','rb')
    lespics = pickle.load(f)
    f.close()
except:
    lespics = {}

pics = dict([(p,{}) for p in paysok])
indicateursRpic = ['Rhosp_patients','Ricu_patients',
                   'Rpositive_rate','Rnew_cases','Rweekly_hosp_admissions','Rweekly_icu_admissions', 'Rnew_deaths',
                   ]

for p in paysok:
    for i in [x for x in indicateursRpic if x in nomsindicateursRp(p)]:
        i1 = i[1:]
        jdep,jfin,v = M[p][i]
        jpic = jaujourdhui
        while P1[p][i][2][jpic-jdep] > 1 and jpic < jourfin: 
            jpic += 1
        if jpic not in [jaujourdhui,jourfin]:
            pics[p][i1] = (jpic, # jour du pic
                           P1[p][i1][2][jpic-jdep]) # valeur de l'indicateur ce jour-là

lespics[aujourdhuih] = pics

f = open(DIRPREV + 'pics.pickle','wb')
pickle.dump(lespics,f)
f.close()

######################################################################
#

def trace_previsions(p,indicateurs,P1):
    for i in indicateurs:
        jdep,jfin,v = M[p][i]
        debut = jaujourdhui - jdep - 60 
        vmax = abs(np.max(v))
        lcourbes = []
        jf = jfin + 1
        df = 60
        dferreurs = min(max(erreurs_moy[p][i]),8*7)
        vP1 = P1[p][i][2]
        v1 = np.minimum(vP1[debut:jf - jdep + df],
                        5*vmax)
        ve = np.array([erreurmoyenne(p,i,j-jf) for j in range(jf, jf + dferreurs)])
        vsup = np.minimum(np.concatenate([vP1[debut:jf - jdep],
                                          np.array([(1 + ve[j-jf] / 100) * vP1[j - jdep]
                                                    for j in range(jf, jf + dferreurs)
                                                    if j - jdep < len(vP1)])]),
                          5*vmax)
        vinf = np.minimum(np.concatenate([vP1[debut:jf - jdep],
                                          np.array([(max(0,1 - ve[j-jf] / 100)) * vP1[j - jdep]
                                                    for j in range(jf, jf + dferreurs)
                                                    if j - jdep < len(vP1)])]),
                          5*vmax)
        lcourbes += [(zipper([jour_de_num[j] for j in range(jdep + debut, jdep + debut + len(v1))], # jf + df)],
                             v1),
                      'prévision au ' + jour_de_num[jf], prev)]
        for ivp,vp in enumerate([vsup, vinf]):
            lcourbes += [(zipper([jour_de_num[j] for j in range(jdep + debut,jf + dferreurs)],
                                 vp),
                          'sup' if ivp == 0 else 'inf', prev)]
        lcourbes = lcourbes[::-1] +  [(zipper([jour_de_num[j] for j in range(jdep + debut,jfin+1)],
                                              v[debut:]),
                                       i,real)]
        lj = []
        for (courbe,nom,t) in lcourbes:
            lj = lj + [x[0] for x in courbe]
        dcourbes = [dict(courbe) for (courbe,nom,t) in lcourbes]
        lj = sorted(list(set(lj))) # liste des jours en abcisse
        courbemin = [(j,min([dcourbe[j] for dcourbe in dcourbes if j in dcourbe]))
                     for j in lj]
        courbemax = [(j,max([dcourbe[j] for dcourbe in dcourbes if j in dcourbe]))
                     for j in lj]
        lcourbes = [(courbemax,'',prev)] + lcourbes + [(courbemin,'',prev)]
        titre = traduitindicateur(i)
        options = ('series: {'  #https://material.io/design/color/the-color-system.html#tools-for-picking-colors
                   + " 0: { areaOpacity : 1.0 ,  color : '#B0BEC5', lineWidth: 0}," # courbemax #80CBC4
                   + ' '.join([str(k) + ": {areaOpacity : 0.0 , lineDashStyle: [10, 2] }, " 
                               for k in range(1,len(lcourbes) - 3)])
                   + str(len(lcourbes) - 3)
                   + " : { areaOpacity : 0.0 , lineWidth: 4, color : 'blue', lineDashStyle: [6, 2]}," #prevision
                   + str(len(lcourbes) - 2) + " : { areaOpacity : 0.0 , lineWidth: 4, color : 'blue' }," #reelle
                   + str(len(lcourbes) - 1) + ": { areaOpacity : 1.0 ,  color : 'white', lineWidth: 0}," # courbemin
                   + '}')
        nomchart = i
        #print(nomchart)
        lescharts[p][('prévisions avec erreurs',nomchart)] = trace_charts(DIRPREV, lcourbes, titre = titre,
                                                                          options = options, area = True)

######################################################################
# previsions à 28 jours dans le passé

fin = jaujourdhui
duree_evaluation = 300 #500
mois_evaluation = duree_evaluation // 30
nsemaines = 6

def previsions_passe(listepays):
    erreurs = dict([(p,dict([(i,dict([(k*7,[]) for k in range(1,nsemaines+1)]))
                             for i in nomsindicateursRp(p) + nomsindicateursReffp(p)]))
                    for p in listepays]) # paysok
    P1s = {}
    print('prévisions passées')
    for j in range(fin + 1, fin - duree_evaluation - 1, -7): #range(fin - duree_evaluation, fin + 1,7):
        print(j, jour_de_num[j], end = ', ', flush = True)
        CPj = coefficients_prevision(M,jourdepartprevision = j)
        P1 = prevoit_indicateursR(M,CPj,jourdepartprevision = j)
        P1s[j] = P1
        for p in erreurs:
            for i in erreurs[p]:
                jdep,jfin,v = M[p][i]
                for k in range(1,nsemaines+1): 
                    if j+k*7 <= jfin:
                        erreurs[p][i][7*k].append(abs((P1[p][i][2][j+k*7 -jdep] - v[j+k*7 -jdep])
                                                      / v[j+k*7 -jdep]))
    print('fin des prévisions passées')
    return(erreurs,P1s)

# ne pas mettre de ' dans ces textes (à cause de javascriot)
traduit = {'hosp_patients': 'nombre de patients hospitalisés',
           'icu_patients' : 'nombre de patients en soins intensifs',
           'new_cases' : 'nombre de nouveaux cas',
           'positive_rate': 'taux de positivité des tests',
           'weekly_hosp_admissions': 'nombre de patients hospitalisés par jour',
           'weekly_icu_admissions': 'nombre de nouveaux patients soins critiques par jour',
           'new_deaths': 'nombre de nouveaux décès',}

dtraduitindicateur = {'hosp_patients': 'hospitalisations',
                      'icu_patients' : 'soins intensifs',
                      'new_cases' : 'nouveaux cas',
                      'positive_rate': 'taux de positivité',
                      'weekly_hosp_admissions': 'nouvelles hospitalisations',
                      'weekly_icu_admissions': 'nouveaux soins critiques',
                      'new_deaths': 'nouveaux décès',}

def traduitindicateur(i):
    if i[0] == 'R':
        return('R ' + traduitindicateur(i[1:]))
    elif i in dtraduitindicateur:
        return(dtraduitindicateur[i])
    else:
        return(i)

def trace_previsions_passes(p,indicateurs,P1s,dureefutur = 28): # minimum tous les 7 jours
    for i in indicateurs:
        #debut = 600-duree_evaluation #if i not in nomsindicateursRp(p) else 500-duree_evaluation + 100
        debut = 600-duree_evaluation
        jdep,jfin,v = M[p][i]
        vmax = abs(np.max(v))
        lcourbes = []
        for j in range(fin + 1, fin - duree_evaluation - 1, - max(7,7)):
            df = 60 if j == fin + 1 else dureefutur
            P1 = P1s[j]
            v1 = np.minimum(P1[p][i][2][debut:j - jdep + df],
                            5*vmax)
            lcourbes += [(zipper([jour_de_num[j] for j in range(jdep + debut,jdep + debut + len(v1))],#j + df)],
                                 v1),
                          'début:' + jour_de_num[j], prev)]
        lcourbes = lcourbes[::-1] +  [(zipper([jour_de_num[j] for j in range(jdep + debut,jfin+1)],
                                                  v[debut:]),
                                           i,real)]
        lj = []
        for (courbe,nom,t) in lcourbes:
            lj = lj + [x[0] for x in courbe]
        dcourbes = [dict(courbe) for (courbe,nom,t) in lcourbes]
        lj = sorted(list(set(lj))) # liste des jours en abcisse
        courbemin = [(j,min([dcourbe[j] for dcourbe in dcourbes if j in dcourbe]))
                     for j in lj]
        courbemax = [(j,max([dcourbe[j] for dcourbe in dcourbes if j in dcourbe]))
                     for j in lj]
        lcourbes = [(courbemax,'',prev)] + lcourbes + [(courbemin,'',prev)]
        titre = traduitindicateur(i)
        options = ('series: {'  #https://material.io/design/color/the-color-system.html#tools-for-picking-colors
                   + " 0: { areaOpacity : 1.0 ,  color : '#B0BEC5', lineWidth: 0}," # courbemax #80CBC4
                   + ' '.join([str(k) + ": {areaOpacity : 0.0 , lineDashStyle: [10, 2] }, " 
                               for k in range(1,len(lcourbes) - 3)])
                   + str(len(lcourbes) - 3)
                   + " : { areaOpacity : 0.0 , lineWidth: 4, color : 'blue', lineDashStyle: [6, 2]}," #prevision
                   + str(len(lcourbes) - 2) + " : { areaOpacity : 0.0 , lineWidth: 4, color : 'blue' }," #reelle
                   + str(len(lcourbes) - 1) + ": { areaOpacity : 1.0 ,  color : 'white', lineWidth: 0}," # courbemin
                   + '}')
        nomchart = i
        #print(nomchart)
        lescharts[p][('prévisions passées',nomchart)] = trace_charts(DIRPREV, lcourbes, titre = titre,
                                                                     options = options, area = True)

def trace_donnees(p,donnees):
    for i in donnees:
        debut = 600-duree_evaluation if i != "taux du variant omicron suspecté" else 600-duree_evaluation + 260
        jdep,jfin,v = M[p][i]
        jfin2 = min(jfin,jaujourdhui)
        vmax = abs(np.max(v))
        lcourbes = []
        lcourbes += [(zipper([jour_de_num[j] for j in range(jdep + debut,jfin2+1)],
                             v[debut:-(jfin - jfin2)] if jfin2 != jfin else v[debut:]),
                      i,real)]
        titre = i
        nomchart = i 
        #print(nomchart)
        lescharts[p][('données',nomchart)] = trace_charts(DIRPREV, lcourbes, titre = titre)

##########################
# les erreurs de prévision

#erreur_rea_28, erreur_hospi_28, erreur_cases_28 = 0,0,0

erreurs_moy = dict([(p,{}) for p in paysok])

def erreurmoyenne(p,i,j):
    k = j //  7
    dj = j %  7
    e = erreurs_moy[p][i][7*k] * (1 - dj /  7) + erreurs_moy[p][i][7*(k+1)] * dj /  7
    return(e)

try:
    f = open(DIRPREV + 'erreurs.pickle','rb')
    leserreurs = pickle.load(f)
    f.close()
except:
    leserreurs = {}

#erreurs = dict([(p,{}) for p in paysok])

iscore1 = ['icu_patients','hosp_patients']
iscore2 = ['Rnew_tests', 'Rpositive_rate','weekly_hosp_admissions','new_deaths']

# moyenne des erreurs pour les indicateurs iscore1 et 2, jusqu'à 6 semaines
def score(p,erreurs): # lerreurs liste d'erreurs pour les indicateurs
    s = []
    for i in erreurs[p]:
        for k in range(1,nsemaines+1): 
            #print(ni,i,k-1)
            if k in [1,2,3,4,5,6] and i in iscore1: #or i in iscore2: # semaines pour le score
                s.append(100*erreurs[p][i][7*k])
    #print(s)
    return(np.sqrt(np.mean(np.array(s)**2)))


def bilan_erreurs(listepays):
    scores = {}
#    global erreur_rea_28, erreur_hospi_28, erreur_cases_28
    erreurs,P1s = previsions_passe(listepays)
    for p in listepays:
        for i in erreurs[p]:
            for k in range(1,nsemaines + 1):
                erreurs[p][i][7*k] = np.sqrt(np.mean(np.array(erreurs[p][i][7*k])**2)) #np.mean(erreurs[p][i][7*k])
    for p in listepays:
        lindic = [i for i in nomsindicateursR + nomsindicateursReff if i in erreurs[p]]
        s  = score(p,erreurs)
        for i in lindic:
            erreurs_moy[p][i] = dict([(7*k,100*erreurs[p][i][7*k]) for k in range(1,nsemaines+1)])
            erreurs_moy[p][i][0] = 0
        scores[p] = s
        if p == 'France':
            print('-------------------------------')
            print('%s   7j' % (p[:16]+'.'*(16-len(p))), end = '')
            for k in range(2,nsemaines+1):
                print('  %dj'% (7*k), end = '')
            print('')
            print('---------')
            for i in lindic:
                print('%s' % i[:16]+'.'*(16-len(i)), end = '')
                for k in range(1,nsemaines+1):
                    v = erreurs_moy[p][i][7*k]
                    if v > 50:
                        print('     ', end = '')
                    else:
                        print(' %3.0f%s' % (v,'%'), end = '')
                print('')
            print('--- ' + p + ' score: %3.2f' % s)
    return(scores,erreurs,P1s)

# prend du temps
scores,erreurs,P1s = bilan_erreurs(paysok)

print(scores)

leserreurs[aujourdhuih] = erreurs

f = open(DIRPREV + 'erreurs.pickle','wb')
pickle.dump(leserreurs,f)
f.close()

######################################################################
# tracés google charts et tableaux de données pour la synthese html

atracer = ['hosp_patients','icu_patients','positive_rate',
           'new_cases','weekly_hosp_admissions','weekly_icu_admissions','new_deaths',
]
atracer = atracer + ['R' + x for x in atracer]

def atracerp(p):
    return([i for i in atracer if i in M[p]])

for p in paysok:
    print(p)
    trace_previsions(p,atracerp(p),P1)
    print('fait')
    if True: #p == 'France':
        trace_previsions_passes(p,
                                atracerp(p),
                                P1s,
                                dureefutur = 30)


######################################################################
# erreurs moyennes, faudrait calculer les quantiles pour les forecasthub
quantiles1 = [0.010,0.025,0.050,0.100,0.150,0.200,0.250,0.300,0.350,0.400,0.450,0.500,0.550,0.600,0.650,0.700,0.750,0.800,0.850,0.900,0.950,0.975,0.990]
quantiles2 = [0.025,0.100,0.250,0.500,0.750,0.900,0.975] #N wk ahead inc case

def erreurs_moyennes(p,i):
    jdep,jfin,v = M[p][i]
    debut = jaujourdhui - jdep - 60 
    vmax = abs(np.max(v))
    lcourbes = []
    jf = jfin + 1
    df = 60
    dferreurs = min(max(erreurs_moy[p][i]),8*7)
    vP1 = P1[p][i][2]
    v1 = np.minimum(vP1[debut:jf - jdep + df],
                    5*vmax)
    ve = np.array([min(100,erreurmoyenne(p,i,j-jf)) for j in range(jf, jf + dferreurs)])
    return(ve)

'''
p = 'France'
i = 'icu_patients'
erreurs_moyennes(p,i)
p = 'United States'
i = 'weekly_hosp_admissions'
erreurs_moyennes(p,i)
'''
######################################################################
# forecasthub
'''
https://github.com/reichlab/covid19-forecast-hub/blob/master/data-processed/README.md#Data-formatting

nom de la prévision
YYYY-MM-DD-team-model.csv

'''
team = 'prolix'
model = 'euclidean'
colonnes = ['forecast_date','target','target_end_date',
            'location','type','quantile','value']
jourssemaine = ['lundi', 'mardi', 'mercredi',
                'jeudi', 'vendredi', 'samedi', 'dimanche']

def joursemaine(j):
    j0 = num_de_jour('2021-12-31') + 3 # vendredi + 3 = lundi
    return(jourssemaine[(j - j0) % 7])

def target_end_date(j):
    js = j
    while joursemaine(js) != 'samedi':
        js += 1
    if joursemaine(j) in ['dimanche','lundi']:
        return(js)
    else:
        return(js + 7)

lines = [colonnes]

p = 'United States'
quantile = True

def lines_indic(i,colonne):
    ve = erreurs_moyennes(p,i)
    jdep,jfin,v = P1[p][i]
    nmax = 7 #8
    if i == 'new_cases': nmax == 7 #8
    if i == 'new_deaths': nmax == 7 #20
    for n in range(1,nmax + 1):
        samedi_n = target_end_date(jaujourdhui + (n-1)*7)
        value = int(sum(v[samedi_n - 6 - jdep:samedi_n + 1 - jdep]))
        dv =  min(100, 2 * ve[samedi_n - 3 - jaujourdhui]) if samedi_n - 3 - jaujourdhui < len(ve) else 100
        if quantile:
            quantiles = quantiles1 if colonne != 'N wk ahead inc case' else quantiles2
            for q in quantiles:
                lines.append([aujourdhui,
                              str(n) + colonne[1:],
                              jour_de_num[samedi_n],
                              'US',
                              'quantile',
                              "%0.3f" % q,
                              int(value * (100 - dv + q * 2 * dv) / 100)])
        lines.append([aujourdhui,
                      str(n) + colonne[1:],
                      jour_de_num[samedi_n],
                      'US',
                      'point',
                      'NA',
                      value])

lines_indic('new_cases','N wk ahead inc case')
lines_indic('new_deaths','N wk ahead inc death')

i = 'daily_hosp_admissions'
jdep,jfin,v = P1[p][i]
ve = erreurs_moyennes(p,i)

for n in range(0,min(jfin - jaujourdhui, 130) + 1):
    dv = min(100, 2 * ve[n]) if n < len(ve) else 100
    if quantile:
        for q in quantiles1:
            lines.append([aujourdhui,
                          str(n) + ' day ahead inc hosp',
                          jour_de_num[jaujourdhui + n],
                          'US',
                          'quantile',
                          "%0.3f" % q,
                          int(v[jaujourdhui + n - jdep] * (100 - dv + q * 2 * dv) / 100)
                          ])
    lines.append([aujourdhui,
                  str(n) + ' day ahead inc hosp',
                  jour_de_num[jaujourdhui + n],
                  'US',
                  'point',
                  'NA',
                  int(v[jaujourdhui + n - jdep])
    ])

f = open('forecast_hub_US/' + team + '-' + model + '/' + aujourdhui + '-' + team + '-' + model + '.csv','w')
for l in lines:
    f.write(','.join([x if type(x) is str else str(int(x)) 
                      for x in l]) + '\n')

f.close()

print('----------------------------------------------------------------------')
print('forecast US ecrite',flush = True)

'''
p = 'United States'
i = 'Rdaily_hosp_admissions'
plt.plot(P1[p][i][2])
plt.plot(M[p][i][2])
plt.grid();plt.show()

p = 'United States'
i = 'Rpositive_rate'
i = 'Rnew_cases'
plt.plot(P1[p][i][2][400:])
plt.plot(M[p][i][2][400:])
plt.grid();plt.show()

i = 'Rnew_cases'
i = 'Rweekly_hosp_admissions'
correlations[p][i]

for i0 in correlations[p][i]:
  if i0 != 'date':
    plt.plot(P1[p][i0][2])
    plt.plot(M[p][i0][2])

plt.grid(); plt.show()

  
'''

######################################################################
# forecast hub europe

# https://github.com/covid19-forecast-hub-europe/covid19-forecast-hub-europe/wiki/Forecast-format

# ['Belgium', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'France', 'Germany', 'Ireland', 'Israel', 'Italy', 'Luxembourg', 'Malta', 'Portugal', 'Slovenia', 'Switzerland', 'United Kingdom', 'United States']
codespays = {'Belgium': 'BE',
             'Cyprus': 'CY',
             'Czechia': 'CZ',
             'Denmark': 'DK',
             'Estonia': 'EE',
             'France': 'FR',
             'Germany': 'DE',
             'Ireland': 'IE',
             'Italy': 'IT',
             'Luxembourg': 'LU',
             'Malta': 'MT',
             'Portugal': 'PT',
             'Slovenia': 'SI',
             'Switzerland': 'CH',
             'United Kingdom': 'GB',
             }

lines = [colonnes]

quantile = True
def lines_indicp(p,i,colonne):
    ve = erreurs_moyennes(p,i)
    jdep,jfin,v = P1[p][i]
    cp = codespays[p]
    nmax = 7 #8
    for n in range(1,nmax + 1):
        samedi_n = target_end_date(jaujourdhui + (n-1)*7)
        value = int(sum(v[samedi_n - 6 - jdep:samedi_n + 1 - jdep]))
        dv =  min(100, 2 * ve[samedi_n - 3 - jaujourdhui]) if samedi_n - 3 - jaujourdhui < len(ve) else 100
        if quantile:
            for q in quantiles1:
                lines.append([aujourdhui,
                              str(n) + colonne[1:],
                              jour_de_num[samedi_n],
                              cp,
                              'quantile',
                              "%0.3f" % q,
                              int(value * (100 - dv + q * 2 * dv) / 100)])
        lines.append([aujourdhui,
                      str(n) + colonne[1:],
                      jour_de_num[samedi_n],
                      cp,
                      'point',
                      'NA',
                      value])

for p in codespays:
    if p in paysok:
        lines_indicp(p,'new_cases','N wk ahead inc case')
        lines_indicp(p,'new_deaths','N wk ahead inc death')
        if p in ['France','United Kingdom']:
            lines_indicp(p,'weekly_hosp_admissions','N wk ahead inc hosp')

f = open('forecast_hub_EU/' + team + '-' + model + '/' + aujourdhui + '-' + team + '-' + model + '.csv','w')
for l in lines:
    f.write(','.join([x if type(x) is str else str(int(x)) 
                      for x in l]) + '\n')

f.close()
print('----------------------------------------------------------------------')
print('forecast EU ecrite',flush = True)

######################################################################

mobilitesgoogle = ['retail_and_recreation_percent_change_from_baseline',
                   'grocery_and_pharmacy_percent_change_from_baseline',
                   'parks_percent_change_from_baseline',
                   'transit_stations_percent_change_from_baseline',
                   'workplaces_percent_change_from_baseline',
                   'residential_percent_change_from_baseline']

trace_donnees('France',
              mobilitesgoogle + (mobilitesapple if useapple else [])
              + nomseauxusees + nomsvariants)

def tableaux_erreurs(erreurs):
    taberreurs = {}
    for p in erreurs:
        s  = score(p,erreurs)
        t = [[p + ('<br>score: %3.2f' % s)] + [('%dj'% (7*k)) for k in range(1,nsemaines+1)]]
        lindic = [i for i in nomsindicateursR + nomsindicateursReff if i in erreurs[p]]
        for i in lindic:
            t.append([i] + [('%3.0f%s' % (100*erreurs[p][i][7*k],'%'))
                            if 100*erreurs[p][i][7*k] < 30
                            else ''
                            for k in range(1,nsemaines+1)])
        taberreurs[p] = t
    return(taberreurs)

tableauxerreurs = tableaux_erreurs(erreurs)

toutlescriptdescharts = []

def charthtml(c):
    toutlescriptdescharts.append(c['script'])
    return(c['div'])

def leschartsfun(p,x):
    try:
        return(lescharts[p][x])
    except:
        print('problème chart', p, x, x in lescharts[p])
        return({'id':'','href':'','script':'','div':''})

def ecritpreverreurs_navbar(f,p):
    f.write(navbar_dropdown('navbarpreverreurs' + p + 'previsions',
                            'Prévisions',
                            [(traduitindicateur(nom),
                              [("valeur de l'indicateur", leschartsfun(p,('prévisions avec erreurs',nom))['href']),
                               ('taux de reproduction R', leschartsfun(p,('prévisions avec erreurs','R' + nom))['href'])])
                             for nom in atracerp(p) if nom[0] != 'R']))

def ecritprevpassees_navbar(f,p):
    f.write('Prévisions passées: précision moyenne <a href="#precision">[2]</a>: <b>' + ('%3.2f' % scores[p]) + '%</b><p>')
    if True: #scores[p] <= 20:
        f.write(navbar_dropdown('navbarprevpassees' + p + 'previsionspassees',
                                'Prévisions passées',
                                [(traduitindicateur(nom),
                                  [("valeur de l'indicateur", leschartsfun(p,('prévisions passées',nom))['href']),
                                   ('taux de reproduction R', leschartsfun(p,('prévisions passées','R' + nom))['href'])])
                                 for nom in atracerp(p) if nom[0] != 'R']))

def chartpicval(p,i):
    i1 = i[1:]
    s = [(j, jour_de_num[lespics[j][p][i1][0]], lespics[j][p][i1][1])
         for j in lespics if i1 in lespics[j][p]]
    if s != []:
        lcourbes2 = [(zipper([jj for (jj,j,v) in s], [float(v) for (jj,j,v) in s]),
                     'nombre de patients', prev)]
        lechart2 = trace_charts(DIRPREV, lcourbes2, titre = "Prévision du " +  traduit[i1] + " au pic ")
        return((i1,lechart2))
    else:
        #print('pas de pic déjà prévu pour', i1)
        return((i1,{'id':'','href':'','script':'','div':'pas de pic prévu'}))

def chartpicjours(p,indicateurs):
    lcourbes = []
    for i in indicateurs:
        i1 = i[1:]
        s = [(j, jour_de_num[lespics[j][p][i1][0]], lespics[j][p][i1][1])
             for j in lespics if i1 in lespics[j][p]]
        if s != []:
            lcourbes.append((zipper([jj for (jj,j,v) in s], [j for (jj,j,v) in s]),
                             traduitindicateur(i1), prev))
        else:
            pass
            #print('pas de pic déjà prévu pour', i1)
    return(trace_charts(DIRPREV, lcourbes, titre = "Jours prévu pour les pics en fonction du jour de prévision"))

def ecritpics_navbar(f,p):
    f.write('<a id="pics"></a>'
            + vspace + "<h4>Évolution des pics prévus pour les indicateurs du Covid19 (France).</h4>"
            + "Le pic est déterminé par le premier jour où le R associé atteint la valeur 1 en décroissant.<p>"
            + 'En abcisse: jour où a été effectuée la prévision.<p>'
            + "Les irrégularités ont lieu lorsque des données manquent pour les prévisions (typiquement lorsque les sites où se trouvent les données sont mis à jour ou bien inaccessibles), ou lorsque les coefficients modélisant l'effet de la vaccinations fluctuent.<p>"
            + table2([['<h5>Jours prévus pour les pics des indicateurs</h5>',
                   '<h5>Valeurs des indicateurs à leurs pics</h5>'],
                  [charthtml(chartpicjours(p,indicateursRpic)),
                   navbar_dropdown('navbarpics' + p + 'pics',
                                   p,
                                   [(traduitindicateur(i), [('indicateur', ch2['href'])])
                                    for (i,ch2) in [chartpicval(p,i) for i in indicateursRpic]])]]))

######################################################################
# page de synthèse
# upload la synthèse sur cp.lpmib.fr:
# https://cp.lpmib.fr/medias/covid19/_synthese.html
######################################################################

synthese = DIRPREV + '_synthese.html'
f = open(synthese, 'w', encoding = 'utf8')

vspace = '<p> <br> <br></p>'*1 # ruse dégueue pour pas que la navbar se superpose

f.write(debut1)

f.write('''
<nav class="navbar fixed-top navbar-expand-lg navbar-dark bg-success">
  <a class="navbar-brand" href="#">Covid19: la suite?</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavAltMarkup" aria-controls="navbarNavAltMarkup" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNavAltMarkup">
    <div class="navbar-nav">
             <a class="nav-item nav-link active" href="#pics">Évolution des pics prévus (France)</a>
             '''
        + '\n'.join(['<a class="nav-item nav-link active" href="#previsions ' + p + '">'
                     + p + '</a>'
                     for p in ['France']])
        + '''      
     <li class="nav-item dropdown">
        <a class="nav-link dropdown-toggle" href="#" id="navbarDropdown" role="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
          Régions
        </a>
        <div class="dropdown-menu" aria-labelledby="navbarDropdown">
''')
for reg in nomsregions:
    f.write('''
          <a class="dropdown-item" href="#previsions '''+ reg + '''">''' + reg + '''</a>''')

f.write('''
        </div>
      </li>
'''
        + '\n'.join(['<a class="nav-item nav-link active" href="#previsions ' + p + '">'
                     + p + '</a>'
                     for p in sorted([p for p in paysok if p != 'France' and p not in nomsregions])])
        + '''<a class="nav-item nav-link active" href="https://github.com/loicpottier/covid19">Code</a>
    </div>
  </div>
</nav>

''')

f.write(vspace + '<div class="container-fluid">'
        + '<p>' + time.asctime(now) + '</p>'
        + '<p>loic.pottier@gmail.com<a href="https://twitter.com/Loic_Pottier_"><img src="https://help.twitter.com/content/dam/help-twitter/brand/logo.png" width = 50></a></p>'
        )

f.write("<h4>Données et prévisions des indicateurs principaux de l'épidémie de covid19 en France et dans le monde</h4>")
f.write('''Prévisions participant aux efforts européens <a href="https://covid19forecasthub.eu">covid19forecasthub.eu</a>
          et US <a href="https://covid19forecasthub.org">covid19forecasthub</a> (forecast model: prolix-euclidean).<br>
          Pour la France, consulter aussi <a href="https://modelisation-covid19.pasteur.fr/">la modélisation de l'équipe de S.Cauchemez</a>.<br>''')
tpic = []
for i in indicateursRpic:
    i1 = i[1:]
    if i1 in lespics[aujourdhuih]['France']:
        jpicf,vpicf = lespics[aujourdhuih]['France'][i1]
        v = ('%3.0f' % vpicf) if vpicf > 1 else (('%3.1f' % (100 * vpicf)) + '%')
        tpic.append([jour_de_num[jpicf], # pour trier ensuite
                     traduit[i1],
                     joli2(jour_de_num[jpicf]),
                     v])
        print('pic prévu pour',i1,':',joli2(jour_de_num[jpicf]),v)
    else:
        print('pas de pic prévu pour',i1)

tpic = ([['indicateur','jour du pic prévu',"valeur de l'indicateur au pic"]]
        + [x[1:] for x in sorted(tpic, key = lambda x:x[0])])

def affcoefnaif(p,i,c):
    return(('%d' % (100 * coefsnaifs[p][i][c]))
           + '%'
           + ((' (±%d' % (100 * ecoefsnaifs[p][i][c]))
              + '%)'
              if ecoefsnaifs[p][i][c] != 0 else '')
    )

f.write(table2([["<h5>Pics prévus pour la France:</h5><p>"
                 + (table2h(tpic) if len(tpic) > 1 else "<i>Pas de pic prévu pour l'instant</i>"),
                 '''<h5>Méthode</h5>
                 La méthode employée est mathématique, fondée sur la géométrie euclidienne et l'algèbre linéaire. Elle est décrite en détails en anglais dans <a href="https://www.medrxiv.org/content/10.1101/2021.04.13.21255418v1">ce preprint sur MedRxiv</a>, et en français <a href="https://hal.archives-ouvertes.fr/hal-03183712v1">ici</a>.<br>
                 Elle procède en trois étapes.
                 <ul><li>La première est de déterminer des décalages de temps entre les données temporelles <a href="#donnees">[0]</a>:
                     <ul><li> les contextes (mesures quotidiennes de mobilité de Google, d'Apple, date, saison, et, pour la France, taux de coronavirus dans les eaux usées),
                         <li> les indicateurs de l'épidémie (hospitalisations, soins critiques, cas positifs, tests, décès),
                         <li> les taux de reproduction des indicateurs (le R effectif <a href="#Reff">[3]</a>).
                     </ul>
                     Cela se fait en calculant les décalages qui maximisent les corrélations entre données.
                    <li> Puis on détermine les transformations linéaires \(C\) qui permettent, à partir des 5 données décalées les plus corrélées \(A\), d'obtenir les R effectifs \(B\) avec des erreurs \(||AC-B||\) minimales  (qui s'avèrent être de l'ordre de 5%, sauf pour le taux de positivité des tests, pour lequel l'erreur est variable mais supérieure à 10%).
                    <li> Enfin on utilise ces transformations linéaires pour prévoir les valeurs futures des R effectifs, puis, par intégration discrète, des indicateurs de l'épidémie.</ul>

                 On a adapté cette méthode pour tenir compte de la proportion de la population qui est vaccinée, ainsi que de la proportion estimée de la population qui a été en contact avec le virus. Le principe est de diviser le R réel par la proportion de la population qui n'est ni vaccinée, ni n'a été infectée, de réaliser les prévisions avec ces valeurs de R (comme si la population était entièrement naive face au virus), puis de multiplier par la même proportion pour obtenir la prévision réelle.<br>
                 Néanmoins, une proportion des personnes vaccinées peut contracter la maladie, de même qu'une proportion des personnes déjà infectées par le passé.<br>
                 Ainsi, par exemple, on considère qu'en France, '''
                 + '''sont protégés de l'hospitalisation en soins intensifs '''
                 + affcoefnaif('France','Ricu_patients',"coefvaccinsc")
                 + ''' des vaccinés avec 2 doses, '''
                 + affcoefnaif('France','Ricu_patients',"coefvaccinsc3")
                 + ''' des vaccinés avec 3 doses, '''
                 + affcoefnaif('France','Ricu_patients',"coefinfectes")
                 + ''' des infectés  <a href="#scauchemez">[1],</a>'''
                 + ''' et que '''
                 + affcoefnaif('France','Ricu_patients',"coefinfectes_vaccines")
                 + ''' des infectés sont vaccinés.<br>'''
                 + '''Toujours pour la France, sont protégés de l'infection '''
                 + affcoefnaif('France','Rnew_cases',"coefvaccinsc")
                 + ''' des vaccinés avec 2 doses, '''
                 + affcoefnaif('France','Rnew_cases',"coefvaccinsc3")
                 + ''' des vaccinés avec 3 doses.<br>'''
                 + '''Ces proportions sont déterminées de manière à minimiser l'erreur sur les prévisions de 1 à 6 semaines des nombres de patients en soins intensifs ou hospitalisés sur les '''
                 + str(mois_evaluation) + ''' derniers mois.<p>
                 La première (vaccinés avec 2 doses) est assez précisément déterminée, mais les 3 autres beaucoup moins.<p>'''
                 + '''La méthode produit alors des prévisions en France à 14 jours avec une erreur moyenne sur les '''
                 + str(mois_evaluation) + ''' derniers mois de <b>'''
                 + ("%2.f" % erreurs_moy['France']['icu_patients'][14])
                 + '''%</b> pour le nombre de patients en soins intensifs, et de <b>'''
                 + ("%2.f" % erreurs_moy['France']['hosp_patients'][14])
                 + '''%</b> pour le nombre de patients hospitalisés.<br>'''
                 + '''Pour 1 mois les erreurs sont de <b>'''
                 + ("%2.f" % erreurs_moy['France']['icu_patients'][28])
                 + '''%</b> et de <b>'''
                 + ("%2.f" % erreurs_moy['France']['hosp_patients'][28])
                 + '''%</b>.<p>''',
                 '''<h5>Method</h5>
<p>Our method is mathematical, based on Euclidean geometry and linear algebra. It is described in detail in English in <a href="https://www.google.com/url?q=https://www.medrxiv.org/content/10.1101/2021.04.13.21255418v1&amp;sa=D&amp;source=editors&amp;ust=1641231981708000&amp;usg=AOvVaw0R5gZeFI5DePU_U7Rrwccp">&nbsp;</a><a href="https://www.google.com/url?q=https://www.medrxiv.org/content/10.1101/2021.04.13.21255418v1&amp;sa=D&amp;source=editors&amp;ust=1641231981709000&amp;usg=AOvVaw0YHdU8v7D_P491pyUJxrVy">this preprint on MedRxiv</a> , and in French <a href="https://www.google.com/url?q=https://hal.archives-ouvertes.fr/hal-03183712v1&amp;sa=D&amp;source=editors&amp;ust=1641231981709000&amp;usg=AOvVaw3LORZsUxCql_PtQb7caR-L">&nbsp;</a><a href="https://www.google.com/url?q=https://hal.archives-ouvertes.fr/hal-03183712v1&amp;sa=D&amp;source=editors&amp;ust=1641231981709000&amp;usg=AOvVaw3LORZsUxCql_PtQb7caR-L">here</a>.</p><p>It proceeds in three stages.</p>

<ul>
<li>The first is to determine time offsets between the temporal data. <a href="https://www.google.com/url?q=https://cp.lpmib.fr/medias/covid19/_synthese.html%23donnees&amp;sa=D&amp;source=editors&amp;ust=1641231981710000&amp;usg=AOvVaw1UFXxYuIBR9-jGTD-GgtH7">&nbsp;</a><a href="https://www.google.com/url?q=https://cp.lpmib.fr/medias/covid19/_synthese.html%23donnees&amp;sa=D&amp;source=editors&amp;ust=1641231981710000&amp;usg=AOvVaw1UFXxYuIBR9-jGTD-GgtH7">[0]</a>:

                 <ul>
                 <li class="c4 c9 li-bullet-0">contexts (daily mobility measurements from Google, Apple, date, season, and, for France, coronavirus rate in wastewater),</li>
                 <li class="c4 c9 li-bullet-0">epidemic indicators (hospitalizations, intensive care unit, cases, tests, deaths),</li>
                 <li class="c4 c9 li-bullet-0">the reproduction rates of the indicators (the effective R <a href="https://www.google.com/url?q=https://cp.lpmib.fr/medias/covid19/_synthese.html%23Reff&amp;sa=D&amp;source=editors&amp;ust=1641231981710000&amp;usg=AOvVaw39pLVGiaRzWygpKaMMr7-U">&nbsp;</a><a href="https://www.google.com/url?q=https://cp.lpmib.fr/medias/covid19/_synthese.html%23Reff&amp;sa=D&amp;source=editors&amp;ust=1641231981711000&amp;usg=AOvVaw3LHsqM1P5T5IgBhc1DksoD">[3]</a>).</li>
                 </ul>

This is done by calculating the offsets that maximize the correlations between data.</li>

<li>Then we determine the linear transformations \(C\) which make it possible, from the 5 most correlated shifted data \(A\), to obtain the effective Rs \(B\) with minimal errors \(||AC-B||\) (which turn out to be of the order of 5%, except for the positivity rate of the tests, for which the error is variable but greater than 10%).</li>

<li>Finally, these linear transformations are used to predict the future values &#8203;&#8203;of the effective Rs, then, by discrete integration, the epidemic indicators.</li>
</ul>

<p>This method has been adapted to take into account the proportion of the population that is vaccinated, as well as the estimated proportion of the population that has been in contact with the virus. The principle is to divide the real R by the proportion of the population which is neither vaccinated nor has been infected, to carry out the forecasts with these values &#8203;&#8203;of R (as if the population were entirely naive in the face of the virus), then multiply by the same proportion to get the actual forecast.</p><p>However, a proportion of people vaccinated can contract the disease, as well as a proportion of people already infected in the past.</p>'''
                 '''Thus, for example, it is considered that in France '''
                 + affcoefnaif('France','Rnew_cases',"coefvaccinsc")
                 + ''' of those vaccinated with 2 doses are protected, '''
                 + affcoefnaif('France','Rnew_cases',"coefvaccinsc3")
                 + ''' of those vaccinated with 3 doses, '''
                 + affcoefnaif('France','Rnew_cases',"coefinfectes")
                 + ''' of those infected <a href="#scauchemez">[1],</a>. '''
                 + ''' and that '''
                 + affcoefnaif('France','Rnew_cases',"coefinfectes_vaccines")
                 + ''' of those infected are vaccinated.<br>'''
                 '''<p>These proportions are determined in such a way as to minimize the error on the 1 to 6 week forecasts of the numbers of patients in intensive care or hospitalized over the last 10 months.The first is quite precisely determined, but the other 3 much less.</p>'''
                 '''<p>The method then produces forecasts in France at 14 days with an average error over the last '''
                 + str(mois_evaluation) + ''' months of <b>'''
                 + ("%2.f" % erreurs_moy['France']['icu_patients'][14])
                 + '''%</b> for the number of patients in intensive care, and of <b>'''
                 + ("%2.f" % erreurs_moy['France']['hosp_patients'][14])
                 + '''%</b> for the number of hospitalized patients.</p>'''
                 '''<p>For 1 month the errors are <b>'''
                 + ("%2.f" % erreurs_moy['France']['icu_patients'][28])
                 + '''%</b> and <b>'''
                 + ("%2.f" % erreurs_moy['France']['hosp_patients'][28])
                 + '''%</b>.<p>'''
                 ]]))

f.write("<h5>La suite présente les prévisions pour la France et d'autres pays.</h5>")

f.write("<p>Prévisions: courbes en pointillé. Les courbes supérieures et inférieures correspondent aux prévisions avec une erreur prise comme l'erreur moyenne (quadratique) sur les prévisions passées.<br>"
        + "Courbe en trait plein: données réelles.</p>")

######################################################################
p = 'France'
f.write('<a id="previsions ' + p + '"></a>' + vspace
        + '<h3>' + p + '</h3>')
ecritpreverreurs_navbar(f,p)
ecritprevpassees_navbar(f,p)
ecritpics_navbar(f,p)
if False:
    f.write('<h4> Erreurs de prévision moyennes sur 10 mois.</h4>'
            + table2h(tableauxerreurs[p]))

f.write("<a id=\"mobilitesgoogleauxusees\"></a>"
        + table2([["<h4>Coronavirus dans les eaux usées, taux du variant omicron suspecté</h4>",
                   '<h4>Données de mobilité de Google ' + ('et Apple' if useapple else '') + '</h4>'],
                  [tabs([(traduitindicateur(nom),
                          charthtml(leschartsfun(p,('données',nom))))
                         for nom in nomseauxusees]
                        +
                        [(traduitindicateur(nom),
                          charthtml(leschartsfun(p,('données',nom))))
                         for nom in nomsvariants]),
                   navbar_dropdown('mobilitesgoogle' + p,
                                   p,
                                   [(traduitindicateur(nom), [('donnée', leschartsfun(p,('données',nom))['href'])])
                                    for nom in mobilitesgoogle + (mobilitesapple if useapple else [])])
                   ]]))

#################### les autres pays
def html_pays(p,f):
    if scores[p] < 70:
        f.write('<a id="previsions ' + p + '"></a>' + vspace
                + '<h3>' + p + '</h3>')
        ecritpreverreurs_navbar(f,p)
        ecritprevpassees_navbar(f,p)
        try:
            ecritpics_navbar(f,p)
        except:
            print('pas de pics prévus')
        if False:
            f.write('<h4> Erreurs de prévision moyennes sur 10 mois.</h4>'
                    + table2h(tableauxerreurs[p]))
    else:
        f.write('<a id="previsions ' + p + '"></a>' + vspace
                + '<h3>' + p + '</h3>')
        f.write('Erreurs de prévision > 40%.<p>')

if True:
    f.write('<h2>Régions de France</h2>')
    for p in sorted(nomsregions,
                    key = lambda p: scores[p]):
        html_pays(p,f)
    f.write('<h2>Pays du monde</h2>')
    for p in sorted([p for p in paysok if p != 'France' and p not in nomsregions],
                    key = lambda p: scores[p]):
        html_pays(p,f)
    
####################

f.write(vspace
        + '<h3>Notes</h3>'
        + '<a id="donnees">[0] </a>'
        + '<ul><li><a href="https://github.com/owid/covid-19-data">https://github.com/owid/covid-19-data</a>'
        + '<li><a href="https://www.data.gouv.fr">https://www.data.gouv.fr</a>'
        + '<li><a href="https://www.data.gouv.fr/fr/datasets/surveillance-du-sars-cov-2-dans-les-eaux-usees-1/#_">données Obepine</a>'
        + '<li><a href="https://www.google.com/covid19/mobility/">https://www.google.com/covid19/mobility</a>'
        + '<li><a href="https://covid19.apple.com/mobility">https://covid19.apple.com/mobility</a>'
        + '</ul>'
        + '<a id="scauchemez">[1] </a>'
        + "Le nombre de personnes infectées est estimée suivant l'idée développée par <a href=\"https://modelisation-covid19.pasteur.fr/realtime-analysis/infected-population/\">l'équipe de Simon Cauchemez</a><p>"
        + '<a id="precision">[2] </a>'
        + " Il s'agit de la moyenne des erreurs sur les prévisions de 1 à 6 semaines, durant 10 mois, pour les hospitalisations et les soins critiques.<p>"
        + '<a id="Reff">[3] </a>'
        + " Le R réel est le taux de reproduction mesuré d'un indicateur \(f\), il est donné par la formule \(R = e^{s \\frac {f'} f}\), avec \(s\) l'intervalle sériel, pris égal à \(4.11\). Si R est supérieur à 1, l'épidémie progresse, s'il est inférieur à 1, elle régresse. La valeur du paramètre  \(s\) n'est pas vraiment importante de ce point de vue, du moment qu'elle est fixe et positive. Un maximum de R correspond à un changement dans la dynamique épidémique: sa croissance commence à ralentir jusqu'à ce qu'un pic soit atteint, et que l'épidémie régresse.<p>"
)

f.write(''' 
 <script type="text/javascript">
      google.charts.load('current', {'packages':['corechart'], 'language': 'fr'});
      google.charts.setOnLoadCallback(drawChart);

      function drawChart() {
      ''' + '\n'.join(toutlescriptdescharts) + '''
      }
 </script>''')

f.write(fin1)
f.close()
print('synthèse écrite')

if uploader:
    os.system('scp previsions_quotidiennes/_synthese.html lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19/_synthese.html')
    os.system('cd ' + DIRPREV + ';tar -cvf groba.tgz charts >/dev/null')
    os.system('scp ' + DIRPREV + 'groba.tgz lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19/groba.tgz >/dev/null')
    os.system("ssh lpmib@ssh-lpmib.alwaysdata.net 'cd testdjango2/testdjango2/medias/covid19 && tar -xvf groba.tgz >/dev/null'")
    os.system('cd ..')
    print('synthèse uploadée')


######################################################################
# ajustement de coefvaccinc, coefinfectes et coefinfectes_vaccines avec 100 tests de prévisions dans le passé

# on teste avec paystest
paystest = paysok #['France']
paysok = paystest

def arrondi(x): return(int(x*100)/100)

if testcoefvac:
    res = dict([(p,[]) for p in paystest])
    mcoefsnaifs = copy.deepcopy(coefsnaifs)
    ecoefsnaifs = copy.deepcopy(coefsnaifs)
    '''
----- France -----------------------------------------------------------------
1014 score 18.03
scores des 5 premiers: ['17.43', '17.43', '17.44', '17.45', '17.46']
premier
Rnew_cases
   coefinfectes  4 58
   coefinfectes_vaccines 96 29
   coefvaccinsc 52  2
   coefvaccinsc3 90 20
Rnew_deaths
   coefinfectes  4 58
   coefinfectes_vaccines 96 29
   coefvaccinsc 56 12
   coefvaccinsc3 15 39
Ricu_patients
   coefinfectes  4 58
   coefinfectes_vaccines 96 29
   coefvaccinsc 70  8
   coefvaccinsc3 10 32
Rhosp_patients
   coefinfectes  4 58
   coefinfectes_vaccines 96 29
   coefvaccinsc 70  4
   coefvaccinsc3 87  7
Rpositive_rate
   coefinfectes  4 58
   coefinfectes_vaccines 96 29
   coefvaccinsc 76  6
   coefvaccinsc3 40 22
'''
    for kk in range(10000):
        t0 = time.time()
        for p in paystest:
            ci = random.random()
            civ = random.random()
            for i in indicateurstest:
                coefsnaifs[p][i]["coefvaccinsc"] = random.random() 
                coefsnaifs[p][i]["coefvaccinsc3"] = random.random()
                coefsnaifs[p][i]["coefinfectes"] = ci
                coefsnaifs[p][i]["coefinfectes_vaccines"] = civ
        p = 'France'
        coefsnaifs[p]['Ricu_patients']["coefvaccinsc"] = 0.70
        coefsnaifs[p]['Rhosp_patients']["coefvaccinsc"] = 0.70
        coefsnaifs[p]['Rpositive_rate']["coefvaccinsc"] = 0.76
        coefsnaifs[p]['Rnew_cases']["coefvaccinsc"] = 0.52
        # previsions à 28 jours dans le passé
        fin = num_de_jour(aujourdhui) 
        erreurs = dict([(p,dict([(i,dict([(k*7,[]) for k in range(1,nsemaines + 1)]))
                                 for i in nomsindicateursRp(p) + nomsindicateursReffp(p)]))
                        for p in paystest]) # paysok
        for j in range(fin - duree_evaluation, fin + 1,7):
            #print(j)
            CPj = coefficients_prevision(M,jourdepartprevision = j)
            P1 = prevoit_indicateursR(M,CPj,jourdepartprevision = j)
            for p in erreurs:
                for i in erreurs[p]:
                    jdep,jfin,v = M[p][i]
                    for k in range(1,nsemaines + 1):
                        if j+k*7 <= jfin:
                            erreurs[p][i][7*k].append(abs((P1[p][i][2][j+k*7 -jdep] - v[j+k*7 -jdep])
                                                          / v[j+k*7 -jdep]))
        for p in paystest:
            for i in erreurs[p]:
                for k in range(1,nsemaines + 1):
                    erreurs[p][i][7*k] = np.sqrt(np.mean(np.array(erreurs[p][i][7*k])**2))
        #######################
        for p in paystest[::-1]:
            res[p].append((score(p,erreurs), copy.deepcopy(coefsnaifs[p])))
            res1 = sorted(res[p],key = lambda x: x[0])
            print('-----', p, '-----------------------------------------------------------------')
            print(kk, 'score','%3.2f' % score(p,erreurs))
            print('scores des 5 premiers:',["%2.3f" % s for s,x in res1[:5]])
            print('premier')
            for i in indicateurstest:
                print(i) #,res1[0][1][i])
                for c in sorted(res1[0][1][i]):
                    v = [x[i][c] for s,x in res1[:1]] #res1[:5]]
                    v5 = [x[i][c] for s,x in res1[:5]]
                    mv = np.mean(v5)
                    ev = np.std(v5)
                    #ev = np.sqrt(np.mean([(v5j - mv) ** 2 for v5j in v5]))
                    mcoefsnaifs[p][i][c] = mv
                    ecoefsnaifs[p][i][c] = ev
                    if p == 'France':
                        ecoefsnaifs[p]['Ricu_patients']["coefvaccinsc"] = 0.08
                        ecoefsnaifs[p]['Rhosp_patients']["coefvaccinsc"] = 0.04
                        ecoefsnaifs[p]['Rpositive_rate']["coefvaccinsc"] = 0.06
                        ecoefsnaifs[p]['Rnew_cases']["coefvaccinsc"] = 0.02
                    print('  ',c, "%2.0f" % (100 * mv), "%2.0f" % (100 * ecoefsnaifs[p][i][c]))
        f = open('bestcoefsnaifs.pickle','wb')
        pickle.dump((mcoefsnaifs,ecoefsnaifs),f)
        f.close()
        print("%3.2fs" % (time.time() - t0))

    os.system('cp bestcoefsnaifs.pickle ' + DIRCOVID19 + 'coefsnaifs.pickle')

######################################################################
# test de coefficients de vaccination avec score restrient aux erreurs de prev lineaire sur les R

if testcoefvacR:
    paystest = ['France']
    res = dict([(p,[]) for p in paystest])
    mcoefsnaifs = copy.deepcopy(coefsnaifs)
    ecoefsnaifs = copy.deepcopy(coefsnaifs)
    for kk in range(1000000):
        t0 = time.time()
        for p in paystest:
            ci = random.random()
            civ = random.random()
            for i in indicateurstest:
                coefsnaifs[p][i]["coefvaccinsc"] = random.random() 
                coefsnaifs[p][i]["coefvaccinsc3"] = random.random()
                coefsnaifs[p][i]["coefinfectes"] = ci
                coefsnaifs[p][i]["coefinfectes_vaccines"] = civ
        CPj = coefficients_prevision(M,jourdepartprevision = jaujourdhui + 1)
        P1 = prevoit_indicateursR(M,CPj,jourdepartprevision = jaujourdhui + 1)
        erreurs = dict([(p,[]) for p in paystest])
        for p in paystest:
            for i in nomsindicateursRp(p):
                jdep,jfin,v = P1[p][i]
                C,dependances,e = CPj[p][i]
                erreurs[p].append(e)
            erreurs[p] = np.sqrt(np.mean([e**2 for e in erreurs[p]]))
        #######################
        for p in paystest[::-1]:
            res[p].append((erreurs[p], copy.deepcopy(coefsnaifs[p])))
            res1 = sorted(res[p],key = lambda x: x[0])
            print('-----', p, '-----------------------------------------------------------------')
            print(kk, 'erreur','%3.4f' % erreurs[p])
            print('erreur des 5 premiers:',["%2.4f" % s for s,x in res1[:5]])
            print('premier')
            for i in indicateurstest:
                print(i) #,res1[0][1][i])
                for c in sorted(res1[0][1][i]):
                    v5 = [x[i][c] for s,x in res1[:5]]
                    mv = np.mean(v5)
                    ev = np.std(v5)
                    mcoefsnaifs[p][i][c] = mv
                    ecoefsnaifs[p][i][c] = ev
                    print('  ',c, "%2.0f" % (100 * mv), "%2.0f" % (100 * ecoefsnaifs[p][i][c]))
        f = open('bestcoefsnaifsR.pickle','wb')
        pickle.dump((mcoefsnaifs,ecoefsnaifs),f)
        f.close()
        print("%3.3fs" % (time.time() - t0))

    os.system('cp bestcoefsnaifsR.pickle ' + DIRCOVID19 + 'coefsnaifsR.pickle')

'''
----- France -----------------------------------------------------------------
99999 erreur 0.0810
erreur des 5 premiers: ['0.0625', '0.0639', '0.0643', '0.0645', '0.0653']
premier
Rnew_cases
   coefinfectes 41 28
   coefinfectes_vaccines 60 23
   coefvaccinsc 64  8
   coefvaccinsc3 45 30
Rnew_deaths
   coefinfectes 41 28
   coefinfectes_vaccines 60 23
   coefvaccinsc 60 10
   coefvaccinsc3 63 16
Ricu_patients
   coefinfectes 41 28
   coefinfectes_vaccines 60 23
   coefvaccinsc 62  6
   coefvaccinsc3 85  6
Rhosp_patients
   coefinfectes 41 28
   coefinfectes_vaccines 60 23
   coefvaccinsc 60  9
   coefvaccinsc3 83 17
Rpositive_rate
   coefinfectes 41 28
   coefinfectes_vaccines 60 23
   coefvaccinsc 61  3
   coefvaccinsc3 70 14
0.471s
----- France -----------------------------------------------------------------
414554 erreur 0.0753
erreur des 5 premiers: ['0.0600', '0.0615', '0.0617', '0.0619', '0.0624']
premier
Rnew_cases
   coefinfectes 41 32
   coefinfectes_vaccines 44 31
   coefvaccinsc 69  4
   coefvaccinsc3 34 20
Rnew_deaths
   coefinfectes 41 32
   coefinfectes_vaccines 44 31
   coefvaccinsc 67  2
   coefvaccinsc3 73 25
Ricu_patients
   coefinfectes 41 32
   coefinfectes_vaccines 44 31
   coefvaccinsc 64  3
   coefvaccinsc3 84 11
Rhosp_patients
   coefinfectes 41 32
   coefinfectes_vaccines 44 31
   coefvaccinsc 63  4
   coefvaccinsc3 69 12
Rpositive_rate
   coefinfectes 41 32
   coefinfectes_vaccines 44 31
   coefvaccinsc 63  4
   coefvaccinsc3 65  9
1.090s

'''
######################################################################
'''
f = open('bestcoefsnaifs.pickle','rb')
mcoefsnaifs,ecoefsnaifs = pickle.load(f)
f.close()
p = 'France'
print('France')
for i in indicateurstest:
    print(i)
    for c in coefsnaifs[p][i]:
        print('  ',c, "%2.0f" % (100 * mcoefsnaifs[p][i][c]), "%2.0f" % (100 * ecoefsnaifs[p][i][c]))
'''


