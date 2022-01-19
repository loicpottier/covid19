# Prévision des indicateurs du covid pour des pays du monde
# python3 monde.py new

from outils import *

recharger = False #True
touslespays = True #False
touteslesregions = True
useapple = True
useomicron = False # le 7 janv omicron apparait dans les dépendances de hosp_patients et fout la merde: score 77...
testcoefvac = False
testcoefvacR = False
uploader = True
lissagesindicateurs = 4 #4
lissagesmobilites = 2 #4
lissagesReff = 2 # 12

if len(sys.argv) > 1 and sys.argv[1] == 'new':
    recharger = True
    print('on charge les dernières données disponibles')

if len(sys.argv) > 1 and sys.argv[1] == 'newtout':
    recharger = True
    touslespays = True
    print('on charge les dernières données disponibles, tous les pays')

if len(sys.argv) > 1 and sys.argv[1] == 'newtest':
    recharger = True
    testcoefvac = True
    uploader = False
    print('on charge les dernières données disponibles, tests pour coefvac, sans upload')

if len(sys.argv) > 1 and sys.argv[1] == 'test':
    testcoefvac = True
    uploader = False
    print('tests pour coefvac, sans upload')

if len(sys.argv) > 1 and sys.argv[1] == 'testR':
    testcoefvacR = True
    uploader = False
    print('tests pour coefvac avec R, sans upload')

if len(sys.argv) > 1 and sys.argv[1] == 'local':
    uploader = False
    print('données locales, sans upload')

jourfin = num_de_jour('2022-06-30')

# toutes les données du monde, jour par jour, pays par pays
if recharger:
    try:
        print('chargement des données du covid dans le monde')
        #https://github.com/owid/covid-19-data/raw/master/public/data/owid-covid-data.csv
        world = chargecsv('https://github.com/owid/covid-19-data/raw/master/public/data/owid-covid-data.csv',
                          sep = ',')[:-1]
        f = open(DIRCOVID19 + 'world.pickle','wb')
        pickle.dump(world,f)
        f.close()
    except:
        print('****************** echec ****************************')

f = open(DIRCOVID19 + 'world.pickle','rb')
world = pickle.load(f)
f.close()

def get(x,champ):
    return(x[world[0].index(champ)])

x = world[[i for i in range(len(world)) if 'France' in world[i]][-6]]
x1 = world[[i for i in range(len(world)) if 'France' in world[i]][-5]]

['FRA', 'Europe', 'France', '2021-11-04', '7292220.0', '9397.0', '6276.429', '118804.0', '46.0', '34.714', '107930.154', '139.082', '92.896', '1758.386', '0.681', '0.514', '1.15', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '99714581.0', '51331330.0', '46075779.0', '3350002.0', '163493.0', '117725.0', '147.58', '75.97', '68.2', '4.96', '1742.0', '66.67', '67564251.0', '122.578', '42.0', '19.718', '13.079', '38605.671', '', '86.06', '4.77', '30.1', '35.6', '', '5.98', '82.66', '0.901', '', '', '', '']

# tous les noms des données
world[0]
['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths', 'new_deaths', 'new_deaths_smoothed', 'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million', 'total_deaths_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million', 'reproduction_rate', 'icu_patients', 'icu_patients_per_million', 'hosp_patients', 'hosp_patients_per_million', 'weekly_icu_admissions', 'weekly_icu_admissions_per_million', 'weekly_hosp_admissions', 'weekly_hosp_admissions_per_million', 'new_tests', 'total_tests', 'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'positive_rate', 'tests_per_case', 'tests_units', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters', 'new_vaccinations', 'new_vaccinations_smoothed', 'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred', 'new_vaccinations_smoothed_per_million', 'stringency_index', 'population', 'population_density', 'median_age', 'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index', 'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative', 'excess_mortality', 'excess_mortality_cumulative_per_million']

# sélection de données
worldind = ['location', 'date',
            'new_cases', 'new_deaths',
            #'new_cases_per_million', 'new_deaths_per_million',
            'reproduction_rate',
            'icu_patients', #'icu_patients_per_million',
            'hosp_patients', #'hosp_patients_per_million',
            'weekly_icu_admissions', #'weekly_icu_admissions_per_million',
            'weekly_hosp_admissions', #'weekly_hosp_admissions_per_million', # tous les 7 jours sinon rien
            'new_tests', #'new_tests_per_thousand',
            'positive_rate',
            #'people_vaccinated', 'people_fully_vaccinated',
            'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred',
            #'total_boosters',
            'total_boosters_per_hundred',
            #'population', 'population_density',
            #'median_age', 'aged_65_older', 'aged_70_older', 'gdp_per_capita',
            #'cardiovasc_death_rate', 'diabetes_prevalence',
            #'female_smokers', 'male_smokers',
            #'hospital_beds_per_thousand', 'life_expectancy',
            #'human_development_index'
]

lespays = sorted(list(set([get(x,'location') for x in world[1:]])))
print('nombre de pays:',len(lespays))# 237 pays

# indices des indicateurs
indind = [world[0].index(ind) for ind in worldind]
# indice du nom des pays
indpays = world[0].index('location')

popmem = {}

def population(p):
    if p in popmem:
        return(popmem[p])
    else:
        x = world[[i for i in range(len(world)) if p in world[i]][-6]]
        pop = float(get(x,'population'))
        popmem[p] = pop
        return(pop)

######################################################################
# traitement des données des pays
indicateurspays = dict([(p,[[x[i] for i in indind]
                            for x in world if x[indpays] == p])
                        for p in lespays])

def to_float(x):
    try:
        return(float(x))
    except:
        return(None)

# convertit en float, jours en nombres, vire le nom du pays
for p in indicateurspays:
    indicateurspays[p] = [[num_de_jour(x[1])] + [to_float(v) for v in x[2:]]
                   for x in indicateurspays[p]]

nomsindicateurs = worldind[1:]

######################################################################
# extrapolation par jour des données hebdomadaires

def extrapole_semaine_cumul(lval, manquante = None):
    lval = [manquante if x == None else x for x in lval]
    for i,s in enumerate(lval):
        if i >= 7 and s != manquante:
            v0 = lval[i-7]
            if v0 != None:
                for k in range(1,8):
                    lval[i-7 + k] = v0 + k * (s - 7 * v0) / 28 # je sais plus comment j'ai trouvé ca...
    return(lval)

for indic in ['weekly_hosp_admissions','weekly_icu_admissions']: #'weekly_hosp_admissions_per_million']:
    ind = nomsindicateurs.index(indic)
    for p in lespays:
        lval = [x[ind] for x in indicateurspays[p]]
        if set(lval) != {None}:
            #print(p,indic)
            lvalext = extrapole_semaine_cumul(lval)
            for i,x in enumerate(indicateurspays[p]):
                x[ind] = lvalext[i]

######################################################################
# les indicateurs de chaque pays: jour de début, jour de fin, liste des valeurs.

paysindic = dict([(p,{}) for p in lespays])

for p in lespays:
    for ind,indicateur in enumerate(nomsindicateurs):
        lv = [x[ind] for x in indicateurspays[p]]
        if set(lv) != {None}:
            j = 0
            while lv[j] == None:
                j = j + 1
            jmin = j
            j = len(lv) - 1
            while lv[j] == None:
                j = j - 1
            jmax = j
            jdep = indicateurspays[p][0][0]
            paysindic[p][indicateur] = (jdep + jmin, jdep + jmax, lv[jmin:jmax+1])

######################################################################
# extrapoler linéairement les donnees manquantes (seulement dans les trous, ne touche pas au bouts)
# lissage

def extrapole_manquantes(ld, manquante = 0, valmanquante = 0):
    # on extrapole linéairement les données manquantes, ne touche pas au bouts
    ld1 = ld[:]
    for j in range(len(ld)):
        if ld[j] == manquante:
            suiv = valmanquante
            jsuiv = j
            try:
                for u in range(j+1,len(ld)):
                    if ld[u] != manquante:
                        suiv = ld[u]
                        jsuiv = u
                        raise NameError('ok')
            except:
                pass
            prev = valmanquante
            jprev = j
            try:
                for u in range(j-1,-1,-1):
                    if ld[u] != manquante:
                        prev = ld[u]
                        jprev = u
                        raise NameError('ok')
            except:
                pass
            if jprev == j or jsuiv == j:
                pass # en fait valmanquante ne sert pas
            else:
                ld1[j] = prev + (j - jprev)/(jsuiv-jprev) * (suiv - prev)
            #print(j,jprev,jsuiv,suiv,prev,ld[j])
    return(ld1)

extrapole_manquantes([None,None,None,1,2,None,None,5,None,None], manquante = None, valmanquante = 0)

for p in paysindic:
    for i in paysindic[p]:
        jdep,jfin,v = paysindic[p][i]
        v = extrapole_manquantes(v, manquante = None, valmanquante = 0)
        paysindic[p][i] = (jdep,jfin,lissage(v,7,repete = lissagesindicateurs))

# extrapoler
def extrapole_lin_amorti(v,n): # n nombre de jours de prévision
    l = len(v)
    v1 = list(v)
    vmin,vmax = min(v1),max(v1)
    for j in range(n):
        d = (1.1*(n-j)/n # >1 : amortisement
             + 1*(1-(n-j)/n)
             )
        dv = ((v1[-1] - np.mean(v1[-14:])) / (14/2)) / d
        a = v1[-1] + dv
        v1.append(a)
    return(np.array(v1))

# extrapoler les mobilites
def extrapole_mobilite(v,n): # n nombre de jours de prévision
    l = len(v)
    v1 = list(v)
    m0 = np.mean(v[-395:-365])
    m1 = np.mean(v[-30:])
    for j in range(n):
        a = v1[-365] + m1 - m0
        v1.append(a)
    return(np.array(v1))

nomsvaccination = ['people_vaccinated_per_hundred',
                   'people_fully_vaccinated_per_hundred',
                   'total_boosters_per_hundred']

######################################################################
# données de la France par data.gouv.fr: plus récentes
# https://www.data.gouv.fr/fr/pages/donnees-coronavirus/
# https://www.data.gouv.fr/fr/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/

if recharger:
    #donnees-hospitalieres-covid19-2022-01-18-19h06.csv 
    csv1 = chargecsv('https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7',
                     zip = False,
                     sep = ';')
    #donnees-hospitalieres-nouveaux-covid19-2022-01-18-19h06.csv
    csv2 = chargecsv('https://www.data.gouv.fr/fr/datasets/r/6fadff46-9efd-4c53-942a-54aca783c30c',
                     zip = False,
                     sep = ';')
    # https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-resultats-des-tests-virologiques-covid-19/
    csv3 = chargecsv('https://www.data.gouv.fr/fr/datasets/r/dd0de5d9-b5a5-4503-930a-7b08dc0adc7c',
                     zip = False,
                     sep = ';')
    csv3dep = chargecsv('https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675',
                        zip = False,
                        sep = ';')
    #https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-indicateurs-sur-les-mutations/
    csv4 = chargecsv('https://www.data.gouv.fr/fr/datasets/r/848debc4-0e42-4e3b-a176-afc285ed5401',
                     zip = False,
                     sep = ';')
    print('donnees France ok', flush=True)
    csv5 = chargecsv('https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD',
                     zip = False,
                     sep = ',')
    csv6 = chargecsv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv',
                     zip = False,
                     sep = ',')
    csv7 = chargecsv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv',
                     zip = False,
                     sep = ',')
    print('donnees US ok', flush=True)
    f = open(DIRCOVID19 + 'hospfrance.pickle','wb')
    pickle.dump((csv1,csv2,csv3,csv3dep,csv4,csv5,csv6,csv7),f)
    f.close()

f = open(DIRCOVID19 + 'hospfrance.pickle','rb')
csv1,csv2,csv3,csv3dep,csv4,csv5,csv6,csv7 = pickle.load(f)
f.close()

####################
# regions et departements

f = open('regions.csv','r')
s = f.read()
f.close()

ls = [x.split('\t') for x in s.split('\n')]
nomregion_num_deps = {}
dep_region = {}
dep_nom = {}
for x in ls:
    d = x[0]
    if d[1] not in 'ab':
        dep_nom[int(d)] = x[1]
    r = x[3] #nom de region
    if r not in nomregion_num_deps and r != 'Corse':
        nomregion_num_deps[r] = [int(x[2])] # numero de region
    try:
        nomregion_num_deps[r].append(int(x[0])) # le departement
        dep_region[int(x[0])] = int(x[2]) # numero de region
    except:
        pass # la corse...

region_deps = dict([(nomregion_num_deps[r][0],nomregion_num_deps[r][1:]) for r in nomregion_num_deps]) # numeros des regions et departements
region_nom = dict([(nomregion_num_deps[x][0], x)
                   for x in nomregion_num_deps])

for reg in region_deps:
    nomreg = region_nom[reg]
    popmem[nomreg] = 0
    for dep in region_deps[reg]:
        popmem[nomreg] += population_dep[dep]

#departements = [dep for reg in region_deps for dep in region_deps[reg]]

for reg in region_deps:
    nomreg = region_nom[reg]
    paysindic[nomreg] = {}

###################
data = csv1[1:-1]
jours = sorted(set([x[2] for x in data]))

def vdata(k):
    vjours = dict([(j,[]) for j in jours])
    for x in data:
        if x[1] == '"0"': # tous les sexes
            j = x[2]
            vjours[j].append(to_float(x[k]))
    v = np.array([sum(vjours[j]) for j in sorted(vjours)])
    return(v)

hosp = vdata(3)
rea = vdata(4)

jdep = num_de_jour(jours[0])
jfin = num_de_jour(jours[-1])

paysindic['France']['icu_patients'] = (jdep,jfin,rea)
paysindic['France']['hosp_patients'] = (jdep,jfin,hosp)

def vdata_reg(k):
    vjours = dict([(reg, dict([(j,[]) for j in jours]))
                   for reg in region_deps])
    for x in data:
        if x[1] == '"0"': # tous les sexes
            j = x[2]
            dep = x[0].replace('"','')
            if dep not in ['2A','2B']:
                dep = int(dep)
                if dep in dep_region:
                    vjours[dep_region[dep]][j].append(to_float(x[k]))
    v = dict([(reg, np.array([sum(vjours[reg][j])
                              for j in sorted(vjours[reg])]))
              for reg in region_deps])
    return(v)

hosp_reg = vdata_reg(3)
rea_reg = vdata_reg(4)

for reg in region_deps:
    nomreg = region_nom[reg]
    paysindic[nomreg]['icu_patients'] = (jdep,jfin,rea_reg[reg])
    paysindic[nomreg]['hosp_patients'] = (jdep,jfin,hosp_reg[reg])

################### 
data = csv2[1:-1]
jours = sorted(set([x[1] for x in data]))

def vdata(k):
    vjours = dict([(j,[]) for j in jours])
    for x in data:
        j = x[1]
        vjours[j].append(to_float(x[k]))
    v = np.array([sum(vjours[j]) for j in sorted(vjours)])
    return(v)

hosp = vdata(2)
rea = vdata(3)
deces = vdata(4)

jdep = num_de_jour(jours[0])
jfin = num_de_jour(jours[-1])

paysindic['France']['weekly_hosp_admissions'] = (jdep,jfin,hosp)
paysindic['France']['weekly_icu_admissions'] = (jdep,jfin,rea)
paysindic['France']['new_deaths'] = (jdep,jfin,deces)

def vdata_reg(k):
    vjours = dict([(reg, dict([(j,[]) for j in jours]))
                   for reg in region_deps])
    for x in data:
        j = x[1]
        dep = x[0].replace('"','')
        if dep not in ['2A','2B']:
            dep = int(dep)
            if dep in dep_region:
                vjours[dep_region[dep]][j].append(to_float(x[k]))
    v = dict([(reg, np.array([sum(vjours[reg][j])
                              for j in sorted(vjours[reg])]))
              for reg in region_deps])
    return(v)

hosp_reg = vdata_reg(2)
rea_reg = vdata_reg(3)
deces_reg = vdata_reg(4)

for reg in region_deps:
    nomreg = region_nom[reg]
    paysindic[nomreg]['weekly_hosp_admissions'] = (jdep,jfin,hosp_reg[reg])
    paysindic[nomreg]['weekly_icu_admissions'] = (jdep,jfin,rea_reg[reg])
    paysindic[nomreg]['new_deaths'] = (jdep,jfin,deces_reg[reg])

###################
# global france
data = csv3[1:-1]
jours = sorted(set([x[1] for x in data]))

def vdata(k):
    vjours = {}
    for x in data:
        if x[8] == '0': # tous les ages
            j = x[1]
            vjours[j] = to_float(x[k])
    v = np.array([vjours[j] for j in sorted(vjours)])
    return(v)

pos = vdata(4)
test = vdata(7)
jdep = num_de_jour(jours[0])
jfin = num_de_jour(jours[-1])

#paysindic['France']['new_cases'] = (jdep,jfin,pos) # en retard par rapport à owid
paysindic['France']['new_tests'] = (jdep,jfin,test)
paysindic['France']['positive_rate'] = (jdep,jfin,pos/test)

# par départements, on regroupe par régions
data = csv3dep[1:-1]
jours = sorted(set([x[1] for x in data]))

def vdata_reg(k):
    vjours = dict([(reg, dict([(j,[]) for j in jours]))
                   for reg in region_deps])
    for x in data:
        if x[4] == '0': # tous les ages
            j = x[1]
            dep = x[0].replace('"','')
            if dep not in ['2A','2B']:
                dep = int(dep)
                if dep in dep_region:
                    vjours[dep_region[dep]][j].append(to_float(x[k]))
    v = dict([(reg, np.array([sum(vjours[reg][j])
                              for j in sorted(vjours[reg])]))
              for reg in region_deps])
    return(v)

pos_reg = vdata_reg(2)
test_reg = vdata_reg(3)

for reg in region_deps:
    nomreg = region_nom[reg]
    paysindic[nomreg]['new_cases'] = (jdep,jfin,pos_reg[reg]) # en retard par rapport à owid
    paysindic[nomreg]['new_tests'] = (jdep,jfin,test_reg[reg])
    paysindic[nomreg]['positive_rate'] = (jdep,jfin,pos_reg[reg]/test_reg[reg])

###################
# variants, mutation C = L452R portée par Delta, pas portée par Omicron
data = csv4[1:-1]

jours = sorted(set([x[1][11:] for x in data]))

def vdata(k):
    vjours = {}
    for x in data:
        j = x[1][11:]
        vjours[j] = to_float(x[k])
    v = np.array([vjours[j] for j in sorted(vjours)])
    return(v)

jdep = num_de_jour(jours[0]) - 3 # car les données sont des moyennes sur la semaine passée
jfin = num_de_jour(jours[-1]) - 3
L452R = vdata(-1)
L452R = np.concatenate([np.zeros(jdep - 863),L452R])
omin = np.max(L452R)
np.argmax(L452R)
L452R[:np.argmax(L452R)] = omin
L452R = (100 - L452R) - (100 - omin)


if useomicron:
    paysindic['France']["taux du variant omicron suspecté"] = (863,jfin,L452R)
    nomsvariants = ["taux du variant omicron suspecté"]
else:
    nomsvariants = []

#########
# lissage
for i in ['icu_patients','hosp_patients',
          'weekly_hosp_admissions','weekly_icu_admissions','new_deaths',
          'new_cases','new_tests','positive_rate',
]:
    for p in ['France'] + [region_nom[r] for r in region_nom]:
        if i in paysindic[p]:
            jdep,jfin,v = paysindic[p][i]
            paysindic[p][i] = (jdep,jfin,lissage(v,7,repete = lissagesindicateurs))

######################################################################
# US

def toint(x):
    try:
        return(int(x))
    except:
        return(0)

data = {}

for x in csv5[1:-1]:
    h = (toint(x[csv5[0].index('previous_day_admission_adult_covid_confirmed')])
         + toint(x[csv5[0].index('previous_day_admission_pediatric_covid_confirmed')]))
    j = x[1].replace('/','-')
    if j not in data:
        data[j] = h
    else:
        data[j] = data[j] + h

jours = sorted(data)
jdep = num_de_jour(jours[0]) - 1
jfin = num_de_jour(jours[-1]) - 1
hosp = np.array([data[j] for j in jours])
hosp = lissage(hosp,7,repete = lissagesindicateurs)
paysindic['United States']['daily_hosp_admissions'] = (jdep,jfin,hosp)
''' bizarre, pas pareil que 
https://github.com/reichlab/covid19-forecast-hub/blob/master/data-truth/truth-Incident%20Hospitalizations.csv
2021-12-18,US,US,7358
2021-12-25,US,US,9395

plt.plot(hosp[926 - 730:] * 8) # environ 8 fois plus faible car les covid suspectes ne sont pas comptes
plt.plot(paysindic['United States']['daily_hosp_admissions'][2])
plt.grid();plt.show()
'''

def ajoute0(x):
    return(x if int(x) >= 10 else '0' + x)

jdep = csv6[0][11].split('/')
jdep = '-'.join(['20' + jdep[2],ajoute0(jdep[0]),ajoute0(jdep[1])])
jfin = csv6[0][-1].split('/')
jfin = '-'.join(['20' + jfin[2],ajoute0(jfin[0]),ajoute0(jfin[1])])
jdep = num_de_jour(jdep) #- 1
jfin = num_de_jour(jfin) #- 1

casescum = np.zeros(jfin - jdep + 1)
for x in csv6[1:-1]:
    for j in range(jfin - jdep + 1):
        casescum[j] = casescum[j] + int(x[j - (jfin - jdep + 1)])

cases = casescum[1:] - casescum[:-1]
jdep = jdep + 1
cases = lissage(cases,7,repete = lissagesindicateurs)
paysindic['United States']['new_cases'] = (jdep,jfin,cases)

jdep = csv7[0][12].split('/')
jdep = '-'.join(['20' + jdep[2],ajoute0(jdep[0]),ajoute0(jdep[1])])
jfin = csv7[0][-1].split('/')
jfin = '-'.join(['20' + jfin[2],ajoute0(jfin[0]),ajoute0(jfin[1])])
jdep = num_de_jour(jdep)# - 1
jfin = num_de_jour(jfin)# - 1

deathscum = np.zeros(jfin - jdep + 1)
for x in csv7[1:-1]:
    for j in range(jfin - jdep + 1):
        deathscum[j] = deathscum[j] + int(x[j - (jfin - jdep + 1)])

deaths = deathscum[1:] - deathscum[:-1]
jdep = jdep + 1
deaths = lissage(deaths,7,repete = lissagesindicateurs)
paysindic['United States']['new_deaths'] = (jdep,jfin,deaths)

print('donnees US chargees',flush = True)

'''
plt.plot(deaths[789-752:])
plt.plot(paysindic['United States']['new_deaths'][2])
plt.grid();plt.show()
'''

######################################################################
# donnees google
######################################################################

nomsmobilites = ['date',
                 'retail_and_recreation_percent_change_from_baseline',
                 'grocery_and_pharmacy_percent_change_from_baseline',
                 'parks_percent_change_from_baseline',
                 'transit_stations_percent_change_from_baseline',
                 'workplaces_percent_change_from_baseline',
                 'residential_percent_change_from_baseline']

if recharger:
    print('chargement des données de mobilité de Google')
    # https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv
    response = urllib.request.urlopen('https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip')
    encoding = response.info().get_content_charset() or "utf-8"
    data = response.read()      # a `bytes` object
    f = open('Region_Mobility_Report_CSVs.zip','wb')
    f.write(data)
    f.close()
    import zipfile
    with zipfile.ZipFile('Region_Mobility_Report_CSVs.zip', 'r') as zip_ref:
        zip_ref.extractall('mobilite_google')
    mobilitepays = {}
    for f in [f for f in os.listdir('mobilite_google') if '2020' in f]:
        ls2020 = loadcsv('mobilite_google/' + f, sep = ',', end = '\n')[:-1]
        ls2021 = loadcsv('mobilite_google/' + f.replace('2020','2021'), sep = ',', end = '\n')[:-1]
        ls2022 = loadcsv('mobilite_google/' + f.replace('2020','2022'), sep = ',', end = '\n')[:-1]
        p = ls2020[1][1]
        if p in lespays:
            #print(p)
            google = ls2020 + ls2021[1:] + ls2022[1:]
            mobilitepays[p] = [[x[google[0].index(m)] for m in nomsmobilites]
                               for x in google[1:]
                               if x[google[0].index('sub_region_1')] == '']
    f = open(DIRCOVID19 + 'mobilitepays.pickle','wb')
    pickle.dump(mobilitepays,f)
    
    f.close()
    #os.system('rm -rf mobilite_google Region_Mobility_Report_CSVs.zip')

f = open(DIRCOVID19 + 'mobilitepays.pickle','rb')
mobilitepays = pickle.load(f)
f.close()

paysok = [p for p in sorted(mobilitepays)
          if '' not in mobilitepays[p][-1]]

# convertit en float, jours en nombres
for p in paysok:
    mobilitepays[p] = [[num_de_jour(x[0])] + [to_float(v) for v in x[1:]]
                       for x in mobilitepays[p]]

jourminmob = min([min([x[0] for x in mobilitepays[p]]) for p in paysok])
jourmaxmob = max([max([x[0] for x in mobilitepays[p]]) for p in paysok])

mobilitepays = dict([(p,mobilitepays[p])
                     for p in paysok
                     if max([x[0] for x in mobilitepays[p]]) == jourmaxmob
                     and min([x[0] for x in mobilitepays[p]]) == jourminmob])

######################################################################
# données apple
######################################################################

if recharger and useapple:
    print("chargement des données de mobilité d'Apple")
    now = time.localtime(time.time())
    def sdep(d):
        dep = str(d)
        if d < 10:
            dep = '0' + str(d)
        return(dep)
    j = str(now.tm_year) + '-' + sdep(now.tm_mon) + '-' + sdep(now.tm_mday)
    s = chargejson('https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json',zip = False)
    basePath = s['basePath']
    csvPath = s['regions']['en-us']['csvPath']
    data = None
    while data == None and j != '2020-11-01':
        try:
            #print(j)
            # attention cette url change régulièrement,
            # il faut inspecter le reseau dans le navigateur pour voir cette url...
            # ou charger le json ci-dessus! si son url ne change pas...
            # (trouvé dans le source javascript .js  avec le lien dans la page https://covid19.apple.com/mobility)
            #url = 'https://covid19-static.cdn-apple.com/covid19-mobility-data/2023HotfixDev9/v3/en-us/applemobilitytrends-' + j + '.csv'
            url = 'https://covid19-static.cdn-apple.com' + basePath + csvPath
            print('url apple:',url)
            data = chargecsv(url,zip = False,sep = ',')
            #print('data ok')
        except:
            print(j)
            j = jour_de_num[num_de_jour(j)-1]
    f = open(DIRCOVID19 + 'mobilitepaysapple.pickle','wb')
    pickle.dump(data,f)
    f.close()

if useapple:
    f = open(DIRCOVID19 + 'mobilitepaysapple.pickle','rb')
    data = pickle.load(f)
    f.close()    
    jours = data[0][6:]
    # données des pays
    datapays = [x for x in data if x[0] == 'country/region']
    #les pays avec les 3 données
    vapple = ['driving', 'walking', 'transit']
    lp = [[x[1] for x in datapays if v in x]
          for v in vapple]
    paysok = [p for p in lp[0] if p in lp[1] and p in lp[2] and p in paysok]
    print('données google:',jour_de_num[jourmaxmob])
    print('données apple:',jours[-1])
    jourminmob = max(jourminmob,num_de_jour(jours[0]))
    jourmaxmob = min(jourmaxmob,num_de_jour(jours[-1]))
    p = 'France'
    dp = [[x for x in datapays if v in x and p in x][0][6:]
          for v in vapple]
    for p in paysok:
        dp = [[x for x in datapays if v in x and p in x][0][6:][jourminmob - num_de_jour(jours[0]):
                                                                jourmaxmob - num_de_jour(jours[0]) + 1]
              for v in vapple]
        dp = [[to_float(v) for v in x] for x in dp]
        ndate = nomsmobilites.index('date')
        for x in mobilitepays[p]:
            j = x[ndate]
            if j < jourminmob:
                for v in vapple:
                    x.append(None)
            elif j > jourmaxmob:
                for iv,v in enumerate(vapple): # on reprend la valeur la plus récente
                    x.append(dp[iv][num_de_jour(jours[-1]) - jourminmob])
            else:
                for iv,v in enumerate(vapple):
                    x.append(dp[iv][j - jourminmob])
    mobilitesapple = vapple
    nomsmobilites = nomsmobilites + mobilitesapple

######################################################################
# lissage des mobilites

for p in paysok:
    mobp = mobilitepays[p]
    for im,m in enumerate(nomsmobilites):
        if m != 'date':
            lval = extrapole_manquantes([x[im] for x in mobp], manquante = None)
            lval = lissage(lval, 7, repete = lissagesmobilites)
            for i,x in enumerate(mobp):
                x[im] = lval[i]

######################################################################
# eaux usées en France
######################################################################
# eaux usées obepine
# https://www.data.gouv.fr/fr/datasets/surveillance-du-sars-cov-2-dans-les-eaux-usees-1/#_
# 05-01-2022 stations, et on moyenne
if recharger:
    print("chargement des eaux usées")
    url_eaux_usees = 'https://www.data.gouv.fr/fr/datasets/r/a632790a-8d03-41da-842b-8aac673fe278'
    csv = chargecsv(url_eaux_usees, sep = r'[,;]')[:-1]
    csv = [x if x[0] != '' else x[1:]
           for x in csv]
    lieux_pos = list(set([(x[0],x[5],x[6]) for x in csv[1:]]))
    lieux = sorted([x[0] for x in lieux_pos])
    lieux_jours = dict([(l,[]) for l in lieux])
    for x in csv[1:]:
        lieux_jours[x[0]].append(num_de_jour(x[2]))
    jourmax = int(round(np.mean([max(lieux_jours[l]) for l in lieux])))
    print('dernier jour des eaux usées:',jour_de_num[jourmax])
    # dernieres données il y 12 jours en moyenne: on prolongera lineairement a j - 5
    jours = [j for j in range(num_de_jour('2020-05-01'),min(jaujourdhui, max(jaujourdhui - 5, jourmax + 7)) + 1)]
    njours = len(jours)
    val_lieu = np.zeros((len(lieux),njours,1))
    #['Station', 'Code_Sandre', 'Date', 'Indicateur', 'IPQDE', 'Longitude_station', 'Latitude_station']
    for x in csv[1:]:
        l,_,j,v = x[:4]
        val_lieu[lieux.index(l),num_de_jour(j) - jours[0],0] = float(v)
    #val_lieu[lieux.index('CANNES'),:,0]
    valmin = np.min(val_lieu[:,:,0])
    # on prolonge linéairement jusqu'à jourmax
    for l in range(len(lieux)):
        lv = [j for j in range(njours) if val_lieu[l,j,0] != 0]
        if lv != []:
            jmax = max(lv)
            jmin = min(lv)
            v = val_lieu[l,jmin:jmax+1,0]
            j1 = jmax - jmin
            vm = np.mean(v[j1 - 7:j1 + 1])
            pv = (v[j1] - vm) / 3.5
            prev = [max(valmin,v[j1] + pv * k) for k in range(1,njours - jmax)]
            val_lieu[l,jmax+1:,0] = prev
    val = np.zeros((njours,2))
    # moyenne des lieux où on a des valeurs
    for j in jours:
        jj = j - jours[0]
        lv = [val_lieu[l,jj,0] for l in range(len(lieux))]
        lv = [x for x in lv if x != 0]
        if lv == []:
            vd = 0
        else:
            vd = np.mean(lv)
        val[jj,0] = vd
    val[:,1] = derivee(val[:,0],7)
    f = open(DIRCOVID19 + 'eauxusees.pickle','wb')
    pickle.dump((jours,val),f)
    f.close()

f = open(DIRCOVID19 + 'eauxusees.pickle','rb')
jours,val = pickle.load(f)
f.close()

val1 = val[:,0] #extrapole_lin_amorti(val[:,0], jourfin - jours[-1])
vald = val[:,1] #extrapole_lin_amorti(val[:,1], jourfin - jours[-1])

if True:
    paysindic['France']['eaux usées'] = (jours[0],jours[-1],val1)
    paysindic['France']['dérivée eaux usées'] = (jours[0],jours[-1],vald)
    nomseauxusees = ['eaux usées','dérivée eaux usées']
else:
    nomseauxusees = []

######################################################################
######################################################################
# on dispose maintenant de
# indicateurspays: nomsindicateurs, dates variables
# mobilitepays: nomsmobilites

if  touslespays:
    paysok = [p for p in ['Belgium', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'France', 'Germany', 'Ireland', 'Israel', 'Italy', 'Luxembourg', 'Malta', 'Portugal', 'Slovenia', 'Switzerland', 'United Kingdom', 'United States']
          if p in paysok]
    
else:
    paysok = ['France','Germany','United Kingdom', 'Italy', 'United States']

if touteslesregions:
    nomsregions = sorted([region_nom[reg] for reg in region_nom])
else:
    nomsregions = []

######################################################################
# dictionnaire des vecteurs des mobilites et des indicateurs

M = dict([(p,{}) for p in paysok + nomsregions])

ndate = nomsmobilites.index('date')
for p in paysok:
    mp = mobilitepays[p]
    for im,m in enumerate(nomsmobilites):
        lval = [(x[ndate],x[im])
                for x in mp
                if x[im] != None]
        jdeb = lval[0][0]
        M[p][m] = (jdeb,jdeb + len(lval) - 1,
                   np.array([x[1] for x in lval])) # jdeb est le premier jour du vecteur
    # date
    jdep,jfin,v = M[p]['date']
    M[p]['date'] = (jdep, jourfin, np.concatenate([v,[j for j in range(jfin+1,jourfin+1)]]))

ndate = nomsindicateurs.index('date')
for p in paysok + nomsregions:
    for indic in paysindic[p]:
        M[p][indic] = paysindic[p][indic]

nomsindicateursReff = ['new_cases', 'new_deaths', 'icu_patients',
                       'hosp_patients', 'daily_hosp_admissions','weekly_hosp_admissions',
                       'weekly_icu_admissions','new_tests','positive_rate',
                       ]
nomsindicateursnonR = ['people_vaccinated_per_hundred',
                       'people_fully_vaccinated_per_hundred','total_boosters_per_hundred']

for p in paysok + nomsregions:
    for indic in nomsindicateursReff:
        if indic in M[p]:
            jdeb,jfin,v = M[p][indic]
            rv = r0(np.maximum(lissage(v,7,repete = lissagesReff),0),
                    derive=7,maxR = 3)
            M[p]['R' + indic] = (jdeb,jfin,rv)

nomsindicateursR = ['R' + x for x in nomsindicateursReff]
nomsindicateursM = nomsindicateurs + nomsindicateursR

####################
# mobilites et eaux usees des regions: pour l'instant ce sont les nationales...
if touteslesregions:
    for reg in nomsregions:
        for i in nomsmobilites + nomseauxusees + nomsindicateursnonR:
            M[reg][i] = M['France'][i]

paysok = paysok + nomsregions

####################
# ajouter la valeur 0 pour le passé
for p in paysok:
    for indic in paysindic[p]:
        if indic in ['people_vaccinated_per_hundred',
                     'people_fully_vaccinated_per_hundred','total_boosters_per_hundred']:
            jdep,jfin,v = M[p][indic]
            j = num_de_jour('2020-04-01')
            M[p][indic] = j,jfin,np.concatenate([[0]*(jdep-j),v])

print(paysok)

######################################################################
# on ajoute la saison
jourfinsaisons = num_de_jour('2023-05-31')
saison = [-1]*(jourfinsaisons + 1 - jourminmob)
moissaisons = [['03','04','05'],['06','07','08'],['09','10','11'],['12','01','02']]
nomssaisons = ['printemps','été','automne','hiver']
vectsaisons = [[],[],[],[]]

for s in range(4):
    vs = saison[:]
    for j in range(jourminmob,jourfinsaisons + 1):
        js = jour_de_num[j]
        if js[5:7] in moissaisons[s]:
            vs[j - jourminmob] = 1
        else:
            vs[j - jourminmob] = -1 + random.random() * 0.1 # pour éviter les déterminants nuls
    vs = lissage(vs,31,repete = 6) # attention: longueur impaire pour la fenetre de lissage
    vectsaisons[s] = vs

nomsmobilites = nomsmobilites + nomssaisons
for p in paysok:
    for s in range(4):
        M[p][nomssaisons[s]] = (jourminmob,jourfinsaisons,vectsaisons[s] + 1)

######################################################################
# utilisation des vaccinés

# infectés: récupéré de correlation.py
# au 14 nov 2021: 35.4% des francais
propinfectes = 0.354
# population: 67390000
popfrance = 67390000
# nombre d'hospitalises depuis le debut de la pandémie:
nhosp = sum(M['France']['weekly_hosp_admissions'][2] / 7)
(popfrance * propinfectes) / nhosp # = 329.16098610198742
# infectes = hospitalises * 329
infectes_hopitalises = 329

def calcule_infected(M,p,jourfin,jdepprev  = None):
    pop = population(p)
    if 'weekly_hosp_admissions' in paysindic[p]:
        jdep,jfin,vh = M[p]['weekly_hosp_admissions']
        # lent
        #vi = np.array([(sum(vh[:j - jdep]) / 7 * infectes_hopitalises) / pop
        #               for j in range(jdep,jfin + 1)])
        vi = np.zeros(jfin - jdep + 1)
        vi[0] = vh[0]
        for j in range(1,jfin - jdep + 1):
            vi[j] = vi[j-1] + vh[j]
        vi = (vi / 7 * infectes_hopitalises) / pop
        if jdepprev != None:
            jfin = min(jfin,jdepprev - 1)
        Mvi = (jdep,jfin,vi[:jfin - jdep + 1])
    else:
        Mvi = (jourminmob,jourmaxmob,np.zeros(jourmaxmob-jourminmob+1))
    jdep,jfin,v = Mvi
    v = extrapole_lin_amorti(v,jourfin - jdep + 1 - len(v))
    return((jdep,jourfin,v))

for p in paysok:
    if p in nomsregions:
        M[p]['infected_per_hundred'] = calcule_infected(M,'France',jourfin)
    else:
        M[p]['infected_per_hundred'] = calcule_infected(M,p,jourfin)

nomsmobilites = nomsmobilites # + ['infected_per_hundred']

######################################################################
######################################################################
# Désormais M contient les données des pays: mobilite, indicateurs, Reff des indicateurs
# M[pays][donnée] =(jour début, jour fin, valeurs de la donnée)
######################################################################
######################################################################

print('dates des dernières données pour la France')
for i in sorted(M['France'],key = lambda x: M['France'][x][1]):
    print(M['France'][i][1],jour_de_num[M['France'][i][1]],i)

######################################################################
# correlations, overlap complet: le plus court bouge dans le plus long

def correlate(x,y):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return(np.correlate(x,y,mode = 'valid')) 
    else:
        return(np.correlate(x,y,mode = 'valid')/(nx*ny))

correlate([1,2,3,4],[2,3,4,5])
correlate([1,2,3,4],[1,2,3,4,5])

decmax = 40 # pour les calculs
decmaxaccepte = 30 # au dela, on vire
# x et y vecteurs
# x0 y0 jours de début de x et y
# dépendance de y par rapport à x, donc donne le décalage de y dans le passé qui maximise leur correlation.
def correlation(x,y,x0,y0,decmax = decmax,decmaxaccepte = decmaxaccepte):
    ############ test: on garde la partie la plus récente
    dx = 0 #len(x) // 4
    dy = 0 #len(y) // 4
    x = x[dx:]
    y = y[dy:]
    x0 = x0 + dx
    y0 = y0 + dy
    ############
    x1 = x0 + len(x) - 1
    y1 = y0 + len(y) - 1
    z1 = max(x1,y1)
    z0 = min(x0,y0)
    # vx longueur z1 - z0 + decmax
    # vy longueur z1 - z0 + decmax
    # se terminent le meme jour
    vxm = np.mean(x)
    vym = np.mean(y)
    vx = np.concatenate([np.zeros(max(0,x0-z0 + decmax)),
                         x - vxm,
                         np.zeros(z1-x1)])
    vy = np.concatenate([np.zeros(max(0,y0-z0 + decmax)),
                         y - vym,
                         np.zeros(z1-y1)])
    vvx = vx.flatten()
    vvy = vy.flatten()[decmax:]
    lcsm = correlate(vvx,vvy)
    d = np.argmax(np.abs(lcsm))
    corr = lcsm[d]
    d = decmax - d
    if d == 0 or d > decmaxaccepte:
        d = 0
        corr = lcsm[-1]
    return([d,corr])

######################################################################
# calcul des correlations et des décalages entre
# indicateurs de Reff
# et mobilites ou indicateurs

def nomsindicateursMp(p):
    r = [x for x in nomsindicateursM
         if x in paysindic[p] or x in ['R' + y for y in paysindic[p]]]
    if p == 'France':
        r = r + nomseauxusees + nomsvariants
    return(r)

memnomsindicateursRp = {}
def nomsindicateursRp(p):
    if p in memnomsindicateursRp:
        return(memnomsindicateursRp[p])
    else:
        res = [x for x in nomsindicateursR
               if x in ['R' + y for y in paysindic[p]]]
        memnomsindicateursRp[p] = res
        return(res)

def nomsindicateursReffp(p):
    return([x for x in nomsindicateursReff if x in paysindic[p]])

correlations = dict([(p,dict([(indic,{})
                              for indic in nomsindicateursRp(p)]))
                     for p in paysok])

mincorrelation = 0.034
touteslesdependances = False
eauxuseesdependances = True
variantsdependances = True

for p in paysok:
    for indic in nomsindicateursRp(p):
        y0,y1,y = M[p][indic]
        for m in nomsmobilites:
            x0,x1,x = M[p][m]
            d,c = correlation(x[:jourmaxmob - jourminmob + 1],y,x0,y0)
            if d != 0 and abs(c) > mincorrelation:
                correlations[p][indic][m] = (d,c)
        for indic1 in (([x for x in nomsindicateursMp(p) if x != 'date']
                        if touteslesdependances else nomsindicateursRp(p))
                       + (nomseauxusees if p == 'France' and eauxuseesdependances else [])
                       + (nomsvariants if p == 'France' and variantsdependances else [])):
            x0,x1,x = M[p][indic1]
            d,c = correlation(x,y,x0,y0)
            if indic1 == "taux du variant omicron suspecté":
                print(d,c,indic,indic1)
            if d != 0 and abs(c) > mincorrelation:
                correlations[p][indic][indic1] = (d,c)

# limiter le nombre de correlations
ncorrelationsmax = 5

for p in paysok:
    for i in correlations[p]:
        l = sorted(correlations[p][i],
                   key = lambda i1: - abs(correlations[p][i][i1][1]))
        d = [(i1,correlations[p][i][i1]) for i1 in l]
        correlations[p][i] = dict(d[:ncorrelationsmax])

######################################################################
# calculs des coefficients de prevision

favoriser_jours_recents = False

def extrait_vecteur(jdep,v,j1,j2):
    return(v[j1-jdep:j2-jdep+1])

def matricenorme(d):
    Q = np.zeros((d,d))
    if favoriser_jours_recents:
        for j in range(d):
            Q[j,j] = j**3 + 1 #j**1 + 1 #np.exp(10* j/d) #j**3 + 1
        Q = Q / sum([Q[j,j] for j in range(d)]) * d
    else: # Q = I
        for j in range(d):
            Q[j,j] = 1
    return(Q)

def coefficients_prev_lin(p,i,jourdepartprevision = None):
    jdepi,jfini,vi = M[p][i]
    joursdepart = [jdepi]
    joursfin = [jdepi + len(vi) - 1]
    for x in correlations[p][i]:
        dec,c = correlations[p][i][x]
        jdep,jfin,v = M[p][x]
        joursdepart.append(jdep + dec) #debut de prevision possible
        joursfin.append(jdep + dec + len(v) - 1) #debut de prevision possible
    jourdepprev = max(joursdepart)
    jourfinprev = min(joursfin)
    if jourdepartprevision != None:
        jourfinprev = min(jourfinprev,jourdepartprevision)
    dureeprev = jourfinprev - jourdepprev + 1
    dependances = sorted(correlations[p][i]) # pour éviter que l'ordre change inopinément
    C = np.zeros(len(dependances))
    try:
        L = np.zeros((dureeprev,len(dependances)))
        for ix,x in enumerate(dependances):
            dec,c = correlations[p][i][x]
            jdep,jfin,v = M[p][x]
            L[:,ix] = extrait_vecteur(jdep,v,jourdepprev-dec,jourfinprev-dec)
        H = extrait_vecteur(jdepi,vi,jourdepprev,jourfinprev)
        Q = matricenorme(dureeprev)
        LtQ = np.transpose(L) @ Q
        A = LtQ @ L
        B = LtQ @ H
        e = 0
        try:
            C = np.linalg.inv(A) @ B
            H1 = L @ C - H
            e = np.linalg.norm(H1) / (0.00000001 + np.linalg.norm(H))
        except:
            pass #print('determinant nul',p,i)
        return(C,dependances,e)
    except:
        return(C,dependances,0)

indicateurstest = ['Rnew_cases', 'Rnew_deaths', 'Ricu_patients', 'Rhosp_patients', 'Rpositive_rate']

'''
----- France -----------------------------------------------------------------
546 score 17.02
scores des 5 premiers: ['16.95', '16.95', '16.95', '16.95', '16.95']
premier
Rnew_cases
   coefinfectes 18 12
   coefinfectes_vaccines 49 34
   coefvaccinsc 52  2
   coefvaccinsc3 52 25
Rnew_deaths
   coefinfectes 18 12
   coefinfectes_vaccines 49 34
   coefvaccinsc 28 24
   coefvaccinsc3 47 32
Ricu_patients
   coefinfectes 18 12
   coefinfectes_vaccines 49 34
   coefvaccinsc 70  8
   coefvaccinsc3 34 28
Rhosp_patients
   coefinfectes 18 12
   coefinfectes_vaccines 49 34
   coefvaccinsc 70  4
   coefvaccinsc3 94  5
Rpositive_rate
   coefinfectes 18 12
   coefinfectes_vaccines 49 34
   coefvaccinsc 76  6
   coefvaccinsc3  6  3
'''
if False:
    #f = open(DIRCOVID19 + 'coefsnaifs.pickle','rb')
    f = open('bestcoefsnaifs7.pickle','rb')
    coefsnaifs,ecoefsnaifs = pickle.load(f)
    f.close()
    p = 'France'
    print('France')
    for i in indicateurstest:
        print(i)
        for c in coefsnaifs[p][i]:
            print('  ',c, "%2.0f" % (100 * coefsnaifs[p][i][c]), "%2.0f" % (100 * ecoefsnaifs[p][i][c]))
else:
    coefvaccinsc = 0.68 # proportion de vaccinés qui sont protégés.
    coefvaccinsc3 = 0.90 # proportion de vaccinés avec 3 doses qui sont protégés.
    coefinfectes = 0.58 # proportion d'infectés qui sont protégés.
    coefinfectes_vaccines = 0.45 # proportion d'infectés qui sont vaccinés.
    coefsnaifs = dict([(p,
                        dict([(i,
                               {"coefvaccinsc" : coefvaccinsc,
                                "coefvaccinsc3" : coefvaccinsc3,
                                "coefinfectes" : coefinfectes,
                                "coefinfectes_vaccines" : coefinfectes_vaccines})
                              for i in nomsindicateursR]))
                       for p in paysok])
    ecoefsnaifs = dict([(p,
                         dict([(i,
                                {"coefvaccinsc" : 0,
                                 "coefvaccinsc3" : 0,
                                 "coefinfectes" : 0,
                                 "coefinfectes_vaccines" : 0})
                               for i in nomsindicateursR]))
                        for p in paysok])
    if True: # vient de bestcoefsnaifs7, donne 17.49 au lieu 18.38
        p = 'France'
        coefsnaifs[p]['Ricu_patients']["coefvaccinsc"] = 0.70 #0.63 #0.70
        ecoefsnaifs[p]['Ricu_patients']["coefvaccinsc"] = 0.08 #0.06 #0.08
        coefsnaifs[p]['Rhosp_patients']["coefvaccinsc"] = 0.70 #0.63 #0.70
        ecoefsnaifs[p]['Rhosp_patients']["coefvaccinsc"] = 0.04 #0.03 #0.04
        coefsnaifs[p]['Rhosp_patients']["coefvaccinsc3"] = 0.94
        ecoefsnaifs[p]['Rhosp_patients']["coefvaccinsc3"] = 0.05
        coefsnaifs[p]['Rpositive_rate']["coefvaccinsc"] = 0.76 #0.64 #0.76
        ecoefsnaifs[p]['Rpositive_rate']["coefvaccinsc"] = 0.06 #0.05 #0.06
        coefsnaifs[p]['Rnew_cases']["coefvaccinsc"] = 0.52 #0.72 #0.52
        ecoefsnaifs[p]['Rnew_cases']["coefvaccinsc"] = 0.12 #0.02

def coef_naif(M,p,i,j):
    decal = 17
    jdep1,jfin1,v1 = M[p]['people_vaccinated_per_hundred']
    jdep2,jfin2,v2 = M[p]['infected_per_hundred']
    try:
        jdep3,jfin3,v3 = M[p]['total_boosters_per_hundred']
        if j - decal < jdep3:
            boost = 0
        elif j - decal > jfin3:
            boost = v3[jfin3 - jdep3]
        else:
            boost = v3[j - decal - jdep3]
    except:
        boost = 0
    if j - decal < jdep1:
        vac = 0
    elif j - decal > jfin1:
        vac = v1[jfin1 - jdep1]
    else:
        vac = v1[j - decal - jdep1]
    if j - decal < jdep2:
        inf = 0
    elif j - decal > jfin2:
        inf = v2[jfin2 - jdep2]
    else:
        inf = v2[j - decal - jdep2]
    return(max(0.1,
               1
               # les vaccinés avec 2 doses
               - coefsnaifs[p][i]["coefvaccinsc"] * (vac - boost) / 100. 
               # les vaccinés avec 3 doses
               - coefsnaifs[p][i]["coefvaccinsc3"] * boost / 100. 
                  # les infectés pas vaccines
               - (coefsnaifs[p][i]["coefinfectes"] 
                  * (1 - coefsnaifs[p][i]["coefinfectes_vaccines"]) 
                  * inf / 100.)))

# fait comme si personne n'était vacciné ni n'avait eu le covid
# on multiplie les Reff par population / (population - vaccines_ou_infectes)

def remet_naif(M,p,remet):
    for x in nomsindicateursRp(p): #[y for y in ['R' + z for z in paysindic[p]] if y in nomsindicateursR]:
        x0,x1,vx = M[p][x]
        for j in range(x0,x1+1):
            coef = coef_naif(M,p,x,j)
            if not remet:
                coef = 1 / coef
            vx[j - x0] = vx[j - x0] / coef


def coefficients_prevision(M,jourdepartprevision = None):
    coefficients_prevision = dict([(p,{}) for p in paysok])
    for p in paysok:
        # on remet naif avant de calculer les coef de prevision
        remet_naif(M,p,True)
        for i in correlations[p]:
            try:
                coefficients_prevision[p][i] = coefficients_prev_lin(p,i,jourdepartprevision = jourdepartprevision)
            except:
                print('problème coefficients_prevision avec', p,i,jourdepartprevision)
        # on enlève après
        remet_naif(M,p,False)
    return(coefficients_prevision)

######################################################################
# prevision des indicateurs

# pour l'amortissement des discontinuités réel/prévision
def applatit(c): # c entre 0 et 1
    d = 3
    if c < 1/2:
        return(c**d / (1/2)**d * 1/2)
    else:
        cc = 1 - c
        return(1 - cc**d / (1/2)**d * 1/2)

#plt.plot([applatit(c) for c in [x/100 for x in range(101)]]);plt.grid(); plt.show()

# prevoit pour le pays p, l'indicateur i le jour j
# ne modifie pas M
def prevoit_jour(M,P,CP,p,i,j,jourdepartprevision, trace = False):
    jdepi,jfini,v = P[p][i]
    if i in nomsindicateursR:
        C,dependances,e = CP[p][i]
        L = np.zeros(len(dependances))
        coef = coef_naif(P,p,i,j)
        for ix,x in enumerate(dependances): 
            dec,c = correlations[p][i][x]
            jdepx,jfinx,vx = P[p][x]
            L[ix] = vx[j - dec - jdepx]
            if x in nomsindicateursR:
                L[ix] = L[ix] / coef # on remet naif
        vj = L @ C # valeur prévue pour le jour j
        if i in nomsindicateursR:
            vj = vj * coef # on enleve naif
        if trace: print('-----\n',j,coef,vj,'\n',
                        L, '\n',
                        dependances)
        # lissage de la discontinuite au depart de la prevision, sur dureangle jours
        jdep0,jfin0,v0 = M[p][i]
        jfin0 = min(jfin0, jourdepartprevision - 1)
        jdep,jfin,v = P[p][i]
        dureeangle = 6*7
        pente = (v0[jfin0 - jdep] - v0[jfin0 - jdep - 7]) / 7
        c = 1 if j >= jfin0 + dureeangle else math.sin(math.pi / 2 * (j - jfin0) / dureeangle) # 0 <= c <= 1
        c = applatit(c)
        vj = (1 - c) * (v0[jfin0 - jdep] + pente * (j - jfin0)) + c * vj
    elif i in nomsindicateursReff:
        iR = 'R' + i
        jdepR,_,vR = P[p][iR]
        # f'/ f:
        dvlogdep = math.log(max(0.01,vR[j - jdepR - 1])) / intervalle_seriel
        vj = v[j - jdepi - 1] + dvlogdep * v[j - jdepi - 1]
        if trace: print('-------\n','j',j,'jdepR',jdepR,'jdepi',jdepi,'\n',
                        1+ dvlogdep, vR[j - jdepR - 1],j-jdepi, j-jdepR,'\n',
                        vj,' = ', v[j - jdepi - 1], ' * ', 1+ dvlogdep)
    elif j <= jfini:
        vj = v[j - jdepi]
    else:
        print('on devrait avoir déjà extrapolé')
        vj = v[jfini - jdepi] # dernière valeur connue, mais on devrait avoir déjà extrapolé.
    return(vj)

# prevoit tous les indicateurs R à partir du premier jour où un indicateur est inconnu,
# ou bien d'un jour donné (jourdepartprevision)
# et jusqu'à jourfinprevision
def prevoit_indicateursR(M,CP,jourdepartprevision = jourfin,jourfinprevision = jourfin):
    # on allonge les vecteurs jusqu'à jourfinprevision avec des zéros
    # ou bien avec des extrapolations tangentes amorties
    P = copy.deepcopy(M)
    for p in paysok:
        for i in M[p]:
            jdep,jfin0,v = M[p][i]
            jfin = min(jfin0,jourdepartprevision - 1)
            v = v[:jfin - jdep + 1]
            if i in nomsindicateursReffp(p) +  nomsindicateursRp(p):
                P[p][i] = (jdep,jfin,np.concatenate([v,np.zeros(jourfinprevision - jdep + 1 - len(v))]))
            elif i == 'infected_per_hundred':
                P[p][i] = calcule_infected(P,p,jourfinprevision,jdepprev = jourdepartprevision)
            elif i == 'date':
                P[p][i] = (jdep, jourfinprevision,
                           np.concatenate([v,[j for j in range(jfin + 1, jourfinprevision + 1)]]))
            elif i in nomssaisons:
                pass
            elif i in nomsmobilites:
                P[p][i] = (jdep, jourfinprevision,
                           extrapole_mobilite(v,jourfinprevision - jdep + 1 - len(v)))
            else:
                P[p][i] = (jdep, jourfinprevision,
                           extrapole_lin_amorti(v,jourfinprevision - jdep + 1 - len(v)))
    for p in paysok:
        # départ de la prévision: le premier jour où un indicateurR est inconnu
        jdebutprev = jourdepartprevision
        for i in M[p]:
            jdep,jfin,v = M[p][i] # P[p][i]
            jdebutprev = min(jdebutprev, jfin + 1)
        # decalage au premier jour 
        dvj0 = dict([(i,0) for i in M[p]])
        # prevoit tout, jour par jour, si la valeur n'est pas connue
        for j in range(jdebutprev,jourfinprevision+1):
            for i in ([x for x in M[p] if x in nomsindicateursRp(p)]
                      + [x for x in M[p] if x not in nomsindicateursRp(p)]):
                jdep0,jfin0,v0 = M[p][i]
                jfin0 = min(jfin0, jourdepartprevision - 1)
                jdep,jfin,v = P[p][i]
                if j == jfin0 and i in nomsindicateursRp(p):
                    dvj0[i] = P[p][i][2][j - jdep] - prevoit_jour(M,P,CP,p,i,j,jourdepartprevision)
                if j > jfin0: #jourdepartprevision : # valeur non connue (ou déjà prévue)
                    vj = prevoit_jour(M,P,CP,p,i,j,jourdepartprevision) + dvj0[i]
                    if i in nomsindicateursReffp(p) +  nomsindicateursRp(p):
                        vj = max(vj,0)
                    v[j - jdep] = vj
                    P[p][i] = jdep,max(j,jfin),v # mise a jour des bornes
                    #if p == 'France' and i == 'Rnew_cases': print(j,'valeur:',vj)
    return(P)

######################################################################
# affiche les corrélations

def affiche_correlations(p):
    print('-'*70)
    print("correlations " + p )
    for i0 in sorted(correlations[p]):
        print('-------------------')
        print(i0)
        l = sorted(correlations[p][i0],
                   key = lambda i: - abs(correlations[p][i0][i][1])) 
        for i in l:
            d,c = correlations[p][i0][i]
            print("%2d %3.2f %s" % (d,c,i))
