# Prévision des indicateurs du covid pour des pays du monde
# python3 monde.py new

from outils import *

recharger = False #True
touslespays = False
useapple = True
testcoefvac = False
uploader = True

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
    print('on charge les dernières données disponibles, tests pour coefvac')

if len(sys.argv) > 1 and sys.argv[1] == 'local':
    uploader = False
    print('données locales, sans upload')

jourfin = num_de_jour('2022-02-28')

# toutes les données du monde, jour par jour, pays par pays
if recharger:
    try:
        print('chargement des données du covid dans le monde')
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
            'people_vaccinated', 'people_fully_vaccinated',
            'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred',
            'total_boosters', 'total_boosters_per_hundred',
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
                    lval[i-7 + k] = v0 + k * (s - 7 * v0) / 28
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

lissagesindicateurs = 4

for p in paysindic:
    for i in paysindic[p]:
        jdep,jfin,v = paysindic[p][i]
        v = extrapole_manquantes(v, manquante = None, valmanquante = 0)
        paysindic[p][i] = (jdep,jfin,lissage(v,7,repete = lissagesindicateurs))

# extrapoler les vaccinations
def extrapole_lin_amorti(v,n): # n nombre de jours de prévision
    l = len(v)
    v1 = list(v)
    for j in range(n):
        d = (1.1*(n-j)/n # >1 : amortisement
             + 1*(1-(n-j)/n)
             )
        #dv = ((v1[-1] - np.mean([y for y in v1[-14:]])) / (14/2)) / d
        dv = ((v1[-1] - np.mean(v1[-14:])) / (14/2)) / d
        v1.append(v1[-1] + dv)
    return(np.array(v1))

nomsvaccination = ['people_vaccinated', 'people_fully_vaccinated', 'people_vaccinated_per_hundred',
                   'people_fully_vaccinated_per_hundred', 'total_boosters',
                   'total_boosters_per_hundred']

######################################################################
# données de la France par data.gouv.fr: plus récentes

if recharger:
    csv1 = chargecsv('https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7',
                     zip = False,
                     sep = ';')
    csv2 = chargecsv('https://www.data.gouv.fr/fr/datasets/r/6fadff46-9efd-4c53-942a-54aca783c30c',
                     zip = False,
                     sep = ';')
    # https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-resultats-des-tests-virologiques-covid-19/
    csv3 = chargecsv('https://www.data.gouv.fr/fr/datasets/r/dd0de5d9-b5a5-4503-930a-7b08dc0adc7c',
                     zip = False,
                     sep = ';')
    #https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-indicateurs-sur-les-mutations/
    csv4 = chargecsv('https://www.data.gouv.fr/fr/datasets/r/848debc4-0e42-4e3b-a176-afc285ed5401',
                     zip = False,
                     sep = ';')
    f = open(DIRCOVID19 + 'hospfrance.pickle','wb')
    pickle.dump((csv1,csv2,csv3,csv4),f)
    f.close()

f = open(DIRCOVID19 + 'hospfrance.pickle','rb')
csv1,csv2,csv3,csv4 = pickle.load(f)
f.close()

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

################### 
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


paysindic['France']["taux du variant omicron suspecté"] = (863,jfin,L452R)
nomsvariants = ["taux du variant omicron suspecté"]

#########
# lissage
for i in ['icu_patients','hosp_patients',
          'weekly_hosp_admissions','weekly_icu_admissions','new_deaths',
          'new_cases','new_tests','positive_rate',
]:
    jdep,jfin,v = paysindic['France'][i]
    paysindic['France'][i] = (jdep,jfin,lissage(v,7,repete = lissagesindicateurs))
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
        p = ls2020[1][1]
        if p in lespays:
            #print(p)
            google = ls2020 + ls2021[1:]
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
            #print(url)
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

lissagesmobilites = 2 #4
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
# 15-12-2021 stations, et on moyenne
if recharger:
    print("chargement des eaux usées")
    url_eaux_usees = 'https://www.data.gouv.fr/fr/datasets/r/fcff0415-2b78-47d7-a5c3-fe9512c65824'
    csv = chargecsv(url_eaux_usees, sep = ';')[:-1]
    try:
        lieux_pos = list(set([(x[0],x[5],x[6]) for x in csv[1:]]))
    except:
        print('ah les cons ils ont changé le séparateur...')
        csv = chargecsv(url_eaux_usees, sep = ',')[:-1]
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
# on dispose maintenant de
# indicateurspays: nomsindicateurs, dates variables
# mobilitepays: nomsmobilites

if  touslespays:
    paysok = [p for p in ['Belgium', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'France', 'Germany', 'Ireland', 'Israel', 'Italy', 'Luxembourg', 'Malta', 'Portugal', 'Slovenia', 'Switzerland', 'United Kingdom', 'United States']
          if p in paysok]
else:
    paysok = ['France','Germany','United Kingdom', 'Italy']

######################################################################
# dictionnaire des vecteurs des mobilites et des indicateurs

M = dict([(p,{}) for p in paysok])

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
for p in paysok:
    for indic in paysindic[p]:
        M[p][indic] = paysindic[p][indic]

lissagesReff = 12
nomsindicateursReff = ['new_cases', 'new_deaths', 'icu_patients', 'hosp_patients', 'weekly_hosp_admissions', 'weekly_icu_admissions','new_tests','positive_rate',
                       ]
nomsindicateursnonR = ['people_vaccinated', 'people_fully_vaccinated', 'people_vaccinated_per_hundred','people_fully_vaccinated_per_hundred','total_boosters', 'total_boosters_per_hundred']

for p in paysok:
    indicp = indicateurspays[p]
    for indic in nomsindicateursReff:
        if indic in M[p]:
            jdeb,jfin,v = M[p][indic]
            rv = r0(np.maximum(lissage(v,7,repete = lissagesReff),0),
                    derive=7,maxR = 3)
            M[p]['R' + indic] = (jdeb,jfin,rv)

nomsindicateursR = ['R' + x for x in nomsindicateursReff]
nomsindicateursM = nomsindicateurs + nomsindicateursR

# ajouter la valeur 0 pour le passé
for p in paysok:
    for indic in paysindic[p]:
        if indic in ['people_vaccinated', 'people_fully_vaccinated', 'people_vaccinated_per_hundred',
                     'people_fully_vaccinated_per_hundred','total_boosters', 'total_boosters_per_hundred']:
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

def population(p):
    x = world[[i for i in range(len(world)) if p in world[i]][-6]]
    pop = float(get(x,'population'))
    return(pop)

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

def nomsindicateursRp(p):
    return([x for x in nomsindicateursR
            if x in ['R' + y for y in paysindic[p]]])

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

# déterminé avec 100 calculs de prévisions dans le passé, voir à la fin du fichier.
#15.958633688486707 (0.77, 0.89, 0.83, 0.68)
#15.521863755385837 (0.74, 0.81, 0.52, 0.63)
#15.522096673781599 (0.77, 0.97, 0.16, 0.12)
coefvaccinsc = 0.74 # proportion de vaccinés qui sont protégés.
coefvaccinsc3 = 0.97 # proportion de vaccinés avec 3 doses qui sont protégés.
coefinfectes = 0.16 # proportion d'infectés qui sont protégés.
coefinfectes_vaccines = 0.12 # proportion d'infectés qui sont vaccinés.

def coef_naif(M,p,j):
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
               - coefvaccinsc * (vac - boost) / 100. # les vaccinés avec 2 doses
               - coefvaccinsc3 * boost / 100. # les vaccinés avec 3 doses
               - coefinfectes * (1 - coefinfectes_vaccines) * inf / 100.)) # les infectés pas vaccines

# fait comme si personne n'était vacciné ni n'avait eu le covid
# on multiplie les Reff par population / (population - vaccines_ou_infectes)

def remet_naif(M,p,remet):
    for x in nomsindicateursRp(p): #[y for y in ['R' + z for z in paysindic[p]] if y in nomsindicateursR]:
        x0,x1,vx = M[p][x]
        for j in range(x0,x1+1):
            coef = coef_naif(M,p,j)
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

# prevoit pour le pays p, l'indicateur i le jour j
# ne modifie pas M
def prevoit_jour(M,CP,p,i,j):
    jdepi,jfini,v = M[p][i]
    if i in nomsindicateursR:
        C,dependances,e = CP[p][i]
        L = np.zeros(len(dependances))
        coef = coef_naif(M,p,j)
        for ix,x in enumerate(dependances): 
            dec,c = correlations[p][i][x]
            jdepx,jfinx,vx = M[p][x]
            L[ix] = vx[j - dec - jdepx]
            if x in nomsindicateursR:
                L[ix] = L[ix] / coef # on remet naif
        vj = L @ C # valeur prévue pour le jour j
        if i in nomsindicateursR:
            vj = vj * coef # on enleve naif
    elif i in nomsindicateursReff:
        iR = 'R' + i
        jdepR,_,vR = M[p][iR]
        # f'/ f:
        dvlogdep = math.log(max(0.01,vR[j - jdepR - 1])) / intervalle_seriel
        vj = v[j - jdepi - 1] + dvlogdep * v[j - jdepi - 1]
    elif j <= jfini:
        vj = v[j - jdepi]
    else:
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
            else:
                P[p][i] = (jdep, jourfinprevision,
                           extrapole_lin_amorti(v,jourfinprevision - jdep + 1 - len(v)))
    for p in paysok:
        # départ de la prévision: le premier jour où un indicateurR est inconnu
        jdebutprev = jourdepartprevision
        for i in M[p]:
            jdep,jfin,v = P[p][i]
            jdebutprev = min(jdebutprev, jfin + 1)
        # decalage au premier jour 
        dvj0 = dict([(i,0) for i in M[p]])
        for i in M[p]:
            if i in nomsindicateursRp(p):
                jdep,jfin,v = P[p][i]
                dvj0[i] = P[p][i][2][jdebutprev - 1 - jdep] - prevoit_jour(P,CP,p,i,jdebutprev - 1)
        # prevoit tout, jour par jour, si la valeur n'est pas connue
        for j in range(jdebutprev,jourfinprevision+1):
            for i in M[p]:
                jdep,jfin,v = P[p][i]
                if j > jfin or j >= jourdepartprevision : # valeur non connue (ou déjà prévue)
                    vj = prevoit_jour(P,CP,p,i,j) + dvj0[i]
                    if i in nomsindicateursReffp(p) +  nomsindicateursRp(p):
                        vj = max(vj,0)
                    v[j - jdep] = vj
                    P[p][i] = jdep,max(j,jfin),v # mise a jour des bornes
                    #if p == 'France' and i == 'Rnew_cases': print(j,'valeur:',vj)
            if False and 'weekly_hosp_admissions' in M[p]:
                pop = population(p)
                jdep,jfin,vh = M[p]['weekly_hosp_admissions']
                jdep1,jfin1,vh1 = M[p]['infected_per_hundred']
                if j <= jfin:
                    vh1[j-jdep1] = vh1[j-jdep1-1] + vh[j - jdep] / 7 * infectes_hopitalises / pop
        for i in nomsindicateursRp(p):
            pop = population(p)
            jdep,jfin,v = P[p][i]
            vl = v #vl = lissage(v,7,repete = 6) #repete = 6
            P[p][i] = jdep,jfin,vl
    return(P)

######################################################################
# affiche les corrélations en France

if True:
    print('-'*70)
    print("correlations['France']")
    for i0 in sorted(correlations['France']):
        print('-------------------')
        print(i0)
        l = sorted(correlations['France'][i0],
                   key = lambda i: - abs(correlations['France'][i0][i][1])) 
        for i in l:
            d,c = correlations['France'][i0][i]
            print("%2d %3.2f %s" % (d,c,i))

######################################################################
# prévisions des pays

CPj = coefficients_prevision(M,jourdepartprevision = jaujourdhui + 1)
P1 = prevoit_indicateursR(M,CPj,jourdepartprevision = jaujourdhui + 1)

print('----------- erreurs de prévision France ----------------')
p = 'France'
for i in nomsindicateursR:
    jdep,jfin,v = P1[p][i]
    C,dependances,e = CPj[p][i]
    print('prevision de', i + '.'*(30 - len(i)) , ("%3.0f" % (100*e)) + '%')


print('----------------------------------------------------------------------')

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
                   'Rpositive_rate','Rnew_cases','Rweekly_hosp_admissions', 'Rnew_deaths',
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
                        2*vmax)
        ve = np.array([erreurmoyenne(p,i,j-jf) for j in range(jf, jf + dferreurs)])
        vsup = np.minimum(np.concatenate([vP1[debut:jf - jdep],
                                          np.array([(1 + ve[j-jf] / 100) * vP1[j - jdep]
                                                    for j in range(jf, jf + dferreurs)])]),
                          2*vmax)
        vinf = np.minimum(np.concatenate([vP1[debut:jf - jdep],
                                          np.array([(max(0,1 - ve[j-jf] / 100)) * vP1[j - jdep]
                                                    for j in range(jf, jf + dferreurs)])]),
                          2*vmax)
        lcourbes += [(zipper([jour_de_num[j] for j in range(jdep + debut,jf + df)],
                             v1),
                      jour_de_num[jf], prev)]
        for vp in [vsup, vinf]:
            lcourbes += [(zipper([jour_de_num[j] for j in range(jdep + debut,jf + dferreurs)],
                                 vp),
                          jour_de_num[jf], prev)]
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
        lescharts[p][('prévisions avec erreurs',nomchart)] = trace_charts(lcourbes, titre = titre,
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
                        try:
                            erreurs[p][i][7*k].append(abs((P1[p][i][2][j+k*7 -jdep] - v[j+k*7 -jdep])
                                                          / v[j+k*7 -jdep]))
                        except:
                            print(p,i,j+k*7,jdep,jfin,j,k)
                            erreurs[p][i][7*k].append(1)
    print('fin des prévisions passées')
    return(erreurs,P1s)

traduit = {'hosp_patients': 'nombre de patients hospitalisés',
           'icu_patients' : 'nombre de patients en soins intensifs',
           'new_cases' : 'nombre de nouveaux cas',
           'positive_rate': 'taux de positivité des tests',
           'weekly_hosp_admissions': 'nombre de patients hospitalisés par jour',
           'new_deaths': 'nombre de nouveaux décès',}

dtraduitindicateur = {'hosp_patients': 'hospitalisations',
                      'icu_patients' : 'soins intensifs',
                      'new_cases' : 'nouveaux cas',
                      'positive_rate': 'taux de positivité',
                      'weekly_hosp_admissions': 'nouvelles hospitalisations',
                      'new_deaths': 'nouveaux décès',}

def traduitindicateur(i):
    if i[0] == 'R':
        return('R ' + traduitindicateur(i[1:]))
    elif i in dtraduitindicateur:
        return(dtraduitindicateur[i])
    else:
        return(i)

def trace_previsions_passes(p,indicateurs,P1s,dureefutur = 28):
    for i in indicateurs:
        #debut = 600-duree_evaluation #if i not in nomsindicateursRp(p) else 500-duree_evaluation + 100
        debut = 600-duree_evaluation
        jdep,jfin,v = M[p][i]
        vmax = abs(np.max(v))
        lcourbes = []
        for j in range(fin + 1, fin - duree_evaluation - 1, -7):
            df = 60 if j == fin + 1 else dureefutur
            P1 = P1s[j]
            v1 = np.minimum(P1[p][i][2][debut:j - jdep + df],
                            2*vmax)
            lcourbes += [(zipper([jour_de_num[j] for j in range(jdep + debut,j + df)],
                                 v1),
                          jour_de_num[j], prev)]
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
        lescharts[p][('prévisions passées',nomchart)] = trace_charts(lcourbes, titre = titre,
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
        lescharts[p][('données',nomchart)] = trace_charts(lcourbes, titre = titre)

##########################
# les erreurs de prévision

#erreur_rea_28, erreur_hospi_28, erreur_cases_28 = 0,0,0

erreurs_moy = dict([(p,{}) for p in paysok])

def erreurmoyenne(p,i,j):
    k = j // 7
    dj = j % 7
    e = erreurs_moy[p][i][7*k] * (1 - dj / 7) + erreurs_moy[p][i][7*(k+1)] * dj / 7
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

# moyenne des erreurs pour les indicateurs iscore1 et 2, jusqu'à 4 semaines
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
    for p in erreurs:
        for i in erreurs[p]:
            for k in range(1,nsemaines + 1):
                erreurs[p][i][7*k] = np.sqrt(np.mean(np.array(erreurs[p][i][7*k])**2)) #np.mean(erreurs[p][i][7*k])
    for p in erreurs:
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
           'new_cases','weekly_hosp_admissions','new_deaths',
]
atracer = atracer + ['R' + x for x in atracer]

def atracerp(p):
    return([i for i in atracer if i in M[p]])

for p in paysok:
    print(p)
    trace_previsions(p,atracerp(p),P1)
    if p == 'France':
        trace_previsions_passes(p,
                                atracerp(p),
                                P1s,
                                dureefutur = 30)

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

def charthtml(c):
    return(c['script'] + c['div'])


def ecritpreverreurs(f,p):
    f.write('<a id="previsions ' + p + '"></a>' + vspace
            + '<h3>' + p + '</h3>')
    f.write("<p>Prévisions: courbes en pointillé. Les courbes supérieures et inférieures correspondent aux prévisions avec une erreur prise comme l'erreur moyenne (quadratique) sur les prévisions passées.<br>"
            + "Courbe en trait plein: données réelles.</p>"
            + table2([[tabs([(traduitindicateur(nom),
                              charthtml(leschartsfun(p,('prévisions avec erreurs',nom))))
                             for nom in atracerp(p) if nom[0] != 'R']),
                       tabs([(traduitindicateur(nom),
                              charthtml(leschartsfun(p,('prévisions avec erreurs',nom))))
                             for nom in atracerp(p) if nom[0] == 'R'])]]))

def ecritprevpassees(f,p):
    f.write('<a id="previsions ' + p + '"></a>' + vspace
            + '<h3>' + p + '</h3>')
    f.write('Prévisions passées. Précision moyenne <a href="#precision">[2]</a>: <b>' + ('%3.2f' % scores[p]) + '%</b><p>')
    if True: #scores[p] <= 20:
        f.write("<p>Courbe en trait plein: données réelles.<br>"
                + "Courbes en pointillés: données approximées puis prévues à partir du jour \(j\) "
                + "en utilisant uniquement les données des jours précédant \( j\).</p>"
                + table2([[tabs([(traduitindicateur(nom),
                                  charthtml(leschartsfun(p,('prévisions passées',nom))))
                                 for nom in atracerp(p) if nom[0] != 'R']),
                           tabs([(traduitindicateur(nom),
                                  charthtml(leschartsfun(p,('prévisions passées',nom))))
                                 for nom in atracerp(p) if nom[0] == 'R'])]]))

def chartpicval(p,i):
    i1 = i[1:]
    s = [(j, jour_de_num[lespics[j][p][i1][0]], lespics[j][p][i1][1])
         for j in lespics if i1 in lespics[j][p]]
    if s != []:
        lcourbes2 = [(zipper([jj for (jj,j,v) in s], [float(v) for (jj,j,v) in s]),
                     'nombre de patients', prev)]
        lechart2 = trace_charts(lcourbes2, titre = "Prévision du " +  traduit[i1] + " au pic ")
        return((i1,lechart2))
    else:
        #print('pas de pic déjà prévu pour', i1)
        return((i1,{'script':'','div':'pas de pic prévu'}))

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
    return(trace_charts(lcourbes, titre = "Jours prévu pour les pics"))

def ecritpics(f,p):
    f.write('<a id="pics"></a>'
            + vspace + "<h4>Évolution des pics prévus pour les indicateurs du Covid19 (France).</h4>"
            + "Le pic est déterminé par le premier jour où le R associé atteint la valeur 1 en décroissant.<p>"
            + 'En abcisse: jour où a été effectuée la prévision.<p>'
            + "Les irrégularités ont lieu lorsque des données manquent pour les prévisions (typiquement lorsque les sites où se trouvent les données sont mis à jour ou bien inaccessibles).<p>"
            + table2([['<h5>Jours prévus pour les pics des indicateurs</h5>',
                   '<h5>Valeurs des indicateurs à leurs pics</h5>'],
                  [charthtml(chartpicjours(p,indicateursRpic)),
                   tabs([(traduitindicateur(i),charthtml(ch2))
                         for (i,ch2) in [chartpicval(p,i) for i in indicateursRpic]])]]))

######################################################################
# page de synthèse
# upload la synthèse sur cp.lpmib.fr:
# https://cp.lpmib.fr/medias/covid19/_synthese.html
######################################################################

def leschartsfun(p,x):
    try:
        return(lescharts[p][x])
    except:
        print('problème chart', x)
        return({'script':'','div':''})

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
                     for p in ['France'] + sorted([p for p in paysok if p != 'France'])])
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
        
f.write(table2([["<h5>Pics prévus pour la France:</h5><p>"
                 + (table2h(tpic) if len(tpic) > 1 else "<i>Pas de pic prévu pour l'instant</i>"),
                 '''<h5>Méthode</h5>
                 La méthode employée est mathématique, elle est décrite en détails en anglais dans <a href="https://www.medrxiv.org/content/10.1101/2021.04.13.21255418v1">ce preprint sur MedRxiv</a>, et en français <a href="https://hal.archives-ouvertes.fr/hal-03183712v1">ici</a>.<br>
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
                 Ainsi on considère que sont protégés '''
                 + ('%d' % (100 * coefvaccinsc)) + '''% des vaccinés avec 2 doses, '''
                 + ('%d' % (100 * coefvaccinsc3)) + '''% des vaccinés avec 3 doses, '''
                 + ('%d' % (100 * coefinfectes)) + '''% des infectés <a href="#scauchemez">[1],</a>'''
                 + ''' et que '''
                 + ('%d' % (100 * coefinfectes_vaccines)) + '''% des infectés sont vaccinés.<br>
Ces proportions sont déterminées de manière à minimiser l'erreur sur les prévisions de 1 à 6 semaines des nombres de patients en soins intensifs ou hospitalisés sur les '''
                 + str(mois_evaluation) + ''' derniers mois.<p>
                 La première est assez précisément déterminée, mais les 3 autres beaucoup moins.<p>'''
                 + '''La méthode produit alors des prévisions en France à 14 jours avec une erreur moyenne sur les '''
                 + str(mois_evaluation) + ''' derniers mois de <b>'''
                 + ("%2.f" % erreurs_moy['France']['icu_patients'][14])
                 + '''%</b> pour le nombre de patients en soins intensifs, et de <b>'''
                 + ("%2.f" % erreurs_moy['France']['hosp_patients'][14])
                 + '''%</b> pour le nombre de patients hospitalisés.<br>'''
                 + '''Pour 1 mois les erreurs sont de '''
                 + ("%2.f" % erreurs_moy['France']['icu_patients'][28])
                 + ''' et de '''
                 + ("%2.f" % erreurs_moy['France']['hosp_patients'][28])
                 + '''.<p>'''
                 ]]))

f.write("<h5>La suite présente les prévisions pour la France et d'autres pays.</h5>")

p = 'France'
ecritpreverreurs(f,p)
ecritprevpassees(f,p)
ecritpics(f,p)
f.write('<h4> Erreurs de prévision moyennes sur 10 mois.</h4>'
        + table2h(tableauxerreurs[p]))
####################
f.write("<a id=\"mobilitesgoogleauxusees\"></a>"
        + table2([["<h4>Coronavirus dans les eaux usées, taux du variant omicron suspecté</h4>",
                   '<h4>Données de mobilité de Google ' + ('et Apple' if useapple else '') + '</h4>'],
                  [tabs([(traduitindicateur(nom),
                          leschartsfun(p,('données',nom))['script']
                          + leschartsfun(p,('données',nom))['div'])
                         for nom in nomseauxusees]
                        +
                        [(traduitindicateur(nom),
                          leschartsfun(p,('données',nom))['script']
                          + leschartsfun(p,('données',nom))['div'])
                         for nom in nomsvariants]),
                  tabs([(traduitindicateur(nom),
                         leschartsfun(p,('données',nom))['script']
                         + leschartsfun(p,('données',nom))['div'])
                        for nom in mobilitesgoogle + (mobilitesapple if useapple else [])])]]))

#################### les autres pays
if True:
    for p in sorted([p for p in paysok if p != 'France'],
                    key = lambda p: scores[p]):
        if scores[p] < 40:
            ecritpreverreurs(f,p)
            try:
                ecritpics(f,p)
            except:
                print('pas de pics prévus')
            f.write('<h4> Erreurs de prévision moyennes sur 10 mois.</h4>'
                    + table2h(tableauxerreurs[p]))
        else:
            f.write('<a id="previsions ' + p + '"></a>' + vspace
                    + '<h3>' + p + '</h3>')
            f.write('Erreurs de prévision > 40%.<p>')
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

f.write(fin1)
f.close()
print('synthèse écrite')

if uploader:
    os.system('scp previsions_quotidiennes/_synthese.html lpmib@ssh-lpmib.alwaysdata.net:testdjango2/testdjango2/medias/covid19/_synthese.html')
    print('synthèse uploadée')


######################################################################
# ajustement de coefvaccinc, coefinfectes et coefinfectes_vaccines avec 100 tests de prévisions dans le passé

# on teste avec paystest
paystest = ['France']
paysok = paystest

if not testcoefvac:
    print('Pas de test de coefvaccinc, coefinfectes et coefinfectes_vaccines')
    exit()

res = []
for kk in range(100):
    coefvaccinsc = random.random() #0.40 #random.random()
    coefvaccinsc3 = random.random()
    coefinfectes = random.random()
    coefinfectes_vaccines = random.random()
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
    for p in erreurs:
        for i in erreurs[p]:
            for k in range(1,nsemaines + 1):
                erreurs[p][i][7*k] = sum(erreurs[p][i][7*k])/ len(erreurs[p][i][7*k])
    # les erreurs
    r = []
    for p in erreurs:
        for i in nomsindicateursR + nomsindicateursReff :
            r.append([100*erreurs[p][i][7*k] for k in range(1,nsemaines + 1)])
    #######################
    p = 'France'
    # lindic = [i for i in nomsindicateursR + nomsindicateursReff if i in erreurs[p]]
    res.append((score(p,erreurs),coefvaccinsc,coefvaccinsc3,coefinfectes,coefinfectes_vaccines,r))
    print('-----',kk,'%3.2f' % score(p,erreurs),
          '%3.2f' % coefvaccinsc,'%3.2f' % coefvaccinsc3,'%3.2f' % coefinfectes,'%3.2f' % coefinfectes_vaccines,
          [' '.join(['%3.0f' % x for x in y]) for y in r])

def arrondi(x): return(int(x*100)/100)

res = [(s,arrondi(a),arrondi(b),arrondi(c),arrondi(d),[[arrondi(x) for x in t] for t in y]) for (s,a,b,c,d,y) in res]
print(res)

f = open('resultat','w')
f.write(str(res))
f.close()

import os
import random
from outils import *
############################

f = open('resultat', 'r')
res = eval(f.read())
f.close()

#tri

res1 = sorted(res,key = lambda x: x[0])

for x in res1[:10]: 
    print(x[0], x[1:5])

print('moyenne 10 premiers', [moyenne([x[k] for x in res1[:10]]) for k in range(1,5)])
print('moyenne 5 premiers', [moyenne([x[k] for x in res1[:5]]) for k in range(1,5)])

import matplotlib.pyplot as plt
plt.plot([  sum([y[2] for y in x[5][2:4]]) for x in res1])
plt.grid();plt.show(False)

for k in range(1,nsemaines + 1):
    plt.plot(lissage([x[k] for x in res1],7))

#plt.plot(lissage([x[0] for x in res1],7))
#plt.plot(lissage([moyenne(x[1:3]) for x in res1],7))
plt.grid();plt.show(False)

