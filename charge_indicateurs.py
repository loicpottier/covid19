from outils import *

def get(data,x,champ):
    try:
        return(x[data[0].index(champ)])
    except:
        return(x[data[0].index('"' + champ + '"')].replace('"',''))
      

departements = ['0' + str(x) for x in range(1,10)] + [str(x) for x in range(10,96) if x != 20]

######################################################################
# https://www.data.gouv.fr/fr/pages/donnees-coronavirus
# indicateurs covid
######################################################################
# urgences quotidien
'''
dep	integer	Departement
date_de_passage	string($date)	Date de passage
sursaud_cl_age_corona	integer	Tranche d'âge des patients
nbre_pass_corona	integer	Nombre de passages aux urgences pour suspicion de COVID-19 
nbre_pass_tot	integer	Nombre de passages aux urgences total
nbre_hospit_corona	integer	Nombre d'hospitalisations parmi les passages aux urgences pour suspicion de COVID-19 
nbre_pass_corona_h	integer	Nombre de passages aux urgences pour suspicion de COVID-19 - Hommes
nbre_pass_corona_f	integer	Nombre de passages aux urgences pour suspicion de COVID-19 - Femmes
nbre_pass_tot_h	integer	Nombre de passages aux urgences total - Hommes
nbre_pass_tot_f	integer	Nombre de passages aux urgences total - Femmes
nbre_hospit_corona_h	integer	Nombre d'hospitalisations parmi les passages aux urgences pour suspicion de COVID-19 - Hommes
nbre_hospit_corona_f	integer	Nombre d'hospitalisations parmi les passages aux urgences pour suspicion de COVID-19 - Femmes
nbre_acte_corona	integer	Nombres d'actes médicaux SOS Médecins pour suspicion de COVID-19
nbre_acte_tot	integer	Nombres d'actes médicaux SOS Médecins total
nbre_acte_corona_h	integer	Nombres d'actes médicaux SOS Médecins pour suspicion de COVID-19 - Hommes
nbre_acte_corona_f	integer	Nombres d'actes médicaux SOS Médecins pour suspicion de COVID-19 - Femmes
nbre_acte_tot_h	integer	Nombres d'actes médicaux SOS Médecins total - Hommes
nbre_acte_tot_f	integer	Nombres d'actes médicaux SOS Médecins total - Femmes
'''

csv = chargecsv('https://www.data.gouv.fr/fr/datasets/r/eceb9fb4-3ebc-4da3-828d-f5939712600a',
                zip = False,
                sep = ';')[:-1] # enlever le [''] de la fin

td = dict([(x,[]) for x in departements])
for x in csv:
    dep = get(csv,x,'dep')
    if dep in departements and get(csv,x,'sursaud_cl_age_corona') == '0':
        lv = [get(csv,x,c)
              for c in ['date_de_passage',
                        'nbre_pass_corona',
                        'nbre_hospit_corona',
                        'nbre_acte_corona']]
        td[dep].append(lv)

maxjours = max([len(td[x]) for x in td])
td = dict([(x,td[x]) for x in td
           if len(td[x]) == maxjours and len([y for y in td[x] if '' in y]) == 0])
jours = [x[0] for x in td[list(td)[0]]] 
deps = sorted([x for x in td])

dataurge = {'nom': 'urgences',
            'titre': 'urgences quotidien',
            'dimensions': ['departements', 'jours'],
            'jours': jours,
            'departements': [int(d) for d in deps],
            'valeurs': np.array([[mfloat(x[1]) for x in td[dep]] for dep in deps])}
datahospiurge = {'nom': 'hospitalisation urgences',
                 'titre': 'hospitalisations urgences quotidien',
                 'dimensions': ['departements', 'jours'],
                 'jours': jours,
                 'departements': [int(d) for d in deps],
                 'valeurs': np.array([[mfloat(x[2]) for x in td[dep]] for dep in deps])}
datasosmedecin = {'nom': 'sosmedecin',
                  'titre': 'sosmedecin quotidien',
                  'dimensions': ['departements', 'jours'],
                  'jours': jours,
                  'departements': [int(d) for d in deps],
                  'valeurs': np.array([[mfloat(x[3]) for x in td[dep]] for dep in deps])}

print('urge, hospiurge, sosmedecin ok', jours[-1])
######################################################################
# hospitalieres
'''
dep	integer	Département	1
sexe	integer	Sexe 	0
jour	string($date)	Date de notification
hosp	integer	Nombre de personnes actuellement hospitalisées
rea	integer	Nombre de personnes actuellement en réanimation ou soins intensifs
rad	integer	Nombre cumulé de personnes retournées à domicile
dc	integer	Nombre cumulé de personnes décédées à l'hôpital
'''
csv = chargecsv('https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7',
                      zip = False,
                      sep = ';')

td = dict([(x,[]) for x in departements])
for x in csv:
    dep = get(csv,x,'dep')
    if dep in departements and get(csv,x,'sexe') == '0':
        lv = [get(csv,x,c)
              for c in ['jour',
                        'hosp',
                        'rea',
                        'dc']]
        td[dep].append(lv)

maxjours = max([len(td[x]) for x in td])
td = dict([(x,td[x]) for x in td
           if len(td[x]) == maxjours and len([y for y in td[x] if '' in y]) == 0])
jours = [x[0] for x in td[list(td)[0]]] 
deps = sorted([x for x in td])

datareatot = {'nom': 'réanimations',
            'titre': 'réanimation',
            'dimensions': ['departements', 'jours'],
            'jours': jours,
            'departements': [int(d) for d in deps],
            'valeurs': np.array([[mfloat(x[2]) for x in td[dep]] for dep in deps])}
datahospitot = {'nom': 'hospitalisations',
            'titre': 'hospitalisations',
            'dimensions': ['departements', 'jours'],
            'jours': jours,
            'departements': [int(d) for d in deps],
            'valeurs': np.array([[mfloat(x[1]) for x in td[dep]] for dep in deps])}
datadecestot = {'nom': 'décès',
            'titre': 'décès',
            'dimensions': ['departements', 'jours'],
            'jours': jours,
            'departements': [int(d) for d in deps],
            'valeurs': np.array([[mfloat(x[3]) for x in td[dep]] for dep in deps])}

print('hospitot ok', jours[-1])

######################################################################
# hospitalieres quotidien
'''
dep	integer	Département
jour	string($date)	Date de notification 
incid_hosp	string 	Nombre quotidien de personnes nouvellement hospitalisées
incid_rea	integer	Nombre quotidien de nouvelles admissions en réanimation 
incid_dc	integer	Nombre quotidien de personnes nouvellement décédées
incid_rad	integer	Nombre quotidien de nouveaux retours à domicile 
'''
csv = chargecsv('https://www.data.gouv.fr/fr/datasets/r/6fadff46-9efd-4c53-942a-54aca783c30c',
                      zip = False,
                      sep = ';')

td = dict([(x,[]) for x in departements])
for x in csv:
    dep = get(csv,x,'dep')
    if dep in departements:
        lv = [get(csv,x,c)
              for c in ['jour',
                        'incid_hosp',
                        'incid_rea',
                        'incid_dc']]
        td[dep].append(lv)

maxjours = max([len(td[x]) for x in td])
td = dict([(x,td[x]) for x in td
           if len(td[x]) == maxjours and len([y for y in td[x] if '' in y]) == 0])
jours = [x[0] for x in td[list(td)[0]]] 
deps = sorted([x for x in td])

datahospi = {'nom': 'nouv hospitalisations',
             'titre': 'nouvelles hospitalisations',
             'dimensions': ['departements', 'jours'],
             'jours': jours,
             'departements': [int(d) for d in deps],
             'valeurs': np.array([[mfloat(x[1]) for x in td[dep]] for dep in deps])}
datarea = {'nom': 'nouv réanimations',
             'titre': 'nouvelles réanimations',
             'dimensions': ['departements', 'jours'],
             'jours': jours,
             'departements': [int(d) for d in deps],
             'valeurs': np.array([[mfloat(x[2]) for x in td[dep]] for dep in deps])}
datadeces = {'nom': 'nouv décès',
             'titre': 'nouveaux décès',
             'dimensions': ['departements', 'jours'],
             'jours': jours,
             'departements': [int(d) for d in deps],
             'valeurs': np.array([[mfloat(x[3]) for x in td[dep]] for dep in deps])}

print('hospi ok', jours[-1])
######################################################################
# hospitalisation classes d'ages, par régions
'''
reg	integer	Region								
cl_age90	integer	Classe age 								
jour	string($date)	Date de notification 							
hosp	integer	Nombre de personnes actuellement hospitalisées
rea	integer	Nombre de personnes actuellement en réanimation ou soins intensifs
rad	integer	Nombre cumulé de personnes retournées à domicile		
dc	integer	Nombre cumulé de personnes décédées
'''
csv = chargecsv('https://www.data.gouv.fr/fr/datasets/r/08c18e08-6780-452d-9b8c-ae244ad529b3',
                         zip = False,
                         sep = ';')

datahospiage = {}
regions = ['11','24','27','28','32','44','52','53','75','76','84','93']

deps = sorted([x for x in depregion])
depsind = dict([(dep,k) for (k,dep) in enumerate(deps)])

for age in ['0'] + [str(x)+'9' for x in range(9)]+['90']: # 0 c est le total des tous les ages
    tr = dict([(x,[]) for x in regions])
    for x in csv:
        reg = get(csv,x,'reg')
        if reg in regions and get(csv,x,'cl_age90') == age:
            lv = [get(csv,x,c)
                  for c in ['jour',
                            'hosp',
                            'rea',
                            'dc']]
            tr[reg].append(lv)
    maxjours = max([len(tr[x]) for x in tr])
    tr = dict([(x,tr[x]) for x in tr
               if len(tr[x]) == maxjours and len([y for y in tr[x] if '' in y]) == 0])
    jours = [x[0] for x in tr[list(tr)[0]]] 
    regs = sorted([x for x in tr])
    t = np.zeros((len(deps),len(jours)))
    for reg in regs:
        v = [mfloat(x[1]) for x in tr[reg]]
        popreg = sum([population_dep[dep] for dep in regiondep[int(reg)]])
        for dep in regiondep[int(reg)]:
            t[depsind[dep],:] = np.array(v[:]) * population_dep[dep] / popreg
    datahospiage[age] = {'nom': 'hospi '+ age,
                         'titre': 'hospitalisations '+ age,
                         'dimensions': ['departements', 'jours'],
                         'jours': jours,
                         'departements': deps,
                         'valeurs': t}

print('hospiage ok', jours[-1])

######################################################################
# tests
# https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-resultats-des-tests-virologiques-covid-19/
# ils font chier, ca n'arrête pas de changer, la structure de ce csv...
# ['dep', 'jour', 'P', 'T', 'cl_age90']
# https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675
csv = chargecsv('https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675',
                zip = False,
                sep = ';')

# a disparu

dataposage = {}
datatauxposage = {}

for age in ['0'] + [str(x)+'9' for x in range(9)]+['90']: # 0 c est le total des tous les ages
    td = dict([(x,[]) for x in departements])
    for x in csv:
        dep = get(csv,x,'dep')
        if dep == '00': print(x)
        if dep in departements and get(csv,x,'cl_age90') == age:
            lv = [get(csv,x,c)
                  for c in ['jour',
                            'P',
                            'T']]
            td[dep].append(lv)
    maxjours = max([len(td[x]) for x in td])
    td = dict([(x,td[x]) for x in td
               if len(td[x]) == maxjours and len([y for y in td[x] if '' in y]) == 0])
    jours = [x[0] for x in td[list(td)[0]]] 
    deps = sorted([x for x in td])
    dataposage[age] = {'nom': 'positifs'+ (' ' + age if age != '0' else ''),
                       'titre': 'positifs '+ (age if age != '0' else ''),
                       'dimensions': ['departements', 'jours'],
                       'jours': jours,
                       'departements': [int(d) for d in deps],
                       'valeurs': np.array([[mfloat(x[1]) for x in td[dep]] for dep in deps])}
    datatauxposage[age] = {'nom': 'taux positifs'+ (' ' + age if age != '0' else ''),
                           'titre': 'taux positifs '+ (age if age != '0' else ''),
                           'dimensions': ['departements', 'jours'],
                           'jours': jours,
                           'departements': [int(d) for d in deps],
                           'valeurs': np.array([[mfloat(x[1])/(0.00001 + mfloat(x[2]))
                                                 for x in td[dep]] for dep in deps])}

datapos = dataposage['0']
datatauxpos = datatauxposage['0']

print('positifs age ok', jours[-1])

######################################################################
# r0

dataR = {'nom': 'R',
         'titre': 'R: taux de reproduction',
         'dimensions': ['departements', 'jours'],
         'jours': dataurge['jours'],
         'departements': dataurge['departements'],
         'valeurs': np.array([lissage(r0(lissage(lissage(dataurge['valeurs'][dep],7),7),derive=14),7)
                              for dep in range(len(dataurge['departements']))])
}
#plt.plot(np.transpose(dataR['valeurs']));plt.show()

######################################################################
# exces de deces durant les semaines 11 à 14 de 2020: 9 mars au 5 avril
# https://www.data.gouv.fr/fr/datasets/niveaux-dexces-de-mortalite-standardise-durant-lepidemie-de-covid-19/
# https://www.data.gouv.fr/fr/datasets/r/055ebba4-89dc-4996-962e-71dde6aaf7a6
'''
Le Z-score est calculé par la formule : (nombre observé – nombre attendu)/ écart-type du nombre attendu.

Les cinq catégories d'excès sont définies de la façon suivante :

    Pas d’excès : indicateur standardisé de décès (Z-score) <2
    Excès modéré de décès : indicateur standardisé de décès (Z-score) compris entre 2 et 4,99
    Excès élevé de décès : indicateur standardisé de décès (Z-score) compris entre 5 et 6,99 :
    Excès très élevé de décès : indicateur standardisé de décès (Z-score) compris entre 7 et 11,99 : Excès exceptionnel de décès indicateur standardisé de décès (Z-score) supérieur à 12
'''
Z = {1:1, 2:3.5, 3:6, 4:9.5, 5: 15}

deps = dataurge['departements']
csv = chargecsv('https://www.data.gouv.fr/fr/datasets/r/055ebba4-89dc-4996-962e-71dde6aaf7a6')
sdeps = [str(y) if y > 9 else '0' + str(y) for y in deps]
vdeps = {}
for x in csv:
    d = x[0].replace('"','')
    if str(d) in sdeps:
        d = int(d)
        if x[2] == '"0"':
            v = Z[int(x[3].replace('"',''))]
            if d in vdeps:
                vdeps[d] += v
            else:
                vdeps[d] = v


jours = dataurge['jours']
data = np.zeros((len(deps),len(jours)))
for(d,dep) in enumerate(deps):
    data[d,:] = vdeps[dep]

dataexcesdeces = {'nom': 'excesdeces',
                  'titre': 'excès des décès',
                  'dimensions': ['departements', 'jours'],
                  'jours': jours,
                  'departements': deps,
                  'valeurs': data}

# décès dûs au covid au 17 mai 2020
# https://www.data.gouv.fr/fr/datasets/donnees-de-certification-electronique-des-deces-associes-au-covid-19-cepidc/
csv2 = chargecsv('https://www.data.gouv.fr/fr/datasets/r/d0420516-e193-4c57-a68f-5b082345b439')
val = dict([(int(x[0]),x[2]) for x in csv2
            if x[0] in sdeps and x[1] == '0' and x[3] == '2020-05-17'])
data = np.zeros((len(deps),len(jours)))
for(d,dep) in enumerate(deps):
    data[d,:] = val[dep]

    datadeces17mai = {'nom': 'deces17mai',
                      'titre': 'décès au 17 mai 2020',
                      'dimensions': ['departements', 'jours'],
                      'jours': jours,
                      'departements': deps,
                      'valeurs': data}

######################################################################
#

indicateurs = [dataurge, datahospiurge, datasosmedecin, 
               datareatot, datahospitot, datadecestot,
               datahospi, datarea, datadeces,
               datahospiage, # par région
               dataposage,
               datapos,
               datatauxposage,
               datatauxpos,
               dataR,
               dataexcesdeces,
               datadeces17mai
]

import pickle
f = open('indicateurs.pickle','wb')
pickle.dump(indicateurs,f)
f.close()
