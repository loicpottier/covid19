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
                        'nbre_hospit_corona']]
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

print('urge et hospiurge ok', jours[-1])
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

datareatot = {'nom': 'réanimation total',
            'titre': 'total réanimation',
            'dimensions': ['departements', 'jours'],
            'jours': jours,
            'departements': [int(d) for d in deps],
            'valeurs': np.array([[mfloat(x[2]) for x in td[dep]] for dep in deps])}
datahospitot = {'nom': 'hospitalisations total',
            'titre': 'total hospitalisations',
            'dimensions': ['departements', 'jours'],
            'jours': jours,
            'departements': [int(d) for d in deps],
            'valeurs': np.array([[mfloat(x[1]) for x in td[dep]] for dep in deps])}
datadecestot = {'nom': 'décès total',
            'titre': 'total décès',
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

datahospi = {'nom': 'hospitalisations',
             'titre': 'hospitalisations',
             'dimensions': ['departements', 'jours'],
             'jours': jours,
             'departements': [int(d) for d in deps],
             'valeurs': np.array([[mfloat(x[1]) for x in td[dep]] for dep in deps])}
datarea = {'nom': 'réanimations',
             'titre': 'réanimations',
             'dimensions': ['departements', 'jours'],
             'jours': jours,
             'departements': [int(d) for d in deps],
             'valeurs': np.array([[mfloat(x[2]) for x in td[dep]] for dep in deps])}
datadeces = {'nom': 'décès',
             'titre': 'décès',
             'dimensions': ['departements', 'jours'],
             'jours': jours,
             'departements': [int(d) for d in deps],
             'valeurs': np.array([[mfloat(x[3]) for x in td[dep]] for dep in deps])}

print('hospi ok', jours[-1])
######################################################################
# hospitalieres classes d ages
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

for age in ['0'] + [str(x)+'9' for x in range(9)]+['90']: # 0 c est le total des tous les ages
    td = dict([(x,[]) for x in regions])
    for x in csv:
        reg = get(csv,x,'reg')
        if reg in regions and get(csv,x,'cl_age90') == age:
            lv = [get(csv,x,c)
                  for c in ['jour',
                            'hosp',
                            'rea',
                            'dc']]
            td[reg].append(lv)
    maxjours = max([len(td[x]) for x in td])
    td = dict([(x,td[x]) for x in td
               if len(td[x]) == maxjours and len([y for y in td[x] if '' in y]) == 0])
    jours = [x[0] for x in td[list(td)[0]]] 
    regs = sorted([x for x in td])
    datahospiage[age] = {'nom': 'hospi '+ age,
                         'titre': 'hospitalisations '+ age,
                         'dimensions': ['regions', 'jours'],
                         'jours': jours,
                         'regions': regs,
                         'valeurs': np.array([[mfloat(x[1]) for x in td[reg]] for reg in regs])}

print('hospiage ok', jours[-1])

######################################################################
# tests
# ['dep', 'jour', 'P', 'T', 'cl_age90', 'pop']

csv = chargecsv('https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675',
                zip = False,
                sep = ';')

population_dep = {}
for x in csv:
    if len(x) >= 5 and x[4] == '0':
        try:
            population_dep[int(x[0])] = int(x[5])
        except:
            pass

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

def r0(l):
    intervalle_seriel = 4.11 # = math.log(3.296)/0.29 
    # l1: log de l
    l1 = [math.log(x) if x>0 else 0 for x in l]
    # dérivée de l1
    dl1 = derivee(l1,largeur=7) 
    # r0 instantané
    lr0 = [min(4000,math.exp(c*intervalle_seriel)) for c in dl1]
    return(lr0)

dataR = {'nom': 'R',
         'titre': 'R: taux de reproduction',
         'dimensions': ['departements', 'jours'],
         'jours': dataurge['jours'],
         'departements': dataurge['departements'],
         'valeurs': np.array([r0(lissage(dataurge['valeurs'][dep],7))
                              for dep in range(len(dataurge['departements']))])
}

######################################################################
#

indicateurs = [dataurge, datahospiurge,
               datareatot, datahospitot, datadecestot,
               datahospi, datarea, datadeces,
               datahospiage, # par région
               dataposage,
               datapos,
               datatauxposage,
               datatauxpos,
               dataR
]
