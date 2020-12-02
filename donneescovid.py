#python3
from outils import *

######################################################################
# on récupère les données
print('download des données de santepubliquefrance')

######################################################################
# réanimation
j = chargejson('https://geodes.santepubliquefrance.fr/GC_infosel.php?lang=fr&allindics=0&nivgeo=dep&view=map2&codgeo=06&dataset=covid_hospit&indic=rea&filters=sexe=0')

data = j['content']['data']['covid_hospit']
datarea = {'nom': 'rea',
           'titre': j['content']['indicateurs'][0]['c_lib_indicateur'],
           'dimensions': ['zones','jours'],
           'jours': sorted(list(set([x['jour'] for x in data]))),
           'zones': ['fra@FR','dep@06']}
datarea['valeurs'] = np.array([[[x['rea'] for x in data if x['jour'] == j 
                                 and x['sexe'] == '0' and x['territory'] == zone][0]
                                for j in datarea['jours']]
                               for zone in datarea['zones']])
######################################################################
#Nombre d'hospitalisations parmi les passages aux urgences pour suspicion de COVID-19 - Quotidien
j = chargejson('https://geodes.santepubliquefrance.fr/GC_infosel.php?lang=fr&allindics=0&nivgeo=dep&view=map2&codgeo=06&dataset=sursaud_corona_quot&indic=nbre_hospit_corona&filters=sursaud_cl_age_corona=0')

data = j['content']['data']['sursaud_corona_quot']
datahospi = {'nom': 'hospiurgences',
             'titre': j['content']['indicateurs'][0]['c_lib_indicateur'],
             'dimensions': ['zones','jours'],
             'jours': sorted(list(set([x['date_de_passage'] for x in data]))),
             'zones': ['fra@FR','dep@06']}
datahospi['valeurs'] = np.array([[[x['nbre_hospit_corona'] 
                                   for x in data if x['date_de_passage'] == j 
                                   and x['sursaud_cl_age_corona'] == '0'
                                   and x['territory'] == zone][0]
                                  for j in datahospi['jours']]
                                 for zone in datahospi['zones']])

######################################################################
# urgences
j = chargejson('https://geodes.santepubliquefrance.fr/GC_infosel.php?lang=fr&allindics=0&nivgeo=dep&view=map2&codgeo=06&dataset=sursaud_corona_quot&indic=nbre_pass_corona&filters=sursaud_cl_age_corona=0')

data = j['content']['data']['sursaud_corona_quot']
dataurge = {'nom': 'urge',
            'titre': j['content']['indicateurs'][0]['c_lib_indicateur'],
            'dimensions': ['zones','jours'],
            'jours': sorted(list(set([x['date_de_passage'] for x in data]))),
            'zones': ['fra@FR','dep@06']}
dataurge['valeurs'] = np.array([[[x['nbre_pass_corona'] 
                                  for x in data if x['date_de_passage'] == j 
                                  and x['sursaud_cl_age_corona'] == '0'
                                  and x['territory'] == zone][0]
                                 for j in dataurge['jours']]
                                for zone in dataurge['zones']])
######################################################################
# décès
j = chargejson('https://geodes.santepubliquefrance.fr/GC_infosel.php?lang=fr&allindics=0&nivgeo=dep&view=map2&codgeo=06&dataset=covid_hospit_incid&indic=incid_dc')

data = j['content']['data']['covid_hospit_incid']
datadeces = {'nom': 'deces',
             'titre': j['content']['indicateurs'][0]['c_lib_indicateur'],
             'dimensions': ['zones','jours'],
             'jours': sorted(list(set([x['jour'] for x in data]))),
             'zones': ['fra@FR','dep@06']}
datadeces['valeurs'] = np.array([[[x['incid_dc'] for x in data
                                   if x['jour'] == j 
                                   and x['territory'] == zone][0]
                                  for j in datadeces['jours']]
                                 for zone in datadeces['zones']])
######################################################################
# indicateurs france
trace([(zipper(t['jours'],lissage(t['valeurs'][0],7))[-120:],t['nom'],'-')
       for t in [datarea, datahospi, dataurge, datadeces]]
      + [(zipper(datarea['jours'],
                 derivee(lissage(30*datarea['valeurs'][0],7),7))[-120:],
          '30*rea/jour','-')],
      'indicateurs France',
      'donnees/_indicateurs_covid_france')
trace([(zipper(t['jours'],
               derivee(derivee(lissage(t['valeurs'][0],7),7),7))[-120:],
        t['nom'],'-')
       for t in [datarea, datahospi, dataurge, datadeces]],
      'dérivée seconde indicateurs alpes maritimes',
      'donnees/_derivee_seconde_indicateurs_covid_france')
######################################################################
# indicateurs 06
trace([(zipper(t['jours'],lissage(t['valeurs'][1],7))[-120:],t['nom'],'-')
       for t in [datarea, datahospi, dataurge, datadeces]]
      + [(zipper(datarea['jours'],
                 derivee(lissage(30*datarea['valeurs'][1],7),7))[-120:],
          '30*rea/jour','-')],
      'indicateurs alpes maritimes',
      'donnees/_indicateurs_covid_06')
trace([(zipper(t['jours'],
               derivee(derivee(lissage(t['valeurs'][1],7),7),7))[-120:],
        t['nom'],'-')
       for t in [datarea, datahospi, dataurge, datadeces]],
      'dérivée seconde indicateurs alpes maritimes',
      'donnees/_derivee_seconde_indicateurs_covid_06')

######################################################################
# nombre de cas positifs
######################################################################

# a partir du '2020-05-13'
j = chargejson('https://geodes.santepubliquefrance.fr/GC_indic.php?lang=fr&prodhash=5bdf572d&indic=p&dataset=sp_pos_quot&view=map2&filters=cl_age90=19,jour=2020-11-02')

data = j['content']['zonrefs'][0]['values']
datapositifage = {'nom': 'positifage',
                  'titre': j['content']['indic']['c_lib_indicateur'],
                  'dimensions': ['ages','jours'],
                  'jours': sorted(list(set([x['jour'] for x in data]))),
                  'ages': [str(x)+'9' for x in range(9)]+['90']}
# population par age
popages = {'09': 7410000,
        '19': 7980000,
        '29': 7230000,
        '39': 8030000,
        '49': 8320000,
        '59': 8490000,
        '69': 7780000,
        '79': 5570000,
        '89': 3190000,
        '90':  900000, #plus de 90
        }

popages['20-70'] = sum([popages[x] for x in ['29','39','49','59','69']])
popages['>70'] = sum([popages[x] for x in ['79','89','90']])
popages['tout'] = sum([popages[x] for x in datapositifage['ages']])

def positifs(age):
    if age == '20-70':
        return(np.sum(np.array([positifs(age) for age in ['29','39','49','59','69']]),
                      axis = 0))
    elif age == '>70':  
        return(np.sum(np.array([positifs(age) for age in ['79','89','90']]),axis = 0))
    elif age == 'tout':  
        return(np.sum(np.array([positifs(age)
                                for age in [str(x)+'9' for x in range(9)]+['90']]),
                      axis = 0))
    else:
        return(np.array([x['p'] for x in data
                         if x['cl_age90'] == age]))

datapositifage['valeurs'] = np.array([positifs(age) for age in datapositifage['ages']])

######################################################################
# nombres de cas de covid par tranche d'âge
trace([(zipper(datapositifage['jours'],
               lissage(100*datapositifage['valeurs'][a]/popages[age],7))[-70:],
        age,'-')
       for (a,age) in enumerate(datapositifage['ages'])],
      "nouveaux cas France (% de la classe d'age)",
      'donnees/_positifs_france',
      xlabel = 45)

trace([(zipper(datapositifage['jours'],
               lissage(100*datapositifage['valeurs'][a]/popages[age],7))[-70:],
        age,'-')
       for (a,age) in enumerate(['09','19','29','20-70','>70'])],
      "nouveaux cas France (% de la classe d'age)",
      'donnees/_positifs_france_0-19',
      xlabel = 45)
# total
trace([(zipper(datapositifage['jours'],
               lissage(np.sum(datapositifage['valeurs'], axis = 0),7))[-120:],
        'positifs','-')],
      "nouveaux cas France (lissage sur 7 jours)",
      'donnees/_nouveaux_cas_france',
      xlabel = 45)

######################################################################
# total des cas positifs par tranches d'âge
# avant le '2020-05-26' inclus
jp = chargejson('https://geodes.santepubliquefrance.fr/GC_indic.php?lang=fr&prodhash=b066af17&indic=nb_pos&dataset=covid_troislabo_quot&view=map2&filters=clage_covid=A,jour=2020-05-26')
datap = jp['content']['zonrefs'][0]['values']

jt = chargejson('https://geodes.santepubliquefrance.fr/GC_indic.php?lang=fr&prodhash=b066af17&indic=nb_test&dataset=covid_troislabo_quot&view=map2&filters=clage_covid=A,jour=2020-05-26')
datat = jt['content']['zonrefs'][0]['values']

jours0 = sorted(list(set([x['jour'] for x in datap]))) # dernier '2020-05-26'
valeurs0 = np.array([[[[x['nb_pos'],datat[k]['nb_test']] for (k,x) in enumerate(datap)
                       if x['jour'] == j and x['clage_covid'] == age][0]
                      for age in ['A','B','C','D','E']]
                     for j in jours0])

# total des cas positifs au 13 mai
positifs_13_mai = np.sum(valeurs0[:,:,0][:jours0.index('2020-05-12')])
# on extrapole les nombres de positifs car les tranches changent le 13 mai
lp0 = [0] + list(np.sum(valeurs0[:,:,0][:jours0.index('2020-05-12')], axis = 0))
lp = integrate(lp0)
la = [0,15,45,65,75,100]
la2 = [0,10,20,30,40,50,60,70,80,90,100]
def f(a):
    i = 0
    while a >= la[i] and i < len(la) - 1:
        i += 1
    #print(i,la[i])
    return(lp[i-1] + (a - la[i-1])/(la[i] - la[i-1]) * (lp[i] - lp[i-1]))

lp1 = [f(x) for x in la2]
# aproximation des nombres de positifs par tranches d'âges de 10 ans:
lp11 = [lp1[k+1] - lp1[k] for k in range(len(lp1)-1)]

trace([(zipper(datapositifage['jours'],
               np.array(integrate(datapositifage['valeurs'][a],
                                  initial = lp11[a]))/popages[age]*100),
        age,'-')
       for (a,age) in enumerate(datapositifage['ages'])],
      "nombre de cas total en France (% de la classe d'age)",
      'donnees/_total_positifs_france')

######################################################################
# r0

# 3.296 = r0 expérimental donné par pas mal d'études

#https://academic.oup.com/jtm/advance-article/doi/10.1093/jtm/taaa115/5876265
# donne 3
intervalle_seriel = 4.11 # = math.log(3.296)/0.29 

def r0(l):
    # l1: log de l
    l1 = [math.log(x) if x>0 else 0 for x in l]
    # dérivée de l1
    dl1 = derivee(l1,largeur=7) 
    # r0 instantané
    lr0 = [min(4000,math.exp(c*intervalle_seriel)) for c in dl1]
    return(lr0)

trace([(zipper(t['jours'],r0(lissage(t['valeurs'][0],7)))[-120:],t['nom'],'-')
       for t in [datahospi, dataurge]],
      'r0 France',
      'donnees/_r0_france')

trace([(zipper(t['jours'],r0(lissage(t['valeurs'][1],7)))[-120:],t['nom'],'-')
       for t in [datahospi, dataurge]],
      'r0 06',
      'donnees/_r0_06')
######################################################################
# taux de positivité des tests
# https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-resultats-des-tests-virologiques-covid-19/
# france
data = chargecsv('https://www.data.gouv.fr/fr/datasets/r/dd0de5d9-b5a5-4503-930a-7b08dc0adc7c')
datafrance = [(x[1],int(float(x[4])),int(float(x[7]))) for x in data if x[-1] == '0']
# 06
data = chargecsv('https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675')
data06 = [(x[1],int(x[2]),int(x[3])) for x in data if x[-1] == '0' and x[0] == '06']
datatests = {'nom': 'tests',
             'titre': 'taux de positivité des tests',
             'dimensions': ['zones','jours','tests'],
             'jours': [x[0] for x in datafrance],
             'zones': ['france','06'],
             'tests': ['positifs', 'effectués']}
datatests['valeurs'] = np.array([[x[1:] for x in datafrance],
                                 [x[1:] for x in data06]])

tmax = np.max(datatests['valeurs'][:,:,1])

trace([(zipper(datatests['jours'],
               lissage(100 * datatests['valeurs'][t,:,0]
                       /datatests['valeurs'][t,:,1],7))[-120:],
        datatests['zones'][t],
        '-')
       for t in [0,1]]
      +[(zipper(datatests['jours'],
                lissage(10 * datatests['valeurs'][0,:,1]/tmax,7))[-120:],
         '10*t/' + str(tmax),
         '-')],
      'taux de positivité des tests',
      'donnees/_taux_pos_france')
######################################################################
print('----------------------------------------------------------------------')
print('france')
for t in [datarea, datahospi, dataurge, datadeces]:
    print(t['nom'],zipper(t['jours'],t['valeurs'][0])[-3:])
print('alpes-maritimes')
for t in [datarea, datahospi, dataurge, datadeces]:
    print(t['nom'],zipper(t['jours'],t['valeurs'][1])[-3:])






    
    
