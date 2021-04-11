from outils import *

f = open(DIRCOVID19 + 'indicateurs.pickle','rb')
indicateurs = pickle.load(f)
f.close()

dataurge, datahospiurge, datareatot, datahospitot, datahospi, datarea, datadeces, datahospiage, dataposage, datapos, datatauxposage, datatauxpos, dataexcesdeces, datadeces17mai = indicateurs

#from correlation import *

######################################################################
# https://modelisation-covid19.pasteur.fr/realtime-analysis/infected-population/
# https://doi.org/10.1101/2020.09.16.20195693
# entre le 4 mai et le 23 juin

infectes_regions = {'Ile-de-France': (552,6348),
                    'Alsace-Champagne-Ardenne-Lorraine': (270,3434),
                    'Aquitaine-Limousin-Poitou-Charentes': (161,4843)} #4846

infectes_ages = {'<40': (245,2262), # (infectés sérologie,testés)
                 '40-50': (332,2897),
                 '50-60': (176,3019),
                 '60-70': (133,3272),
                 '>70': (97,3175)}
    
# quelques calculs de probabilites conditionnelles
# proba d'être infecté, sachant l'age
def pinf_age(a):
    return(infectes_ages[a][0] / infectes_ages[a][1])

# proba d'avoir l'âge a (parmi les testés)
def page(a):
    return(infectes_ages[a][1] / sum([infectes_ages[a][1] for a in infectes_ages]))

# proba d'être infecté (parmi les testés): formule des probabilités totales
# 6.7%
def pinf():
    return(sum([pinf_age(a) * page(a) for a in ages]))

# proba d'être infecté, sachant la région
def pinf_region(r):
    return(infectes_regions[r][0] / infectes_regions[r][1])

# proba d'être dans la région r (parmi les testés)
def pregion(r):
    return(infectes_regions[r][1] / sum([infectes_regions[r][1] for r in infectes_regions]))

# proba d'être infecté (parmi les testés), égal à pinf() normalement (en fait il y a une différence de 3 dans le total des testés, selon l'âge ou la région)
def pinf2():
    return(sum([pinf_region(r) * pregion(r) for r in infectes_regions]))

# on suppose que infectes_ages et infectes_regions sont des variables indépendantes, même parmi les infectés.
# proba d'être infecté, sachant l'âge et la région
def pinf_age_region(a,r):
    return(pinf_age(a) * pinf_region(r) / pinf())

ages = ['<40', '40-50', '50-60', '60-70', '>70']

pinf_age_region('<40','Ile-de-France')
sum([pinf_age_region(a,'Ile-de-France') * page(a) for a in ages])
pinf_region('Ile-de-France')
sum([pinf_age_region('<40',r) * pregion(r) for r in infectes_regions])
pinf_age('<40')

######################################################################
# on extrapole aux hospitalisations,
# on suppose que pour un âge donné, le nombre d'hospitalisés est proportionnel à celui des infectés
# p.ex. si on double les infectés, les hospitalisés doublent aussi.
# et indépendemment de la région

# hospitalisations cumulées jusqu'au 4 mai (à diviser par la durée moyenne d'hospitalisation si on veut le nobre total de personnes hospitalisées, mais on s'en fiche en fait)

ldep = (regions['Ile-de-France']
        + regions['Alsace-Champagne-Ardenne-Lorraine']
        + regions['Aquitaine-Limousin-Poitou-Charentes'])

poptot = sum([population_dep[d] for d in ldep])

j0 = num_de_jour('2020-05-04') - num_de_jour(datahospiage['09']['jours'][0])

# cumul hospitalises dans les departements ldep, tranche d'age a, jusqu'au jour j 
def hospi(ldep,a,j):
    return(int(np.sum([np.sum(datahospiage[a]['valeurs'][dep,:j])
                       for dep in [datahospiage[a]['departements'].index(d)
                                   for d in ldep]])))

def hospis(a,r,j):
    ldep = regions[r]
    dh = {'<40': sum([hospi(ldep,a,j) for a in ['09','19','29','39']]),
          '40-50': hospi(ldep,'49',j),
          '50-60': hospi(ldep,'59',j),
          '60-70': hospi(ldep,'69',j),
          '>70': sum([hospi(ldep,a,j) for a in ['79','89','90']])}
    return(dh[a])

hospis('<40','Ile-de-France',j0)

def hospis_regions_jour(lr,j):
    return(sum([sum([hospis(a,r,j) for a in ages])
                for r in lr]))

# dans les 3 régions testées
def hospis_ages_jour(la,j):
    return(sum([sum([hospis(a,r,j) for r in infectes_regions])
                for a in la]))

# proba d'être infecté pour l'âge a, dans la région r, jusqu'au jour j
def pinf_age_region_jour(a,r,j):
    # variation des hospi de j à j0
    dh =  hospis(a,r,j) / hospis(a,r,j0)
    return(pinf_age_region(a,r) * dh)

def pinf_age_jour(a,j):
    dh =  hospis_ages_jour([a],j) / hospis_ages_jour([a],j0)
    return(pinf_age(a) * dh)

def pinf_region_jour(r,j):
    dh =  hospis_regions_jour([r],j) / hospis_regions_jour([r],j0)
    return(pinf_region(r) * dh)

def pinf_jour(j):
    dh =  hospis_regions_jour(infectes_regions,j)/hospis_regions_jour(infectes_regions,j0)
    return(pinf() * dh)

# les infectés par régions, tous âges confondus

def pop_region(r):
    return(sum([population_dep[d] for d in regions[r]]))

def pinf_regionf_jour(r,j):
    if r in infectes_regions:
        return(pinf_region_jour(r,j))
    else:
        # on se fonde sur les 3 régions
        h0 = hospis_regions_jour([r],j0)
        h30 = hospis_regions_jour(infectes_regions,j0)
        dh =  hospis_regions_jour([r],j) / hospis_regions_jour([r],j0)
        pop3 = sum([pop_region(r) for r in infectes_regions])
        return(pinf_jour(j0) * ((h0 / pop_region(r)) / (h30 / pop3)) * dh)

def pinf_regions_jour(lr,j):
    return(sum([pinf_regionf_jour(r,j) * pop_region(r) for r in lr])
           / sum([pop_region(r) for r in lr]))

def pinf_france_jour(j):
    return(pinf_regions_jour(regions,j))

# les infectes par age en France

def pinf_agef_jour(a,j):
    # on se fonde sur les 3 régions
    h0 =  sum([hospis(a,r,j0) for r in regions])
    h30 = hospis_ages_jour([a],j0)
    dh =  sum([hospis(a,r,j) for r in regions]) / h0
    pop3 = sum([pop_region(r) for r in infectes_regions])
    return(pinf_age_jour(a,j0) * ((h0 / population_france) / (h30 / pop3)) * dh)

######################################################################
# extrapolations au 10 avril (le jour où j'ai écrit le code)

jj1 = aujourdhui
j1 = num_de_jour(jj1) - num_de_jour(datahospiage['09']['jours'][0])

pinf_jour(j1)

pinf_france_jour(j1)

print("proportion de personnes infectées par SARSCOV2 au " + jj1 + ":")
print("----------------------------------------------------------------------")
print("par région:")
for r in sorted(regions):
    print(r + ":",("%.1f" % (100*pinf_regions_jour([r],j1))) + '%')

print("----------------------------------------------------------------------")
print("par tranche d'âges:")
for a in ages:
    print(a + ":",("%.1f" % (100*pinf_agef_jour(a,j1))) + '%')

######################################################################
# graphiques

lr = {}
for r in regions:
    print(r)
    lr[r] = [pinf_regions_jour([r],j)
             for j in range(j0,j1+1,7)]

trace([(zipper([jour_de_num[num_de_jour(datahospiage['09']['jours'][0]) + j]
                for j in range(j0,j1+1,7)],
               [100*x for x in lr[r]]),
        r,'-')
       for r in regions],
      "proportion de personnes infectées par le SARS-CoV-2 par région (en %)",
      DIRSYNTHESE + "infectes_regions")

la = {}
for a in ages:
    print(a)
    la[a] = [pinf_agef_jour(a,j)
             for j in range(j0,j1+1,7)]

trace([(zipper([jour_de_num[num_de_jour(datahospiage['09']['jours'][0]) + j]
                for j in range(j0,j1+1,7)],
               [100*x for x in la[a]]),
        a,'-')
       for a in ages],
      "proportion de personnes infectées par le SARS-CoV-2 par tranche d'âge (en %)",
      DIRSYNTHESE + "infectes_ages")               
    
trace([(zipper([jour_de_num[num_de_jour(datahospiage['09']['jours'][0]) + j]
                for j in range(j0,j1+1,7)],
               [100*pinf_france_jour(j) for j in range(j0,j1+1,7)]),
        'France','-')],
      "proportion de personnes infectées par le SARS-CoV-2 en France (en %)",
      DIRSYNTHESE + "infectes_france")               
