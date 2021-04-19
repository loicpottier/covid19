# python3 copieversion.py
# bonne version
# tout sauf sosmedecin
# correlations et decalages avec derivees
# prevision par minimum quadratique
# animations
# evaluation par comparaison avec extrapolations lineaire et quadratique

import os

version = 20
dir = 'version' + str(version)
while dir in os.listdir('..') or dir+'.tgz' in os.listdir('..'):
    version += 1
    dir = 'version' + str(version)

print('version', version)
os.system('mkdir ../version' + str(version))
os.system('cp copieversion.py urlcache.py outils.py charge_contextes.py charge_indicateurs.py correlation.py synthese.py evaluation.py popinfectee.py ../version' + str(version))



