# python3 copieversion.py
# bonne version
# tout sauf sosmedecin
# correlations et decalages avec derivees
# prevision par minimum quadratique
# animations
# evaluation par comparaison avec extrapolations lineaire et quadratique

import os

version = 13
dir = 'version' + str(version)
while dir in os.listdir('..'):
    version += 1
    dir = 'version' + str(version)

print('version', version)
os.system('tar -cvf ../version' + str(version) + '.tgz '
          + '../versioncourante')


