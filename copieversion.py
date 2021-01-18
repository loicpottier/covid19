# python3 copieversion.py

import os

version = 13
dir = 'version' + str(version)
while dir in os.listdir():
    version += 1
    dir = 'version' + str(version)

print('version', version)
os.system('mkdir version' + str(version))
for file in ['urlcache', 'outils',
             'charge_contextes', 'charge_indicateurs',
             'correlation', 'copieversion', 'synthese2', 'evaluation'
             ]:
    os.system('cp ' + file + '.py version' + str(version))

for file in ['ajour2']:
    os.system('cp ' + file + ' version' + str(version))


