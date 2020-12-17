import sys
from urlcache import *

try:
    if sys.argv[1] == 'nouveau':
        efface_cache()
        print('cache effacé, on charge les dernières données disponibles')
except:
    print('on utilise le cache')
######################################################################
import previsions
import synthese_donnees

######################################################################
try:
    if sys.argv[1] == 'nouveau':
        sauve_cache()
except:
    pass
