import urllib.request
import gzip
import pickle

#DIRCOVID19 = '/home/loic/covid19/' # pour les gros fichiers
######################################################################
# Version DEV

DIRCOVID19 = '/home/loic/covid19dev/' # pour les gros fichiers
######################################################################
DIRCOVID19UK = '/home/loic/covid19UK/' # pour les gros fichiers
DIRCACHE = DIRCOVID19 + 'cache/'

try:
    # dictionnaire des contenus des urls
    f = open(DIRCACHE + 'cache.pickle','rb')
    cache = pickle.load(f)
    f.close()
except:
    print('pas de cache')
    cache = {}

compteur = 0

def efface_cache():
    global cache
    cache = {}
    f = open(DIRCACHE + 'cache.pickle','wb')
    pickle.dump(cache,f)
    f.close()

def sauve_cache():
    print("""
**********************************************************************
                        SAUVEGARDE DU CACHE
**********************************************************************
""")          
    f = open(DIRCACHE + 'cache.pickle','wb')
    pickle.dump(cache,f)
    f.close()

def urltextecache(url, zip = False, use_cache = True):
    global compteur
    if url in cache and use_cache:
        #print('déjà dans le cache')
        return(cache[url])
    else:
        response = urllib.request.urlopen(url)
        encoding = response.info().get_content_charset() or "utf-8"
        data = response.read()      # a `bytes` object
        if zip:
            data = gzip.decompress(data)
        texte = data.decode(encoding)
        cache[url] = texte
        compteur += 1
        if compteur%100 == 0:
            sauve_cache()
        return(texte)

def urltexte(url, zip = False):
    response = urllib.request.urlopen(url)
    encoding = response.info().get_content_charset() or "utf-8"
    data = response.read()      # a `bytes` object
    if zip:
        data = gzip.decompress(data)
    texte = data.decode(encoding)
    return(texte)
