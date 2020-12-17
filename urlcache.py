import urllib.request
import gzip
try:
    from cache.cache import cache # dictionnaire des contenus des urls
except:
    cache = {}

dircache = '/home/loic/Dropbox/covid19/previsions/cache/'

compteur = 0

def efface_cache():
    global cache
    cache = {}
    f = open(dircache + 'cache.py','w')
    f.write('cache = {}')
    f.close()

def sauve_cache():
    print("""
**********************************************************************
                        SAUVEGARDE DU CACHE
**********************************************************************
""")          
    f = open(dircache + 'cache.py','w')
    f.write('cache = ' + str(cache))
    f.close()

def urltexte(url, zip = False):
    global compteur
    if url in cache:
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

