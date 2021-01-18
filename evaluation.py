from correlation import *
import matplotlib.animation as animation

######################################################################
# animation des prévisions passées

utiliser_proportions = True
present = aujourdhui

#plt.plot(np.mean(MF[ni('urgences'),:,:],axis = 0));plt.show()
#plt.plot(np.mean(M[ni('urgences'),:,:],axis = 0));plt.show()

def courbes_prev_duree(x, duree, passe = 100, futur = 60, pas = 1):
    global intervalle,M,MR,MRD,coefs,coefficients
    intervalle0,M0,MR0,MRD0,coefs0,coefficients0 = copy.deepcopy([intervalle,M,MR,MRD,coefs,
                                                                  coefficients])
    x0,x1 = intervalle[x]
    lcourbes = []
    for j in range(0,duree,pas):
        print('prévision -' + str(j) + '\n')
        xdep = x1 - 1 - j
        depart = jour_de_num[jours[0] + xdep]
        M = creeM(depart)
        normalise_data(M)
        MR = proportionsM(M)
        MRD = deriveM(MR)
        coefs = calcule_correlations()
        coefficients = [calcule_coefficients(y) for y in range(nnoms)]
        MF,MRF,MRDF = prevoit_tout(futur, depart = depart)
        f = np.mean if noms[x] in donnees_proportions else np.sum
        lcourbes.append(zipper(jourstext[xdep:xdep+futur],
                               f(MF[x,:,xdep:xdep+futur], axis = 0)))
    reel = zipper(jourstext[int(x1)-passe:int(x1)],
                  f(M0[x,:,int(x1)-passe:int(x1)], axis = 0))
    intervalle,M,MR,MRD,coefs,coefficients = intervalle0,M0,MR0,MRD0,coefs0,coefficients0
    return((reel,lcourbes))

def axejours(lj):
    n = len(lj)
    lk = [n-1 - 7*k for k in range(n//7+1)][::-1]
    ljaxe = [joli(lj[k]) for k in lk]
    plt.xticks(lk,ljaxe,rotation = 70,fontsize = 8)

def val(jours,l):
    d = dict(l)
    return([d[j] if j in d else None for j in jours])

def anime_previsions(nom, duree = 60, futur = 60):
    x = ni(nom)
    reel, previsions = courbes_prev_duree(x, duree, futur = futur)
    previsions = previsions[::-1]
    courbes = [reel] + previsions
    fig, ax = plt.subplots()
    lj = []
    for courbe in courbes:
        lj = lj + [x[0] for x in courbe]
    lj = sorted(list(set(lj))) # liste des jours concernés par les courbes
    axejours(lj)
    plt.grid()
    plt.title('prévisions passées: ' + nom)
    # courbe réel
    lv = val(lj,reel)
    plt.plot(lv,'-', linewidth = 2)
    prev, = plt.plot(lj,val(lj,previsions[0]), '-')
    def init():
        ax.set_xlim(0, len(lj))
        ax.set_ylim(0,max([max([x[1] for x in c]) for c in courbes]))
        return prev,
    def update(frame):
        prev.set_data(lj,val(lj,previsions[frame]))
        return prev,
    ani = animation.FuncAnimation(fig, update, frames= np.array(list(range(len(previsions)))),
                        init_func=init, blit=True)
    plt.rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'
    DPI=180
    writer = animation.FFMpegWriter(fps=5, bitrate=10000)
    ani.save("synthese2/previsions_" + nom + ".mp4", writer = writer, dpi=DPI) 

anime_previsions('urgences',duree = 90, futur = 60)
anime_previsions('réanimations',duree = 90, futur = 60)
anime_previsions('hospitalisations',duree = 90, futur = 60)
anime_previsions('taux positifs',duree = 90, futur = 60)
anime_previsions('positifs',duree = 90, futur = 60)
anime_previsions('sosmedecin',duree = 90, futur = 60)
