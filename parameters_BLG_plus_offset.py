from matplotlib import cm, rc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['mathtext.default'] = 'regular' #make tex font size the same as normal text
rc('mathtext', fontset='stixsans')
rcParams.update({'font.size': 9})

fig, ax=plt.subplots(1, 4, constrained_layout=True)


# to dobry zewstaw do sprawdzenia gdzie sie zamyka przerwa w top BLG
Vtmin = 1
Vtmax = 3
Vbmin = -11
Vbmax = -10

# to zewstaw do sprawdzenia gdzie sie zamyka przerwa w bottom BLG
#Vtmin = -8
#Vtmax = -6
#Vbmin = 3
#Vbmax = 4

Vt = 0
#Vtmax = 8.01
Vbmin = -60
Vbmax = 60.0

#dV = 0.02
dV = 2.0
# zakres zgodny z BLG_parameters.py: np.linspace(-60, 60, 61)
x1 = np.linspace(Vbmin, Vbmax, int((Vbmax - Vbmin) / dV) + 1)
nx = x1.shape[0]
    
# x1 to napięcie na dolnej bramce
print(x1)

n1 = x1.copy()
nt = x1.copy()
nb = x1.copy()
U1 = x1.copy()
Vg1 = x1.copy()
n2 = x1.copy()
U2 = x1.copy()
Vg2 = x1.copy()
fun4 = x1.copy()

#yi = np.arange(-8., 8.01, 0.1)
#xi = np.arange(-8.0, 8.01, 0.1)
xi = x1.copy()

#gx,gy = np.meshgrid(xi,yi)

levels = np.arange(0.0, 1, 2)

cmap=cm.RdBu_r

# probujemy rozwiazac ten uklad rownan :):)
def n_solve(functions,variables):
    func = lambda x: [ f(*x) for f in functions]
    return fsolve(func, variables, full_output=False, xtol=1e-10, maxfev=5000)
#    return broyden1(func, variables)

# Funkcje pomocnicze do zabezpieczenia przed błędami numerycznymi
def safe_sqrt(x):
    """Bezpieczny sqrt - zwraca 0 dla ujemnych wartości."""
    return np.sqrt(np.maximum(x, 0))

def safe_log(x):
    """Bezpieczny log - zabezpieczenie przed log(0)."""
    return np.log(np.maximum(x, 1e-15))

Ct = 0#3.04 
#Cb = 2.026
Cb = 0.653

#Ct = 552.635
#Cb = 552.635
Vt = 0#x1[0]
Vb = x1[0]
gamma1 = 0.39
hvf2 = (6.39)**2 
n_t = 118.57 
# parametry zgodne z domyślnymi ustawieniami w BLG_parameters.py
Cg = 230.0
#Cr = 218.45 # pomiedzy dwoma BLG? Trzeba bedzie wstawic z grubosci hBN
Cr = 2.55 # pomiedzy dwoma BLG, jest 80 nm hBN

#Cg = 552.635
#Cr = 276.32

#n0t = -0.2 #-0.5 # to przesuwanie tak jakby byl intrinsic doping.
#n0b = 0 # to przesuwanie tak jakby byl intrinsic doping.
n0t = 0
n0b = 0
#dn0t = 13 # proba zmiany "na sile" asymmetry parameter
#dn0b = -14 # proba zmiany "na sile" asymmetry parameter
dn0t = 0 # proba zmiany "na sile" asymmetry parameter
dn0b = 0 # proba zmiany "na sile" asymmetry parameter

f1t = lambda n1,dn1,U1,Vg1, n2,dn2,U2,Vg2: -dn1 + dn0t + Cr * ((Vg2 - U2/2) - (Vg1 + U1/2)) - Ct * (Vt - (Vg1 - U1/2)) - 2 * Cg * U1 
f2t = lambda n1,dn1,U1,Vg1, n2,dn2,U2,Vg2: dn1 + n_t / 2 / gamma1 * U1 * safe_log(np.abs(n1) / n_t / 2 + 0.5 * safe_sqrt( (n1 / n_t)**2 + (U1 / 2 / gamma1)**2))
f3t = lambda n1,dn1,U1,Vg1, n2,dn2,U2,Vg2: -n1 + n0t + Cr * ((Vg2 - U2/2) - (Vg1 + U1/2)) + Ct * (Vt - (Vg1 - U1/2)) 
f4t = lambda n1,dn1,U1,Vg1, n2,dn2,U2,Vg2: Vg1 + safe_sqrt(gamma1**2/2 + U1**2/4 + (hvf2 * 1e-5 * np.pi) * np.abs(n1) - gamma1/2 * safe_sqrt(gamma1**2 + (4 * hvf2 * 1e-5 * np.pi) * np.abs(n1) * (1 + (U1/gamma1)**2)) ) * (-np.sign(n1) if n1 != 0 else -1)
f1b = lambda n1,dn1,U1,Vg1, n2,dn2,U2,Vg2: -dn2 + n0b + Cb * (Vb - (Vg2 + U2/2)) - Cr * ((Vg1 + U1/2) - (Vg2 - U2/2)) - 2 * Cg * U2
f2b = lambda n1,dn1,U1,Vg1, n2,dn2,U2,Vg2: dn2 + dn0b + n_t / 2 / gamma1 * U2 * safe_log(np.abs(n2) / n_t / 2 + 0.5 * safe_sqrt( (n2 / n_t)**2 + (U2 / 2 / gamma1)**2))
f3b = lambda n1,dn1,U1,Vg1, n2,dn2,U2,Vg2: -n2 + Cb * (Vb - (Vg2 + U2/2)) + Cr * ((Vg1 + U1/2) - (Vg2 - U2/2)) 
f4b = lambda n1,dn1,U1,Vg1, n2,dn2,U2,Vg2: Vg2 + safe_sqrt(gamma1**2/2 + U2**2/4 + (hvf2 * 1e-5 * np.pi) * np.abs(n2) - gamma1/2 * safe_sqrt(gamma1**2 + (4 * hvf2 * 1e-5 * np.pi) * np.abs(n2) * (1 + (U2/gamma1)**2)) ) * (-np.sign(n2) if n2 != 0 else -1)

# w nowym solverze domyślnie rozwiązujemy pojedynczy przypadek Ct = 0
Cts = np.array([Ct])
# tablice w ktorych beda wyniki dla wszystkich Ct do zapisu
U1s = []
U2s = []
Vg1s = []
Vg2s = []
n1s = []
n2s = []

print(x1.shape[0], Cts.shape[0])

# zapis do pliku do porownaania
f = open('nowe.txt', 'w') #note 'w' = write mode
for k in range(0, Cts.shape[0]): #, Cts.shape[0]-2):
    Ct = Cts[k]
    for i in np.arange(x1.shape[0]):
        Vt = 0#x1[i]
        Vb = x1[i]
        # Lepsze wartości początkowe skalowane z Vb
        scale = max(abs(Vb), 1.0)
        init_guess = [0.1*scale, 0.01*scale, 0.01, 0.1*Vb if abs(Vb)>0.1 else 0.01, 
                      0.1*scale, 0.01*scale, 0.01, 0.1*Vb if abs(Vb)>0.1 else 0.01]
        # Użyj poprzedniego rozwiązania jako punktu startowego jeśli dostępne
        if i > 0:
            init_guess = [n1[i-1], nt[i-1], U1[i-1], Vg1[i-1], n2[i-1], nb[i-1], U2[i-1], Vg2[i-1]]
        res = n_solve([f1t,f2t,f3t,f4t, f1b,f2b,f3b,f4b], init_guess)
        #print(res)
        n1[i] = res[0]
        nt[i] = res[1]
        U1[i] = res[2]
        Vg1[i] = res[3]
        n2[i] = res[4]
        nb[i] = res[5]
        U2[i] = res[6]
        Vg2[i] = res[7]
        fun4[i] = f4t(n1[i], nt[i], U1[i], Vg1[i], n2[i], nb[i], U2[i], Vg2[i])
        #print(n[0], U[0], Vg[0])
    '''
    gridn1 = griddata( (x1, y1), n1, (gx,gy), method="nearest")
    gridnt = griddata( (x1, y1), nt, (gx,gy), method="nearest")
    gridU1 = griddata( (x1, y1), U1, (gx,gy), method="nearest")
    gridVg1= griddata( (x1, y1), Vg1, (gx,gy), method="nearest")
    
    gridn2 = griddata( (x1, y1), n2, (gx,gy), method="nearest")
    gridnb = griddata( (x1, y1), nb, (gx,gy), method="nearest")
    gridU2 = griddata( (x1, y1), U2, (gx,gy), method="nearest")
    gridVg2= griddata( (x1, y1), Vg2, (gx,gy), method="nearest")
    
    gridf4 = griddata( (x1, y1), fun4, (gx,gy), method="nearest")
    '''
    # for i in range(4):
    # 	if(i==0):
    # 		grid = gridU1
    # 	elif(i==1):
    # 		grid = gridU2
    # 	elif(i==2):
    # 		grid = gridVg1
    # 	elif(i==3):
    # 		grid = gridVg2
    # 	ax[i].set_box_aspect(1)
    # 	#ax[i].contour(xi,yi,grid, levels, linestyles='dashed')
    # 	pa=ax[i].contourf(xi,yi,grid,100, cmap=cmap)
    # 	ax[i].set_ylabel("$\mathrm{V_{bg}}$ (V)", labelpad=-15)
    # 	ax[i].set_xlabel("$\mathrm{V_{tg}}$ (V)")
    
    # 	#usuwanie bialych konturow
    # 	for c in pa.collections:
    # 		c.set_edgecolor("face" )
    		
    # 	divider = make_axes_locatable(ax[i])
    # 	cax = divider.append_axes('top', size='2%', pad=-2.8)
    # 	cbar=fig.colorbar(pa, cax=cax, orientation='horizontal')
    # 	limit = np.floor(np.amax(np.absolute(grid))*10)/10
    # 	cbar.set_ticks([-limit, np.amax(grid)])
    # 	if(i==0):
    # 		cbar.set_label("$U_2 (eV)$" )
    # 	elif(i==1):
    # 		cbar.set_label("$U_1 (eV)$" )
    # 	if(i==2):
    # 		cbar.set_label("$V_{G2} (eV)$" )
    # 	if(i==2):
    # 		cbar.set_label("$V_{G1} (eV)$" )
    
    # 	cbar.ax.xaxis.set_ticks_position("top")
    # 	cbar.ax.xaxis.set_label_position("top")


    for i in range(x1.shape[0]):
        #for j in range(gx.shape[1]):
        #Ct = Cts[k]
        #Vt = gx[i,j]
        Vb = x1[i]
        #f.write('%f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t\n ' % (Cts[k], gx[i,j], gy[i,j], gridn1[i,j], gridU1[i,j], gridVg1[i,j], gridn2[i,j], gridU2[i,j], gridVg2[i,j],
        #     f1t(gridn1[i,j], gridnt[i,j], gridU1[i,j], gridVg1[i,j], gridn2[i,j], gridnb[i,j], gridU2[i,j], gridVg2[i,j]), 
        #     f2t(gridn1[i,j], gridnt[i,j], gridU1[i,j], gridVg1[i,j], gridn2[i,j], gridnb[i,j], gridU2[i,j], gridVg2[i,j]),
        #     f3t(gridn1[i,j], gridnt[i,j], gridU1[i,j], gridVg1[i,j], gridn2[i,j], gridnb[i,j], gridU2[i,j], gridVg2[i,j]), 
        #     f4t(gridn1[i,j], gridnt[i,j], gridU1[i,j], gridVg1[i,j], gridn2[i,j], gridnb[i,j], gridU2[i,j], gridVg2[i,j])))
        f.write('%f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t\n ' % (x1[i], n1[i], U1[i], Vg1[i], n2[i], U2[i], Vg2[i],
             f1t(n1[i], nt[i], U1[i], Vg1[i], n2[i], nb[i], U2[i], Vg2[i]), 
             f2t(n1[i], nt[i], U1[i], Vg1[i], n2[i], nb[i], U2[i], Vg2[i]),
             f3t(n1[i], nt[i], U1[i], Vg1[i], n2[i], nb[i], U2[i], Vg2[i]), 
             f4t(n1[i], nt[i], U1[i], Vg1[i], n2[i], nb[i], U2[i], Vg2[i])))
        #f.write("\n ")
    f.write("\n ")
    
    U1s.append(U1)
    U2s.append(U2)
    Vg1s.append(Vg1)
    Vg2s.append(Vg2)
    n1s.append(n1)
    n2s.append(n2)
f.close()

# zapis do wczytania do fortrana
f = open('U1s.txt', 'w') #note 'w' = write mode
for k in range(Cts.shape[0]):
    for i in range(x1.shape[0]):
        f.write('%.12e\n ' % (U1s[k][i]))
f.close()

f = open('Vg1s.txt', 'w') #note 'w' = write mode
for k in range(Cts.shape[0]):
    for i in range(x1.shape[0]):
        f.write('%.12e\n ' % (Vg1s[k][i]))
f.close()


# zapis do wczytania do fortrana
f = open('U2s.txt', 'w') #note 'w' = write mode
for k in range(Cts.shape[0]):
    for i in range(x1.shape[0]):
        f.write('%.12e\n ' % (U2s[k][i]))
f.close()

f = open('Vg2s.txt', 'w') #note 'w' = write mode
for k in range(Cts.shape[0]):
    for i in range(x1.shape[0]):
        f.write('%.12e\n ' % (Vg2s[k][i]))
f.close()


# zapis do wczytania do fortrana
f = open('n1s.txt', 'w') #note 'w' = write mode
for k in range(Cts.shape[0]):
    for i in range(x1.shape[0]):
        f.write('%.12e\n ' % (n1s[k][i]))
f.close()

# zapis do wczytania do fortrana
f = open('n2s.txt', 'w') #note 'w' = write mode
for k in range(Cts.shape[0]):
    for i in range(x1.shape[0]):
        f.write('%.12e\n ' % (n2s[k][i]))
f.close()

f = open('mask.txt', 'w') #note 'w' = write mode
for i in range(x1.shape[0]):
    f.write('%f\n ' % (np.abs(fun4[i])>1e-6))
f.close()

# plt.tight_layout()
# filename=f"ntot.jpg"
# plt.savefig(filename,bbox_inches='tight', transparent=True)

# plt.show()



