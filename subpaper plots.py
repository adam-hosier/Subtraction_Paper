from cmath import inf
from xml.etree.ElementTree import QName
import matplotlib.pyplot as plt 
import numpy as np 
import scipy
import math as m 
import lmfit 
import scipy.integrate as integrate

def pot1(x, a, V):
    return V*(x/a)**2

def pot2(x, a, V):
    return V*(2*np.log(x/a)+1)

def potT(x, a, V):
    return np.piecewise(x, [x<=a, x>a], [lambda x: pot1(x, a, V), lambda x: pot2(x, a, V)])

def funa(x, Q, kT, I, a, V):
    return np.exp(-Q*pot1(x, a, V)/kT)

def fun1(x, Q, kT, I, a, V):
    return np.exp(-Q*potT(x, a, V)/kT)

def fun2(x, Q, KT, I, a, V):
    return np.exp(-Q*pot2(x, a, V)/kT)

r_ebeam = 35*10**(-6)
Q = 76
Qlight = 10
I = 1                ## normalized intensity 
a = r_ebeam       
delta = 1     ##micrometers
ep0 = 8.854187*10**(-12)
ke = 8.8987551*10**(9)
mel = 9.11*10**(-31)
eVJ = 1.60218*10**(-19)
kb = 1.380649*10**(-23)
tempest = 1e7 #3.35e7
#V = 15              ##Volts
kT = kb*tempest
echarge = 1.60217663*10**(-19)
xp = np.linspace(-400, 400, num=10000)*10**(-6)
xp2 = np.linspace(0, 5000, num=10000)*10**(-6)

Ie = 0.150                  #current of ebeam A
Eebeam = 18*1000            #energy of ebeam eV
ve = np.sqrt(2*echarge*Eebeam/mel)
lame = Ie / ve
lami = 0.5*lame
lami1 = 0.6*lame 
lami2 = 0.5*lame 
V_e = lame / (4*np.pi*ep0)
V_i = lami / (4*np.pi*ep0)
V_i1 = lami1 / (4*np.pi*ep0)
V_i2 = lami2 / (4*np.pi*ep0)
V = V_e


idist = fun1(xp2, 65*echarge, kT, I, a, V)
idist66 = fun1(xp2, 66*echarge, kT, I, a, V)
idistlight = fun1(xp2, Qlight*echarge, kT, I, a, V)
maxval = np.max(idist)/2
maxval66 = np.max(idist66)/2
maxvallight = np.max(idistlight)/2
ind1 = np.where(idist<=maxval)[0][0]
ind66 = np.where(idist66<=maxval66)[0][0]
indlight = np.where(idistlight<=maxvallight)[0][0]
ioncloudw = xp2[ind1]
ioncloudw66 = xp2[ind66]
ioncloudwlight = xp2[indlight]


#################################

Vdtube = 0

shiftVe = np.max(np.abs(potT(xp2, a, -V_e)))
shiftVi = np.max(np.abs(potT(xp2, ioncloudw, V_i)))
shiftVi60p = np.max(np.abs(potT(xp2, ioncloudw, V_i1)))
shiftVi50p = np.max(np.abs(potT(xp2, ioncloudw, V_i2)))
shiftVi66 = np.max(np.abs(potT(xp2, ioncloudw66, V_i)))
shiftVilight = np.max(np.abs(potT(xp2, ioncloudwlight, V_i)))
ebeampot = -(potT(xp2, a, -V_e)+shiftVe)

ionpot = -(potT(xp2, ioncloudw, V_i)-shiftVi)
ionpot60p = -(potT(xp2, ioncloudw, V_i1)-shiftVi60p)
ionpot50p = -(potT(xp2, ioncloudw, V_i2)-shiftVi50p)
ionpot66 = -(potT(xp2, ioncloudw66, V_i)-shiftVi66)
ionpotlight = -(potT(xp2, ioncloudwlight, V_i)-shiftVilight)

sumpot = (-(potT(xp2, ioncloudw, V_i)-shiftVi))+(-(potT(xp2, a, -V_e)+shiftVe))
sumpot60p = (-(potT(xp2, ioncloudw, V_i1)-shiftVi60p))+(-(potT(xp2, a, -V_e)+shiftVe))
sumpot50p = (-(potT(xp2, ioncloudw, V_i2)-shiftVi50p))+(-(potT(xp2, a, -V_e)+shiftVe))
sumpot66 = (-(potT(xp2, ioncloudw66, V_i)-shiftVi66))+(-(potT(xp2, a, -V_e)+shiftVe))
sumpotlight = (-(potT(xp2, ioncloudwlight, V_i)-shiftVilight))+(-(potT(xp2, a, -V_e)+shiftVe))


newboltz11_ = np.exp(-Qlight*echarge*(sumpot)/(kT))
newboltz11_only = np.exp(-Qlight*echarge*(ebeampot)/(kT))
newboltz65_ = np.exp(-65*echarge*(sumpot)/(kT))
newboltz66_ = np.exp(-66*echarge*(sumpot)/kT)
newboltz11_66 = np.exp(-Qlight*echarge*(sumpot66)/(kT))

##light ion  boltz
boltzli = np.exp(-Qlight*echarge*(ionpotlight)/(kT))
boltzlisum = np.exp(-Qlight*echarge*(sumpotlight)/(kT))

volnum = r_ebeam*2*np.pi*1*10**(-2)
mag = lame
esig = r_ebeam/(np.sqrt(2*np.log(2)))
edist = mag*np.exp(-(xp2)**2 / (2*esig**2))

beamlen = 1*10**(-2)
beamarea = np.pi*r_ebeam**2

K = lame*(1e-2) / (esig**2)
# print(ve)
# print(lame/(ve*beamarea*echarge*1e10))
# print(volnum)
qint = integrate.quad(lambda x:(2/(esig*np.sqrt(2*np.pi)))*np.exp(-(x)**2 / (2*esig**2)), 0, 500e-6)
mag = qint[0]

qint2 = integrate.quad(lambda x:(K)*x*np.exp(-(x)**2 / (2*esig**2)), 0, 500e-6)
mag2 = qint2[0]
#edist = (1/mag)*np.exp(-(xp2)**2 / (2*esig**2))
edist = (mag)*np.exp(-(xp2)**2 / (2*esig**2))
# print(lame*1e-2)
# print('here ', qint2)
ilim = 500e-5#r_ebeam*1
bint1 = integrate.quad(lambda x:(np.exp(-Qlight*echarge*((-(potT(x, ioncloudw, V_i)-shiftVi))+(-(potT(x, a, -V_e)+shiftVe)))/(kT))),0,ilim)
bnorm1 = bint1[0]
eint1 = integrate.quad(lambda x:(1/bnorm1)*(np.exp(-Qlight*echarge*((-(potT(x, ioncloudw, V_i)-shiftVi))+(-(potT(x, a, -V_e)+shiftVe)))/(kT)))*(K*x*np.exp(-(x)**2 / (2*esig**2))),0,ilim)

bint2 = integrate.quad(lambda x:(np.exp(-Qlight*echarge*((-(potT(x, ioncloudw, V_i)-shiftVi))+(-(potT(x, a, -V_e)+shiftVe)))/(kT))),0,ilim)
bnorm2 = bint2[0]
eint2 = integrate.quad(lambda x:(1/bnorm2)*(np.exp(-Qlight*echarge*((-(potT(x, ioncloudw, V_i)-shiftVi))+(-(potT(x, a, -V_e)+shiftVe)))/(kT))),0,ilim)

bint3 = integrate.quad(lambda x:(np.exp(-Qlight*echarge*((-(potT(x, ioncloudw66, V_i)-shiftVi))+(-(potT(x, a, -V_e)+shiftVe)))/(kT))),0,ilim)
bnorm3 = bint3[0]
eint3 = integrate.quad(lambda x:(np.exp(-Qlight*echarge*((-(potT(x, ioncloudw66, V_i)-shiftVi))+(-(potT(x, a, -V_e)+shiftVe)))/(kT)))*(K*x*np.exp(-(x)**2 / (2*esig**2))),0,ilim)

bint4 = integrate.quad(lambda x:(np.exp(-Qlight*echarge*((-(potT(x, ioncloudw66, V_i)-shiftVi))+(-(potT(x, a, -V_e)+shiftVe)))/(kT))),0,ilim)
bnorm4 = bint4[0]
eint4 = integrate.quad(lambda x:(np.exp(-Qlight*echarge*((-(potT(x, ioncloudw66, V_i)-shiftVi))+(-(potT(x, a, -V_e)+shiftVe)))/(kT))),0,ilim)

bint9 = integrate.quad(lambda x:(np.exp(-Qlight*echarge*((-(potT(x, a, -V_e)+shiftVe)))/(kT))),0,ilim)
bnorm9 = bint9[0]
eint9 = integrate.quad(lambda x:(1/bnorm9)*(np.exp(-Qlight*echarge*((-(potT(x, a, -V_e)+shiftVe)))/(kT)))*(K*x*np.exp(-(x)**2 / (2*esig**2))),0,ilim)

bint10 = integrate.quad(lambda x:(np.exp(-Qlight*echarge*((-(potT(x, a, -V_e)+shiftVe)))/(kT))),0,ilim)
bnorm10 = bint10[0]
eint10 = integrate.quad(lambda x:(1/bnorm10)*(np.exp(-Qlight*echarge*((-(potT(x, a, -V_e)+shiftVe)))/(kT))),0,ilim)

evalvolume = np.pi*(ilim**2)*(1e-2)*(1e2)**3
#print(lame*1e-2/(1e12*1e-2*echarge*np.pi*r_ebeam**2)*(1e-2)**3) 
#print(ve)
##average electron density for light ion
print('electron beam: ', lame*1e-2/(1e12*1e-2*echarge*np.pi*r_ebeam**2)*(1e-2)**3)
print('q = ', str(Qlight),'+ & 65+ ',eint1[0]/(echarge*eint2[0]*evalvolume*1e12))       #w.r.t. Q=65+
print('q = ', str(Qlight),'+ & 66+ ', eint3[0]/(echarge*eint4[0]*evalvolume*1e12))       #w.r.t. Q = 66+
print('q = ', str(Qlight),'+ only: ', eint9[0]/(echarge*eint10[0]*evalvolume*1e12))



plt.figure() 
#plt.title('potential')
plt.axvline(x=r_ebeam*1e6, c='k', ls='--', label='Radius of electron beam')
plt.axvline(x=ioncloudw*1e6, c='tab:grey',ls='--', label='Radius of ion cloud')
plt.plot(xp2*1e6, -(potT(xp2, a, -V_e)+shiftVe), label='Electron beam')
plt.plot(xp2*1e6, ionpot, c='r', label='heavy ion cloud (70%)')
#plt.plot(xp2*1e6, -(potT(xp2, ioncloudwlight, V_i)-shiftVilight), label='light ion cloud')
plt.plot(xp2*1e6, sumpot,c='g', label='heavy sum (70%)')
#plt.plot(xp2*1e6, (-(potT(xp2, ioncloudwlight, V_i)-shiftVilight))+(-(potT(xp2, a, -V_e)+shiftVe)), label='light sum')

#plt.plot(xp2*1e6, -(potT(xp2, a, -V_e)+shiftVe), label='ebeam')
plt.plot(xp2*1e6, ionpot60p,c='r',ls='-.', label='heavy ion cloud (60%)')
plt.plot(xp2*1e6, sumpot60p,c='g',ls='-.', label='heavy sum (60%)')

plt.plot(xp2*1e6, ionpot50p,c='r',ls='--', label='heavy ion cloud (50%)')
plt.plot(xp2*1e6, sumpot50p,c='g',ls='--', label='heavy sum (50%)')


#plt.plot(xp2, Q65_ebeam_pot, label='total potential')
plt.xlim(np.min(0), 5000)
plt.ylabel('Potential (V)')
plt.xlabel('Radial distance (micrometers)')
plt.legend()
plt.show()
plt.close() 

#print(1e6*xp2)
plt.figure() 
plt.axvline(x=r_ebeam*1e6, ls='--',c='k', label='Radius of Electron Beam')
plt.plot(xp2*1e6, edist/mag, label='e-beam distribution')
plt.plot(xp2*1e6, newboltz65_/np.max(newboltz65_), label='Q = 65+')
plt.plot(xp2*1e6, newboltz66_/np.max(newboltz66_), label='Q = 66+')
plt.plot(xp2*1e6, newboltz11_/np.max(newboltz11_), label='Q = '+str(Qlight)+'+ & heavy ions (65+)')
plt.plot(xp2*1e6, newboltz11_66/np.max(newboltz11_66), label='Q = '+str(Qlight)+'+ & heavy ions (66+)')
#plt.plot(xp2*1e6, boltzli/np.max(boltzli), label='Q = 11+ (ion only potential)')
#plt.plot(xp2*1e6, newboltz11_only/np.max(newboltz11_only), label='Q = 11+ (ebeam potential only)')
plt.plot(xp2*1e6, boltzlisum/np.max(boltzlisum), label='Q = '+str(Qlight)+'+')
plt.xlim(np.min(0), 500)
#plt.xlim(np.min(0), 10000)
#plt.xscale('log')
plt.ylim(0, 1.05)
plt.ylabel('Relative radial distribution (arb)')
plt.xlabel('Radial distance (micrometers)')
#plt.ylim(0,1.5)
plt.legend()
plt.show()
plt.close() 