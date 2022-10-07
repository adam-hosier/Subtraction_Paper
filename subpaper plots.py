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

Q = 76
I = 1                ## normalized intensity 
a = 100       
delta = 1     ##micrometers
ep0 = 8.854187*10**(-12)
ke = 8.8987551*10**(9)
V = 10              ##Volts
kT = 100 #.01*Q*echarge*V
echarge = 1.60217663*10-19
xp = np.linspace(-400, 400, num=10000)
xp2 = np.linspace(0.1, 400, num=1000)
xp2a = np.linspace(a+delta, 400)

testf = fun1(xp2, Q, kT, I, a, V)


testint2 = integrate.quad(lambda x: fun1(x, Q, kT, I, a, V), 0, inf)
inorm = testint2[0]


iepot = []
ipot2 = []

for i in range(len(xp2)):
    ionpot = integrate.quad(lambda x: (Q/inorm)*fun1(x, Q, kT, I, a, V), 0.01+1, xp2[i])
    ipot2.append(ke*ionpot[0]/xp2[i]+1)
    epot = potT(xp2[i], a, V)
    iepot.append(epot+ke*ionpot[0]/xp2[i])



# plt.figure() 
# plt.title('Potential')
# plt.plot(xp2, pot1(xp2, a, V), label='potential 1 (r/a)^2')
# plt.plot(xp2, potT(xp2, a, V), label='e-beam potential')
# plt.plot(xp2, ipot2, label='ion potential')
# plt.plot(xp2a, pot2(xp2a, a, V), label='potential 2 (ln)')
# #plt.plot(xp2, iepot, label='ion + e-beam potential')
# plt.xlim(np.min(xp2), np.max(xp2))
# plt.legend()
# plt.show()
# plt.close() 


plt.figure() 
plt.title('boltzmann distributions')
#plt.plot(xp2, funa(xp2, Q, kT, I, a, V), label='potential 1 (r/a)^2')
plt.plot(xp2, fun1(xp2, Q, kT, I, a, V), label='total potential')
#plt.plot(xp2a, fun2(xp2a, Q, kT, I, a, V), label='potential 2 (ln)')
plt.xlim(np.min(0), np.max(xp2))
#plt.ylim(0, 100)
plt.legend()
plt.show()
plt.close() 