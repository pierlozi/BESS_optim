import numpy as np
import matplotlib.pyplot as plt

def eff_DG(alpha, beta, Pr_DG, P_DG):
    return 1/(alpha + beta * Pr_DG / P_DG)

def fuel_cons(alpha, beta, Pr_DG, P_DG):
    return alpha * P_DG  + beta * Pr_DG 

Pr_DG = 1
P_DG = np.linspace(0, Pr_DG)

# here I am plotting 
plt.plot(P_DG/Pr_DG, P_DG/fuel_cons(0.246, 0.08145, Pr_DG, P_DG), label = 'a=0.246 b=0.08145')
plt.plot(P_DG/Pr_DG, P_DG/fuel_cons(0.24, 0.084, Pr_DG, P_DG), label = 'a=0.24 b=0.084')
plt.xlabel('Power to Rated Power ratio[-]')
plt.ylabel('Efficiency [kWh/l]')
plt.legend(loc = 'best')

plt.grid(True)
plt.show()
plt.close()
