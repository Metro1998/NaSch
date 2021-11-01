import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import poisson, nbinom
from utilities.utilities import cal
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection='3d')
xx = np.arange(0.5, 1.05, 0.05)
yy = np.arange(0.25, 0.80, 0.05)
X, Y = np.meshgrid(xx, yy)

Z = np.load(file='../NaSch/result/z.npy')
Z = np.tile(np.array(Z), (2, 2))
print(Z.shape)
ax.plot_surface(X=X, Y=Y, Z=Z, alpha=0.3, camp='winter')
ax.contour(X, Y, Z, zdir='x', offset=-6, cmap="rainbow")
ax.contour(X, Y, Z, zdir='y', offset=6, cmap="rainbow")

ax.set_xlabel('P in Negative Binomial Distribution ')
ax.set_xlim(0.4, 1.1)
ax.set_ylabel('Mu in Poisson Distribution')
ax.set_ylim(0.15, 0.85)
ax.set_zlabel('Average Travel Time')
ax.set_zlim(0, 200)
ax.set_title('')

plt.show()

"""
fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.set(xlim=[-1, 5], ylim=[0, 1], title='Vehicle Arrival Fitting Curve (peak period)',
        xlabel='num of vehicles arrived (1/s)', ylabel='possibility')
ax2.set(xlim=[-1, 5], ylim=[0, 1], title='Vehicle Arrival Fitting Curve (flat period)',
        xlabel='num of vehicles arrived (1/s)', ylabel='possibility')
ax3.set(xlim=[-1, 5], ylim=[0, 1], title='Pedestrian Arrival Fitting Curve (peak period)',
        xlabel='num of pedestrians arrived (1/s)', ylabel='possibility')
ax4.set(xlim=[-1, 5], ylim=[0, 1], title='Pedestrian Arrival Fitting Curve (flat period)',
        xlabel='num of pedestrians arrived (1/s)', ylabel='possibility')
prob1 = poisson.pmf(k=np.arange(0, 5, 1), mu=16.50 / 300)
prob2 = poisson.pmf(k=np.arange(0, 5, 1), mu=17.67 / 300)
prob3 = nbinom.pmf(k=np.arange(0, 5, 1), n=335 / 300, p=0.9)
prob4 = nbinom.pmf(k=np.arange(0, 5, 1), n=340 / 300, p=0.83)
ax1.plot(np.arange(0, 5, 1), prob1)
for i in range(5):
    ax1.text(np.arange(0, 5, 1)[i], prob1[i], (round(prob1[i], 6)), color='grey')
ax2.plot(np.arange(0, 5, 1), prob2)
for i in range(5):
    ax2.text(np.arange(0, 5, 1)[i], prob2[i], (round(prob2[i], 6)), color='grey')
ax3.plot(np.arange(0, 5, 1), prob3)
for i in range(5):
    ax3.text(np.arange(0, 5, 1)[i], prob3[i], (round(prob3[i], 6)), color='grey')
ax4.plot(np.arange(0, 5, 1), prob4)
for i in range(5):
    ax4.text(np.arange(0, 5, 1)[i], prob4[i], (round(prob4[i], 6)), color='grey')
plt.tight_layout()

plt.show()
"""
"""
fig = plt.figure(figsize=(8, 9))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set(xlim=[-1, 5], ylim=[0, 1], title='Vehicle Arrival Fitting Curve (Poisson)',
        xlabel='num of vehicles arrived (1/s)', ylabel='possibility')
ax2.set(xlim=[-1, 5], ylim=[0, 1], title='Pedestrian Arrival Fitting Curve (Negative Binomial)',
        xlabel='num of pedestrians arrived (1/s)', ylabel='possibility')
prob1 = poisson.pmf(k=np.arange(0, 5, 1), mu=16.50 / 300)
prob2 = poisson.pmf(k=np.arange(0, 5, 1), mu=17.67 / 300)
prob4 = nbinom.pmf(k=np.arange(0, 5, 1), n=335 / 300, p=0.9)
prob3 = nbinom.pmf(k=np.arange(0, 5, 1), n=340 / 300, p=0.83)
ax1.plot(np.arange(0, 5, 1), prob1, label='peak period')
for i in range(3):
    ax1.text(np.arange(0, 5, 1)[i], prob1[i], (round(prob1[i], 3)), color='grey')
ax1.plot(np.arange(0, 5, 1), prob2, label='flat period')
ax2.plot(np.arange(0, 5, 1), prob3, label='peak period')
for i in range(3):
    ax2.text(np.arange(0, 5, 1)[i], prob3[i] + 0.02, (round(prob3[i], 3)), color='grey')
ax2.plot(np.arange(0, 5, 1), prob4, label='flat period')
for i in range(3):
    ax2.text(np.arange(0, 5, 1)[i], prob4[i], (round(prob4[i], 3)), color='grey')
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.show()
"""
