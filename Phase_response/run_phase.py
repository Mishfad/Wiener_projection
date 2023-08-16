import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os
import sys 

directory = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(directory) # # Getting the parent directory name where the current directory is present.
sys.path.append(parent_path)  # # setting path... # equialent to  sys.path.append('../Wiener_projection')
##############################################################################################################################        
# From src/* userdefined library
##############################################################################################################################        
from src.network_functions import continuous_data_generation
from src.network_functions import create_network
from src.network_functions import create_A_B_coeffs
from src.network_functions import restart_record_data_generation
from src.Wiener_filter import Wiener_time
from src.Wiener_filter import compute_Wiener_coeffs

from src.plot_functions import plot_directed_graph
from src.plot_functions import plot_phase_response
from src.plot_functions import plot_topology
from src.plot_functions import phase_shift
##############################################################################################################################        

##############################################################################################################################        
#############       main part of the file
##############################################################################################################################
nNodes=6
nTrajectories=10000
nSamples=64
noise_pow= np.ones((nNodes)) # 2*np.random.uniform(0,1,size=(nNodes)) #

# Create the network and the Adjacency matrix
# print(np.sqrt(noise_pow))
Adj=create_network(nNodes)

###################################################################################################################
SMALL_SIZE = 11
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig = plt.figure(tight_layout=True)
gsp = gs.GridSpec(2, 2)
ax = fig.add_subplot(gsp[0, 0])
plot_directed_graph(Adj)
ax.set_title('Generative graph')
# plt.figtext(0.15,.96,'Generative graph', fontsize=18, ha='left')
plt.get_current_fig_manager().resize(2700,1500)

print(Adj)
# A,B=create_A_B_coeffs(Adj)

AB=np.load('/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/working_phase/AB_coeffs.npz')
A=AB['A']
B=AB['B'].reshape(nNodes,nNodes,3)

print("A:", A, "\nB:",B)

# x_data is (nTrajectories,nNodes,nSamples), where the 3rd dimension corresponds to different frequencies
y=restart_record_data_generation(nTrajectories,nSamples,B,A,noise_pow)

original_data=y
print("Max value =",np.amax(np.abs(original_data)))
original_data= np.divide( original_data,np.amax(np.abs(original_data)))
print("Max value after scaling=",np.amax(np.abs(original_data)))
nNodes=original_data.shape[1]
W,W_mag=compute_Wiener_coeffs(nNodes,original_data)
print("W_mag=",W_mag)
W_avg=(W_mag+np.transpose(W_mag))/2
print(W_avg)

moral_graph=W_avg>.1
# print("phase=",np.angle(W[2,:,3]),"\n Average=",np.average(np.abs(np.angle(W[2,:,3]))))


ax = fig.add_subplot(gsp[0, 1])
ax.set_title('Estimated topology')
# plt.figtext(0.75,.96,'Estimated topology', fontsize=18, ha='left')
plot_topology(moral_graph)
ax = fig.add_subplot(gsp[1, 0])
ax.set_title('Estimated directed graph')
plot_directed_graph(moral_graph)
ax = fig.add_subplot(gsp[1, 1])
ax.set_title(r'$\angle W_2[3]$')
plt.plot(phase_shift(np.angle(W[2,:,3])))
plt.axis([0, nSamples, 0, 6.3])

plt.show(block=False)
# move_figure(fig,0,0)
# plot_digraph()
# fig.savefig('/Users/mishfad/Library/CloudStorage/GoogleDrive-veedu002@umn.edu/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/working_phase/Graphs.pdf',format='pdf')
plot_phase_response(W)

plt.figure()
plt.plot(W_avg.ravel(),range(nNodes*nNodes),'r*')
plt.show()