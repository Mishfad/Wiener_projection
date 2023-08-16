import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os
import sys 
##############################################################################################################################
# do_conti_RR=1 : continuous sampling
# do_conti_RR=0 : Restart and record sampling
do_conti_RR=1
##############################################################################################################################

##############################################################################################################################        
# Find current and paretn directory to add src functions
##############################################################################################################################        
# # directory reach
directory = os.path.dirname(os.path.realpath(__file__))
# # Getting the parent directory name where the current directory is present.
parent_path = os.path.dirname(directory)
print(directory)
print(parent_path)
# # setting path
sys.path.append(parent_path)  # equialent to  sys.path.append('../Wiener_projection')
##############################################################################################################################        
# From Wiener_projection/src/* userdefined library
##############################################################################################################################        
from src.network_functions import continuous_data_generation
from src.network_functions import create_network
from src.network_functions import create_A_B_coeffs
from src.network_functions import restart_record_data_generation
from src.network_functions import Do_continuous_data_generation
from src.network_functions import DO_restart_record_data_generation
from src.network_functions import compute_fft

from src.Wiener_filter import Wiener_time
from src.Wiener_filter import compute_Wiener_coeffs

from src.plot_functions import plot_directed_graph
from src.plot_functions import plot_phase_response
from src.plot_functions import plot_topology
from src.plot_functions import phase_shift
from src.plot_functions import plot_histogram
##############################################################################################################################        

SMALL_SIZE = 13
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

##############################################################################################################################        
#             main part of the file
##############################################################################################################################
nNodes=6
nSamples=100000
nfft=64
nTrajectories=int(nSamples/nfft)
noise_pow= np.ones((nNodes)) # 2*np.random.uniform(0,1,size=(nNodes)) #

#############################################################################################
# Create the network and the Adjacency matrix
Adj=create_network(nNodes)
# print("Adjacency matrix=",Adj)
A,B=create_A_B_coeffs(Adj)
# AB=np.load('AB_coeffs.npz')
# A=AB['A']
# B=AB['B'].reshape(nNodes,nNodes,3)
# print("A:", A, "\nB:",B)

## Initialize y1 and y2 to perform intervention
y1_1= np.zeros([nfft]); y1_1[0:(int(nfft/2)-1)]=0        
y2_1= np.zeros([nfft]); y2_1[0:int(nfft/2)-1]=2
y1=np.tile(y1_1,(int(np.ceil(nSamples/nfft)),1)).flatten()
y2=np.tile(y2_1,(int(np.ceil(nSamples/nfft)),1)).flatten()

########################################################################################
# compute time-series with intervention at node 2
# do_conti_RR=1 : continous; 0
# do_conti_RR=0 : Restart and record
if do_conti_RR:
    TS_do_x2_y1=Do_continuous_data_generation(1,y1,nSamples,B,A,noise_pow)
    TS_do_x2_y2=Do_continuous_data_generation(1,y2,nSamples,B,A,noise_pow)
    ## Compute DFT
    data_do_x2_y1=compute_fft(TS_do_x2_y1,nfft)
    data_do_x2_y2=compute_fft(TS_do_x2_y2,nfft)
else:
    data_do_x2_y1=DO_restart_record_data_generation(1,y1,nTrajectories, nfft,B,A,noise_pow)
    data_do_x2_y2=DO_restart_record_data_generation(1,y2,nTrajectories, nfft,B,A,noise_pow)

########################################################################################
#           Plot some histograms to verify that the distributions are Gaussian              
########################################################################################
nCount=6
fig = plt.figure(tight_layout=True)
gsp = gs.GridSpec(2, nCount)
plt.get_current_fig_manager().resize(2700,1500)
freq_index=1
for ind in range(6):
    ax = fig.add_subplot(gsp[0,ind])
    plot_histogram(np.imag(data_do_x2_y1[:,ind,freq_index]),bin=10,title='X'+str(ind+1)+'/do(X2=y1), Freq_index:'+str(freq_index))
    ax = fig.add_subplot(gsp[1,ind])
    plot_histogram(np.imag(data_do_x2_y2[:,ind,freq_index]),bin=10,title='X'+str(ind+1)+'/do(X2=y2), Freq_index:'+str(freq_index))

# plt.savefig('Do_histogram_freq_1.pdf',format='pdf')

###################################################################################################################
# computing the statistics of $\hat{X}_i$
avg1=np.zeros([nNodes,nfft],dtype=np.complex128)
std1=np.zeros([nNodes,nfft],dtype=np.complex128)
avg2=np.zeros([nNodes,nfft],dtype=np.complex128)
std2=np.zeros([nNodes,nfft],dtype=np.complex128)
for ind in range(nNodes):
    for freq_index in range(1,nfft):
        avg1[ind,freq_index]=np.mean(data_do_x2_y1[:,ind,freq_index])
        avg2[ind,freq_index]=np.mean(data_do_x2_y2[:,ind,freq_index])
        std1[ind,freq_index]=np.std(data_do_x2_y1[:,ind,freq_index])
        std2[ind,freq_index]=np.std(data_do_x2_y2[:,ind,freq_index])
        # print('\nX'+str(ind+1)+'/do(X2=y1), Freq_index:'+str(freq_index), 'Mean',np.mean(data_do_x2_y1[:,ind,freq_index]), "Std:",np.std(data_do_x2_y1[:,ind,freq_index]))
        # std1[ind,freq_index]=np.std(data_do_x2_y1[:,ind,freq_index])
        # print('\nX'+str(ind+1)+'/do(X2=y1), Freq_index:'+str(freq_index), 'Mean',np.mean(data_do_x2_y1[:,ind,freq_index]), "Std:",)

########################################################################################
#       Plot the statistics of the estimated distributions
########################################################################################

fig = plt.figure(tight_layout=True)
gsp = gs.GridSpec(4, nNodes-1)
xvals=np.arange(nfft)
omega_N=[ r""+eval("\"$f_{"+str(xvals[ind])+"}$"+"\"") for ind in range(0,np.size(xvals),15)]
idx=[ind for ind in range(0,np.size(xvals),15)]
plt.get_current_fig_manager().resize(2700,1500)
for ind in range(nNodes-1):
    ax = fig.add_subplot(gsp[0,ind])
    ax.plot(xvals,np.real(avg1[ind,:]),label='do(y1)')
    ax.plot(xvals,np.real(avg2[ind,:]),label='do(y2)')
    ax.plot(xvals,np.real(avg2[ind,:]-avg1[ind,:]),label='diff')
    # ax.axis([0, nfft, 0, 6.3])
    ax.set_title(r'$\Re${Mean('+eval('\'$X_'+str(ind+1)+'$\'')+")}", fontsize=SMALL_SIZE, ha='center')
    plt.xticks(xvals[idx],omega_N)
    ax.set_xlabel('DFT points')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(.05,0.05))

    # plt.tight_layout()
    
#------------------------------------------------------------------------------------------
    ax = fig.add_subplot(gsp[1,ind])
    ax.plot(xvals,np.imag(avg1[ind,:]),label='do(y1)')
    ax.plot(xvals,np.imag(avg2[ind,:]),label='do(y2)')
    ax.plot(xvals,np.imag(avg2[ind,:]-avg1[ind,:]),label='diff')
    # ax.axis([0, nfft, 0, 6.3])
    ax.set_title(r'$\Im${Mean('+eval('\'$X_'+str(ind+1)+'$\'')+")}", fontsize=SMALL_SIZE, ha='center')
    plt.xticks(xvals[idx],omega_N)
    ax.set_xlabel('DFT points')
#------------------------------------------------------------------------------------------
    ax = fig.add_subplot(gsp[2,ind])
    ax.plot(xvals,std1[ind,:],label='do(y1)')

    ax.plot(xvals,std2[ind,:],label='do(y2)')
    ax.set_title(r'$\sigma$('+eval('\'$X_'+str(ind+1)+'$\'')+")", fontsize=SMALL_SIZE, ha='center')
    plt.xticks(xvals[idx],omega_N)
    ax.set_xlabel('DFT points')
    # plt.legend(bbox_to_anchor=(1.05, 1.0))#, loc='upper right')

# plt.savefig('Do_conti_statistics_full.pdf',format='pdf')


###########################################################################################################
## Drawing the generative graph for reference
fig = plt.figure(tight_layout=True)
gsp = gs.GridSpec(2, 2)
plt.get_current_fig_manager().resize(2700,1500)
ax = fig.add_subplot(gsp[0, 0])
plot_directed_graph(Adj)
ax.set_title('Generative graph')
plt.show()
# plt.figtext(0.15,.96,'Generative graph', fontsize=18, ha='left')