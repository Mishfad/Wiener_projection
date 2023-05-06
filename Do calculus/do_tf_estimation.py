import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from itertools import combinations
import control as ct
# from Wiener_projection import *


def phase_shift(w):
    # Shifts the phase from [-pi,pi] to [0,2pi]
    # receives a vector of phases and returns the modified phase
    for ind in range(w.shape[0]):
        # print(w.shape)
        # w_mod=w[ind]
        if w[ind]<0:
            w[ind]=2*np.pi+w[ind]
    return w

#########################################################################################
# functions from Wiener_projection
def Wiener_Proj(X,Z):
    # Project X on to Z, Z=a0x_0+a1x1+...am xm, 
    # where m is the number of columns of X
    X_star = X.conj().T
    XX=np.matmul(X_star, X)
    if np.isscalar(XX):
        return np.matmul(np.divide( X_star,XX), Z)
    elif XX.all==0:
        return np.complex128(0)
    else:
        # print("X:",X)
        return np.matmul(np.matmul(np.linalg.inv(np.matmul(X_star, X)), X_star), Z)


def compute_Wiener_coeffs(nNodes,data):
    # Returns W and W_mag
    # W: nNodes X nSamples X nNodes
    # W_mag: returns h-infinity norm over frequencies
    nSamples=data.shape[2]
    W=np.zeros([nNodes,nSamples,nNodes],dtype='complex')
    W_mag=np.zeros([nNodes,nNodes])
    for proj_ind in range(nNodes):
        # print("proj_ind=",proj_ind)
        W1=np.zeros([nSamples,nNodes],dtype='complex')
        # print(W1.shape)
        for freq_index in range(nSamples):
            fft_data=data[:,:,freq_index]
            # print(fft_data.shape)
            # print(freq_index)
            Z=fft_data[:,proj_ind]
            X_bar=np.delete(fft_data,proj_ind,1)
            # print(Z.shape,X_bar.shape)
            # print(X_bar[:3,:])
            W1[freq_index,:]=np.insert(Wiener_Proj(X_bar,Z),proj_ind,0)
        # W1_max=np.amax(abs(W1),axis=0)
        W[proj_ind,:,:]=W1
        W_mag[proj_ind,:]=np.amax(abs(W1),axis=0)
    return W,W_mag

def compute_partial_Wiener_coeffs(data,i,j,z=[]):
    # Returns W and W_mag
    # W: nNodes X nSamples X nNodes
    # W_mag: returns h-infinity norm over frequencies
    c=np.append(j,np.int32(z))
    # c=np.unique(c)
    indexes = np.unique(c, return_index=True)[1]
    c=[c[index] for index in sorted(indexes)]
    n1=len(c)
    nSamples=data.shape[2]
    W=np.zeros([nSamples,n1],dtype='complex')
    W_mag=np.zeros([n1])
    print(i,j,z)
    for freq_index in range(nSamples):
        fft_data=data[:,:,freq_index]
        # print(fft_data.shape)
        # print(i,c,freq_index)
        X=fft_data[:,i]
        X_bar=fft_data[:,c]
        # print(Z.shape,X_bar.shape)
        # print(c,freq_index,X_bar[:3,:])
        W[freq_index,:]=Wiener_Proj(X_bar,X)
    W_mag=np.amax(abs(W),axis=0)
    return W,W_mag


def Run_PC_Wiener_test(data):
    nNodes=data.shape[1]
    C={}
    for ind1 in range(nNodes):
        for ind2 in range(nNodes):
            if ind1>=ind2:
                continue
            flag=False
            # the complement set through which we condition
            D=[i for i in range(nNodes) if i != ind1 and i !=ind2]
            print("D ",D)
            # iterate over the combinations of various size in the increasing order
            for ind_cond in range(nNodes-1):
                if flag==True:
                    break
                combination_set=[i for i in combinations(D,ind_cond)]
                # print("combination"+str(ind_cond)+" =",combination_set)
                for c in combination_set:
                    # print("\n combination set: ",c)
                    W_i_jZ,W_mag_i_jZ=compute_partial_Wiener_coeffs(data,ind1,ind2,c)
                    print("Projection coefficient of X"+str(ind1+1)+" on "+str(np.append(ind2,c)+1)+"=",W_mag_i_jZ)
                    if W_mag_i_jZ[0]<0.1:
                        print("Separation set of ( X"+str(ind1+1)+", X"+str(ind2+1),") is : ",np.array(c)+1)
                        # c=[l+1 for l in c]
                        C[ind1,ind2]=c
                        flag=True
                        break
                        c=np.array([ind1])
                    if np.shape(c)[0]==nNodes-2:
                        print("No separation set for ( X"+str(ind1+1)+", X"+str(ind2+1),")")
                        C[ind1,ind2]=None
    return C
                


def plot_topology(adjacency_matrix):
    # import pygraphviz as pgv
    nNodes=adjacency_matrix.shape[0]
    G=nx.Graph()
    G.add_nodes_from(['X'+str(ind) for ind in range(1,nNodes+1)])
    # G.add_nodes_from([(2,{"color":"red"}),(3,{"color":"green"})])
    for ind1 in range(nNodes):
        for ind2 in range(nNodes):
            if adjacency_matrix[ind1,ind2]:
                # print(ind1,ind2)
                G.add_edge('X'+str(ind1+1),'X'+str(ind2+1))
# explicitly set positions
    pos = {'X'+str(1): (-1, 0.3), 'X'+str(2): (0, 0.05), 
           'X'+str(3): (1.7, 0.22), 'X'+str(4): (4.5, 0.285), 
           'X'+str(5): (4.5, 0.03), 'X'+str(6): (8, 0.00)}
    options = {
    "font_size": 26,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5
    }
    nx.draw_networkx(G, pos, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.001)

    # nx.draw_networkx(G, with_labels=True, 
    #                     font_weight='bold', 
    #                     font_size=30, node_size=4000, arrowsize=14)


def plot_directed_graph(adjacency_matrix):
    # import pygraphviz as pgv
    nNodes=adjacency_matrix.shape[0]
    G=nx.DiGraph()
    G.add_nodes_from(['X'+str(ind) for ind in range(1,nNodes+1)])
    # G.add_nodes_from([(2,{"color":"red"}),(3,{"color":"green"})])
    for ind1 in range(nNodes):
        for ind2 in range(nNodes):
            if adjacency_matrix[ind1,ind2]:
                # print(ind1,ind2)
                G.add_edge('X'+str(ind2+1),'X'+str(ind1+1))
    # explicitly set positions
    pos = {'X'+str(1): (-1, 0.3), 'X'+str(2): (0, 0), 
           'X'+str(3): (1.7, 0.22), 'X'+str(4): (4, 0.255), 
           'X'+str(5): (4.5, 0.03), 'X'+str(6): (8, 0.13)}
    options = {
    "font_size": 26,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 3,
    "width": 2,
    }

    nx.draw_networkx(G, pos, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.001)
    # nx.draw_networkx(G, with_labels=True, 
    #                     font_weight='bold', 
    #                     font_size=30, node_size=4000, arrowsize=14)



##############################################################################################################################
# Creates a function to generate data according to the restart and record format
# Parameters are 
# 1) Number of independent trajectories, 
# 2) NUmber of samples per trajectory
# 3) B and A
# 4) Noise power (noise is i.i.d. Gausian)
def restart_record_data_generation(nTrajectories,nSamples,B,A,noise_pow):
    print("Generating restart record data.. nfft: ",nSamples,"nTrajectories: ",nTrajectories)
    nNodes=B.shape[0]
    y=np.zeros((nTrajectories,nNodes,nSamples),dtype=complex)
    for ind_Traj in range(nTrajectories):
        x=np.zeros((nSamples,nNodes))
        x[0,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
        x[1,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[0,:]
        x[2,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[1,:]-A[:,1]*x[0,:]
        for ind_Samp in range(3,nSamples):
            x[ind_Samp,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[ind_Samp-1,:]-A[:,1]*x[ind_Samp-2,:]-A[:,2]*x[ind_Samp-3,:]
            x[ind_Samp,:]=x[ind_Samp,:]+np.dot(B[:,:,0],x[ind_Samp-1,:])+np.dot(B[:,:,1],x[ind_Samp-2,:])+np.dot(B[:,:,2],x[ind_Samp-3,:])
        X=np.fft.fft(x,axis=0)
        y[ind_Traj,:,:]=np.transpose(X) #X is (nFreq,nSamples)
    return y
##############################################################################################################################        

def continuous_data_generation(nSamples,B,A,noise_pow):
    print("Generating continous data.. nSamples: ",nSamples)
    nNodes=B.shape[0]
    x=np.zeros((nSamples,nNodes))
    x[0,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
    x[1,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[0,:]+np.dot(B[:,:,0],x[0,:])
    x[2,:]=np.random.randn(nNodes)*np.sqrt(noise_pow) - A[:,0]*x[1,:] - A[:,1]*x[0,:] + np.dot(B[:,:,0],x[1,:])+np.dot(B[:,:,1],x[0,:])
    for ind_Samp in range(3,nSamples):
        x[ind_Samp,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[ind_Samp-1,:]-A[:,1]*x[ind_Samp-2,:]-A[:,2]*x[ind_Samp-3,:]
        x[ind_Samp,:]=x[ind_Samp,:]+np.dot(B[:,:,0],x[ind_Samp-1,:])+np.dot(B[:,:,1],x[ind_Samp-2,:])+np.dot(B[:,:,2],x[ind_Samp-3,:])
    return x
##############################################################################################################################

##############################################################################################################################
# Creates a function to generate data according to the restart and record format
# Parameters are 
# 1) i: X_i that is performing do-operation
# 2) xi: the value that Xi is being set to (a vector or list of length nfft(nSamples here))
# 3) Number of independent trajectories, 
# 4) NUmber of samples per trajectory
# 5) B and A
# 6) Noise power (noise is i.i.d. Gausian)

def DO_restart_record_data_generation(i,xi,nTrajectories,nSamples,B,A,noise_pow):
    print("Generating data.. nSamples: ",nSamples,"nTrajectories: ",nTrajectories,"Intervention at",i)
    nNodes=B.shape[0]
    # print(B)
    y=np.zeros((nTrajectories,nNodes,nSamples),dtype=complex)
    for ind_Traj in range(nTrajectories):
        # print(ind_Traj)
        x=np.zeros((nSamples,nNodes))
        x[0,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
        x[0,i]=xi[0]
        x[1,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[0,:]
        x[1,i]=xi[1]
        x[2,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[1,:]-A[:,1]*x[0,:]
        x[2,i]=xi[2]
        for ind_Samp in range(3,nSamples):
            x[ind_Samp,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[ind_Samp-1,:]-A[:,1]*x[ind_Samp-2,:]-A[:,2]*x[ind_Samp-3,:]
            x[ind_Samp,i]=xi[ind_Samp]
            # print(ind_Samp, "Previous sample: ",x[ind_Samp-1,:])
            # print(ind_Samp, "Present sample: ",x[ind_Samp,:])
            x[ind_Samp,:]=x[ind_Samp,:]+np.dot(B[:,:,0],x[ind_Samp-1,:])+np.dot(B[:,:,1],x[ind_Samp-2,:])+np.dot(B[:,:,2],x[ind_Samp-3,:])
            x[ind_Samp,i]=xi[ind_Samp]
            # print("B*X(t-1)",np.dot(B,x[ind_Samp-1,:]))
            # print(x[ind_Samp,:])
        # print(x)
        # y=np.tile(x[:2,0],[3,1])
        # print(y)
        X=np.fft.fft(x,axis=0)
        # print("Shape: ",X.shape)
        # print("\n\nfft:",X)
        y[ind_Traj,:,:]=np.transpose(X) #X is (nFreq,nSamples)
        # print("y:",y[ind_Traj,:,:])
    return y
##############################################################################################################################        

def compute_fft(data,nfft):
    nSamples,nNodes=data.shape
    nTrajectories=np.int32(nSamples/nfft) 
    y=np.zeros((nTrajectories,nNodes,nfft),dtype=complex)
    for ind in range(nTrajectories-1): # discard final few residual samples
        x=data[ind*(nfft):(ind+1)*nfft,:]
        X=np.fft.fft(x,axis=0)
        y[ind,:,:]=np.transpose(X) #X is (nfft,nNodes)
    return y



def create_network(nNodes=None):
# Creates the graph with nNodes.
# Returen Adj 
    Adj1=np.zeros((nNodes,nNodes))
    # Adj1[0][1]=1; 
    Adj1[1,0]=1 # 1->2
    # Adj1[2,0]=1 # 1->3
    Adj1[2,1]=1 # 2->3
    # Adj1[4,0]=1 # 1->5
    Adj1[4,1]=1 # 2->5
    Adj1[4,2]=1 # 3->5
    Adj1[4,3]=1 # 4->5
    Adj1[5,4]=1 # 5->6
    return Adj1

def create_A_B_coeffs(Adj,filter=None):
    # Create A and B matrix for IIR transfer functions
    # A is coeffs of interaction with the past of self, 
    # B interaction with the other nodes 
    # returns A,B
    nNodes=Adj.shape[0]
    # A=(np.random.uniform(0,1,size=(nNodes,3)))
    # A=np.ones((nNodes,3))
    A=np.array([[1, .6,.3],[1, .4,.2],[1, .7,.4],[1, .5,.3],[1, .6,.3],[1, .7,.3] ])
    A1=np.random.uniform(0,1,size=(nNodes,1))
    A_scale=np.repeat(A1, 3, axis=1)
    A=A*A_scale
    B=np.random.uniform(0,1,size=(nNodes,nNodes,3))
    Adj=np.repeat(Adj[:, :, np.newaxis], 3, axis=2)
    B=B*Adj
    B[:,:,1:3]=0
    # print("\nA=",A,"\nB0=",B[:,:,0],"\nB2=",B[:,:,2])
    # print("\nB2=",B[:,:,2])
    # np.savez('/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/Do calculus/AB_coeffs',A=A,B=B.reshape(B.shape[0], -1))
    return A,B

def plot_histogram(d,colr='#0504aa',bin='auto',title='My Very Own Histogram'):
    h, xedges, yedges,image = plt.hist2d(x=np.real(d),y=np.imag(d), bins=bin, color=colr)
    # n, bins, patches = plt.hist(x=d, bins=bin, color=colr,
    #                         alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title(title, fontsize=10)
    # plt.text(23, 45, r'$\mu=15, b=3$')
    # maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    # print(np.amin(d))
    # plt.xlim(left=-25,right=np.maximum(30,np.amax(abs(d))))
    # plt.ylim(left=-25,right=np.maximum(30,np.amax(abs(d))))


##############################################################################################################################        
#############       main part of the file
##############################################################################################################################
nNodes=6
nTrajectories=10000
nSamples=100000
nfft=64
noise_pow= np.ones((nNodes)) # 2*np.random.uniform(0,1,size=(nNodes)) #

# Create the network and the Adjacency matrix
# print(np.sqrt(noise_pow))
Adj=create_network(nNodes)


# print("Adjacency matrix=",Adj)
A,B=create_A_B_coeffs(Adj)

AB=np.load('/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/Do calculus/AB_coeffs.npz')
A=AB['A']
B=AB['B'].reshape(nNodes,nNodes,3)

# print("A:", A, "\nB:",B)

# x_data is (nTrajectories,nNodes,nSamples), where the 3rd dimension corresponds to different frequencies
x=restart_record_data_generation(nTrajectories, nfft,B,A,noise_pow)

##########################################################################################################################
# Full Projection
W,W_mag=compute_Wiener_coeffs(nNodes,x)
print("W_mag on full projection=\n",W_mag)
moral_graph=W_mag>.1


fig = plt.figure(tight_layout=True)
gsp = gs.GridSpec(2, 2)
plt.get_current_fig_manager().resize(2700,1500)
ax = fig.add_subplot(gsp[0, 0])
plot_directed_graph(Adj)
ax.set_title('Generative graph')
# plt.figtext(0.15,.96,'Generative graph', fontsize=18, ha='left')

ax = fig.add_subplot(gsp[0, 1])
ax.set_title('Estimated topology Wiener coeffs, nSample='+str(nSamples)+' nfft: '+str(nfft))
# plt.figtext(0.75,.96,'Estimated topology', fontsize=18, ha='left')
plot_topology(moral_graph)
# file_name='/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/continous_data_generation/Wiener_filter.pdf'
# plt.savefig(file_name,format='pdf')


# ax = fig.add_subplot(gsp[1, 0])
# ax.set_title('Estimated directed graph')
# plot_directed_graph(moral_graph)
ax = fig.add_subplot(gsp[1, 1])
ax.set_title('Phase response of (2,3) ')
plt.plot(phase_shift(np.angle(W[2,:,3])))
plt.axis([0, nfft, 0, 6.3])
plt.show(block=False)

##########################################################################################################################

# y1= np.random.uniform(-1,1,[nfft]) 
y1= np.zeros([nfft]); y1[0]=1
y2= np.zeros([nfft]); y2[0]= y1[0]+1#2*np.ones([nfft])

# data_do_x1_y1=DO_restart_record_data_generation(0,y1,nTrajectories, nfft,B,A,noise_pow)
# data_do_x1_y2=DO_restart_record_data_generation(0,y2,nTrajectories, nfft,B,A,noise_pow)
data_do_x2_y1=DO_restart_record_data_generation(1,y1,nTrajectories, nfft,B,A,noise_pow)
data_do_x2_y2=DO_restart_record_data_generation(1,y2,nTrajectories, nfft,B,A,noise_pow)

# temp1=data_do_x1_y1.reshape(data_do_x1_y1.shape[0], -1)
# temp2=data_do_x1_y2.reshape(data_do_x1_y2.shape[0], -1)
# temp3=data_do_x2_y1.reshape(data_do_x2_y1.shape[0], -1)
# temp4=data_do_x2_y2.reshape(data_do_x2_y2.shape[0], -1)

# np.savez('/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/Do calculus/data_do_X1_y1',
#          data=data_do_x1_y1.reshape(data_do_x1_y1.shape[0], -1))
# np.savez('/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/Do calculus/data_do_X1_y2',
#          data=data_do_x1_y2.reshape(data_do_x1_y2.shape[0], -1))
# np.savez('/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/Do calculus/data_do_X2_y1',
#          data=data_do_x2_y1.reshape(data_do_x2_y1.shape[0], -1))
# np.savez('/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/Do calculus/data_do_X2_y2',
#          data=data_do_x2_y2.reshape(data_do_x2_y2.shape[0], -1))


# data=np.load('/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/Do calculus/data_do_X1_y1.npz')
# data_do_x1_y1=data['data'].reshape(nTrajectories,nNodes,nfft)
# data=np.load('/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/Do calculus/data_do_X1_y2.npz')
# data_do_x1_y2=data['data'].reshape(nTrajectories,nNodes,nfft)
# data=np.load('/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/Do calculus/data_do_X2_y1.npz')
# data_do_x2_y1=data['data'].reshape(nTrajectories,nNodes,nfft)
# data=np.load('/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/Do calculus/data_do_X2_y2.npz')
# data_do_x2_y2=data['data'].reshape(nTrajectories,nNodes,nfft)


# w01_do_x1_y1,w01_mag=compute_partial_Wiener_coeffs(data_do_x1_y1,0,1,[])
# w10_do_x1_y1,w10_mag=compute_partial_Wiener_coeffs(data_do_x1_y1,1,0,[])
# w01_do_x1_y2,w21_mag=compute_partial_Wiener_coeffs(data_do_x1_y2,0,1,[])
# w10_do_x1_y2,w12_mag=compute_partial_Wiener_coeffs(data_do_x1_y2,1,0,[])

# w01_do_x2_y1,w01_mag=compute_partial_Wiener_coeffs(data_do_x2_y1,0,1,[])
# w10_do_x2_y1,w10_mag=compute_partial_Wiener_coeffs(data_do_x2_y1,1,0,[])
# w01_do_x2_y2,w21_mag=compute_partial_Wiener_coeffs(data_do_x2_y2,0,1,[])
# w10_do_x2_y2,w12_mag=compute_partial_Wiener_coeffs(data_do_x2_y2,1,0,[])


# fig = plt.figure(tight_layout=True)
# gsp = gs.GridSpec(2, 2)
# xvals=np.arange(nfft)
# plt.get_current_fig_manager().resize(2700,1500)
# # for ind in range(nNodes):
# ax = fig.add_subplot(gsp[0,0])
# ax.plot(xvals,abs(w01_do_x1_y1),label='W_01 | do(X1=y1)')
# ax.plot(xvals,abs(w01_do_x1_y2),label='W_01 | do(X1=y2)')
# ax.plot(xvals,abs(w01_do_x1_y2-w01_do_x1_y1),label='diff')
# ax.set_title('Wiener coeff of projecting X'+str(1)+"to X2|do(X1)", fontsize=10, ha='center')
# ax.legend()

# ax = fig.add_subplot(gsp[0,1])
# ax.plot(xvals,abs(w10_do_x1_y1),label='W_10 | do(X1=y1)')
# ax.plot(xvals,abs(w10_do_x1_y2),label='W_10 | do(X1=y2)')
# ax.plot(xvals,abs(w10_do_x1_y2-w10_do_x1_y1),label='diff ')
# # ax.axis([0, nfft, 0, 6.3])
# ax.set_title('Wiener coeff of projecting X'+str(2)+"to X1|do(X1)", fontsize=10, ha='center')
# ax.legend()

# ax = fig.add_subplot(gsp[1,0])
# ax.plot(xvals,abs(w01_do_x2_y1),label='W_01 | do(X2=y1)')
# ax.plot(xvals,abs(w01_do_x2_y2),label='W_01 | do(X2=y2)')
# ax.plot(xvals,abs(w01_do_x2_y2-w01_do_x2_y1),label='diff')
# ax.set_title('Wiener coeff of projecting X'+str(1)+"to X2|do(X2)", fontsize=10, ha='center')
# ax.legend()

# ax = fig.add_subplot(gsp[1,1])
# ax.plot(xvals,abs(w10_do_x2_y1),label='W_10 | do(X2=y1)')
# ax.plot(xvals,abs(w10_do_x2_y2),label='W_10 | do(X2=y2)')
# ax.plot(xvals,abs(w10_do_x2_y2-w10_do_x2_y1),label='diff | do(X1=y2)')
# # ax.axis([0, nfft, 0, 6.3])
# ax.set_title('Wiener coeff of projecting X'+str(2)+"to X1|do(X2)", fontsize=10, ha='center')
# ax.legend()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
w10,w10_mag=compute_partial_Wiener_coeffs(x,1,0,[])
w54,w54_mag=compute_partial_Wiener_coeffs(x,5,4,[])
# w01,w01_mag=compute_partial_Wiener_coeffs(x,0,1,[])
response10 = w10.ravel()
response54 = w54.ravel()
# response01 = w01.ravel()
freqs = 2*np.pi/nfft*np.arange(nfft)
F10 = ct.frd(response10, freqs)
F54 = ct.frd(response54, freqs)

num10=np.array([0])
den=np.array([1])
num10=(np.append(num10,B[1,0,:]))
num54=np.flip(np.append(np.array([0]),B[5,4,:]))
den1=(np.append(den,A[1,:]))
den5=(np.append(den,A[5,:]))
# print(num,den)
sys1=ct.tf(num10,den1)
sys54=ct.tf(num54,den5)
# print(sys1)
plt.figure()
freqlog=np.logspace(-1.2,.8,200)
# mag_H01,phase_H01,omega_H01=ct.bode_plot(sys1,label="$H_{10}$",dB = True, Hz = True, deg='False')
ct.bode_plot(sys1,freqlog,label="$H_{10}$",dB = True, Hz = True, deg='False')
ct.bode_plot(sys54,freqs,label="$H_{54}$",dB = True, Hz = True)
ct.bode_plot(F10,freqs,label="W_01",dB = True, Hz = True)
ct.bode_plot(F54,freqs,label="W_54",dB = True, Hz = True)
plt.legend()
# plt.show(False)

# fig = plt.figure(tight_layout=True)
# plt.semilogx(omega_H01,mag_H01,label='$H_{01}$')
# plt.semilogx(omega_W01,mag_W01,label='$W_10_do(X1=y1)$')
# plt.legend()
# # phas_W01=2*np.pi-phase_W01
# phase_W01=phase_W01%(2*np.pi)*180/np.pi
# plt.semilogx(omega_H01,(phase_H01),label='$H_{01}$')
# plt.semilogx(omega_W01,np.unwrap(phase_W01),label='$W_10_do(X1=y1)$')

# mag_H01,phase_H01,omega_H01=ct.bode_plot(F2,freqs,label="W_10_do(X1=y2)",dB = True, Hz = True)
# ct.bode_plot(F1,freqs,deg='False',label="W_10_do(X1=y1)")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# x3_1=DO_restart_record_data_generation(2,y1,nTrajectories, nfft,B,A,noise_pow)

# w32,w32_mag=compute_partial_Wiener_coeffs(x3_1,2,1,[])
w52  ,w52_mag =compute_partial_Wiener_coeffs(x,4,1,[2])
w52_2,w52_mag2=compute_partial_Wiener_coeffs(data_do_x2_y2,4,1,[2])
response52_1 = (w52[:,0]).ravel()
response52_2 = w52_2[:,0].ravel()
F52_1 = ct.frd(response52_1, freqs)
F52_2 = ct.frd(response52_2, freqs)
F52_diff = ct.frd(response52_2-response52_1, freqs)

num=np.array([0])
den=np.array([1])
num=np.flip(np.append(num,B[4,1,:]))
den=np.flip(np.append(den,A[4,:]))
# print(num,den)
sys52=ct.tf(num,den)
# print(sys1)
plt.figure()
mag_H52,phase_H52,omega_H52=ct.bode_plot(sys52,label="$H_{52}$",dB = True, Hz = True, deg='False')
mag_W52,phase_W052,omega_W52=ct.bode_plot(F52_1,freqs,label="W_52_do(X2=y1)",dB = True, Hz = True)
# mag_W52,phase_W052,omega_W52=ct.bode_plot(F52_2,freqs,label="W_52_do(X2=y2)",dB = True, Hz = True)
mag_W52,phase_W052,omega_W52=ct.bode_plot(F52_diff,freqs,label="W_52:do--diff",dB = True, Hz = True)
plt.legend()
plt.show()

#########################################################################################################################
# PC test
print("Running PC test")
# C=Run_PC_Wiener_test(original_data)
Top_est=np.zeros([nNodes,nNodes])
# for c in C:
#     # print("\n",c)
#     if C[c]==None: Top_est[c]=1
# Top_est=Top_est+Top_est.T
# W_mag
# plt.figure()
# # ax=plt.gca
# # ax.set_title("Estimated topology")
# plot_topology(Top_est)
# ax=plt.gca()
# ax.set_title("Estimated topology PC-Wiener: continous data_generation, nSample="+str(nSamples)+" nfft: "+str(nfft))
# plt.get_current_fig_manager().resize(2700,1500)
# # plt.savefig('/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/continous_data_generation/Top_est_PC_Wiener.pdf',format='pdf')
# ##########################################################################################################################
# plt.show()
# fig=plt.get_current_fig_manager()
# print("phase=",np.angle(W[2,:,3]),"\n Average=",np.average(np.abs(np.angle(W[2,:,3]))))