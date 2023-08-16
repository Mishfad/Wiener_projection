import numpy as np

##############################################################################################################################
# Creates a function to generate data according to the restart and record format
# Parameters are 
# 1) Number of independent trajectories, 
# 2) NUmber of samples per trajectory
# 3) B and A
# 4) Noise power (noise is i.i.d. Gausian)
def restart_record_data_generation(nTrajectories,nSamples,B,A,noise_pow):
    nNodes=B.shape[0]
    # print(B)
    y=np.zeros((nTrajectories,nNodes,nSamples),dtype=complex)
    for ind_Traj in range(nTrajectories):
        # print(ind_Traj)
        x=np.zeros((nSamples,nNodes))
        x[0,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
        x[1,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[0,:]
        x[2,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[1,:]-A[:,1]*x[0,:]
        for ind_Samp in range(3,nSamples):
            x[ind_Samp,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[ind_Samp-1,:]-A[:,1]*x[ind_Samp-2,:]-A[:,2]*x[ind_Samp-3,:]
            # print(ind_Samp, "Previous sample: ",x[ind_Samp-1,:])
            # print(ind_Samp, "Present sample: ",x[ind_Samp,:])
            x[ind_Samp,:]=x[ind_Samp,:]+np.dot(B[:,:,0],x[ind_Samp-1,:])+np.dot(B[:,:,1],x[ind_Samp-2,:])+np.dot(B[:,:,2],x[ind_Samp-3,:])
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

def continuous_data_generation(nSamples,B,A,noise_pow):
    nNodes=B.shape[0]
    print("Generating continous data.. nSamples: ",nSamples," for nNodes :",nNodes)
    x=np.zeros((nSamples,nNodes))
    x[0,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
    x[1,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[0,:]
    x[2,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[1,:]-A[:,1]*x[0,:]
    for ind_Samp in range(3,nSamples):
        x[ind_Samp,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[ind_Samp-1,:]-A[:,1]*x[ind_Samp-2,:]-A[:,2]*x[ind_Samp-3,:]
        # print(ind_Samp, "Previous sample: ",x[ind_Samp-1,:])
        # print(ind_Samp, "Present sample: ",x[ind_Samp,:])
        x[ind_Samp,:]=x[ind_Samp,:]+np.dot(B[:,:,0],x[ind_Samp-1,:])+np.dot(B[:,:,1],x[ind_Samp-2,:])+np.dot(B[:,:,2],x[ind_Samp-3,:])
        # print("B*X(t-1)",np.dot(B,x[ind_Samp-1,:]))
        # print(x[ind_Samp,:])
        # print(x)
        # X=np.fft.fft(x,axis=0)
        # y=np.tile(x[:2,0],[3,1])
        # print(y)
        # print("Shape: ",X.shape)
        # print("\n\nfft:",X)
        # y[ind_traj,:,:]=np.transpose(X) #X is (nFreq,nNodes)
        # print("y:",y[ind_Traj,:,:])
    return x
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

##############################################################################################################################

def Do_continuous_data_generation(i,xi,nSamples,B,A,noise_pow):
    # print("running Do_continuous_data_generation... for node")
    print("Generating continous data.. nSamples: ",nSamples,"Intervention at node ",i)
    nNodes=B.shape[0]
    x=np.zeros((nSamples,nNodes))
    x[0,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
    x[0,i]=xi[0]
    x[1,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[0,:]+np.dot(B[:,:,0],x[0,:])
    x[1,i]=xi[1]
    x[2,:]=np.random.randn(nNodes)*np.sqrt(noise_pow) - A[:,0]*x[1,:] - A[:,1]*x[0,:] + np.dot(B[:,:,0],x[1,:])+np.dot(B[:,:,1],x[0,:])
    x[2,i]=xi[2]
    for ind_Samp in range(3,nSamples):
        x[ind_Samp,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[ind_Samp-1,:]-A[:,1]*x[ind_Samp-2,:]-A[:,2]*x[ind_Samp-3,:]
        x[ind_Samp,i]=xi[ind_Samp]
        x[ind_Samp,:]=x[ind_Samp,:]+np.dot(B[:,:,0],x[ind_Samp-1,:])+np.dot(B[:,:,1],x[ind_Samp-2,:])+np.dot(B[:,:,2],x[ind_Samp-3,:])
        x[ind_Samp,i]=xi[ind_Samp]
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
    # file_loc='/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/Python_codes/Wiener_projection_simulation/continous_data_generation/'
    # np.savez(file_loc+"AB_coeffs_conti",A=A,B=B.reshape(B.shape[0], -1))
    return A,B

