import numpy as np
import scipy.signal as sig
from itertools import combinations


#########################################################################################
# functions from Wiener_projection
#########################################################################################


def Least_Squares_Solution(X,Z):
    # Project X on to Z, Z=a0x_0+a1x1+...am xm, 
    # where m is the number of columns of X
    X_star = X.conj().T
    XX=np.matmul(X_star, X)
    if np.isscalar(XX):
        return np.matmul(np.divide( X_star,XX), Z)
    else:
        return np.matmul(np.matmul(np.linalg.inv(np.matmul(X_star, X)), X_star), Z)

def Wiener_Proj(X,Z):
    # Project X on to Z, Z=a0x_0+a1x1+...am xm, 
    # where m is the number of columns of X
    return Least_Squares_Solution(X,Z)

#########################################################################################
# Compute full Wiener coeffs for each frequency
#########################################################################################

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
            W1[freq_index,:]=np.insert(Least_Squares_Solution(X_bar,Z),proj_ind,0)
        # W1_max=np.amax(abs(W1),axis=0)
        W[proj_ind,:,:]=W1
        W_mag[proj_ind,:]=np.amax(abs(W1),axis=0)
    return W,W_mag
#########################################################################################
# Partial Wiener Coeffs for conditional uncorrelatedness test (for example, in PC test)
#########################################################################################
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
    for freq_index in range(nSamples):
        fft_data=data[:,:,freq_index]
        # print(fft_data.shape)
        # print(freq_index)
        X=fft_data[:,i]
        X_bar=fft_data[:,c]
        # print(Z.shape,X_bar.shape)
        # print(X_bar[:3,:])
        W[freq_index,:]=Least_Squares_Solution(X_bar,X)
    W_mag=np.amax(abs(W),axis=0)
    return W,W_mag


#########################################################################################
# PC test with Partial Wiener Coeffs for conditional uncorrelatedness test 
#########################################################################################
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
                
#########################################################################################
# Wiener filtering in time, VAR based approach with non-causal Wiener filtering
#########################################################################################
def Wiener_time(x,i,L=5):
    # ||xi-YB||^2
    m=(2*L+1)
    xi=x[:,i]
    C=[ind for ind in range(x.shape[1])]
    C=np.delete(C,i)
    y=np.zeros([x.shape[0],m*C.shape[0]])
    for ind in range(x.shape[0]-L-1,L,-L):
        y[ind,:]=x[ind-L:ind+L+1,C].reshape(m*len(C))
        # to reshape back, use y[ind,:].reshape((2*L+1),len(C)). Need to perform it with Wiener coefficients
    
    ## Try with CVX
    
    ########################### 
    beta=Least_Squares_Solution(xi,y)
    w=beta.reshape(m,len(C))
    # fig, ax1 = plt.subplots()
    h_inf_mag=np.zeros(x.shape[1])
    for ind_nodes in range(len(C)):
        print(w[:,ind_nodes])
        # sys=tf(w[ind_nodes,:],np.append([np.zeros(L-1)],1))
        # omega = np.logspace(-4, 2, 1001)
        # H = sys(omega * 1j)
        # # Find the highest
        # print(np.abs(H).max())
        [freq,r]=sig.freqz(w[:,ind_nodes],np.append([np.zeros(L-1)],1),50)
        h_inf_mag[C[ind_nodes]]=r.max()
        print(ind_nodes, r.max())
        # ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(r))
        # ax2.plot(freq, angles)
        # ax2.set_ylabel('Angle(radians)')
        # ax2.grid(True)
        # ax2.axis('tight')
    # plt.show()
    return h_inf_mag
    # print(np.max(np.abs(w),0))
    # print(np.abs(w)>1e-2)
    # w_FFT=np.fft.fft(w)
    # print(np.abs(w_FFT)) # 0,2,4-->0,1,3
    # print("place holder")
