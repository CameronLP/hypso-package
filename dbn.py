"Translated from Matlab into Python by JLG - 11.2022"


import numpy as np
import sklearn.decomposition as de

def sig(x):
    '''
    sigmoid
    '''
    return 1/(1 + np.exp(-x))

class TrainingOptions():
    def __init__(self, MaxIter=5, BatchSize=200, Verbose=False, StepRatio=0.1,
                 InitialMomentum=0.5, FinalMomentum=0.9, WeightCost=0.2,
                 InitialMomentumIter=5, WeightLimit=2.0):
        self.MaxIter=MaxIter
        self.BatchSize=BatchSize
        self.Verbose=bool(Verbose)
        self.StepRatio=StepRatio
        self.InMomentum=InitialMomentum #momentum for first five iterations
        self.FiMomentum=FinalMomentum #momentum after that
        self.WeightCost=WeightCost #costs of weight update
        self.WeightLimit=WeightLimit
        self.InMoIter=InitialMomentumIter


class RBM():
    '''
    The Restricted Boltzmann Machine class
    '''
    def __init__(self, dimV, dimH, corr_limit=1.0, sep_rate=0.5, mean_zero=False,
                W_dr=True, b_dr=True, c_dr=True):
        self.W = (np.random.rand(dimV, dimH)-0.5)*0.1
        self.W -= np.mean(self.W, axis=0)
        #spacing = dimV // dimH
        #offset = spacing // 2
        #for i in range(dimH):
        #    self.W[spacing * i:spacing * (i+1), i] = 1.0
        #self.W -= np.mean(self.W, axis=0)
        #self.W /= np.sqrt(np.sum(self.W**2, axis=0))
        self.b = np.random.rand(dimH)*0.1 -0.5
        self.c = np.zeros(dimV)#np.random.rand(dimV)*0.1 -0.5
        self.to = TrainingOptions()
        self.corr_limit = corr_limit
        self.sep_rate = sep_rate
        self.mean_zero = mean_zero
        self.b_dr=b_dr
        self.W_dr=W_dr
        self.c_dr = c_dr
        ma = np.arange(dimH**2).reshape((dimH,dimH))
        m1 = ma // dimH
        m2 = ma % dimH
        self.tmask = m1 > m2
        
        #W_corr = W_new.transpose()@ W_new
        #W_adj = np.abs(W_corr)>max_corr
        #W_new = (1 + sep_rate) * W_new - sep_rate * W_new @ (W_corr*W_adj)
        
    def hid(self, pix):
        '''The hidden layer given a visible layer'''
        return sig(pix @ self.W + self.b)
    
    def vis(self, pix):
        '''The visible layer given a hidden layer'''
        return sig(pix @ self.W.T + self.c)
        
    def train(self, img):
        # initialize training
        N_pix = img.shape[0]
        N_hid = len(self.b)
        N_vis = len(self.c)
        
        #initialize deltas
        deltaW   = np.zeros((N_vis, N_hid))
        deltaB   = np.zeros(N_hid)
        deltaC   = np.zeros(N_vis)
        
        momentum = self.to.InMomentum
        
        if self.to.Verbose:
            H    = sig( img[::10] @ self.W + self.b  )
            Vr   = sig(  H @ self.W.T + self.c  )
            err  = np.power( img[::10] - Vr, 2 );
            rmse = np.sqrt( np.sum(err) / len(err) )
            print("starting", rmse)
        
        for it in range(self.to.MaxIter):
            if it > self.to.InMoIter:
                momentum = self.to.FiMomentum
            
            ind = np.random.permutation(N_pix)
            for batch in range(0,N_pix,self.to.BatchSize):
                bstart = batch #* self.to.BatchSize
                bend = np.min([bstart + self.to.BatchSize, N_pix-1])
                bind = ind[bstart:bend]
                
                #Gibbs sampling step 0
                vis0  = img[bind]
                hid0 = self.hid(vis0)
                #print(hid0)
                
                #Gibbs sampling step 1
                #hid0 = np.random.rand(*hid0.shape) < hid0
                vis1 = self.vis(hid0)

                hid1 = self.hid(vis1)
                vis2 = self.vis(hid1)
                hid2 = self.hid(vis2)
                
                posprods = hid0.T @ vis0
                negprods = hid2.T @ vis2
                
                dW = (posprods - negprods).T
                dB = (np.sum(hid0) - np.sum(hid2))
                dC = (np.sum(vis0) - np.sum(vis2))
                
                deltaW =  momentum * deltaW + (self.to.StepRatio / self.to.BatchSize) * dW
                deltaB =  momentum * deltaB + (self.to.StepRatio / self.to.BatchSize) * dB
                deltaC =  momentum * deltaC + (self.to.StepRatio / self.to.BatchSize) * dC
                
                #print(deltaC)
                
                #update network weights
                if self.W_dr:
                    self.W += deltaW #- self.to.WeightCost * self.W
                if self.b_dr:
                    self.b += deltaB - self.to.WeightCost * self.b
                if self.c_dr:
                    self.c += deltaC - self.to.WeightCost * self.c
                
                if self.mean_zero:
                    self.W -= self.W.mean(axis=0)
                    
                if self.corr_limit < 1:
                    #do a little de-correlation, asymmetrically
                    W_dm = self.W - self.W.mean(axis=0)
                    W = W_dm/np.sqrt(np.sum(W_dm**2, axis=0))
                    W_corr = W.transpose()@ W
                    W_adj = W_corr>self.corr_limit
                    W_adj[self.tmask] = False
                    W_new = (1 + self.sep_rate) * W - self.sep_rate * W @ (W_corr*W_adj)
                    W_dm = W_new * np.maximum(np.sqrt(np.sum(W_dm**2, axis=0)), W.shape[1]/W.shape[0])
                    self.W = W_dm  + self.W.mean(axis=0)
                
            if self.to.Verbose:
                H    = sig( img[::10] @ self.W + self.b  )
                Vr   = sig(  H @ self.W.T + self.c  )
                err  = np.power( img[::10] - Vr, 2 );
                rmse = np.sqrt( np.sum(err) / len(err) )
                print(it, rmse)
                #print(self.c.mean(), self.c.std())
                
class DBN():
    def __init__(self, n_bands=12,
                 training_options_DBN=TrainingOptions(BatchSize = 5000,
                                                      StepRatio = 0.9,
                                                      MaxIter = 30,
                                                      WeightCost = 0.01,
                                                      WeightLimit=1.0),
                 training_options_RBM1=TrainingOptions(MaxIter=10),
                 training_options_RBM=TrainingOptions(),
                 Verbose=False):
        self.to1 = training_options_DBN
        self.toRBM1 = training_options_RBM1
        self.to = self.to1
        self.toRBM = training_options_RBM
        if Verbose:
            self.to.Verbose=True
            self.toRBM.Verbose=True
            self.toRBM1.Verbose=True
        self.N_hidden = [n_bands]
        self.bound = np.argmin(self.N_hidden)
    
    def _fit(self, img):
        Nodes = [img.shape[-1], *self.N_hidden, img.shape[-1]]
        N_rbm = len(Nodes) - 1
        if self.to1.Verbose:
            print([(Nodes[i], Nodes[i+1]) for i in range(N_rbm)])
        self.rbm = [RBM(Nodes[i], Nodes[i+1]) for i in range(N_rbm)]
        for rbm in self.rbm:
            rbm.to = self.toRBM
        self.scale = img.max()
            
    def fit(self, img):
        self._fit(img)
        self.rbm[0].to = self.toRBM1
        self.rbm[0].corr_limit = 0.0
        self.rbm[0].sep_rate = 0.1
        #self.rbm[0].mean_zero=True
        #self.rbm[0].c_dr = False
        if self.to1.Verbose:
            print("starting joint RBM training")
        self.train_joint_rbm(img/self.scale)
        self.rbm[0].corr_limit = 1.0
        self.to = self.to1
        self.train(img/self.scale)

    def H_all(self, img):
        allH = []
        v = np.copy(img)
        for rbm in self.rbm:
            H = rbm.hid(v)
            allH.append(H)
            v = H
        return allH
    
    def transform(self, img):
        rbm = self.rbm[0]
        v = rbm.hid(img/self.scale)
        for i in range(1,self.bound+1):
            rbm = self.rbm[i]
            v = rbm.hid(v)
        return v.transpose()
        
    def inverse_transform(self, dr):
        rbm = self.rbm[self.bound + 1]
        v = rbm.hid(dr.transpose())
        for i in range(self.bound+2, len(self.rbm)):
            rbm = self.rbm[i]
            v = rbm.hid(v)
        return v*self.scale

    def make_saving_dict(self):
        saving_dict = {}
        saving_dict["W0"] =  self.rbm[0].W
        saving_dict["W1"] =  self.rbm[1].W
        saving_dict["b0"] =  self.rbm[0].b
        saving_dict["b1"] =  self.rbm[1].b
        saving_dict["scale"] = self.scale
        return saving_dict

    def save(self, filename):
        """
        assumes just a 2 layers of rbms
        """
        saving_dict = {}
        saving_dict["W0"] =  self.rbm[0].W
        saving_dict["W1"] =  self.rbm[1].W
        saving_dict["b0"] =  self.rbm[0].b
        saving_dict["b1"] =  self.rbm[1].b
        saving_dict['scale'] = self.scale
        np.savez(filename, **saving_dict)

    def trained_weights(self):
        return self.make_saving_dict()


    def load(self, filename):
        "we assume it's already trained"
        loaded = np.load(filename)
        self.N_hidden = [loaded["W0"].shape[-1]]
        Nodes = [loaded["W0"].shape[0],
                 *self.N_hidden,
                 loaded["W0"].shape[0]]
        N_rbm = len(Nodes) - 1
        self.rbm = [RBM(Nodes[i], Nodes[i+1]) for i in range(N_rbm)]
        
        self.rbm[0].W = loaded["W0"]
        self.rbm[1].W = loaded["W1"]
        self.rbm[0].b = loaded["b0"]
        self.rbm[1].b = loaded["b1"]
        self.scale = loaded["scale"]

    def load_from_props(self, W0, W1, b0, b1, scale):
        #meta parameters already defined, should only be used when they are known
        self.N_hidden = [W0.shape[-1]]
        Nodes = [W0.shape[0], *self.N_hidden, W0.shape[0]]
        N_rbm = len(Nodes) - 1
        self.rbm = [RBM(Nodes[i], Nodes[i + 1]) for i in range(N_rbm)]

        self.rbm[0].W = W0
        self.rbm[1].W = W1
        self.rbm[0].b = b0
        self.rbm[1].b = b1
        self.scale = scale
        
    
    def endecode(self, img, r=-1):
        if r == -1:
            r = self.N_hidden[0]
        if r != self.N_hidden[0]:
            print("note that dbns work only with a fixed number of bands")
        return self.inverse_transform(self.transform(img))
        
    def train_joint_rbm(self, img):
        #v = np.copy(img) # I don't like doing this, but I don't see a good way around it
        # we could skip the first if the hidden layers are small
        for i, rbm in enumerate(self.rbm):
            if self.to1.Verbose:
                print("training rbm: ", i)
            if (i > 0) & (len(self.rbm)==2):
                rbm.W[:] = self.rbm[i-1].W.T[:]
                rbm.b[:] = self.rbm[i-1].c[:]
                rbm.c[:] = self.rbm[i-1].b[:]
                rbm.train(v)
                v = rbm.hid(v)
            elif i == 0:
                rbm.train(img)
                v = rbm.hid(img)
            else:
                rbm.train(v)
                v = rbm.hid(img)
    
    def train(self, img):
        N_rbm = len(self.rbm)
        N_pix = len(img)
        
        if self.to.Verbose:
            rmse = np.sqrt(np.sum((self.endecode(img) - img)**2)/len(img))
            print(rmse/self.scale)

        deltaDbn = DBN(n_bands=self.N_hidden[0])
        deltaDbn._fit(img)
        
        momentum = self.to.InMomentum
        for it in range(self.to.MaxIter):
            if it > self.to.InMoIter:
                momentum = self.to.FiMomentum

            it_steprat = self.to.StepRatio * (1-it/self.to.MaxIter)
        
            ind = np.random.permutation(N_pix)
            for batch in range(0,N_pix,self.to.BatchSize):
                bstart = batch 
                bend = np.min([bstart + self.to.BatchSize, N_pix-1])
                bind = ind[bstart:bend]
                
                all_H = self.H_all(img[bind])
                
                for n in range(len(all_H)-1,-1,-1):
                    derSgm = np.multiply(all_H[n], ( 1 - all_H[n] ))                    
                    
                    if ( n+1 > (len(self.rbm)-1) ):
                        der = (all_H[-1] - img[bind])
                    else:
                        der = np.multiply(derSgm, (der @ self.rbm[n+1].W.T ))
                                          
                    if ( n-1 > -1 ):
                        inn = np.concatenate((np.ones((len(bind), 1)), all_H[n-1]), axis=1)
                    else:   
                        inn = np.concatenate((np.ones((len(bind),1)), img[bind]), axis=1)
    
                        
                    deltaWb = inn.T @ der / len(bind)
                    deltab  = deltaWb[0]
                    deltaW  = deltaWb[1:]
                                          
                    deltaDbn.rbm[n].W = momentum * deltaDbn.rbm[n].W  
                    deltaDbn.rbm[n].b = momentum * deltaDbn.rbm[n].b
                    deltaDbn.rbm[n].W = deltaDbn.rbm[n].W - it_steprat * deltaW
                    deltaDbn.rbm[n].b = deltaDbn.rbm[n].b - it_steprat * deltab
                    
                
                for i, rbm in enumerate(self.rbm):       
                    #testing code
                    #change this to an update option
                    
                    if rbm.W_dr:
                        #wl = self.to.WeightCost * rbm.W * (np.abs(rbm.W) > self.to.WeightLimit)
                        rbm.W += deltaDbn.rbm[i].W #-  wl
                    if rbm.b_dr:
                        wl = self.to.WeightCost * rbm.b * (np.abs(rbm.b) > self.to.WeightLimit)
                        rbm.b += deltaDbn.rbm[i].b - wl
                    
                    if rbm.mean_zero:
                        # always leave one non-zero to accord with PCA
                        rbm.W[:,1:] -= rbm.W[:,1:].mean(axis=0)
                    
                    if rbm.corr_limit < 1:         
                    #do a little de-correlation, asymmetrically
                        W_dm = rbm.W - rbm.W.mean(axis=0)
                        W = W_dm/np.sqrt(np.sum(W_dm**2, axis=0))
                        W_corr = W.transpose()@ W
                        W_adj = np.abs(W_corr)>rbm.corr_limit
                        W_adj[rbm.tmask] = False
                        W_new = (1 + rbm.sep_rate) * W - rbm.sep_rate * W @ (W_corr*W_adj)
                        W_dm = W_new * np.sqrt(np.sum(W_dm**2, axis=0)) 
                        rbm.W = W_dm + rbm.W.mean(axis=0)
                                          
            if self.to.Verbose:
                rmse = np.sqrt(np.sum((self.endecode(img) - img)**2)/len(img))
                print(rmse/self.scale)
                
        return 0 
    
        
