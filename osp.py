"""
This code was written as part of the ELO-Hyp project, and is not yet published, so please don't share it
Eventually it will be available in more official channels

JLGarrett 2023.12.01

"""

import numpy as np
from scipy.linalg import pinv

def atgp(img, n=0):
    if n == 0:
        n = img.shape[-1]

    targets = np.zeros((n, img.shape[-1]))
    Po = np.eye(img.shape[-1])
    sigs = np.zeros(n, dtype=np.uint64)

    for i in range(n):
        orth = img@Po
        sigs[i] = np.argmax(np.multiply(orth, orth).sum(axis=1))
        #print(np.mean((orth**2).sum(axis=-1)))
        targets[i,:] = orth[sigs[i]]
        #The normalization doesn't really matter, but...
        targets[i,:] = targets[i,:]#/np.sqrt((targets[i]**2).sum())
        U = np.array([targets[j] for j in range(i+1)])
        Po = np.eye(img.shape[-1])-pinv(U)@U


    return targets, sigs

class OSP():
    def __init__(self, n_bands=0):
        self.n_bands = n_bands

    def fit(self, img):
        if self.n_bands == 0:
            self.n_bands = img.shape[-1]
        self.proj, _ = atgp(img, self.n_bands)

    def set_r(self, r):
        if r == -1:
            r = self.n_bands
        return r

    def transform(self, img, r=-1):
        r = self.set_r(r)
        return self.proj[:r]@(img).transpose()

    def inverse_transform(self, tr_img, r=-1):
        r = self.set_r(r)
        inv = pinv(self.proj[:r])
        return (inv@tr_img).transpose()
    
    def trained_weights(self):
        props = {
            'projection' : self.proj,
         }
        return props

    def endecode(self, img, r=-1):
        r = self.set_r(r)
        return self.inverse_transform(self.transform(img, r), r)

    def save(self, file_name):
        np.savez(file_name, osp_proj=self.proj)

    def load(self, file_name):
        loaded = np.load(file_name)
        self.proj = loaded['osp_proj']
        self.n_bands = self.proj.shape[-1]
        
    def load_from_props(self, projection):
        #meta parameters already defined, should only be used when they are known
        self.proj = projection
        self.n_bands = self.proj.shape[-1]


def atgpII(img, n=0):
    if n == 0:
        n = img.shape[-1]
        
    targets = np.zeros((n, img.shape[-1]), dtype=np.float32)
    sigs = np.zeros(n, dtype=np.uint64)
    mysigs = np.zeros(img.shape[0], dtype=np.float64)
    mysigs[:] = np.multiply(img, img).sum(axis=-1)
    img_copy = np.zeros(img.shape, dtype=np.float32)
    img_copy[:] = img
    
    for i in range(n):
        sigs[i] = np.argmax(mysigs)
        pre_target = img_copy[sigs[i]].astype(np.float32)
        targets[i,:] = pre_target/np.sqrt((pre_target**2).sum())
        img_copy -= np.outer((img_copy@targets[i,:]),targets[i])
        mysigs[:] = np.multiply(img_copy, img_copy).sum(axis=-1)
        #for j in range(i):
            #print(targets[j], pre_target)
        #    print(i, j, targets[j,:]@pre_target, mysigs[sigs[i]])
        #    pre_target -= (targets[j,:]@pre_target)*targets[j,:]
        #    #print((((targets[j,:]@pre_target)*pre_target)**2).sum())
        #targets[i,:] = pre_target/np.sqrt((pre_target**2).sum())
        #mysigs -= np.float32(img)@targets[i]
        #print(img[sigs[i]]@targets[i], (img[sigs[i]]@targets[i])**2)
        #print(mysigs.shape)
        #print(img@targets[i])
        #print(np.mean(mysigs))
        #get signal
        
        #sigs[i] = np.argmax(np.multiply(orth, orth).sum(axis=1))
        #print(np.mean((orth**2).sum(axis=-1)))
        #targets[i,:] = orth[sigs[i]]
        #The normalization doesn't really matter, but...
        #targets[i,:] = targets[i,:]#/np.sqrt((targets[i]**2).sum())
        #U = np.array([targets[j] for j in range(i+1)])
        
    return targets, sigs
