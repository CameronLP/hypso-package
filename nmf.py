"""
This code was written by Daniela Lupu and Joseph Garrett as part of the ELO-Hyp project, and is not yet published, so please don't share it
Eventually it will be available in more official channels

JLGarrett 2023.12.01

"""

import numpy as np
import scipy.optimize as opt
import osp as osp
import copy

EPS = 1e-4

def nmf(Img, r, n=-1,m=-1,mb_in=-1, mb_l=-1,
        gamma_m=0.1, gamma_s=0.3, delta=1.0, alpha = 0.1, beta=0.05, theta=0.3,
        version=2, tol = 0.1, verbose=False, maxiter=10000,
        remoditer=10000009, mask=np.array([], dtype='bool'),
        no_S_smoothness=True, no_M_smoothness=True,
        zeta = 0.5, initial_M = np.array([]), up_pix=1,
       weighted = True, weights = (), apow=5):
    """ Nonnegative matrix factorization 1/2 \| A - MS\|^2 + alpha sum g(M) + beta sum g(S)
    n,m = spatial dimension
    """
    b, pixels = Img.shape

    
    #initialization
    #tol = 0.0001
    crit =1

    #gamma_s = 0.3 # rule: gamma_m < gamma_s
    #delta = 0.9
    #alpha = 0.1 # regularizer for smoothness
    #beta = 0.05
    #theta = 0.3 # control level of sparseness; usually less 0.3
    if version==2:
        s_lim = 1 - theta * (np.sqrt(b)-1)/np.sqrt(b)
    else:
        s_lim = -1
    #version = 1  # 1 - control sparseness via C; 0 - without sparseness control, 2 - control sparseness via projection

    if not no_S_smoothness:
        # minibatching only works without smoothness constraint for now
        mb_l = -1
        mb_in = -1

    # for now, set minibatching input equal to minibatching length
    #mb_l = mb_in

    if weighted==True:
        if len(weights)==0:
            weights = 1/np.sum(Img**2, axis=0)
    
    if len(mask) > 0:
        if mask.dtype=='bool':
            pmask = mask
            nmask = np.ones(np.sum(pmask), dtype=np.float32)
        else:
            pmask = mask > 0
            nmask = mask
    else:
        pmask = np.ones(Img.shape[1:], dtype='bool')
        nmask = pmask
    _, in_pixels = Img[:, pmask].shape
    pindex = np.arange(in_pixels, dtype=np.uint32)
    
    if mb_l > 0:
        np.random.shuffle(pindex)
        #print(pindex[:12])
        sel_pix = get_pix(pindex, mb_l)
        ##print(sel_pix)
        #print(pindex[:12])
        alength = mb_l
        A=Img[:,pmask][:,sel_pix]
        #print(Img)
        #print(Img[:, pmask])
        #print(A)

    else:
        A = Img
        alength = pixels
        sel_pix = np.arange(alength)
    Amax = np.max(A)
    Amin = np.min(A)
    print(Amax, Amin)
    if mb_l == -1:
        if mb_in >= 1:
            mb_in = alength
        elif np.abs(mb_in-0.5)< 0.5:
            mb_in = int(np.floor(mb_in*alength))
       

    #define the in/out pixel mask

    #indx = np.random.randint(0, alength, r)
    # Initialize M and S
    if len(initial_M) == 0:
        #if weighted:
        #    num = np.sum(nmask**apow*A, axis=-1)
        #    denom = np.sum(nmask**apow)
        #    mean = (num/denom)**(1/apow)
        #else:
        #    mean = np.mean(A, axis=-1)
        #indx = np.argsort(np.sum(np.abs(A.T-mean), axis=-1).T)[:r]
        _, indx = osp.atgp(np.multiply(nmask[sel_pix], A).transpose(), n=r+1)
        indx = indx[1:]
        print('idx', indx)
        if len(mask) == 0:
            #_, indx = osp.atgp(A.transpose(), n=r)
            #indx = np.random.randint(0,alength, r)
            M = A[:, indx]
        else:
            #_, indx = osp.atgp((A).transpose(), n=r)
            #args = np.argsort(np.abs(1-mask[sel_pix]))
            #if args[0] != args[1]:
            #else:
            #    indx = np.random.randint(0, alength, r)
            #    _, indx = osp.atgp(mask[sel_pix]*A.transpose(), n=r)
            M = A[:, indx]/mask[sel_pix][indx]
    else:
        M = np.copy(initial_M)
        
    if version==5:
        M /= np.sqrt(np.sum(M**2, axis=0))

    #_, indx = osp.atgp(A.transpose(), n=r)
    print("M-initializes")
    #print(M)

    if (version==3):
        if len(mask)>0:
            S = nmf_project(A, M, no_negative_residuals=True, mask=mask[pmask][sel_pix], delta=delta)
        else:
            S = nmf_project(A, M, no_negative_residuals=True, delta=delta)
    else:
        S = nmf_project(A, M, s_lim=s_lim, delta=delta)
    #np.array([opt.nnls(M, i)[0] for i in A.transpose()]).transpose()

    #S =  np.random.rand(r, pixels)
    #S = S/np.linalg.norm(S, axis = 0)  # S column norm 1
    
    # Augment A to compute S
    try:
        Aa = np.append(A, delta * nmask[sel_pix].reshape(1,-1), axis = 0)
    except NameError:
        Aa = np.append(A, delta * np.ones((1, alength)), axis=0)


    
    def smoothness(A, gamma):
        p = A.shape[1]
        H = np.empty(A.shape)
        Gp = np.empty(A.shape)
        for i in range(1, p-1):
            H[:,i]= 2/gamma *(np.exp( - np.power((A[:,i] - A[:, i+1]),2)/gamma) + np.exp( -np.power((A[:,i] - A[:, i-1]),2)/gamma)) 
            Gp[:,i]= 2/gamma *( (A[:,i] - A[:, i+1])* np.exp( -np.power((A[:,i] - A[:, i+1]),2)/gamma) + (A[:,i] - A[:, i-1])*np.exp( -np.power((A[:,i] - A[:, i-1]),2)/gamma)) 

        H[:,0]= 2/gamma *np.exp( -np.power((A[:,0] - A[:, 1]),2)/gamma) 
        H[:, p-1]= 2/gamma *np.exp( -np.power((A[:,p-1] - A[:, p-2]),2)/gamma) 
        Gp[:,0]= 2/gamma * (A[:,0] - A[:, 1])*np.exp( -np.power((A[:,0] - A[:, 1]),2)/gamma)
        Gp[:,p-1]= 2/gamma *(A[:,p-1] - A[:, p-2])*np.exp( -np.power((A[:,p-1] - A[:, p-2]),2)/gamma)

        return H, Gp
    
    def smooth(M, gamma_m):
        gm= 0
        b, r = M.shape
        for i in range(1,b-1):
            for j in range(1,r-1):
                gm = gm +4 - np.exp(- (M[i,j] - M[i-1, j])**2/ gamma_m) - np.exp(- (M[i,j] - M[i+1, j])**2/ gamma_m)- np.exp(- (M[i,j] - M[i, j-1])**2/ gamma_m)- np.exp(- (M[i,j] - M[i, j+1])**2/ gamma_m)   
        gm = gm +2 - np.exp(- (M[0,0] - M[1, 0])**2/ gamma_m) - np.exp(- (M[0,0] - M[0, 1])**2/ gamma_m)
        gm = gm +2 - np.exp(- (M[0,r-1] - M[0, r-2])**2/ gamma_m) - np.exp(- (M[0,r-1] - M[1, r-1])**2/ gamma_m)
        gm = gm +2 - np.exp(- (M[b-1,0] - M[b-2, 0])**2/ gamma_m) - np.exp(- (M[b-1,0] - M[b-1, 1])**2/ gamma_m)
        gm = gm +2 - np.exp(- (M[r-1,r-1] - M[r-1, r-2])**2/ gamma_m) - np.exp(- (M[r-1,r-1] - M[r-2, r-1])**2/ gamma_m)
        for j in range(1,r-1):
            gm = gm +3 - np.exp(- (M[0,j] - M[0, j-1])**2/ gamma_m) - np.exp(- (M[0,j] - M[0, j+1])**2/ gamma_m)- np.exp(- (M[0,j] - M[1, j])**2/ gamma_m)   
            gm = gm +3 - np.exp(- (M[b-1,j] - M[b-1, j-1])**2/ gamma_m) - np.exp(- (M[b-1,j] - M[b-1, j+1])**2/ gamma_m)- np.exp(- (M[b-1,j] - M[b-2, j])**2/ gamma_m)   
        for i in range(1, b-1):
            gm = gm + 3 - np.exp(- (M[i,0] - M[i+1, 0])**2/ gamma_m)- np.exp(- (M[i,0] - M[i-1, 0])**2/ gamma_m) - np.exp(- (M[i,0] - M[i, 1])**2/ gamma_m)
            gm = gm + 3 - np.exp(- (M[i,r-1] - M[i+1, r-1])**2/ gamma_m)- np.exp(- (M[i,r-1] - M[i-1, r-1])**2/ gamma_m) - np.exp(- (M[i,r-1] - M[i, r-2])**2/ gamma_m)
        return gm

    # Objective function computation
    if no_M_smoothness:
        gm = 0
    else:
        gm =smooth(M, gamma_m)
    gs=0
    if not no_S_smoothness:
        for i in range(r):
            gs = gs + smooth(S[i,:].reshape(n,m) , gamma_s)
        
    objF_old =  alpha* gm + beta* gs
    if version == 1:
        C = (1 - theta) * np.eye(r) + theta / r * np.ones((r, r))
        objF_old = objF_old + np.linalg.norm(A - M @ C @ S, "fro")
    else:
        objF_old = objF_old + np.linalg.norm(A - M @ S, "fro")
        C = 0

    
    if verbose:
        print('original:', objF_old)
    if verbose:
        obj_terms = {}
        obj_terms['gs'] = []
        obj_terms['gm'] = []
        obj_terms['err'] = []
        obj_terms['obj'] = []
    iterate = 0
    if len(mask) > 0:
        local_mask = mask[pmask][sel_pix]
        lmask = local_mask > 0
    
    S[S<EPS] = EPS
    criteria = True
    while (criteria)&(iterate < maxiter):
        #print(np.linalg.norm(A - M @ S, "fro"))
        if ((iterate + 1)%remoditer)==0:
            S = nmf_project(A, M, s_lim=s_lim, delta=delta)
        elif (mb_in > 0)&(iterate > 0):
            new_pix = get_pix(pindex, mb_in)
            sel_pix[:-mb_in] = sel_pix[mb_in:]
            sel_pix[-mb_in:] = new_pix[:]
            if len(mask) > 0:
                local_mask = mask[pmask][sel_pix]
                lmask = local_mask > 0
            A = Img[:,pmask][:, sel_pix]
            S[:, :-mb_in] = copy.deepcopy(S[:,mb_in:])
            if version == 3:
                if len(mask)==0:
                    S[:, -mb_in:] = nmf_project(A[:, -mb_in:], M, no_negative_residuals=True,
                                                delta=delta)
                else:
                    S[:, -mb_in:] = nmf_project(A[:,-mb_in:], M, no_negative_residuals=True,
                                                mask=local_mask[-mb_in:], delta=delta)
            else:
                S[:, -mb_in:] = nmf_project(A[:, -mb_in:], M, s_lim=s_lim, delta=delta)

            try:
                Aa = np.append(A, delta * nmask[sel_pix].reshape((1,-1)), axis = 0)
            except NameError:
                Aa = np.append(A, delta * np.ones((1, alength)), axis=0)
        #Aa[Aa <= 0] = EPS

        
        # Update M
        #print(np.linalg.norm(A - M @ S, "fro"))
        if no_M_smoothness:
            Hm, Gm = 0, 0
        else:
            Hm, Gm = smoothness(M, gamma_m)
        if len(mask) > 0 and mask.dtype == 'bool':
            num =  A[:, pmask[sel_pix]]#S[:, pmask[sel_pix]].T #A[:, pmask[sel_pix]] @ #+ alpha * (M * Hm - Gm)
            denom =  S[:, pmask[sel_pix]] #M @#@ S[:, pmask[sel_pix]].T #+ alpha * M * Hm)
        else:
            num = A 
            denom = M @ S
            
        #print("denom1", denom)
        if weighted:
            num = np.multiply(weights[sel_pix], num)
            denom = np.multiply(weights[sel_pix], denom)
        if len(mask) > 0 and mask.dtype == 'bool':
            num = num @ S[:, pmask[sel_pix]].T
            denom = denom @ S[:, pmask[sel_pix]].T #+ alpha * M * Hm)
        else:
            num = num @ S.T
            denom = denom @ S.T
        #print("denom2", denom)
        if alpha > 0:
            num += alpha * (M * Hm - Gm)
            denom +=  alpha * M * Hm
        #print(denom)
        #print("1m",M)
        M = M * num / denom
        if version==5:
                # normalize spectra
                M /= np.sqrt(np.sum(M**2, axis=0))
        #print("2m",M)
        #else:
        #    M = M * (A @ S.T + alpha * (M * Hm - Gm)) / (M @ S @ S.T + alpha * M * Hm + EPS)
        #M = M *(A@S.T +alpha * (M* Hm -Gm))/ (M@S@S.T + alpha * M * Hm)
        #M[M>(15*Amax)] = 15*Amax
        #M[M<0] *= -1
        M[M==0] = EPS
        #M[np.isnan(M)] = Amin

        #Augment M
        Ma = np.append(M, delta* np.ones((1,r)), axis =0)
        
        # Update S
        #print(np.linalg.norm(A - M @ S, "fro"), 's')
        if no_S_smoothness:
            Hs, Gs = 0, 0
        else:
            Hs, Gs = smoothness(S, gamma_s)

        #print(np.linalg.norm(A - M @ S, "fro"), 's2')
        if version == 1:
            S = S * (C.T @ Ma.T @ Aa + beta * (S * Hs - Gs)) / (C.T @ Ma.T @ Ma @ C @ S + beta * S * Hs)
        else:
            #print(S.mean())
            #print((Ma.T @ Ma @ S + beta * S * Hs).mean())
            #print(((Ma.T @ Aa + beta * (S * Hs - Gs)).mean()))
            num = Aa
            denom = Ma @ S
            if weighted:
                num = np.multiply(weights[sel_pix], num)
                denom = np.multiply(weights[sel_pix], denom)
            num = Ma.T @ num + beta * (S * Hs - Gs)
            denom = Ma.T @ denom + beta * S * Hs
            S_new = S * num / denom
            #S_new = S * (Ma.T @ Aa + beta * (S * Hs - Gs)) / (Ma.T @ Ma @ S + beta * S * Hs)
            S[~np.isnan(S_new)] = S_new[~np.isnan(S_new)]
            #print(S.mean())

        #for i, band in enumerate(S):
        #    if band.max() < 0.7:
        #        S[i] *= 0.7/S[i].max()
        if version>1:
            # normalize everything
            #print(np.linalg.norm(A - M @ S, "fro"), 'norm')
            #
            #if len(mask) > 0:
            #    S = normalize(S, mask=local_mask)
                #print(local_mask)
                #print(local_mask>0)
            #else:
            #    Ssum = S.sum(axis=0)
            #    S /= Ssum

            # calculate the original labels (weighted by total vector size) in case something vanishes
            original_labels = np.argmax(np.multiply(S.T, 1 / S.sum(axis=1)).T, axis=0)
            #print(np.linalg.norm(A - M @ S, "fro"), 'midnorm')
            if version==2:
                # do the L1 projection on all of the layers of S
                for matr in S:
                    L1_proj(matr, s_lim=s_lim, zeta=zeta)
            elif version==3:
                # introduce sparsity by the 'no negative residuals' principle
                for i in range(len(S)):
                    idx = [j for j in range(len(S))]
                    idx.remove(i)
                    min_resid = (A - M[:,idx] @ S[idx, :]).min(axis=0)
                    S[i, min_resid < 0] *= zeta
            if version==4:
                # introduce sparsity through a minimum group magnitude
                for i in range(len(S)):
                    if len(mask) > 0:
                        max_pix = np.argsort(np.abs(local_mask-S[i]))[:up_pix]
                        old_diff = local_mask[max_pix]-S[i, max_pix]
                        new_diff = (zeta)*old_diff
                        S[i, max_pix] = local_mask[max_pix]- new_diff
                    else:
                        max_pix = np.argsort(np.abs(1-S[i]))[:up_pix]
                        #print(S[i, max_pix])
                        old_diff = 1-S[i, max_pix]
                        new_diff = (zeta)*old_diff
                        S[i, max_pix] = 1- new_diff
                    #idx = [j for j in range(len(S))]
                    #idx.remove(i)
                    #for j in idx:
                    #    S[j, max_pix] *= zeta
            


            # fix any zeros that appeared
            #Ssum = S.sum(axis=0)
            #zeros = (Ssum == 0)
            #S[original_labels[zeros], zeros] = 1
            #if len(mask)>0:
            #    S[:,mask[sel_pix]==0] = 0
            S[S<EPS] = EPS
            # normalize again
            #if len(mask) > 0:
            #    S = normalize(S, mask=local_mask)
            #else:
            #    Ssum = S.sum(axis=0)
            #    S /= Ssum
            #print(np.linalg.norm(A - M @ S, "fro"), 'afternorm')

        #Compute objective function
        if no_M_smoothness:
            gm=0
        else:
            gm = smooth(M, gamma_m)
        gs = 0
        if not no_S_smoothness:
            for i in range(r):
                gs = gs + smooth(S[i, :].reshape(n, m), gamma_s)
        
        objF = alpha* gm + beta* gs
        if weighted:
            if version == 1:
                err = np.linalg.norm(np.multiply(weights[sel_pix],A - M @ C @ S), "fro")
            else:
                err = np.linalg.norm(np.multiply(weights[sel_pix], A - M @ S), "fro")
        else:
            if version == 1:
                err = np.linalg.norm(A - M @ C @ S, "fro")
            else:
                err = np.linalg.norm(A - M @ S, "fro")



        objF += err

        crita = objF_old -objF
        old_crit = crit
        crit = crita / objF
        
        if verbose:
            print(iterate, objF, crit, crita)
            obj_terms['gs'].append(gs)
            obj_terms['gm'].append(gm)
            obj_terms['err'].append(err)
            obj_terms['obj'].append(objF)
        objF_old = objF
        
        criteria = np.abs(crit-tol/2) > tol/2
        


        iterate += 1


    if verbose:
        return M, C, S, obj_terms

    return M, C, S

def normalize(S, mask=()):
    if len(mask)==0:
        S = S / S.sum(axis=0)
    else:
        S = S / S.sum(axis=0) * mask
    return S

def nmf_project(A, M, maxiter=100, delta=10, tol=0.9, s_lim = -1, verbose=False, no_negative_residuals=False,
                mask=(), zeta=0.5):
    b, pixels = A.shape
    r = M.shape[-1]
    if len(mask)>1:
        Aa = np.append(A, delta * mask.reshape(1,-1), axis = 0)
    else:
        Aa = np.append(A, delta * np.ones((1, A.shape[1])), axis=0)

    Ma = np.append(M, delta* np.ones((1,r)), axis =0)
    if verbose:
        print("starting initial fitting")
    S = np.array([opt.nnls(Ma, i, maxiter=1000)[0] for i in Aa.transpose()], dtype=np.float32).transpose()
    #S = normalize(S, mask)

    if no_negative_residuals:
        original_labels = np.argmax(np.multiply(S.T, 1 / S.sum(axis=1)).T, axis=0)

        # do the L1 projection on all of the layers of S
        for i in range(len(S)):
            idx = [j for j in range(len(S))]
            idx.remove(i)
            min_resid = (A - M[:, idx] @ S[idx, :]).min(axis=0)
            S[i, min_resid < 0] *= zeta
            #print((min_resid[S[i]>0] < 0).sum(), ' negative residuals')

        # fix any zeros that appeared
        Ssum = S.sum(axis=0)
        zeros = (Ssum == 0)
        S[original_labels[zeros], zeros] = 1
        S[:,mask==0] = 0
        # normalize again
        #Ssum = S.sum(axis=0)
        #S /= Ssum

    elif s_lim > 0:
        original_labels = np.argmax(np.multiply(S.T, 1 / S.sum(axis=1)).T, axis=0)

        # do the L1 projection on all of the layers of S
        for matr in S:
            L1_proj(matr, s_lim=s_lim, zeta=zeta)

        # fix any zeros that appeared
        Ssum = S.sum(axis=0)
        zeros = (Ssum == 0)
        S[original_labels[zeros], zeros] = 1

        # normalize again
        #S = normalize(S, mask)

    objF_old = np.linalg.norm(A - M @ S, "fro")
    crit = tol + 1
    iterate = 0
    if verbose:
        print("starting loop")
    while (np.abs(crit) > tol) & (iterate < maxiter):
        # Augment M
        Ma = np.append(M, delta * np.ones((1, r)), axis=0)

        # Update S
        S = S * (Ma.T @ Aa) / (Ma.T @ Ma @ S + EPS)
        #S = S / S.sum(axis=0)#np.linalg.norm(S, axis=0)  #
        if no_negative_residuals:
            original_labels = np.argmax(np.multiply(S.T, 1 / S.sum(axis=1)).T, axis=0)

            # do the L1 projection on all of the layers of S
            for i in range(len(S)):
                idx = [j for j in range(len(S))]
                idx.remove(i)
                min_resid = (A - M[:, idx] @ S[idx, :]).min(axis=0)
                S[i, min_resid < 0] *= zeta

            # fix any zeros that appeared
            Ssum = S.sum(axis=0)
            zeros = (Ssum == 0)
            S[original_labels[zeros], zeros] = 1
            
            # fix problems with zero labelling
            S[:,mask==0] = 0

            # normalize again
            #S = normalize(S, mask)
        elif s_lim>0:
            # calculate the original labels (weighted by total vector size) in case something vanishes
            original_labels = np.argmax(np.multiply(S.T, 1 / S.sum(axis=1)).T, axis=0)

            # do the L1 projection on all of the layers of S
            for matr in S:
                L1_proj(matr, s_lim=s_lim, zeta=zeta)

            # fix any zeros that appeared
            Ssum = S.sum(axis=0)
            zeros = (Ssum == 0)
            S[original_labels[zeros], zeros] = 1
            S[:,mask==0] = 0
            # don't normalize again


        objF = np.linalg.norm(A - M @ S, "fro")


        crit = objF_old - objF

        if verbose:
            print(objF, crit)
        objF_old = objF

        iterate += 1

    return S

def L1(mat):
    norm = np.sum(np.abs(mat))/(len(mat))
    return norm

def L2(mat):
    norm = np.sqrt(np.sum(mat**2)/len(mat))
    return norm

def sparseness(mat):
    return 1 - L1(mat)/L2(mat)
    
def L1_proj(mat, s_lim=0.5, zeta=0.0):
    '''
    sparse partial projection
    designed to be fast
    '''
    msort = np.argsort(mat)
    l1 = L1(mat)
    l2 = L2(mat)
    if l1/l2 < s_lim:
        return mat
    else:
        projlim = s_lim*l2*len(mat)
        csum = np.cumsum(mat[msort][::-1])
        to_remove = msort[~(csum<projlim)[::-1]]
        mat[to_remove] *= zeta
        return mat

class NMF():
    '''
    To use with smoothness, set no_S_smoothness to false and input
    '''
    def __init__(self, n_bands=12,
                 gamma_m=0.1, gamma_s=0.3, predelta=0.5,
                 alpha=0.1, beta=0.05, theta=0.3, version=2, tol=0.1,
                 verbose=False, maxiter=10000, remoditer=20000, mask=(),
                 no_S_smoothness=True, no_M_smoothness=True,
                 mb_in=-1, mb_l=-1, zeta=0.5, weighted=False):
        self.r = n_bands
        self.gamma_m = gamma_m
        self.gamma_s = gamma_s
        self.predelta = predelta
        self.alpha = alpha
        self.beta = beta
        self.version = version
        self.tol = tol
        self.verbose = verbose
        self.maxiter = maxiter
        self.remoditer = remoditer
        self.mask = mask
        self.no_S_smoothness = no_S_smoothness
        self.no_M_smoothness = no_M_smoothness
        self.theta = theta
        self.mb_in = mb_in
        self.mb_l = mb_l
        self.zeta=zeta
        self.weighted=weighted

    def _fit(self, img, n, m):
        if not self.no_S_smoothness:
            self.n = img.shape[0]
            self.m = img.shape[1]

    def fit(self, img):
        if len(img.shape)==2:
            self.delta = self.predelta*img.mean(axis=0).sum()
            out = nmf(img.transpose(), r=self.r, gamma_m=self.gamma_m,
                      gamma_s=self.gamma_s, delta=self.delta, alpha=self.alpha,
                      beta=self.beta, theta=self.theta, version=self.version,
                      tol=self.tol, verbose=self.verbose, maxiter=self.maxiter,
                      remoditer=self.remoditer, mask=self.mask,
                      no_S_smoothness=self.no_S_smoothness,
                      no_M_smoothness=self.no_M_smoothness,
                      mb_in=self.mb_in, mb_l=self.mb_l, zeta=self.zeta,
                     weighted=self.weighted)
        elif len(img.shape)==3:
            sh = img.shape
            self._fit(img, sh[0], sh[1])
            self.delta = self.predelta*img.mean(axis=(0,1)).sum()
            out = nmf(img.reshape((-1, sh[-1])).transpose(), n=self.n, m=self.m,
                        r = self.r, gamma_m = self.gamma_m,
                        gamma_s = self.gamma_s, delta = self.delta, alpha = self.alpha,
                        beta = self.beta, theta = self.theta, version = self.version,
                        tol = self.tol, verbose = self.verbose, maxiter = self.maxiter,
                        remoditer = self.remoditer, mask = self.mask(),
                        no_S_smoothness = self.no_S_smoothness,
                        no_M_smoothness = self.no_M_smoothness, zeta=self.zeta,
                     weighted=self.weighted)
        self.M = out[0]
        return out

    def transform(self, img):
        try:
            proj = nmf_project(img.transpose(), self.M, delta=self.delta, tol=self.tol, zeta=self.zeta)
        except AttributeError:
            self.delta = self.predelta * img.mean(axis=0).sum()
            proj = nmf_project(img.transpose(), self.M, delta=self.delta, tol=self.tol, zeta=self.zeta)
        return proj

    def inversetransform(self, rep):
        return (self.M@rep).transpose()

    def make_saving_dict(self):
        saving_dict = {}
        saving_dict['M'] = self.M
        return saving_dict

    def load_from_props(self, M):
        self.M = M

    def save(self, filename):
        saving_dict = self.make_saving_dict()
        np.savez(filename, **saving_dict)
    def trained_weights(self):
        return self.make_saving_dict()

    def load(self, filename):
        loaded = np.load(filename)
        self.M = loaded['M']

    def endecode(self, img):
        self.delta = self.predelta*img.mean(axis=(0,1)).sum()
        return self.inversetransform(self.transform(img))

def get_pix(pix_array, n_to_select):
    '''
    note that this does adjust the pixel array
    '''
    sel_pix = copy.deepcopy(pix_array[:n_to_select])
    pix_array[:-n_to_select] = pix_array[n_to_select:]
    pix_array[-n_to_select:] = copy.deepcopy(sel_pix[:])
    return sel_pix
