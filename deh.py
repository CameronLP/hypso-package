"""
This code was written as part of the ELO-Hyp project, and is not yet published, so please don't share it
Eventually it will be available in more official channels

JLGarrett 2023.12.01

"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import scipy.optimize as opt
import sklearn.svm as svm
import tables as tab
import nmf as nmf
import time  

class Node():
    def __init__(self, spatial_map, classifier, status=True):
        self.map = spatial_map
        self.classifier = classifier
        #self.splitter = None
        #self.status=status # True = open, False = closed

def svm_to_endmembers(coef, intercept, data):
    abs_coef = np.sqrt(np.sum(coef**2))
    coef_norm = coef/abs_coef
    # adjust data to account for the intercept offset 
    shifted_data = (data.T + (intercept/abs_coef*coef_norm).T).T
    # project data onto the SVM plane
    proj_coefficients = shifted_data@coef_norm.T
    # remove the part of the data that is out-of-plane
    planar_data = data - proj_coefficients*coef_norm
    # average the planar data to find the mean
    endmember_mean = np.mean(planar_data, axis=0)
    # get the two endmembers
    endmember0 = endmember_mean - coef_norm/(2*abs_coef)
    endmember1 = endmember_mean + coef_norm/(2*abs_coef)
    #if np.isnan(endmember0).any():
    #    print(abs_coef, coef_norm, endmember_mean, proj_coefficients, shifted_data)
    return endmember0, endmember1


class DEH():
    def __init__(self, no_negative_residuals = True, subfit_params={}):
        #fdig, fparta, fpartb,
        '''
        fdig must be able to find endmembers of unlabelled data
        fpart must be able to both (a) train a set of weights for a given distribution 
        and (b) find the distribution for a given set of weights
        fpart can thus be a function on either labelled or unlabelled data
        '''
        self.nodes = {}
        self.NR = not no_negative_residuals # negative residuals
        self.subfit_parameters = subfit_params
        self.remainder_id = 'none'
        self.splitting_size = 100000
        self.max_depth = 10
        self.root='average3'
        self.threshold = 0.1
        self.min_size = 1000
        self.tol = 0.001
        self.tolb = 0.05
        self.mb_l = -1 
        self.mb_in = -1
        self.svm='threshold'
        self.mixed_layers = True
        self.double_fit=True
        self.approach_rate=0.1
        self.spread = 2
        self.plot_size = (239,171)
        self.plot_aspect = 0.3
        self.zeta=0.1
        self.up = 0
        self.history = []
        self.S_MAX=2
        self.weight_power=2
        self.verbose = False
        self.Cnorm = 1
        self.max_iter = 200
        #self.sf = 253
        #return self
        
    def parameter_initialization(self, image):
        # takes the image in the ordinary (e.g. N x b), rather than nmf-style 
        self.delta = image.sum(axis=1).mean()
        self.weights = 1/np.sum(np.abs(image)**self.weight_power, axis=1)
        #self.nodes[''] = Node(spatial_map = np.ones(image.shape[0], dtype=bool),
        #                      classifier = image.mean(axis=0),
        #                      status=False)
    
    def print_nmax(self, level):
        return 0
        n_nodes = 0
        l_nodes = []
        maxes = []
        for node in self.nodes:
            if len(node)==level:
                n_nodes += 1
                l_nodes.append(node)
        S = np.zeros((n_nodes,) + self.nodes[l_nodes[0]].map.shape)
        for i, node in enumerate(l_nodes):
            S[i,:] = self.nodes[node].map[:]
            maxes.append(self.nodes[node].map.max())
        sums = np.sum(S, axis = 0)
        if self.verbose:
            print(n_nodes, 'nmax ',S.max(), maxes)

        
        
    def update_lower_level_spatial(self, level, data):
        L_nodes = [i for i in self.nodes if len(i)==level]
        endnodes = self.get_end_nodes()
        for node in L_nodes:
            if len(node)==level:
                if node + '1' in self.nodes:
                    nmap = self.nodes[node].map
                    omap = (self.nodes[node+'0'].map + self.nodes[node+'1'].map)
                    og0 = omap > 0
                    
                    S = np.array([self.nodes[node+'0'].map,
                                   self.nodes[node+'1'].map])
                    S[:,og0] *= nmap[og0]/omap[og0]
                    
                    n_endnodes = 0
                    LN = len(node)
                    for enode in endnodes:
                        if enode[:LN]==node:
                            n_endnodes += 1
                            
                    for enode in endnodes:
                        if enode[:LN]==node:
                            smap = np.copy(self.nodes[enode].map)
                            smap[og0] *= nmap[og0]/omap[og0]
                            ii = int(enode[LN])
                            S = self.setS_to_threshold(S, ii, np.minimum(smap, self.nodes[enode].map))
                  
                    self.nodes[node+'0'].map = S[0]
                    self.nodes[node+'1'].map = S[1]
    
    def bipropagate(self, data, vary_spectra=True):
        depth = self.get_depth()
        for level in range(depth):
            self.spectral_from_spatial(level, data)
            self.spatial_from_spectral(level, data)
            if level < (depth -1):
                self.update_lower_level_spatial(level+1, data)
            if vary_spectra:
                self.update_spectra(data, level, utype=self.root)

    
    def train_clap_cycle(self, image):
        orig_S = self.end_node_map()
        self.full_fit(image)
        self.update_intermediate_node_maps()
        self.fix_endnode_maps()
        #self.trim_empty_nodes()
        self.update_intermediate_node_maps()
        self.display_level(self.get_depth())
        self.bipropagate(image)
        #self.propagate_spatial_upward(image)
        #self.propagate_spectral_downward(image)
        self.propagate_one_child_nodes()
        #self.train_open_nodes(image)
        new_S = self.end_node_map()
        delta = (np.abs(orig_S - new_S)).sum(axis=0)
        self.history[-1].append(delta.sum()/orig_S.shape[1])
        if self.verbose:
            print("delta: ", (delta.sum()/orig_S.shape[1]))
        self.trim_empty_nodes()
        return delta.sum()/orig_S.shape[1]
    
    def one_loop_clap(self, image):
        self.training = 'one_loop'
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        self.display_level(1)
        self.train_clap_cycle(image)
        self.display_level(1)
        while (len(self.nodes_to_split())>0) and (self.get_depth()<self.max_depth):
            #self.full_fit(image)
            self.update_spectra(image, self.get_depth(), utype=self.root)
            self.add_another_node_layer(image)
            self.train_clap_cycle(image)
            self.display_level(self.get_depth())
            
            
    def train(self, image, loop='closed', cycle='clap'):
        self.start = time.time()
        if loop=='open':
            self.feed_forward_train(image)
        elif loop=='one_loop':
            if cycle=='slap':
                self.one_loop_train(image)
            elif cycle=='clap':
                self.one_loop_clap(image)
        elif loop =='closed':
            if cycle=='slap':
                self.train_slap(image)
            elif cycle=='clap':
                self.train_clap(image)
    
    def train_clap(self, image):
        self.training = 'closed loop'
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        self.train_clap_cycle(image)
        delta = 1
        while (len(self.nodes_to_split())>0) and (self.get_depth()<self.max_depth):
            self.update_spectra(image, self.get_depth(), utype=self.root)
            self.add_another_node_layer(image)
            delta = 1
            while delta > self.tolb:
                delta=self.train_clap_cycle(image)
                self.display_level(self.get_depth())
    
    def propagate_spatial_upward(self, data):
        depth = self.get_depth()
        for level in range(depth-1, -1, -1):
            #print(level)
            self.spectral_from_spatial(level, data)
            
    
    def propagate_spectral_downward(self, data, vary_spectra=True):
        depth = self.get_depth()
        for level in range(depth):
            if self.verbose:
                print("level ", level)
            self.spatial_from_spectral(level, data)
            if vary_spectra:
                self.update_spectra(data, level, utype=self.root)
                     
            
    def update_all_spectra(self, data, utype='spatial'):
        depth = self.get_depth()
        for level in range(depth+1):
            self.update_spectra(data, level, utype)
          
        
    def ff_norm_S(self):
        end_nodes = self.get_end_nodes()
        S = np.zeros((len(end_nodes), len(self.nodes[''].map)), dtype=np.float32)
                     
        for i, x in enumerate(self.end_nodes):
            S[i] = self.nodes[x].map
        
        S /= S.sum(axis=0)
        
        for i, x in enumerate(self.end_nodes):
            self.nodes[x].map = S[i]
                     
        
        
    def feed_forward_train(self, image):
        self.training = 'open_loop'
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        if self.verbose:
            print(self.nodes_to_split())
        while (len(self.nodes_to_split()) > 0) & (self.get_depth() < self.max_depth):
            o_depth = self.get_depth()
            self.add_another_node_layer(image)
            self.ff_norm_S()
            for nodeID in self.nodes:
                if len(nodeID)==o_depth:
                    try:
                        self.nodes[nodeID].splitter = classifiers_2_svm(self.nodes[nodeID+'0'].classifier,
                                                            self.nodes[nodeID+'1'].classifier)
                    except KeyError:
                        pass
            self.update_spectra(image, self.get_depth(), utype=self.root)
            rem = self.remainder_at_level(image, self.get_depth())
            err = np.sum(self.weights*np.sum(rem**2, axis=-1))
            self.history.append([time.time()-self.start,
                                 self.get_depth(),
                                 err
                                ])
            if self.verbose:
                self.display_level(self.get_depth())
            
    def one_loop_train(self, image):
        self.training = 'one_loop'
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        self.display_level(1)
        self.train_1_cycle(image)
        self.display_level(1)
        while (len(self.nodes_to_split())>0) and (self.get_depth()<self.max_depth):
            #self.full_fit(image)
            self.update_spectra(image, self.get_depth())
            self.add_another_node_layer(image)
            self.train_1_cycle(image)
            self.display_level(self.get_depth())
            #self.full_fit(image)
            
    def forward_svm_train(self, image):
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        self.display_level(1)
        self.train_1_cycle(image)
        self.display_level(1)
        while (len(self.nodes_to_split())>0) and (self.get_depth()<self.max_depth):
            #self.full_fit(image)
            self.update_spectra(image, self.get_depth(), utype=self.root)
            self.add_another_node_layer(image)
            self.update_intermediate_node_maps()
            self.display_level(self.get_depth())
            self.propagate_spatial_upward(image)
            self.propagate_spectral_downward(image)
            self.propagate_one_child_nodes()
            self.display_level(self.get_depth())
        
            
    def train_slap(self, image):
        self.training = 'closed loop'
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        self.train_1_cycle(image)
        delta = 1
        while (len(self.nodes_to_split())>0) and (self.get_depth()<self.max_depth):
            self.update_spectra(image, self.get_depth(), utype=self.root)
            self.add_another_node_layer(image)
            delta = 1
            while delta > self.tolb:
                delta=self.train_1_cycle(image)
                self.display_level(self.get_depth())
            #self.full_fit(image)
            #print('full fit')
            #self.display_level(self.get_depth())
        #self.train_1_cycle(image)
    def train_cycle(self, image):
        delta = 1
        while delta > self.tolb:
            delta=self.train_1_cycle(image)
            if self.verbose():
                self.display_level(self.get_depth())
        
    def stablize_cycle(self, image):
        self.propagate_spectral_downward(image)
        self.propagate_one_child_nodes()
        S = []
        for i in self.end_nodes:
                S.append(self.nodes[i].map)
        S = np.array(S)
        S = self.set_to_threshold(S)
        for i, x in enumerate(self.end_nodes):
            self.nodes[x].map = S[i]
        orig_S = self.end_node_map()
        self.update_intermediate_node_maps()
        self.propagate_spatial_upward(image)
        self.propagate_spectral_downward(image)
        self.propagate_one_child_nodes()

    def predict(self, image):
        self.nodes[''].map=np.ones(image.shape[0])
        self.remainder_at_level(image, 0)
        self.propagate_spectral_downward(image, vary_spectra=False)
        self.propagate_one_child_nodes()
        S = []
        self.get_end_nodes()
        for i in self.end_nodes:
            S.append(self.nodes[i].map)
        S = np.array(S)
        return S    
        
    def base_predict(self, image):
        classifiers = self.get_end_classifiers()
        try:
            S = np.float32(nmf.nmf_project(image.T, classifiers.T, delta=self.delta, verbose=True,
                      tol=self.tol))
        except AttributeError:
            self.parameter_initialization(image)
            S = np.float32(nmf.nmf_project(image.T, classifiers.T, delta=self.delta, verbose=True,
                      tol=self.tol))
        self.get_end_nodes()
        for i, x in enumerate(self.end_nodes):
            self.nodes[x].map = S[i]
        self.update_intermediate_node_maps()
            
        return S
        
    def train_1_cycle(self, image):
        orig_S = self.end_node_map()
        self.full_fit(image)
        self.update_intermediate_node_maps()
        self.fix_endnode_maps()
        #self.trim_empty_nodes()
        self.update_intermediate_node_maps()
        self.display_level(self.get_depth())
        #self.bipropagate(image)
        self.propagate_spatial_upward(image)
        self.propagate_spectral_downward(image)
        self.propagate_one_child_nodes()
        #self.train_open_nodes(image)
        new_S = self.end_node_map()
        delta = (np.abs(orig_S - new_S)).sum(axis=0)
        if self.verbose:
            print("delta: ", (delta.sum()/orig_S.shape[1]))
        self.trim_empty_nodes()
        return delta.sum()/orig_S.shape[1]
        
        
    def train_single_open_node(self, image, node):
        rimage = self.get_reduced_dataset(image, node)
        # train on rimage
        
    def upstream_nodes(self, nodeID):
        #get all possible upstream nodes
        #l = len(nodeID)
        unodes = [i for i in self.nodes if nodeID[:len(i)]==i]
        unodes.remove(nodeID)
        return unodes
    
    def downstream_nodes(self, nodeID):
        #get all possible downstream nodes
        l = len(nodeID)
        dnodes = [i for i in self.nodes if i[:l]==nodeID]
        dnodes.remove(nodeID)
        return dnodes
    
    def get_end_nodes(self):
        d = self.get_depth()
        self.end_nodes = [i for i in self.nodes if len(i)==d]
        #for node in self.end_nodes:
        #    if node + '0' in self.nodes:
        #        self.nodes[node].status = False
        #self.end_nodes = [i for i in self.nodes if self.nodes[i].status]
        return self.end_nodes
    
    def end_node_map(self):
        end_nodes = self.get_end_nodes()
        S = np.zeros((len(end_nodes), self.nodes[''].map.shape[0]), dtype=np.float32)
        for i, x in enumerate(end_nodes):
            S[i] = self.nodes[x].map
        return S
    
    def get_end_classifiers(self):
        end_nodes = self.get_end_nodes()
        self.end_classifiers = np.array([self.nodes[i].classifier for i in self.end_nodes])
        return self.end_classifiers
    
    def get_rel_spec(self, image):
        rel_pix = []
        for i in self.end_nodes:
            self.nodes[i].map[rel_pix] *= 0.01
            new_pix = np.argmin(np.abs(1-self.nodes[i].map)) 
            self.nodes[i].map[rel_pix] *= 100
            rel_pix.append(new_pix)
        rel_spec = image[rel_pix]
        return rel_spec
    
    def update_classifiers(self, data): 
        for i in self.nodes:
            self.nodes[i].classifier = self.weighted_average(i, data, 3)
    
    def update_spectra(self, data, level, utype='spatial'):
        S = []
        rel_nodes = []
        if utype=='spatial':
            for i in self.nodes:
                if len(i)==level:
                    S.append(self.nodes[i].map)
                    rel_nodes.append(i)
            S = np.array(S)
            rel_spec = np.array([opt.nnls(np.multiply(np.sqrt(self.weights), S).T,
                     (np.multiply(np.sqrt(self.weights), data.T[i])))[0] for i in range(data.shape[1])]).T
            for i, x in enumerate(rel_nodes):
                self.nodes[x].classifier = rel_spec[i]
        elif 'average' in utype:
            p = int(utype[7:])
            for i in self.nodes:
                if len(i)==level:
                    self.nodes[i].map = np.minimum(self.S_MAX, self.nodes[i].map)
                    self.nodes[i].classifier = self.weighted_average(i, data, p)
    
    def full_fit(self, image):
        classifiers = self.get_end_classifiers()
        
        if self.root == 'pixel':
            rel_pix = []
            for i in self.end_nodes:
                old = np.copy(self.nodes[i].map[rel_pix])
                self.nodes[i].map[rel_pix] = 0
                new_pix = np.argmin(np.abs(1-self.nodes[i].map)) 
                self.nodes[i].map[rel_pix] = old
                rel_pix.append(new_pix)
            rel_spec = image[rel_pix]
        elif self.root == 'model':
            rel_spec = np.copy(classifiers)
        elif 'average' in self.root:
            p = int(self.root[7:])
            rel_spec = []
            self.update_all_spectra(image, utype=self.root)
            for i in self.end_nodes:
                rel_spec.append(self.weighted_average(i, image, p))
            rel_spec = np.array(rel_spec)
            if self.mixed_layers:
                d = self.get_depth()
                npa = np.array([self.nodes[''].classifier for x in self.end_nodes])
                rel_spec = np.zeros_like(npa, dtype=np.float32)
                rel_spec[:] += npa[:]
                div = 1
                for i, x in enumerate(self.end_nodes):
                    for j in range(1,len(x)+1):
                        if (np.sum(self.nodes[x[:j]].classifier[:]) > 0):
                            rel_spec[i,:] += self.nodes[x[:j]].classifier[:]*1.5**(j)
                            #div += (j+1)
                        else:
                            #divx = div + (j+1)
                            rel_spec[i,:] += rel_spec[i,:]*1.5**(j)
                            #div += (j+1)
                divisor = (1.5**np.arange(0,d+1)).sum()
                rel_spec /= divisor#(d+1)
                #rel_spec /= divisor
        elif self.root == 'spatial':
            S = []
            for i in self.end_nodes:
                S.append(self.nodes[i].map)
            S = np.array(S)
            S = self.set_to_threshold(S)
            if self.mixed_layers:
                self.update_intermediate_node_maps()
                self.update_all_spectra(image)
                d = self.get_depth()
                rel_spec = 0*np.array([self.nodes[x].classifier for x in self.end_nodes])
                for i, x in enumerate(self.end_nodes):
                    for j in range(1,len(x)+1):
                        if not np.isnan(self.nodes[x[:j]].classifier[:]).any():
                            rel_spec[i,:] += self.nodes[x[:j]].classifier[:]
                        else:
                            rel_spec[i,:] *= (j+1)/j
                divisor = np.arange(0,d+1).sum()
                rel_spec /= d
            else:
                rel_spec = np.array([opt.nnls(np.multiply(np.sqrt(self.weights), S).T,
                     (np.multiply(np.sqrt(self.weights), image).T)[i])[0] for i in range(image.shape[1])]).T
                for i, x in enumerate(self.end_nodes):
                    self.nodes[x].classifier = rel_spec[i]
        print(rel_spec[:,0])
        # this nnls is kept, because it allows transition from zero, which is necessary for the feedback mechanism to work
        S = np.float32(nmf.nmf_project(image.T, rel_spec.T, delta=self.delta, verbose=True,
                      tol=self.tol, zeta=self.zeta))
        #S = self.set_to_threshold(S)
        #S = self.set_to_threshold(S, do_set=False)
        if self.double_fit:
            if False:
                self.update_intermediate_node_maps()
                self.update_all_spectra(image)
                d = self.get_depth()
                rel_spec = 0*np.array([self.nodes[x].classifier for x in self.end_nodes])
                for i, x in enumerate(self.end_nodes):
                    for j in range(1,len(x)+1):
                        if not np.isnan(self.nodes[x[:j]].classifier[:]).any():
                            rel_spec[i,:] += self.nodes[x[:j]].classifier[:]
                        else:
                            rel_spec[i,:] *= (j+1)/j
                divisor = np.arange(0,d+1).sum()
                rel_spec /= d
                out = rel_spec.T
            else:
                S[S<0] = -S[S<0]
                S = self.set_to_threshold(S)
                spec = self.update_spectra_MUR(rel_spec, S, image)
                ##out = np.array([opt.nnls(np.multiply(np.sqrt(self.weights),S).T,
                ##         (np.multiply(np.sqrt(self.weights), image.T[i])))[0] for i in range(image.shape[1])])
        
        
        
        #print(out.shape)
            #spec = out.T
            #S = np.float32(nmf.nmf_project(image.T, spec.T, delta=self.delta, verbose=True, tol=self.tol))
            S = self.update_abundances_MUR(spec, S, image)
            #S = self.set_to_threshold(S)
        S[S<0] = -S[S<0]
        S /= S.sum(axis=0)
        print('smax', S.max(), S.sum(axis=0).shape, S.sum(axis=0).max(), S.sum(axis=1).shape, S.sum(axis=1).max())
        S /= S.sum(axis=0)
        am = np.argmax(S)
        n = am // S.sum(axis=0).shape[0]
        pid = am - S.sum(axis=0).shape[0]*n
        print('smax', S.max(), np.argmax(S), n , pid, S[:,  pid])
        
        self.history.append([time.time() - self.start,
                             self.get_depth(),
                             self.get_error(spec.T, S, image)])
        
        #S = self.set_to_threshold(S)
        #S /= S.sum(axis=0)
        #out = np.array([opt.nnls(S.T,
        #             (image.T)[i])[0] for i in range(image.shape[1])])
        #spec = out
        #S = np.float32(nmf.nmf_project(image.T, spec, delta=self.delta, verbose=True, 
        #                               no_negative_residuals=(not self.NR), tol=self.tol))
        for i, x in enumerate(self.end_nodes):
            self.nodes[x].map = S[i]
    
    def get_error(self, M, S, data):
        esum = 0 
        #print(M.shape, S.shape, data.shape)
        for i in range(len(data)):
            esum += self.weights[i] * np.sum((data[i] - M@S[:,i])**2)
        return esum
    
    def fix_endnode_maps(self):
        en = self.get_end_nodes()
        for node in en:
            if self.nodes[node].map.sum()==0:
                for i in range(1, self.get_depth()):
                    twin = (node[:-i] + str(1-int(node[-i])))
                    try:
                        tnz = self.nodes[twin].map > 0
                        new_value = tnz.min()
                        amin = np.argmin(self.nodes[twin].map - new_value)
                        self.nodes[node].map[amin] = new_value
                    except KeyError:
                        pass
                    
                
    def update_intermediate_node_maps(self):
        depth = self.get_depth()
        unupdated = list(self.nodes.keys())
        for i in self.end_nodes:
            unupdated.remove(i)
        for level in range(depth-1,-1,-1):
            for i in self.nodes:
                if len(i)==level:
                    try:
                        self.nodes[i].map = self.nodes[i+'0'].map + self.nodes[i+'1'].map
                    except KeyError:
                        self.nodes[i].map = self.nodes[i+'0'].map
        
    def get_depth(self):
        my = list(self.nodes.keys())
        my.sort(key=len)
        self.depth = len(my[-1])
        return self.depth
    
    def remainder_at_level(self, data, level):
        local_data = np.copy(data)   
        for k in self.nodes:
            if ((len(k)==level)):
                local_data -= np.outer(self.nodes[k].map,
                                       self.nodes[k].classifier)
        self.remainder = local_data
        self.remainder_id = str(level)
        return self.remainder
    
    def remainder_at_node(self, data, nodeID, residual_type='proportional'):
        if self.remainder_id!=str(len(nodeID)):
            self.remainder_at_level(data, np.max([len(nodeID), 0]))
        
        if residual_type=='proportional':
            residual = (self.nodes[nodeID].map * \
                            (self.remainder + self.nodes[nodeID].classifier).T).T
        elif residual_type=='full':
            residual = (self.remainder + \
                         np.outer(self.nodes[nodeID].map,self.nodes[nodeID].classifier))
            
        non_negative_map = (residual < 0)
        #True should mean non-negative
        non_negative_map = ~(non_negative_map.max(axis=-1))
        
        return residual, non_negative_map
    
    def train_node_from_spatial(self, nodeID, data):
        try:
            for i in range(2):
                if np.sum(self.nodes[nodeID + str(i)].map)==0:
                    return -1
        except KeyError:
            pass
        try:
            try:
                #self.print_nmax(len(nodeID))
                coefs, intercept, dmap = self.partition_node(nodeID, data)
                
                if np.isnan(intercept):
                    return -1
                
                e0, e1 = svm_to_endmembers(coefs, 
                                           intercept,
                                           data[dmap])
            except ValueError:
                e0, e1 = self.nodes[nodeID].outmembers[0], self.nodes[nodeID].outmembers[1] 
            #print("intercept, ", intercept)
            
            
            r = self.approach_rate
            try:
                self.nodes[nodeID].outmembers = [r*e0.flatten() + (1-r)*self.nodes[nodeID].outmembers[0],
                                   r*e1.flatten() + (1-r)*self.nodes[nodeID].outmembers[1]]
                #self.nodes[nodeID].splitter = [self.approach_rate*coefs \
                #                               + (1-self.approach_rate)*self.nodes[nodeID].splitter[0],
                #                               self.approach_rate*intercept \
                #                               + (1-self.approach_rate)*self.nodes[nodeID].splitter[1]]
            except AttributeError:
                self.nodes[nodeID].outmembers = [e0.flatten(), e1.flatten()]
                
            self.nodes[nodeID].splitter = classifiers_2_svm(self.nodes[nodeID].outmembers[0],
                                                            self.nodes[nodeID].outmembers[1])
        except TypeError:
            print('typeerror')
            pass
    
    def newest_train_node_from_spatial(self, nodeID, data):
        try:
            print("trainging ", nodeID)
            nchild0 = nodeID + '0'
            nchild1 = nodeID + '1'
            Sfit = np.array([self.nodes[nchild0].map,
                             self.nodes[nchild1].map])
            local, nnr = self.remainder_at_node(data, nodeID)
            #print(local.shape)
            #coefs, intercept, _ = self.partition_node(nodeID, data)
            #e0, e1 = svm_to_endmembers(coefs, 
            #     intercept,
            #      data[self.nodes[nodeID].map >= (1-self.threshold)])
            if self.NR:
                out = np.array([opt.nnls(np.multiply(np.sqrt(self.weights),Sfit).T,
                            (np.multiply(np.sqrt(self.weights),local).T)[i])[0] for i in range(local.shape[1])])
                out = np.array(train_reconstructing_svm(local, self.nodes[nchild0].map,
                                               self.nodes[nchild1].map, b=5000, r=0.001,
                                         maxit=10, verbose=False, size=10000, m=0))
            else:
                nonzeros = self.nodes[nodeID].map>0
                nnr_merge = np.minimum(nnr, nonzeros, dtype=bool)
                child0 = np.argsort(self.nodes[nchild0].map)[-self.min_size:]
                child1 = np.argsort(self.nodes[nchild1].map)[-self.min_size:]
                joined_pts = np.concatenate([child0, child1])
                #out = np.array([opt.nnls((Sfit.T)[nnr_merge],
                #            (local[nnr_merge].T)[i])[0] for i in range(local.shape[1])])
                #coefs, intercept, dmap = self.partition_node(nodeID, data)
                #e0, e1 = svm_to_endmembers(coefs, 
                # intercept,
                # data[dmap])
                out = np.array(train_reconstructing_svm(local[nnr_merge], self.nodes[nchild0].map[nnr_merge],
                                               self.nodes[nchild1].map[nnr_merge], b0=1000000000, r=0.000000,
                                         maxit=20, verbose=False, size=-1, m=0))
                                         #initial_conditions = out.T))
                #
                #out = np.array([opt.nnls((Sfit.T)[nnr_merge],
                #            (local[nnr_merge].T)[i])[0] for i in range(local.shape[1])])
            r = self.approach_rate
            try:
                self.nodes[nodeID].outmembers = [r*out[0] + (1-r)*self.nodes[nodeID].outmembers[0],
                                   r*out[1] + (1-r)*self.nodes[nodeID].outmembers[1]]
                #self.nodes[nodeID].splitter = [self.approach_rate*coefs \
                #                               + (1-self.approach_rate)*self.nodes[nodeID].splitter[0],
                #                               self.approach_rate*intercept \
                #                               + (1-self.approach_rate)*self.nodes[nodeID].splitter[1]]
            except AttributeError:
                self.nodes[nodeID].outmembers = [out[0], out[1]]
            self.nodes[nodeID].splitter = classifiers_2_svm(*self.nodes[nodeID].outmembers)
            print(self.nodes[nodeID].splitter[1])
        except KeyError:
            print('keyerror')
            pass
        
        
    def train_node_from_spectral(self, nodeID, data):
        MAX_S = 2
        try:
            local, nnr = self.remainder_at_node(data, nodeID)
            coef, intercept = self.nodes[nodeID].splitter[0], self.nodes[nodeID].splitter[1]
            self.nodes[nodeID].map[self.nodes[nodeID].map==0] = -1
            class_map = classify_from_partition((local.T/self.nodes[nodeID].map).T, coef, intercept)
            self.nodes[nodeID].map[self.nodes[nodeID].map==-1] = 0
            scale_map = class_map * self.nodes[nodeID].map
            S = np.array([(self.nodes[nodeID].map-scale_map).flatten(), scale_map.flatten()])
            #ensure nothing vanishes
            for i in range(2):
                if False:#S[i].sum()==0:
                    print('testing deletion code ', nodeID)
                    e_map = ((local - np.outer(self.nodes[nodeID].map,
                                               self.nodes[nodeID].classifier))**2).sum(axis=-1)
                    worst_pix = np.argmax(e_map)
                    print(S.shape)
                    S[i, worst_pix] = self.nodes[nodeID].map[worst_pix]
                    S[i, worst_pix] = self.nodes[nodeID].map[worst_pix]
            
            
            self.nodes[nodeID + '0'].map = np.minimum(MAX_S,S[0].flatten())
            self.nodes[nodeID + '1'].map = np.minimum(MAX_S,S[1].flatten())
        except AttributeError:
            #print(nodeID, "AttributeError")
            self.nodes[nodeID + '0'].map = np.minimum(MAX_S,self.nodes[nodeID].map)
            pass
        
    def old_train_node_from_spectral(self, nodeID, data):
        try:
            nchild0 = nodeID + '0'
            nchild1 = nodeID + '1'
            spectra = np.array([self.nodes[nchild0].classifier,
                                self.nodes[nchild1].classifier])
            local, nnr = self.remainder_at_node(data, nodeID)
            #S = np.float32(nmf.nmf_project(local.T, spectra.T,
            #                               delta=self.delta,
            #                               mask=self.nodes[nodeID].map))
            
            pos = self.nodes[nodeID].map > 0
            S = np.zeros((2, nnr.shape[0]), dtype=np.float32)
            S[:,pos] = scaled_2class_svm(local[pos], self.nodes[nodeID].map[pos], 
                                  self.nodes[nchild0].classifier,
                                  self.nodes[nchild1].classifier)
            
            
            
            
            #if not self.NR:
            if False:
                local, nnr = self.remainder_at_node(data, nodeID,
                                                    residual_type='full')
                nnr0 = (local - np.outer(S[1],self.nodes[nchild1].classifier)).min(axis=1) > 0
                nnr1 = (local - np.outer(S[0],self.nodes[nchild0].classifier)).min(axis=1) > 0
                
                S[0,~nnr0] = 0
                S[1,~nnr0] = self.nodes[nodeID].map[~nnr0]
                
                S[1,~nnr1] = 0
                S[0,~nnr1] = self.nodes[nodeID].map[~nnr1]
                
                both_neg = np.logical_and(~nnr0, ~nnr1)
                S0_greater = S[0] > S[1]
                S[0, np.logical_and(both_neg, S0_greater)] = \
                    self.nodes[nodeID].map[np.logical_and(both_neg, S0_greater)]
                S[1, np.logical_and(both_neg, S0_greater)] = 0
                S[0, np.logical_and(both_neg, ~S0_greater)] = 0
                S[1, np.logical_and(both_neg, ~S0_greater)] = \
                    self.nodes[nodeID].map[np.logical_and(both_neg, ~S0_greater)]
                
            self.nodes[nchild0].map = S[0]
            self.nodes[nchild1].map = S[1]
        except KeyError:
            nchild0 = nodeID + '0'
            pass
    
    def post_split_apply_nnr(self, nodeID, data):
        try:
            nchild0 = nodeID + '0'
            nchild1 = nodeID + '1'
            
            spectra = np.array([self.nodes[nchild0].classifier,
                                self.nodes[nchild1].classifier])
            local, nnr = self.remainder_at_node(data, nodeID)
            
            S = np.array([self.nodes[nchild0].map,self.nodes[nchild1].map])
            nnr0 = (local - np.outer(S[1],self.nodes[nchild1].classifier)).min(axis=1) > 0
            nnr1 = (local - np.outer(S[0],self.nodes[nchild0].classifier)).min(axis=1) > 0

            S[0,~nnr0] = 0
            S[1,~nnr0] = self.nodes[nodeID].map[~nnr0]

            S[1,~nnr1] = 0
            S[0,~nnr1] = self.nodes[nodeID].map[~nnr1]

            both_neg = np.logical_and(~nnr0, ~nnr1)
            S0_greater = S[0] > S[1]
            S[0, np.logical_and(both_neg, S0_greater)] = \
                self.nodes[nodeID].map[np.logical_and(both_neg, S0_greater)]
            S[1, np.logical_and(both_neg, S0_greater)] = 0
            S[0, np.logical_and(both_neg, ~S0_greater)] = 0
            S[1, np.logical_and(both_neg, ~S0_greater)] = \
                self.nodes[nodeID].map[np.logical_and(both_neg, ~S0_greater)]
                
            self.nodes[nchild0].map = S[0]
            self.nodes[nchild1].map = S[1]
        except KeyError:
            pass

    def spectral_from_spatial(self, level, data):
        self.update_spectra(data, level)
        for k in self.nodes:
            if len(k)==level:
                if self.svm=='threshold':
                    self.train_node_from_spatial(k, data)
                elif self.svm=='all':
                    self.newest_train_node_from_spatial(k, data)
                           
    def spatial_from_spectral(self, level, data):
        for k in self.nodes:
            if len(k)==level:
                self.train_node_from_spectral(k, data)                
                
    def get_open_nodes(self):
        open_nodes = []
        for node in self.nodes:
            if node + '1' in self.nodes:
                node0split = (node + '01') in self.nodes
                node1split = (node + '11') in self.nodes
                if not (node0split or node1split):
                    open_nodes.append(node)
        lens = np.array([len(i) for i in open_nodes])
        maxl = lens.max()
        self.open_nodes = [i for i in open_nodes if len(i) == maxl]
        return self.open_nodes
    
    def train_node(self, data, node):
        i = node
        local, nnr = self.remainder_at_node(data, i)
        initial_M = np.array([self.nodes[i+j].classifier \
                              for j in ['0','1']]).T
        nonzeros = self.nodes[i].map>0
        if self.NR:
            output = nmf.nmf(local.T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
               no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
               mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, mask=self.nodes[i].map,
               initial_M=initial_M, weights=self.weights)
        else:
            nnr_merge = np.minimum(nnr, nonzeros, dtype=bool)
            output = nmf.nmf(local[nnr_merge].T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
               no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
               mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, mask=self.nodes[i].map[nnr_merge],
               initial_M=initial_M, zeta=self.zeta, up_pix=self.up, weights=self.weights[nnr_merge])
        S = np.array((nmf.nmf_project(local.T, output[0],
                           delta=self.delta, mask = self.nodes[i].map
                                     )),
                      dtype=np.float32)
        for j in range(2):
            l = i + str(j)
            self.nodes[l] = Node(spatial_map=S[j], classifier=output[0][:,j])
        
    
    def train_open_nodes(self, data):
        nodes = self.get_open_nodes()
        for i in nodes:
            print('retraining ', i)
            local, nnr = self.remainder_at_node(data, i)
            initial_M = np.array([self.nodes[i+j].classifier \
                                  for j in ['0','1']]).T
            nonzeros = self.nodes[i].map>0
            
            try:
                if self.NR:
                    output = nmf.nmf(local.T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
                       no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
                       mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, mask=self.nodes[i].map,
                       initial_M=initial_M, weights=self.weights)
                else:
                    nnr_merge = np.minimum(nnr, nonzeros, dtype=bool)
                    output = nmf.nmf(local[nnr_merge].T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
                       no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
                       mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, mask=self.nodes[i].map[nnr_merge],
                       initial_M=initial_M, zeta=self.zeta, up_pix=self.up,
                                     weights=self.weights[nnr_merge])
                S = np.array((nmf.nmf_project(local.T, output[0],
                                   delta=self.delta, mask = self.nodes[i].map
                                             )),
                              dtype=np.float32)
                for j in range(2):
                    l = i + str(j)
                    self.nodes[l] = Node(spatial_map=S[j], classifier=output[0][:,j])
            except ValueError:
                self.delete_node(i)
                
    def trim_empty_nodes(self):
        old_nodes = list(self.nodes.keys())
        old_nodes.sort()
        for node in old_nodes[::-1]:
            if self.nodes[node].map.sum() == 0:
                try:
                    self.delete_node(node)
                except KeyError:
                    pass
                    
    def delete_node(self, node):
        print('deleting ', node)
        #assert
        try:
            del self.nodes[node[:-1]].splitter
            print("deleted splitter from ", node[:-1])
        except AttributeError:
            print("no splitter to delete on ", node[:-1])
        all_nodes = list(self.nodes.keys())
        print(all_nodes)
        if (node[-1]=='0')&((node[:-1] + '1') in all_nodes):
            twin = (node[:-1] + '1')
            l = len(twin)
            print(twin)
            #self.nodes.pop(node)
            for i in all_nodes:
                print(i)
                if (node in i):
                    print('inner ', i, ' ', i[:len(node)])
                    if i[:len(node)]==node:
                        print('popping ', i)
                        self.nodes.pop(i)
                        print('popped ', i)
                    else:
                        print(i, ' not popped ', i[:len(node)])
                print('done with ', i)
            for i in all_nodes:
                if (twin in i):
                    print(i)
                    if i[:len(twin)]==twin:
                        new_name = list(i)
                        new_name[l-1] = '0'
                        new_name = ''.join(new_name)
                        print("naming ", i, " as ", new_name)
                        self.nodes[new_name]=self.nodes[i]
                        self.nodes.pop(i)
            
        else:
            self.nodes.pop(node)            
            for i in all_nodes:
                if (node in i):
                    if i[:len(node)]==node:
                        self.nodes.pop(i)
        
        print(self.nodes.keys())
        
            
    
    def initialize_nodes(self, data):
        self.nodes[''] = Node(spatial_map = np.ones(data.shape[0], dtype=bool),
                              classifier = data.mean(axis=0),
                              status=False)
        
        args = np.argsort((data**2).sum(axis=-1))
        L = len(args)
        initial_M = np.array([data[args[int(np.ceil(3*L/4))]], data[args[int(np.floor(L/4))]]]).T
        #initial_M = np.array([data[args[1]], data[args[-2]]]).T
        if self.NR:
            output = nmf.nmf(data.T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
               no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
               mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, weights=self.weights)#, initial_M=initial_M)#, mask=self.nodes[i].map)
        else: 
            output = nmf.nmf(data.T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
               no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
               mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, zeta=self.zeta,
                             up_pix=self.min_size, initial_M = initial_M)
                            #weights=self.weights)#, mask=self.nodes[i].map[nnr])
        S = np.float32(nmf.nmf_project(data.T, output[0], delta=self.delta))#,
                                      #no_negative_residuals=(not self.NR)))
        for i in range(2):
            self.nodes[str(i)] = Node(spatial_map=S[i], classifier=output[0][:,i])
        self.nodes[''].status=False
                         
    def nodes_to_split(self):
        end_nodes = self.get_end_nodes()
        splitting_nodes = []
        for node in end_nodes:
            if self.nodes[node].map.sum() > self.splitting_size:
                splitting_nodes.append(node)
        return splitting_nodes
        
    def add_another_node_layer(self, data):
        to_split = self.nodes_to_split()
        level = self.get_depth()
        lowest_nodes = [i for i in self.nodes if len(i)==level]
        #print(lowest_nodes)
        for i in lowest_nodes:
            if i in to_split:
                print("splittting ", i)
                local, nnr = self.remainder_at_node(data, i)
                nonzeros = self.nodes[i].map>0
                if self.NR:
                    output = nmf.nmf(local[nonzeros].T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
                           no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
                       mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, mask=self.nodes[i].map[nonzeros],
                                    weights=self.weights)
                    S = np.array((nmf.nmf_project(local.T, output[0],
                             delta=self.delta, mask = self.nodes[i].map)), dtype=np.float32)
                else: 
                    nnr_merge = (np.minimum(nnr, nonzeros, dtype=bool)).flatten()
                    #print(nnr_merge.sum(), nnr_merge)
                    #l_vals = np.multiply(local[nonzeros].T, local[nonzeros].T).sum(axis=1)
                    #selected = np.argsort(self.nodes[i].map[nonzeros])[-2:]
                    avg = self.nodes[i].classifier
                    l_vals = np.sum((local[nonzeros]-avg)**2, axis=-1)
                    selected = np.argsort(l_vals)[:2]#self.nodes[i].map[nonzeros])[-2:]
                    initial_M = local[nonzeros][selected].T/self.nodes[i].map[nonzeros][selected]
                    if np.isnan(initial_M).any():
                        initial_M = data[nonzeros][selected].T

                    output = nmf.nmf(local[nnr_merge].T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
                       no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
                       mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, mask=self.nodes[i].map[nnr_merge], zeta=self.zeta,
                                    up_pix=self.min_size,  weights=self.weights[nnr_merge])#, initial_M = initial_M)
                    #print(output[0].shape,output[0])
                    S = np.array((nmf.nmf_project(local.T, output[0],#output[0],
                                 delta=self.delta, mask = self.nodes[i].map,
                                 no_negative_residuals=False)), dtype=np.float32)
                for j in range(2):
                    l = i + str(j)
                    #Sprint(l)
                    self.nodes[l] = Node(spatial_map=S[j], classifier=output[0][:,j])#initial_M[:,j])
                self.nodes[i].outmembers= output[0].T
            else:
                l = i + '0'
                self.nodes[l] = Node(spatial_map=self.nodes[i].map, classifier=self.nodes[i].classifier)
            self.nodes[i].status=False
                
    def get_one_child_nodes(self):
        one_child_nodes =  []
        for node in self.nodes:
            if (node + '0' in self.nodes)&(node + '1' not in self.nodes):
                one_child_nodes.append(node)
        one_child_nodes.sort(key=len)
        return one_child_nodes
    
    def propagate_one_child_nodes(self):
        ocns = self.get_one_child_nodes()
        for node in ocns:
            self.nodes[node+'0'].map = self.nodes[node].map
            self.nodes[node+'0'].classifier = self.nodes[node].classifier
            
    def weighted_average(self, node, data, p):
        spatial = self.nodes[node].map
        weight_sum = (np.multiply(self.weights, (1-np.abs(1-spatial))**p)).sum()
        weight_spectra = (np.multiply(self.weights, (1-np.abs(1-spatial))**p)*data.T).sum(axis=-1)
        if weight_sum > 0:
            return (weight_spectra / weight_sum)
        else:
            return 0*weight_spectra


    def setS_to_threshold(self, S_to_set, sseti, S_to_read):
        n_max = self.min_size
        
        maxes = np.argsort(S_to_read)[-(n_max):]
        below_threshold = np.zeros_like(S_to_read, dtype=bool)# < (1-self.threshold)) & (S[i] > 0)
        below_threshold[maxes] = ((S_to_read[maxes]) < (1-self.threshold)) & (S_to_read[maxes] > 0)
        #print(maxes.shape, below_threshold.shape)
        delta = (1-self.threshold)/S_to_read[below_threshold] 
        for j in range(len(S_to_set)):
            if j != sseti:
                S_to_set[j, below_threshold] *= (self.threshold)/np.maximum(1-S_to_set[sseti, below_threshold],1e-16)
        S_to_set[sseti, below_threshold] *= delta
        #print(S[i, maxes])

        return S_to_set
    
    def set_to_threshold(self, S, do_set=True):
        n_max = self.min_size
        if not do_set:
            for i in range(len(S)):
                maxes = np.argsort(S[i])[-(n_max):]
                #print(S[i, maxes])
                #print(S.shape)
                
        for i in range(len(S)):
            maxes = np.argsort(S[i])[-(n_max):]
            #print(S[i, maxes])
            below_threshold = np.zeros_like(S[i], dtype=bool)# < (1-self.threshold)) & (S[i] > 0)
            below_threshold[maxes] = ((S[i,maxes]) < (1-self.threshold)) & (S[i, maxes] > 0)
            #print(maxes.shape, below_threshold.shape)
            delta = (1-self.threshold)/S[i, below_threshold] 
            for j in range(len(S)):
                if j != i:
                    S[j, below_threshold] *= (self.threshold)/(1-S[i, below_threshold])
            S[i, below_threshold] *= delta
            #print(S[i, maxes])
            
        return S

    def display_level(self, level):
        count=0
        for i in self.nodes:
            if len(i)==level:
                count += 1
        print(count)

        fig, ax = plt.subplots(count, figsize=(8,2*count))
        for i, a in enumerate(ax):
            a.set_xticks([])
            a.set_yticks([])

        counter = 0
        for i in self.nodes:
            if len(i)==level:
                ax[counter].imshow(np.rot90(self.nodes[i].map.reshape(self.plot_size)),
                                   aspect=self.plot_aspect, vmin=0, vmax=1, interpolation='nearest')
                ax[counter].set_ylabel(self.nodes[i].map.sum())
                ax[counter].set_title(i + " {:.2f}".format(self.nodes[i].map.max()))
                counter += 1
        plt.show()
                
    

    def partition_node(self, node, data):
        min_size = self.min_size#int(np.ceil(np.max([self.min_size / 2**(len(node)), 1])))
        print("partitioning ", node)
        if (node +'0' and node+'1') in self.nodes:
            lima = self.threshold 
            limb = np.sort(np.abs(1-self.nodes[node +'0'].map))[(min_size+1)]
            if limb == 0:
                try:
                    limb = self.nodes[node+'0'].map[self.nodes[node+'0'].map>0].min()
                except ValueError:
                    other_min = self.nodes[node+'1'].map[self.nodes[node+'1'].map>0].min()
                    arg = np.argmin(np.abs(other_min-self.nodes[node+'1'].map))
                    self.nodes[node+'0'].map[arg] = self.nodes[node+'1'].map[arg]
                    self.nodes[node+'1'].map[arg] = 0
            print(lima, limb)
            #print(np.sort(self.nodes[node +'0'].map)[-(min_size+1):])
            map0 = np.array(np.abs(1-self.nodes[node +'0'].map) <= min(lima, limb))
            #if map0.sum()==0:
            #    arg = np.argmax(self.nodes[node +'0'].map)
            #    map0[arg] = True
            lima = self.threshold 
            limb = np.sort(np.abs(1-self.nodes[node +'1'].map))[(min_size+1)]
            if limb == 0:
                try:
                    limb = self.nodes[node+'1'].map[self.nodes[node+'1'].map>0].min()
                except ValueError:
                    other_min = self.nodes[node+'0'].map[self.nodes[node+'0'].map>0].min()
                    arg = np.argmin(np.abs(other_min-self.nodes[node+'0'].map))
                    self.nodes[node+'1'].map[arg] = self.nodes[node+'0'].map[arg]
                    self.nodes[node+'0'].map[arg] = 0
            print(lima, limb)
            #print(np.sort(self.nodes[node +'1'].map)[-(min_size+1):])
            map1 = np.array(np.abs(1-self.nodes[node +'1'].map) <=  min(lima, limb))
                
            Xs = data[np.logical_xor(map1, map0)]
            ys = np.zeros(np.logical_xor(map1, map0).sum(), dtype=bool)

            ys[map1[np.logical_xor(map1, map0)]]=True
            if len(set(ys.tolist())) == 1:
                return np.nan, np.nan, np.nan
            
            sdev = np.std(Xs)
            
            print("about to train the svm")
            test1 = svm.LinearSVC(max_iter=100000, dual=False, tol=1e-4, C=self.Cnorm/sdev**2,
                                  class_weight='balanced')
            test1.fit(Xs, ys)
            x_guess = test1.predict(Xs)
            print(np.mean(x_guess[ys==1]==ys[ys==1]))
            print(np.mean(x_guess[ys==0]==ys[ys==0]))
            d_out = data@test1.coef_.T + test1.intercept_
            d_out2 = classify_from_partition(data, test1.coef_,test1.intercept_)

            print("svm trained")
            return test1.coef_*(1-self.threshold), test1.intercept_*(1-self.threshold), map1 | map0
        else:
            print('node {} as no children'.format(node))
            
            
    def save(self, filename, title='Deep Endmember Hierarchy', save_labels=()):
        d = self.get_depth()
        
        class Save_Node(tab.IsDescription):
            name = tab.StringCol(itemsize=d)
            classifier = tab.Float32Col(self.nodes[''].classifier.shape)
            splitter_w = tab.Float32Col(self.nodes[''].classifier.shape)
            intercept = tab.Float32Col()
            
        if filename[-2:] != 'h5':
            print("can only save as h5 file")
            return -1
        
        h5file = tab.open_file(filename, mode='w', title=title) 
        try:
            table = h5file.create_table("/", "nodes", Save_Node)
            h5file.create_array("/", "history", self.history)
            node_save = table.row
            for node in self.nodes:
                node_save['name'] = node
                node_save['classifier'] = self.nodes[node].classifier.astype(np.float32)
                try:
                    node_save['splitter_w'] = self.nodes[node].splitter[0].astype(np.float32)
                    node_save['intercept'] = float(self.nodes[node].splitter[1])
                except AttributeError:
                    pass
                node_save.append()
                table.flush()
            table.attrs.NR = self.NR 
            table.attrs.subfit_parameters = self.subfit_parameters
            table.attrs.remainder_id = self.remainder_id
            table.attrs.splitting_size = self.splitting_size
            table.attrs.max_depth = self.max_depth
            table.attrs.root = self.root
            table.attrs.threshold = self.threshold
            table.attrs.training = self.training
            table.attrs.min_size = self.min_size
            table.attrs.tol = self.tol 
            table.attrs.tolb = self.tolb
            table.attrs.mb_l = self.mb_l  
            table.attrs.mb_in = self.mb_in
            table.attrs.svm = self.svm
            table.attrs.mixed_layers = self.mixed_layers
            table.attrs.double_fit=self.double_fit
            table.attrs.approach_rate=self.approach_rate
            table.attrs.spread = self.spread
            if len(save_labels)>0:
                table.attrs.cluster_labels = save_labels
            #self.plot_size = (239,171)
            #self.plot_aspect = 0.3
            table.attrs.zeta = self.zeta
            h5file.close()
        except ValueError:
            print("ValueError occured on saving")
            h5file.close()
            
        return 0
    
    def load(self, filename):
        self.h5file = tab.open_file(filename, mode='a')
        table = self.h5file.root.nodes
        for row in table.iterrows():
            node = row['name'].decode('ascii')
            self.nodes[node] = Node(None, row['classifier'])
            if np.sum(np.abs(row['splitter_w'])) > 0:
                self.nodes[node].splitter = (row['splitter_w'], row['intercept']) 
                
    def display_spectra(self, spectra_list, names = (), wl=(), normalizer = 1, **kwargs):
        if len(wl)==0:
            wl = np.arange(len(self.nodes[''].classifier))
        for i, x in enumerate(spectra_list):
            if len(names)==len(spectra_list):
                plt.plot(wl, self.nodes[x].classifier/normalizer, label = names[i])
            else:
                plt.plot(wl, self.nodes[x].classifier/normalizer, label = i)
        plt.legend(**kwargs)

    def display_map(self, spectrum, ax=plt):
        ax.imshow(np.rot90(self.nodes[spectrum].map.reshape(self.plot_size)),
                  aspect=self.plot_aspect)

    def display_maps(self, spectra, names = (), figsize=()):
        count = len(spectra)
        if len(figsize) > 0:
            fig, ax = plt.subplots(count, figsize=figsize)
        else:
            fig, ax = plt.subplots(count)
        for i, spec in enumerate(spectra):
            if len(names)==len(spectra):
                name = names[i]
            else:
                name = spec
            self.display_map(spec, ax=ax[i])
            ax[i].set_title(name)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
            
    def update_abundances_MUR(self, M, S, data):
        # will be performed pixel-wise
        S_out = np.zeros_like(S)
        Ma = np.append(M.T, self.delta*np.ones((1,M.shape[0])), axis=0)
        for i in range(len(S.T)):
            Da = np.append(data.T[:,i], self.delta)

            num = Da.T
            denom = Ma @ S[:,i]
            
            #print(num.shape, denom.shape)
            
            num = Ma.T @ num 
            denom = Ma.T @ denom

            S_out[:,i] = S[:,i] * num / denom
        return S_out
    
    def update_spectra_MUR(self, M, S, data):
        M_out = np.zeros_like(M)
        for i in range(len(M.T)):
            num = data[:,i]
            denom = M[:,i] @ S

            #print(num.shape, denom.shape)

            num = np.multiply(self.weights, num)
            denom = np.multiply(self.weights, denom)

            num = num @ S.T
            denom = denom @ S.T

            denom = np.maximum(denom, 1e-32)
            M_out[:,i] = M[:,i] * num / denom
        return M_out

def classify_from_partition(data, coef, intercept, threshold=0, spread=1):
        base = (coef@data.T + intercept + 1)/(2)
        trimmed = np.array(np.maximum(np.minimum(base, 1.0),0), dtype=np.float32)
        return trimmed
    
def svm_from_classifiers(image, classifier0, classifier1):
    pw = (classifier1 - classifier0)
    w = 2 * pw / np.dot(pw, pw)
    d = -np.dot(pw, classifier1 + classifier0)/ np.dot(pw, pw)
    
    #print(image.shape, w.shape)
    lambdas = w@image + d
    lambdas = (lambdas + 1)/2
    lambdas = np.minimum(1, lambdas)
    lambdas = np.maximum(0, lambdas)
    return lambdas

def classifiers_2_svm(classifier0, classifier1):
    pw = (classifier1 - classifier0)
    
    w = 2 * pw / np.dot(pw, pw)
    if np.isnan(w).any():
        print("theres a nan ", pw, classifier0, classifier1)
    d = -np.dot(pw, classifier1 + classifier0)/ np.dot(pw, pw)
    
    return w, d

def scaled_2class_svm(resid_image, scaling_factors, classifier0, classifier1):
    scaled_image = (resid_image).T / scaling_factors
    scaled_lambdas = svm_from_classifiers(scaled_image, classifier0, classifier1)
    two_classes = [1-scaled_lambdas, scaled_lambdas]
    two_classes *= scaling_factors
    return two_classes
