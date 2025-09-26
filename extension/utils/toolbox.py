import numpy as np
import scipy
import scipy.sparse

def adjacencyMat(F):
    '''
    adjacencyMat computes vertex adjacency matrix

    Inputs:
    F: |F|-by-3 numpy ndarray of face indices

    Outputs:
    A: sparse.csr_matrix with size |V|-by-|V|
    '''

    idx = np.array([[0,1], [1,2], [2,0]]) # assume we have simplex with DOF=3
    edgeIdx1 = np.reshape(F[:,idx[:,0]], (np.product(F.shape)))
    edgeIdx2 = np.reshape(F[:,idx[:,1]], (np.product(F.shape)))
    data = np.ones([len(edgeIdx1)])
    numVert = np.amax(F)+1
    A = scipy.sparse.csr_matrix((data, (edgeIdx1, edgeIdx2)), shape=(numVert,numVert), dtype = np.int32)
    return A

def findIdx(F, VIdx):
    '''
    FINDIDX finds desired indices in the ndarray

    Inputs:
    F: |F|-by-dim numpy ndarray 
    VIdx: a list of indices

    Output:
    r, c: row/colummn indices in the ndarray
    '''
    mask = np.in1d(F.flatten(),VIdx)
    try:
        nDim = F.shape[1]
    except:
        nDim = 1
    r = np.floor(np.where(mask)[0] / (nDim*1.0) ).astype(int)
    c = np.where(mask)[0] % nDim
    return r,c

def midPointUpsampling(V,F,numIter=1):
    '''
    midPointUpsampling do mid point upsampling 

    Inputs:
        V (|V|,3) numpy array of vertex positions
        F (|F|,3) numpy array of face indices
        numIter number of upsampling to perform

    Outputs:
        V |V|-by-3 numpy array of new vertex positions
        F |F|-by-3 numpy array of new face indices
        upOpt |Vup|-by-|V| numpy array of upsampling operator

    TODO:
        add boundary constraints 
    '''
    for iter in range(numIter):
        nV = V.shape[0]
        nF = F.shape[0]

        # compute new vertex positions
        hE = np.concatenate( (F[:,[0,1]], F[:,[1,2]], F[:,[2,0]]), axis=0 )
        hE = np.sort(hE, axis = 1)
        E, hE2E = np.unique(hE, axis=0, return_inverse=True)
        nE = E.shape[0]
        newV = (V[E[:,0],:] + V[E[:,1],:]) / 2.0
        V = np.concatenate( (V, newV), axis = 0 )

        # compute updated connectivity
        i2 = nV       + np.arange(nF)
        i0 = nV+nF    + np.arange(nF)
        i1 = nV+nF+nF + np.arange(nF)

        hEF0 = np.concatenate( (F[:,0:1], i2[:,None], i1[:,None]), axis=1 )
        hEF1 = np.concatenate( (F[:,1:2], i0[:,None], i2[:,None]), axis=1 )
        hEF2 = np.concatenate( (F[:,2:3], i1[:,None], i0[:,None]), axis=1 )
        hEF3 = np.concatenate( (i0[:,None], i1[:,None], i2[:,None]), axis=1 )
        hEF  = np.concatenate( (hEF0, hEF1, hEF2, hEF3), axis=0 )

        hE2E =  np.concatenate( (np.arange(nV), hE2E+nV), axis=0 )
        uniqV = np.unique(F)
        F = hE2E[hEF]

        # upsampling for odd vertices
        rIdx = uniqV
        cIdx = uniqV
        val = np.ones((len(uniqV),))

        # upsampling for even vertices
        rIdx = np.concatenate( (rIdx, nV+np.arange(nE),  nV+np.arange(nE)) )
        cIdx = np.concatenate( (cIdx, E[:,0],  E[:,1]) )
        val = np.concatenate( (val, np.ones(2*nE)*0.5) )

        # upsampling operator
        if iter == 0:
            S = scipy.sparse.coo_matrix( (val, (rIdx,cIdx)), shape = (nV+nE, nV) )
        else:
            tmp = scipy.sparse.coo_matrix( (val, (rIdx,cIdx)), shape = (nV+nE, nV) )
            S = tmp * S

    return V, F, S

def readOBJ(filepath):
    """
    READOBJ read .obj file

    Input:
      filepath a string of mesh file path
    Output:
      V (|V|,3) numpy array of vertex positions
      F (|F|,3) numpy array of face indices
    """
    V = []
    F = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    while True:
        for line in lines:
            if line == "":
                break
            elif line.strip().startswith("vn"):
                continue
            elif line.strip().startswith("vt"):
                continue
            elif line.strip().startswith("v"):
                vertices = line.replace("\n", "").split(" ")[1:]
                vertices = np.delete(vertices,np.argwhere(vertices == np.array([''])).flatten())
                V.append(list(map(float, vertices)))
            elif line.strip().startswith("f"):
                t_index_list = []
                for t in line.replace("\n", "").split(" ")[1:]:
                    t_index = t.split("/")[0]
                    try: 
                        t_index_list.append(int(t_index) - 1)
                    except ValueError:
                        continue
                F.append(t_index_list)
            else:
                continue
        break
    V = np.asarray(V)
    F = np.asarray(F)
    return V, F

def writeOBJ(fileName,V,F):
    f = open(fileName, 'w')
    for ii in range(V.shape[0]):
        string = 'v ' + str(V[ii,0]) + ' ' + str(V[ii,1]) + ' ' + str(V[ii,2]) + '\n'
        f.write(string)
    Ftemp = F + 1
    for ii in range(F.shape[0]):
        string = 'f ' + str(Ftemp[ii,0]) + ' ' + str(Ftemp[ii,1]) + ' ' + str(Ftemp[ii,2]) + '\n'
        f.write(string)
    f.close()