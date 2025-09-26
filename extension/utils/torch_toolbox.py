import torch
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import numpy as np

def computeBarycentric2D(p, UV, F):
    '''
    computeBarycentric2D computes berycentric coordinate or queryUV in fUV
    Inputs:
        p  length 2 array
        UV |UV| by 2 array
        F  |F| by 3 array
    Outputs:
        B  |F| by 3 array of barycentric coordinate from p to all F
    '''

    a = UV[F[:,0],:]
    b = UV[F[:,1],:]
    c = UV[F[:,2],:]

    nF = F.size(0)
    v0 = b-a
    v1 = c-a
    v2 = p.unsqueeze(0)-a

    d00 = v0.mul(v0).sum(1)
    d01 = v0.mul(v1).sum(1)
    d11 = v1.mul(v1).sum(1)
    d20 = v2.mul(v0).sum(1)
    d21 = v2.mul(v1).sum(1)
    denom = d00.mul(d11) - d01.mul(d01)
    denom = 1.0 / denom

    v = (d11.mul(d20) - d01.mul(d21)).mul(denom)
    w = (d00.mul(d21) - d01.mul(d20)).mul(denom)
    u = 1 - v - w

    B = torch.cat((u.unsqueeze(1),v.unsqueeze(1),w.unsqueeze(1)), dim = 1)
    return B

def faceAreas(V, F):
    """
    FACEAREAS computes area per face 

    Input:
        V (|V|,3) torch float tensor of vertex positions
        F (|F|,3) torch long tensor of face indices
    Output:
        FA (|F|,) torch tensor of face area
    """
    vec1 = V[F[:,1],:] - V[F[:,0],:]
    vec2 = V[F[:,2],:] - V[F[:,0],:]
    FN = torch.cross(vec1, vec2) / 2
    FA = torch.sqrt(torch.sum(FN.pow(2),1))
    return FA

def faceNormals(V, F):
    """
    FACENORMALS computes unit normals per face 

    Input:
        V (|V|,3) torch float tensor of vertex positions
        F (|F|,3) torch long tensor of face indices
    Output:
        FN (|F|,3) torch tensor of face normals
    """
    vec1 = V[F[:,1],:] - V[F[:,0],:]
    vec2 = V[F[:,2],:] - V[F[:,0],:]
    FN = torch.cross(vec1, vec2) / 2
    l2norm = torch.sqrt(torch.sum(FN.pow(2),1))
    nCol = FN.size()[1]
    for cIdx in range(nCol):
        FN[:,cIdx] /= l2norm
    return FN

def findIdx(F, VIdx):
    '''
    FINDIDX finds desired indices in a torch tensor

    Inputs:
    F: |F|-by-dim torch tensor 
    VIdx: a list of indices

    Output:
    r, c: row/colummn indices in the torch tensor
    '''

    def isin(ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)

    mask = isin(F.view(-1),VIdx)
    try:
        nDim = F.shape[1]
    except:
        nDim = 1
    r = torch.floor(torch.where(mask)[0] / (nDim*1.0) ).long()
    c = torch.where(mask)[0] % nDim
    return r,c

def intersect1d(tensor1, tensor2):
    '''
    intersect1d return intersected elements between tensor1 and tensor2
    '''
    aux = torch.cat((tensor1, tensor2),dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]

def isInTriangle(P,UV,F):
    # INOUTTRI detect inside(True)/outside(False) of a query point P from 
    # a triangle mesh in R2 (V,F)
    
    # Inputs:
    #     P:  length 2 vector
    #     UV: nUV x 2 vertices in 2D
    #     F:  nF x 3 faces
       
    # Outputs:
    #     isInF: nF x 1 bool whether P is in triangle Fi

    nUV = UV.size(0)
    P1 = UV[F[:,0],:]
    P2 = UV[F[:,1],:]
    P3 = UV[F[:,2],:]
    P12 = P1 - P2
    P23 = P2 - P3
    P31 = P3 - P1
    
    detP31P23 = rowwiseDet2D(P31,  P23)
    detP30P23 = rowwiseDet2D(P3-P, P23)
    detP12P31 = rowwiseDet2D(P12,  P31)
    detP10P31 = rowwiseDet2D(P1-P, P31)
    detP23P12 = rowwiseDet2D(P23,  P12)
    detP20P12 = rowwiseDet2D(P2-P, P12)

    isInF = (detP31P23 * detP30P23) >= -1e-6
    isInF = isInF & ((detP12P31 * detP10P31) >= -1e-6)
    isInF = isInF & ((detP23P12 * detP20P12) >= -1e-6)
    
    return isInF

def normalizeRow(X):
    """
    NORMALIZEROW normalizes the l2-norm of each row in a np array 

    Input:
        X: n-by-m torch tensor
    Output:
        X_normalized: n-by-m row normalized torch tensor
    """
    l2norm = torch.sqrt(torch.sum(X.pow(2),1))
    X = X / l2norm.unsqueeze(1)
    return X

def normalizeUnitArea(V,F):
    '''
    NORMALIZEUNITAREA normalize a shape to have total surface area 1

    Inputs:
        V (|V|,3) torch array of vertex positions
        F (|F|,3) torch array of face indices

    Outputs:
        V |V|-by-3 torch array of normalized vertex positions
    '''
    totalArea = torch.sum(faceAreas(V,F))
    V = V / torch.sqrt(totalArea)
    return V

def normalizeUnitCube(V):
    '''
    NORMALIZEUNITCUBE normalize a shape to the bounding box by 0.5,0.5,0.5

    Inputs:
        V (|V|,3) torch array of vertex positions

    Outputs:
        V |V|-by-3 torch array of normalized vertex positions
    '''
    V = V - torch.min(V,0)[0].unsqueeze(0)
    V = V / torch.max(V.view(-1)) / 2.0
    return V

def plotMesh(Vin,F, \
    showEdges=False,\
    alpha=0.5):
    """
    PLOTMESH plot a triangle mesh

    Input:
      V (|V|,3) torch tensor of vertex positions
      F (|F|,3) torch tensor of face indices
    Output:
      None
    """
    V = Vin.clone()
    if V.size(1) == 2:
        V = torch.cat((V, torch.zeros((V.size(0),1))), dim=1)
    FN = faceNormals(V,F)
    V = V.data.numpy()
    F = F.data.numpy()
    FN = FN.data.numpy()

    # compute colors for rendering
    
    z = (FN[:,2] + 3) / 5
    face_color = np.array([144.0/ 255.0, 210.0/ 255.0, 236.0/ 255.0])
    face_color = z[:,None]*face_color

    # open a figure
    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca(projection='3d')

    # get vertices 
    vtx = V[F,:]
    
    # plot
    if showEdges == True:
        mesh = a3.art3d.Poly3DCollection(vtx, linewidths=.5,  edgecolors=[0,0,0], alpha=alpha)
    else:
        mesh = a3.art3d.Poly3DCollection(vtx, alpha=alpha)
    
    # add face color
    mesh.set_facecolor(face_color)
    
    # add mesh to figures
    ax.add_collection3d(mesh)

    # set figure axis 
    actV = np.unique(F.flatten())
    axisRange = np.array([np.max(V[actV,0])-np.min(V[actV,0]), np.max(V[actV,1])-np.min(V[actV,1]), np.max(V[actV,2])-np.min(V[actV,2])])
    r = np.max(axisRange) / 2.0
    mean = np.mean(V[actV,:], 0)
    ax.set_xlim(mean[0]-r, mean[0]+r)
    ax.set_ylim(mean[1]-r, mean[1]+r)
    ax.set_zlim(mean[2]-r, mean[2]+r)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax

def randomSampleMesh(V, F, nPt):
    """
    RANDOMSAMPLEMESH randomly samples nPt points on a triangle mesh

    Input:
        V (|V|,3) torch float tensor of vertex positions
        F (|F|,3) torch long tensor of face indices
        nPt number of points to sample
    Output:
        P (nPt,3) torch float tensor of sampled point positions
    """
    nF = F.size()[0]

    FIdx = torch.randint(nF, (nPt,))
    bary = torch.rand(nPt, 3)
    rowSum = torch.sum(bary, 1)
    bary[:,0] /= rowSum
    bary[:,1] /= rowSum
    bary[:,2] /= rowSum

    b0 = bary[:,0:1].repeat(1,3)
    b1 = bary[:,1:2].repeat(1,3)
    b2 = bary[:,2:3].repeat(1,3)

    P = b0*V[F[FIdx,0],:] + b1*V[F[FIdx,1],:] + b2*V[F[FIdx,2],:]
    return P

def readOBJ(filepath):
    """
    READOBJ read .obj file

    Input:
      filepath a string of mesh file path
    Output:
      V (|V|,3) torch tensor of vertex positions (float)
      F (|F|,3) torch tensor of face indices (long)
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
    V_th = torch.from_numpy(V).float()
    F_th = torch.from_numpy(F).long()
    return V_th, F_th

def roll1d(x, n):  
    return torch.cat((x[-n:], x[:-n]))

def rowwiseDet2D(v1List,v2List):
    '''
    rowwiseDet2D computes the determinant between two sets of 2D vectors.
    This is equivalent of 

    for ii in range(v1List.size(0)):
        v1 = v1List[ii,:];
        v2 = v2List[ii,:];
        detList[ii] = det(v1,v2);

    Inputs:
      v1List nV x 2 matrix
      v2List nV x 2 matrix

    Outputs:
      detList nV x 1 determinant
    '''
    assert(v1List.size(1) == 2)
    assert(v2List.size(1) == 2)
    assert(v1List.size(0) == v2List.size(0))

    nV = v1List.size(0)
    M = torch.zeros((2,2,nV))
    M[0,:,:] = v1List.t()
    M[1,:,:] = v2List.t()
    Mvec = M.view(2*2,nV)
    detList = Mvec[0,:] * Mvec[3,:] - Mvec[1,:] * Mvec[2,:]
    return detList

def setdiff1d(tensor1, tensor2):
    '''
    setdiff1d returns non-intersected elements between tensor1 and tensor2
    '''
    idx = torch.ones_like(tensor1, dtype=torch.bool)
    for ele in tensor2:
        idx = idx & (tensor1 != ele)
    diffEle = tensor1[idx]
    return diffEle

def vertexNormals(V,F):
    vec1 = V[F[:,1],:] - V[F[:,0],:]
    vec2 = V[F[:,2],:] - V[F[:,0],:]
    FN = torch.cross(vec1, vec2) / 2

    rIdx = F.view(-1)
    cIdx = torch.arange(F.size(0))
    cIdx = cIdx.unsqueeze(1).repeat(1,3).view(-1)
    val = torch.ones(cIdx.size(0))

    I = torch.cat([rIdx,cIdx], 0).reshape(2, -1)
    W = torch.sparse.FloatTensor(I, val, torch.Size([V.size(0),F.size(0)]))
    VN = torch.sparse.mm(W, FN)
    VN = normalizeRow(VN)
    return VN

def writeOBJ(fileName,V_torch,F_torch):
    '''
    WRITEOBJ write a mesh into OBJ file

    Input:
      filepath a string of mesh file path
      V (|V|,3) torch tensor of vertex positions (float)
      F (|F|,3) torch tensor of face indices (long)
    '''
    V = V_torch.data.numpy()
    F = F_torch.data.numpy()
    f = open(fileName, 'w')
    for ii in range(V.shape[0]):
        string = 'v ' + str(V[ii,0]) + ' ' + str(V[ii,1]) + ' ' + str(V[ii,2]) + '\n'
        f.write(string)
    Ftemp = F + 1
    for ii in range(F.shape[0]):
        string = 'f ' + str(Ftemp[ii,0]) + ' ' + str(Ftemp[ii,1]) + ' ' + str(Ftemp[ii,2]) + '\n'
        f.write(string)
    f.close()
