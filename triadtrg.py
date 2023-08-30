import numpy as np
import sys
from scipy.linalg import eigh
import warnings
# from sklearn.utils.extmath import randomized_svd


def getU(q, nums):
    # qs = q.shape
    # O = eigh(q, subset_by_index=[qs[0]-nums, qs[0]-1])
    # O = eigh(q, eigvals=(qs[0]-nums, qs[0]-1))
    if not np.allclose(q, q.conjugate().transpose()):
        warnings.warn("q matrix is not Hermitian by allclose.")
    # O = eigh(q)
    O = np.linalg.eigh(q)
    U = O[1]
    e = O[0]
    idx = np.abs(e).argsort()[::-1]
    # print("largest =", np.max(e))
    sys.stdout.flush()
    return (U[:,idx])[:,:nums]


# def split(matrix, cut=None, split='both'):
#     """
#     Splits a matrix in half using the SVD.
    
#     Parameters
#     ----------
#     matrix : The matrix to be split.
#     cut    : (optional, default None) The number of states to keep on the internal 
#              index.

#     Returns
#     -------
#     left  : The left side of the split matrix.
#     right : The right side of the split matrix.
#     alpha : The size of the internal index.

#     """
#     # assert np.allclose(np.dot(left, np.dot(np.diag(s), right)), matrix)
#     if (cut is not None):
#         left, s, right = randomized_svd(matrix, n_components=cut) 
#         alpha = min([len(s[s > 1e-14]), cut])
#         if split == 'both':
#             left = np.dot(left, np.diag(np.sqrt(s))[:, :alpha])
#             right = np.dot(np.diag(np.sqrt(s))[:alpha, :], right)
#             # assert np.allclose(left.dot(right), matrix)
#             return (left, right, alpha)
#         elif split == 'left':
#             # alpha = min([len(s[s > 1e-14]), cut])
#             left = np.dot(left, np.diag(s)[:, :alpha])
#             right = right[:alpha, :]
#             # assert np.allclose(left.dot(right), matrix)
#             return (left, right, alpha)
#         elif split == 'right':
#             # alpha = min([len(s[s > 1e-14]), cut])
#             left = left[:, :alpha]
#             right = np.dot(np.diag(s)[:alpha, :], right)
#             # assert np.allclose(left.dot(right), matrix)
#             return (left, right, alpha)
#         else:
#             raise ValueError("split must be a valid option.")
#     else:
#         left, s, right = np.linalg.svd(matrix, full_matrices=False)
#         if split == 'both':
#             alpha = len(s[s > 1e-14])
#             left = np.dot(left, np.diag(np.sqrt(s))[:, :alpha])
#             right = np.dot(np.diag(np.sqrt(s))[:alpha, :], right)
#             # assert np.allclose(left.dot(right), matrix)
#             return (left, right, alpha)
#         elif split == 'left':
#             alpha = len(s[s > 1e-14])
#             left = np.dot(left, np.diag(s)[:, :alpha])
#             right = right[:alpha, :]
#             # assert np.allclose(left.dot(right), matrix)
#             return (left, right, alpha)
#         elif split == 'right':
#             alpha = len(s[s > 1e-14])
#             left = left[:, :alpha]
#             right = np.dot(np.diag(s)[:alpha, :], right)
#             # assert np.allclose(left.dot(right), matrix)
#             return (left, right, alpha)
#         else:
#             raise ValueError("split must be a valid option.")


def split(matrix, cut=None, split='both'):
    """
    Splits a matrix in half using the SVD.
    
    Parameters
    ----------
    matrix : The matrix to be split.
    cut    : (optional, default None) The number of states to keep on the internal 
             index.

    Returns
    -------
    left  : The left side of the split matrix.
    right : The right side of the split matrix.
    alpha : The size of the internal index.

    """
    # assert np.allclose(np.dot(left, np.dot(np.diag(s), right)), matrix)
    if (cut is not None):
        left, s, right = np.linalg.svd(matrix, full_matrices=False)
        # left, s, right = randomized_svd(matrix, n_components=cut) 
        alpha = min([len(s[s > 1e-14]), cut])
        if split == 'both':
            left = np.dot(left, np.diag(np.sqrt(s))[:, :alpha])
            right = np.dot(np.diag(np.sqrt(s))[:alpha, :], right)
            # assert np.allclose(left.dot(right), matrix)
            return (left, right, alpha)
        elif split == 'left':
            # alpha = min([len(s[s > 1e-14]), cut])
            left = np.dot(left, np.diag(s)[:, :alpha])
            right = right[:alpha, :]
            # assert np.allclose(left.dot(right), matrix)
            return (left, right, alpha)
        elif split == 'right':
            # alpha = min([len(s[s > 1e-14]), cut])
            left = left[:, :alpha]
            right = np.dot(np.diag(s)[:alpha, :], right)
            # assert np.allclose(left.dot(right), matrix)
            return (left, right, alpha)
        else:
            raise ValueError("split must be a valid option.")
    else:
        left, s, right = np.linalg.svd(matrix, full_matrices=False)
        if split == 'both':
            alpha = len(s[s > 1e-14])
            left = np.dot(left, np.diag(np.sqrt(s))[:, :alpha])
            right = np.dot(np.diag(np.sqrt(s))[:alpha, :], right)
            # assert np.allclose(left.dot(right), matrix)
            return (left, right, alpha)
        elif split == 'left':
            alpha = len(s[s > 1e-14])
            left = np.dot(left, np.diag(s)[:, :alpha])
            right = right[:alpha, :]
            # assert np.allclose(left.dot(right), matrix)
            return (left, right, alpha)
        elif split == 'right':
            alpha = len(s[s > 1e-14])
            left = left[:, :alpha]
            right = np.dot(np.diag(s)[:alpha, :], right)
            # assert np.allclose(left.dot(right), matrix)
            return (left, right, alpha)
        else:
            raise ValueError("split must be a valid option.")

def bond_weight(matrix, k=0, cut=None):
    """
    Splits a matrix in half using the SVD.
    
    Parameters
    ----------
    matrix : The matrix to be split.
    k      : (optional, default zero) The hyperparameter value for the svd
    cut    : (optional, default None) The number of states to keep on the internal 
             index.

    Returns
    -------
    left  : The left side of the split matrix.
    right : The right side of the split matrix.
    bw    : The bond weight matrix.

    """
    if (cut is not None):
        left, s, right = np.linalg.svd(matrix, full_matrices=False)
        # left, s, right = randomized_svd(matrix, n_components=cut) 
        alpha = min([len(s[s > 1e-14]), cut])
        stil = np.sqrt(s**(1-k))
        bw = np.diag((s[:alpha])**k)
        left = np.dot(left, np.diag(stil)[:, :alpha])
        right = np.dot(np.diag(stil)[:alpha, :], right)
        return (left, right, bw)
    else:
        left, s, right = np.linalg.svd(matrix, full_matrices=False)
        alpha = len(s[s > 1e-14])
        stil = np.sqrt(s**(1-k))
        bw = np.diag((s[:alpha])**k)
        left = np.dot(left, np.diag(stil)[:, :alpha])
        right = np.dot(np.diag(stil)[:alpha, :], right)
        return (left, right, bw)


# def coarse_grain(tensor, nx, ny, nz, nt, dbond, triads=None):
#     """The main coarse graining function."""
#     if triads is not None:
#         assert tensor is None
#         fourD_net = Four_Dimensional_Triad_Network(dbond, triads=triads)
#     else:
#         fourD_net = Four_Dimensional_Triad_Network(dbond)
#         fourD_net.get_triads(tensor)
        
#     for x in range(nx):
#         fourD_net.update_triads()
#         fourD_net.normalize()

#     threeD_net = Three_Dimensional_Triad_Network(dbond, triads=fourD_net.make_3d_triads())
#     del(fourD_net)
#     for y in range(ny):
#         threeD_net.update_triads()
#         threeD_net.normalize()

#     twoD_net = Two_Dimensional_Triad_Network(dbond, triads=make_2d_triads())
#     del(threeD_net)
#     for z in range(nz-1):
#         twoD_net.update_triads()
#         twoD_net.normalize()

#     TM = twoD_net.trace_update()
#     del(twoD_net)
#     for t in range(nt):
#         TM = TM.dot(TM)
#         norm = np.linalg.norm(TM)
#         TM /= norm
        
class Four_Dimensional_Triad_Network:
    """
    A 4D RG instance.
    """
    def __init__(self, dbond, triads=None, normlist=None):
        self.dbond = dbond
        if normlist is not None:
            self.lognorms = normlist
        else:
            self.lognorms = list()
        if triads is not None:
            assert (len(triads) == 6)
            self.A = triads[0]
            self.B = triads[1]
            self.C = triads[2]
            self.D = triads[3]
            self.E = triads[4]
            self.F = triads[5]

    def coarse_grain(self, normalize=True):
        """The main coarse graining function."""
        # print("Coarsening fourth dimension...")
        dirs = ['x', 'y', 'z', 't']
        for d in dirs:
            print("doing " + d)
            self.update_triads()
            if normalize:
                self.normalize()
        print("Done.")

    def norm(self,):
        """ computes the norm of the full tensor from the triads."""
        As = self.A.shape
        bs = self.B.shape
        cs = self.C.shape
        ds = self.D.shape
        es = self.E.shape
        fs = self.F.shape

        leftside = np.tensordot(self.A, self.A.conjugate(),
                                axes=([0,1], [0,1])).reshape((As[2]**2))
        rightside = np.tensordot(self.B, self.B.conjugate(),
                                 axes=([1], [1])).transpose((0,2,1,3)).reshape((bs[0]**2, bs[2]**2))
        leftside = np.dot(leftside, rightside)
        rightside = np.tensordot(self.C, self.C.conjugate(),
                                 axes=([1], [1])).transpose((0,2,1,3)).reshape((cs[0]**2, cs[2]**2))
        leftside = np.dot(leftside, rightside)
        rightside = np.tensordot(self.D, self.D.conjugate(),
                                 axes=([1], [1])).transpose((0,2,1,3)).reshape((ds[0]**2, ds[2]**2))
        leftside = np.dot(leftside, rightside)
        rightside = np.tensordot(self.E, self.E.conjugate(),
                                 axes=([1], [1])).transpose((0,2,1,3)).reshape((es[0]**2, es[2]**2))
        leftside = np.dot(leftside, rightside)
        rightside = np.tensordot(self.F, self.F.conjugate(),
                                 axes=([1,2], [1,2])).reshape((fs[0]**2))
        norm = np.dot(leftside, rightside)
        return np.sqrt(norm)
    


    def normalize(self,):
        """normalize each tensor, and return the total."""
        normA = np.linalg.norm(self.A)
        normB = np.linalg.norm(self.B)
        normC = np.linalg.norm(self.C)
        normD = np.linalg.norm(self.D)
        normE = np.linalg.norm(self.E)
        normF = np.linalg.norm(self.F)

        norm_list = [normA, normB, normC, normD, normE, normF]

        self.A /= normA
        self.B /= normB
        self.C /= normC
        self.D /= normD
        self.E /= normE
        self.F /= normF

        self.lognorms.append(np.sum(np.log(norm_list)))
        # norm = self.norm()
        # smallnorm = norm**(1./6)
        # self.A /= smallnorm
        # self.B /= smallnorm
        # self.C /= smallnorm
        # self.D /= smallnorm
        # self.E /= smallnorm
        # self.F /= smallnorm
        # self.lognorms.append(np.log(norm))

    def reconstruct(self,):
        one = np.einsum('ija, akl', self.A, self.B)
        two = np.einsum('ijka, alm', one, self.C)
        one = np.einsum('ijkla, amn', two, self.D)
        two = np.einsum('ijklma, ano', one, self.E)
        one = np.einsum('ijklmna, aop', two, self.F)
        return one
        
    def get_triads(self, tensor):
        """
        Break down a tensor into the triads.
        A is (left, front, right)
        B is (left, away, right)
        C is (left, towards, right)
        D is (left, up, right)
        E is (left, down, right)
        F is (left, back, right)
        
        """

        ts = tensor.shape
        rest = tensor.reshape((ts[0]*ts[1], ts[2]*ts[3]*ts[4]*ts[5]*ts[6]*ts[7]))
        self.A, rest, alpha = split(rest, cut=self.dbond)
        self.A = self.A.reshape((ts[0], ts[1], alpha))

        rest = rest.reshape((alpha*ts[2], ts[3]*ts[4]*ts[5]*ts[6]*ts[7]))
        self.B, rest, beta = split(rest, cut=self.dbond)
        self.B = self.B.reshape((alpha, ts[2], beta))

        rest = rest.reshape((beta*ts[3], ts[4]*ts[5]*ts[6]*ts[7]))
        self.C, rest, alpha = split(rest, self.dbond)
        self.C = self.C.reshape((beta, ts[3], alpha))

        rest = rest.reshape((alpha*ts[4], ts[5]*ts[6]*ts[7]))
        self.D, rest, beta = split(rest, self.dbond)
        self.D = self.D.reshape((alpha, ts[4], beta))

        rest = rest.reshape((beta*ts[5], ts[6]*ts[7]))
        self.E, self.F, alpha = split(rest, cut=self.dbond)
        self.E = self.E.reshape((beta, ts[5], alpha))

        self.F = self.F.reshape((alpha, ts[6], ts[7]))

    def getq(self,A,B,C,D,E,F):
        As = A.shape       # get everyone's shapes
        bs = B.shape
        cs = C.shape
        ds = D.shape
        
        # s1 = np.einsum('iak, jal', A, A.conjugate()).reshape((As[0]**2, As[2]**2))
        s1 = np.tensordot(A, A.conjugate(), axes=([1], [1])).transpose((0,2,1,3))
        s1 = s1.reshape((As[0]**2, As[2]**2))
        # print(np.allclose(s1, a1))
        # s2 = np.einsum('iak, jal', B, B.conjugate()).reshape((bs[0]**2, bs[2]**2))
        s2 = np.tensordot(B, B.conjugate(), axes=([1], [1])).transpose((0,2,1,3))
        s2 = s2.reshape((bs[0]**2, bs[2]**2))
        # print(np.allclose(s2, a2))
        AB = np.dot(s1, s2)
        # s2 = np.einsum('iak, jal', C, C.conjugate()).reshape((cs[0]**2, cs[2]**2))
        s2 = np.tensordot(C, C.conjugate(), axes=([1], [1])).transpose((0,2,1,3))
        s2 = s2.reshape((cs[0]**2, cs[2]**2))
        # print(np.allclose(s2, a2))
        ABC = np.dot(AB, s2)     # A B C
        r1 = np.tensordot(F, F.conjugate(), axes=([1,2], [1,2]))
        r2 = np.tensordot(E, r1, axes=([2], [0]))
        # r2 = np.einsum('ika, jla', r2, E.conjugate())
        r2 = np.tensordot(r2, E.conjugate(), axes=([1,2], [1,2]))
        r2 = np.tensordot(D, r2, axes=([2], [0]))
        r2 = np.tensordot(r2, D.conjugate(), axes=([2],[2])).transpose((0,2,1,3))
        r3 = np.einsum('ijaa', r2)
        DD = r2.reshape((ds[0]**2, ds[1]**2))
        r4 = np.tensordot(C, r3, axes=([2],[0]))
        r4 = np.tensordot(r4, C.conjugate(), axes=([2],[2])).transpose((0,2,1,3))
        CC = r4.reshape((cs[0]**2, cs[1]**2))
        Q = AB.dot(CC)
        Q = Q.dot(DD.transpose())
        Q = Q.dot(ABC.transpose()).reshape((As[0], As[0], As[0], As[0]))
        Q = Q.transpose((0,2,1,3)).reshape((As[0]**2, As[0]**2))
        assert np.allclose(Q, Q.conjugate().transpose())
        return Q

        
        # return S1

    # def gets2(self, D):
    #     ds = D.shape
    #     # DD = np.einsum('iak, jal', D, D.conjugate()).reshape((ds[0]**2, ds[2]**2))
    #     DD = np.tensordot(D, D.conjugate(), axes=([1], [1])).transpose((0,2,1,3))
    #     DD = DD.reshape((ds[0]**2, ds[2]**2))
    #     return DD

    # def getr23(self, D, E, F):
    #     ds = D.shape
    #     es = E.shape
    #     fs = F.shape

    #     # r1 = np.einsum('iab, jab', F, F.conjugate())
    #     r1 = np.tensordot(F, F.conjugate(), axes=([1,2], [1,2]))
    #     # r2 = np.einsum('ija, ak', E, r1)
    #     r2 = np.tensordot(E, r1, axes=([2], [0]))
    #     # r2 = np.einsum('ika, jla', r2, E.conjugate())
    #     r2 = np.tensordot(r2, E.conjugate(), axes=([2], [2])).transpose((0,2,1,3))

    #     mid = np.einsum('ijaa', r2)
    #     # left = np.einsum('ija, ak', D, mid)
    #     left = np.tensordot(D, mid, axes=([2], [0]))
    #     # r3 = np.einsum('ika, jla', left, D.conjugate()).reshape((ds[0]**2, ds[1]**2))
    #     r3 = np.tensordot(left, D.conjugate(), axes=([2], [2])).transpose((0,2,1,3))
    #     r3 = r3.reshape((ds[0]**2, ds[1]**2))
    #     r2 = r2.reshape((es[0]**2, es[1]**2))
    #     return (r2, r3)

            
    # def getq(self, A, B, C, D, E, F):
    #     """make the q matrix."""
    #     S1 = self.gets1(A, B, C)
    #     ss = S1.shape
    #     xx = int(np.rint(np.sqrt(ss[0])))
    #     S2 = self.gets2(D)
    #     R2, R3 = self.getr23(D, E, F)

    #     Q = np.dot(S1, S2)
    #     Q = np.dot(Q, R2)
    #     Q = np.dot(Q, R3.transpose())
    #     Q = np.dot(Q, S1.transpose())
    #     Q = Q.reshape((xx,xx,xx,xx)).transpose((0,2,1,3))
    #     Q = Q.reshape((xx**2, xx**2))
    #     # assert np.allclose(Q, Q.conjugate().transpose())

    #     return Q

    def getwq(self, A, B, C,D,E,F):
        As = A.shape       # get everyone's shapes
        bs = B.shape
        cs = C.shape
        ds = D.shape
        es = E.shape
        fs = F.shape
        # s1 = np.einsum('bai, baj', A, A.conjugate())
        s1 = np.tensordot(A, A.conjugate(), axes=([0,1], [0,1]))
        # s2 = np.einsum('ajk, ai', B, s1)
        s2 = np.tensordot(B, s1, axes=([0], [0])).transpose((2,0,1))
        # s2 = np.einsum('aik, ajl', s2, B.conjugate()).reshape((bs[1]**2, bs[2]**2))
        s2 = np.tensordot(s2, B.conjugate(), axes=([0], [0])).transpose((0,2,1,3))
        s2 = s2.reshape((bs[1]**2, bs[2]**2))

        # s1 = np.einsum('iak, jal', C, C.conjugate()).reshape((cs[0]**2, cs[2]**2))
        s1 = np.tensordot(C, C.conjugate(), axes=([1], [1])).transpose((0,2,1,3))
        s1 = s1.reshape((cs[0]**2, cs[2]**2))
        r1 = np.tensordot(F, F.conjugate(), axes=([1,2], [1,2]))
        r2 = np.tensordot(E, r1, axes=([2], [0]))
        # r2 = np.einsum('ika, jla', r2, E.conjugate())
        r2 = np.tensordot(r2, E.conjugate(), axes=([1,2], [1,2]))
        r2 = np.tensordot(D, r2, axes=([2], [0]))
        r2 = np.tensordot(r2, D.conjugate(), axes=([2],[2])).transpose((0,2,1,3))
        r3 = np.einsum('ijaa', r2)
        DD = r2.reshape((ds[0]**2, ds[1]**2))
        r4 = np.tensordot(C, r3, axes=([2],[0]))
        r4 = np.tensordot(r4, C.conjugate(), axes=([2],[2])).transpose((0,2,1,3))
        CC = r4.reshape((cs[0]**2, cs[1]**2))
        Q = s2.dot(CC)
        Q = Q.dot(DD.transpose())
        Q = Q.dot(s1.transpose())
        Q = Q.dot(s2.transpose())
        Q = Q.reshape((bs[1], bs[1], bs[1], bs[1])).transpose((0,2,1,3))
        Q = Q.reshape((bs[1]**2, bs[1]**2))
        assert np.allclose(Q, Q.conjugate().transpose())
        return Q


        # S1 = np.dot(s2, s1)     # A B C
        # return S1

    # def getws2(self, D):
    #     ds = D.shape
    #     DD = np.einsum('iak, jal', D, D.conjugate()).reshape((ds[0]**2, ds[2]**2))
    #     return DD

    # def getwr23(self, D, E, F):
    #     ds = D.shape
    #     es = E.shape
    #     fs = F.shape

    #     r1 = np.einsum('iab, jab', F, F.conjugate())
    #     r2 = np.einsum('ija, ak', E, r1)
    #     r2 = np.einsum('ika, jla', r2, E.conjugate())

    #     mid = np.einsum('ijaa', r2)
    #     left = np.einsum('ija, ak', D, mid)
    #     r3 = np.einsum('ika, jla', left, D.conjugate()).reshape((ds[0]**2, ds[1]**2))
    #     r2 = ee.reshape((es[0]**2, es[1]**2))
    #     return (r2, r3)


    # def getwq(self, A, B, C, D, E, F):
    #     """make the special q for w update."""
    #     S1 = self.getws1(A, B, C)
    #     ss = S1.shape
    #     xx = int(np.rint(np.sqrt(ss[0])))
    #     S2 = self.gets2(D)
    #     R2, R3 = self.getr23(D, E, F)

    #     Q = np.dot(S1, S2)
    #     Q = np.dot(Q, R2)
    #     Q = np.dot(Q, R3.transpose())
    #     Q = np.dot(Q, S1.transpose())
    #     Q = Q.reshape((xx,xx,xx,xx)).transpose((0,2,1,3))
    #     Q = Q.reshape((xx**2, xx**2))
    #     # assert np.allclose(Q, Q.conjugate().transpose())
    #     return Q


    # def updateBC(self, w, left):
    #     """
    #     Here B and C are (left, middle, right)
    #     W is (top, bot, free)
    #     left is (free, top, bot)
    #     right is (free, top, bot)
    #     """
    #     bs = self.B.shape
    #     cs = self.C.shape
    #     ws = w.shape
    #     ws = (int(np.rint(np.sqrt(ws[0]))), int(np.rint(np.sqrt(ws[0]))), ws[1])

    #     ls = left.shape
    #     ls = (ls[0], int(np.rint(np.sqrt(ls[1]))), int(np.rint(np.sqrt(ls[1]))))
    #     # print(ls)
    #     # rs = right.shape

    #     # top = np.einsum('iaj, akl', self.B, w.reshape(ws))
    #     top = np.tensordot(self.B, w.reshape(ws), axes=([1], [0]))
    #     # Big = np.einsum('ijak, lam', top, self.B)
    #     Big = np.tensordot(top, self.B, axes=([2], [1]))
    #     Big = Big.transpose((0,3,2,1,4))
    #     # Big = np.einsum('iab, abjkl', left.reshape(ls), Big).reshape((ls[0]*ws[2], bs[2]**2))
    #     Big = np.tensordot(left.reshape(ls), Big, axes=([1,2], [0,1]))
    #     Big = Big.reshape((ls[0]*ws[2], bs[2]**2))
    #     self.B, mid, alpha = split(Big, cut=self.dbond)
    #     self.B = self.B.reshape((ls[0], ws[2], alpha))
    #     mid = mid.reshape((alpha, bs[2], bs[2]))

    #     # top = np.einsum('iaj, akl', self.C, w.reshape(ws))
    #     top = np.tensordot(self.C, w.reshape(ws), axes=([1], [0]))
    #     # Big = np.einsum('ijak, lam', top, self.C)
    #     Big = np.tensordot(top, self.C, axes=([2], [1]))
    #     Big = Big.transpose((0,3,2,1,4))
    #     # Big = np.einsum('iab, abjkl', mid, Big).reshape((alpha*ws[2], cs[2]**2))
    #     Big = np.tensordot(mid, Big, axes=([1,2], [0,1]))
    #     Big = Big.reshape((alpha*ws[2], cs[2]**2))
    #     self.C, mid, beta = split(Big, cut=self.dbond)
    #     self.C = self.C.reshape((alpha, ws[2], beta))

    #     return mid


    # def updateDE(self, left):
    #     """
    #     Here B and C are (left, middle, right)
    #     W is (top, bot, free)
    #     left is (free, top, bot)
    #     """

    #     ls = left.shape
    #     ds = self.D.shape
    #     es = self.E.shape

    #     # ED = np.einsum('iaj, kal', self.E, self.D).transpose((0,2,1,3))
    #     ED = np.tensordot(self.E, self.D, axes=([1], [1])).transpose((0,2,1,3))
    #     ED = ED.reshape((es[0]*ds[0], es[2]*ds[2]))
    #     ED_left, ED_right, gamma = split(ED, cut=self.dbond)
    #     # Big = np.einsum('ija, akl', self.D, ED_left.reshape((es[0], ds[0], gamma)))
    #     Big = np.tensordot(self.D, ED_left.reshape((es[0], ds[0], gamma)),
    #                        axes=([2], [0]))
    #     Big = Big.transpose((0,2,1,3)).reshape((ds[0]*ds[0], ds[1], gamma))
    #     # self.D = np.einsum('ia, ajk', left, Big).reshape((ls[0], ds[1], gamma))
    #     self.D = np.tensordot(left, Big, axes=([1], [0])).reshape((ls[0], ds[1], gamma))

    #     # Big = np.einsum('ija, akl', ED_right.reshape((gamma, es[2], ds[2])), self.E)
    #     Big = np.tensordot(ED_right.reshape((gamma, es[2], ds[2])), self.E,
    #                        axes=([2], [0]))
    #     Big = Big.transpose((0,2,1,3)).reshape((gamma*es[1], es[2]**2))
    #     self.E, mid, delta = split(Big, cut=self.dbond)
    #     self.E = self.E.reshape((gamma, es[1], delta))

    #     return mid


    def updateE(self, W):
        es = self.E.shape
        ws = W.shape
        ws = (int(np.rint(np.sqrt(ws[0]))), int(np.rint(np.sqrt(ws[0]))), ws[1])

        top = np.tensordot(self.E, W.reshape(ws), axes=([1],[0]))
        bigE = np.tensordot(top, self.E, axes=([2],[1])).transpose((0,3,2,1,4))
        return bigE

    def updateB(self, W):
        bs = self.B.shape
        ws = W.shape
        ws = (int(np.rint(np.sqrt(ws[0]))), int(np.rint(np.sqrt(ws[0]))), ws[1])
        top = np.tensordot(self.B, W.reshape(ws), axes=([1],[0]))
        bigB = np.tensordot(top, self.B, axes=([2],[1])).transpose((0,3,2,1,4))
        return bigB
        
    
    # def updateA(self, U, V):
    #     As = self.A.shape
    #     us = U.shape
    #     us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
    #     vs = V.shape
    #     vs = (int(np.rint(np.sqrt(vs[0]))), int(np.rint(np.sqrt(vs[0]))), vs[1])
    #     # cs = cap.shape
        
    #     # one = np.einsum('iaj, kal', V.reshape(vs), self.A)
    #     one = np.tensordot(V.reshape(vs), self.A, axes=([1], [1]))
    #     # two = np.einsum('aij, akl', U.reshape(us), self.A)
    #     two = np.tensordot(U.reshape(us), self.A, axes=([0], [0]))
    #     # self.A = np.einsum('piqk, qjpl', two, one).reshape((us[2]*vs[2], As[2]**2))
    #     Big = np.tensordot(two, one, axes=([0, 2], [2, 0])).transpose((0, 2, 1, 3))
    #     Big = Big.reshape((us[2]*vs[2], As[2]**2))
    #     self.A, mid, alpha = split(Big, cut=self.dbond)
    #     self.A = self.A.reshape((us[2], vs[2], alpha))

    #     return mid


    def make_mid(self,):
        mid = np.tensordot(self.C, self.D, axes=([1],[1])).transpose((0,2,1,3))
        return mid


    def updateEF(self, U, V, W):
        us = U.shape
        us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
        vs = V.shape
        vs = (int(np.rint(np.sqrt(vs[0]))), int(np.rint(np.sqrt(vs[0]))), vs[1])
        ws = W.shape
        ws = (int(np.rint(np.sqrt(ws[0]))), int(np.rint(np.sqrt(ws[0]))), ws[1])
        # cs = cap.shape
        fs = self.F.shape
        ds = self.D.shape
        cs = self.C.shape
        es = self.E.shape

        # one = np.einsum('iqj, kql', self.D, V.reshape(vs))
        one = np.tensordot(self.F, V.reshape(vs), axes=([1], [1]))
        # two = np.einsum('ijp, pkl', self.D, U.reshape(us))
        two = np.tensordot(self.F, U.reshape(us), axes=([2], [0]))
        # self.D = np.einsum('iqpl, jpqk', two, one).reshape((ds[0]**2, vs[2]*us[2]))
        Big = np.tensordot(two, one, axes=([1,2], [2,1])).transpose((0,2,3,1))
        Big = np.tensordot(self.updateE(W), Big, axes=([3,4],[0,1]))
        # Big = Big.reshape((fs[0]**2, vs[2]*us[2]))
        # self.F = np.dot(cap, Big).reshape((cs[0], vs[2], us[2]))
        Big = np.tensordot(self.D, Big, axes=([2],[0])).transpose((0,2,1,3,4,5))
        Big = np.tensordot(self.make_mid(), Big, axes=([2,3],[0,1])) # D^8
        Big = Big.transpose((0,1,3,4,5,2)).reshape((cs[0]*ds[0]*ws[2]*vs[2], us[2]*ds[1]))
        Big, self.F, alpha = split(Big, cut=self.dbond) # D^8
        self.F = self.F.reshape((alpha, us[2], ds[1]))
        Big = Big.reshape((cs[0]*ds[0]*ws[2], vs[2]*alpha))
        Big, self.E, beta = split(Big, cut=self.dbond)
        self.E = self.E.reshape((beta, vs[2], alpha))
        Big = Big.reshape((cs[0], ds[0], ws[2], beta))
        return Big
        # Big, self.D, gamma = split(Big, cut=self.dbond)
        # self.D = self.D.reshape((gamma, ws[2], beta))
        # return Big.reshape((cs[0], ds[0], gamma))

    def updateABCD(self, cap, U, V, W):
        us = U.shape
        us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
        vs = V.shape
        vs = (int(np.rint(np.sqrt(vs[0]))), int(np.rint(np.sqrt(vs[0]))), vs[1])
        cs = cap.shape
        ws = W.shape
        ws = (int(np.rint(np.sqrt(ws[0]))), int(np.rint(np.sqrt(ws[0]))), ws[1])
        As = self.A.shape
        bs = self.B.shape
        cs = self.C.shape
        # ds = self.D.shape
        ms = cap.shape
        # one = np.einsum('iaj, kal', V.reshape(vs), self.A)
        one = np.tensordot(V.reshape(vs), self.A, axes=([1], [1]))
        # two = np.einsum('aij, akl', U.reshape(us), self.A)
        two = np.tensordot(U.reshape(us), self.A, axes=([0], [0]))
        # self.A = np.einsum('piqk, qjpl', two, one).reshape((us[2]*vs[2], As[2]**2))
        Big = np.tensordot(two, one, axes=([0, 2], [2, 0])).transpose((0, 2, 1, 3))
        Big = np.tensordot(Big, self.updateB(W), axes=([2,3],[0,1]))
        Big = np.tensordot(Big, self.C, axes=([4],[0])).transpose((0,1,2,4,3,5))
        Big = np.tensordot(Big, cap, axes=([4,5],[0,1])).transpose((3,0,1,2,4,5)) # D^8
        Big = Big.reshape((cs[1]*us[2], vs[2]*ws[2]*ms[2]*ms[3]))
        self.A, Big, alpha = split(Big, cut=self.dbond)
        self.A = self.A.reshape((cs[1], us[2], alpha))
        Big = Big.reshape((alpha*vs[2], ws[2]*ms[2]*ms[3]))
        self.B, Big, beta = split(Big, cut=self.dbond)
        self.B = self.B.reshape((alpha, vs[2], beta))
        Big = Big.reshape((beta*ws[2], ms[2]*ms[3]))
        self.C, self.D, gamma = split(Big, cut=self.dbond)
        self.C = self.C.reshape((beta, ws[2], gamma))
        self.D = self.D.reshape((gamma, ms[2], ms[3]))
        
        

        
    def update_triads(self, different_updates=True):
        # U part
        q = self.getq(self.A, self.B, self.C,
                      self.D, self.E, self.F)
        if (self.A.shape[0]**2 < self.dbond):
            U = getU(q, self.A.shape[0]**2)
            # U = np.eye(U.shape[0])
        else:
            U = getU(q, self.dbond)
        if different_updates:
            # V part
            q = self.getq(self.A.transpose((1,0,2)), self.B, self.C,
                          self.D, self.E, self.F.transpose((0,2,1)))
            if (self.A.shape[1]**2 < self.dbond):
                V = getU(q, self.A.shape[1]**2)
                # V = np.eye(V.shape[0])
            else:
                V = getU(q, self.dbond)
            # W part
            q = self.getwq(self.A, self.B, self.C,
                           self.D, self.E, self.F)
            if (self.B.shape[0]**2 < self.dbond):
                W = getU(q, self.B.shape[0]**2)
                # W = np.eye(W.shape[0])
            else:
                W = getU(q, self.dbond)
        else:
            W = V = U
        mid = self.updateEF(U, V, W)
        self.updateABCD(mid, U, V, W)
        print(self.A.shape, self.B.shape, self.C.shape, self.D.shape, self.E.shape, self.F.shape)
        # mid = self.updateA(U, V)
        # mid = self.updateBC(W, mid)
        # mid = self.updateDE(mid)
        # self.updateF(mid, U, V)

        
    def make_3d_triads(self,):
        """ trace over D and E and give it to C.  Now
            A, B, C and F are the four tensors for 3D."""
        mid = np.einsum('iab, baj', self.D, self.E)
        self.C = np.einsum('ija, ak', self.C, mid)
        return [self.A, self.B, self.C, self.F]

    def tensor_trace(self,):
        mid = np.tensordot(self.C, self.D, axes=([1,2], [1,0]))
        mid = np.tensordot(self.B, mid, axes=([2], [0]))
        mid = np.tensordot(mid, self.E, axes=([1,2], [1,0]))
        mid = np.tensordot(self.A, mid, axes=([2], [0]))
        mid = np.tensordot(mid, self.F, axes=([0,1,2], [2,1,0]))
        return mid
        #self.lognorms.append(np.log(mid))

    def get_lognorms(self,):
        lognorms = self.lognorms[:]
        lognorms.append(np.log(self.tensor_trace()))
        return lognorms

        
    
    

class ThreeDimensionalTriadNetwork:
    """
    A 3D RG instance.
    """
    def __init__(self, dbond, triads=None, normlist=None, time_first=True):
        """ dbond is the maximum size of internal indices. """
        self.dbond = dbond
        # self.sdbond = 2*self.dbond
        # self.tdbond = 3*self.dbond
        self.imp = False
        self.nnimp = False
        self.time_first = time_first
        self.Xlist = list()
        if normlist is not None:
            self.lognorms = normlist
        else:
            self.lognorms = list()
        if triads is not None:
            self.A = triads[0]
            self.B = triads[1]
            self.C = triads[2]
            self.D = triads[3]
            # bond weight order is [z, y, x]
            # it rolls to [y, x, z]
            self.bond_weights = [np.eye(self.B.shape[1])]*3
        print("Bond dimension =", self.dbond)

    def coarse_grain(self, normalize=True, all_vols=False, hyp_k=0):
        """The main coarse graining function."""
        # print("Coarsening third dimension...")
        self.hyp_k = hyp_k
        dirs = ['x', 'y', 'z']
        for d in dirs:
            print("doing " + d)
            X = self.makeX1()
            self.Xlist.append(X)
            print("X1 = ", X)
            self.update_triads()
            if normalize:
                self.normalize()
        if all_vols:
            if self.imp:
                return (self.get_lognorms(), self.get_imp_ratio())
            else:
                return self.get_lognorms()
        # print("Done.")


    def normalize(self,):
        """normalize each tensor, and return the total."""
        normA = np.linalg.norm(self.A)
        normB = np.linalg.norm(self.B)
        normC = np.linalg.norm(self.C)
        normD = np.linalg.norm(self.D)

        self.A /= normA
        self.B /= normB
        self.C /= normC
        self.D /= normD
        print("normalized the pure triads.")
        if self.nnimp:
            self.Aimp1 /= normA
            self.Bimp1 /= normB
            self.Cimp1 /= normC
            self.Dimp1 /= normD
            self.Aimp2 /= normA
            self.Bimp2 /= normB
            self.Cimp2 /= normC
            self.Dimp2 /= normD
            print("normalized the nn impure triads.")
        if self.imp:
            self.Aimp /= normA
            self.Bimp /= normB
            self.Cimp /= normC
            self.Dimp /= normD
            print("normalized the impure triads.")
        self.lognorms.append(np.log(normA*normB*normC*normD))
            
            
        # norm = self.norm()
        # print("norm = ", norm)
        # self.A /= norm**0.25
        # self.B /= norm**0.25
        # self.C /= norm**0.25
        # self.D /= norm**0.25
        # print("normalized the pure triads.")
        # if self.nnimp:
        #     self.Aimp1 /= norm**0.25
        #     self.Bimp1 /= norm**0.25
        #     self.Cimp1 /= norm**0.25
        #     self.Dimp1 /= norm**0.25
        #     self.Aimp2 /= norm**0.25
        #     self.Bimp2 /= norm**0.25
        #     self.Cimp2 /= norm**0.25
        #     self.Dimp2 /= norm**0.25
        #     print("normalized the nn impure triads.")
        # if self.imp:
        #     self.Aimp /= norm**0.25
        #     self.Bimp /= norm**0.25
        #     self.Cimp /= norm**0.25
        #     self.Dimp /= norm**0.25
        #     print("normalized the impure triads.")
        # self.lognorms.append(np.log(norm))

    def reconstruct(self,):
        """
        naive reconstruction of T tensor.
        """
        one = np.einsum('ija, akl', self.A, self.B)
        two = np.einsum('ijka, alm', one, self.C)
        one = np.einsum('ijkla, amn', two, self.D)
        return one

    def norm(self,):
        """ Compute the norm of the tensor. """
        As = self.A.shape
        bs = self.B.shape
        cs = self.C.shape
        ds = self.D.shape

        left_cap = np.dot(self.A.reshape((As[0]*As[1], As[2])).transpose().conjugate(),
                          self.A.reshape((As[0]*As[1], As[2])))
        left_cap = left_cap.reshape((As[2]**2))
        left_mid = np.dot(self.B.transpose((0, 2, 1)).reshape((bs[0]*bs[2], bs[1])).conjugate(),
                          self.B.transpose((1, 0, 2)).reshape((bs[1], bs[0]*bs[2])))
        left_mid = left_mid.reshape((bs[0], bs[2], bs[0], bs[2])).transpose((0, 2, 1, 3))
        left_mid = left_mid.reshape((bs[0]**2, bs[2]**2))
        right_mid = np.dot(self.C.transpose((0, 2, 1)).reshape((cs[0]*cs[2], cs[1])).conjugate(),
                           self.C.transpose((1, 0, 2)).reshape((cs[1], cs[0]*cs[2])))
        right_mid = right_mid.reshape((cs[0], cs[2], cs[0], cs[2])).transpose((0, 2, 1, 3))
        right_mid = right_mid.reshape((cs[0]**2, cs[2]**2))

        left_mid = np.dot(left_mid, right_mid)

        right_cap = np.dot(self.D.reshape((ds[0], ds[1]*ds[2])).conjugate(),
                           self.D.reshape((ds[0], ds[1]*ds[2])).transpose())
        right_cap = right_cap.reshape((ds[0]**2))

        left_mid = np.dot(left_cap, left_mid)
        left_mid = np.dot(left_mid, right_cap)
        return np.sqrt(left_mid)
        
        
    def get_triads(self, tensor):
        """ 
        Break down a tensor into the 3D triads. 
        A is (left, front, right)
        B is (left, down, right)
        C is (left, up, right)
        D is (left, back, right)
        
        """
        if not self.time_first:
            ts = tensor.shape
            ss = (ts[2], ts[0], ts[1], ts[4], ts[5], ts[3])
            rest = np.transpose(tensor, ((2,0,1,4,5,3))).reshape((ss[0]*ss[1], ss[2]*ss[3]*ss[4]*ss[5]))
            self.A, rest, alpha = split(rest, cut=self.dbond)
            self.A = self.A.reshape(ss[0], ss[1], alpha)
            
            rest = rest.reshape((alpha*ss[2], ss[3]*ss[4]*ss[5]))
            self.B, rest, beta = split(rest, cut=self.dbond)
            self.B = self.B.reshape((alpha, ss[2], beta))
            
            rest = rest.reshape((beta*ss[3], ss[4]*ss[5]))
            self.C, self.D, gamma = split(rest, cut=self.dbond)
            self.C = self.C.reshape((beta, ss[3], gamma))
            
            self.D = self.D.reshape((gamma, ss[4], ss[5]))
        else:
            ss = tensor.shape
            rest = tensor.reshape((ss[0]*ss[1], ss[2]*ss[3]*ss[4]*ss[5]))
            self.A, rest, alpha = split(rest, cut=self.dbond)
            self.A = self.A.reshape(ss[0], ss[1], alpha)
            
            rest = rest.reshape((alpha*ss[2], ss[3]*ss[4]*ss[5]))
            self.B, rest, beta = split(rest, cut=self.dbond)
            self.B = self.B.reshape((alpha, ss[2], beta))
            
            rest = rest.reshape((beta*ss[3], ss[4]*ss[5]))
            self.C, self.D, gamma = split(rest, cut=self.dbond)
            self.C = self.C.reshape((beta, ss[3], gamma))
            
            self.D = self.D.reshape((gamma, ss[4], ss[5]))
            self.bond_weights = [np.eye(self.B.shape[1])]*3


    def load_nn_imp(self, triads1, triads2):
        """
        Load in the two impure tensors for the nearest neighbor
        interaction.

        """
        self.Aimp1, self.Bimp1, self.Cimp1, self.Dimp1 = triads1
        self.Aimp2, self.Bimp2, self.Cimp2, self.Dimp2 = triads2
        self.nnimp = True
            
            
    def get_nn_imp_triads(self, tensor1, tensor2):
        """ 
        Break down two impure tensors into their triads. 
        A is (left, front, right)
        B is (left, down, right)
        C is (left, up, right)
        D is (left, back, right)
        
        """
        
        if self.time_first:
            ss = tensor1.shape
            rest = tensor1.reshape((ss[0]*ss[1], ss[2]*ss[3]*ss[4]*ss[5]))
            self.Aimp1, rest, alpha = split(rest, cut=self.dbond)
            self.Aimp1 = self.Aimp1.reshape(ss[0], ss[1], alpha)
            
            rest = rest.reshape((alpha*ss[2], ss[3]*ss[4]*ss[5]))
            self.Bimp1, rest, beta = split(rest, cut=self.dbond)
            self.Bimp1 = self.Bimp1.reshape((alpha, ss[2], beta))
            
            rest = rest.reshape((beta*ss[3], ss[4]*ss[5]))
            self.Cimp1, self.Dimp1, gamma = split(rest, cut=self.dbond)
            self.Cimp1 = self.Cimp1.reshape((beta, ss[3], gamma))
            
            self.Dimp1 = self.Dimp1.reshape((gamma, ss[4], ss[5]))
            
            ss = tensor2.shape
            rest = tensor2.reshape((ss[0]*ss[1], ss[2]*ss[3]*ss[4]*ss[5]))
            self.Aimp2, rest, alpha = split(rest, cut=self.dbond)
            self.Aimp2 = self.Aimp2.reshape(ss[0], ss[1], alpha)
            
            rest = rest.reshape((alpha*ss[2], ss[3]*ss[4]*ss[5]))
            self.Bimp2, rest, beta = split(rest, cut=self.dbond)
            self.Bimp2 = self.Bimp2.reshape((alpha, ss[2], beta))
            
            rest = rest.reshape((beta*ss[3], ss[4]*ss[5]))
            self.Cimp2, self.Dimp2, gamma = split(rest, cut=self.dbond)
            self.Cimp2 = self.Cimp2.reshape((beta, ss[3], gamma))
            
            self.Dimp2 = self.Dimp2.reshape((gamma, ss[4], ss[5]))
            self.nnimp = True
        else:
            ss = tensor1.shape
            ss = (ss[2],ss[0],ss[1],ss[4],ss[5],ss[3])
            rest = np.transpose(tensor1, ((2,0,1,4,5,3))).reshape((ss[0]*ss[1], ss[2]*ss[3]*ss[4]*ss[5]))
            self.Aimp1, rest, alpha = split(rest, cut=self.dbond)
            self.Aimp1 = self.Aimp1.reshape(ss[0], ss[1], alpha)
            
            rest = rest.reshape((alpha*ss[2], ss[3]*ss[4]*ss[5]))
            self.Bimp1, rest, beta = split(rest, cut=self.dbond)
            self.Bimp1 = self.Bimp1.reshape((alpha, ss[2], beta))
            
            rest = rest.reshape((beta*ss[3], ss[4]*ss[5]))
            self.Cimp1, self.Dimp1, gamma = split(rest, cut=self.dbond)
            self.Cimp1 = self.Cimp1.reshape((beta, ss[3], gamma))
            
            self.Dimp1 = self.Dimp1.reshape((gamma, ss[4], ss[5]))
            
            ss = tensor2.shape
            ss = (ss[2],ss[0],ss[1],ss[4],ss[5],ss[3])
            rest = np.transpose(tensor2, ((2,0,1,4,5,3))).reshape((ss[0]*ss[1], ss[2]*ss[3]*ss[4]*ss[5]))
            self.Aimp2, rest, alpha = split(rest, cut=self.dbond)
            self.Aimp2 = self.Aimp2.reshape(ss[0], ss[1], alpha)
            
            rest = rest.reshape((alpha*ss[2], ss[3]*ss[4]*ss[5]))
            self.Bimp2, rest, beta = split(rest, cut=self.dbond)
            self.Bimp2 = self.Bimp2.reshape((alpha, ss[2], beta))
            
            rest = rest.reshape((beta*ss[3], ss[4]*ss[5]))
            self.Cimp2, self.Dimp2, gamma = split(rest, cut=self.dbond)
            self.Cimp2 = self.Cimp2.reshape((beta, ss[3], gamma))
            
            self.Dimp2 = self.Dimp2.reshape((gamma, ss[4], ss[5]))
            self.nnimp = True

        
        
    def get_impure_triads(self, tensor):
        """ 
        Break down an impure tensor into the 3D triads. 
        A is (left, front, right)
        B is (left, down, right)
        C is (left, up, right)
        D is (left, back, right)
        
        """
        
        if not self.time_first:
            ss = tensor.shape
            ss = (ss[2],ss[0],ss[1],ss[4],ss[5],ss[3])
            rest = np.transpose(tensor, ((2,0,1,4,5,3))).reshape((ss[0]*ss[1], ss[2]*ss[3]*ss[4]*ss[5]))
            self.Aimp, rest, alpha = split(rest, cut=self.dbond)
            self.Aimp = self.Aimp.reshape(ss[0], ss[1], alpha)
            
            rest = rest.reshape((alpha*ss[2], ss[3]*ss[4]*ss[5]))
            self.Bimp, rest, beta = split(rest, cut=self.dbond)
            self.Bimp = self.Bimp.reshape((alpha, ss[2], beta))
            
            rest = rest.reshape((beta*ss[3], ss[4]*ss[5]))
            self.Cimp, self.Dimp, gamma = split(rest, cut=self.dbond)
            self.Cimp = self.Cimp.reshape((beta, ss[3], gamma))
            
            self.Dimp = self.Dimp.reshape((gamma, ss[4], ss[5]))
            self.imp = True
        else:
            ss = tensor.shape
            rest = tensor.reshape((ss[0]*ss[1], ss[2]*ss[3]*ss[4]*ss[5]))
            self.Aimp, rest, alpha = split(rest, cut=self.dbond)
            self.Aimp = self.Aimp.reshape(ss[0], ss[1], alpha)
            
            rest = rest.reshape((alpha*ss[2], ss[3]*ss[4]*ss[5]))
            self.Bimp, rest, beta = split(rest, cut=self.dbond)
            self.Bimp = self.Bimp.reshape((alpha, ss[2], beta))
            
            rest = rest.reshape((beta*ss[3], ss[4]*ss[5]))
            self.Cimp, self.Dimp, gamma = split(rest, cut=self.dbond)
            self.Cimp = self.Cimp.reshape((beta, ss[3], gamma))
            
            self.Dimp = self.Dimp.reshape((gamma, ss[4], ss[5]))
            self.imp = True
        print("made impure triads.")


    def makeX1(self,):
        mid = np.tensordot(self.B, self.C, axes=([1], [1]))
        mid = np.tensordot(mid, mid, axes=([1,2], [2,1]))
        ends = np.tensordot(self.A, self.D, axes=([0,1],[2,1]))
        full = np.tensordot(mid, ends, axes=([0,3],[0,1]))
        full = np.tensordot(full, ends, axes=([0,1], [1,0]))
        
        mid = np.tensordot(self.B, self.C, axes=([1,2], [1,0]))
        other = np.tensordot(self.A, self.D, axes=([0,1], [2,1]))
        trace = np.trace(np.dot(other.transpose(), mid))
        numerator = trace**2

        return numerator / full


    def make_2d_triads(self,):
        """ build the A and B tensors for the 2d triad run."""
        As = self.A.shape
        bs = self.B.shape
        cs = self.C.shape
        ds = self.D.shape
        mid = np.dot(self.B.reshape((bs[0], bs[1]*bs[2])),
                     self.C.transpose((1, 0, 2)).reshape((cs[1]*cs[0], cs[2])))
        left, right, alpha = split(mid, cut=self.dbond)
        A2d = np.dot(self.A.reshape((As[0]*As[1], As[2])), left).reshape((As[0], As[1], alpha))
        B2d = np.dot(right, self.D.reshape((ds[0], ds[1]*ds[2]))).reshape((alpha, ds[1], ds[2]))
        return [A2d, B2d]
        

    def make_T(self,):
        """ Makes the 2d tensor from the triads. """
        As = self.A.shape
        bs = self.B.shape
        cs = self.C.shape
        ds = self.D.shape

        mid = np.dot(self.B.reshape((bs[0], bs[1]*bs[2])),
                     self.C.transpose((1, 0, 2)).reshape((cs[1]*cs[0], cs[2])))
        left = np.dot(self.A.reshape((As[0]*As[1], As[2])), mid)
        left = np.dot(left, self.D.reshape((ds[0], ds[1]*ds[2])))
        return left.reshape((As[0], As[1], ds[1], ds[2])).transpose((0, 3, 2, 1))


    def getS(self, AorB):
        """ A or B is structured (left, front, alpha) """
        As = AorB.shape
        # one = AorB.transpose((1, 0, 2)).reshape((As[1], As[0]*As[2]))
        # want = np.dot(one.transpose(), one)
        # want = want.reshape((As[0], As[2], As[0], As[2]))
        # want = want.transpose((0, 2, 1, 3)).reshape((As[0]**2, As[2]**2))
        # want = np.einsum('iak, jal', AorB, AorB).reshape((As[0]**2, As[2]**2))
        want = np.tensordot(AorB, AorB.conjugate(), axes=([1], [1])).transpose((0,2,1,3))
        # want.shape = (As[0]**2, As[2]**2)
        want = want.reshape((As[0]**2, As[2]**2))
        return want

    def getR23(self, C, D, B):
        """
        C is structured (beta top gamma)
        D is structured (gamma away right)
        """
        cs = C.shape
        # ds = D.shape
        # one = np.einsum('iab, jab', D, D)
        one = np.tensordot(D, D.conjugate(), axes=([1,2], [1,2]))
        # two = np.einsum('ija, ak', C, one)
        two = np.tensordot(C, one, axes=([2], [0]))
        # two = np.einsum('ika, jla', two, C)
        two = np.tensordot(two, C.conjugate(), axes=([2], [2])).transpose((0,2,1,3))
        r2 = two.reshape((cs[0]**2, cs[1]**2))
        bs = B.shape
        # tws = two.shape
        one = np.einsum('ijaa', two)
        # two = np.einsum('ija, ak', B, one)
        two = np.tensordot(B, one, axes=([2], [0]))
        # two = np.einsum('ika, jla', two, B)
        two = np.tensordot(two, B.conjugate(), axes=([2], [2])).transpose((0,2,1,3))
        r3 = two.reshape((bs[0]**2, bs[1]**2))
        # print(cs, ds)
        # one = D.reshape((ds[0], ds[1]*ds[2]))
        # one = np.dot(one, one.transpose())
        # print(one.shape)
        # one = np.dot(C.reshape((cs[0]*cs[1], cs[2])), one)
        # one = np.dot(C.reshape((cs[0]*cs[1], cs[2])), one.transpose())
        # one =  one.reshape((cs[0], cs[1], cs[0], cs[1])).transpose((0, 2, 1, 3))
        return (r2, r3)


    # def getQ(self, s1, s2, r2, r3):
    #     """
    #     Computes the Q matrix from smaller S and R
    #     matrices using triads.

    #     """
    #     ss = s1.shape
    #     x = int(np.rint(np.sqrt(ss[0])))
    #     temp = s1.dot(s2)
    #     temp = temp.dot(r2)
    #     temp = temp.dot(r3.transpose())
    #     temp = temp.dot(s1.transpose())
    #     temp = temp.reshape((x, x, x, x)).transpose((2, 0, 3, 1))
    #     return temp.reshape((x**2, x**2))

    # def make_q_from_triads(self, A, B, C, D):
    #     """
    #     A generic make Q method for any four triads.

    #     """
    #     s1 = self.getS(A)
    #     s2 = self.getS(B)
    #     r2, r3 = self.getR23(C, D, B)
    #     q = self.getQ(s1, s2, r2, r3)
    #     return q

    # def get_UUdag(self, A, B, C, D):
    #     """
    #     From the Q matrix this gets the isometry and the
    #     singular values from the eigenvalues.

    #     """
    #     q = self.make_q_from_triads(A, B, C, D)
    #     assert np.allclose(q, q.conjugate().transpose())
    #     evals_left, Uleft = np.linalg.eigh(q)
    #     Udag = Uleft.dot(np.diag(1/np.sqrt(np.abs(evals_left))))
    #     U = Uleft.dot(np.diag(np.sqrt(np.abs(evals_left))))
    #     return (U, Udag)

    def getQ_left_front(self, s1, s2, r2, r3):
        ss = s1.shape
        x = int(np.rint(np.sqrt(ss[0])))
        temp = s1.dot(s2)
        temp = temp.dot(r2)
        temp = temp.dot(r3.transpose())
        temp = temp.dot(s1.transpose())
        temp = temp.reshape((x, x, x, x)).transpose((2, 0, 3, 1))
        return temp.reshape((x**2, x**2))

    def getQ_right_back(self, s1, s2, r2, r3):
        ss = s1.shape
        x = int(np.rint(np.sqrt(ss[0])))
        temp = s1.dot(s2)
        temp = temp.dot(r2)
        temp = temp.dot(r3.transpose())
        temp = temp.dot(s1.transpose())
        temp = temp.reshape((x, x, x, x)).transpose((0, 2, 1, 3))
        return temp.reshape((x**2, x**2))

    # def make_q_from_triads(self, A, B, C, D, which):
    #     """
    #     A generic make Q method for any four triads.

    #     """
    #     s1 = self.getS(A)
    #     s2 = self.getS(B)
    #     r2, r3 = self.getR23(C, D, B)
    #     if which == 'lf':
    #         q = self.getQ_left_front(s1, s2, r2, r3)
    #     elif which == 'rb':
    #         q = self.getQ_right_back(s1, s2, r2, r3)
    #     else:
    #         raise ValueError("must be 'lf' or 'rb'")
    #     return q


    def get_UUdag(self, A, B, C, D, which):
        """
        From the Q matrix this gets the isometry and the
        singular values from the eigenvalues.

        """
        # q = self.make_q_from_triads(A, B, C, D)
        s1 = self.getS(A)
        s2 = self.getS(B)
        r2, r3 = self.getR23(C, D, B)
        if which == 'lf':
            q = self.getQ_left_front(s1, s2, r2, r3)
        elif which == 'rb':
            q = self.getQ_right_back(s1, s2, r2, r3)
        else:
            raise ValueError("must be 'lf' or 'rb'")
        assert np.allclose(q, q.conjugate().transpose())
        evals_left, Uleft = np.linalg.eigh(q)
        # idx = np.abs(evals_left).argsort()[::-1]
        # res = np.sum(evals_left[idx][self.dbond:])
        Udag = Uleft.dot(np.diag(1/np.sqrt(np.abs(evals_left))))
        U = Uleft.dot(np.diag(np.sqrt(np.abs(evals_left))))
        return (U, Udag)
        # print("largest =", np.max(e))
        # sys.stdout.flush()
        # return (res, Uleft[:, idx])

    def update_triads(self, getV=True):
        """
        Computes the squeezers for each direction, then
        updates each direction using them.

        """
        # make the left isometry
        U, Udag = self.get_UUdag(self.A, self.B, self.C, self.D, which='lf')
        # make the right isometry
        W, Wdag = self.get_UUdag(self.D.transpose((2,1,0)),
                                 self.C.transpose((2,1,0)),
                                 self.B.transpose((2,1,0)),
                                 self.A.transpose((2,1,0)), which='rb')
        center = W.conjugate().transpose().dot(U)
        u, vdag, bwlr = bond_weight(center, k=self.hyp_k, cut=self.dbond)
        print("lr shape", bwlr.shape)
        left_isometry = Udag.dot(vdag.transpose())  # possible conjugate?
        # if alpha <= self.dbond:
        #     left_isometry = Udag.dot(vdag.transpose())  # possible conjugate?
        # else:
        #     left_isometry = Udag.dot(vdag.transpose()[:, :self.dbond])
        right_isometry = Wdag.dot(u)
        # if alpha <= self.dbond:
        #     right_isometry = Wdag.dot(u)
        # else:
        #     right_isometry = Wdag.dot(u[:, :self.dbond])
        # done with left and right
        # starting front and back
        # make the back isometry
        U, Udag = self.get_UUdag(self.D.transpose((1,2,0)),
                                 self.C.transpose((2,1,0)),
                                 self.B.transpose((2,1,0)),
                                 self.A.transpose((2,1,0)), which='rb')        
        # make the front isometry
        W, Wdag = self.get_UUdag(self.A.transpose((1,0,2)),
                                 self.B,
                                 self.C,
                                 self.D, which='lf')
        center = W.conjugate().transpose().dot(U)
        u, vdag, bwfb = bond_weight(center, k=self.hyp_k, cut=self.dbond)
        print("fb size", bwfb.shape)
        back_isometry = Udag.dot(vdag.transpose())
        front_isometry = Wdag.dot(u)
        # center = W.conjugate().transpose().dot(U)
        # u, vdag, alpha = split(center)
        # if alpha <= self.dbond:
        #     back_isometry = Udag.dot(vdag.transpose())
        # else:
        #     back_isometry = Udag.dot(vdag.transpose()[:, :self.dbond])
        # if alpha <= self.dbond:
        #     front_isometry = Wdag.dot(u)
        # else:
        #     front_isometry = Wdag.dot(u[:, :self.dbond])
        if self.imp:
            self.make_new_impure_triads(left_isometry, right_isometry,
                                        front_isometry, back_isometry)
        if self.nnimp:
            self.contract_nn_triads(left_isometry, right_isometry,
                                    front_isometry, back_isometry)
            self.nnimp = False
            self.imp = True
        # self.make_new_triads(U, V)
        self.make_new_triads(left_isometry, right_isometry,
                             front_isometry, back_isometry)
        self.bond_weights[2] = bwlr
        self.bond_weights[1] = bwfb
        self.bond_weights = [self.bond_weights[1], self.bond_weights[2],
                             self.bond_weights[0]]


    def makeD(self, U, V):
        """
        U is ordered (left-top left-bottom left)
        V is ordered (front-top front-bottom front)
        """
        us = U.shape
        us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
        bs = self.B.shape
        cs = self.C.shape
        # ds = self.D.shape
        vs = V.shape
        vs = (int(np.rint(np.sqrt(vs[0]))), int(np.rint(np.sqrt(vs[0]))), vs[1])
        
        # one = np.einsum('iqj, kql', self.D, V.reshape(vs))
        # print(self.D.shape, vs)
        V = V.reshape(vs)
        V = np.einsum('abk, ia, jb', V, np.sqrt(self.bond_weights[1]),
                      np.sqrt(self.bond_weights[1]))  # these could come from Q?
        one = np.tensordot(self.D, V, axes=([1], [1]))
        U = U.reshape(us)
        U = np.einsum('abk, ia, jb', U, np.sqrt(self.bond_weights[2]),
                      np.sqrt(self.bond_weights[2]))
        # two = np.einsum('ijp, pkl', self.D, U.reshape(us))
        two = np.tensordot(self.D, U, axes=([2], [0]))
        # self.D = np.einsum('iqpl, jpqk', two, one).reshape((ds[0]**2, vs[2]*us[2]))
        self.D = np.tensordot(two, one, axes=([1,2], [2,1]))
        # self.D = self.D.reshape((ds[0]**2, vs[2]*us[2]))
        one = np.tensordot(self.C, self.D, axes=([2], [0])).transpose((0,3,4,2,1))
        
        # M = np.einsum('iwk, jwl', self.B, self.C).reshape((bs[0]*cs[0], bs[2]*cs[2]))
        self.B = np.tensordot(self.B, self.bond_weights[0],
                              axes=([1], [0])).transpose((0,2,1))
        two = np.tensordot(self.B, self.C, axes=([1], [1])).transpose((0,2,1,3))

        one = np.tensordot(two, one, axes=([2,3], [0,1]))
        one = one.reshape((bs[0]*cs[0]*vs[2], us[2]*cs[1]))
        G, self.D, alpha = split(one, cut=self.dbond, split='left')
        self.D = self.D.reshape((alpha, us[2], cs[1]))
        # print(self.D.shape)
        return G.reshape((bs[0], cs[0], vs[2], alpha))

    def makeDimp(self, U, V):
        """
        U is ordered (left-top left-bottom left)
        V is ordered (front-top front-bottom front)
        """
        us = U.shape
        us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
        bs = self.B.shape
        bsi = self.Bimp.shape
        cs = self.C.shape
        csi = self.Cimp.shape
        ds = self.D.shape
        vs = V.shape
        vs = (int(np.rint(np.sqrt(vs[0]))), int(np.rint(np.sqrt(vs[0]))), vs[1])
        
        # one = np.einsum('iqj, kql', self.D, V.reshape(vs))
        # print(self.D.shape, vs)
        one = np.tensordot(self.D, V.reshape(vs), axes=([1], [1]))
        # two = np.einsum('ijp, pkl', self.D, U.reshape(us))
        two = np.tensordot(self.Dimp, U.reshape(us), axes=([2], [0]))
        # self.D = np.einsum('iqpl, jpqk', two, one).reshape((ds[0]**2, vs[2]*us[2]))
        self.Dimp = np.tensordot(two, one, axes=([1,2], [2,1]))
        # self.D = self.D.reshape((ds[0]**2, vs[2]*us[2]))
        one = np.tensordot(self.Cimp, self.Dimp,
                           axes=([2], [0])).transpose((0,3,4,2,1))
        
        # M = np.einsum('iwk, jwl', self.B, self.C).reshape((bs[0]*cs[0], bs[2]*cs[2]))
        two = np.tensordot(self.Bimp, self.C, axes=([1], [1])).transpose((0,2,1,3))

        one = np.tensordot(two, one, axes=([2,3], [0,1]))
        one = one.reshape((bsi[0]*cs[0]*vs[2], us[2]*csi[1]))
        G, self.Dimp, alpha = split(one, cut=self.dbond, split='left')
        self.Dimp = self.Dimp.reshape((alpha, us[2], csi[1]))
        # print(self.D.shape)
        return G.reshape((bsi[0], cs[0], vs[2], alpha))



    def makeDnnimp(self, U, V):
        """
        U is ordered (left-top left-bottom left)
        V is ordered (front-top front-bottom front)
        """
        us = U.shape
        us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
        bs = self.B.shape
        cs = self.C.shape
        ds = self.D.shape
        vs = V.shape
        vs = (int(np.rint(np.sqrt(vs[0]))), int(np.rint(np.sqrt(vs[0]))), vs[1])
        
        # one = np.einsum('iqj, kql', self.D, V.reshape(vs))
        # print(self.D.shape, vs)
        one = np.tensordot(self.Dimp2, V.reshape(vs), axes=([1], [1]))
        # two = np.einsum('ijp, pkl', self.D, U.reshape(us))
        two = np.tensordot(self.Dimp1, U.reshape(us), axes=([2], [0]))
        # self.D = np.einsum('iqpl, jpqk', two, one).reshape((ds[0]**2, vs[2]*us[2]))
        self.Dimp = np.tensordot(two, one, axes=([1,2], [2,1]))
        # self.D = self.D.reshape((ds[0]**2, vs[2]*us[2]))
        one = np.tensordot(self.Cimp1, self.Dimp,
                           axes=([2], [0])).transpose((0,3,4,2,1))
        
        # M = np.einsum('iwk, jwl', self.B, self.C).reshape((bs[0]*cs[0], bs[2]*cs[2]))
        two = np.tensordot(self.Bimp1, self.Cimp2, axes=([1], [1])).transpose((0,2,1,3))

        one = np.tensordot(two, one, axes=([2,3], [0,1]))
        one = one.reshape((bs[0]*cs[0]*vs[2], us[2]*cs[1]))
        G, self.Dimp, alpha = split(one, cut=self.dbond, split='left')
        self.Dimp = self.Dimp.reshape((alpha, us[2], cs[1]))
        # print(self.D.shape)
        return G.reshape((bs[0], cs[0], vs[2], alpha))


    
    def makeA(self, G, U, V):
        """
        U is ordered (left-top left-bottom left)
        V is ordered (front-top front-bottom front)
        """
        us = U.shape
        us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
        # As = self.A.shape
        bs = self.B.shape
        gs = G.shape
        vs = V.shape
        vs = (int(np.rint(np.sqrt(vs[0]))), int(np.rint(np.sqrt(vs[0]))), vs[1])
        
        V = V.reshape(vs)
        V = np.einsum('abk, ia, jb', V, np.sqrt(self.bond_weights[1]),
                      np.sqrt(self.bond_weights[1]))  # these could come from Q?
        # one = np.einsum('iaj, kal', V.reshape(vs), self.A)
        one = np.tensordot(V.conjugate(), self.A, axes=([1], [1]))
        U = U.reshape(us)
        U = np.einsum('abk, ia, jb', U, np.sqrt(self.bond_weights[2]),
                      np.sqrt(self.bond_weights[2]))
        # two = np.einsum('aij, akl', U.reshape(us), self.A)
        two = np.tensordot(U.conjugate(), self.A, axes=([0], [0]))
        # self.A = np.einsum('piqk, qjpl', two, one).reshape((us[2]*vs[2], As[2]**2))
        self.A = np.tensordot(one, two, axes=([0,2], [2,0]))
        self.A = np.tensordot(self.B, self.A, axes=([0], [1])).transpose((0,3,4,2,1))
        # self.A = self.A.reshape((us[2]*vs[2], As[2]**2))
        
        one = np.tensordot(self.A, G, axes=([2,4], [0,1]))
        one = one.reshape((bs[1]*us[2], vs[2]*gs[2]*gs[3]))

        self.A, G, alpha = split(one, cut=self.dbond, split='right')
        self.A = self.A.reshape((bs[1], us[2], alpha)) # check ordering
        # print(self.A.shape)
        return G.reshape((alpha, vs[2], gs[2], gs[3]))

    def makeAimp(self, G, U, V):
        """
        U is ordered (left-top left-bottom left)
        V is ordered (front-top front-bottom front)
        """
        us = U.shape
        us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
        As = self.A.shape
        bs = self.B.shape
        gs = G.shape
        vs = V.shape
        vs = (int(np.rint(np.sqrt(vs[0]))), int(np.rint(np.sqrt(vs[0]))), vs[1])
        
        
        # one = np.einsum('iaj, kal', V.reshape(vs), self.A)
        one = np.tensordot(V.reshape(vs).conjugate(), self.A, axes=([1], [1]))
        # two = np.einsum('aij, akl', U.reshape(us), self.A)
        two = np.tensordot(U.reshape(us).conjugate(), self.Aimp, axes=([0], [0]))
        # self.A = np.einsum('piqk, qjpl', two, one).reshape((us[2]*vs[2], As[2]**2))
        self.Aimp = np.tensordot(one, two, axes=([0,2], [2,0]))
        self.Aimp = np.tensordot(self.B, self.Aimp,
                                 axes=([0], [1])).transpose((0,3,4,2,1))
        # self.A = self.A.reshape((us[2]*vs[2], As[2]**2))

        
        one = np.tensordot(self.Aimp, G, axes=([2,4], [0,1]))
        one = one.reshape((bs[1]*us[2], vs[2]*gs[2]*gs[3]))

        self.Aimp, G, alpha = split(one, cut=self.dbond, split='right')
        self.Aimp = self.Aimp.reshape((bs[1], us[2], alpha)) # check ordering
        # print(self.A.shape)
        return G.reshape((alpha, vs[2], gs[2], gs[3]))


    def makeAnnimp(self, G, U, V):
        """
        U is ordered (left-top left-bottom left)
        V is ordered (front-top front-bottom front)
        """
        us = U.shape
        us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
        As = self.A.shape
        bs = self.B.shape
        gs = G.shape
        vs = V.shape
        vs = (int(np.rint(np.sqrt(vs[0]))), int(np.rint(np.sqrt(vs[0]))), vs[1])
        
        
        # one = np.einsum('iaj, kal', V.reshape(vs), self.A)
        one = np.tensordot(V.reshape(vs).conjugate(), self.Aimp2, axes=([1], [1]))
        # two = np.einsum('aij, akl', U.reshape(us), self.A)
        two = np.tensordot(U.reshape(us).conjugate(), self.Aimp1, axes=([0], [0]))
        # self.A = np.einsum('piqk, qjpl', two, one).reshape((us[2]*vs[2], As[2]**2))
        self.Aimp = np.tensordot(one, two, axes=([0,2], [2,0]))
        self.Aimp = np.tensordot(self.Bimp2, self.Aimp,
                                 axes=([0], [1])).transpose((0,3,4,2,1))
        # self.A = self.A.reshape((us[2]*vs[2], As[2]**2))

        
        one = np.tensordot(self.Aimp, G, axes=([2,4], [0,1]))
        one = one.reshape((bs[1]*us[2], vs[2]*gs[2]*gs[3]))

        self.Aimp, G, alpha = split(one, cut=self.dbond, split='right')
        self.Aimp = self.Aimp.reshape((bs[1], us[2], alpha)) # check ordering
        # print(self.A.shape)
        return G.reshape((alpha, vs[2], gs[2], gs[3]))

    
    
    def makeBC(self, G):
        gs = G.shape

        self.B, self.C, alpha = split(G.reshape((gs[0]*gs[1], gs[2]*gs[3])),
                                      cut=self.dbond)
        self.B = self.B.reshape((gs[0], gs[1], alpha)) # check ordering
        self.C = self.C.reshape((alpha, gs[2], gs[3])) # check ordering

    def makeBCimp(self, G):
        gs = G.shape

        self.Bimp, self.Cimp, alpha = split(G.reshape((gs[0]*gs[1], gs[2]*gs[3])),
                                      cut=self.dbond)
        self.Bimp = self.Bimp.reshape((gs[0], gs[1], alpha)) # check ordering
        self.Cimp = self.Cimp.reshape((alpha, gs[2], gs[3])) # check ordering


        
    # def make_new_triads(self, U, V):        
    def make_new_triads(self, left, right, front, back):        
        G = self.makeD(right, back)
        G = self.makeA(G, left, front)
        self.makeBC(G)

    def make_new_impure_triads(self, left, right, front, back):
        G = self.makeDimp(right, back)
        G = self.makeAimp(G, left, front)
        self.makeBCimp(G)

    def contract_nn_triads(self, left, right, front, back):
        G = self.makeDnnimp(right, back)
        G = self.makeAnnimp(G, left, front)
        self.makeBCimp(G)
        
        
    def get_lognorms(self,):
        lognorms = self.lognorms[:]
        mid = np.tensordot(self.B, self.C, axes=([1,2], [1,0]))
        other = np.tensordot(self.A, self.D, axes=([0,1], [2,1]))
        trace = np.trace(np.dot(other.transpose(), mid))
        print("trace = ", trace)
        if trace < 0:
            print("negative trace!")
        lognorms.append(np.log(np.abs(np.trace(np.dot(other.transpose(), mid)))))
        return lognorms

    def get_imp_ratio(self,):
        mid = np.tensordot(self.B, self.C, axes=([1,2], [1,0]))
        other = np.tensordot(self.A, self.D, axes=([0,1], [2,1]))
        pure = np.trace(np.dot(other.transpose(), mid))

        mid = np.tensordot(self.Bimp, self.Cimp, axes=([1,2], [1,0]))
        other = np.tensordot(self.Aimp, self.Dimp, axes=([0,1], [2,1]))
        impure = np.trace(np.dot(other.transpose(), mid))
        return impure/pure

        
    def tensor_trace(self,):
        mid = np.tensordot(self.B, self.C, axes=([1,2], [1,0]))
        other = np.tensordot(self.A, self.D, axes=([0,1], [2,1]))
        trace = np.trace(np.dot(other.transpose(), mid))
        print("trace =", trace)
        if trace < 0:
            print("negative trace!")
        self.lognorms.append(np.log(np.abs(trace)))

    # def imp_trace_ratio(self,):
    #     mid = np.tensordot(self.B, self.C, axes=([1,2], [1,0]))
    #     other = np.tensordot(self.A, self.D, axes=([0,1], [2,1]))
    #     pure = np.trace(np.dot(other.transpose(), mid))

    #     mid = np.tensordot(self.Bimp, self.Cimp, axes=([1,2], [1,0]))
    #     other = np.tensordot(self.Aimp, self.Dimp, axes=([0,1], [2,1]))
    #     impure = np.trace(np.dot(other.transpose(), mid))
    #     return impure/pure
        
        # """
        # U is ordered (left-top left-bottom left)
        # V is ordered (front-top front-bottom front)
        # """
        # us = U.shape
        # us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
        # As = self.A.shape
        # bs = self.B.shape
        # cs = self.C.shape
        # ds = self.D.shape
        # vs = V.shape
        # vs = (int(np.rint(np.sqrt(vs[0]))), int(np.rint(np.sqrt(vs[0]))), vs[1])
        
        
        # # one = np.einsum('iaj, kal', V.reshape(vs), self.A)
        # one = np.tensordot(V.reshape(vs), self.A, axes=([1], [1]))
        # # two = np.einsum('aij, akl', U.reshape(us), self.A)
        # two = np.tensordot(U.reshape(us), self.A, axes=([0], [0]))
        # # self.A = np.einsum('piqk, qjpl', two, one).reshape((us[2]*vs[2], As[2]**2))
        # self.A = np.tensordot(two, one, axes=([0, 2], [2, 0])).transpose((0, 2, 1, 3))
        # self.A = self.A.reshape((us[2]*vs[2], As[2]**2))
        
        # # one = np.einsum('iqj, kql', self.D, V.reshape(vs))
        # one = np.tensordot(self.D, V.reshape(vs), axes=([1], [1]))
        # # two = np.einsum('ijp, pkl', self.D, U.reshape(us))
        # two = np.tensordot(self.D, U.reshape(us), axes=([2], [0]))
        # # self.D = np.einsum('iqpl, jpqk', two, one).reshape((ds[0]**2, vs[2]*us[2]))
        # self.D = np.tensordot(two, one, axes=([1,2], [2,1])).transpose((0,2,3,1))
        # # self.D = self.D.reshape((ds[0]**2, vs[2]*us[2]))
        # UU = np.tensordot(self.C, self.D, axes=([2], [0]))
        
        # # M = np.einsum('iwk, jwl', self.B, self.C).reshape((bs[0]*cs[0], bs[2]*cs[2]))
        # M = np.tensordot(self.B, self.C, axes=([1], [1]))

        # one = np.tensordot(M, UU, axes=([1,3], [0,2])).transpose((0,1,3,2,4))
        # one = one.reshape((bs[0]*cs[0]*vs[2], cs[1]*us[2]))
        # G, self.D, alpha = split(one, cut=self.dbond, split='left')
        # self.D = self.D.reshape((alpha, cs[1], us[2])) # check ordering
        # return G.reshape((bs[0], cs[0], vs[2], alpha))
                
        # # M = M.reshape((bs[0]*cs[0], bs[2]*cs[2]))
        # # print(bs, cs)
        # # print(M.shape)
        # # left, right, dd = split(M, cut=self.dbond)
        # # print("here")
        
        # # one = np.einsum('jka, ial', self.B,
        # #                 left.reshape((bs[0], cs[0], dd))).reshape((bs[0]**2, bs[1], dd))
        # one = np.tensordot(self.B, left.reshape((bs[0], cs[0], dd)), axes=([2], [1]))
        # one = one.transpose((2,0,1,3)).reshape((bs[0]**2, bs[1], dd))
        # self.A, Ar, dda = split(self.A, cut=self.dbond)
        # self.A = self.A.reshape((us[2], vs[2], dda))
        # # self.B = np.einsum('ia, ajk', Ar, one)
        # self.B = np.tensordot(Ar, one, axes=([1], [0]))
        
        # # one = np.einsum('ial, ajk', right.reshape((dd, bs[2], cs[2])),
        # #                 self.C).reshape((dd, cs[1], cs[2]**2))
        # one = np.tensordot(right.reshape((dd, bs[2], cs[2])), self.C, axes=([1], [0]))
        # one = one.transpose((0,2,3,1)).reshape((dd, cs[1], cs[2]**2))
        # Dl, self.D, ddd = split(self.D, cut=self.dbond)
        # self.D = self.D.reshape((ddd, vs[2], us[2])) 
        # # self.C = np.einsum('ija, ak', one, Dl)
        # self.C = np.tensordot(one, Dl, axes=([2], [0]))



        

# def getS(AorB):
#     """ A or B is structured (left, front, alpha) """
#     As = AorB.shape
#     # one = AorB.transpose((1, 0, 2)).reshape((As[1], As[0]*As[2]))
#     # want = np.dot(one.transpose(), one)
#     # want = want.reshape((As[0], As[2], As[0], As[2]))
#     # want = want.transpose((0, 2, 1, 3)).reshape((As[0]**2, As[2]**2))
#     # want = np.einsum('iak, jal', AorB, AorB).reshape((As[0]**2, As[2]**2))
#     want = np.tensordot(AorB, AorB, axes=([1], [1])).transpose((0,2,1,3))
#     want = want.reshape((As[0]**2, As[2]**2))
#     return want

# def getR23(C, D, B):
#     """
#     C is structured (beta top gamma)
#     D is structured (gamma away right)
#     """
#     cs = C.shape
#     # ds = D.shape
#     # one = np.einsum('iab, jab', D, D)
#     one = np.tensordot(D, D, axes=([1,2], [1,2]))
#     # two = np.einsum('ija, ak', C, one)
#     two = np.tensordot(C, one, axes=([2], [0]))
#     # two = np.einsum('ika, jla', two, C)
#     two = np.tensordot(two, C, axes=([2], [2])).transpose((0,2,1,3))
#     r2 = two.reshape((cs[0]**2, cs[1]**2))
#     bs = B.shape
#     # tws = two.shape
#     one = np.einsum('ijaa', two)
#     # two = np.einsum('ija, ak', B, one)
#     two = np.tensordot(B, one, axes=([2], [0]))
#     # two = np.einsum('ika, jla', two, B)
#     two = np.tensordot(two, B, axes=([2], [2])).transpose((0,2,1,3))
#     r3 = two.reshape((bs[0]**2, bs[1]**2))
#     # print(cs, ds)
#     # one = D.reshape((ds[0], ds[1]*ds[2]))
#     # one = np.dot(one, one.transpose())
#     # print(one.shape)
#     # one = np.dot(C.reshape((cs[0]*cs[1], cs[2])), one)
#     # one = np.dot(C.reshape((cs[0]*cs[1], cs[2])), one.transpose())
#     # one =  one.reshape((cs[0], cs[1], cs[0], cs[1])).transpose((0, 2, 1, 3))
#     return (r2, r3)


# def getQ(s1, s2, r2, r3):
#     ss = s1.shape
#     x = int(np.rint(np.sqrt(ss[0])))
#     temp = s1.dot(s2)
#     temp = temp.dot(r2)
#     temp = temp.dot(r3.transpose())
#     temp = temp.dot(s1.transpose())
#     temp = temp.reshape((x, x, x, x)).transpose((2, 0, 3, 1))
#     return temp.reshape((x**2, x**2))

# def getU(q, nums):
#     qs = q.shape
#     # O = eigh(q, subset_by_index=[qs[0]-nums, qs[0]-1])
#     # O = eigh(q, eigvals=(qs[0]-nums, qs[0]-1))
#     if not np.allclose(q, q.conjugate().transpose()):
#         warnings.warn("q matrix is not Hermitian by allclose.")
#     # O = eigh(q)
#     O = np.linalg.eigh(q)
#     U = O[1]
#     e = O[0]
#     idx = e.argsort()[::-1]
#     # print("largest =", np.max(e))
#     sys.stdout.flush()
#     return (U[:,idx])[:,:nums]





class TwoDimensionalTriadNetwork:

    def __init__(self, dbond, triads=None, normlist=None, time_first=True):
        """
        Two dimensional triad initialization.  The triads
        are ordered (left, up, alpha) (alpha, down, right)
        in the time_first=True setting.  Otherwise they
        start with A and B flipped, and transposed as in
        the update with space and time switched.
 
        """
        self.dbond = dbond
        self.imp = False
        self.nnimp = False
        self.time_first = time_first
        if normlist is not None:
            self.lognorms = normlist
        else:
            self.lognorms = list()
        if triads is not None:
            self.A = triads[0]
            self.B = triads[1]
            if not self.time_first:
                self.A, self.B = self.B.transpose((1,2,0)), self.A.transpose((2,0,1))

    def reconstruct(self,):
        # tensor = np.einsum('ija, akl', self.A, self.B)
        tensor = np.tensordot(self.A, self.B, axes=([2], [0]))
        return tensor
        
    def norm(self,):
        # one = np.einsum('abi, abj', self.A, self.A)
        one = np.tensordot(self.A, self.A.conjugate(), axes=([0,1], [0,1]))
        # two = np.einsum('iab, jab', self.B, self.B)
        two = np.tensordot(self.B, self.B.conjugate(), axes=([1,2], [1,2]))
        norm = np.sqrt(np.trace(one.dot(two.transpose())))
        return norm

    def coarse_grain(self, normalize=True, all_vols=False):
        """ Does the coarse graining along a single direction."""
        # print("Coarsening second dimension...")
        dirs = ['x', 'y']
        for d in dirs:
            print("Doing " + d)
            self.update_triads()
            if normalize:
                self.normalize()
        if all_vols:
            if self.imp:
                return (self.getlognorms(), self.imp_tensor_ratio())
            else:
                return self.getlognorms()
        # print("Done.")
                

    def normalize(self,):
        """normalize each tensor, and return the total."""
        # normA = np.linalg.norm(self.A)
        # normB = np.linalg.norm(self.B)

        # self.A /= normA
        # self.B /= normB
        # norm = normA * normB
        # self.lognorms.append(np.log(norm))
        norm = self.norm()
        self.A /= np.sqrt(norm)
        self.B /= np.sqrt(norm)
        self.lognorms.append(np.log(norm))
        if self.imp:
            self.Aimp /= np.sqrt(norm)
            self.Bimp /= np.sqrt(norm)
        if self.nnimp:
            self.Aimp1 /= np.sqrt(norm)
            self.Aimp2 /= np.sqrt(norm)
            self.Bimp1 /= np.sqrt(norm)
            self.Bimp2 /= np.sqrt(norm)
            

    def get_triads(self, tensor):
        """
        A is (left, top, alpha)
        B is (alpha, bot, right)
        where alpha points right for A, and left for B.
        """
        # if (self.A is not None) or (self.B is not None):
        #     warnings.warn("A and B are already defined.")
        ts = tensor.shape
        rest = tensor.reshape((ts[0]*ts[1], ts[2]*ts[3]))
        self.A, self.B, alpha = split(rest, cut=self.dbond)
        self.A = self.A.reshape((ts[0], ts[1], alpha))
        self.B = self.B.reshape((alpha, ts[2], ts[3]))
        if not self.time_first:
            self.A, self.B = self.B.transpose((1,2,0)), self.A.transpose((2,0,1))
        # print(np.allclose(tensor, self.reconstruct()))


    def get_impure_triads(self, tensor):
        """
        A is (left, top, alpha)
        B is (alpha, bot, right)
        where alpha points right for A, and left for B.
        """
        # if (self.A is not None) or (self.B is not None):
        #     warnings.warn("A and B are already defined.")
        ts = tensor.shape
        rest = tensor.reshape((ts[0]*ts[1], ts[2]*ts[3]))
        self.Aimp, self.Bimp, alpha = split(rest, cut=self.dbond)
        self.Aimp = self.Aimp.reshape((ts[0], ts[1], alpha))
        self.Bimp = self.Bimp.reshape((alpha, ts[2], ts[3]))
        self.imp = True
        if not self.time_first:
            self.Aimp, self.Bimp = self.Bimp.transpose((1,2,0)), self.Aimp.transpose((2,0,1))
        print(self.Aimp.shape, self.Bimp.shape)
            


    def get_nn_imp_triads(self, tensor1, tensor2):
        ts = tensor1.shape
        rest = tensor1.reshape((ts[0]*ts[1], ts[2]*ts[3]))
        self.Aimp1, self.Bimp1, alpha = split(rest, cut=self.dbond)
        self.Aimp1 = self.Aimp1.reshape((ts[0], ts[1], alpha))
        self.Bimp1 = self.Bimp1.reshape((alpha, ts[2], ts[3]))
        if not self.time_first:
            self.Aimp1, self.Bimp1 = self.Bimp1.transpose((1,2,0)), self.Aimp1.transpose((2,0,1))

        ts = tensor2.shape
        rest = tensor2.reshape((ts[0]*ts[1], ts[2]*ts[3]))
        self.Aimp2, self.Bimp2, alpha = split(rest, cut=self.dbond)
        self.Aimp2 = self.Aimp2.reshape((ts[0], ts[1], alpha))
        self.Bimp2 = self.Bimp2.reshape((alpha, ts[2], ts[3]))
        if not self.time_first:
            self.Aimp2, self.Bimp2 = self.Bimp2.transpose((1,2,0)), self.Aimp2.transpose((2,0,1))
            
        self.nnimp = True

        

    def getQ(self,):
        As = self.A.shape
        # S1 = np.einsum('iak, jal', self.A, self.A.conjugate()).reshape((As[0]**2, As[2]**2))
        S1 = np.tensordot(self.A, self.A.conjugate(), axes=([1], [1])).transpose((0,2,1,3))
        S1 = S1.reshape((As[0]**2, As[2]**2))
        bs = self.B.shape
        # S2 = np.einsum('ika, jla', self.B, self.B.conjugate())
        S2 = np.tensordot(self.B, self.B.conjugate(), axes=([2], [2])).transpose((0,2,1,3))
        R2 = np.einsum('ijaa', S2)
        S2 = S2.reshape((bs[0]**2, bs[1]**2))
        # R3 = np.einsum('ija, ak', self.A, R2)
        R3 = np.tensordot(self.A, R2, axes=([2], [0]))
        # R3 = np.einsum('ika, jla', R3, self.A.conjugate()).reshape((As[0]**2, As[1]**2))
        R3 = np.tensordot(R3, self.A.conjugate(), axes=([2], [2])).transpose((0,2,1,3))
        R3 = R3.reshape((As[0]**2, As[1]**2))

        Q = S1.dot(S2)
        Q = Q.dot(R3.transpose()).reshape((As[0], As[0], As[0], As[0]))
        Q = Q.transpose((0, 2, 1, 3)).reshape((As[0]**2, As[0]**2))
        # Q = 0.5 * (Q + Q.conjugate().transpose())
        # print(Q)
        assert np.allclose(Q, Q.conjugate().transpose())
        return Q

    def update_triads(self,):
        """
        updates the triads starting from computing
        Q upto updating the individual triads.

        """
        q = self.getQ()
        if (self.A.shape[0]**2 < self.dbond):
            U = getU(q, self.A.shape[0]**2)
            # print(U.shape)
            # assert np.allclose(U.dot(U.conjugate().transpose()), np.eye(U.shape[0]))
            # print(U.transpose().conjugate().dot(U))
            # assert np.allclose(U.transpose().dot(U), np.eye(U.shape[0]))
            # U = np.eye(q.shape[0])
        else:
            U = getU(q, self.dbond)
            # U = np.eye(q.shape[0])
        if self.imp:
            self.make_new_impure_triads(U)
        if self.nnimp:
            self.do_nn_triad_update(U)
            self.nnimp = False
            self.imp = True
        self.make_new_triads(U)

    def make_new_triads(self, U):
        # U and V are (top, bot, prime)
        us = U.shape
        us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
        # vs = V.shape
        # vs = (int(np.rint(np.sqrt(vs[0]))), int(np.rint(np.sqrt(vs[0]))), vs[1])
        As = self.A.shape
        bs = self.B.shape

        # top = np.einsum('ajk, ali', self.A, U.reshape(us)).reshape((us[2]*As[1], As[2]*us[1]))
        top = np.tensordot(self.A, U.reshape(us).conjugate(), axes=([0], [0])).transpose((3,0,1,2))
        top = top.reshape((us[2]*As[1], As[2]*us[1]))
        
        # mid = np.einsum('iak, jal', self.B, self.A).reshape((bs[0]*As[0], bs[2]*As[2]))
        mid = np.tensordot(self.B, self.A, axes=([1], [1])).transpose((0,2,1,3))
        mid = mid.reshape((bs[0]*As[0], bs[2]*As[2]))
        # left, right, alpha = split(mid, cut = self.dbond)
        top = np.dot(top, mid)
        # bot = np.einsum('ial, jka', U.reshape(us), self.B).reshape((us[0]*bs[0], bs[1]*us[2]))
        bot = np.tensordot(U.reshape(us), self.B, axes=([1], [2])).transpose((0,2,3,1))
        bot = bot.reshape((us[0]*bs[0], bs[1]*us[2]))

        top = np.dot(top, bot)
        self.B, self.A, alpha = split(top, cut=self.dbond)
        self.B = self.B.reshape((us[2], As[1], alpha)).transpose((2,0,1))
        self.A = self.A.reshape((alpha, bs[1], us[2])).transpose((1,2,0))
        # self.A = np.dot(top, left).reshape((us[2], As[1], alpha))
        # self.B = np.dot(right, bot).reshape((alpha, bs[1], us[2]))

    # def make_new_impure_triads(self, U):
    #     # U and V are (top, bot, prime)
    #     us = U.shape
    #     us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
    #     As = self.A.shape
    #     bs = self.B.shape
    #     ibs = self.Bimp.shape
    #     ias = self.Aimp.shape
        
    #     # top = np.einsum('ajk, ali', self.A, U.reshape(us)).reshape((us[2]*As[1], As[2]*us[1]))
    #     top = np.tensordot(self.Aimp, U.reshape(us).conjugate(), axes=([0], [0])).transpose((3,0,1,2))
    #     top = top.reshape((us[2]*ias[1], ias[2]*us[1]))
        
    #     # mid = np.einsum('iak, jal', self.B, self.A).reshape((bs[0]*As[0], bs[2]*As[2]))
    #     mid = np.tensordot(self.Bimp, self.A, axes=([1], [1])).transpose((0,2,1,3))
    #     mid = mid.reshape((ibs[0]*As[0], ibs[2]*As[2]))
    #     # left, right, alpha = split(mid, cut = self.dbond)
    #     top = np.dot(top, mid)
    #     # bot = np.einsum('ial, jka', U.reshape(us), self.B).reshape((us[0]*bs[0], bs[1]*us[2]))
    #     bot = np.tensordot(U.reshape(us), self.B, axes=([1], [2])).transpose((0,2,3,1))
    #     bot = bot.reshape((us[0]*bs[0], bs[1]*us[2]))
        
    #     top1 = np.dot(top, bot)
    #     # Bimp1, Aimp1, alpha = split(top, cut=self.dbond)
    #     # Bimp1 = Bimp1.reshape((us[2], ias[1], alpha)).transpose((2,0,1))
    #     # Aimp1 = Aimp1.reshape((alpha, bs[1], us[2])).transpose((1,2,0))

    #     top = np.tensordot(self.A, U.reshape(us).conjugate(), axes=([0], [0])).transpose((3,0,1,2))
    #     top = top.reshape((us[2]*As[1], As[2]*us[1]))
        
    #     # mid = np.einsum('iak, jal', self.B, self.A).reshape((bs[0]*As[0], bs[2]*As[2]))
    #     mid = np.tensordot(self.B, self.Aimp, axes=([1], [1])).transpose((0,2,1,3))
    #     mid = mid.reshape((bs[0]*ias[0], bs[2]*ias[2]))
    #     # left, right, alpha = split(mid, cut = self.dbond)
    #     top = np.dot(top, mid)
    #     # bot = np.einsum('ial, jka', U.reshape(us), self.B).reshape((us[0]*bs[0], bs[1]*us[2]))
    #     bot = np.tensordot(U.reshape(us), self.Bimp, axes=([1], [2])).transpose((0,2,3,1))
    #     bot = bot.reshape((us[0]*ibs[0], ibs[1]*us[2]))
        
    #     top2 = np.dot(top, bot)
    #     top = 0.5*(top1 + top2)
    #     self.Bimp, self.Aimp, alpha = split(top, cut=self.dbond)
    #     self.Bimp = self.Bimp.reshape((us[2], ias[1], alpha)).transpose((2,0,1))
    #     self.Aimp = self.Aimp.reshape((alpha, ibs[1], us[2])).transpose((1,2,0))


    def make_new_impure_triads(self, U):
        # U and V are (top, bot, prime)
        us = U.shape
        us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
        As = self.A.shape
        bs = self.B.shape
        ibs = self.Bimp.shape
        ias = self.Aimp.shape
        
        # top = np.einsum('ajk, ali', self.A, U.reshape(us)).reshape((us[2]*As[1], As[2]*us[1]))
        top = np.tensordot(self.Aimp, U.reshape(us).conjugate(), axes=([0], [0])).transpose((3,0,1,2))
        top = top.reshape((us[2]*ias[1], ias[2]*us[1]))
        
        # mid = np.einsum('iak, jal', self.B, self.A).reshape((bs[0]*As[0], bs[2]*As[2]))
        mid = np.tensordot(self.Bimp, self.A, axes=([1], [1])).transpose((0,2,1,3))
        mid = mid.reshape((ibs[0]*As[0], ibs[2]*As[2]))
        # left, right, alpha = split(mid, cut = self.dbond)
        top = np.dot(top, mid)
        # bot = np.einsum('ial, jka', U.reshape(us), self.B).reshape((us[0]*bs[0], bs[1]*us[2]))
        bot = np.tensordot(U.reshape(us), self.B, axes=([1], [2])).transpose((0,2,3,1))
        bot = bot.reshape((us[0]*bs[0], bs[1]*us[2]))
        
        top = np.dot(top, bot)
        # Bimp1, Aimp1, alpha = split(top, cut=self.dbond)
        # Bimp1 = Bimp1.reshape((us[2], ias[1], alpha)).transpose((2,0,1))
        # Aimp1 = Aimp1.reshape((alpha, bs[1], us[2])).transpose((1,2,0))

        self.Bimp, self.Aimp, alpha = split(top, cut=self.dbond)
        self.Bimp = self.Bimp.reshape((us[2], ias[1], alpha)).transpose((2,0,1))
        self.Aimp = self.Aimp.reshape((alpha, bs[1], us[2])).transpose((1,2,0))


    def do_nn_triad_update(self, U):
        # U and V are (top, bot, prime)
        us = U.shape
        us = (int(np.rint(np.sqrt(us[0]))), int(np.rint(np.sqrt(us[0]))), us[1])
        ibs = self.Bimp1.shape
        ias = self.Aimp1.shape
        
        # top = np.einsum('ajk, ali', self.A, U.reshape(us)).reshape((us[2]*As[1], As[2]*us[1]))
        top = np.tensordot(self.Aimp1, U.reshape(us).conjugate(), axes=([0], [0])).transpose((3,0,1,2))
        top = top.reshape((us[2]*ias[1], ias[2]*us[1]))
        
        # mid = np.einsum('iak, jal', self.B, self.A).reshape((bs[0]*As[0], bs[2]*As[2]))
        mid = np.tensordot(self.Bimp1, self.Aimp2, axes=([1], [1])).transpose((0,2,1,3))
        mid = mid.reshape((ibs[0]*ias[0], ibs[2]*ias[2]))
        # left, right, alpha = split(mid, cut = self.dbond)
        top = np.dot(top, mid)
        # bot = np.einsum('ial, jka', U.reshape(us), self.B).reshape((us[0]*bs[0], bs[1]*us[2]))
        bot = np.tensordot(U.reshape(us), self.Bimp2, axes=([1], [2])).transpose((0,2,3,1))
        bot = bot.reshape((us[0]*ibs[0], ibs[1]*us[2]))
        
        top = np.dot(top, bot)
        self.Bimp, self.Aimp, alpha = split(top, cut=self.dbond)
        self.Bimp = self.Bimp.reshape((us[2], ias[1], alpha)).transpose((2,0,1))
        self.Aimp = self.Aimp.reshape((alpha, ibs[1], us[2])).transpose((1,2,0))


        
    def traceupdate(self,):
        As = self.A.shape
        bs = self.B.shape

        one = self.reconstruct()
        final = np.tensordot(one, one, axes=([1,2], [2,1]))
        final = final.transpose((0,2,1,3)).reshape((As[0]**2, bs[2]**2))
        # one = np.einsum('iak, jal', self.A, self.B)
        # one = np.tensordot(self.A, self.B, axes=([1], [1])).transpose((0,2,1,3))
        # two = np.einsum('iak, jal', self.B, self.A)
        # two = np.tensordot(self.B, self.A, axes=([1], [1])).transpose((0,2,1,3))
        # final = np.einsum('iabl, bjka', one, two).reshape((As[0]**2, bs[2]**2))
        # final = np.tensordot(one, one, axes=([1,2], [3,0])).transpose((0,2,3,1))
        # final = final.reshape((As[0]**2, bs[2]**2))
        
        return final

    def close_boundary(self,):
        """ makes a 1D transfer matrix from the two triads."""
        want = np.einsum('iab, baj', self.A, self.B)
        return want

    def getlognorms(self,):
        lognorms = self.lognorms[:]
        want = np.tensordot(self.A, self.B, axes=([1,2], [1,0]))
        lognorms.append(np.log(np.trace(want)))
        return lognorms

    def tensor_trace(self,):
        want = np.tensordot(self.A, self.B, axes=([1,2], [1,0]))
        self.lognorms.append(np.log(np.trace(want)))


    def imp_tensor_ratio(self,):
        want = np.tensordot(self.Aimp, self.Bimp, axes=([1,2], [1,0]))
        imp = np.trace(want)

        want = np.tensordot(self.A, self.B, axes=([1,2], [1,0]))
        pure = np.trace(want)
        return imp/pure

        


class One_Dimensional_Triad_Network:

    def __init__(self, matrix, normlist=None):
        if normlist is not None:
            self.lognorms = normlist
        self.A = matrix

    def coarse_grain(self, nt, normalize=True):
        print("Coarsening first dimension...")
        for i in range(nt):
            print(i+1, "of", nt)
            self.A = np.dot(self.A, self.A)
            if normalize:
                self.normalize()
        print("Done.")

    def normalize(self,):
        norm = np.linalg.norm(self.A)

        self.A /= norm

        self.lognorms.append(np.log(norm))

    def trace(self,):
        self.lognorms.append(np.log(np.trace(self.A)))

    def normlist(self,):
        return np.array(self.lognorms)
    
            

    
    
    
def update(tensor, U):
    ts = tensor.shape
    Us = U.shape
    Us = (int(np.sqrt(Us[0])), int(np.sqrt(Us[0])), Us[1])
    theresult = 0
    for i in range(ts[2]):
        bot = np.reshape(np.transpose(U), (Us[2]*Us[0], Us[1]))       
        bot = np.reshape(np.dot(bot, np.reshape(tensor[:,:,i,:], (ts[0], ts[1]*ts[3]))), (Us[2], Us[0], ts[1], ts[3]))
        bot = np.reshape(np.transpose(bot, (0,3,1,2)), (Us[2]*ts[3], Us[0]*ts[1]))
        
        top = np.reshape(U, (Us[0], Us[1]*Us[2]))
        top = np.reshape(np.dot(np.reshape(np.transpose(tensor[:,:,:,i], (0,2,1)), (ts[0]*ts[2], ts[1])), top), (ts[0], ts[2], Us[1], Us[2]))
        top = np.reshape(np.transpose(top, (0,2,1,3)), (ts[0]*Us[1], ts[2]*Us[2]))
        
        theresult += np.transpose(np.reshape(np.dot(bot, top), (Us[2], ts[3], ts[2], Us[2])), (0,3,2,1)) 
    sys.stdout.flush()
    return theresult

# def update(tensor, Ul, Ur):
#     ts = tensor.shape
#     Us = Ul.shape
#     Us = np.reshape(Ul, (int(sqrt(Us[0])), int(sqrt(Us[0])), Us[1])).shape
#     theresult = 0
#     for i in range(ts[2]):
#         bot = np.reshape(np.transpose(Ul), (Us[2]*Us[0], Us[1]))       
#         bot = np.reshape(np.dot(bot, np.reshape(tensor[:,:,i,:], (ts[0], ts[1]*ts[3]))), (Us[2], Us[0], ts[1], ts[3]))
#         bot = np.reshape(np.transpose(bot, (0,3,1,2)), (Us[2]*ts[3], Us[0]*ts[1]))
        
#         top = np.reshape(Ur, (Us[0], Us[1]*Us[2]))
#         top = np.reshape(np.dot(np.reshape(np.transpose(tensor[:,:,:,i], (0,2,1)), (ts[0]*ts[2], ts[1])), top), (ts[0], ts[2], Us[1], Us[2]))
#         top = np.reshape(np.transpose(top, (0,2,1,3)), (ts[0]*Us[1], ts[2]*Us[2]))
        
#         theresult += np.transpose(np.reshape(np.dot(bot, top), (Us[2], ts[3], ts[2], Us[2])), (0,3,2,1)) 
#     sys.stdout.flush()
#     return theresult


# def symupdate(pure, impure, Uten):
#     pts = pure.shape
#     ipts = impure.shape
#     Us = Uten.shape
#     Us = np.reshape(Uten, (int(sqrt(Us[0])), int(sqrt(Us[0])), Us[1])).shape
#     impbot = 0
#     imptop = 0
#     for i in range(pts[2]):
#         bot = np.reshape(np.transpose(Uten), (Us[2]*Us[0], Us[1]))
#         bot = np.reshape(np.dot(bot, np.reshape(impure[:,:,i,:], (ipts[0], ipts[1]*ipts[3]))), (Us[2], Us[0], ipts[1], ipts[3]))
#         bot = np.reshape(np.transpose(bot, (0,3,1,2)), (Us[2]*ipts[3], Us[0]*ipts[1]))

#         top = np.reshape(Uten, (Us[0], Us[1]*Us[2]))
#         top = np.reshape(np.dot(np.reshape(np.transpose(pure[:,:,:,i], (0,2,1)), (pts[0]*pts[2], pts[1])), top), (pts[0], pts[2], Us[1], Us[2]))
#         top = np.reshape(np.transpose(top, (0,2,1,3)), (pts[0]*Us[1], pts[2]*Us[2]))

#         impbot += np.transpose(np.reshape(np.dot(bot, top), (Us[2], ipts[3], pts[2], Us[2])), (0,3,2,1))

#         bot = np.reshape(np.transpose(Uten), (Us[2]*Us[0], Us[1]))
#         bot = np.reshape(np.dot(bot, np.reshape(pure[:,:,i,:], (pts[0], pts[1]*pts[3]))), (Us[2], Us[0], pts[1], pts[3]))
#         bot = np.reshape(np.transpose(bot, (0,3,1,2)), (Us[2]*pts[3], Us[0]*pts[1]))
                
#         top = np.reshape(Uten, (Us[0], Us[1]*Us[2]))
#         top = np.reshape(np.dot(np.reshape(np.transpose(impure[:,:,:,i], (0,2,1)), (ipts[0]*ipts[2], ipts[1])), top), (ipts[0], ipts[2], Us[1], Us[2]))
#         top = np.reshape(np.transpose(top, (0,2,1,3)), (ipts[0]*Us[1], ipts[2]*Us[2]))

#         imptop += np.transpose(np.reshape(np.dot(bot, top), (Us[2], pts[3], ipts[2], Us[2])), (0,3,2,1))
#     sys.stdout.flush()
#     return (impbot+imptop)

def getU2d(tensor, nums):
    # Assume tensor is organized as left, right, up, down per index
    ts = tensor.shape
    top = np.reshape(np.transpose(tensor, (0,3,2,1)), (ts[0]*ts[3], ts[2]*ts[1]))
    top = np.reshape(np.dot(top, np.transpose(top)), (ts[0], ts[3], ts[0], ts[3]))
    top = np.reshape(np.transpose(top, (0,2,1,3)), (ts[0]*ts[0], ts[3]*ts[3]))

    bot = np.reshape(np.transpose(tensor, (0,2,1,3)), (ts[0]*ts[2], ts[1]*ts[3]))
    bot = np.reshape(np.dot(bot, np.transpose(bot)), (ts[0], ts[2], ts[0], ts[2]))
    bot = np.reshape(np.transpose(bot, (0,2,1,3)), (ts[0]*ts[0], ts[2]*ts[2]))
    
    Q = np.reshape(np.dot(top, np.transpose(bot)), tuple([ts[0]]*4))
    Q = np.reshape(np.transpose(Q, (0,2,1,3)), tuple([ts[0]**2]*2))
    # print Q
    # Q = np.real(Q + np.conjugate(Q))*0.5
    sys.stdout.flush()
    
    O = eigh(Q, eigvals=(ts[0]**2 - nums, ts[0]**2-1))
    # O = eigh(Q)
    e = O[0]
    idx = e.argsort()[::-1]
    print("largest =", np.max(e))
    sys.stdout.flush()
    return (O[1][:,idx])[:,:nums]




def traceupdate(tensor):
    ts = tensor.shape

    top = np.reshape(tensor, (ts[0]*ts[1], ts[2]*ts[3]))
    bot = np.reshape(np.transpose(tensor, (0,1,3,2)), (ts[0]*ts[1], ts[3]*ts[2]))

    bot = np.reshape(np.dot(top, bot.T), (ts[0], ts[1], ts[0], ts[1]))

    sys.stdout.flush()
    return np.reshape(np.transpose(bot, (0,2,1,3)), (ts[0]*ts[0], ts[1]*ts[1]))

# def imptraceupdate(tensor, mimp):
#     """
#     This takes a tensor as while performing the trace-update, it inserts an impure
#     link that represents a link in the polyakov loop between the two tensors.
#     """

#     ts = tensor.shape

#     top = np.reshape(np.dot(np.reshape(tensor, (ts[0]*ts[1]*ts[2], ts[3])), mimp), (ts[0]*ts[1], ts[2]*ts[3]))
#     bot = np.reshape(np.transpose(tensor, (0,1,3,2)), (ts[0]*ts[1], ts[3]*ts[2]))

#     bot = np.reshape(np.dot(top, bot.T), (ts[0], ts[1], ts[0], ts[1]))

#     sys.stdout.flush()
#     return np.reshape(np.transpose(bot, (0,2,1,3)), (ts[0]*ts[0], ts[1]*ts[1]))


# def imptraceupdateopen(tensor, L, D, mimp):
# #     Dbond = 2*D+1
# #     M = np.zeros((Dbond, Dbond))
# #     for x, xp in product(range(-D, D+1), repeat=2):
# #         M[x+D, xp+D] = ttilde(xp-x, 2*kappa)

# #     L = la.cholesky(M)
# #     if not np.allclose(np.dot(L, L.T), M):
# #         raise ValueError("cholesky not reproducing M correctly.")

#     ts = tensor.shape
#     top = np.dot(np.reshape(tensor, (ts[0]*ts[1]*ts[2], ts[3])), L.T[:,D])
#     top = np.reshape(top, (ts[0]*ts[1], ts[2]))
#     top = np.dot(top, mimp)
#     bot = np.reshape(np.transpose(tensor, (0,1,3,2)), (ts[0]*ts[1]*ts[3], ts[2]))
#     bot = np.dot(bot, L.T[:,D])
#     bot = np.reshape(bot, (ts[0]*ts[1], ts[3]))

#     bot = np.reshape(np.dot(top, bot.T), (ts[0], ts[1], ts[0], ts[1]))

#     sys.stdout.flush()
#     return np.reshape(np.transpose(bot, (0,2,1,3)), (ts[0]*ts[0], ts[1]*ts[1]))



# def traceupdateopen(tensor, L, D):

#     ts = tensor.shape
#     top = np.dot(np.reshape(tensor, (ts[0]*ts[1]*ts[2], ts[3])), L.T[:,D])
#     top = np.reshape(top, (ts[0]*ts[1], ts[2]))
#     bot = np.reshape(np.transpose(tensor, (0,1,3,2)), (ts[0]*ts[1]*ts[3], ts[2]))
#     bot = np.dot(bot, L.T[:,D])
#     bot = np.reshape(bot, (ts[0]*ts[1], ts[3]))

#     bot = np.reshape(np.dot(top, bot.T), (ts[0], ts[1], ts[0], ts[1]))

#     sys.stdout.flush()
#     return np.reshape(np.transpose(bot, (0,2,1,3)), (ts[0]*ts[0], ts[1]*ts[1]))





# def traceupdateopen1(tensor, L, D):
# #     Dbond = 2*D+1
# #     M = np.zeros((Dbond, Dbond))
# #     for x, xp in product(range(-D, D+1), repeat=2):
# #         M[x+D, xp+D] = ttilde(xp-x, 2*kappa)

# #     L = la.cholesky(M)
# #     if not np.allclose(np.dot(L, L.T), M):
# #         raise ValueError("cholesky not reproducing M correctly.")

#     ts = tensor.shape
#     top = np.dot(np.reshape(tensor, (ts[0]*ts[1]*ts[2], ts[3])), L.T[:,D+1])
#     top = np.reshape(top, (ts[0]*ts[1], ts[2]))
#     bot = np.reshape(np.transpose(tensor, (0,1,3,2)), (ts[0]*ts[1]*ts[3], ts[2]))
#     bot = np.dot(bot, L.T[:,D])
#     bot = np.reshape(bot, (ts[0]*ts[1], ts[3]))

#     bot = np.reshape(np.dot(top, bot.T), (ts[0], ts[1], ts[0], ts[1]))

#     sys.stdout.flush()
#     return np.reshape(np.transpose(bot, (0,2,1,3)), (ts[0]*ts[0], ts[1]*ts[1]))

# def imptraceupdateopen(tensor, mimp, L, D):
#     ts = tensor.shape

#     top = np.reshape(np.dot(np.reshape(tensor, (ts[0]*ts[1]*ts[2], ts[3])), L.T[:,D]), (ts[0]*ts[1], ts[2]))
#     top = np.dot(top, mimp)
#     bot = np.dot(np.reshape(np.transpose(tensor, (0,1,3,2)), (ts[0]*ts[1]*ts[3], ts[2])), L.T[:,D])
#     bot = np.reshape(bot, (ts[0]*ts[1], ts[3]))
#     bot = np.reshape(np.dot(top, bot.T), (ts[0], ts[1], ts[0], ts[1]))

#     sys.stdout.flush()
#     return np.reshape(np.transpose(bot, (0,2,1,3)), (ts[0]*ts[0], ts[1]*ts[1]))


# def symtraceupdate(tensor1, tensor2):
#     ts = tensor1.shape

#     top = np.reshape(tensor1, (ts[0]*ts[1], ts[2]*ts[3]))
#     bot = np.reshape(np.transpose(tensor2, (0,1,3,2)), (ts[0]*ts[1], ts[3]*ts[2]))

#     impbot = np.reshape(np.dot(top, bot.T), (ts[0], ts[1], ts[0], ts[1]))

#     top = np.reshape(tensor2, (ts[0]*ts[1], ts[2]*ts[3]))
#     bot = np.reshape(np.transpose(tensor1, (0,1,3,2)), (ts[0]*ts[1], ts[3]*ts[2]))

#     imptop = np.reshape(np.dot(top, bot.T), (ts[0], ts[1], ts[0], ts[1]))
#     sys.stdout.flush()
#     return np.reshape(np.transpose(imptop+impbot, (0,2,1,3)), (ts[0]*ts[0], ts[1]*ts[1]))

