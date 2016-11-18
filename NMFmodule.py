import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

class NMF:

    # r is the rang we want
    def __init__(self, A, r, eps=10 ** -4, T=500, **kwargs):
        self.rank = r
        self.tol = eps
        self.maxiter = T
        self.set = False

        try:
            self.A = sp.matrix(A)
        except ValueError("Matrix incorrectly defined."):
            exit()
        except:
            exit("Unknow error occured.")

        m, n = self.A.shape

        if "seed" in kwargs.keys():
            self.seed = kwargs["seed"]
        else:
            self.seed = False

        if "num" in kwargs.keys():
            self.num = kwargs["num"]  # koliko puta zelimo ponoviti postupka sa slucajno geneririranim matricama
        else:
            self.num = 1

        if "W0" in kwargs.keys():
            try:
                self.W0 = sp.matrix(kwargs["W0"])
                if (m, r) != self.W0.shape:
                    raise ValueError
                else:
                    self.set = True
            except:
                self.W0 = sp.maximum(sp.matrix(sp.random.normal(size=(m, r))), 0)
        else:
            self.W0 = sp.maximum(sp.matrix(sp.random.normal(size=(m, r))), 0)

        if "H0" in kwargs.keys():
            try:
                self.H0 = sp.matrix(kwargs["H0"])
                if (r, n) != H0.shape:
                    raise ValueError
                else:
                    self.set = True
            except:
                self.H0 = sp.maximum(sp.matrix(sp.random.normal(size=(r, n))), 0)
        else:
            self.H0 = sp.maximum(sp.matrix(sp.random.normal(size=(r, n))), 0)

        if "rw" in kwargs.keys():
            self.rw = rw
        else:
            self.rw = 1

    # ------------------------------------

    # ------------------------------------


    def reload(self):

        if self.set:
            return False
        else:
            m, n = self.A.shape
            self.W0 = sp.maximum(sp.matrix(sp.random.normal(size=(m, self.rank))), 0)
            self.H0 = sp.maximum(sp.matrix(sp.random.normal(size=(self.rank, n))), 0)
        return True

    # -----------------------------------

    def factorize(self):
        ''' comment how the function works '''

        # preprocessing

        rep = {
            "hist_obj": [],
            "relerr1": [],
            "relerr2": [],
            "iterator": 0
        }

        Anorm = np.linalg.norm(self.A, ord="fro")
        W0 = sp.matrix(self.W0 / np.linalg.norm(self.W0, ord="fro") * np.sqrt(Anorm))
        H0 = sp.matrix(self.H0 / np.linalg.norm(self.H0, ord="fro") * np.sqrt(Anorm))
        Wm = W0
        Hm = H0

        Ht = H0.T
        Hs = H0 * Ht
        AHt = self.A * Ht

        obj0 = 0.5 * Anorm ** 2
        nstall = 0
        t0 = 1
        Lw = 1
        Lh = 1

        rw = self.rw

        # iterations
        # comment on how iterations work

        for i in range(self.maxiter):

            # updating W

            Lw0 = Lw
            Lw = np.linalg.norm(Hs)
            Gw = Wm * Hs - AHt
            W = np.maximum(0, Wm - Gw / Lw)
            Wt = W.T
            Ws = Wt * W

            # updating H

            Lh0 = Lh
            Lh = np.linalg.norm(Ws)
            Gh = (Ws * Hm) - (Wt * self.A)
            H = np.maximum(0, Hm - Gh / Lh)
            Ht = H.T
            Hs = H * Ht
            AHt = self.A * Ht

            # reporting

            obj = 0.5 * (
                sum(sum(sp.array(sp.multiply(Ws, Hs)))) - 2 * sum(sum(sp.array(sp.multiply(W, AHt)))) + Anorm ** 2)
            # debug

            rep["hist_obj"].append(float(obj))
            rep["relerr1"].append(float(sp.absolute(obj - obj0) / (obj0 + 1)))
            rep["relerr2"].append(float(sp.sqrt(2 * obj) / Anorm))

            # stop check

            # debug

            crit = rep["relerr1"][-1] < self.tol
            if crit:
                nstall += 1
            else:
                nstall = 0

            if nstall >= 3 or rep["relerr2"][-1] < self.tol:
                break

            # correction
            t = (1 + sp.sqrt(1 + 4 * t0 ** 2)) / 2
            if obj > obj0:
                Wm = W0
                Hm = H0
                Ht = H0.T
                Hs = H0 * Ht
                AHt = self.A * Ht
            else:
                w = (t0 - 1) / t
                ww = min(w, rw * sp.sqrt(Lw0 / Lw))
                wh = min(w, rw * sp.sqrt(Lh0 / Lh))
                Wm = W + ww * (W - W0)
                Hm = H + wh * (H - H0)
                W0 = W
                H0 = H
                t0 = t
                obj0 = obj

        rep["iterator"] = i
        self.rep = rep
        self.W = W
        self.H = H

    # -----------------------------------

    def graphic(self):
        report = self.rep
        n = len(report["hist_obj"])

        plt.plot(report["hist_obj"], range(n), c="red")
        plt.plot(report["relerr1"], range(n), c="blue")
        plt.plot(report["relerr2"], range(n), c="green")

        plt.show()

    # ------------------------------------

    def start(self):

        self.factorize()
        num = self.num
        temp = self.rep["hist_obj"][-1]
        W0 = self.W0
        H0 = self.H0

        curri = 0

        if not (num <= 1 or self.set):
            for i in range(1, num):
                self.reload()
                self.factorize()

                if self.rep["hist_obj"][-1] < temp:
                    W0 = self.W0
                    H0 = self.H0
                    temp = self.rep["hist_obj"][-1]

            self.W0 = W0
            self.H0 = H0

            self.factorize()

        self.graphic()

# np.linalg.norm(ord = "fro")
