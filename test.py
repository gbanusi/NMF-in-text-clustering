import scipy as sp

import NMFmodule as nm

x = sp.absolute(sp.random.normal(size=(15, 12)))

a = nm.NMF(x, 8, num=100, seed=123)
a.factorize()
a.start()

print(a.W)
print("\n")
print(a.H)
