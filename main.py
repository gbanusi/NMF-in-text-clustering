__author__ = 'gregor'
import term_document_matrix as tdm
import numpy
import scipy as sp

# tdm.parseDocument("resources/")

r = 20
n = 500
H0 = numpy.array(sp.maximum(sp.matrix(sp.random.normal(size=(r, n))), 0))

tdm.results(H0, [numpy.random.randint(1, 20) for i in range(0, n)])

# crs = tdm.load_sparse_csr("resources/csv/reut2-001.npz")
# print(crs.toarray())
# numpy.savetxt("foo.csv", crs.toarray(), delimiter=",")