__author__ = 'gregor'
import term_document_matrix as tdm
import numpy

tdm.parseDocument("resources/")

# crs = tdm.load_sparse_csr("resources/csv/reut2-001.npz")
# print(crs.toarray())
# numpy.savetxt("foo.csv", crs.toarray(), delimiter=",")