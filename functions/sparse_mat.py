from common_imports import *

def sparse_mat(op: SparsePauliOp) -> sp.csr_matrix: # converts a SparsePauliOp to a matrix
    return op.to_matrix(sparse=True)