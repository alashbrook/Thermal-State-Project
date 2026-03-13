from common_imports import *
from sparse_mat import sparse_mat

def build_liouvillian(H_op: SparsePauliOp, L_ops: list[SparsePauliOp], D: np.ndarray, a_lat: float) -> sp.csr_matrix:
    H = sparse_mat(H_op).tocsr()
    d = H.shape[0]
    I = sp.identity(d, format="csr")

    # -i [H, ρ]  ->  -i (I⊗H - H^T⊗I) vec(ρ)
    Lsuper = -1j * (sp.kron(I, H, format = "csr") - sp.kron(H.T, I, format = "csr"))

    Lmats = [sparse_mat(Lop).tocsr() for Lop in L_ops] # Lindblad matrices

    # Dissipator: a^2 sum_{n1,n2} D_{n1n2} ( L(n2) ρ L†(n1) - 1/2 {L†(n1)L(n2), ρ} )
    pref = (a_lat**2)
    for n1 in range(len(Lmats)):
        L1 = Lmats[n1]
        L1dag = L1.getH()
        L1star = L1.conjugate() # (L†)^T == L*

        for n2 in range(len(Lmats)):
            w = pref * D[n1, n2]
            if w == 0.0:
                continue

            L2 = Lmats[n2]
            M = (L1dag @ L2).tocsr()

            term_jump = sp.kron(L1star, L2, format = "csr")
            term_ac1 = sp.kron(I, M, format = "csr")
            term_ac2 = sp.kron(M.T, I, format = "csr")

            Lsuper += w * (term_jump - 0.5 * term_ac1 - 0.5*term_ac2)

    return Lsuper.tocsr()