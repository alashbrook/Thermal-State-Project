from common_imports import *
from ketbra_from_ints import ketbra_from_ints

def build_J_operator(L_ops: list[SparsePauliOp], n_sys: int) -> SparsePauliOp:
    """
    J = sum_{j=1}^m ( |j><0| ⊗ L_j + |0><j| ⊗ L_j^† )
    """
    m = len(L_ops)
    n_aux = math.ceil(math.log2(m+1))
    if (1 << n_aux) < (m + 1):
        raise ValueError("n_aux sizing failed")

    total_qubits = n_aux + n_sys
    J = SparsePauliOp.from_list([("I" * total_qubits, 0.0)])

    for j, Lj in enumerate(L_ops, start=1):
        if j > m:
            break

        ket_j_bra_0 = ketbra_from_ints(j, 0, n_aux) # |j><0|
        ket_0_bra_j = ketbra_from_ints(0, j, n_aux) # |0><j|

        Lj_dag = Lj.adjoint()

        J += ket_j_bra_0.tensor(Lj)
        J += ket_0_bra_j.tensor(Lj_dag)

    return J.simplify()