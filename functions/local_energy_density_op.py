from common_imports import *

def local_energy_density_op(H: SparsePauliOp, n: int, split_identity: bool = False) -> SparsePauliOp:
    """
    Constructs h_n from H by distributing each Pauli term equally among qubits.

    Qiskit convention: Pauli label rightmost char is qubit 0.
    """
    N = H.num_qubits
    if not(0 <= n < N):
        raise ValueError(f"n must be in [0, {N-1}]")

    terms = []
    for label, coeff in H.to_list(): # .to_list() makes labels like "IIZXI..."
        # Finding which qubits the terms acts on
        support = [q for q in range(N) if label[-1 - q] != "I"] # q=0 is the rightmost char
        if len(support) == 0:
            # Then we have a pure identity term (global energy shift)
            if split_identity:
                # Distributing equally across all sites
                terms.append((label, coeff / N))
            continue

        if n in support:
            terms.append((label, coeff / len(support)))

    
    if not terms:
        # Returns explicit zero operator with correct width
        return SparsePauliOp.from_list([("I"*N, 0.0)])

    return SparsePauliOp.from_list(terms).simplify()