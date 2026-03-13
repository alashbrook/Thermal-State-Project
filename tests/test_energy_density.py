import numpy as np
from qiskit.quantum_info import SparsePauliOp

from get_schwinger_hamiltonian import get_schwinger_hamiltonian
from local_energy_density_op import local_energy_density_op


def test_local_energy_density_matches_hamiltonian_up_to_identity_shift():
    L = 4
    a = 1.0
    m = 0.5
    e = 0.71

    H = get_schwinger_hamiltonian(L, m, e, a)

    hsum = sum(
        (local_energy_density_op(H, n) for n in range(H.num_qubits)),
        SparsePauliOp.from_list([("I" * H.num_qubits, 0.0)])
    )

    diff = (H - hsum).simplify()

    if len(diff.coeffs) == 0:
        assert True
        return

    labels = diff.paulis.to_labels()
    coeffs = diff.coeffs

    identity_label = "I" * H.num_qubits

    for label, coeff in zip(labels, coeffs):
        if label != identity_label:
            assert abs(coeff) < 1e-10, f"Found non-identity leftover term: {label} with coeff {coeff}"