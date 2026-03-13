from common_imports import *

from common_imports import *

def get_schwinger_hamiltonian(L: int, m: float, e: float, a: float, include_shifts: bool = True) -> SparsePauliOp:
    if L <= 0:
        raise ValueError("L must be >= 1.")
    if a == 0:
        raise ValueError("a must be nonzero.")

    terms = []

    # Kinetic term
    ## H_kin = sum_{i=0}^{L-2} (1/(4a)) (X_i X_{i+1} + Y_i Y_{i+1})
    kin_coeff = 1.0 / (4.0 * a)
    for i in range(L - 1):
        terms.append(("XX", [i, i + 1], kin_coeff))
        terms.append(("YY", [i, i + 1], kin_coeff))

    # Mass term
    ## H_mass = sum_i m * (-1)^i * (Z_i + I)/(2a)
    for i in range(L):
        coeff = m * ((-1) ** i) / (2.0 * a)
        terms.append(("Z", [i], coeff))
        if include_shifts: # global energy shift
            terms.append(("I", [0], coeff))

    # Electric term
    ## E_i = (e/2) (S_i + C_i),  S_i = sum_{j<i} Z_j,  C_i = sum_{j<i} (-1)^j
    ## E_i^2 = (e^2/4) [ S_i^2 + 2 C_i S_i + C_i^2 ]
    ## S_i^2 = i*I + 2 sum_{j<k<i} Z_j Z_k
    base_I = (a / 2.0) * (e**2 / 4.0)
    zz_coeff = (a * e**2) / 4.0
    z_per_link_unit = (a * e**2) / 4.0

    for i in range(1, L + 1):
        # C_i = sum_{j=0}^{i-1} (-1)^j = 1 if i is odd else 0
        C = i % 2

        # Identity from i*I + C^2*I
        if include_shifts:
            terms.append(("I", [0], base_I * (i + C))) # since C^2 = C for C in {0, 1}

        # linear Z terms: (a/2)*(e^2/4)*(2 C) * sum_{j<i} Z_j = (a e^2/4)*C * sum_{j<i} Z_j
        if C == 1:
            for j in range(i):
                terms.append(("Z", [j], z_per_link_unit))

        # ZZ terms: (a/2)*(e^2/4)*(2) * sum_{j<k<i} Z_j Z_k = (a e^2/4) * sum_{j<k<i} Z_j Z_k
        for j in range(i):
            for k in range(j + 1, i):
                terms.append(("ZZ", [j, k], zz_coeff))

    H = SparsePauliOp.from_sparse_list(terms, num_qubits=L)
    return H.simplify()