from common_imports import *

def build_lindblad_cycle_circuit(
    H_sys: SparsePauliOp,
    L_ops: list[SparsePauliOp],
    dt: float,
    n_cycles: int,
    measure_system: bool = True,
    init_state: Optional[InitSpec] = None,
) -> QuantumCircuit:
    """
    U_J(k) = exp(-i * J * sqrt(dt)) on (aux_block_k + system)
    U_H    = exp(-i * H_sys * dt)   on system

    At the end, measuring only system qubits (ancillas are traced out).
    """
    n_sys = H_sys.num_qubits
    m = len(L_ops)
    n_aux = math.ceil(math.log2(m + 1)) # aux dimension is m+1

    sys = QuantumRegister(n_sys, "s")
    anc = QuantumRegister(n_cycles * n_aux, "a") # ancillas up front

    if measure_system:
        c_sys = ClassicalRegister(n_sys, "cs")
        qc = QuantumCircuit(sys, anc, c_sys)
    else:
        qc = QuantumCircuit(sys, anc)

    apply_init_to_system(qc, sys, init_state)
    
    J = build_J_operator(L_ops, n_sys = n_sys)

    for k in range(n_cycles):
        start = k * n_aux
        aux_block = [anc[start + i] for i in range(n_aux)]

        # exp(-i * sqrt(dt) * J) on [aux_block] + [sys]
        qc.append(
            PauliEvolutionGate(J, time=np.sqrt(dt)),
            aux_block + list(sys) # [ a[start], a[start+1], ..., a[start+n_aux-1], s[0], s[1], ..., s[n_sys-1] ]
        )

        # exp(-i * dt * H_sys) on system
        qc.append(
            PauliEvolutionGate(H_sys, time=dt),
            list(sys)
        )

    if measure_system:
        qc.measure(sys, c_sys)

    return qc