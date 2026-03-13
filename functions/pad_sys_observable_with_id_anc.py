from common_imports import *

def pad_sys_observable_with_id_anc(obs_sys: SparsePauliOp, n_tot: int, n_sys: int, anc_first: bool = True) -> SparsePauliOp:
    """
    Embeds a system-only observable into the full circuit space by tensoring identities
    on ancillas.
    """
    n_anc = n_tot - n_sys
    if n_anc < 0:
        raise ValueError("n_tot must be >= n_sys")
    if n_anc == 0:
        return obs_sys

    I_anc = SparsePauliOp.from_list([("I" * n_anc, 1.0)])

    return (I_anc.tensor(obs_sys) if anc_first else obs_sys.tensor(I_anc)).simplify()