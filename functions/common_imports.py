# For circuit setup
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
#from qiskit_ibm_runtime.circuit import MidCircuitMeasure
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate

# For transpilation
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# For backends and running circuits via Estimator or Sample
#from qiskit_ibm_runtime import EstimatorV2 as Estimator
#from qiskit_ibm_runtime import SamplerV2 as Sampler
#from qiskit_ibm_runtime import QiskitRuntimeService

# For math and data analysis
import math
import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from typing import Optional, Union
