from sys import argv
from enum import Enum
import numpy as np
import itertools

# stores the amplitudes of all possible quantum states, like |000> and |101> for a 3 qubit state
q_state = []
# the number of qubits q_state represents
n = 0

# GateType - an enum for the various supported gates of our QASM simulator
class GateType(Enum):
    IDENTITY = 0
    HADAMARD = 1
    PAULI_X = 2
    PAULI_Y = 3
    PAULI_Z = 4
    ROTATE_X = 5
    ROTATE_Y = 6
    ROTATE_Z = 7
    UNITARY = 8

# Gate - a simple data type that stores the operator and operands of each gate
#   gtype       - (enum GateType) for each supported quantum gate
#   target      - (int) the index of the target qubit, also the operand of single qubit gates
#   is_control  - (bool) False if no control qubit, True otherwise
#   control     - (int) the index of the control qubit, -1 if is_control is False
#   parameters  - (List) stores the various parameters for a UNITARY or rotate gate, None otherwise
class Gate:
    def __init__(self, gtype, target, is_control, control, parameters):
        self.gtype = gate_type
        self.target = target
        self.is_control = is_control
        self.control = control
        self.parameters = parameters

# init_state() - Initializes the simulated quantum state with the |0> state on each qubit
#                Uses the n global variable, which stores the number of represented qubits
#   Effect: initializes q_state with 2^n elements, first element is the |00...0> state
#   Return: nothing
def init_state():
    q_state = np.zeros(2 ** n, dtype=complex)
    q_state[0] = 1 + 0j

# getGateType() - returns the matrix representation of a quantum gate
#   Parameter: gate_type - (enum GateType) a quantum gate
#   Return: (2 dimensional np.array) the matrix representation of gate_type
def getGateMatrix(gate_type):
    if gate_type == GateType.IDENTITY:
        return np.array([[1, 0],
                         [0, 1]])
    elif gate_type == GateType.HADAMARD:
        return (1. / np.sqrt(2)) * np.array([[1,  1],
                                             [1, -1]])
    else:
        raise Exception("getGateMatrix(): unsupported Gate operation (" + str(gate_type) + ")")

# applySingleGate() - Applies a single-qubit gate to the quantum state
#   Parameters: gate - (type Gate) the gate to apply
#   Effect: updates q_state with the gate operation
#   Return: nothing
def applySingleGate(gate):
    target_index = gate.target
    if gate.is_control == True:
        raise Exception("applySingleGate(): must use applyCGate() for control gates")
    if target_index >= n:
        raise Exception("applySingleGate(): target (" + str(target_index) 
        + ") must be less than n(" + str(n) + ")")
    # Select the gate
    gate_matrix = getGateMatrix(gate.gtype)

    # identity matrix
    identity = np.array([[1, 0],
                         [0, 1]])

    # transform gate into the 2^n by 2^n matrix
    resultant_matrix = identity if target_index < n - 1 else gate_matrix

    for _ in range(n - target_index - 1): # pad operator from the front
        resultant_matrix = np.kron(resultant_matrix, identity)
    
    if target_index < n - 1:
        resultant_matrix = np.kron(resultant_matrix, gate_matrix)

    for _ in range(target_index): # pad rest of the gate
        resultant_matrix = np.kron(resultant_matrix, identity)

    # matrix multiply with the quantum state
    q_state = np.matmul(resultant_matrix, q_state)


if __name__ == '__main__':
    main()
