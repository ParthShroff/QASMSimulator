from sys import argv
from enum import Enum
import numpy as np
from itertools import combinations #this is for testing only

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
        self.gtype = gtype
        self.target = target
        self.is_control = is_control
        self.control = control
        self.parameters = parameters


# init_state() - Initializes the simulated quantum state with the |0> state on each qubit
#                Uses the n global variable, which stores the number of represented qubits
#   Effect: initializes q_state with 2^n elements, first element is the |00...0> state
#   Return: nothing
def init_state():
    if n <= 0 or type(n) != int:
        raise Exception("init_state(): n must be initialized with a positive integer")
    global q_state
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
        + ") must be less than n (" + str(n) + ")")
    # Select the gate
    gate_matrix = getGateMatrix(gate.gtype)

    # identity matrix
    identity = np.array([[1, 0],
                         [0, 1]])

    # transform gate into the 2^n by 2^n matrix: G_i = I x ... x I x G x I x ... x I x I
    resultant_matrix = identity if target_index < n - 1 else gate_matrix

    for _ in range(n - target_index - 2): # pad operator from the front
        resultant_matrix = np.kron(resultant_matrix, identity)
    
    if target_index < n - 1:
        resultant_matrix = np.kron(resultant_matrix, gate_matrix)

    for _ in range(target_index): # pad rest of the gate
        resultant_matrix = np.kron(resultant_matrix, identity)

    # matrix multiply with the quantum state
    global q_state
    q_state = np.matmul(resultant_matrix, q_state)

# applyCGate() - Applies a control gate to the quantum state
#   Parameters: gate - (type Gate) the gate to apply
#               target - (int) the qubit index to apply the gate to
#   Effect: updates q_state with the gate operation
#   Return: nothing
def applyCGate(gate):
    target_index = gate.target
    control_index = gate.control
    if gate.is_control == False:
        raise Exception("applyCGate(): gate operation must be a control gate")
    if target_index >= n:
        raise Exception("applyCGate(): target (" + str(target_index) + ") must be less than n (" + str(n) + ")")
    if control_index >= n or control_index < 0:
        raise Exception("applyCGate(): control (" + str(control_index) + ") must be nonnegative")
    if target_index == control_index:
        raise Exception("applyCGate(): target (" + str(target_index) + ") != control ("+ str(control_index) + ")")

    # Select the gate
    gate_matrix = getGateMatrix(gate.gtype) # gate with GateType PAULI_X and is_control=True == CNOT

    # useful matrices
    identity = np.array([[1, 0],
                         [0, 1]])
    zero_proj = np.array([1, 0], # |0><0|
                         [0, 0])
    one_proj  = np.array([0, 0], # |1><1|
                         [0, 1])

    # transform gate into the 2^n by 2^n matrix
    # based on the logic provided by: 
    # https://quantumcomputing.stackexchange.com/questions/4252/how-to-derive-the-cnot-matrix-for-a-3-qubit-system-where-the-control-target-qu/4254#4254

    # the switch factor is either the 'control' or the 'target'
    zero_proj_switch_factor = zero_proj if control_index > target_index else identity
    one_proj_switch_factor = one_proj if control_index > target_index else gate_matrix

    # initial term value
    zero_proj_term = identity if max(target_index, control_index) < n - 1 else zero_proj_switch_factor
    one_proj_term = identity if max(target_index, control_index) < n - 1 else one_proj_switch_factor

    # padding for qubits before either the target or control
    for _ in range(n - max(target_index, control_index) - 2):
        zero_proj_term = np.kron(zero_proj_term, identity)
        one_proj_term = np.kron(one_proj_term, identity)

    # multiply the switch factor if we didn't already
    if max(target_index, control_index) < n - 1:
        zero_proj_term = np.kron(zero_proj_term, zero_proj_switch_factor)
        one_proj_term = np.kron(one_proj_term, one_proj_switch_factor)

    # padding for qubits between the control and target
    for _ in range(max(target_index, control_index) - min(target_index, control_index) - 1):
        zero_proj_term = np.kron(zero_proj_term, identity)
        one_proj_term = np.kron(one_proj_term, identity)

    # multiply the opposite switch factor, target or control
    zero_proj_term = np.kron(zero_proj_term, identity if control_index > target_index else zero_proj)
    one_proj_term = np.kron(one_proj_term, identity if control_index > target_index else one_proj)

    # pad the rest of the qubits
    for _ in range(max(target_index, control_index)): # pad rest of the gate
        zero_proj_term = np.kron(zero_proj_term, identity)
        one_proj_term = np.kron(one_proj_term, identity)

    resultant_matrix = zero_proj_term + one_proj_term

    # matrix multiply with the quantum state
    global
    q_state = np.matmul(resultant_matrix, q_state)


# For simple by-hand testing:
def main():
    global n
    #TODO make test cases for CNOT

    # single qubit gate, multiple positions test
    # n_max = 4 # total qubits to test, 1 through n_max
    # test_gates = [GateType.IDENTITY, GateType.HADAMARD] # the gates to test
    # for i in range(1, n_max + 1):
    #     n = i
    #     print(str(i) + "-Qubit State Test")
    #     for test_gate in test_gates:
    #         print("    " + str(test_gate))
    #         for j in range(1, i+1):
    #             combination = list(combinations(list(range(i)), j))

    #             for k in range(len(combination)):
    #                 print("        Applied to: " + str(combination[k]))
    #                 init_state()
    #                 print("            Initial state: " + str(list(np.round(q_state, 3))))
    #                 for l in combination[k]:
    #                     gate = Gate(test_gate, l, False, -1, None)
    #                     applySingleGate(gate)
    #                 print("            Final state: " + str(list(np.round(q_state, 3))))


if __name__ == '__main__':
    main()
