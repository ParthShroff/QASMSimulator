from sys import argv
from enum import Enum
import numpy as np
from itertools import combinations  # this is for testing only

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

    def getGType(self):
        return self.gtype

    def getControl(self):
        return self.is_control

    def setTarget(self, target):
        self.target = target

    def setControl(self, control):
        self.control = control


class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def getType(self):
        return self.type

    def getValue(self):
        return self.value

class Type(Enum):
    INV = 1
    QREG = 2
    CREG = 3
    GATE = 4
    QUBIT = 5
    CBIT = 6
    MEASURE = 7
    ARROW = 8
    OP = 9
    LPARAN = 10
    RPARAN = 11
    CONST = 12

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
        return (1. / np.sqrt(2)) * np.array([[1, 1],
                                             [1, -1]], dtype=complex)
    elif gate_type == GateType.PAULI_X:
        return np.array([[0, 1],
                         [1, 0]], dtype=complex)
    elif gate_type == GateType.PAULI_Y:
        return np.array([[0, -1j],
                         [1j, 0]], dtype=complex)
    elif gate_type == GateType.PAULI_Z:
        return np.array([[1, 0],
                         [0, -1]], dtype=complex)
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

    for _ in range(n - target_index - 2):  # pad operator from the front
        resultant_matrix = np.kron(resultant_matrix, identity)

    if target_index < n - 1:
        resultant_matrix = np.kron(resultant_matrix, gate_matrix)

    for _ in range(target_index):  # pad rest of the gate
        resultant_matrix = np.kron(resultant_matrix, identity)

    # matrix multiply with the quantum state
    global q_state
    q_state = np.matmul(resultant_matrix, q_state)


# applyCGate() - Applies a control gate to the quantum state
#   Parameters: gate - (type Gate) the gate to apply
#               target - (int) the qubit index to apply the gate to
# #   Effect: updates q_state with the gate operation
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
        raise Exception("applyCGate(): target (" + str(target_index) + ") != control (" + str(control_index) + ")")

    # Select the gate
    gate_matrix = getGateMatrix(gate.gtype)  # gate with GateType PAULI_X and is_control=True == CNOT

    # useful matrices
    identity = np.array([[1, 0],
                         [0, 1]])
    zero_proj = np.array([[1, 0],  # |0><0|
                          [0, 0]])
    one_proj = np.array([[0, 0],  # |1><1|
                         [0, 1]])

    # transform gate into the 2^n by 2^n matrix
    # based on the logic provided by:
    # https://quantumcomputing.stackexchange.com/questions/4252/how-to-derive-the-cnot-matrix-for-a-3-qubit-system-where-the-control-target-qu/4254

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
    one_proj_term = np.kron(one_proj_term, gate_matrix if control_index > target_index else one_proj)

    # pad the rest of the qubits
    for _ in range(min(target_index, control_index)):  # pad rest of the gate
        zero_proj_term = np.kron(zero_proj_term, identity)
        one_proj_term = np.kron(one_proj_term, identity)

    resultant_matrix = zero_proj_term + one_proj_term

    # matrix multiply with the quantum state
    global q_state
    q_state = np.matmul(resultant_matrix, q_state)

def tokenizer(inputLine):
    tokenList = []
    gates = ['h', 'x', 't', 'tdg', 'sdg', 's', 'z', 'p', 'rz', 'rx', 'ry', 'rxx', 'rzz', 'sx', 'sxdg', 'id', 'cx']
    splited = customDelim(inputLine)
    for token in splited:
        if token in gates:
            gateObj = parseGate(token)
            newToken = Token(Type.GATE, gateObj)
        elif token == "->":
            newToken = Token(Type.ARROW, token)
        elif token == "qreg":
            newToken = Token(Type.QREG, token)
        elif token == "creg":
            newToken = Token(Type.CREG, token)
        elif token == "pi" or token.isnumeric():
            if token == "pi":
                newToken = Token(Type.CONST, np.pi)
            else:
                newToken = Token(Type.CONST, int(token))
        elif token[0] == 'q' and token[1] == '[' and token[2].isnumeric() and token[3] == ']':
            if tokenList[len(tokenList)-1].getType() == Type.GATE and not(tokenList[len(tokenList)-1].getValue().getControl()):
                tokenList[len(tokenList) - 1].getValue().setTarget(int(token[2]))
            elif tokenList[len(tokenList)-1].getType() == Type.GATE and tokenList[len(tokenList)-1].getValue().getControl():
                tokenList[len(tokenList) - 1].getValue().setControl(int(token[2]))
            elif tokenList[len(tokenList) - 1].getType() == Type.QUBIT and tokenList[len(tokenList) - 2].getType() == Type.GATE and tokenList[len(tokenList) - 2].getValue().getControl():
                tokenList[len(tokenList) - 2].getValue().setTarget(int(token[2]))
            elif tokenList[len(tokenList) - 1].getType() == Type.GATE:
                tokenList[len(tokenList) - 1].getValue().setTarget(int(token[2]))
            newToken = Token(Type.QUBIT, token[2])

        elif token[0] == 'c' and token[1] == '[' and token[2].isnumeric() and token[3] == ']':
            newToken = Token(Type.CBIT, token[2])
        elif token == "measure":
            newToken = Token(Type.MEASURE, token)
        elif token == '+' or token == '-' or token == '*' or token == '/':
            newToken = Token(Type.OP, token)
        elif token == '(' or token == ')':
            if token == '(':
                newToken = Token(Type.LPARAN, token)
            else:
                newToken = Token(Type.RPARAN, token)
        else:
            newToken = Token(Type.INV, token)

        tokenList.append(newToken)

    return tokenList

def parseGate(token):

    if token == 'id':
        newGate = Gate(GateType.IDENTITY, -1, False, -1, None)
    elif token == 'h':
        newGate = Gate(GateType.HADAMARD, -1, False, -1, None)
    elif token == 'x':
        newGate = Gate(GateType.PAULI_X, -1, False, -1, None)
    elif token == 'y':
        newGate = Gate(GateType.PAULI_Y, -1, False, -1, None)
    elif token == 'z':
        newGate = Gate(GateType.PAULI_Z, -1, False, -1, None)
    elif token == 'cx':
        newGate = Gate(GateType.PAULI_X, -1, True, -1, None)

    return newGate

def customDelim(input):
    inputString = input
    for delim in ',;()':
        inputString = inputString.replace(delim, ' ')
    return inputString.split()

def result(filepath, shots):
    with open(filepath) as fp:
        for line in fp:
            curTokList = tokenizer(line)
            for i in range(len(curTokList)):
                tok = curTokList[i]
                if tok.getType() == Type.INV:
                    continue
                elif tok.getType() == Type.QREG:
                    global n
                    n = int(curTokList[i + 1].getValue())
                    init_state()
                elif tok.getType() == Type.GATE:
                    if tok.getValue().getControl():
                        applyCGate(tok.getValue())
                    else:
                        applySingleGate(tok.getValue())
                elif tok.getType() == Type.MEASURE and curTokList[i + 2].getType() == Type.ARROW:
                    print("   Final state: " + str(list(np.round(q_state, 3))))
                else:
                    continue

# For simple by-hand testing:
def main():
    if len(argv) < 3:
        print(f"usage: {argv[0]} <file>")
    filepath = argv[1]
    shots = argv[2]
    result(filepath, shots)

if __name__ == '__main__':
    main()
