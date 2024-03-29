from sys import argv
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
import random
import itertools
import warnings
import argparse

# disables the ComplexWarning message
warnings.filterwarnings('ignore')

# stores the amplitudes of all possible quantum states, like |000> and |101> for a 3 qubit state
q_state = []
# the number of qubits q_state represents
n = 0

# number of shots to make
shots = 0
# internal parameter for noisy gates: p = 0 unchanged q_state, p = 1 completely changed q_state
p = 0.03

# flag parameters
is_noisy = False
verbose = False
show_prob_graph = False
show_shots_graph = False


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

    def setParams(self, params):
        self.parameters = params


class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def getType(self):
        return self.type

    def getValue(self):
        return self.value


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
def getGateMatrix(gate):
    gate_type = gate.gtype
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
    elif gate_type == GateType.ROTATE_X:
        theta = gate.parameters[0]
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                         [1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    elif gate_type == GateType.ROTATE_Y:
        theta = gate.parameters[0]
        return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                         [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    elif gate_type == GateType.ROTATE_Z:
        phi = gate.parameters[0]
        return np.array([[np.exp(-1j * phi / 2), 0],
                         [0, np.exp(1j * phi / 2)]], dtype=complex)
    elif gate_type == GateType.UNITARY:
        theta = gate.parameters[0]
        phi = gate.parameters[1]
        lmbda = gate.parameters[2]
        first = np.cos(theta / 2)
        second = -np.exp(1j * lmbda) * np.sin(theta / 2)
        third = np.exp(1j * phi) * np.sin(theta / 2)
        fourth = np.exp(1j * (lmbda + phi)) * np.cos(theta / 2)
        return np.array([[first, second],
                         [third, fourth]], dtype=complex)
    else:
        raise Exception("getGateMatrix(): unsupported Gate operation (" + str(gate_type) + ")")


# applySingleGate() - Applies a single-qubit gate to the quantum state
#   Parameters: gate - (type Gate) the gate to apply
#   Effect: updates q_state with the gate operation
#   Return: nothing
def applySingleGate(gate):
    target_index = gate.target
    if gate.is_control:
        raise Exception("applySingleGate(): must use applyCGate() for control gates")
    if target_index >= n:
        raise Exception("applySingleGate(): target (" + str(target_index)
                        + ") must be less than n (" + str(n) + ")")
    # Select the gate
    gate_matrix = getGateMatrix(gate)

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
    if not gate.is_control:
        raise Exception("applyCGate(): gate operation must be a control gate")
    if target_index >= n:
        raise Exception("applyCGate(): target (" + str(target_index) + ") must be less than n (" + str(n) + ")")
    if control_index >= n or control_index < 0:
        raise Exception("applyCGate(): control (" + str(control_index) + ") must be nonnegative")
    if target_index == control_index:
        raise Exception("applyCGate(): target (" + str(target_index) + ") != control (" + str(control_index) + ")")

    # Select the gate
    gate_matrix = getGateMatrix(gate)  # gate with GateType PAULI_X and is_control=True == CNOT

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


# measure_state() - measures the entire quantum state for each possible outcome and collapses the state (end of program)
# Assumes the QASM file applies the measurement to all qubits.
# Calls the show_data method
# Effect: exits out the program
def measure_state():
    prob = []
    shots_per_state = []
    global q_state
    global p

    if is_noisy:
        temp_q_state = np.copy(q_state)
        # apply a random unitary gate for each qubit
        for i in range(n):
            theta = np.arccos(1 - 2 * random.uniform(0, 1) * p)
            phi = p * (2 * random.uniform(0, 1) - 1) * np.pi
            lmbda = p * (2 * random.uniform(0, 1) - 1) * np.pi
            noisy_gate = Gate(GateType.UNITARY, i, False, -1, [theta, phi, lmbda])
            applySingleGate(noisy_gate)

        # distribute the shots according to the randomized state
        for state in q_state:
            shots_per_state.append(int(shots * np.square(state)))

        # restore q_state to its original state
        q_state = np.copy(temp_q_state)
    else:
        # distribute the shots according to the state
        for state in q_state:
            shots_per_state.append(int(shots * np.square(state)))

    # fill out the theoretical probabilities
    for state in q_state:
        prob.append(np.real(np.square(state)))

    # print out the probabilities
    show_data(prob, shots_per_state)

    # "collapse" quantum state
    exit()


# show_data() - Prints to stdout the final state and distribution of shots
# If the verbose argument is true, print out the theoretical and actual probabilities
# If the histograms arguments are true, shows the histogram in a window
# Effect: None
def show_data(prob, shots_per_state):
    print("Final state: " + str(list(np.round(q_state, 3))), end='\n\n')
    binary_states = list(itertools.product('01', repeat=n))
    if verbose:
        print("Theoretical Probabilities")
        for i in range(2 ** n):
            print("    Pr(|" + str(''.join(binary_states[i]) + ">) = " + str(prob[i])))
        print("\nActual Probabilities")
        for i in range(2 ** n):
            print("    Pr(|" + str(''.join(binary_states[i]) + ">) = " + str(shots_per_state[i] / shots)))
        print()

    print("Shots Taken")
    for i in range(2 ** n):
        print("    Shots(|" + str(''.join(binary_states[i]) + ">) = " + str(shots_per_state[i])))

    if show_prob_graph:
        probability_bargraph(prob)
    if show_shots_graph:
        shots_bargraph(shots_per_state)


# probability_bargraph() - Displays the bar graph for the probabilities of each quantum state outcome
# Parameter - prob (numpy.ndarray List) is a list containing the probabilities for each corresponding
#               outcome
# Effect: None
def probability_bargraph(prob):
    binary_states = []
    for b in list(itertools.product('01', repeat=n)):
        binary_states.append('|' + ''.join(b) + '>')

    plt.xlabel('Measurement Outcome', fontsize=12)
    plt.ylabel('Theoretical Probability', fontsize=12)
    plt.bar(binary_states, prob)
    plt.show()


# shots_bargraph() - Displays the bar graph for the shot frequency of each quantum state outcome
# Parameter - shots_per_state (int List) is a list containing the frequency for each corresponding
#               outcome
# Effect: None
def shots_bargraph(shots_per_state):
    binary_states = []
    for b in list(itertools.product('01', repeat=n)):
        binary_states.append('|' + ''.join(b) + '>')

    plt.xlabel('Measurement Outcome', fontsize=12)
    plt.ylabel('Shot Frequency', fontsize=12)
    plt.bar(binary_states, shots_per_state)
    plt.show()


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
    CONST = 12


# global variables for handling theta and phi parsing
thetaIndex = 0
phiIndex = 0


# tokenizer() - tokenizes the input QASM file
def tokenizer(inputLine):
    tokenList = []
    gates = ['h', 'x', 'y', 'z', 'rz', 'rx', 'ry', 'id', 'cx', 'u']
    splited = customDelim(inputLine)
    prevGate = 'inv'
    prevGateIndex = -1
    count = 0
    for token in splited:
        if token in gates:
            gateObj = parseGate(token)
            prevGate = token
            prevGateIndex = count
            newToken = Token(Type.GATE, gateObj)
        elif token == "->":
            newToken = Token(Type.ARROW, token)
        elif token == "qreg":
            newToken = Token(Type.QREG, token)
        elif token == "creg":
            newToken = Token(Type.CREG, token)
        elif "pi" in token or token.isnumeric() or ("+" in token or "-" in token or "*" in token or "/" in token):
            if token == "pi":
                newToken = Token(Type.CONST, np.pi)
            elif token.isnumeric():
                newToken = Token(Type.CONST, int(token))
            else:
                newToken = Token(Type.CONST, token)
        elif token[0] == 'q' and token[1] == '[' and token[2].isnumeric() and token[3] == ']':
            if prevGate == 'u':
                tokenList[prevGateIndex].getValue().setTarget(int(token[2])) # Sets default qubit parameter (target qubit) unitary matrices

            elif prevGate == 'rx' or prevGate == 'ry' or prevGate == 'rz':
                tokenList[prevGateIndex].getValue().setTarget(int(token[2])) # Sets default qubit parameter (target qubit) rotate matrices

            elif tokenList[len(tokenList) - 1].getType() == Type.GATE and not (tokenList[len(tokenList) - 1].getValue().getControl()):
                tokenList[len(tokenList) - 1].getValue().setTarget(int(token[2])) # Sets the default single qubit parameter (target field) for any other gate

            elif tokenList[len(tokenList) - 1].getType() == Type.GATE and tokenList[len(tokenList) - 1].getValue().getControl():
                tokenList[len(tokenList) - 1].getValue().setControl(int(token[2])) # Sets the control qubit for a control gate

            elif tokenList[len(tokenList) - 1].getType() == Type.QUBIT and tokenList[len(tokenList) - 2].getType() == Type.GATE and tokenList[len(tokenList) - 2].getValue().getControl():
                tokenList[len(tokenList) - 2].getValue().setTarget(int(token[2])) # Sets the target qubit for a control gate

            elif tokenList[len(tokenList) - 1].getType() == Type.GATE:
                tokenList[len(tokenList) - 1].getValue().setTarget(int(token[2])) # Sets the default single qubit parameter (target field) for any other gate

            newToken = Token(Type.QUBIT, token[2])

        elif token[0] == 'c' and token[1] == '[' and token[2].isnumeric() and token[3] == ']':
            newToken = Token(Type.CBIT, token[2])
        elif token == "measure":
            newToken = Token(Type.MEASURE, token)
        elif token == '+' or token == '-' or token == '*' or token == '/':
            newToken = Token(Type.OP, token)
        else:
            newToken = Token(Type.INV, token)
        tokenList.append(newToken)
        count = count + 1

    return tokenList


# parseGate() - Returns an initial gate object for each token ID
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
    elif token == 'u':
        newGate = Gate(GateType.UNITARY, -1, False, -1, None)
    elif token == 'rx':
        newGate = Gate(GateType.ROTATE_X, -1, False, -1, None)
    elif token == 'ry':
        newGate = Gate(GateType.ROTATE_Y, -1, False, -1, None)
    elif token == 'rz':
        newGate = Gate(GateType.ROTATE_Z, -1, False, -1, None)
    return newGate


# customDelim() - Takes an input string and separates string based on ',', ';', '(', ')'
def customDelim(input):
    inputString = input
    for delim in ',;()':
        inputString = inputString.replace(delim, ' ')
    return inputString.split()


# parseTokens() - Takes in a file directory as an input
# tokenizes the input line by line and performs respect operations by update Gate Objects and calling Gate Operations
def parseTokens(filepath):
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
                    elif tok.getValue().getGType() == GateType.UNITARY:
                        theta = parseTheta(curTokList, i)
                        phi = parsePhi(curTokList, thetaIndex)
                        lmbda = parseLambda(curTokList, phiIndex)
                        params = [theta, phi, lmbda]
                        tok.getValue().setParams(params)
                        applySingleGate(tok.getValue())
                    elif tok.getValue().getGType() == GateType.ROTATE_X or tok.getValue().getGType() == GateType.ROTATE_Y or tok.getValue().getGType() == GateType.ROTATE_Z:
                        param1 = parseTheta(curTokList, i)
                        params = [param1]
                        tok.getValue().setParams(params)
                        applySingleGate(tok.getValue())
                    else:
                        applySingleGate(tok.getValue())
                elif tok.getType() == Type.MEASURE and curTokList[i + 2].getType() == Type.ARROW:
                    measure_state()
                else:
                    continue


#parseTheta() - Takes as input a tokenlist and a position in the token array and evaluates the expression specified for the theta positional argument
def parseTheta(curTokList, i):
    j = i + 1
    tok = curTokList[j]
    prevTok = tok.getType()
    if type(tok.getValue()) == str and "pi" in tok.getValue():
        stringToParse = parseWithPi(tok.getValue())
    else:
        stringToParse = str(tok.getValue())
    global thetaIndex
    while j < len(curTokList):
        if curTokList[j].getType() == Type.CONST and prevTok == Type.CONST:
            break
        else:
            if type(tok.getValue()) == str and "pi" in curTokList[j].getValue():
                stringToParse = parseWithPi(curTokList[j].getValue())
            else:
                stringToParse = stringToParse + str(curTokList[j].getValue())
        prevTok = curTokList[j].getType()
        j = j + 1
    thetaIndex = j
    return eval(stringToParse)


#parsePhi() - Takes as input a tokenlist and a position in the token array and evaluates the expression specified for the phi positional argument
def parsePhi(curTokList, i):
    j = i + 1
    tok = curTokList[j]
    prevTok = tok.getType()
    if type(tok.getValue()) == str and "pi" in tok.getValue():
        stringToParse = parseWithPi(tok.getValue())
    else:
        stringToParse = str(tok.getValue())
    global phiIndex
    while j < len(curTokList):
        if curTokList[j].getType() == Type.CONST and prevTok == Type.CONST:
            break
        else:
            if type(tok.getValue()) == str and "pi" in curTokList[j].getValue():
                stringToParse = parseWithPi(curTokList[j].getValue())
            else:
                stringToParse = stringToParse + str(curTokList[j].getValue())
        prevTok = curTokList[j].getType()
        j = j + 1
    phiIndex = j
    return eval(stringToParse)

#parseLambda() - Takes as input a tokenlist and a position in the token array and evaluates the expression specified for the lambda positional argument
def parseLambda(curTokList, i):
    j = i + 1
    tok = curTokList[j]
    prevTok = tok.getType()
    if type(tok.getValue()) == str and "pi" in tok.getValue():
        stringToParse = parseWithPi(tok.getValue())
    else:
        stringToParse = str(tok.getValue())
    j = j + 1
    while j < len(curTokList) and (curTokList[j].getType() == Type.CONST or curTokList[j].getType() == Type.OP):
        if curTokList[j].getType() == Type.CONST and prevTok == Type.CONST:
            break
        else:
            if type(tok.getValue()) == str and "pi" in curTokList[j].getValue():
                stringToParse = parseWithPi(curTokList[j].getValue())
            else:
                stringToParse = stringToParse + str(curTokList[j].getValue())
        prevTok = curTokList[j].getType()
        j = j + 1
    return eval(stringToParse)


def parseWithPi(exp):
    index = exp.index('pi')
    piVal = np.pi
    return exp[0:index] + str(piVal) + exp[index + 2:len(exp)]


def main():
    if len(argv) < 3:
        print(f"usage: {argv[0]} <file>")
    filepath = argv[1]
    global shots
    shots = int(argv[2])

    parser = argparse.ArgumentParser(description='Run QASM Compiler')
    parser.add_argument('filepath', action="store")
    parser.add_argument('shots', action="store", type=int)
    parser.add_argument('-n', action='store_true', default=False,
                        dest='isNoisy',
                        help='Enable Noisy Flag')
    parser.add_argument('-v', action='store_true', default=False,
                        dest='isVerbose',
                        help='Enable Verbose Flag')
    parser.add_argument('-g', action='store_true', default=False,
                        dest='showGraph',
                        help='Enable Show Probability Graph Flag')
    parser.add_argument('-s', action='store_true', default=False,
                        dest='showShots',
                        help='Enable Show Shots Graph Flag')
    results = parser.parse_args()

    global is_noisy
    global verbose
    global show_prob_graph
    global show_shots_graph

    is_noisy = results.isNoisy
    verbose = results.isVerbose
    show_prob_graph = results.showGraph
    show_shots_graph = results.showShots
    parseTokens(filepath)


if __name__ == '__main__':
    main()
