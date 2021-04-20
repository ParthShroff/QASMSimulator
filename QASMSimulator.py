from sys import argv
from enum import Enum
import numpy as np
import itertools

qubitArray = []
finalOutputQubits = []

gateArray = []
qubitIndexArray = []

shotArray = []

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


class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def getType(self):
        return self.type

    def getValue(self):
        return self.value


def main():
    if len(argv) < 3:
        print(f"usage: {argv[0]} <file>")
    filepath = argv[1]
    shots = argv[2]
    with open(filepath) as fp:
        cnt = 0
        for line in fp:
            curTokList = tokenizer(line)
            for i in range(len(curTokList)):
                tok = curTokList[i]
                if tok.getType() == Type.INV:
                    continue
                elif tok.getType() == Type.QREG:
                    initializeQubitArray(int(curTokList[i + 1].getValue()))
                elif tok.getType() == Type.GATE:
                    gateArray.append(tok.getValue())
                elif tok.getType() == Type.QUBIT and curTokList[i - 1].getType() == Type.GATE:
                    qubitIndexArray.append(tok.getValue())
                elif tok.getType() == Type.MEASURE and curTokList[i + 2].getType() == Type.ARROW:
                    measureQubit(curTokList[i + 1].getValue(), curTokList[i + 3].getValue())
                else:
                    continue
    for i in range(len(finalOutputQubits)):
        print("The state for qubit " + str(i) + " is: \n" + str(finalOutputQubits[i]))

    stateArray = list(["".join(x) for x in itertools.product(["0","1"], repeat=len(finalOutputQubits))])

    for i in range(len(stateArray)):
        prob = 1
        for j in range(len(stateArray[i])):
            if int(stateArray[i][j]) == 0:
                prob = prob * np.square(qubitArray[len(finalOutputQubits)-1-j][0, 0])
                print("the prob is: " + str(prob))
            if int(stateArray[i][j]) == 1:
                prob = prob * np.square(qubitArray[len(finalOutputQubits)-1-j][1, 0])
                print("the prob is: " + str(prob))
        shotArray.append(np.multiply(int(shots), prob))

    for i in range(len(stateArray)):
        print("The state for qubit |" + stateArray[i] + "> and the theoretical frequency is: \n" + str(shotArray[i]))

    fp.close()



def measureQubit(qubitIndex, cbitIndex):
    for i in range(len(gateArray)):
        if int(qubitIndexArray[i]) == int(qubitIndex):
            applyGate(gateArray[i], int(qubitIndex))
    finalOutputQubits.insert(int(cbitIndex), qubitArray[int(qubitIndex)])

def applyGate(gate, qIndex):
    if gate == 'h':
        hadamard = 1. / np.sqrt(2) * np.array([[1, 1],
                                               [1, -1]])
        qubitArray[int(qIndex)] = np.matmul(hadamard, qubitArray[int(qIndex)])
    elif gate == 'x':
        xGate = np.array([[0, 1],
                            [1, 0]])
        qubitArray[int(qIndex)] = np.matmul(xGate, qubitArray[int(qIndex)])
    elif gate == 'y':
        yGate = np.array([[0, -1j],
                            [1j, 0]])
        qubitArray[int(qIndex)] = np.matmul(yGate, qubitArray[int(qIndex)])
    elif gate == 'z':
        zGate = np.array([[1, 0],
                            [0, -1]])
        qubitArray[int(qIndex)] = np.matmul(zGate, qubitArray[int(qIndex)])
    elif gate == 't':
        tGate = np.array([[1, 0],
                        [0, np.exp(np.pi*1j/4)]])
        qubitArray[int(qIndex)] = np.matmul(tGate, qubitArray[int(qIndex)])
    elif gate == 's':
        sGate = np.array([[1, 0],
                        [0, np.exp(np.pi*1j/2)]])
        qubitArray[int(qIndex)] = np.matmul(sGate, qubitArray[int(qIndex)])
    elif gate == 'sdg':
        sdgGate = np.array([[1, 0],
                        [0, np.exp(-1*np.pi*1j/2)]])
        qubitArray[int(qIndex)] = np.matmul(sdgGate, qubitArray[int(qIndex)])


def initializeQubitArray(length):
    for i in range(length):
        qubitArray.append(np.array([[1], [0]]))


def tokenizer(inputLine):
    tokenList = []
    gates = ['h', 'x', 't', 'tdg', 'sdg', 's', 'z', 'p', 'rz', 'rx', 'ry', 'rxx', 'rzz', 'sx', 'sxdg', 'id']
    splited = customDelim(inputLine)
    for token in splited:
        if token in gates:
            newToken = Token(Type.GATE, token)
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


def customDelim(input):
    inputString = input
    for delim in ',;':
        inputString = inputString.replace(delim, ' ')
    return inputString.split()


if __name__ == '__main__':
    main()
