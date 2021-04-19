from sys import argv
from enum import Enum
import numpy as np




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
    with open(filepath) as fp:
        cnt = 0
        for line in fp:
            curTokList = tokenizer(line)
            for tok in curTokList:
                print("The token type is: " + str(tok.getType()) + " and the token value is: " + tok.getValue())




def tokenizer(inputLine):
    tokenList = []
    gates = ['h', 'x', 't', 'tdg', 'sdg', 's', 'z', 'p', 'rz', 'rx', 'ry', 'rxx', 'rzz', 'sx', 'sxdg']
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
