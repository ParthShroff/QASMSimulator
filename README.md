# QASMSimulator

QASMSimulator is our implementation of a general purpose QASM compiler (OPENQASM).
The compiler was made by Parth Shroff and Richard Noh. The test cases and Bell's Inequality analysis was done by Aaron Cheung

## Usage

```
QASMSimulator.py filepath shots [-h] [-n] [-v] [-g] [-s]

Run QASM Compiler

positional arguments:
  filepath
  shots

optional arguments:
  -h, --help  show this help message and exit
  -n          Enable Noisy Flag: Simulates quantum noise
  -v          Enable Verbose Flag: Displays the theoretical and actual probabilties of the simulation
  -g          Enable Show Probability Graph Flag: Opens a new window displaying a probability bar graph for each outcome
  -s          Enable Show Shots Graph Flag: Opens a new window displaying a shot frequency bar graph for each outcome
```

## Simulation
QASMSimulator is a general-purpose, N-qubit quantum computer simulator. The quantum state is stored as a 2^N-vector that stores the amplitudes of component outcomes of an N-qubit system. This vector is multiplied by a 2^N by 2^N Kronecker product matrix representation of a quantum gate.

Supported quantum gates:
    Identity
    Hadamard
    Pauli X, Y, and Z
    Rotate X, Y, and Z
    Unitary
    All single control gate variants (CNOT, CHadamard, etc)

Using these matrices, we are able to simulate entangled states. All other single gate operations can be substituted with an appropriate unitary gate.

# Example
python QASMSimulator.py "test_cases/bell-a'b.txt" 1024 -n -s -p

The command above runs the simulator on the "bell-a'b.txt" file, with 1024 shots, noise enabled, and displays the shots frequency graph and the probability graph.

Note that the program assumes that whenever a measurement command is given, the file is requesing a total state measurement, rather than a partial one. Additionally, if no measurement command is given in a file, the simulator assumes there is such a command at the end of the file.