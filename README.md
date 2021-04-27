# QASMSimulator

QASMSimulator is our implementation of a general purpose QASM compiler (OPENQASM).
The compiler was made by Parth Shroff and Richard Noh. The test cases and Bell's Inequality analysis was done by Aaron Cheung



## Usage

```
QASMSimulatorTemp.py [-h] [-n] [-v] [-g] [-s] filepath shots

Run QASM Compiler

positional arguments:
  filepath
  shots

optional arguments:
  -h, --help  show this help message and exit
  -n          Enable Noisy Flag
  -v          Enable Verbose Flag
  -g          Enable Show Graph Flag
  -s          Enable Show Shots Flag
```

## Simulation
Our focus was to simulate Bell's Inequality but the compiler is general purpose
