# Contains all the functions from notebooks Part_1, Decompositions, Part_2 and Part_3

import pennylane as qml
from pennylane import numpy as np
import copy


# Convert a complex array to a string in the form of a sum of basis states
def statevector_to_braket(statevector):
    num_qubits = int(np.log2(len(statevector)))
    braket_notation = ""

    for i in range(2 ** num_qubits):
        amplitude = statevector[i]
        if amplitude != 0:
            basis_state = bin(i)[2:].zfill(num_qubits)
            basis_state_str = f"|{basis_state}>"
            braket_notation += f"{amplitude:.2f} * {basis_state_str} + "
            
    braket_notation = braket_notation[:-3]

    return braket_notation


# Create a quantum circuit in pennylane from a list of operations
def create_circuit_from_operations(ops, num_wires, dev):
    @qml.qnode(dev)
    def circuit():
        for op in ops:
            op_type = op.__class__.__name__
            op_params = op.parameters
            op_wires = op.wires
            
            getattr(qml, op_type)(*op_params, wires=op_wires)
        
        return qml.state()

    return circuit

# Create a quantum circuit in pennylane from a list of operations and measurements
def create_circuit_from_operations_measurements(ops, measurements, num_wires, dev):
    @qml.qnode(dev)
    def circuit():
        for op in ops:
            op_type = op.__class__.__name__ 
            op_params = op.parameters
            op_wires = op.wires 

            getattr(qml, op_type)(*op_params, wires=op_wires)

        measurement_calls = []
        for meas in measurements:
            meas_type = meas.return_type
            meas_wires = meas.wires
            meas_obs = meas.obs

            if meas_type == qml.measurements.Sample:
                if meas_obs is not None:
                    measurement_calls.append(qml.sample(meas_obs))
                else:
                    measurement_calls.append(qml.sample(wires=meas_wires))
            elif meas_type == qml.measurements.Expectation:
                measurement_calls.append(qml.expval(meas_obs))
            elif meas_type == qml.measurements.Probability:
                measurement_calls.append(qml.probs(wires=meas_wires))
            elif meas_type == qml.measurements.State:
                measurement_calls.append(qml.state())
            elif meas_type == qml.measurements.Variance:
                measurement_calls.append(qml.var(meas_obs))
            else:
                raise ValueError(f"Unsupported measurement type: {meas_type}")

        if len(measurement_calls) == 1:
            return measurement_calls[0]
        else:
            return tuple(measurement_calls)

    return circuit


# Inserts a random Pauli gate on the wire corresponding to the gates with input probabilities
def make_noisy(alpha, beta, ops):
    index = 0
    while index < len(ops):
        op = ops[index]
        num_wires = len(op.wires)
        op_wires = op.wires.tolist()
        if num_wires == 1 and np.random.rand() < alpha:
            new_op = np.random.choice([qml.PauliX, qml.PauliY, qml.PauliZ])(wires=op_wires)
            ops.insert(index + 1, new_op)
            index += 1
        elif num_wires == 2 and np.random.rand() < beta:
            new_wire = np.random.choice(op_wires)
            new_op = np.random.choice([qml.PauliX, qml.PauliY, qml.PauliZ])(wires=new_wire)
            ops.insert(index + 1, new_op)
            index += 1
        index += 1


# Inserts a random Pauli gate decomposed to the target basis on the wire corresponding to the gates with input probabilities
def make_noisy_target_basis(alpha, beta, ops):
    noise_options = [
        [qml.PauliX],
        [lambda wires: qml.RZ(np.pi, wires=wires)],
        [lambda wires: qml.RZ(np.pi, wires=wires), qml.PauliX]
    ]
    
    index = 0
    while index < len(ops):
        op = ops[index]
        num_wires = len(op.wires)
        
        if num_wires == 1:
            prob = alpha
            selected_wire = op.wires[0]
        elif num_wires == 2:
            prob = beta
            selected_wire = np.random.choice(op.wires)
        else:
            prob = 0
        
        if prob > 0 and np.random.rand() < prob:
            choice = np.random.choice(len(noise_options))
            gates = noise_options[choice]
            new_ops = [gate(wires=selected_wire) if callable(gate) else gate(wires=selected_wire) for gate in gates]
            ops[index + 1:index + 1] = new_ops
            index += len(new_ops)
        
        index += 1



# Define supported gates
supported_gates = [
    'u', 'cx', 'rz', 'ry', 'rx', 'swap', 'cz', 'cy',
    'cu', 'crx', 'cs', 'ct', 'ssx', 'x', 'sx', 'id',
    'y', 'z', 'h', 's', 't'
]

# Define ZY decomposition
def zy_decomposition(theta, phi, lam, rho):
    delta = lam
    beta = phi
    gamma = theta
    alpha = beta / 2 + delta / 2 + rho
    return alpha, beta, gamma, delta

# Define the Gate class which decomposes gates to the target basis
class Gate:
    GATE_REQUIREMENTS = {
        'single_qubit': {'qubits': 1, 'params': 0},
        'one_control_two_qubits': {'qubits': 2, 'params': 0},
        'multi_control': {'qubits_min': 2, 'params': 0},
        'with_params': {
            'u': {'qubits': 1, 'params': 4},
            'cu': {'qubits': 2, 'params': 4},
            'rz': {'qubits': 1, 'params': 1},
            'ry': {'qubits': 1, 'params': 1},
            'rx': {'qubits': 1, 'params': 1},
            'crx': {'qubits': 2, 'params': 1},
            'crz': {'qubits': 2, 'params': 1},
            'ssx': {'qubits': 'variable', 'params': 1},
            'controlledphaseshift': {'qubits': 2, 'params': 1}
        }
    }

    DECOMPOSITION_MAP = {
        'y': lambda q, p: [Gate('u', q, [np.pi, np.pi / 2, np.pi / 2, 0])],
        'z': lambda q, p: [Gate('u', q, [0, 0, np.pi, 0])],
        'h': lambda q, p: [Gate('u', q, [np.pi / 2, 0, np.pi, 0])],
        'ry': lambda q, p: Gate._decompose_ry_gate_static(q, p),
        'rx': lambda q, p: [Gate('u', q, [p[0], -np.pi / 2, np.pi / 2, 0])],
        's': lambda q, p: [Gate('u', q, [0, 0, np.pi / 2, 0])],
        't': lambda q, p: [Gate('u', q, [0, 0, np.pi / 4, 0])],
        'id': lambda q, p: [],
        'ssx': lambda q, p: Gate._decompose_ssx(q, p).decompose(),
        'cy': lambda q, p: [
            Gate('rz', q[0], [-np.pi / 2]),
            Gate('cx', q),
            Gate('rz', q[0], [np.pi / 2])
        ],
        'cz': lambda q, p: [
            Gate('u', [q[0]], [np.pi / 2, 0, np.pi, 0]),
            Gate('cx', q),
            Gate('u', [q[0]], [np.pi / 2, 0, np.pi, 0])
        ],
        'crz': lambda q, p: [
        Gate('rz', q[1], [p[0] / 2]),
        Gate('cx', q), 
        Gate('rz', q[1], [-p[0] / 2]),
        Gate('cx', q), 
        ],
        'controlledphaseshift': lambda q, p: [
            Gate('rz', q[0], [p[0] / 2]),
            Gate('cx', q), 
            Gate('rz', q[1], [-p[0] / 2]),
            Gate('cx', q),  
            Gate('rz', q[1], [p[0] / 2])
        ],
        'swap': lambda q, p: [
            Gate('cx', q),
            Gate('cx', [q[1], q[0]]),
            Gate('cx', q)
        ],
        'crx': lambda q, p: Gate._decompose_cu_gate_static(q, p, 'crx'),
        'cu': lambda q, p: Gate._decompose_cu_gate_static(q, p, 'cu'),
        'cs': lambda q, p: Gate._decompose_cu_gate_static(q, p, 'cs'),
        'ct': lambda q, p: Gate._decompose_cu_gate_static(q, p, 'ct'),
    }

    def __init__(self, name, qubits, params=None):
        self.name = name
        self.qubits = self._validate_qubits(qubits)
        self.params = params if params is not None else []

    def _validate_qubits(self, qubits):
        if isinstance(qubits, list):
            return qubits
        return [qubits]

    def target(self):
        return self.qubits[0]

    def control(self):
        return self.qubits[1:]

    def nr_qubits(self):
        return len(self.qubits)

    def decompose(self):
        gates = [self]
        for stage in ['_decompose_multiple_qubit_gates', '_decompose_to_cx_rz_u', '_decompose_u_gate']:
            decomposed = []
            for gate in gates:
                decomposed.extend(getattr(gate, stage)())
            gates = decomposed
        return gates

    def _decompose_multiple_qubit_gates(self):
        if self.nr_qubits() <= 2:
            return [self]
        
        if self.name not in ['cx', 'ssx']:
            return [self]
        
        control_qubits = self.control().copy()
        target_qubit = self.target()
        first_control = control_qubits.pop(0)
        k = self.params[0] if self.name == 'ssx' else 0
        k_succ = k + 1 if k >= 0 else k - 1

        decomposition = [
            Gate('ssx', [target_qubit, first_control], [k_succ]),
            *Gate('cx', [first_control] + control_qubits)._decompose_multiple_qubit_gates(),
            Gate('ssx', [target_qubit, first_control], [-k_succ]),
            copy.deepcopy(Gate('cx', [first_control] + control_qubits)._decompose_multiple_qubit_gates()),
            Gate('ssx', [target_qubit] + control_qubits, [k_succ])
        ]
        return decomposition

    def _decompose_to_cx_rz_u(self):
        decomposition_func = self.DECOMPOSITION_MAP.get(self.name, lambda q, p: [self])
        return decomposition_func(self.qubits, self.params)

    def _decompose_u_gate(self):
        if self.name != 'u':
            return [self]
        
        _, beta, gamma, delta = zy_decomposition(*self.params)
        gates = []
        if delta != 0:
            gates.append(Gate('rz', self.qubits, [delta]))
        gates += Gate('ry', self.qubits, [gamma])._decompose_ry_gate()
        if beta != 0:
            gates.append(Gate('rz', self.qubits, [beta]))
        return gates

    @staticmethod
    def _decompose_ry_gate_static(qubits, params):
        return [
            Gate('rz', qubits, [-np.pi]),
            Gate('sx', qubits),
            Gate('rz', qubits, [np.pi - params[0]]),
            Gate('sx', qubits)
        ]

    def _decompose_ry_gate(self):
        if self.name != 'ry':
            return [self]
        return [
            Gate('rz', self.qubits, [-np.pi]),
            Gate('sx', self.qubits),
            Gate('rz', self.qubits, [np.pi - self.params[0]]),
            Gate('sx', self.qubits)
        ]

    @staticmethod
    def _decompose_ssx(qubits, params):
        return Gate('ssx', qubits, params)

    @staticmethod
    def _decompose_cu_gate_static(qubits, params, gate_type):
        if gate_type == 'crx':
            theta, phi, lam, rho = params[0], -np.pi / 2, np.pi / 2, 0
        elif gate_type == 'cs':
            theta, phi, lam, rho = 0, 0, np.pi / 2, 0
        elif gate_type == 'ct':
            theta, phi, lam, rho = 0, 0, np.pi / 4, 0
        elif gate_type == 'cu':
            theta, phi, lam, rho = params
        else:
            return [Gate(gate_type, qubits, params)]

        alpha, beta, gamma, delta = zy_decomposition(theta, phi, lam, rho)
        phase_shift = [Gate('u', [qubits[0]], [0, 0, alpha, 0])] if alpha != 0 else []
        A = Gate('ry', [qubits[1]], [gamma / 2])._decompose_ry_gate() + [Gate('rz', [qubits[1]], [beta])]
        B = [Gate('rz', [qubits[1]], [-(beta + delta) / 2])] + Gate('ry', [qubits[1]], [-gamma / 2])._decompose_ry_gate()
        C = [Gate('rz', [qubits[1]], [(delta - beta) / 2])]
        CX = [Gate('cx', qubits)]
        
        return C + CX + B + copy.deepcopy(CX) + A + phase_shift

    def _decompose_cu_gate(self):
        return self._decompose_cu_gate_static(self.qubits, self.params, self.name)


def decompose_pennylane_ops(ops):
    decomposed = []
    for op in ops:
        name = op.name.lower()
        params = list(op.parameters)
        wires = [int(wire) for wire in op.wires]

        if name in ['pauli_x', 'x']:
            name = 'x'
        elif name in ['pauli_y', 'y']:
            name = 'y'
        elif name in ['pauli_z', 'z']:
            name = 'z'
        elif name in ['hadamard', 'h']:
            name = 'h'
        elif name == 's':
            name = 's'
        elif name == 't':
            name = 't'
        elif name in ['cnot', 'cx']:
            name = 'cx'
        elif name == 'crx':
            name = 'crx'
        elif name == 'cu3':
            name = 'cu'
            params = params + [0]
        elif name == 'swap':
            name = 'swap'
        elif name in ['cy', 'cz', 'cs', 'ct', 'crz', 'ssx']:
            name = name 
        elif name in ['rx', 'ry', 'rz', 'u', 'id']:
            name = name
        elif name == ['controlledphaseshift', 'cp']:
            name = 'controlledphaseshift'

        gate = Gate(name, wires, params)
        decomposed_gates = gate.decompose()
        decomposed.extend(decomposed_gates)
        
    return decomposed

def get_decomposed_gates(gates):
    decomposed_gates = []
    for gate in gates:
        decomposed_gates.append(Gate(name=gate.name, qubits=gate.qubits, params=gate.params))
    
    return decomposed_gates


# Mapping from custom gate names to PennyLane gate functions
GATE_MAPPING = {
    'cx': qml.CNOT,
    'cz': qml.CZ,
    'cy': qml.CY,
    'swap': qml.SWAP,
    'crx': qml.CRX,
    'cs': qml.CSWAP,
    'ct': qml.CZ,
    'rz': qml.RZ,
    'ry': qml.RY,
    'rx': qml.RX,
    'u': qml.U3,
    'sx': qml.SX,
    'x': qml.PauliX,
    'paulix': qml.PauliX,
    'y': qml.PauliY,
    'z': qml.PauliZ,
    'h': qml.Hadamard,
    's': qml.S,
    't': qml.T,
    'id': qml.Identity,
    'controlledphaseshift': qml.CPhase,
    'crz': qml.CRZ
    # Add more mappings as needed
}


# Convert a list of gates to a PennyLane quantum tape
def convert_gates_to_pennylane_circuit(gates, return_type='state'):
    all_qubits = set()
    for gate in gates:
        all_qubits.update(gate.qubits)
    num_wires = max(all_qubits) + 1
    wires = range(num_wires)

    tape = qml.tape.QuantumTape()

    with tape:
        for gate in gates:
            gate_name = gate.name.lower()
            qubits = gate.qubits
            params = gate.params

            qml_gate = GATE_MAPPING.get(gate_name)

            if qml_gate is None:
                raise ValueError(f"Gate '{gate_name}' is not recognized in GATE_MAPPING.")

            if qml_gate.num_params == 0:
                qml_gate(wires=qubits)
            elif qml_gate.num_params == 1:
                qml_gate(*params, wires=qubits)
            elif qml_gate.num_params == 2:
                qml_gate(*params, wires=qubits)
            elif qml_gate.num_params == 3:
                qml_gate(*params, wires=qubits)
            elif qml_gate.num_params == 4:
                qml_gate(*params, wires=qubits)
            else:
                raise ValueError(f"Unsupported number of parameters ({qml_gate.num_params}) for gate '{gate_name}'.")

    return tape

# Define QFT circuit
def qft(wires):
    n = len(wires)
    
    for j in range(n):
        qml.Hadamard(wires=wires[j])
        
        for k in range(j + 1, n):
            angle = np.pi / (2 ** (k - j))
            qml.CPhase(angle, wires=[wires[k], wires[j]])
    
    for i in range(n // 2):
        qml.SWAP(wires=[wires[i], wires[n - i - 1]])

def adjoint_qft(wires):
    n = len(wires)

    for i in reversed(range(n // 2)):
        qml.SWAP(wires=[wires[i], wires[n - i - 1]])

    for j in reversed(range(n)):
        for k in reversed(range(j + 1, n)):
            angle = -np.pi / (2 ** (k - j))
            qml.CPhase(angle, wires=[wires[k], wires[j]])
        qml.Hadamard(wires=wires[j])


def add_k_fourier_basis(k, wires):
    for i in range(len(wires)):
        qml.RZ(k * np.pi / (2**i), wires=wires[i])


# Generate wires to represent m, k and m+k
def generate_wires(m, k):
    bits_m = m.bit_length()
    bits_k = k.bit_length()
    total = m + k
    bits_solution = total.bit_length()
    wires_m = list(range(bits_m))
    wires_k = list(range(bits_m, bits_m + bits_k))
    wires_solution = list(range(bits_m + bits_k, bits_m + bits_k + bits_solution))
    
    return wires_m, wires_k, wires_solution


# Define the addition circuit
def addition(wires_m, wires_k, wires_solution):
    qft(wires=wires_solution)
    # qml.Barrier()

    for i in range(len(wires_m)):
        qml.ctrl(add_k_fourier_basis, control=wires_m[i])(2 **(len(wires_m) - i - 1), wires_solution)

    for i in range(len(wires_k)):
        qml.ctrl(add_k_fourier_basis, control=wires_k[i])(2 **(len(wires_k) - i - 1), wires_solution)

    # qml.Barrier()
    adjoint_qft(wires=wires_solution)


# Define encoding circuit that prepares the state in the binary representation with input n
def int_to_binary_ops(n, wires):
    binary = bin(n)[2:]
    binary = binary.zfill(len(wires))
    for i, bit in enumerate(binary):
        if bit == '1':
            qml.PauliX(wires=wires[i])


# Define a function to pack all the functions into one

from scipy.linalg import sqrtm

def ket_to_statevector(ket):
    num_qubits = len(ket)
    statevector = np.zeros(2**num_qubits, dtype=complex)
    index = int("".join(map(str, ket)), 2)
    statevector[index] = 1.0
    return statevector

def statevector_to_density_matrix(statevector):
    return np.outer(statevector, np.conj(statevector))

def density_matrix_fidelity(rho, sigma):
    sqrt_rho = sqrtm(rho)
    product = np.dot(sqrt_rho, np.dot(sigma, sqrt_rho))
    sqrt_product = sqrtm(product)
    return np.real(np.trace(sqrt_product))**2

def fidelity_between_kets(ket1, ket2):
    statevector1 = ket_to_statevector(ket1)
    statevector2 = ket_to_statevector(ket2)
    
    density_matrix1 = statevector_to_density_matrix(statevector1)
    density_matrix2 = statevector_to_density_matrix(statevector2)
    
    return density_matrix_fidelity(density_matrix1, density_matrix2)


def noisy_decomposed_quantum_sum_fidelity(m, k, alpha, beta):
    wires_m, wires_k, wires_solution = generate_wires(m, k) 

    dev = qml.device("default.qubit", wires=wires_m + wires_k + wires_solution, shots=1)
    n_wires = len(dev.wires) 
    @qml.qnode(dev)
    def quantum_sum(m, k, wires_m, wires_k, wires_solution):
        int_to_binary_ops(m, wires_m)
        int_to_binary_ops(k, wires_k)

        addition(wires_m, wires_k, wires_solution)

        return qml.sample(wires=wires_solution)

    sample = quantum_sum(m, k, wires_m, wires_k, wires_solution)

    tape = quantum_sum.tape
    ops = tape.operations
    measurements = tape.measurements
    decomposed_gates = decompose_pennylane_ops(ops)
    decomposed_gates_list = get_decomposed_gates(decomposed_gates)
    result = convert_gates_to_pennylane_circuit(decomposed_gates_list, return_type='state')
    tape = qml.tape.QuantumTape(result.operations, measurements)

    quantum_circuit = create_circuit_from_operations_measurements(result.operations, measurements, n_wires, dev)
    compiled_circuit = qml.compile(quantum_circuit)
    noiseless_sum = compiled_circuit()

    make_noisy_target_basis(alpha, beta, tape.operations)
    noisy_tape = qml.tape.QuantumTape(tape.operations, measurements)
    circuit = noisy_tape.circuit

    noisy_quantum_circuit = create_circuit_from_operations_measurements(noisy_tape.operations, measurements, n_wires, dev)
    noisy_compiled_circuit = qml.compile(noisy_quantum_circuit)
    noisy_sum = noisy_compiled_circuit()

    return fidelity_between_kets(noiseless_sum, noisy_sum)
