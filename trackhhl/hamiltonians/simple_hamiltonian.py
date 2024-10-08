from trackhhl.event_model.event_model import Event, Segment
from trackhhl.hamiltonians.hamiltonian import Hamiltonian
from itertools import product, count
from scipy.sparse import eye
from scipy.sparse.linalg import cg
import numpy as np
import time
from scipy.linalg import block_diag
from numpy import array, cross
from numpy.linalg import solve, norm
import cProfile
import pstats

from linear_solvers import HHL
from qiskit import Aer, execute, BasicAer
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis.evolution import SuzukiTrotter
from qiskit.synthesis import QDrift, LieTrotter, MatrixExponential
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import transpile
from qiskit.opflow import I, Z
from qiskit.providers.fake_provider import FakeHanoiV2
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from qiskit import IBMQ, Aer
from qiskit_aer.noise import NoiseModel
from qiskit import QiskitError
from scipy.optimize import curve_fit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

def count_gates_on_qubit(circuit, qubit):
  """
  Counts the number of gates applied to a specific qubit in a circuit.

  Args:
      circuit: The QuantumCircuit object.
      qubit: The Qubit object representing the qubit of interest.

  Returns:
      int: The number of gates applied to the specified qubit.
  """
  gate_count = 0
  for instr in circuit.data:
    if qubit in instr[1]:
      gate_count += 1
  return gate_count


def upscale_pow2(A,b):
    #add a constant the same as the original matrix for this number 
    m = A.shape[0]
    d = int(2**np.ceil(np.log2(m)) - m)
    if d > 0:
        A_tilde = np.block([[A, np.zeros((m, d),dtype=np.float64)],[np.zeros((d, m),dtype=np.float64), 3*np.eye(d,dtype=np.float64)]])
        b_tilde = np.block([b,b[:d]])
        return A_tilde, b_tilde
    else:
        return A, b

class SimpleHamiltonian(Hamiltonian):
    def __init__(self, epsilon, gamma, delta):
        self.epsilon                                    = epsilon
        self.gamma                                      = gamma
        self.delta                                      = delta
        self.Z                                          = None
        self.A                                          = None
        self.b                                          = None
        self.segments                                   = None
        self.segments_grouped                           = None
        self.n_segments                                 = None
    
    
    def construct_segments(self, event: Event):
        
        segments_grouped = []
        segments = []
        n_segments = 0
        segment_id = count()
        for idx in range(len(event.modules)-1):
            from_hits = event.modules[idx].hits
            to_hits = event.modules[idx+1].hits
            #print(to_hits)
            
            segments_group = []
            for from_hit, to_hit in product(from_hits, to_hits):
                seg = Segment(next(segment_id),from_hit, to_hit)
                segments_group.append(seg)
                segments.append(seg)
                n_segments = n_segments + 1
        
            segments_grouped.append(segments_group)
            
        
        self.segments_grouped = segments_grouped
        self.segments = segments
        self.n_segments = n_segments
        
    def construct_hamiltonian(self, event: Event):
        #pr = cProfile.Profile()
        #pr.enable()

        if self.segments_grouped is None:
            self.construct_segments(event)
        A = eye(self.n_segments,format='lil')*(-(self.delta+self.gamma))
        b = np.ones(self.n_segments)*self.delta
        for group_idx in range(len(self.segments_grouped) - 1):
            for seg_i, seg_j in product(self.segments_grouped[group_idx], self.segments_grouped[group_idx+1]):
                if seg_i.hit_to == seg_j.hit_from:
                    cosine = seg_i * seg_j
                    #print(cosine)
                    if abs(cosine - 1) < self.epsilon:
                        A[seg_i.segment_id, seg_j.segment_id] = A[seg_j.segment_id, seg_i.segment_id] =  1
        A = A.tocsc()
        
        self.A, self.b = -A, b
        #pr.disable()
        #stats = pstats.Stats(pr)
        #stats.sort_stats('cumulative')
        #stats.print_stats(10)
        return -A, b
    
        
    def construct_Z(self):
        def find_intersections(ham):
            def find_intersection(v1,v2):

                XA0 = v1
                XA1 = v2
                XB0 = array([0, 0, 0])
                XB1 = array([0, 0, 1])

                UA = (XA1 - XA0) / norm(XA1 - XA0)
                UB = (XB1 - XB0) / norm(XB1 - XB0)
                UC = cross(UB, UA); UC /= norm(UC)

                RHS = XB0 - XA0
                LHS = array([UA, -UB, UC]).T
                parameters = solve(LHS, RHS)
                intersection_point_B = XB0 + parameters[1] * UB
                return intersection_point_B
            intersections = []
            for track in range(len(ham.Z)):
                v1 = np.array([ham.Z[track].hit_from.x, ham.Z[track].hit_from.y, ham.Z[track].hit_from.z])
                v2 = np.array([ham.Z[track].hit_to.x, ham.Z[track].hit_to.y, ham.Z[track].hit_to.z])
                intercept = find_intersection(v1,v2)
                intersections.append(intercept[2])
            matrix = np.zeros((len(intersections), len(intersections)),float)
            np.fill_diagonal(matrix, intersections)
            return matrix
        
        self.Z = []
        for seg in self.segments_grouped:
            self.Z.append(seg)
        self.Z = self.Z[0] + self.Z[1]
        self.Z = find_intersections(self)
        
    
    def suzuki_trotter_circuit(self, A, time, num_slices=1):
        """Generate Suzuki-Trotter approximation for time evolution.

        Args:
            A (np.ndarray): The unitary matrix to be evolved.
            time (float): Total evolution time.
            num_slices (int): Number of Trotter slices.

        Returns:
            QuantumCircuit: The Suzuki-Trotter approximation circuit.
        """
        num_qubits = int(np.log2(A.shape[0]))
        qr = QuantumRegister(num_qubits, name="q")
        circuit = QuantumCircuit(qr, name="SuzukiTrotter")

        hamiltonian = -1j * A * time / num_slices

        for _ in range(num_slices):
            for i in range(num_qubits):
                theta, phi, lam = np.angle(hamiltonian[i, i]), -np.angle(hamiltonian[i, (i + 1) % num_qubits]), -np.angle(hamiltonian[i, (i - 1) % num_qubits])
                circuit.u(theta, phi, lam, qr[i])

        return circuit
    
    def solve_classicaly(self):
        if self.A is None:
            raise Exception("Not initialised")
        
        solution, _ = cg(self.A, self.b, atol=0)
        return solution
    
    def solve_hhl(self, epsilon=0.1):
        start = time.time()
        def power1(self, opt = None) -> "QuantumCircuit":
            
            pauli_op = SparsePauliOp.from_operator(opt.matrix).simplify()
            gate = PauliEvolutionGate(pauli_op)
            st = SuzukiTrotter(order = 1, reps = 1)
            #st = QDrift(reps = 1)
            #st = LieTrotter(reps=4)
            #st = MatrixExponential()
            gate.time = 1 * np.pi
            trotter_circuit = st.synthesize(gate)
            trotter_circuit = transpile(trotter_circuit, optimization_level=3)
            return trotter_circuit
        
        #def test_decomp(self,qc):
        #    pass_manager = PassManager()
        #    pass_manager.append(Unroller(['u1', 'u2', 'u3', 'cx']))
        #    decomposed_circuit = pass_manager.run(qc)
        #    return decomposed_circuit
        
        QuantumCircuit.power1 = power1
        #QuantumCircuit.test_decomp = test_decomp
        #HHL.calculate_norm_efficiently = calculate_norm_efficiently

        if self.A is None:
            raise Exception("Not initialised")
        # Construct the circuit
        A = self.A.todense()
        b = self.b
        
        A, b = upscale_pow2(A,b)
        #print(A,b)
        
        b_circuit = QuantumCircuit(QuantumRegister(int(np.log2(len(b)))), name="init")
        for i in range(int(np.log2(len(b)))):
            b_circuit.h(i)
        hhl_solver = HHL(epsilon=epsilon)
        circuit = hhl_solver.construct_circuit(A, b_circuit, neg_vals=False)
        #if circuit_only: return circuit
        print("HHL time taken:", time.time()-start)
        print('Depth:',circuit.depth())


        #IBMQ.save_account('20691a602d3a3a344177bb53e776576df5b522c10a4934d22275e5cdfb2a86748ec79c286ec9bc0cc0b8fc65fedfbdf48154663069b9d034aa4afbdfc89397bb')
        #provider = IBMQ.load_account()
        #backend = provider.get_backend('ibm_brisbane')
        #noise_model = NoiseModel.from_backend(backend)
        #print(noise_model)
        #
        backend = FakeHanoiV2()
        noise_model = NoiseModel.from_backend(backend)
        #print(noise_model)


        #for qubit in circuit.qubits:
        #    gate_count = count_gates_on_qubit(circuit, qubit)
        #    print("Number of gates on qubit", qubit, ":", gate_count)
        #print(circuit.decompose().decompose().decompose())
        
        print('Transpiled Depth:', transpile(circuit, backend=backend, optimization_level=0).depth(), transpile(circuit, backend=backend, optimization_level=0).count_ops())
        
        #decomposed_circuit = circuit.decompose().decompose().decompose().decompose().decompose().decompose()
        #print(circuit)
        #print(decomposed_circuit)
        #num_layers = 0
        #for instruction in decomposed_circuit:
        #    num_layers += 1
        #print("Depth of the circuit:", num_layers)
        #print(decomposed_circuit.count_ops())
        

        state_vector = Statevector(circuit)
        solution_norm = np.linalg.norm(state_vector)  # Calculate the solution norm
        post_select_qubit = int(np.log2(len(state_vector.data))) - 1  # Pick the correct slice and renormalize it back
        solution_len = len(b)
        base = 1 << post_select_qubit
        solution_vector = state_vector.data[base: base + solution_len].real
        solution_vector = solution_vector / np.linalg.norm(solution_vector) * solution_norm * np.linalg.norm(b)


        
        sims = ['aer_simulator',
            'aer_simulator_statevector',
            'aer_simulator_density_matrix',
            'aer_simulator_stabilizer',
            'aer_simulator_matrix_product_state',
            'aer_simulator_extended_stabilizer',
            'aer_simulator_unitary',
            'qasm_simulator',
            'statevector_simulator',
            'unitary_simulator']
        #noise_model = {"depolarizing": {"probability": 0.8}} 
        #noise_model_object = NoiseModel.from_dict(noise_model)
        solution_vector1 = np.zeros(len(self.b))
        backend = "statevector_simulator1"
        if backend == "statevector_simulator":
            solution_vector1 = self.statevector_post_processing(circuit, noise_model)

        #for sim in sims:
        #    print(sim)
        #    simulator = Aer.get_backend(sim)
        #    simulator.noise_model = noise_model
        #    state_vector = execute(circuit, simulator, shots=4000).result()
        #    #print(state_vector.data())
        #print(are_close, solution_vector1[:len(self.b)] - solution_vector[:len(self.b)])
        # Use this function in your solve_hhl method
        return solution_vector[:len(self.b)], circuit, solution_vector1[:len(self.b)]
    
    def statevector_post_processing(self, circuit, noise_model = None):
        simulator = Aer.get_backend("statevector_simulator")
        if noise_model is not None: simulator.noise_model = noise_model
        state_vector = execute(circuit, simulator, noise_model = noise_model, shots=10000).result()
        state_vector = state_vector.get_statevector()
        solution_norm1 = np.linalg.norm(state_vector)
        post_select_qubit1 = int(np.log2(len(state_vector.data))) - 1 
        solution_len1 = len(self.b)
        base1 = 1 << post_select_qubit1
        solution_vector1 = state_vector.data[base1: base1 + solution_len1].real
        solution_vector1 = solution_vector1 / np.linalg.norm(solution_vector1) * solution_norm1 * np.linalg.norm(self.b)
        return solution_vector1

    def solve_hhl_original(self, epsilon=0.01, circuit_only=False):
        if self.A is None:
            raise Exception("Not initialised")
        # Construct the circuit
        A = self.A.todense()
        b = self.b
        
        A, b = upscale_pow2(A,b)
        
        b_circuit = QuantumCircuit(QuantumRegister(int(np.log2(len(b)))), name="init")
        for i in range(int(np.log2(len(b)))):
            b_circuit.h(i)
        
        hhl_solver = HHL(epsilon=epsilon)
        circuit = hhl_solver.construct_circuit(A, b_circuit, neg_vals=False)
        if circuit_only: return circuit
        
        # Get the final state vector
        state_vector = Statevector(circuit)
        solution_norm = hhl_solver._calculate_norm(circuit)
        
        # Pick the correct slice and renormalise it back
        post_select_qubit = int(np.log2(len(state_vector.data)))-1
        solution_len = len(b)
        base = 1 << post_select_qubit
        solution_vector = state_vector.data[base : base+solution_len].real
        solution_vector = solution_vector/np.linalg.norm(solution_vector)*solution_norm*np.linalg.norm(b)
        
        
        return solution_vector[:len(self.b)]


    def evaluate(self, solution):
        if self.A is None:
            raise Exception("Not initialised")
        
        if isinstance(solution, list):
            sol = np.array([solution, None])
        elif isinstance(solution, np.ndarray):
            if solution.ndim == 1:
                sol = solution[..., None]
            else: sol = solution
            
            
        return -0.5 * sol.T @ self.A @ sol + self.b.dot(sol)
        