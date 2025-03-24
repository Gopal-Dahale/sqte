from qte import QTE
from utils import sample_circuit, batch_conn, bin_to_int
from qiskit_algorithms.time_evolvers.time_evolution_problem import TimeEvolutionProblem
from sqte_result import SQTEResult
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.transpiler import PassManager
from qiskit.synthesis import ProductFormula, LieTrotter
from qiskit.primitives import BaseSampler
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm as sparse_expm
import jax.numpy as jnp


class SQTE(QTE):
	def __init__(
		self,
		product_formula: ProductFormula | None = None,
		sampler: BaseSampler | None = None,
		pm: PassManager | None = None,
		num_timesteps: int = 1,
		verbose: bool = False,
	) -> None:

		self.product_formula = product_formula
		self.num_timesteps = num_timesteps
		self.sampler = sampler
		self.pm = pm
		self.verbose = verbose

	def sort_and_remove_duplicates(self, samples):
		decimal_values = bin_to_int(samples)
		_, unique_indices = np.unique(decimal_values, return_index=True)
		return samples[unique_indices]

	def transpile(self, qc, hamiltonian, aux_operators=None):
		isa_qc = self.pm.run(qc)
		isa_hamiltonian = hamiltonian.apply_layout(isa_qc.layout)
		if aux_operators:
			isa_aux_operators = [
				aux_op.apply_layout(isa_qc.layout) for aux_op in aux_operators
			]
			return isa_qc, isa_hamiltonian, isa_aux_operators

		return isa_qc, isa_hamiltonian

	def project_observable(self, samples, observable):
		d, n = samples.shape
		operator = coo_matrix((d, d), dtype="complex128")

		for i, pauli in enumerate(observable.paulis):
			coeff = observable.coeffs[i]
			row_ids = np.arange(d)

			# qubit wise representation
			diag = np.logical_not(pauli.x)[::-1]
			sign = pauli.z[::-1]
			imag = np.logical_and(pauli.x, pauli.z)[::-1]

			decimal_values = bin_to_int(samples)

			samples_conn, amplitudes = batch_conn(samples, diag, sign, imag)
			decimal_conn = bin_to_int(samples_conn)

			conn_mask = np.isin(
				decimal_conn, decimal_values, assume_unique=True, kind="sort"
			)

			# keep samples that are represented both in the original samples
			# and connected elements
			amplitudes = amplitudes[conn_mask]
			decimal_conn = decimal_conn[conn_mask]
			row_ids = row_ids[conn_mask]

			# Get column indices of non-zero matrix elements
			col_ids = np.searchsorted(decimal_values, decimal_conn)

			operator += coeff * coo_matrix((amplitudes, (row_ids, col_ids)), (d, d))

		return operator

	def evolve(
		self, evolution_problem: TimeEvolutionProblem, union_samples=False
	) -> SQTEResult:

		if evolution_problem.aux_operators is not None and (
			self.sampler is None or self.pm is None
		):
			raise ValueError(
				"The time evolution problem contained ``aux_operators`` but either sampler or "
				"pass manager was not provided. The algorithm continues without calculating these quantities. "
			)

		# ensure the hamiltonian is a sparse pauli op
		hamiltonian = evolution_problem.hamiltonian
		if not isinstance(hamiltonian, (Pauli, SparsePauliOp)):
			raise ValueError(
				f"SQTE only accepts Pauli | SparsePauliOp, {type(hamiltonian)} "
				"provided."
			)

		if isinstance(hamiltonian, Pauli):
			hamiltonian = SparsePauliOp(hamiltonian)

		t_param = evolution_problem.t_param
		free_parameters = hamiltonian.parameters
		if t_param is not None and free_parameters != ParameterView([t_param]):
			raise ValueError(
				f"Hamiltonian time parameters ({free_parameters}) do not match "
				f"evolution_problem.t_param ({t_param})."
			)

		# make sure PauliEvolutionGate does not implement more than one Trotter step
		dt = evolution_problem.time / self.num_timesteps  # pylint: disable=invalid-name
		times = np.linspace(dt, evolution_problem.time, self.num_timesteps)

		if evolution_problem.initial_state is not None:
			initial_state = evolution_problem.initial_state
		else:
			raise ValueError(
				"``initial_state`` must be provided in the ``TimeEvolutionProblem``."
			)

		evolved_state = QuantumCircuit(initial_state.num_qubits)
		evolved_state.append(initial_state, evolved_state.qubits)

		# Empty define to avoid possibly undefined lint error later here
		single_step_evolution_gate = None

		if t_param is None:
			# the evolution gate
			single_step_evolution_gate = PauliEvolutionGate(
				hamiltonian, dt, synthesis=self.product_formula
			)

		energies = []
		observables = []

		samples = []

		for n in range(self.num_timesteps):
			if self.verbose:
				print(f"Time step: {n}")

			# if hamiltonian is time-dependent, bind new time-value at every step to construct
			# evolution for next step
			if t_param is not None:
				time_value = (n + 1) * dt
				bound_hamiltonian = hamiltonian.assign_parameters([time_value])
				single_step_evolution_gate = PauliEvolutionGate(
					bound_hamiltonian,
					dt,
					synthesis=self.product_formula,
				)
			evolved_state.append(single_step_evolution_gate, evolved_state.qubits)

			qc = evolved_state.copy()
			qc.measure_all()
			isa_qc, isa_hamiltonian, isa_aux_operators = self.transpile(
				qc, hamiltonian, evolution_problem.aux_operators
			)

			curr_samples = sample_circuit(self.sampler, isa_qc)

			if len(samples) and union_samples:
				samples = np.r_[samples, curr_samples]
			else:
				samples = curr_samples
				
			# add the basis state |00..0> to the samples
			samples = np.r_[samples, [np.zeros(initial_state.num_qubits, dtype=int)]]
			samples = self.sort_and_remove_duplicates(samples)

			if self.verbose and union_samples:
				print(f"Number of samples in total: {samples.shape[0]}")

			H_proj = self.project_observable(samples, isa_hamiltonian)

			# project initial state
			# currently hardcoded and only works with |00 .. 0>
			psi0_proj = np.zeros(samples.shape[0])
			psi0_proj[0] = 1.0

			# normalize if needed
			norm = np.linalg.norm(psi0_proj)
			if not np.allclose(norm, 1.0):
				psi0_proj /= norm

			U_proj = sparse_expm(-1j * times[n] * H_proj)
			psi_proj = U_proj @ psi0_proj

			# calculate energy
			energy = np.real(psi_proj.conj().T @ H_proj @ psi_proj)
			energies.append(energy)

			if evolution_problem.aux_operators is not None:

				obs = []
				for aux_op in isa_aux_operators:
					O_proj = self.project_observable(samples, aux_op)
					aux_exp = np.real(psi_proj.conj().T @ O_proj @ psi_proj)
					obs.append(aux_exp)

				observables.append(obs)

		evaluated_aux_ops = None
		if evolution_problem.aux_operators is not None:
			evaluated_aux_ops = observables[-1]

		return SQTEResult(
			evolved_state, evaluated_aux_ops, energies, observables  # type: ignore[arg-type]
		)
