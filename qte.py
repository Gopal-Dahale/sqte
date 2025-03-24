from qiskit_algorithms import TrotterQRTE
from utils import estimate_observables
from qiskit_algorithms.time_evolvers.time_evolution_problem import TimeEvolutionProblem
from qiskit_algorithms.time_evolvers.time_evolution_result import TimeEvolutionResult
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.transpiler import PassManager
from qiskit.synthesis import ProductFormula, LieTrotter
from qiskit.primitives import BaseEstimator


class QTE(TrotterQRTE):
    def __init__(
        self,
        product_formula: ProductFormula | None = None,
        estimator: BaseEstimator | None = None,
        pm: PassManager | None = None,
        num_timesteps: int = 1,
        verbose: bool = False,
    ) -> None:

        self.product_formula = product_formula
        self.num_timesteps = num_timesteps
        self.estimator = estimator
        self.pm = pm
        self.verbose = verbose

    def evolve(self, evolution_problem: TimeEvolutionProblem) -> TimeEvolutionResult:

        if evolution_problem.aux_operators is not None and (
            self.estimator is None or self.pm is None
        ):
            raise ValueError(
                "The time evolution problem contained ``aux_operators`` but either estimator or "
                "pass manager was not provided. The algorithm continues without calculating these quantities. "
            )

        # ensure the hamiltonian is a sparse pauli op
        hamiltonian = evolution_problem.hamiltonian
        if not isinstance(hamiltonian, (Pauli, SparsePauliOp)):
            raise ValueError(
                f"QTE only accepts Pauli | SparsePauliOp, {type(hamiltonian)} "
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

        if evolution_problem.initial_state is not None:
            initial_state = evolution_problem.initial_state
        else:
            raise ValueError(
                "``initial_state`` must be provided in the ``TimeEvolutionProblem``."
            )

        evolved_state = QuantumCircuit(initial_state.num_qubits)
        evolved_state.append(initial_state, evolved_state.qubits)

        if evolution_problem.aux_operators is not None:
            # Transpile the circuit
            isa_qc = self.pm.run(evolved_state)
            isa_hams = [
                ham.apply_layout(isa_qc.layout)
                for ham in evolution_problem.aux_operators
            ]

            observables = []
            observables.append(
                estimate_observables(
                    self.estimator,
                    isa_qc,
                    isa_hams,
                    None,
                    evolution_problem.truncation_threshold,
                )
            )
        else:
            observables = None

        # Empty define to avoid possibly undefined lint error later here
        single_step_evolution_gate = None

        if t_param is None:
            # the evolution gate
            single_step_evolution_gate = PauliEvolutionGate(
                hamiltonian, dt, synthesis=self.product_formula
            )

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

            if evolution_problem.aux_operators is not None:
                # Transpile the circuit
                isa_qc = self.pm.run(evolved_state)
                isa_hams = [
                    ham.apply_layout(isa_qc.layout)
                    for ham in evolution_problem.aux_operators
                ]

                observables.append(
                    estimate_observables(
                        self.estimator,
                        isa_qc,
                        isa_hams,
                        None,
                        evolution_problem.truncation_threshold,
                    )
                )

        evaluated_aux_ops = None
        if evolution_problem.aux_operators is not None:
            evaluated_aux_ops = observables[-1]

        return TimeEvolutionResult(
            evolved_state, evaluated_aux_ops, observables  # type: ignore[arg-type]
        )
