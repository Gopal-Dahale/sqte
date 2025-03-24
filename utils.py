from qiskit_algorithms.observables_evaluator import _handle_zero_ops, _prepare_result
import numpy as np
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit import QuantumCircuit
from qiskit_algorithms.list_or_dict import ListOrDict
from collections.abc import Sequence
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Any, Union, List
from qiskit_algorithms.exceptions import AlgorithmError
import jax
import jax.numpy as jnp


def estimate_observables(
    estimator: BaseEstimator,
    quantum_state: QuantumCircuit,
    observables: ListOrDict[BaseOperator],
    parameter_values: Sequence[float] | None = None,
    threshold: float = 1e-12,
) -> ListOrDict[tuple[float, dict[str, Any]]]:
    if isinstance(observables, dict):
        observables_list = list(observables.values())
    else:
        observables_list = observables

    if len(observables_list) > 0:
        observables_list = _handle_zero_ops(observables_list)
        quantum_state = [quantum_state] * len(observables)
        parameter_values_: Sequence[float] | Sequence[Sequence[float]] | None = (
            parameter_values
        )

        pubs = list(zip(quantum_state, observables_list))
        if parameter_values is not None:
            parameter_values_ = [parameter_values] * len(observables)
            pubs = list(zip(quantum_state, observables_list, parameter_values_))

        try:
            estimator_job = estimator.run(pubs)
            estimator_result = estimator_job.result()
            expectation_values = np.array([res.data.evs for res in estimator_result])
            expectation_values = np.squeeze(expectation_values)
            stds = np.array([res.data.stds for res in estimator_result])
        except Exception as exc:
            raise AlgorithmError("The primitive job failed!") from exc

        metadata = estimator_job.result().metadata
        # Discard values below threshold
        observables_means = expectation_values * (
            np.abs(expectation_values) > threshold
        )
        # zip means and metadata into tuples
        observables_results = observables_means
    else:
        observables_results = []

    return _prepare_result(observables_results, observables)


def sample_circuit(
    sampler: BaseSampler,
    quantum_state: QuantumCircuit,
):
    pubs = (quantum_state,)
    try:
        job = sampler.run(pubs)
        samples = job.result()[0].data.meas.get_bitstrings()
        samples = np.array([list(x) for x in samples], dtype=int)
    except Exception as exc:
        raise AlgorithmError("The primitive job failed!") from exc

    return samples


def connected_elements_and_amplitudes_bool(bitstring, diag, sign, imag):
    """Find the connected element to computational basis state |X>."""
    bitstring_mask = bitstring == diag
    return bitstring_mask.astype(int), jnp.prod(
        (-1) ** (jnp.logical_and(bitstring, sign))
        * jnp.array(1j, dtype="complex64") ** (imag)
    )


batch_conn = jax.jit(
    jax.vmap(connected_elements_and_amplitudes_bool, (0, None, None, None))
)


def int_conversion_from_bts_array(bit_array: np.ndarray):
    """Convert a bit array to an integer representation."""
    n_qubits = len(bit_array)
    bitarray_asint = 0.0
    for i in range(n_qubits):
        bitarray_asint = bitarray_asint + bit_array[i] * 2 ** (n_qubits - 1 - i)

    return bitarray_asint.astype("longlong")  # type: ignore


bin_to_int = jax.jit(jax.vmap(int_conversion_from_bts_array, 0, 0))
