""" Perform Bayesian optimization to find the best parameters for a given objective function. """
from typing import Dict, List, Tuple, Any
import abc
import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass
from pathlib import Path
import toml
import matplotlib.pyplot as plt
import numpy as np

try:
    import bayex
except ImportError:
    raise ImportError("ERROR: Bayex not installed, use pip to install before using this function")

class BOObjective(abc.ABC):
    """
    Abstract base class for Bayesian Optimization objectives

    This class serves as an interface. To use it, a user must subclass it and
    implement their own `forward_simulation` and `loss_function` methods
    tailored to their specific scientific problem
    """
    def __init__(self, static_sim_data: Dict[str, Any], float_type='float32'):
        """Initializes the objective and JIT-compiles the evaluation functions."""
        self.static_sim_data = static_sim_data
        loss_fn_with_static_data = partial(self.loss_function, static_data=self.static_sim_data)

        # JIT-compile the function for a single evaluation.
        self._value_and_grad_fn = jax.jit(
            jax.value_and_grad(loss_fn_with_static_data)
        )
        self._vmapped_value_and_grad_fn = jax.jit(
            jax.vmap(jax.value_and_grad(loss_fn_with_static_data))
        )

        self._float_dtype = jnp.float64 if (float_type.lower() == 'float64') else jnp.float32

    @abc.abstractmethod
    def forward_simulation(self, params: Dict[str, jnp.ndarray], static_data: Dict[str, Any]) -> jnp.ndarray:
        """
        The user's core "expensive" simulation logic goes here.

        Args:
            params: A dictionary mapping parameter names (e.g., 'epsilon', 'deltaH')
                    to their JAX array values.

        Returns:
            A JAX array representing the simulation's output feature to be targeted.
        """
        pass

    @abc.abstractmethod
    def final_forward_simulation(self, params: Dict[str, jnp.ndarray], static_data: Dict[str, Any]) -> jnp.ndarray:
        """ Final simulation that will WRITE the export files """
        pass

    @abc.abstractmethod
    def loss_function(self, params: Dict[str, jnp.ndarray], static_data: Dict[str, Any]) -> jnp.ndarray:
        """
        Calculates the scalar loss based on the simulation output.

        Args:
            params: A dictionary of parameters to be passed to the simulation.

        Returns:
            A scalar JAX array representing the loss.
        """
        pass

    def evaluate(self, params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the loss and gradient for a single set of parameters.
        This is the primary interface for a standard, sequential BO loop.

        Args:
            params: A dictionary of standard Python floats for the parameters

        Returns:
            A tuple of (loss, gradient_dict).
        """
        params_jnp = {k: jnp.array(v) for k, v in params.items()}
        loss, grad_jnp = self._value_and_grad_fn(params_jnp)
        grad_py = {k: v.item() for k, v in grad_jnp.items()}
        return loss.item(), grad_py

    def evaluate_batch(self, params_batch: List[Dict[str, float]]) -> Tuple[List[float], List[Dict[str, float]]]:
        """
        Evaluates a batch of parameter sets in parallel using vmap.
        This is the interface for a parallelized BO scheduler.

        Args:
            params_batch: A list of parameter dictionaries.

        Returns:
            A tuple of (list_of_losses, list_of_gradient_dicts).
        """
        if not params_batch:
            return [], []
        pytree = {k: jnp.array([d[k] for d in params_batch]) for k in params_batch[0]}
        losses_jnp, grads_jnp = self._vmapped_value_and_grad_fn(pytree)
        num_items = len(params_batch)
        grads_py = [{k: v[i].item() for k, v in grads_jnp.items()} for i in range(num_items)]
        return losses_jnp.tolist(), grads_py

@dataclass
class BOArgs:
    """
    Arguments for running Bayesian Optimization using bayex
    """
    write_location: Path | str = ''  # Default directory is wherever user runs script from
    num_iterations: int = 20
    num_initial_points: int = 5
    batch_size: int = 4  # Number of simulations to run in parallel per iteration
    parameter_bounds: Dict[str, Tuple[float, float]] = None  # {Parameter1: (lb1, ub1), Parameter2: (lb2, ub2), ...}
    verbose: bool = False
    maximize: bool = False
    acquisition_fn: str = 'PI'
    seed: int = 8

def run_bayesian_optimization(objective: BOObjective, args: BOArgs, visualize_gp: bool = False):
    """ Runs a simple Bayesian Optimization loop using the provided objective and arguments. """
    final_location = Path(args.write_location) / 'results'
    final_location.mkdir(parents=True, exist_ok=True)  # Create a results folder for storing results

    ori_key = jax.random.key(args.seed)
    ini_key, loop_key = jax.random.split(ori_key)
    acq_fn = args.acquisition_fn.upper()

    if acq_fn not in ['EI', 'PI', 'UCB', 'LCB']:
        raise ValueError("ERROR: Acquisition function must be one of `EI`, `PI`, `UCB`, `LCB` for use with bayex.")
    if args.parameter_bounds is None:
        raise ValueError("ERROR: Parameter bounds must be specified in BOArgs.")

    # Initialize the optimizer with the specified bounds
    param_names = list(args.parameter_bounds.keys())
    bounds = jnp.array([args.parameter_bounds[name] for name in param_names])  # shape (n_params, 2)
    lows, highs = bounds[:, 0], bounds[:, 1]

    # Setup design space / bayex optimizer
    space = {name: bayex.domain.Real(low, high) for name, (low, high) in args.parameter_bounds.items()}
    optimizer = bayex.Optimizer(
        domain=space,
        maximize=args.maximize,
        acq=acq_fn
    )

    # Get some prior evaluations to initialize GP
    ini_key, subkey = jax.random.split(ini_key)
    initial_points = jax.random.uniform(
        subkey, shape=(args.num_initial_points, len(param_names)), minval=lows, maxval=highs
    )
    initial_points_dicts = [dict(zip(param_names, p)) for p in initial_points]  # Formatted for evaluator
    params = {i: [] for i in param_names}
    for d in initial_points_dicts:
        for k, v in d.items():
            params[k].append(float(v))

    # Batch out initial_points_dicts and evaluate:
    batches = [initial_points_dicts[i:i + args.batch_size] for i in range(0, len(initial_points_dicts), args.batch_size)]
    all_losses = []
    for b in batches:
        initial_losses, _ = objective.evaluate_batch(b)
        all_losses.extend(initial_losses)

    opt_state = optimizer.init(all_losses, params)
    observed_data = {i: {'x': [], 'y': []} for i in param_names}

    for i in range(args.num_iterations):
        if args.verbose:
            print(f'Beginning iteration {i}')
        key = jax.random.fold_in(loop_key, i)
        new_params = optimizer.sample(key, opt_state)                # TODO: Create custom sampler for batch evals??
        losses, grads = objective.evaluate(new_params)
        _store(observed_data, new_params, losses)
        opt_state = optimizer.fit(opt_state, losses, new_params)

        # Note: Look to use something like BoTorch to use the gradient information instead, right now scikit-optimize
        # does not use this information natively
        if args.verbose:
            if isinstance(losses, list):
                print(f"  - Observed losses at {i}: {[f'{l:.4f}' for l in losses]}")
            else:
                print(f"  - Observed loss at {i}: {losses}")

    # Finally, get the best results for returning:
    best_result = opt_state.best_score
    best_params_dict = opt_state.best_params

    if args.verbose:
        print(f"Minimum loss found: {best_result:.6f}")
        print(f"Optimal parameters: {best_params_dict}")

    final_sim_eval_value, final_toml = objective.final_forward_simulation(
        {k: jnp.array(v) for k, v in best_params_dict.items()},
        objective.static_sim_data
    )

    output_toml = final_location / 'optimized_input_file.toml'
    with open(output_toml, "w") as f:
        toml.dump(final_toml, f)

    if args.verbose:
        print(f"Bayesian Optimization finished...")
        print(f"  - Predicted eval at optimum: {final_sim_eval_value.item():.4f}")

    _visualize_gp(opt_state, observed_data, args, final_location, visualize_gp)
    return best_result, best_params_dict

def _store(tracker_dict, new_xs, new_ys):
    """ Stores xs / ys into the dict """
    for k, v in new_xs.items():
        if isinstance(v, list):
            for vv in v:
                tracker_dict[k]['x'].append(float(vv))
        else:
            tracker_dict[k]['x'].append(float(v))

        # Same losses for all parameters:
        if isinstance(new_ys, list):
            for yy in new_ys:
                tracker_dict[k]['y'].append(float(yy))
        else:
            tracker_dict[k]['y'].append(float(new_ys))


def _visualize_gp(opt_state, observed_data: dict, args: BOArgs, final_location: Path, show_plot: bool = False):
    # Visualize the gaussian process, relies on the bayex optimizer (not compatible with skopt)
    param_name = list(args.parameter_bounds.keys())
    from bayex.gp import gaussian_process

    for p in param_name:
        # Collect relevent data:
        bounds = args.parameter_bounds[p]
        x_plot = jnp.linspace(bounds[0], bounds[1], 200).reshape(-1, 1)
        xs = opt_state.params[p]
        xs = xs[:, None]
        mean, std = gaussian_process(opt_state.gp_state.params, xs, opt_state.ys, opt_state.mask, xt=x_plot)

        # Create Plot
        mean = np.array(mean)
        std = np.array(std)
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, mean, 'b-', label='GP Mean (Estimate)')
        plt.fill_between(x_plot.flatten(), mean - 1.96 * std, mean + 1.96 * std,
                         color='blue', alpha=0.15, label='95% Confidence Interval')

        x_observed, y_observed = np.array(observed_data[p]['x']), np.array(observed_data[p]['y'])
        plt.plot(x_observed, y_observed, 'ko', markersize=8, label='Observations')

        best_x = x_observed[np.argmin(y_observed)]
        best_y = np.min(y_observed)
        plt.plot(best_x, best_y, 'y*', markersize=15, markeredgecolor='k', label=f'Best Found: {best_x:.2f}')

        plt.title('Bayesian Optimization of Epsilon Parameter')
        plt.xlabel(f'{p}')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        if show_plot:
            plt.show()
        op = final_location / f'bayes_opt_{p}_results.png'
        plt.savefig(op, dpi=400)
        plt.close()


def run_bayesian_optimization_skopt(objective: BOObjective, args: BOArgs):
    """ Runs a simple Bayesian Optimization loop using the provided objective and arguments. """
    # NOTE: This is NOT the recommended optimizer, the bayex version is more customized for export
    try:
        from skopt import Optimizer
    except ImportError:
        raise ImportError("ERROR: scikit-optimize not installed, use pip to install before using this function")
    from skopt.space import Real
    import warnings
    warnings.filterwarnings("ignore", module="skopt")  # Remove eventually, artefact of using scikit-optimize

    if args.parameter_bounds is None:
        raise ValueError("Parameter bounds must be specified in BOArgs.")

    # Initialize the optimizer with the specified bounds
    param_names = list(args.parameter_bounds.keys())
    space = [Real(low, high, name=name) for name, (low, high) in args.parameter_bounds.items()]
    optimizer = Optimizer(
        dimensions=space,
        random_state=1,
        n_initial_points=args.num_initial_points
    )

    for i in range(args.num_iterations):
        if args.verbose:
            print(f'Beginning iteration {i}')
        next_points_list = optimizer.ask(n_points=args.batch_size)  # Get next set of points to evaluate
        next_points_dicts = [dict(zip(param_names, p)) for p in next_points_list]
        losses, grads = objective.evaluate_batch(next_points_dicts)

        # Note: Look to use something like BoTorch to use the gradient information instead, right now scikit-optimize
        # does not use this information natively
        if args.verbose:
            print(f"  - Observed losses at {i}: {[f'{l:.4f}' for l in losses]}")

        optimizer.tell(next_points_list, losses)  # Update optimizer for next collection datapoints

    # Finally, get the best results for returning:
    best_result = optimizer.get_result()
    best_params_list = best_result.x
    best_params_dict = dict(zip(param_names, best_params_list))
    best_loss = best_result.fun

    if args.verbose:
        print(f"Minimum loss found: {best_loss:.6f}")
        print(f"Optimal parameters: {best_params_dict}")

    final_sim_eval_value = objective.forward_simulation(
        {k: jnp.array(v) for k, v in best_params_dict.items()},
        objective.static_sim_data
    )

    if args.verbose:
        print(f"Bayesian Optimization finished...")
        print(f"  - Predicted eval at optimum: {final_sim_eval_value.item():.4f}")

    return best_loss, best_params_dict

