"""
Mutation operators for evolutionary model search.

Implements various mutation strategies for model configurations.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import numpy as np
import random
import copy


@dataclass
class MutationConfig:
    """Configuration for mutation operations."""
    mutation_rate: float = 0.15
    mutation_strength: float = 0.2  # Standard deviation for numeric mutations

    # Parameter-specific mutation rates
    numeric_mutation_rate: float = 0.3
    categorical_mutation_rate: float = 0.1
    structural_mutation_rate: float = 0.05

    # Mutation types to use
    allowed_mutations: list[str] = field(default_factory=lambda: [
        "gaussian", "uniform", "categorical", "structural"
    ])


class ModelMutator:
    """
    Applies mutations to model configurations.

    Supports:
    - Gaussian noise for numeric parameters
    - Uniform random for bounded parameters
    - Categorical changes
    - Structural changes (add/remove layers, etc.)
    """

    def __init__(self, config: Optional[MutationConfig] = None):
        self.config = config or MutationConfig()

        # Define parameter ranges for common model configs
        self._param_ranges = {
            # Hyperparameters
            "learning_rate": (1e-5, 1e-1, "log"),
            "batch_size": (16, 512, "int"),
            "hidden_size": (32, 512, "int"),
            "num_layers": (1, 6, "int"),
            "dropout": (0.0, 0.5, "float"),
            "attention_heads": (1, 16, "int"),

            # Trading-specific
            "lookback_window": (24, 720, "int"),
            "prediction_horizon": (1, 24, "int"),
            "ensemble_size": (3, 15, "int"),
            "stop_loss_pct": (0.01, 0.15, "float"),
            "take_profit_pct": (0.02, 0.30, "float"),

            # Feature selection
            "feature_count": (10, 100, "int"),
            "lag_features": (1, 48, "int"),
        }

    def mutate(self, genotype: dict) -> dict:
        """
        Apply random mutations to a genotype.

        Args:
            genotype: Original model configuration

        Returns:
            Mutated genotype (new copy)
        """
        mutated = copy.deepcopy(genotype)

        # Decide whether to mutate each parameter
        for key in mutated:
            if random.random() > self.config.mutation_rate:
                continue

            # Apply mutation based on parameter type
            if key in self._param_ranges:
                mutated[key] = self._mutate_known_param(key, mutated[key])
            elif isinstance(mutated[key], (int, float)):
                mutated[key] = self._mutate_numeric(mutated[key])
            elif isinstance(mutated[key], bool):
                mutated[key] = not mutated[key]  # Flip boolean
            elif isinstance(mutated[key], str):
                mutated[key] = mutated[key]  # Keep strings unchanged for now

        return mutated

    def _mutate_known_param(self, key: str, value: Any) -> Any:
        """Mutate a parameter with known range."""
        min_val, max_val, param_type = self._param_ranges[key]

        if param_type == "log":
            # Log-uniform mutation for scale parameters
            log_min = np.log(min_val)
            log_max = np.log(max_val)
            log_value = np.log(value)
            log_value += np.random.normal(0, self.config.mutation_strength)
            log_value = np.clip(log_value, log_min, log_max)
            return np.exp(log_value)

        elif param_type == "int":
            # Integer mutation
            noise = np.random.normal(0, self.config.mutation_strength * (max_val - min_val))
            new_value = int(value + noise)
            return max(min_val, min(max_val, new_value))

        elif param_type == "float":
            # Float mutation
            noise = np.random.normal(0, self.config.mutation_strength * (max_val - min_val))
            new_value = value + noise
            return max(min_val, min(max_val, new_value))

        return value

    def _mutate_numeric(self, value: float) -> float:
        """Mutate an unknown numeric parameter."""
        # Apply Gaussian noise
        noise = np.random.normal(0, abs(value) * self.config.mutation_strength + 0.01)
        return value + noise

    def batch_mutate(self, genotypes: list[dict]) -> list[dict]:
        """Apply mutations to multiple genotypes."""
        return [self.mutate(g) for g in genotypes]

    def crossover(self, parent1: dict, parent2: dict) -> tuple[dict, dict]:
        """
        Perform crossover between two parents.

        Args:
            parent1: First parent genotype
            parent2: Second parent genotype

        Returns:
            Tuple of two child genotypes
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Get all keys
        all_keys = set(parent1.keys()) | set(parent2.keys())

        for key in all_keys:
            if random.random() < 0.5:
                # Swap values between children
                val1 = child1.get(key, parent2.get(key))
                val2 = child2.get(key, parent1.get(key))
                if val1 is not None:
                    child1[key] = val1
                if val2 is not None:
                    child2[key] = val2

        return child1, child2

    def structured_mutate(
        self,
        genotype: dict,
        mutation_type: str = "random"
    ) -> dict:
        """
        Apply structured mutation to architecture.

        Args:
            genotype: Model configuration
            mutation_type: Type of structural mutation

        Returns:
            Mutated genotype
        """
        mutated = copy.deepcopy(genotype)

        if mutation_type == "add_layer":
            if "num_layers" in mutated:
                mutated["num_layers"] = mutated.get("num_layers", 1) + 1

        elif mutation_type == "remove_layer":
            if "num_layers" in mutated and mutated["num_layers"] > 1:
                mutated["num_layers"] -= 1

        elif mutation_type == "widen":
            if "hidden_size" in mutated:
                scale = random.uniform(1.1, 1.5)
                mutated["hidden_size"] = int(mutated["hidden_size"] * scale)

        elif mutation_type == "narrow":
            if "hidden_size" in mutated:
                scale = random.uniform(0.7, 0.9)
                mutated["hidden_size"] = max(16, int(mutated["hidden_size"] * scale))

        elif mutation_type == "random":
            # Random structural change
            mutations = ["add_layer", "remove_layer", "widen", "narrow"]
            valid_mutations = [m for m in mutations if m in self.config.allowed_mutations]
            if valid_mutations:
                mutation_type = random.choice(valid_mutations)
                return self.structured_mutate(genotype, mutation_type)

        return mutated


def apply_adaptive_mutation(
    genotype: dict,
    fitness: float,
    best_fitness: float,
    worst_fitness: float
) -> dict:
    """
    Apply adaptive mutation based on fitness.

    Lower fitness individuals get stronger mutations.
    """
    if best_fitness == worst_fitness:
        strength = 0.2
    else:
        # Normalize fitness to [0, 1]
        normalized = (fitness - worst_fitness) / (best_fitness - worst_fitness)
        # Lower fitness = stronger mutation
        strength = 0.1 + 0.3 * (1 - normalized)

    config = MutationConfig(mutation_strength=strength)
    mutator = ModelMutator(config)
    return mutator.mutate(genotype)
