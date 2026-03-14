"""
Model population management for evolutionary search.

Tracks individuals, their fitness, and population dynamics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any, Callable
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
import json
import copy


@dataclass
class Individual:
    """
    Single individual in the evolutionary population.

    Represents a model configuration with its fitness.
    """
    id: str
    genotype: dict  # Model configuration
    fitness: float = 0.0
    age: int = 0
    generations_alive: int = 0
    evaluations: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    parent_id: Optional[str] = None
    birth_time: datetime = None
    last_evaluation: datetime = None

    def __post_init__(self):
        if self.birth_time is None:
            self.birth_time = datetime.now(timezone.utc)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def sharpe(self) -> float:
        # Proxy Sharpe based on total PnL and evaluations
        if self.evaluations < 2:
            return 0.0
        return self.total_pnl / (self.evaluations ** 0.5) if self.evaluations > 0 else 0.0

    def update_fitness(self, new_fitness: float):
        """Update fitness with rolling average."""
        if self.evaluations == 0:
            self.fitness = new_fitness
        else:
            # Exponential moving average
            alpha = 0.3
            self.fitness = alpha * new_fitness + (1 - alpha) * self.fitness

        self.evaluations += 1
        self.last_evaluation = datetime.now(timezone.utc)


@dataclass
class PopulationConfig:
    """Configuration for model population."""
    population_size: int = 20
    elite_size: int = 4
    mutation_rate: float = 0.15
    crossover_rate: float = 0.3
    min_fitness_threshold: float = 0.0
    max_age: Optional[int] = None  # None = no age limit
    diversity_threshold: float = 0.1


class ModelPopulation:
    """
    Manages a population of model configurations.

    Supports:
    - Adding/removing individuals
    - Fitness tracking
    - Diversity measurement
    - Population statistics
    """

    def __init__(self, config: Optional[PopulationConfig] = None):
        self.config = config or PopulationConfig()
        self._individuals: dict[str, Individual] = {}
        self._fitness_history: deque[float] = deque(maxlen=1000)
        self._generation: int = 0
        self._next_id: int = 0

    def add_individual(
        self,
        genotype: dict,
        parent_id: Optional[str] = None
    ) -> Individual:
        """Add a new individual to population."""
        individual_id = f"model_{self._next_id}"
        self._next_id += 1

        individual = Individual(
            id=individual_id,
            genotype=copy.deepcopy(genotype),
            parent_id=parent_id
        )

        self._individuals[individual_id] = individual
        return individual

    def remove_individual(self, individual_id: str) -> Optional[Individual]:
        """Remove an individual from population."""
        return self._individuals.pop(individual_id, None)

    def get_individual(self, individual_id: str) -> Optional[Individual]:
        """Get individual by ID."""
        return self._individuals.get(individual_id)

    def get_all_individuals(self) -> list[Individual]:
        """Get all individuals."""
        return list(self._individuals.values())

    def get_elite(self, n: Optional[int] = None) -> list[Individual]:
        """Get top n individuals by fitness."""
        n = n or self.config.elite_size
        sorted_inds = sorted(
            self._individuals.values(),
            key=lambda x: x.fitness,
            reverse=True
        )
        return sorted_inds[:n]

    def get_worst(self, n: int = 1) -> list[Individual]:
        """Get worst n individuals by fitness."""
        sorted_inds = sorted(
            self._individuals.values(),
            key=lambda x: x.fitness
        )
        return sorted_inds[:n]

    def update_fitness(self, individual_id: str, fitness: float):
        """Update fitness of an individual."""
        if individual_id in self._individuals:
            self._individuals[individual_id].update_fitness(fitness)
            self._fitness_history.append(fitness)

    def age_population(self):
        """Increment age of all individuals."""
        for ind in self._individuals.values():
            ind.age += 1
            ind.generations_alive += 1

    def remove_old(self) -> list[Individual]:
        """Remove individuals exceeding max age."""
        if self.config.max_age is None:
            return []

        removed = []
        to_remove = [
            ind_id for ind_id, ind in self._individuals.items()
            if ind.age > self.config.max_age
        ]

        for ind_id in to_remove:
            removed.append(self.remove_individual(ind_id))

        return removed

    def remove_weak(self, threshold: Optional[float] = None) -> list[Individual]:
        """Remove individuals below fitness threshold."""
        threshold = threshold or self.config.min_fitness_threshold

        to_remove = [
            ind_id for ind_id, ind in self._individuals.items()
            if ind.fitness < threshold
        ]

        removed = []
        for ind_id in to_remove:
            removed.append(self.remove_individual(ind_id))

        return removed

    def get_population_stats(self) -> dict:
        """Calculate population statistics."""
        if not self._individuals:
            return {
                "size": 0,
                "avg_fitness": 0,
                "max_fitness": 0,
                "min_fitness": 0,
                "diversity": 0,
                "generation": self._generation
            }

        fitnesses = [ind.fitness for ind in self._individuals.values()]

        return {
            "size": len(self._individuals),
            "avg_fitness": np.mean(fitnesses),
            "max_fitness": np.max(fitnesses),
            "min_fitness": np.min(fitnesses),
            "std_fitness": np.std(fitnesses),
            "diversity": self._calculate_diversity(),
            "generation": self._generation,
            "total_evaluations": sum(ind.evaluations for ind in self._individuals.values())
        }

    def _calculate_diversity(self) -> float:
        """Calculate genetic diversity in population."""
        if len(self._individuals) < 2:
            return 0.0

        # Calculate pairwise differences in genotypes
        genotypes = [ind.genotype for ind in self._individuals.values()]

        # Use Hamming-like distance for dict keys
        total_distance = 0
        comparisons = 0

        for i, g1 in enumerate(genotypes):
            for g2 in genotypes[i+1:]:
                # Count parameter differences
                all_keys = set(g1.keys()) | set(g2.keys())
                diff_count = 0

                for key in all_keys:
                    v1 = g1.get(key)
                    v2 = g2.get(key)

                    if v1 != v2:
                        diff_count += 1

                total_distance += diff_count / len(all_keys) if all_keys else 0
                comparisons += 1

        return total_distance / comparisons if comparisons > 0 else 0.0

    def is_diverse_enough(self) -> bool:
        """Check if population has enough diversity."""
        return self._calculate_diversity() >= self.config.diversity_threshold

    def increment_generation(self):
        """Increment generation counter."""
        self._generation += 1

    def save(self, path: Path):
        """Save population state."""
        data = {
            "generation": self._generation,
            "next_id": self._next_id,
            "config": {
                "population_size": self.config.population_size,
                "elite_size": self.config.elite_size,
                "mutation_rate": self.config.mutation_rate,
                "crossover_rate": self.config.crossover_rate,
            },
            "individuals": {
                ind_id: {
                    "id": ind.id,
                    "genotype": ind.genotype,
                    "fitness": ind.fitness,
                    "age": ind.age,
                    "evaluations": ind.evaluations,
                    "wins": ind.wins,
                    "losses": ind.losses,
                    "total_pnl": ind.total_pnl,
                    "parent_id": ind.parent_id,
                    "birth_time": ind.birth_time.isoformat() if ind.birth_time else None,
                }
                for ind_id, ind in self._individuals.items()
            }
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path):
        """Load population state."""
        if not path.exists():
            return

        with open(path) as f:
            data = json.load(f)

        self._generation = data.get("generation", 0)
        self._next_id = data.get("next_id", 0)

        for ind_id, ind_data in data.get("individuals", {}).items():
            individual = Individual(
                id=ind_data["id"],
                genotype=ind_data["genotype"],
                fitness=ind_data.get("fitness", 0),
                age=ind_data.get("age", 0),
                evaluations=ind_data.get("evaluations", 0),
                wins=ind_data.get("wins", 0),
                losses=ind_data.get("losses", 0),
                total_pnl=ind_data.get("total_pnl", 0),
                parent_id=ind_data.get("parent_id"),
                birth_time=datetime.fromisoformat(ind_data["birth_time"]) if ind_data.get("birth_time") else None
            )
            self._individuals[ind_id] = individual


def create_initial_population(
    base_config: dict,
    population_size: int = 10
) -> ModelPopulation:
    """
    Create initial population from base configuration.

    Generates variants by randomizing parameters.
    """
    population = ModelPopulation(PopulationConfig(population_size=population_size))

    # Add base configuration
    population.add_individual(base_config)

    # Generate variants
    for _ in range(population_size - 1):
        variant = copy.deepcopy(base_config)

        # Randomize some parameters
        for key in variant:
            if isinstance(variant[key], (int, float)):
                # Add noise to numeric parameters
                noise = np.random.normal(0, 0.2)
                if isinstance(variant[key], int):
                    variant[key] = max(1, int(variant[key] * (1 + noise)))
                else:
                    variant[key] = variant[key] * (1 + noise)

        population.add_individual(variant)

    return population
