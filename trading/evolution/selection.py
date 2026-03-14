"""
Selection mechanisms for evolutionary model search.

Implements various selection strategies:
- Tournament selection
- Roulette wheel selection
- Elitism
- Truncation selection
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
import random

from .population import Individual


@dataclass
class SelectionConfig:
    """Configuration for selection operations."""
    method: str = "tournament"  # tournament, roulette, rank, truncation
    tournament_size: int = 3
    elite_size: int = 2
    selection_pressure: float = 2.0  # For rank selection
    min_fitness_threshold: float = -np.inf


class FitnessSelector:
    """
    Handles selection of individuals for reproduction.

    Supports multiple selection strategies.
    """

    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()

    def select(
        self,
        population: list[Individual],
        n: int
    ) -> list[Individual]:
        """
        Select n individuals from population.

        Args:
            population: List of individuals
            n: Number to select

        Returns:
            List of selected individuals
        """
        if not population:
            return []

        method = self.config.method

        if method == "tournament":
            return self._tournament_selection(population, n)
        elif method == "roulette":
            return self._roulette_selection(population, n)
        elif method == "rank":
            return self._rank_selection(population, n)
        elif method == "truncation":
            return self._truncation_selection(population, n)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _tournament_selection(
        self,
        population: list[Individual],
        n: int
    ) -> list[Individual]:
        """Tournament selection: best of k randomly chosen."""
        selected = []

        for _ in range(n):
            # Pick k random individuals
            tournament = random.sample(
                population,
                min(self.config.tournament_size, len(population))
            )
            # Select the best
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)

        return selected

    def _roulette_selection(
        self,
        population: list[Individual],
        n: int
    ) -> list[Individual]:
        """Roulette wheel selection: probability proportional to fitness."""
        # Handle negative fitness
        min_fitness = min(ind.fitness for ind in population)
        adjusted_fitness = [ind.fitness - min_fitness + 1 for ind in population]

        total = sum(adjusted_fitness)
        if total == 0:
            # All fitnesses equal, random selection
            return random.choices(population, k=n)

        probs = [f / total for f in adjusted_fitness]

        selected = []
        for _ in range(n):
            # Use cumulative distribution
            r = random.random()
            cumsum = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    selected.append(population[i])
                    break

        return selected

    def _rank_selection(
        self,
        population: list[Individual],
        n: int
    ) -> list[Individual]:
        """Rank selection: probability based on rank, not fitness."""
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        pop_size = len(sorted_pop)

        # Calculate rank probabilities
        # Linear ranking with selection pressure
        pressure = self.config.selection_pressure
        ranks = np.arange(1, pop_size + 1)

        # Probability: (2 - s)/n + (2r(s-1))/(n(n-1))
        probs = (2 - pressure) / pop_size + \
                (2 * ranks * (pressure - 1)) / (pop_size * (pop_size - 1))
        probs = np.maximum(probs, 0)  # Ensure non-negative
        probs = probs / probs.sum()  # Normalize

        selected = []
        for _ in range(n):
            idx = np.random.choice(pop_size, p=probs)
            selected.append(sorted_pop[idx])

        return selected

    def _truncation_selection(
        self,
        population: list[Individual],
        n: int
    ) -> list[Individual]:
        """Truncation selection: only select from top performers."""
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        top_n = max(n, len(population) // 2)  # At least top half

        if top_n >= len(population):
            return random.choices(population, k=n)

        return random.choices(sorted_pop[:top_n], k=n)

    def select_elites(
        self,
        population: list[Individual],
        n: Optional[int] = None
    ) -> list[Individual]:
        """Select elite individuals (best performers)."""
        n = n or self.config.elite_size
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:n]

    def select_for_removal(
        self,
        population: list[Individual],
        n: int,
        method: str = "worst"
    ) -> list[Individual]:
        """
        Select individuals for removal.

        Args:
            population: List of individuals
            n: Number to remove
            method: Selection method (worst, random, age)

        Returns:
            List of individuals to remove
        """
        if method == "worst":
            sorted_pop = sorted(population, key=lambda x: x.fitness)
            return sorted_pop[:n]

        elif method == "random":
            return random.sample(population, min(n, len(population)))

        elif method == "age":
            # Remove oldest individuals
            sorted_pop = sorted(population, key=lambda x: x.age, reverse=True)
            return sorted_pop[:n]

        return []


def select_parents(
    population: list[Individual],
    n_pairs: int,
    config: Optional[SelectionConfig] = None
) -> list[tuple[Individual, Individual]]:
    """
    Select parent pairs for reproduction.

    Args:
        population: List of individuals
        n_pairs: Number of parent pairs to select
        config: Selection configuration

    Returns:
        List of (parent1, parent2) tuples
    """
    selector = FitnessSelector(config)

    # Select 2n parents
    parents = selector.select(population, 2 * n_pairs)

    # Pair them up
    pairs = []
    for i in range(0, len(parents) - 1, 2):
        pairs.append((parents[i], parents[i + 1]))

    return pairs


def survival_selection(
    population: list[Individual],
    max_size: int,
    config: Optional[SelectionConfig] = None
) -> list[Individual]:
    """
    Select individuals that survive to next generation.

    Args:
        population: Current population
        max_size: Maximum population size
        config: Selection configuration

    Returns:
        List of surviving individuals
    """
    if len(population) <= max_size:
        return population

    config = config or SelectionConfig()
    selector = FitnessSelector(config)

    # Always keep elites
    elites = selector.select_elites(population)
    remaining_slots = max_size - len(elites)

    # Select rest from population
    remaining = [ind for ind in population if ind not in elites]

    if remaining_slots > 0:
        survivors = selector.select(remaining, remaining_slots)
        return elites + survivors
    else:
        return elites[:max_size]
