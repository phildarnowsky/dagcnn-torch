from math import floor, inf
from random import choice

import torch

from torch.optim import Adam
from torch import nn

from .genome import Genome

class Population():
    def __init__(
        self,
        genomes,
        training_loader,
        validation_loader,
        make_optimizer=lambda individual: Adam(individual.parameters()),
        criterion_class=nn.CrossEntropyLoss,
        n_generations=100,
        elitism_fraction=0.2,
        mean_threshold=0.2,
        std_threshold=0.02,
        mutation_probability=0.003
    ):
        self._genomes = genomes
        n_genomes = len(genomes)
        self._n_by_elitism = floor(elitism_fraction * n_genomes)
        self._n_by_breeding = n_genomes - self._n_by_elitism

        self._fitness_cache = {}
        self._n_generations = n_generations
        self._training_loader = training_loader
        self._validation_loader = validation_loader
        self._make_optimizer = make_optimizer
        self._criterion_class = criterion_class
        self._mean_threshold = mean_threshold
        self._std_threshold = std_threshold
        self._mutation_probability = mutation_probability

    def breed(self, generation_start_callback=None, generation_end_callback=None):
        self.generation_index = 0
        while self.generation_index < self._n_generations:
            if generation_start_callback:
                generation_start_callback(self)

            self._breed_next_generation()

            if generation_end_callback:
                generation_end_callback(self)

            self.generation_index += 1

    def _breed_next_generation(self):
        new_genomes = []
        for _ in range(0, self._n_by_elitism):
            elite_genome = self._select_by_slack_binary_tournament()
            new_genomes.append(elite_genome)

        for _ in range(0, self._n_by_breeding):
            parent1 = self._select_by_slack_binary_tournament()
            parent2 = self._select_by_slack_binary_tournament()
            child = parent1.crossover(parent2)
            new_genomes.append(child)

        new_genomes = list(
            map(
                lambda genome: genome.apply_mutations(self._mutation_probability),
                new_genomes
            )
        )
        self._genomes = new_genomes

    def all_fitnesses(self):
        return {genome : self._fitness(genome) for genome in self._genomes}

    def _fitness(self, genome):
        cache_key = genome.to_cache_key()
        if cache_key not in self._fitness_cache:
            self._fitness_cache[cache_key] = self._evaluate_fitness(genome)
        return self._fitness_cache[cache_key]

    def _evaluate_fitness(self, genome):
        individual = genome.to_individual()
        validation_losses = []
        criterion = self._criterion_class()
        optimizer = self._make_optimizer(individual)

        try:
            individual.train()
            for _, (training_examples, training_labels) in enumerate(self._training_loader):
                training_examples = training_examples.cuda()
                training_labels = training_labels.cuda()
                training_predictions = individual(training_examples)

                training_loss = criterion(training_predictions, training_labels.flatten())
                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()

            individual.eval()
            with torch.no_grad():
                for _, (validation_examples, validation_labels) in enumerate(self._validation_loader):
                    validation_examples = validation_examples.cuda()
                    validation_labels = validation_labels.cuda()
                    validation_predictions = individual(validation_examples)

                    validation_loss = criterion(validation_predictions, validation_labels.flatten())
                    validation_losses.append(validation_loss.item())
        except RuntimeError:
            saynow("RUNTIME ERROR CAUGHT! ASSUMING OUT OF CUDA MEMORY! INFINITE LOSS! OH NO!")
            return {'mean': inf, 'std': inf, 'n_parameters': inf}

        validation_losses = torch.tensor(validation_losses)
        n_parameters = sum(list(map(lambda parameter: parameter.size().numel(), individual.parameters())))

        return {'mean': validation_losses.mean().item(), 'std': validation_losses.std().item(), 'n_parameters': n_parameters}

    def _select_by_slack_binary_tournament(self):
        genome1 = choice(self._genomes)
        genome2 = choice(self._genomes)

        fitness1 = self._fitness(genome1)
        fitness2 = self._fitness(genome2)

        if fitness1['mean'] - fitness2['mean'] > self._mean_threshold:
            return genome2
        elif fitness2['mean'] - fitness1['mean'] > self._mean_threshold:
            return genome1
        elif fitness1['std'] - fitness2['std'] > self._std_threshold:
            return genome2
        elif fitness2['std'] - fitness1['std'] > self._std_threshold:
            return genome1
        elif fitness1['n_parameters'] < fitness2['n_parameters']:
            return genome1

        return genome2

    @classmethod
    def make_random(
        cls,
        input_shape,
        n_outputs,
        training_loader,
        validation_loader,
        make_optimizer=lambda individual: Adam(individual.parameters()),
        criterion_class=nn.CrossEntropyLoss,
        n_genomes=100,
        min_n_genes=10,
        max_n_genes=15,
        n_generations=100,
        elitism_fraction=0.2,
        mutation_probability=0.003,
        mean_threshold=0.2,
        std_threshold=0.02
    ):
        genomes = []
        for _ in range(n_genomes):
            genomes.append(Genome.make_random(input_shape, n_outputs, min_n_genes, max_n_genes))
        return cls(
            genomes,
            training_loader,
            validation_loader,
            make_optimizer,
            criterion_class,
            n_generations=n_generations,
            elitism_fraction=elitism_fraction,
            mutation_probability=mutation_probability,
            mean_threshold=mean_threshold,
            std_threshold=std_threshold
        )
