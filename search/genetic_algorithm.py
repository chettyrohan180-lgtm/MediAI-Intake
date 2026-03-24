"""
Genetic Algo for hospital scheduling
"""

import random
import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass

@dataclass
class ResourceSchedule:
    """Represents a schedule of resource allocation"""
    schedule_id: str
    assignments: List[Tuple[str, str, float]]  # (resource, task, time)
    total_cost: float
    total_time: float
    resource_utilization: float
    patient_satisfaction: float

class GeneticResourceOptimizer:
    """GA for scheduling"""
    
    def __init__(self, 
                 population_size: int = 100,
                 generations: int = 200,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1):
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        
        self.population = []
        self.fitness_history = []
        
    def initialize_population(self, tasks: List[Task], resources: List[Resource]):
        """Initialize population with random schedules"""
        self.population = []
        for _ in range(self.population_size):
            schedule = self._generate_random_schedule(tasks, resources)
            self.population.append(schedule)
            
    def _generate_random_schedule(self, tasks: List[Task], resources: List[Resource]) -> ResourceSchedule:
        """Generate random valid schedule"""
        assignments = []
        used_resources = set()
        
        for task in tasks:
            # Randomly assign resources to task
            assigned = random.sample(resources, len(task.required_resources))
            for resource in assigned:
                time_slot = random.uniform(0, 24)  # Hours in day
                resource_id = resource['id'] if isinstance(resource, dict) else getattr(resource, 'id', None)
                assignments.append((resource_id, task.id, time_slot))
                used_resources.add(resource_id)
                
        return ResourceSchedule(
            schedule_id=str(random.randint(1000, 9999)),
            assignments=assignments,
            total_cost=0,
            total_time=0,
            resource_utilization=0,
            patient_satisfaction=0
        )
    
    def fitness_function(self, schedule: ResourceSchedule, 
                        cost_weights: Tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.2)) -> float:
        """Fitness function"""
        # Normalize each objective to [0,1] range
        cost_normalized = 1 / (1 + schedule.total_cost)
        time_normalized = 1 / (1 + schedule.total_time)
        utilization_normalized = schedule.resource_utilization
        satisfaction_normalized = schedule.patient_satisfaction
        
        # Weighted sum
        fitness = (
            cost_weights[0] * cost_normalized +
            cost_weights[1] * time_normalized +
            cost_weights[2] * utilization_normalized +
            cost_weights[3] * satisfaction_normalized
        )
        
        return fitness
    
    def selection(self) -> List[ResourceSchedule]:
        """
        Tournament selection for parent selection
        """
        selected = []
        tournament_size = 3
        
        for _ in range(len(self.population)):
            # Random tournament
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness_score)
            selected.append(winner)
            
        return selected
    
    def crossover(self, parent1: ResourceSchedule, parent2: ResourceSchedule) -> Tuple[ResourceSchedule, ResourceSchedule]:
        """
        Two-point crossover for schedule recombination
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Find crossover points
        point1 = random.randint(0, len(parent1.assignments))
        point2 = random.randint(point1, len(parent1.assignments))
        
        # Create children by swapping segments
        child1_assignments = (parent1.assignments[:point1] + 
                             parent2.assignments[point1:point2] + 
                             parent1.assignments[point2:])
        
        child2_assignments = (parent2.assignments[:point1] + 
                             parent1.assignments[point1:point2] + 
                             parent2.assignments[point2:])
        
        child1 = ResourceSchedule(
            schedule_id=str(random.randint(1000, 9999)),
            assignments=child1_assignments,
            total_cost=0,
            total_time=0,
            resource_utilization=0,
            patient_satisfaction=0
        )
        
        child2 = ResourceSchedule(
            schedule_id=str(random.randint(1000, 9999)),
            assignments=child2_assignments,
            total_cost=0,
            total_time=0,
            resource_utilization=0,
            patient_satisfaction=0
        )
        
        return child1, child2
    
    def mutate(self, schedule: ResourceSchedule, resources: List[Resource]):
        """
        Mutation operator with adaptive mutation rate
        """
        if random.random() > self.mutation_rate or not schedule.assignments:
            return schedule
        
        # Randomly modify some assignments
        mutation_point = random.randint(0, len(schedule.assignments) - 1)
        resource_id, task_id, time = schedule.assignments[mutation_point]
        
        # Either change resource or time
        if random.random() < 0.5:
            # Change resource
            new_resource = random.choice(resources)
            new_resource_id = new_resource['id'] if isinstance(new_resource, dict) else getattr(new_resource, 'id', None)
            schedule.assignments[mutation_point] = (new_resource_id, task_id, time)
        else:
            # Change time
            new_time = random.uniform(0, 24)
            schedule.assignments[mutation_point] = (resource_id, task_id, new_time)
            
        return schedule
    
    def _evaluate_schedule(self, schedule: ResourceSchedule, tasks: List, resources: List):
        """Calculate and store metrics for a schedule based on its assignments"""
        task_dict = {getattr(t, 'id', t): t for t in tasks}
        
        total_time = 0.0
        total_cost = 0.0
        
        # Track resource usage over time: {resource_id: [(start, end)]}
        resource_timeline = {}
        conflicts = 0
        
        for resource_id, task_id, start_time in schedule.assignments:
            task = task_dict.get(task_id)
            duration = getattr(task, 'estimated_duration', 1.0) if task else 1.0
            end_time = start_time + duration
            
            total_time = max(total_time, end_time)
            total_cost += duration * 100  # Base cost simulation
            
            if resource_id not in resource_timeline:
                resource_timeline[resource_id] = []
                
            # Check for overlapping conflicts on the same resource
            for existing_start, existing_end in resource_timeline[resource_id]:
                if not (end_time <= existing_start or start_time >= existing_end):
                    conflicts += 1
            
            resource_timeline[resource_id].append((start_time, end_time))
            
        # Heavily penalize conflicts since a doctor/ICU can't do two things at once
        schedule.total_cost = total_cost + (conflicts * 5000)
        schedule.total_time = total_time + (conflicts * 24)
        
        # Calculate utilization
        schedule.resource_utilization = len(resource_timeline) / max(1, len(resources))
        
        # Satisfaction drops heavily with conflicts and excessive completion times
        base_satisfaction = 1.0
        satisfaction_penalty = (conflicts * 0.3) + (total_time / 200.0)
        schedule.patient_satisfaction = max(0.0, base_satisfaction - satisfaction_penalty)

    def evolve(self, tasks: List, resources: List, 
              verbose: bool = True) -> ResourceSchedule:
        """Main GA loop"""
        # Initialize
        self.initialize_population(tasks, resources)
        
        # Calculate initial fitness
        for schedule in self.population:
            self._evaluate_schedule(schedule, tasks, resources)
            schedule.fitness_score = self.fitness_function(schedule)
        
        best_fitness_over_time = []
        
        for generation in range(self.generations):
            # Selection
            selected = self.selection()
            
            # Crossover
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i+1])
                    offspring.extend([child1, child2])
            
            # Mutation
            offspring = [self.mutate(child, resources) for child in offspring]
            
            # Calculate fitness for offspring
            for child in offspring:
                self._evaluate_schedule(child, tasks, resources)
                child.fitness_score = self.fitness_function(child)
            
            # Elitism: Keep best individuals
            elite_count = int(self.elitism_rate * self.population_size)
            elites = sorted(self.population, key=lambda x: x.fitness_score, reverse=True)[:elite_count]
            
            # Create new population
            self.population = elites + offspring[:self.population_size - elite_count]
            
            # Track best fitness
            best = max(self.population, key=lambda x: x.fitness_score)
            best_fitness_over_time.append(best.fitness_score)
            
            # Adaptive mutation rate
            if generation % 50 == 0:
                self._adapt_mutation_rate(best_fitness_over_time)
            
            if verbose:
                print(f"Generation {generation}: Best Fitness = {best.fitness_score:.4f}")
        
        # Return best schedule
        best_schedule = max(self.population, key=lambda x: x.fitness_score)
        return best_schedule
    
    def _adapt_mutation_rate(self, fitness_history: List[float]):
        """Adapt mutation rate based on fitness stagnation"""
        if len(fitness_history) < 20:
            return
        
        recent_improvement = fitness_history[-1] - fitness_history[-20]
        
        if recent_improvement < 0.01:
            # Stagnation: increase mutation rate
            self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
        else:
            # Progress: decrease mutation rate
            self.mutation_rate = max(0.01, self.mutation_rate * 0.95)