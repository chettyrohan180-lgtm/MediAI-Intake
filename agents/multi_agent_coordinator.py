import asyncio
import heapq
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import networkx as nx
import math

class AgentState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    NEGOTIATING = "negotiating"
    EMERGENCY = "emergency"
    COORDINATING = "coordinating"

class ResourceType(Enum):
    ICU_BED = "icu_bed"
    VENTILATOR = "ventilator"
    SURGERY_ROOM = "surgery_room"
    DOCTOR = "doctor"
    NURSE = "nurse"
    AMBULANCE = "ambulance"

@dataclass
class Task:
    # represents a medical task to be executed
    id: str
    patient_id: str
    task_type: str
    priority: int  # 1-10, 10 highest
    required_resources: List[ResourceType]
    estimated_duration: float
    deadline: float
    dependencies: List[str]
    location: Tuple[int, int]

@dataclass
class Agent:
    # base agent definition
    id: str
    agent_type: str
    capabilities: List[str]
    current_location: Tuple[int, int]
    state: AgentState
    skill_level: float
    availability: float  # Time available
    energy_level: float  # 0-100
    current_task: Optional[str] = None

class MultiAgentCoordinator:
    # MAS coordinator for agents + tasks

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue = []
        self.resource_allocation = {}
        self.negotiation_memory = {}
        self.communication_graph = nx.Graph()

    def register_agent(self, agent: Agent):
        # register new agent
        self.agents[agent.id] = agent
        self.communication_graph.add_node(agent.id, type=agent.agent_type)

    def add_task(self, task: Task):
        # add task to queue with priority
        heapq.heappush(self.task_queue, (-task.priority, task.id))
        self.tasks[task.id] = task

    async def distribute_tasks(self):
        while self.task_queue:
            priority, task_id = heapq.heappop(self.task_queue)
            task = self.tasks[task_id]

            # find agents for this task
            eligible_agents = self._find_eligible_agents(task)

            if not eligible_agents:
                print(f"No eligible agents for task {task_id}")
                continue

            # get bids
            bids = await self._collect_bids(eligible_agents, task)

            # pick winner
            winner = self._select_best_bid(bids)

            # run task
            if winner:
                await self._execute_task(winner, task)
            else:
                print(f"Failed to allocate task {task_id}")

    def _find_eligible_agents(self, task: Task) -> List[Agent]:
        # find agents that can do this task
        eligible = []
        for agent in self.agents.values():
            # Check if agent has required capabilities
            required_capabilities = set(task.required_resources)
            agent_capabilities = set(agent.capabilities)

            if required_capabilities.issubset(agent_capabilities):
                # Check availability
                if agent.state == AgentState.IDLE or agent.state == AgentState.NEGOTIATING:
                    eligible.append(agent)
        return eligible

    async def _collect_bids(self, agents: List[Agent], task: Task) -> Dict[str, float]:
        # collect bids from agents (simulated negotiation)
        bids = {}
        for agent in agents:
            # agent calculates bid based on distance, workload, skill, energy

            distance = self._calculate_distance(agent.current_location, task.location)
            workload = self._calculate_workload(agent)
            skill_match = self._calculate_skill_match(agent, task)

            # Complex bid calculation formula
            bid = (
                distance * 0.3 +
                workload * 0.4 +
                (100 - skill_match) * 0.3
            )

            # Add negotiation memory influence
            if agent.id in self.negotiation_memory:
                bid *= self.negotiation_memory[agent.id].get('reliability', 1.0)

            bids[agent.id] = bid

        return bids

    def _select_best_bid(self, bids: Dict[str, float]) -> Optional[str]:
        # best bid via simple game theory principles
        if not bids:
            return None

        # Minimize cost (bid)
        winner = min(bids.items(), key=lambda x: x[1])[0]

        # Record negotiation outcome for future learning
        self.negotiation_memory[winner] = {
            'success_rate': self.negotiation_memory.get(winner, {}).get('success_rate', 0.5) + 0.1,
            'reliability': bids[winner]
        }

        return winner

    async def _execute_task(self, agent_id: str, task: Task):
        # trigger execution
        agent = self.agents[agent_id]
        agent.state = AgentState.BUSY
        agent.current_task = task.id

        # Simulate task execution
        await asyncio.sleep(task.estimated_duration)

        # Update agent state after completion
        agent.state = AgentState.IDLE
        agent.current_task = None
        agent.energy_level -= task.estimated_duration * 0.1

        # Check if needs rest
        if agent.energy_level < 30:
            agent.state = AgentState.NEGOTIATING

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Euclidean distance"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _calculate_workload(self, agent: Agent) -> float:
        """Calculate agent's current workload"""
        # Based on current task and queue
        base_load = 0 if agent.state == AgentState.IDLE else 0.7

        # Additional load based on energy level
        energy_factor = (100 - agent.energy_level) / 100

        return base_load * (1 + energy_factor)

    def _calculate_skill_match(self, agent: Agent, task: Task) -> float:
        """Calculate how well agent's skills match task"""
        required = set(task.required_resources)
        agent_capabilities = set(agent.capabilities)

        # Jaccard similarity for skill match
        intersection = len(required & agent_capabilities)
        union = len(required | agent_capabilities)

        return (intersection / union) * 100

class CoalitionFormation:

    def __init__(self, coordinator: MultiAgentCoordinator):
        self.coordinator = coordinator
        self.coalitions = {}

    def form_coalition(self, task: Task, required_agents: int) -> List[Agent]:
        # form optimal coalition using Shapley value
        eligible_agents = self.coordinator._find_eligible_agents(task)

        if len(eligible_agents) < required_agents:
            return []

        # Calculate value of each possible coalition
        coalition_values = {}
        all_combinations = self._generate_combinations(eligible_agents, required_agents)

        for combo in all_combinations:
            value = self._calculate_coalition_value(combo, task)
            coalition_values[tuple(combo)] = value

        # Select coalition with maximum value
        best_coalition = max(coalition_values.items(), key=lambda x: x[1])[0]

        # Calculate Shapley values for fair payoff distribution
        shapley_values = self._calculate_shapley_values(best_coalition, coalition_values)

        return list(best_coalition)

    def _calculate_coalition_value(self, coalition: List[Agent], task: Task) -> float:
        """Calculate value of coalition for given task"""
        total_skill = sum(agent.skill_level for agent in coalition)
        diversity_score = len(set(agent.agent_type for agent in coalition))
        coordination_cost = len(coalition) * 0.1  # More agents = more coordination overhead

        return total_skill * diversity_score - coordination_cost

    def _calculate_shapley_values(self, coalition: List[Agent], coalition_values: Dict) -> Dict[str, float]:
        """Calculate Shapley values for fair distribution"""
        shapley = {agent.id: 0.0 for agent in coalition}
        n = len(coalition)

        for i, agent in enumerate(coalition):
            for subset in self._generate_subsets(coalition, agent):
                # Calculate marginal contribution
                with_agent = tuple(sorted([a.id for a in subset] + [agent.id]))
                without_agent = tuple(sorted([a.id for a in subset]))

                value_with = coalition_values.get(with_agent, 0)
                value_without = coalition_values.get(without_agent, 0)

                marginal = value_with - value_without

                # Weight by permutation probability
                weight = (math.factorial(len(subset)) * math.factorial(n - len(subset) - 1)) / math.factorial(n)

                shapley[agent.id] += marginal * weight

        return shapley

    def _generate_combinations(self, agents: List[Agent], k: int) -> List[List[Agent]]:
        """Generate all combinations of size k"""
        from itertools import combinations
        return [list(combo) for combo in combinations(agents, k)]

    def _generate_subsets(self, coalition: List[Agent], exclude_agent: Agent) -> List[List[Agent]]:
        """Generate all subsets not containing exclude_agent"""
        from itertools import chain, combinations
        other_agents = [a for a in coalition if a.id != exclude_agent.id]

        subsets = []
        for r in range(len(other_agents) + 1):
            for combo in combinations(other_agents, r):
                subsets.append(list(combo))
        return subsets