"""
MediAI - Complete Hospital Management System
Integrates all AI/ML components into a unified system
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd

# Import all modules
from agents.multi_agent_coordinator import MultiAgentCoordinator, Task, ResourceType, Agent, AgentState
from search.genetic_algorithm import GeneticResourceOptimizer
from ml.neural_network import MortalityPredictor
from knowledge.table_engine import TableDiagnosisEngine

from vision.xray_analyzer import XRayAnalyzer
from rl.environment import ResourceAllocationEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediAIHospitalSystem:
    """
    Complete hospital management system integrating all AI components
    """
    
    def __init__(self):
        # Initialize all subsystems
        self.agent_coordinator = MultiAgentCoordinator()
        self.resource_optimizer = GeneticResourceOptimizer(generations=3)
        self.mortality_predictor = MortalityPredictor(input_dim=50)
        self.diagnosis_engine = TableDiagnosisEngine()

        self.xray_analyzer = XRayAnalyzer()
        
        # System state
        self.patients = {}
        self.resources = {}
        self.active_cases = []
        self.emergency_queue = []
        
        # Initialize components
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize all system components"""
        logger.info("Initializing MediAI Hospital System...")
        
        # Initialize resources
        self._initialize_resources()
        

        # Load pre-trained models
        self._load_models()
        
        logger.info("System initialized successfully")
    
    def _initialize_resources(self):
        """Initialize hospital resources and register Multi-Agent entities"""
        resources = [
            {'id': 'icu_1', 'type': ResourceType.ICU_BED, 'location': (0, 0), 'capacity': 1},
            {'id': 'icu_2', 'type': ResourceType.ICU_BED, 'location': (0, 1), 'capacity': 1},
            {'id': 'vent_1', 'type': ResourceType.VENTILATOR, 'location': (0, 0), 'capacity': 1},
            {'id': 'surgery_1', 'type': ResourceType.SURGERY_ROOM, 'location': (1, 0), 'capacity': 1},
            {'id': 'doctor_1', 'type': ResourceType.DOCTOR, 'location': (0, 0), 'skill_level': 0.9},
            {'id': 'doctor_2', 'type': ResourceType.DOCTOR, 'location': (1, 1), 'skill_level': 0.8},
        ]
        
        self.resources = {r['id']: r for r in resources}
        
        # Register agents to the Multi-Agent Coordinator!
        for r in resources:
            agent = Agent(
                id=r['id'],
                agent_type=r['type'].value,
                capabilities=[r['type']],
                current_location=r['location'],
                state=AgentState.IDLE,
                current_task=None,
                skill_level=r.get('skill_level', 1.0),
                availability=1.0,
                energy_level=100.0
            )
            self.agent_coordinator.register_agent(agent)
    

    
    def _load_models(self):
        """Load pre-trained ML models"""
        try:
            self.mortality_predictor.load_model('data/models/mortality_model.pth')
            logger.info("Models loaded successfully")
        except:
            logger.warning("No pre-trained models found. Using untrained models.")
    
    async def handle_emergency(self, patient_data: Dict) -> Dict:
        """
        Complete emergency handling pipeline
        """
        patient_id = patient_data['id']
        logger.info(f"Handling emergency for patient {patient_id}")
        
        # Step 1 & 2: Diagnosis using generalized table lookup
        logger.info(f"Analyzing symptoms for patient {patient_id}")
        diagnosis = self.diagnosis_engine.diagnose(patient_data.get('symptoms_text', ''), patient_id)
        
        # Step 3: Analyze X-ray if available
        if 'xray_image' in patient_data:
            xray_result = self.xray_analyzer.analyze(patient_data['xray_image'])
            diagnosis['xray_findings'] = xray_result
        
        # Step 4: Predict mortality risk using neural network
        risk_features = self._extract_risk_features(patient_data, diagnosis)
        mortality_risk = self.mortality_predictor.predict(risk_features)
        
        # Step 5: Create tasks for multi-agent system
        tasks = self._create_medical_tasks(patient_id, diagnosis, mortality_risk)
        
        # Step 6: Optimize resource allocation using genetic algorithm
        optimal_schedule = self.resource_optimizer.evolve(
            tasks, 
            list(self.resources.values())
        )
        
        # Step 7: Coordinate agents for execution
        for task in tasks:
            self.agent_coordinator.add_task(task)
        
        await self.agent_coordinator.distribute_tasks()
        
        mortality_risk_scalar = float(mortality_risk[0][0])
        
        # Step 8: Generate comprehensive report
        report = self._generate_report(patient_id, diagnosis, mortality_risk_scalar, optimal_schedule)
        
        return {
            'patient_id': patient_id,
            'diagnosis': diagnosis,
            'mortality_risk': mortality_risk_scalar,
            'treatment_plan': optimal_schedule,
            'report': report
        }
    
    def _extract_risk_features(self, patient_data: Dict, diagnosis: Dict) -> np.ndarray:
        """Extract features for mortality prediction"""
        features = []
        
        # Demographic features
        features.append(patient_data.get('age', 50) / 100)
        features.append(1 if patient_data.get('gender') == 'male' else 0)
        
        # Vital signs
        features.append(patient_data.get('heart_rate', 80) / 200)
        features.append(patient_data.get('blood_pressure_systolic', 120) / 200)
        features.append(patient_data.get('temperature', 37) / 42)
        
        # Diagnosis severity (from Table)
        features.append(diagnosis.get('severity', 0.2))
        
        # Pad to fixed size
        while len(features) < 50:
            features.append(0)
        
        return np.array([features[:50]])
    
    def _create_medical_tasks(self, patient_id: str, diagnosis: Dict, 
                            mortality_risk: np.ndarray) -> List[Task]:
        """Create tasks for medical procedures"""
        tasks = []
        risk_score = float(mortality_risk[0][0])
        
        # Determine priority based on risk
        priority = 10 if risk_score > 0.7 else 7 if risk_score > 0.4 else 5
        
        # Create diagnosis task
        tasks.append(Task(
            id=f"diagnosis_{patient_id}",
            patient_id=patient_id,
            task_type="diagnosis",
            priority=priority,
            required_resources=[ResourceType.DOCTOR],
            estimated_duration=0.5,
            deadline=datetime.now().timestamp() + 3600,
            dependencies=[],
            location=(0, 0)
        ))
        
        # Create treatment tasks based on diagnosis
        if 'task_type' in diagnosis and diagnosis.get('diagnoses') and diagnosis['diagnoses'][0] != 'Inconclusive':
            res_str = diagnosis.get('resource_type', 'doctor')
            
            # Note: A single Multi-Agent Coordinator node cannot assume both 
            # human capabilities (DOCTOR) and room capabilities (ICU_BED).
            # We assign the task explicitly to human personnel to avoid eligibility lock-outs.
            req_res = [ResourceType.DOCTOR]
                
            task_noun = diagnosis['task_type']
            
            tasks.append(Task(
                id=f"treatment_{patient_id}_{task_noun}",
                patient_id=patient_id,
                task_type=task_noun,
                priority=priority,
                required_resources=req_res,
                estimated_duration=2.0,
                deadline=datetime.now().timestamp() + 86400,
                dependencies=[f"diagnosis_{patient_id}"],
                location=(0, 0)
            ))
            
        return tasks
    
    def _generate_report(self, patient_id: str, diagnosis: Dict, 
                        mortality_risk: float, schedule) -> str:
        """Generate comprehensive medical report"""
        report = f"""
        ========================================
        MEDICAL REPORT - Patient {patient_id}
        ========================================
        
        DIAGNOSIS:
        ----------
        """
        
        for diag in diagnosis.get('diagnoses', []):
            report += f"• {diag}\n"
        
        report += f"""
        MORTALITY RISK ASSESSMENT:
        --------------------------
        Risk Score: {mortality_risk:.2%}
        Risk Level: {'CRITICAL' if mortality_risk > 0.7 else 'HIGH' if mortality_risk > 0.4 else 'MODERATE'}
        
        TREATMENT PLAN:
        --------------
        """
        
        for assignment in schedule.assignments:
            report += f"• Resource: {assignment[0]} -> Task: {assignment[1]} at {assignment[2]:.1f} hours\n"
        
        report += f"""
        EXPLANATION:
        -----------
        {diagnosis.get('explanation', 'No explanation available')}
        
        RECOMMENDATIONS:
        ---------------
        1. Immediate admission to { 'ICU' if mortality_risk > 0.5 else 'regular ward' }
        2. { 'Emergency surgery required' if 'HeartAttack' in diagnosis.get('diagnoses', []) else 'Medical treatment prescribed' }
        3. Follow-up in 24 hours
        
        ========================================
        Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ========================================
        """
        
        return report

async def main():
    """Main execution function"""
    
    # Initialize system
    system = MediAIHospitalSystem()
    
    # Simulate multiple emergency patients
    patients = [
        {
            'id': 'P001',
            'age': 65,
            'gender': 'male',
            'heart_rate': 110,
            'blood_pressure_systolic': 160,
            'temperature': 38.5,
            'symptoms_text': "Severe chest pain, shortness of breath, nausea",
            'xray_image': None
        },
        {
            'id': 'P002',
            'age': 45,
            'gender': 'female',
            'heart_rate': 95,
            'blood_pressure_systolic': 140,
            'temperature': 37.2,
            'symptoms_text': "High fever, persistent cough, difficulty breathing",
            'xray_image': None
        },
        {
            'id': 'P003',
            'age': 72,
            'gender': 'male',
            'heart_rate': 88,
            'blood_pressure_systolic': 155,
            'temperature': 36.8,
            'symptoms_text': "Frequent urination, excessive thirst, blurred vision",
            'xray_image': None
        }
    ]
    
    # Process each patient
    for patient in patients:
        print(f"\n{'='*60}")
        print(f"Processing Patient {patient['id']}")
        print(f"{'='*60}")
        
        result = await system.handle_emergency(patient)
        
        print(result['report'])
        
        # Demo: Show knowledge graph path
        print("\nMedical Knowledge Graph Query:")
        print("Finding path from 'Smoking' to 'HeartAttack'...")
        path = system.medical_knowledge.find_path('Smoking', 'HeartAttack')
        if path:
            print(f"Path found: {' → '.join(path[0])}")
        
        print("\n" + "="*60)
    
    # Compare search algorithms
    print("\n\nSEARCH ALGORITHM COMPARISON")
    print("="*60)
    
    from search.informed_search import AStarSearch
    from search.uninformed_search import BFS, DFS
    
    # Create sample graph
    graph = {
        'ER': {'Triage': 1, 'ICU': 3},
        'Triage': {'ER': 1, 'Lab': 2, 'XRay': 2},
        'ICU': {'ER': 3, 'Surgery': 1},
        'Lab': {'Triage': 2, 'Pharmacy': 1},
        'XRay': {'Triage': 2, 'Radiology': 1},
        'Surgery': {'ICU': 1, 'Recovery': 1},
        'Pharmacy': {'Lab': 1},
        'Radiology': {'XRay': 1},
        'Recovery': {'Surgery': 1}
    }
    
    astar = AStarSearch(graph)
    bfs = BFS(graph)
    dfs = DFS(graph)
    
    start, goal = 'ER', 'Recovery'
    
    print(f"Finding path from {start} to {goal}\n")
    
    astar_path = astar.find_path(start, goal)
    print(f"A* (Informed): {' → '.join(astar_path)} | Cost: {astar_path[1] if astar_path else 'N/A'}")
    
    bfs_path = bfs.find_path(start, goal)
    print(f"BFS: {' → '.join(bfs_path) if bfs_path else 'No path'}")
    
    dfs_path = dfs.find_path(start, goal)
    print(f"DFS: {' → '.join(dfs_path) if dfs_path else 'No path'}")
    
    # Genetic algorithm demonstration
    print("\n\nGENETIC ALGORITHM OPTIMIZATION")
    print("="*60)
    
    # Create sample tasks for optimization
    sample_tasks = [
        Task("T1", "P001", "surgery", 9, [ResourceType.DOCTOR, ResourceType.SURGERY_ROOM], 2, 10, [], (0,0)),
        Task("T2", "P002", "treatment", 7, [ResourceType.DOCTOR, ResourceType.ICU_BED], 1, 8, [], (0,0)),
        Task("T3", "P003", "diagnosis", 5, [ResourceType.DOCTOR], 0.5, 6, [], (0,0)),
    ]
    
    optimizer = GeneticResourceOptimizer(population_size=50, generations=3)
    best_schedule = optimizer.evolve(sample_tasks, list(system.resources.values()))
    
    print(f"Best schedule found with fitness: {best_schedule.fitness_score:.4f}")
    
    print("\n\nSystem Demo Complete!")
    print("MediAI Hospital Management System is ready for production deployment.")

if __name__ == "__main__":
    # If you want to run the terminal demo simulation, uncomment the line below:
    # asyncio.run(main())
    
    print("\n" + "="*60)
    print("Launching Web Dashboard on port 8000...")
    print("="*60)
    
    import uvicorn
    
    try:
        uvicorn.run("api:app", host="127.0.0.1", port=8000)
    except Exception as e:
        print(f"Failed to start dashboard server: {e}\nIs the server already running on port 8000?")