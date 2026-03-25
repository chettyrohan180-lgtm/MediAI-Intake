import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any

import numpy as np

from agents.multi_agent_coordinator import (
    MultiAgentCoordinator,
    Task,
    ResourceType,
    Agent,
    AgentState,
)
from search.genetic_algorithm import GeneticResourceOptimizer
from ml.neural_network import MortalityPredictor
from knowledge.table_engine import TableDiagnosisEngine
from vision.xray_analyzer import XRayAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediAIHospitalSystem:
    """
    Main orchestration layer for the hospital demo.

    This class glues together:
    - symptom-based diagnosis
    - optional X-ray analysis
    - mortality risk estimation
    - resource scheduling / assignment

    It's still a simulation, so a few shortcuts are intentional.
    """

    def __init__(self) -> None:
        self.agent_coordinator = MultiAgentCoordinator()
        self.resource_optimizer = GeneticResourceOptimizer(generations=3)
        self.mortality_predictor = MortalityPredictor(input_dim=50)
        self.diagnosis_engine = TableDiagnosisEngine()
        self.xray_analyzer = XRayAnalyzer()

        self.patients: Dict[str, Dict[str, Any]] = {}
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.active_cases: List[str] = []
        self.emergency_queue: List[str] = []

        self._boot()

    def _boot(self) -> None:
        logger.info("Starting MediAI hospital system...")
        self._initialize_resources()
        self._load_models()
        logger.info("System ready.")

    def _initialize_resources(self) -> None:
        # Small hardcoded pool for demo/testing.
        # In a real deployment this would likely come from a DB or service registry.
        resource_seed = [
            {"id": "icu_1", "type": ResourceType.ICU_BED, "location": (0, 0), "capacity": 1},
            {"id": "icu_2", "type": ResourceType.ICU_BED, "location": (0, 1), "capacity": 1},
            {"id": "vent_1", "type": ResourceType.VENTILATOR, "location": (0, 0), "capacity": 1},
            {"id": "surgery_1", "type": ResourceType.SURGERY_ROOM, "location": (1, 0), "capacity": 1},
            {"id": "doctor_1", "type": ResourceType.DOCTOR, "location": (0, 0), "skill_level": 0.9},
            {"id": "doctor_2", "type": ResourceType.DOCTOR, "location": (1, 1), "skill_level": 0.8},
        ]

        self.resources = {item["id"]: item for item in resource_seed}

        for item in resource_seed:
            agent = Agent(
                id=item["id"],
                agent_type=item["type"].value,
                capabilities=[item["type"]],
                current_location=item["location"],
                state=AgentState.IDLE,
                current_task=None,
                skill_level=item.get("skill_level", 1.0),
                availability=1.0,
                energy_level=100.0,
            )
            self.agent_coordinator.register_agent(agent)

    def _load_models(self) -> None:
        try:
            self.mortality_predictor.load_model("data/models/mortality_model.pth")
            logger.info("Mortality model loaded.")
        except Exception as exc:
            # We don't want the whole demo to fail if the model file is missing.
            logger.warning("Could not load mortality model, falling back to default behavior: %s", exc)

    async def handle_emergency(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        patient_id = patient_data["id"]
        logger.info("Handling emergency case: %s", patient_id)

        self.patients[patient_id] = patient_data
        self.active_cases.append(patient_id)

        diagnosis = self.diagnosis_engine.diagnose(
            patient_data.get("symptoms_text", ""),
            patient_id,
        )

        if patient_data.get("xray_image") is not None:
            try:
                diagnosis["xray_findings"] = self.xray_analyzer.analyze(patient_data["xray_image"])
            except Exception as exc:
                logger.warning("X-ray analysis failed for %s: %s", patient_id, exc)

        risk_features = self._extract_risk_features(patient_data, diagnosis)
        mortality_risk = self.mortality_predictor.predict(risk_features)
        risk_score = float(mortality_risk.item(0))

        tasks = self._create_medical_tasks(patient_id, diagnosis, risk_score)

        # Keep optimizer in the loop, but task execution still depends on coordinator state.
        schedule = self.resource_optimizer.evolve(tasks, list(self.resources.values()))

        for task in tasks:
            self.agent_coordinator.add_task(task)

        await self.agent_coordinator.distribute_tasks()

        report = self._generate_report(
            patient_id=patient_id,
            diagnosis=diagnosis,
            mortality_risk=risk_score,
            schedule=schedule,
        )

        return {
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "mortality_risk": risk_score,
            "treatment_plan": schedule,
            "report": report,
        }

    def _extract_risk_features(self, patient_data: Dict[str, Any], diagnosis: Dict[str, Any]) -> np.ndarray:
        """
        Build a fixed-length feature vector for the mortality model.

        Right now this is intentionally simple and padded to 50 dims because
        the model expects that input shape.
        """
        features: List[float] = []

        # Demographics
        features.append(patient_data.get("age", 50) / 100)
        features.append(1.0 if patient_data.get("gender") == "male" else 0.0)

        # Basic vitals
        features.append(patient_data.get("heart_rate", 80) / 200)
        features.append(patient_data.get("blood_pressure_systolic", 120) / 200)
        features.append(patient_data.get("temperature", 37) / 42)

        # Diagnosis severity score from rule engine
        features.append(float(diagnosis.get("severity", 0.2)))

        while len(features) < 50:
            features.append(0.0)

        return np.array([features[:50]])

    def _create_medical_tasks(
        self,
        patient_id: str,
        diagnosis: Dict[str, Any],
        risk_score: float,
    ) -> List[Task]:
        tasks: List[Task] = []

        if risk_score > 0.7:
            priority = 10
        elif risk_score > 0.4:
            priority = 7
        else:
            priority = 5

        tasks.append(
            Task(
                id=f"diagnosis_{patient_id}",
                patient_id=patient_id,
                task_type="diagnosis",
                priority=priority,
                required_resources=[ResourceType.DOCTOR],
                estimated_duration=0.5,
                deadline=datetime.now().timestamp() + 3600,
                dependencies=[],
                location=(0, 0),
            )
        )

        diagnoses = diagnosis.get("diagnoses", [])
        if diagnosis.get("task_type") and diagnoses and diagnoses[0] != "Inconclusive":
            # For now, we always route treatment through a doctor first.
            # This avoids deadlocks in early demos where specialized resources were unavailable.
            task_type = diagnosis["task_type"]

            tasks.append(
                Task(
                    id=f"treatment_{patient_id}_{task_type}",
                    patient_id=patient_id,
                    task_type=task_type,
                    priority=priority,
                    required_resources=[ResourceType.DOCTOR],
                    estimated_duration=2.0,
                    deadline=datetime.now().timestamp() + 86400,
                    dependencies=[f"diagnosis_{patient_id}"],
                    location=(0, 0),
                )
            )

        return tasks

    def _generate_report(self, patient_id: str, diagnosis: Dict[str, Any], mortality_risk: float, schedule) -> str:
        lines = []
        lines.append("=" * 40)
        lines.append(f"MEDICAL REPORT - Patient {patient_id}")
        lines.append("=" * 40)
        lines.append("")
        lines.append("DIAGNOSIS:")
        lines.append("-" * 10)

        diagnoses = diagnosis.get("diagnoses", [])
        if diagnoses:
            for item in diagnoses:
                lines.append(f"- {item}")
        else:
            lines.append("- No diagnosis available")

        lines.append("")
        lines.append("MORTALITY RISK:")
        lines.append("-" * 15)
        lines.append(f"Risk Score: {mortality_risk:.2%}")

        if mortality_risk > 0.7:
            risk_level = "CRITICAL"
        elif mortality_risk > 0.4:
            risk_level = "HIGH"
        else:
            risk_level = "MODERATE"

        lines.append(f"Risk Level: {risk_level}")
        lines.append("")
        lines.append("TREATMENT PLAN:")
        lines.append("-" * 15)

        for assignment in schedule.assignments:
            lines.append(f"- {assignment[0]} -> {assignment[1]} at {assignment[2]:.1f}h")

        lines.append("")
        lines.append("EXPLANATION:")
        lines.append("-" * 12)
        lines.append(diagnosis.get("explanation", "No explanation available"))
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 16)
        lines.append(f"1. Immediate admission to {'ICU' if mortality_risk > 0.5 else 'regular ward'}")

        if "HeartAttack" in diagnoses:
            lines.append("2. Emergency cardiac intervention should be considered")
        else:
            lines.append("2. Continue with medical treatment and monitoring")

        lines.append("3. Reassess in 24 hours")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)


async def main() -> None:
    system = MediAIHospitalSystem()

    demo_patients = [
        {
            "id": "P001",
            "age": 65,
            "gender": "male",
            "heart_rate": 110,
            "blood_pressure_systolic": 160,
            "temperature": 38.5,
            "symptoms_text": "Severe chest pain, shortness of breath, nausea",
            "xray_image": None,
        },
        {
            "id": "P002",
            "age": 45,
            "gender": "female",
            "heart_rate": 95,
            "blood_pressure_systolic": 140,
            "temperature": 37.2,
            "symptoms_text": "High fever, persistent cough, difficulty breathing",
            "xray_image": None,
        },
        {
            "id": "P003",
            "age": 72,
            "gender": "male",
            "heart_rate": 88,
            "blood_pressure_systolic": 155,
            "temperature": 36.8,
            "symptoms_text": "Frequent urination, excessive thirst, blurred vision",
            "xray_image": None,
        },
    ]

    for patient in demo_patients:
        print("\n" + "=" * 60)
        print(f"Processing {patient['id']}")
        print("=" * 60)

        result = await system.handle_emergency(patient)
        print(result["report"])


if __name__ == "__main__":
    asyncio.run(main())

    print("\nStarting API server on http://127.0.0.1:8000 ...")
    import uvicorn

    try:
        uvicorn.run("api:app", host="127.0.0.1", port=8000)
    except Exception as exc:
        print(f"API startup failed: {exc}")