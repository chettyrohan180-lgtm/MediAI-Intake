# MediAI Hospital Management System

This is a hospital triage and management system designed to explore various AI/ML concepts. It simulates a hospital environment where different AI components work together to manage patients and resources.

## Key Features

- **Mortality Risk Prediction**: Predicts patient mortality risk based on vitals using a Deep Neural Network (MLP).
- **Automated X-Ray Analysis**: Basic diagnostics for medical imaging.
- **Rule-based Diagnosis**: Matches symptoms against a medical knowledge base for rapid triage and severity assessment.
- **Multi-Agent Coordination**: Models doctors, ICU beds, and surgery rooms as autonomous agents that bid on tasks.
- **Scheduling Optimization**: Uses Genetic Algorithms (GA) to find optimal resource schedules and minimize completion times.
- **Optimized Pathfinding**: Implements A*, BFS, and DFS for routing within the hospital network.

## Running the System

To get the dashboard running:

**On Windows:**
Double-click `Run_Dashboard.bat`. This script will set up a virtual environment, install dependencies, and start the FastAPI server.

**On Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app
```
Then navigate to `http://localhost:8000`.

## Project Structure

- `api.py`: FastAPI web server and dashboard logic.
- `main.py`: Core simulation logic and terminal interface.
- `agents/`: Multi-agent coordination and negotiation modules.
- `knowledge/`: Diagnostic logic and medical table engine.
- `ml/`: Machine learning models for clinical prediction.
- `search/`: Optimization and pathfinding algorithms.
- `static/`: Frontend dashboard assets.
- `vision/`: Image processing for medical diagnostics.
