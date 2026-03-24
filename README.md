# MediAI Hospital System

This is a hospital triage and management system I built to test out various AI/ML concepts. It simulates a hospital environment where different AI components work together to manage patients and resources.

## What it does

- Predicts patient mortality risk based on their vitals (`ml/` stuff).
- Does basic automated X-Ray analysis (`vision/`).
- Routes patients using a rule-based table diagnosis and medical knowledge graph (`knowledge/`, `nlp/`).
- Uses a multi-agent system where doctors, ICU beds, and surgery rooms are treated as agents that pick up tasks (`agents/`).
- I also threw in some search algorithms:
  - **Genetic Algorithms** to figure out the best way to schedule procedures.
  - **Pathfinding (A*, BFS, DFS)** to route things optimally around the hospital network (`search/`).
- **Reinforcement Learning** for trying to tweak resource allocation (`rl/`).
- It has a web UI built with FastAPI and plain CSS (`api.py` & `static/`).

## Running it

I made a script to get it running quickly without messing with dependencies manually.

**On Windows:**
Just double-click `Run_Dashboard.bat`. It'll set up a virtual environment, install whatever's in `requirements.txt`, start the server, and open the dashboard in your browser automatically.

**On Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app
```
Then just go to `http://localhost:8000`.

## Code Structure

- `api.py`: The web server.
- `main.py`: The core logic. If you run this directly, it just does a terminal simulation of a few patients.
- `static/`: Frontend stuff.
- The other folders (`ml/`, `vision/`, `agents/`, etc.) hold the logic for those specific sections.
