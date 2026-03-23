# MediAI Hospital System

A full-stack, AI-powered hospital triage and management system that integrates various Artificial Intelligence paradigms into a unified platform.

## Features

- **Deep Learning / Neural Networks**: Predict patient mortality risk dynamically using vital signs and demographic data (`ml/` module).
- **Computer Vision**: Automated X-Ray image analysis and diagnosis (`vision/` module).
- **Knowledge Representation & NLP**: Rule-based table diagnosis engine and medical knowledge graph routing (`knowledge/` and `nlp/` modules).
- **Multi-Agent Systems**: Coordinated agents (doctors, ICU beds, surgery rooms) dynamically distributing tasks and resources (`agents/` module).
- **Search Algorithms & Optimization**: 
  - **Genetic Algorithms** for optimally scheduling medical procedures and allocating resources.
  - **Pathfinding (A*, BFS, DFS)** for optimal routing within the hospital network (`search/` module).
- **Reinforcement Learning**: Environment simulation for resource allocation (`rl/` module).
- **Beautiful Web Interface**: A dynamic UI served via FastAPI and Vanilla CSS (`api.py` & `static/`).

## AI & ML Course Topics Covered

This project practically implements several core topics from the AI & ML syllabus:

- **Problem solving agents**: Implemented via the Multi-Agent Coordinator (`agents/` module) where doctors, ICU beds, and surgery rooms act as autonomous agents resolving tasks.
- **Search strategies - Uninformed**: Implemented Breadth-First Search (BFS) and Depth-First Search (DFS) for hospital routing and traversal (`search/uninformed_search.py`).
- **Search strategies - Informed**: Implemented A* (A-Star) Search for optimal pathfinding and routing within the hospital network (`search/informed_search.py`).
- **Local search algorithms**: Implemented Genetic Algorithms to optimize the scheduling of medical procedures and resource allocation (`search/genetic_algorithm.py`).
- **Knowledge representation**: Implemented a Rule-based Table Diagnosis Engine that categorizes and matches patient symptoms against clinical presentations (`knowledge/table_engine.py`).

## How to Run the Project

To make evaluating and running this code as easy as possible, an automated startup script is provided.

**Windows Users:**
Simply double-click on `Run_Dashboard.bat`. 

This script will automatically:
1. Verify Python is installed.
2. Create an isolated virtual environment (`.venv`).
3. Install all required dependencies from `requirements.txt`.
4. Spin up the FastAPI web server.
5. Pop open `http://localhost:8000` in your default web browser instantly.

**Mac/Linux Users:**
Run the following commands in your terminal:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app
```
Then navigate to `http://localhost:8000`.

## Architecture Overview

- `api.py`: Holds the web server and API endpoints. 
- `main.py`: Contains the core hospital system logic. Running it directly executes a terminal-based simulation validating all AI subsystems.
- `static/index.html`: Holds the dynamic frontend interface. 
- `ml/`, `vision/`, `agents/`, `search/`, `rl/`, `nlp/`, `knowledge/`: Modular domains containing independent logic driving the entire hospital system.
