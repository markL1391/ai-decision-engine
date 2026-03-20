# AI Decision Engine

An explainable AI-supported decision framework that transforms raw metrics into structural indicators, evaluates system capability deterministically, and explains structural limitations using LLMs and RAG.

## Current status
MVP in progress

## Tech stack
- Flask
- SQLite
- SQLAlchemy
- Pydantic

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
``
Endpoints

GET /health

POST /assessments/analyze

GET /assessments/<id>