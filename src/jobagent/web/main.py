"""ASGI entry point:  .venv\\Scripts\\uvicorn.exe jobagent.web.main:app"""
from .app import create_app

app = create_app()
