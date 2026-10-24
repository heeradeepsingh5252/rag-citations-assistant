"""Central config. Read secrets from environment variables, never hardcode them."""
import os

APP_NAME = os.getenv("APP_NAME", "ai-project")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
