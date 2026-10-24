
.PHONY: install test run docker-build docker-run

install:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements-dev.txt

test:
	.venv/bin/python -m pytest

run:
		.venv/bin/python -m uvicorn src.app.main:app --reload --host 127.0.0.1 --port 8000

docker-build:
	docker build -t rag-citations-assistant .

docker-run:
	docker run --rm -p 8000:8000 rag-citations-assistant
