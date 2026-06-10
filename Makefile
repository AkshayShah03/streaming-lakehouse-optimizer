.PHONY: help install test demo lint up down seed connect flink-submit stream train

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install Python deps
	pip install -r requirements.txt

test: ## Run the unit suite (no cluster needed)
	PYTHONPATH=src python -m pytest

demo: ## End-to-end demo on the simulator (no cluster needed)
	python scripts/run_local_demo.py

train: ## Train + persist the cost model
	PYTHONPATH=src python -m lakehouse.optimizer.train --out artifacts/cost_model.joblib

lint: ## Lint
	ruff check src tests

up: ## Start the full local stack
	docker compose up -d --build

down: ## Tear down the stack
	docker compose down -v

seed: ## Create the source table in Postgres
	docker compose exec -T postgres psql -U lake -d fleet < scripts/seed_postgres.sql

connect: ## Register the Debezium CDC connector
	curl -s -X POST -H "Content-Type: application/json" \
	  --data @infra/debezium/postgres-connector.json http://localhost:8083/connectors | jq .

flink-submit: ## Submit the PyFlink Kafka->Iceberg job
	docker compose exec flink-jobmanager flink run -py /opt/src/lakehouse/ingest/flink_job.py

RPS ?= 10000
SECONDS ?= 120
stream: ## Generate telemetry into Postgres (RPS, SECONDS overridable)
	PYTHONPATH=src python -m lakehouse.ingest.telemetry_generator --rps $(RPS) --seconds $(SECONDS)
