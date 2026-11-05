# Convenience entry points for the assessment repository.
# Override PYTHON to point at the interpreter of your virtual environment:
#   make PYTHON=/path/to/venv/bin/python install test
PYTHON ?= python3
PIP    := $(PYTHON) -m pip

TASKS := task1-mcp-server task2-mcp-gateway task3-llm-guardrail task4-model-router

.PHONY: help install test lint format typecheck check audit lock clean \
        run-task1 run-task2-downstream run-task2-gateway run-task3-provider run-task3-gateway \
        run-task4-primary run-task4-secondary run-task4-router \
        demo-task1 demo-task2 demo-task3 demo-task4

help:
	@echo "Targets:"
	@echo "  install        Install all four tasks in editable mode with dev extras"
	@echo "  test           Run the complete test suite"
	@echo "  lint           Run ruff (lint + format check)"
	@echo "  typecheck      Run mypy in strict mode"
	@echo "  check          lint + typecheck + test"
	@echo "  audit          Scan the pinned dependency set for known vulnerabilities (pip-audit)"
	@echo "  lock           Regenerate requirements.lock from the current environment"
	@echo "  run-task1      Start the stdio MCP server (task 1)"
	@echo "  run-task2-*    Start the mock downstream / gateway (task 2)"
	@echo "  run-task3-*    Start the mock provider / guardrail gateway (task 3)"
	@echo "  run-task4-*    Start the mock providers / router (task 4)"
	@echo "  demo-taskN     Scripted end-to-end demonstration for task N"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@for t in $(TASKS); do $(PIP) install --no-deps -e $$t || exit 1; done

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

typecheck:
	$(PYTHON) -m mypy $(addsuffix /src,$(TASKS))

check: lint typecheck test

# pip-audit is deliberately not a dependency of the services; install it in the
# tooling environment (`$(PIP) install pip-audit`) or run it from CI.
audit:
	@$(PYTHON) -c "import pip_audit" 2>/dev/null || { echo "pip-audit is not installed: $(PIP) install pip-audit"; exit 1; }
	$(PYTHON) -m pip_audit --strict -r requirements.lock

lock:
	$(PYTHON) scripts/freeze_lock.py > requirements.lock

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache -o -name '*.egg-info' \) -prune -exec rm -rf {} +
	rm -rf task4-model-router/data

# ---- Task 1: MCP server over stdio -------------------------------------------
run-task1:
	$(PYTHON) -m customer_mcp_server

demo-task1:
	$(PYTHON) task1-mcp-server/scripts/stdio_smoke_test.py

# ---- Task 2: MCP security gateway --------------------------------------------
run-task2-downstream:
	PYTHONPATH=task2-mcp-gateway/src $(PYTHON) -m mock_mcp_server --port 9100

run-task2-gateway:
	MCP_GATEWAY_DOWNSTREAM_URL=http://127.0.0.1:9100/mcp \
	MCP_GATEWAY_TOKENS_FILE=task2-mcp-gateway/config/tokens.dev.json \
	MCP_GATEWAY_ALLOW_DEVELOPMENT_CREDENTIALS=true \
	$(PYTHON) -m mcp_gateway --port 9000

demo-task2:
	bash task2-mcp-gateway/scripts/demo.sh

# ---- Task 3: LLM gateway streaming guardrail ---------------------------------
run-task3-provider:
	PYTHONPATH=task3-llm-guardrail/src $(PYTHON) -m mock_llm_provider --port 9200

run-task3-gateway:
	LLM_GATEWAY_UPSTREAM_BASE_URL=http://127.0.0.1:9200 LLM_GATEWAY_ALLOW_UNAUTHENTICATED=true \
	$(PYTHON) -m llm_guardrail_gateway --port 9300

demo-task3:
	bash task3-llm-guardrail/scripts/demo.sh

# ---- Task 4: rate limiting and model fallback router -------------------------
run-task4-primary:
	PYTHONPATH=task4-model-router/src MOCK_PROVIDER_NAME=primary MOCK_PROVIDER_BEHAVIOR=rate_limited \
	$(PYTHON) -m mock_model_provider --port 9401

run-task4-secondary:
	PYTHONPATH=task4-model-router/src MOCK_PROVIDER_NAME=secondary MOCK_PROVIDER_BEHAVIOR=ok \
	$(PYTHON) -m mock_model_provider --port 9402

run-task4-router:
	MODEL_ROUTER_PRIMARY_URL=http://127.0.0.1:9401 \
	MODEL_ROUTER_SECONDARY_URL=http://127.0.0.1:9402 \
	MODEL_ROUTER_TENANTS_FILE=task4-model-router/config/tenants.dev.json \
	MODEL_ROUTER_DATABASE_PATH=task4-model-router/data/model_router.sqlite3 \
	MODEL_ROUTER_ALLOW_DEVELOPMENT_CREDENTIALS=true \
	$(PYTHON) -m model_router --port 9400

demo-task4:
	bash task4-model-router/scripts/demo.sh
