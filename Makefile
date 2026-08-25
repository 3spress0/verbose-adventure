# Convenience targets for the combined Generative Agents × tgpt project.
# All targets operate from the top-level directory.

.PHONY: help install-tgpt install-pytgpt install-python install-all \
        backend-check smoke-test run-env run-sim clean

PYTHON ?= python3

help:
	@echo "Available targets:"
	@echo "  install-tgpt     Install the tgpt Go CLI"
	@echo "  install-pytgpt   Install the python-tgpt Python package (editable)"
	@echo "  install-python   Install the generative_agents Python deps"
	@echo "  install-all      All of the above"
	@echo "  backend-check    Show which LLM backend tgpt_backend will pick"
	@echo "  smoke-test       Run tgpt_backend + gpt_structure self-test"
	@echo "  run-env          Start the Django environment server (foreground)"
	@echo "  run-sim          Start the simulation backend (foreground)"
	@echo "  clean            Remove __pycache__ directories"

install-tgpt:
	go install github.com/aandrew-me/tgpt/v2@latest

install-pytgpt:
	$(PYTHON) -m pip install -e python-tgpt

install-python:
	$(PYTHON) -m pip install -r generative_agents/environment/frontend_server/requirements.txt
	$(PYTHON) -m pip install numpy

install-all: install-python install-tgpt install-pytgpt
	@echo "All dependencies installed."

backend-check:
	@$(PYTHON) -c "import sys; sys.path.insert(0, '.'); from tgpt_backend import LLMClient; print('Selected backend:', LLMClient().backend)"

smoke-test:
	@$(PYTHON) -c "import sys; sys.path.insert(0, '.'); \
from tgpt_backend import LLMClient; c = LLMClient(); \
print('backend =', c.backend); \
print('chat:', repr(c.chat('hi')[:80])); \
v = c.embed('hello world'); print('embed dim =', len(v), 'norm =', sum(x*x for x in v)**0.5)"

run-env:
	cd generative_agents/environment/frontend_server && $(PYTHON) manage.py runserver

run-sim:
	cd generative_agents/reverie/backend_server && $(PYTHON) reverie.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
