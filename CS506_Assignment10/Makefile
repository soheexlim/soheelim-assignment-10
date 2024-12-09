# Define variables
PYTHON = python3.11  # Change to python3.10 or python3.11 explicitly
VENV = venv
FLASK_APP = app.py

# Check Python version
check-python:
	@$(PYTHON) --version | grep -E "Python 3\.(10|11)" || \
	(echo "Error: Python 3.10 or 3.11 is required." && exit 1)

# Install dependencies and create the virtual environment
install: check-python
	$(PYTHON) -m venv $(VENV)
	./$(VENV)/bin/pip install --upgrade pip
	./$(VENV)/bin/pip install -r requirements.txt

# Generate precomputed embeddings
generate:
	./$(VENV)/bin/python generate_embeddings.py

# Run the Flask application
run:
	FLASK_APP=$(FLASK_APP) FLASK_ENV=development ./$(VENV)/bin/flask run --port 3000

# Clean up virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies
reinstall: clean install
