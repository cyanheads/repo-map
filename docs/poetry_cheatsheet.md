# Python Poetry Cheatsheet

### Create a new project
```
poetry new <project-name>
```

### Add a new lib
```
poetry add <library>
```

### Remove a lib
```
poetry remove <library>
```

### Update a lib
```
poetry update <library>
```

### Get venv path
```
poetry run which python
```

### Run app
```
poetry run python app.py
```

### Run tests
```
poetry run python -m unittest discover
```

### Create script

1 - Edit `pyproject.toml`:
```
[tool.poetry.scripts]
test = 'scripts:test'
```

2 - Create a `scripts.py` file on the root directory of your project:
```python
import subprocess

def test():
    """
    Run all unittests. Equivalent to:
    `poetry run python -u -m unittest discover`
    """
    subprocess.run(
        ['python', '-u', '-m', 'unittest', 'discover']
    )
```

3 - Run script:
```
poetry run test
```

### Disable virtual environment creation
```
poetry config virtualenvs.create false
```

### List configuration
```
poetry config --list
```

### Activate virtual environment
```
poetry shell
```

### Install dependencies
```
poetry install
```

### Show dependencies
```
poetry show
