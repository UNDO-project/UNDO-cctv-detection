# This is the cctv detection application for the UNDO.

To run this project you need to create a virtual environment. Open up a terminal and type the following:

```commandline
python3.12 -m venv venv
```

Activate the newly created env:

```commandline
source "PATH-TO-VENV/venv/bin/activate"
```

If needed you can deactivate the virtual environment from within root of project:

```commandline
deactivate
```

Install the dependencies for this project:

```commandline
pip install -r requirements.txt
```

## Testing 
In order to run the tests first:

```commandline
pip install -r requiremens_tests.txt
```

Then from the root project run:

```commandline
bash ./local_test_pipeline.sh
```

## Code formatting

This project uses [.pre-commit](https://pre-commit.com) hooks to ensure universal code formatting.

To install these use:

```commandline
pre-commit install
```

The hooks will run the [ruff](https://docs.astral.sh/ruff/) formatter with every commit.