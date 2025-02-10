# This is the cctv detection application for the UNDO.

To run this project you need to create a virtual environment. Open up a terminal and type the following:

```commandline
python3.11 -m venv venv
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

## Image labelling
To label images you need to have [Docker](https://www.docker.com) and [label-studio](https://labelstud.io) installed.

From a terminal run the following to pull the latest label-studio image:

```commandline
docker pull heartexlabs/label-studio:latest
```

To run the container:

```commandline
docker run -it -p 8080:8080 -v $(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest
```

Open your web browser and navigate to http://localhost:8080.
Upon first access, you’ll be prompted to create a username and password. 
This account is for local use only and does not require any external registration.


### Disclaimer
Some images used in this project come from the dataset of the [Fuziih CCTV-Exposure](https://github.com/Fuziih/cctv-exposure/tree/main)

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