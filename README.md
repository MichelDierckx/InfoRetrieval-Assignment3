# Project 3 information retrieval

## Description

Assignment 3 for the course Information Retrieval 2024-2025 at UAntwerpen.

## Usage

### Requirements

Install requirements via:

```bash
pip install -r requirements.txt
```

### Running the program

To index, query and rank documents use:

```bash
python3 -m src.scripts.run_ranker -c /path/to/config.ini
```

To evaluate rankings:

```bash
python3 -m src.scripts.run_evaluator -c /path/to/config.ini
```

#### Program arguments

You can provide program arguments in two ways:

1. Via a configuration file using the ```-c``` flag.
2. Directly on the command line.

To view a list of available arguments for each script, use the ```-h``` flag (help option).
