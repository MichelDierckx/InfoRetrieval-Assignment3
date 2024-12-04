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

To generate embeddings for documents use:

```bash
python3 -m src.scripts.run_document_embedder -c /path/to/config.ini
```

To generate an index for document embeddings use:

```bash
python3 -m src.scripts.run_indexer -c /path/to/config.ini
```

To rank documents given an index and queries use:

```bash
python3 -m src.scripts.run_ranker -c /path/to/config.ini
```

To evaluate generated rankings using a reference ranking use:

```bash
python3 -m src.scripts.run_evaluator -c /path/to/config.ini
```

#### Program arguments

You can provide program arguments in two ways:

1. Via a configuration file using the ```-c``` flag.
2. Directly on the command line.

To view a list of available arguments for each script, use the ```-h``` flag (help option).
