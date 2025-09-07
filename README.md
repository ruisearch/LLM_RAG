A benchmark for local RAG system consist of the following key components:

* A dataset from [open-rag-bench](https://github.com/vectara/open-rag-bench), with 1,000 Pdfs from Arxiv and the corresponding question-answer pairs
* A typical local RAG system, where the retriever is deployed locally while the generator (LLM) is local or remote (by API).
* Evaluating system based on [Ragas](https://docs.ragas.io/en/stable/).


# Requirements

\- [Ollama](https://ollama.ai/) version 0.5.7 or higher for local LLM usage.

\- If the LLM is accessed by API, corresponding API key should be set.

# Setup

1. Clone this repository to your local machine.
2. Install UV, [Installation](https://docs.astral.sh/uv/#installation)
3. Create a virtual environment and install the required Python packages by running `uv sync`

# Running the local RAG system

The main script for this RAG is main.py. Its arguments are as follows:

```
usage: main.py [-h] [-m MODEL] [-e EMBEDDING_MODEL] [-p PATH] [-s STORAGE] [-r RELOAD] [-d DEBUG] [-l LOCAL] [--batch]
               [-q QUESTIONS] [-o OUTPUT] [-c CONTEXT]

A Local RAG system.

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The name of the LLM. Defaults to llama3:8b
  -e EMBEDDING_MODEL, --embedding_model EMBEDDING_MODEL
                        The name of embedding model. Defaults to nomic-embed-text
  -p PATH, --path PATH  The path to the directory containing documents to load. Defaults to ./Research/
  -s STORAGE, --storage STORAGE
                        The path to the directory containing database. Defaults to ./storage/
  -r RELOAD, --reload RELOAD
                        Whether reload the database. please type True or False. Defaults to True
  -d DEBUG, --debug DEBUG
                        Whether print the debug information. please type True or False. Defaults to False
  -l LOCAL, --local LOCAL
                        Whether use local model. True=local model, False=API model. Default is True
  --batch               Enable batch testing mode
  -q QUESTIONS, --questions QUESTIONS
                        Path to questions JSON file for batch testing
  -o OUTPUT, --output OUTPUT
                        Path to results JSON file for batch testing
  -c CONTEXT, --context CONTEXT
                        Path to context JSON file for batch testing
```

the command is `uv run main.py [arg....]`

# An example for Benchmark

TODO....