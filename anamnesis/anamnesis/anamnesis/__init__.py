"""Anamnesis — a retrieval-grounded clinical information assistant.

The package is deliberately split so each layer can be tested on its own:
``embeddings`` -> ``vectorstore`` -> ``retrieval`` feed the knowledge side,
``db`` -> ``tools`` feed the structured side. ``agent`` is the only module that
combines them, and ``safety`` sits in front of both.
"""

__version__ = "1.0.0"
