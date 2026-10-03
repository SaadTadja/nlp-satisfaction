"""French client-satisfaction analysis.

Pure-python/numpy modules (``normalize``, ``estimate``, ``aspects``) import
without torch so they can be tested on any machine. ``predict`` pulls in torch
and transformers lazily, inside ``SatisfactionModel.__init__``.
"""

__version__ = "0.1.0"
