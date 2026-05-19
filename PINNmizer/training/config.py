"""
Training configuration helpers.

Configuration and command-line parsing are experiment wiring, not PDE or biology
logic. Keeping them separate prevents long training scripts from hiding the
mathematical parts of the code.

Future target
-------------
Move argument parsing, run configuration dictionaries, and curriculum schedule
parsing here. Keep defaults unchanged during the move.
"""
