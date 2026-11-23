"""SPM / SW-MSR experiment layer for MOSAIC.

Additive: nothing in MOSAIC's own source is modified. The only upstream module
imported directly is database_manager (standard library only), so the base schema
of a run is a real MOSAIC database.
"""

from .config import Config  # noqa: F401
