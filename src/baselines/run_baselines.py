"""Shim removed.

Baseline code moved to `src.baselines.baselines`. Update imports to:

	from src.baselines.baselines import run_baselines

Importing this module now raises an ImportError to surface remaining
references at import time.
"""

raise ImportError(
	"The compatibility shim `src.baselines.run_baselines` has been removed. Import from 'src.baselines.baselines' instead."
)
