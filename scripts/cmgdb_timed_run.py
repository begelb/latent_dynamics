"""Time the study's own adaptive CMGDB call at a chosen subdivision.

No instrumentation: the persisted box map is handed to CMGDB exactly as
_run_lookup_cmgdb does, batch callback included, so the measured time is the
code's, not a wrapper's.
"""
import json, sys, time
sys.path.insert(0, "src")
import CMGDB
from latentdynamics.analysis.hierarchical_precomputed import HierarchicalPrecomputedBoxMap

ROOT = "output/chafee_latent_dimension_study/latent_3d/seed_0"
init, minimum, maximum = (int(v) for v in sys.argv[1:4])

table = HierarchicalPrecomputedBoxMap.load(f"{ROOT}/precomputed_level24_to33", mmap_mode="r")
bounds = json.loads(open(f"{ROOT}/bounds.json").read())

model = CMGDB.Model(
    minimum, maximum, init, 10000,
    [float(v) for v in bounds["lower"]],
    [float(v) for v in bounds["upper"]],
    table,
)
if hasattr(model, "set_batch_map"):
    model.set_batch_map(table.batch)

start = time.perf_counter()
mg, mp = CMGDB.ComputeConleyMorseGraph(model)
elapsed = time.perf_counter() - start
print(f"RESULT subdiv=({init},{minimum},{maximum}) sets={mg.num_vertices()} "
      f"cells={mp.num_vertices()} seconds={elapsed:.1f}")
