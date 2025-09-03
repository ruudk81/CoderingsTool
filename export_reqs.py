import importlib.metadata as m
from packaging.version import Version

IGNORE = {"pip", "setuptools", "wheel"}

best = {}  # lowercased name -> (canonical Name, version)
for d in m.distributions():
    name = d.metadata.get("Name")
    if not name or name in IGNORE:
        continue
    key = name.lower()
    v = d.version
    if key not in best or Version(v) > Version(best[key][1]):
        best[key] = (name, v)

for name, v in sorted(best.values(), key=lambda x: x[0].lower()):
    print(f"{name}=={v}")
