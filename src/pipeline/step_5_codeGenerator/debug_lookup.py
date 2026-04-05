"""Debug script: verify domain key mismatch between ideas and ExtractionMetadata."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "steps"))

from pipeline.step_3_ideaExtractor import models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

variable_key = generate_enhanced_variable_key(
    selected_variables=[VARIABLE],
    is_merged=False,
    sample_size=SAMPLE_SIZE
)

cache_manager = CacheManager()

# 1. Load ExtractionMetadata
metadata = cache_manager.load_metadata_from_cache(
    filename=FILENAME,
    step="extracted_ideas",
    variable_key=variable_key,
    model_cls=models.ExtractionMetadata
)

print("=" * 60)
print("DOMAINS IN ExtractionMetadata (lookup keys)")
print("=" * 60)
if metadata and metadata.domains:
    for d in metadata.domains:
        print(f"  key:        {d['key']!r}")
        print(f"  label:      {d['label']!r}")
        print(f"  definition: {d['definition']!r}")
        print()
else:
    print("  No domains found in metadata!")

# 2. Load ideas and collect unique domain values
data = cache_manager.load_from_cache(
    FILENAME, "extracted_ideas", variable_key, models.IdeasExtractedModel
)

domain_values = set()
for resp in data:
    if not resp.response_ideas:
        continue
    for idea in resp.response_ideas:
        domain = (getattr(idea, 'domain', '') or '').strip()
        if domain:
            domain_values.add(domain)

print("=" * 60)
print("DOMAIN VALUES ON IDEAS (partition keys, before lowercasing)")
print("=" * 60)
for d in sorted(domain_values):
    print(f"  {d!r}")

# 3. Show the mismatch — replicate DomainDiscoverer's lookup logic
print()
print("=" * 60)
print("LOOKUP TEST (replicating DomainDiscoverer)")
print("=" * 60)
if metadata and metadata.domains:
    domains_lookup = {}
    for d in metadata.domains:
        key = d.get('key', '')
        if key:
            domains_lookup[key] = d
        label = d.get('label', '')
        if label:
            domains_lookup[label.lower()] = d

    print(f"  Lookup keys: {sorted(domains_lookup.keys())}")
    print()
    for domain_val in sorted(domain_values):
        lowered = domain_val.lower()
        match = domains_lookup.get(lowered)
        defn = match.get('definition', '') if match else ''
        print(f"  {lowered!r} → {'FOUND' if match else 'MISS'}")
        if defn:
            print(f"    definition: {defn[:80]}...")
