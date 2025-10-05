"""Debug script to understand codebook_reasoning structure"""

def debug_codebook_reasoning(codebook_reasoning):
    """Debug the structure of codebook_reasoning to understand why mappings aren't found"""
    
    print("=== DEBUGGING CODEBOOK_REASONING ===\n")
    
    # 1. Check basic structure
    print("1. Basic Structure:")
    print(f"   Type: {type(codebook_reasoning)}")
    print(f"   Has step1_inputs: {hasattr(codebook_reasoning, 'step1_inputs')}")
    print(f"   Has step4_validated_codes: {hasattr(codebook_reasoning, 'step4_validated_codes')}")
    print(f"   Has cluster_assignments: {hasattr(codebook_reasoning, 'cluster_assignments')}")
    print(f"   Has codebook: {hasattr(codebook_reasoning, 'codebook')}")
    
    # 2. Check step1_inputs
    if hasattr(codebook_reasoning, 'step1_inputs'):
        step1_inputs = codebook_reasoning.step1_inputs
        print(f"\n2. step1_inputs:")
        print(f"   Type: {type(step1_inputs)}")
        print(f"   Number of clusters: {len(step1_inputs) if step1_inputs else 0}")
        if step1_inputs:
            print(f"   Cluster IDs: {list(step1_inputs.keys())[:5]}{'...' if len(step1_inputs) > 5 else ''}")
            # Sample first cluster
            first_key = list(step1_inputs.keys())[0]
            first_value = step1_inputs[first_key]
            print(f"   Sample cluster '{first_key}':")
            print(f"     Type: {type(first_value)}")
            print(f"     Keys: {list(first_value.keys()) if isinstance(first_value, dict) else 'Not a dict'}")
            if isinstance(first_value, dict):
                if 'cluster_text' in first_value:
                    text = first_value['cluster_text']
                    print(f"     cluster_text exists: {len(text)} chars")
                    print(f"     First 100 chars: {text[:100]}...")
                if 'ideas' in first_value:
                    ideas = first_value['ideas']
                    print(f"     ideas exists: {len(ideas) if isinstance(ideas, list) else 'Not a list'}")
    
    # 3. Check step4_validated_codes
    if hasattr(codebook_reasoning, 'step4_validated_codes'):
        step4_validated_codes = codebook_reasoning.step4_validated_codes
        print(f"\n3. step4_validated_codes:")
        print(f"   Type: {type(step4_validated_codes)}")
        print(f"   Number of entries: {len(step4_validated_codes) if step4_validated_codes else 0}")
        if step4_validated_codes:
            print(f"   Keys: {list(step4_validated_codes.keys())[:5]}{'...' if len(step4_validated_codes) > 5 else ''}")
            # Sample first entry
            first_key = list(step4_validated_codes.keys())[0]
            first_value = step4_validated_codes[first_key]
            print(f"   Sample entry '{first_key}':")
            print(f"     Type: {type(first_value)}")
            if isinstance(first_value, dict):
                print(f"     Keys: {list(first_value.keys())}")
                if 'code' in first_value:
                    print(f"     code: '{first_value['code']}'")
                if 'definition' in first_value:
                    print(f"     definition: '{first_value['definition'][:50]}...'")
    
    # 4. Check cluster_assignments
    if hasattr(codebook_reasoning, 'cluster_assignments'):
        cluster_assignments = codebook_reasoning.cluster_assignments
        print(f"\n4. cluster_assignments:")
        print(f"   Type: {type(cluster_assignments)}")
        print(f"   Number of entries: {len(cluster_assignments) if cluster_assignments else 0}")
        if cluster_assignments:
            print(f"   Keys: {list(cluster_assignments.keys())[:5]}{'...' if len(cluster_assignments) > 5 else ''}")
    
    # 5. Check codebook
    if hasattr(codebook_reasoning, 'codebook'):
        codebook = codebook_reasoning.codebook
        print(f"\n5. codebook:")
        print(f"   Type: {type(codebook)}")
        print(f"   Number of codes: {len(codebook) if codebook else 0}")
        if codebook and len(codebook) > 0:
            print(f"   First code: {codebook[0] if codebook else 'None'}")
    
    # 6. Compare cluster IDs between step1 and step4
    if hasattr(codebook_reasoning, 'step1_inputs') and hasattr(codebook_reasoning, 'step4_validated_codes'):
        step1_ids = set(codebook_reasoning.step1_inputs.keys()) if codebook_reasoning.step1_inputs else set()
        step4_ids = set(codebook_reasoning.step4_validated_codes.keys()) if codebook_reasoning.step4_validated_codes else set()
        print(f"\n6. Cluster ID Comparison:")
        print(f"   step1_inputs cluster IDs: {len(step1_ids)}")
        print(f"   step4_validated_codes cluster IDs: {len(step4_ids)}")
        print(f"   Common cluster IDs: {len(step1_ids.intersection(step4_ids))}")
        if step1_ids and step4_ids and not step1_ids.intersection(step4_ids):
            print(f"   ⚠️  WARNING: No common cluster IDs between step1 and step4!")
            print(f"   Sample step1 IDs: {list(step1_ids)[:3]}")
            print(f"   Sample step4 IDs: {list(step4_ids)[:3]}")

# Example usage
if __name__ == "__main__":
    print("Run this function with: debug_codebook_reasoning(codebook_reasoning)")