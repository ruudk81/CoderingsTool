#!/usr/bin/env python3
"""
Test script for the updated ThemeIdentifier with enhanced validation
"""

import asyncio
import sys
sys.path.append('/workspaces/CoderingsTool/src')

from utils.themeIdentifier import ThemeIdentifier
from pydantic import BaseModel, Field

# Mock code model for testing
class MockCode(BaseModel):
    code: str = Field(description="Code text")
    definition: str = Field(description="Code definition")

def create_test_codebook(num_codes: int = 25):
    """Create a test codebook with the specified number of codes"""
    test_codes = []
    
    # Create diverse codes for testing
    code_templates = [
        ("Tevredenheid over {}", "Positieve ervaringen met {}"),
        ("Ontevredenheid over {}", "Negatieve ervaringen met {}"),
        ("Effectiviteit van {}", "De mate waarin {} resultaat oplevert"),
        ("Betrokkenheid bij {}", "De rol van stakeholders in {}"),
        ("Communicatie over {}", "Informatie-uitwisseling betreffende {}"),
    ]
    
    topics = [
        "gezondheidsbevordering", "interventies", "schoolbeleid", 
        "samenwerking", "resultaten", "implementatie", "training",
        "middelen", "ondersteuning", "monitoring"
    ]
    
    for i in range(num_codes):
        template_idx = i % len(code_templates)
        topic_idx = i % len(topics)
        
        code_template, def_template = code_templates[template_idx]
        topic = topics[topic_idx]
        
        # Add variation to make codes unique
        if i >= len(code_templates) * len(topics):
            topic = f"{topic} (variant {i // (len(code_templates) * len(topics))})"
        
        test_codes.append(MockCode(
            code=code_template.format(topic),
            definition=def_template.format(topic)
        ))
    
    return test_codes

async def test_theme_identifier():
    """Test the ThemeIdentifier with enhanced validation"""
    
    print("="*80)
    print("TESTING ENHANCED THEME IDENTIFIER")
    print("="*80)
    
    # Create test data
    codebook = create_test_codebook(25)  # Test with 25 codes
    var_lab = "U geeft aan dat u tevreden of ontevreden bent met de resultaten van de Gezonde School-aanpak op uw school. Wilt u uw antwoord toelichten?"
    
    # Initialize ThemeIdentifier
    identifier = ThemeIdentifier(
        codebook=codebook,
        var_lab=var_lab,
        verbose=True
    )
    
    print(f"\nTest Configuration:")
    print(f"- Total codes: {len(codebook)}")
    print(f"- Batch size: {identifier.batch_size} (should be 10)")
    print(f"- Max retries: {identifier.max_hierarchy_retries} (should be 5)")
    print(f"- Expected batches: {(len(codebook) + identifier.batch_size - 1) // identifier.batch_size}")
    
    # Run the hierarchical theme identification
    try:
        result = await identifier.identify_themes_hierarchical()
        
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        
        if 'hierarchy' in result and result['hierarchy']:
            hierarchy = result['hierarchy']
            print(f"\nFinal Structure:")
            print(f"- Themes: {len(hierarchy.themes)}")
            
            total_codes_in_result = 0
            for theme in hierarchy.themes:
                domain_count = len(theme.domains)
                code_count = sum(len(domain.codes) for domain in theme.domains)
                total_codes_in_result += code_count
                print(f"\n  Theme: {theme.theme_name}")
                print(f"  - Domains: {domain_count}")
                print(f"  - Codes: {code_count}")
            
            print(f"\nValidation:")
            print(f"- Expected codes: {len(codebook)}")
            print(f"- Found codes: {total_codes_in_result}")
            print(f"- Status: {'✅ PASS' if total_codes_in_result == len(codebook) else '❌ FAIL'}")
            
            # Check for duplicates
            all_code_numbers = []
            for theme in hierarchy.themes:
                for domain in theme.domains:
                    for code in domain.codes:
                        all_code_numbers.append(code.code_number)
            
            if len(all_code_numbers) != len(set(all_code_numbers)):
                print(f"⚠️  WARNING: Duplicate codes detected!")
            
        else:
            print("❌ No hierarchy returned")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_theme_identifier())