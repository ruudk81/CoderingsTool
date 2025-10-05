import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

"""
Example/Testing script for CodebookDeduplicator

This script demonstrates how to use the codebook deduplicator utility
with results from the codeGenerator.
"""

from utils.codebookDeduplicator import deduplicate_codebook, print_deduplication_report
from utils.codeGenerator import CodeGeneratorReasoningResults

def test_deduplicator_with_sample_data():
    """Test the deduplicator with mock data"""
    
    # Create mock CodeGeneratorReasoningResults
    # In real usage, this would come from codeGenerator.generate()
    mock_results = CodeGeneratorReasoningResults(
        cluster_results=[],
        step1_inputs={
            "cluster_1": {
                "ideas": [
                    "Ik vind het belangrijk om milieuvriendelijk te zijn",
                    "Duurzaamheid is essentieel voor de toekomst", 
                    "We moeten meer aandacht besteden aan groene energie",
                    "Klimaatverandering is een groot probleem"
                ]
            },
            "cluster_2": {
                "ideas": [
                    "Het milieu beschermen is zeer belangrijk",
                    "Groene technologie helpt de planeet",
                    "Hernieuwbare energie is de oplossing",
                    "We moeten zorgen voor een schone wereld"
                ]
            },
            "cluster_3": {
                "ideas": [
                    "Geld besparen door energie-efficiëntie",
                    "Goedkopere oplossingen voor huishoudens",
                    "Kosteneffectieve maatregelen nemen",
                    "Financiële voordelen van duurzaamheid"
                ]
            }
        },
        step1_summaries={},
        step2_analysis={},
        step3_recommendations={},
        step4_validations={},
        step4_validated_codes={
            "cluster_1": {
                "code": "Milieubewustzijn",
                "definition": "Bewustzijn van het belang van milieubescherming en duurzaamheid",
                "theme_id": "theme_1"
            },
            "cluster_2": {
                "code": "Duurzame_ontwikkeling", 
                "definition": "Aandacht voor duurzame ontwikkeling en milieuvriendelijke praktijken",
                "theme_id": "theme_2"
            },
            "cluster_3": {
                "code": "Kosteneffectiviteit",
                "definition": "Focus op financiële voordelen en kostenbesparingen",
                "theme_id": "theme_3"
            }
        },
        stats={},
        generator_version="test",
        var_lab="Test vraag over milieu",
        total_clusters=3,
        total_ideas=12,
        processing_timestamp="2024-01-01T10:00:00",
        cluster_assignments={
            "cluster_1": {"code": "Milieubewustzijn", "definition": "Bewustzijn van het belang van milieubescherming"},
            "cluster_2": {"code": "Duurzame_ontwikkeling", "definition": "Aandacht voor duurzame ontwikkeling"},
            "cluster_3": {"code": "Kosteneffectiviteit", "definition": "Focus op financiële voordelen"}
        },
        codebook=[
            {"code": "Milieubewustzijn", "definition": "Bewustzijn van het belang van milieubescherming en duurzaamheid"},
            {"code": "Duurzame_ontwikkeling", "definition": "Aandacht voor duurzame ontwikkeling en milieuvriendelijke praktijken"},
            {"code": "Kosteneffectiviteit", "definition": "Focus op financiële voordelen en kostenbesparingen"}
        ],
        cluster_data={}
    )
    
    print("Testing CodebookDeduplicator with sample data...")
    print(f"Original codebook has {len(mock_results.codebook)} codes")
    
    # Run deduplication with lower threshold for testing
    results = deduplicate_codebook(
        reasoning_results=mock_results,
        similarity_threshold=0.7,  # Lower threshold to potentially catch similarities
        language="Dutch"
    )
    
    # Print results
    print_deduplication_report(results)
    
    return results

def example_usage_with_real_data():
    """Example of how to use with real codeGenerator output"""
    
    print("""
EXAMPLE USAGE WITH REAL DATA:
=============================

# Step 1: Run the codeGenerator to get reasoning results
from utils.codeGenerator import InductiveCodeGenerator

code_generator = InductiveCodeGenerator(
    cluster_results=your_cluster_results,
    starter_codes=your_starter_codes, 
    var_lab=your_survey_question
)

reasoning_results = code_generator.generate()

# Step 2: Run codebook deduplication
from utils.codebookDeduplicator import deduplicate_codebook, print_deduplication_report

dedup_results = deduplicate_codebook(
    reasoning_results=reasoning_results,
    similarity_threshold=0.9,  # High threshold for conservative merging
    language="Dutch"  # or "English"
)

# Step 3: Review results
print_deduplication_report(dedup_results)

# Step 4: Access deduplicated codebook
final_codebook = dedup_results.deduplicated_codebook
merge_log = dedup_results.merge_decisions

# Optional: Save results
import json
with open('deduplication_results.json', 'w') as f:
    json.dump({
        'original_codebook': dedup_results.original_codebook,
        'deduplicated_codebook': dedup_results.deduplicated_codebook,
        'stats': dedup_results.processing_stats
    }, f, indent=2)
    """)

if __name__ == "__main__":
    print("CodebookDeduplicator Test Script")
    print("="*50)
    
    try:
        # Test with sample data
        test_results = test_deduplicator_with_sample_data()
        
        print("\nTest completed successfully!")
        print(f"Similarity matrix size: {len(test_results.similarity_matrix)}")
        
        # Show example usage
        example_usage_with_real_data()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()