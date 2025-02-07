"""Script to display the codebook after running thematicLabeller"""

# After running the labeller, use this to see the codebook:

from utils.codebookDisplay import display_codebook, display_phase4_changes

# Display the full codebook
display_codebook(labeller.codebook, "FINAL CODEBOOK (After Phase 4)")

# Show what Phase 4 did
print("\n" + "="*80)
print("PHASE 4 ANALYSIS")
print("="*80)
display_phase4_changes(labeller.codebook_before_phase4, labeller.final_labels)

# Check if Phase 4 actually changed anything
print("\n" + "="*80)
print("DID PHASE 4 CHANGE ANYTHING?")
print("="*80)
print("Note: Current Phase 4 implementation does NOT modify the codebook.")
print("It only creates final_labels mapping clusters to themes/topics.")
print("The codebook remains unchanged from Phase 2.")

# Show some statistics
print("\n📊 Assignment Statistics:")
theme_counts = {}
for cluster_id, labels in labeller.final_labels.items():
    theme = labels['theme'][0]
    theme_counts[theme] = theme_counts.get(theme, 0) + 1

for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"   Theme '{theme}': {count} clusters")