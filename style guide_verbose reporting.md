# Style Guide for Verbose Reporting

This guide documents the verbose reporting patterns used throughout the CoderingsTool pipeline to maintain consistency across all utilities.

## Core Principles

1. **Two-tier information display**:
   - **Default prints**: Basic input/output information, always shown
   - **Verbose reporter**: Detailed processing insights, configuration, statistics (only when verbose=True)

2. **Consistent structure for each processing phase**:
   - Section header (uppercase title)
   - Configuration details (verbose only)
   - Progress indicators
   - Results summary
   - Sample outputs (verbose only)

## Standard Patterns

### Phase Structure

Each major processing phase follows this pattern:

```
📝 PHASE NAME
==============

🔄 Processing step
• Configuration parameter 1
• Configuration parameter 2

[Progress indicators during processing]

• Result statistic 1
• Result statistic 2

📋 Sample outputs:
  [Examples]

✅ Phase completed (X.Xs)
```

### Information Hierarchy

**Always show (default prints)**:
- Phase start/completion messages
- Basic counts (input → output)
- Total processing time
- Major errors

**Show only when verbose=True**:
- Configuration parameters
- Model details
- Detailed statistics
- Transformation counts
- Sample data
- Progress indicators for large datasets

### VerboseReporter Methods

Primary methods to use:
- `section_header(title)` - Major phase titles
- `step_start(name, emoji)` - Begin processing with timing
- `step_complete(message, emoji)` - End processing with elapsed time
- `stat_line(message)` - Statistics with bullet points
- `empty_line()` - Visual spacing
- `error(message)` - Error reporting
- `warning(message)` - Warning messages

### Statistics Formatting

- Use bullet points (•) for all statistics
- Format counts with context: "Category: X items (Y.Z%)"
- Group related statistics together
- Show meaningful comparisons (before → after)

### Progress Reporting

- Show progress for operations > 1000 items
- Update every 500 items
- Include operation name: "Processing responses... 500/2000 (25.0%)"
- Use `progress_line()` method

### Sample Data Display

- Limit to 3-5 examples
- Use random sampling for variety
- Format with clear visual hierarchy:
  ```
  📋 Sample corrections:
    "original text" → "corrected text"
  ```
- Truncate long text to ~80 characters

### Timing and Performance

- Track elapsed time for each phase
- Show processing rates for API operations
- Include bottleneck information when relevant
- Format: "Phase completed in X.XX seconds"

## Emoji Usage

Emojis serve as visual indicators for processing phases:
- 📝 Section headers
- 🔄 Processing/in-progress
- ✅ Completion
- 📋 Sample data
- 💡 Special operations (e.g., idea extraction)
- ❌ Errors
- ⚠️ Warnings

## Consistency Rules

1. **Capitalization**: 
   - Phase names in UPPERCASE
   - Regular messages in sentence case
   - No period at end of stat lines

2. **Spacing**:
   - Empty line before new sections
   - Empty line before sample displays
   - No empty line between related statistics

3. **Indentation**:
   - Use `indent=1` for nested information
   - Maintain visual hierarchy

4. **Error handling**:
   - Always include context in error messages
   - Truncate long text in errors
   - Use verbose_reporter.error() consistently

## Implementation Notes

- Initialize VerboseReporter with capture_logging=True for noisy libraries
- Track statistics throughout processing, display at end
- Collect examples during processing, show in summary
- Use ProcessingStats class for timing when available
- Check verbose_reporter.enabled before detailed operations

## Example Flow

```
verbose_reporter.section_header("PROCESSING PHASE")
verbose_reporter.step_start("Processing items", emoji="🔄")

# Show configuration (verbose only)
if verbose_reporter.enabled:
    verbose_reporter.stat_line("Configuration param: value")

# Main processing with progress
print(f"Processing {total} items...")
# ... processing logic ...

# Results summary
verbose_reporter.stat_line(f"Items processed: {count}")
verbose_reporter.stat_line(f"Changes made: {changes} ({changes/total*100:.1f}%)")

# Sample output (verbose only)
if verbose_reporter.enabled and examples:
    print("\n📋 Sample results:")
    # ... show examples ...

verbose_reporter.step_complete("Processing completed")
```

This guide captures the existing patterns without adding complexity. Follow these conventions to maintain consistency across all pipeline utilities.