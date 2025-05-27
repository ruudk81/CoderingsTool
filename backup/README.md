# CoderingsTool

A text analysis pipeline for processing survey responses from SPSS files. The tool performs text preprocessing, quality filtering, embedding generation, clustering, and thematic labeling of open-ended survey responses.

## Features

- Load survey data from SPSS (.sav) files
- Models for structured data in models.py
- Congfigurations is config.py
- Prompts in prompts.py
- Cache manager for data storage and persistence
- Modular approach:
  1. Text preprocessing including normalization and spell checking
  2. Quality filtering to identify and filter out meaningless responses
  3. Segment responses and generate descriptive codes and code descriptions for each segment
  4. Embed descriptive codes and code descriptions using OpenAI's embedding models
  5. Initial, automatic clustering of micro clusters with HDBSCAN based on UMAP reduced dimensions of embeddings
  6. Merge clusters that cannot semantically be positively differentiated in light of the research question
  7. Hierarchical labelling - node levels 1, 2 and 3. For meta, macro and micro clusters with labels called themes, topics and keywords
  8. Visualization of results in dendrogram (for each node), word cloud (for each theme) and an overall summary
- Orchestration in pipeline.py
- Streamlit app based on pipeline

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CoderingsTool.git
   cd CoderingsTool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

## Usage

### Running the Pipeline

The main pipeline can be run from the command line:

```bash
cd src
python pipeline.py
```

### Configuration

Edit `src/config.py` to modify:

- OpenAI model settings
- File size limits
- Language settings
- Batch processing parameters
- etc.

## Project Structure

```
CoderingsTool/
├── data/                   # Input data files
├── src/
│   ├── app.py             # Web application (if implemented)
│   ├── config.py          # Configuration settings
│   ├── models.py          # Data models
│   ├── pipeline.py        # Main processing pipeline
│   ├── prompts.py         # LLM prompts
│   ├── ui_text.py         # UI text constants
│   └── modules/
│       └── utils/         # Utility modules
└── requirements.txt       # Python dependencies
```

## Requirements

- Python 3.8+
- OpenAI API key for embeddings and LLM features
- Hunspell for spell checking (Dutch and English dictionaries)

See `requirements.txt` for complete dependency list.

## License

[Specify your license here]

## Contributing

[Add contribution guidelines here]