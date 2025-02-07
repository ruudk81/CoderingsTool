# CoderingsTool

A text analysis pipeline for processing survey responses from SPSS files. The tool performs text preprocessing, quality filtering, embedding generation, clustering, and thematic labeling of open-ended survey responses.

## Features

- Load survey data from SPSS (.sav) files
- Text preprocessing including normalization and spell checking
- Quality filtering to identify and filter out meaningless responses
- Segment descriptions generation for response components
- Text embeddings using OpenAI's embedding models
- Hierarchical clustering (meta, meso, micro levels)
- Thematic labeling of clusters
- CSV export/import at each processing stage

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

### Data Processing Steps

1. **Data Import**: Loads survey responses from SPSS files
2. **Preprocessing**: Normalizes text, performs spell checking, and finalizes responses
3. **Segmentation**: Describes segments and filters by quality
4. **Embedding**: Generates embeddings for codes and descriptions
5. **Clustering**: Creates hierarchical clusters of similar responses
6. **Labeling**: Assigns thematic labels to clusters

### Configuration

Edit `src/config.py` to modify:
- OpenAI model settings
- File size limits
- Language settings
- Batch processing parameters

## Project Structure

```
CoderingsTool/
├── data/                    # Input data files
├── src/
│   ├── app.py              # Web application (if implemented)
│   ├── config.py           # Configuration settings
│   ├── models.py           # Data models
│   ├── pipeline.py         # Main processing pipeline
│   ├── prompts.py          # LLM prompts
│   ├── ui_text.py          # UI text constants
│   └── modules/
│       ├── assigning.py    # Assignment logic
│       ├── clustering.py   # Clustering algorithms
│       ├── labelling.py    # Labeling functions
│       ├── preprocessing.py # Preprocessing utilities
│       ├── visualizing.py  # Visualization tools
│       └── utils/          # Utility modules
└── requirements.txt        # Python dependencies
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