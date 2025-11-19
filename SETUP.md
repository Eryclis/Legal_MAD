# Setup and Usage Guide

## Installation

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup environment variables:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Get Groq API Key

1. Go to https://console.groq.com/
2. Sign up/login
3. Create API key
4. Copy key to .env file

## Test Installation

```bash
# Test data loader
python -m src.utils.data_loader

# Test API client (requires GROQ_API_KEY in .env)
python -m src.utils.api_client
```

## Run MAD Experiments

```bash
# Run on 5 questions (test)
python -m src.experiments.run_mad

# Run on more questions (edit script to change sample_size)
# Edit src/experiments/run_mad.py, change sample_size parameter
```

## Results

Results are saved to `results/` directory in JSON format.

## Troubleshooting

**ModuleNotFoundError:**
- Make sure you're in the project root directory
- Activate virtual environment: `source venv/bin/activate`

**Groq API errors:**
- Check API key in .env file
- Check rate limits (14,400 requests/day on free tier)

**Dataset loading errors:**
- Requires internet connection to download from HuggingFace
- First run will download dataset (~100MB)
