# Agentified CAPTCHA Benchmark

A prototype of agent-based CAPTCHA solving benchmark using the A2A (Agent-to-Agent) protocol.

## Overview

This project demonstrates an agentified approach to benchmarking CAPTCHA-solving capabilities using AI agents. It leverages the [A2A protocol](https://a2a-protocol.org/latest/) for agent interoperability and provides a standardized evaluation framework.

It was forked from https://github.com/agentbeats/tutorial.

The benchmark consists of:
- **Green Agent (Judge)**: Orchestrates the CAPTCHA solving assessment and evaluates results
- **Purple Agent (Solver)**: Attempts to solve CAPTCHA challenges using vision-enabled LLMs

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Google Gemini API key (or other vision-enabled LLM API)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/gmsh/agentified-captcha-benchmark-demo.git
cd agentified-captcha-benchmark-demo
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Setup Environment Variables

Copy the sample environment file and add your API key:

```bash
cp sample.env .env
```

Edit `.env` and configure the following:

```bash
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=your-google-api-key-here
```

**To get a Google Gemini API key:**
1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key or use an existing one
3. Copy the key and paste it into your `.env` file

**Note:** Keep your `.env` file secure and never commit it to version control.

### 4. Run the CAPTCHA Scenario

```bash
uv run agentbeats-run scenarios/captcha/scenario.toml
```

Here is the [example output](example_output.txt):

This command will:
- Start the green agent (CAPTCHA judge) on `http://127.0.0.1:9009`
- Start the purple agent (CAPTCHA solver) on `http://127.0.0.1:9019`
- Run the assessment with 10 CAPTCHA samples from the dataset
- Display real-time progress and final results

**Optional flags:**
- `--show-logs`: Display agent outputs during the assessment
- `--serve-only`: Start agents without running the assessment (useful for debugging)

## Dataset

The benchmark uses a sample of the [Kaggle CAPTCHA v2 Images dataset](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images) located in `assets/kaggle-captcha-v2-images/`.

- **Location**: `assets/kaggle-captcha-v2-images/`
- **Format**: PNG images with CAPTCHA text
- **Labels**: Filenames contain the ground truth text

See `assets/kaggle-captcha-v2-images.md` for more details about the dataset.

## Configuration

You can customize the assessment by editing `scenarios/captcha/scenario.toml`:

```toml
[config]
num_samples = 10                              # Number of CAPTCHAs to test
dataset_path = "assets/kaggle-captcha-v2-images"  # Path to dataset
```

## Project Structure

```
agentify-captcha-benchmark/
├── src/agentbeats/              # Core framework
│   ├── green_executor.py        # Base green agent executor
│   ├── models.py                # Pydantic models for evaluation
│   ├── client.py                # A2A messaging utilities
│   ├── client_cli.py            # CLI client
│   └── run_scenario.py          # Scenario orchestration
│
├── scenarios/captcha/           # CAPTCHA benchmark scenario
│   ├── captcha_judge.py         # Green agent (orchestrator/evaluator)
│   ├── captcha_solver.py        # Purple agent (CAPTCHA solver)
│   ├── captcha_judge_common.py  # Shared models and utilities
│   └── scenario.toml            # Configuration file
│
├── assets/                      # Datasets and resources
│   └── kaggle-captcha-v2-images/  # CAPTCHA dataset
│
├── README.md                    # This file
├── README.agentbeats.md         # Original Agentbeats tutorial
├── sample.env                   # Environment variable template
└── pyproject.toml               # Project configuration
```

## How It Works

1. **Assessment Request**: The client sends an assessment request to the green agent with:
   - The solver agent's endpoint
   - Configuration (number of samples, dataset path)

2. **CAPTCHA Distribution**: The green agent:
   - Loads CAPTCHA images from the dataset
   - Sends each image to the purple agent for solving
   - Tracks responses in real-time

3. **Solving**: The purple agent:
   - Receives CAPTCHA images
   - Uses a vision-enabled LLM (e.g., Google Gemini) to analyze the image
   - Returns the predicted text

4. **Evaluation**: The green agent:
   - Compares predictions against ground truth labels
   - Calculates accuracy metrics
   - Generates a detailed evaluation report

5. **Results**: The assessment produces:
   - Real-time task updates showing progress
   - Final accuracy score
   - Detailed per-sample results as artifacts

## Development

### Running Agents Manually

For debugging, you can start agents manually in separate terminals:

```bash
# Terminal 1: Start green agent
python scenarios/captcha/captcha_judge.py --host 127.0.0.1 --port 9009

# Terminal 2: Start purple agent
python scenarios/captcha/captcha_solver.py --host 127.0.0.1 --port 9019

# Terminal 3: Run client
python -m agentbeats.client_cli scenarios/captcha/scenario.toml
```

### Modifying the Solver

The solver agent (`scenarios/captcha/captcha_solver.py`) can be modified to:
- Use different LLM providers (OpenAI, Anthropic, etc.)
- Implement custom preprocessing
- Add retry logic or fallback strategies
- Enhance prompting techniques

### Adding New Scenarios

See `README.agentbeats.md` for detailed instructions on creating new assessment scenarios using the Agentbeats framework.

## Troubleshooting

**Issue**: `ImportError` or missing dependencies
- **Solution**: Run `uv sync` to install all dependencies

**Issue**: API key errors
- **Solution**: Ensure your `.env` file has a valid `GOOGLE_API_KEY` set

**Issue**: Dataset not found
- **Solution**: Verify that `assets/kaggle-captcha-v2-images/` exists and contains CAPTCHA images

**Issue**: Connection errors
- **Solution**: Ensure ports 9009 and 9019 are not in use by other applications

## Cost Considerations

- Each CAPTCHA image requires one vision API call to the LLM
- Default configuration tests 10 samples
- Consider using cheaper/free models during development (e.g., Gemini Flash)

## Next Steps

- **Expand the dataset**: Add more CAPTCHA samples for robust evaluation
- **Advanced CAPTCHAs**: Include more challenging variants
- **Deploy to platform**: Publish agents to [agentbeats.org](https://agentbeats.org) for public benchmarking

## References

- [A2A Protocol Documentation](https://a2a-protocol.org/latest/)
- [Agentbeats Platform](https://agentbeats.org)
- [Original Agentbeats Tutorial](README.agentbeats.md)

## License

MIT License - see LICENSE file for details
