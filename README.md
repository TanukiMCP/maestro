# TanukiMCP Maestro

**Meta-Agent Ensemble for Systematic Task Reasoning and Orchestration**

Transform any language model into a superintelligent system through advanced orchestration, quality verification, and automated workflow management.

---

## What is MAESTRO?

MAESTRO is an open protocol and software platform that amplifies the capabilities of any AI model—making even free or small models perform at superintelligent levels. It does this by orchestrating multiple specialist agents, applying rigorous quality control, and automating complex workflows. MAESTRO is compatible with the Model Context Protocol (MCP), supports HTTP and containerized deployment, and is designed for both developers and non-technical users.

**In short:**
- For regular users: MAESTRO lets you get expert-level results from any AI, reliably and affordably.
- For developers: MAESTRO provides a modular, extensible orchestration and validation framework for building robust AI-powered systems.

---

## Key Features and Benefits

### For Everyone
- **Superintelligent Results**: Achieve GPT-4-level quality from any model
- **Reliability**: Automated verification and error recovery at every step
- **Cost Efficiency**: Reduce API/model costs by up to 90%
- **Universal Compatibility**: Works with any MCP-compatible client, HTTP, or Docker

### For Developers
- **Drop-in Integration**: Standard MCP server, HTTP API, and container support
- **Extensible Architecture**: Add custom engines, tools, and agent profiles
- **Production Ready**: Built-in logging, monitoring, and error handling
- **Collaborative Orchestration**: Human-in-the-loop fallback and validation

---

## How It Works: Architecture Overview

MAESTRO uses a layered, modular architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    MAESTRO Protocol                        │
├─────────────────────────────────────────────────────────────┤
│  Orchestration Layer                                       │
│  ├── Operator Profiles (Specialist, Analyst, Critic, etc.) │
│  ├── Workflow Planning & Execution                         │
│  └── Quality Control & Verification                        │
├─────────────────────────────────────────────────────────────┤
│  Intelligence Amplification Engine (IAE)                   │
│  ├── Mathematics, Quantum Physics, Data Analysis, Language │
│  └── Code Quality Engines                                  │
├─────────────────────────────────────────────────────────────┤
│  Enhanced Tool Ecosystem                                   │
│  ├── Web Intelligence (Search, Scrape, Analysis)           │
│  ├── Code Execution (Multi-language, Validation)           │
│  ├── Error Handling (Adaptive, Recovery)                   │
│  └── Temporal Context (Time-aware reasoning)               │
├─────────────────────────────────────────────────────────────┤
│  Transport & Integration                                   │
│  ├── MCP Protocol, HTTP/SSE, Docker, Smithery              │
└─────────────────────────────────────────────────────────────┘
```

**Collaborative Orchestration:**
- MAESTRO detects ambiguity, missing context, or requirement conflicts and can request user input to clarify or refine tasks.
- Multi-stage validation ensures quality at every step, with automated and manual checks as needed.

---

## Quick Start

### 1. Docker (Recommended)
```bash
docker run -p 8000:8000 tanukimcp/maestro:latest
```

### 2. Python Package
```bash
pip install tanukimcp-maestro
python -m src.main
```

### 3. From Source
```bash
git clone https://github.com/tanukimcp/maestro.git
cd maestro
pip install -e .
python -m src.main
```

### 4. Smithery Cloud Deployment
```bash
smithery deploy tanukimcp/maestro
```

**Verify Installation:**
```bash
curl http://localhost:8000/health
```

---

## Usage Examples

### Orchestrate a Complex Task (Python)
```python
import asyncio
from src.maestro_tools import MaestroTools

async def main():
    tools = MaestroTools()
    result = await tools.handle_tool_call("maestro_orchestrate", {
        "task_description": "Analyze the current state of AI research and provide insights",
        "complexity_level": "moderate"
    })
    print(result[0].text)

asyncio.run(main())
```

### Mathematical Analysis
```python
result = await tools.handle_tool_call("maestro_iae", {
    "analysis_request": "Calculate the eigenvalues of a 3x3 matrix [[1,2,3],[4,5,6],[7,8,9]]",
    "engine_type": "mathematical",
    "precision_level": "high"
})
```

### Enhanced Web Search
```python
result = await tools.handle_tool_call("maestro_search", {
    "query": "latest developments in quantum computing 2024",
    "max_results": 5,
    "temporal_filter": "recent"
})
```

### HTTP API Example
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "maestro_orchestrate",
      "arguments": {
        "task_description": "Your task here",
        "quality_threshold": 0.8
      }
    }
  }'
```

---

## Tool Reference (Core Tools)

| Tool                   | Purpose                        | Key Features                        |
|------------------------|--------------------------------|--------------------------------------|
| `maestro_orchestrate`  | Workflow orchestration         | Multi-agent planning, validation     |
| `maestro_iae`          | Computational analysis         | Math, quantum, data, language, code  |
| `maestro_search`       | Enhanced web search            | LLM analysis, temporal filtering     |
| `maestro_scrape`       | Web scraping                   | Content extraction, structured data  |
| `maestro_execute`      | Code/workflow execution        | Secure, multi-language, validation   |
| `maestro_error_handler`| Adaptive error handling        | Recovery, context analysis           |
| `maestro_temporal_context` | Time-aware reasoning        | Temporal analysis, context currency  |
| `maestro_iae_discovery`| Engine discovery               | Optimal computation selection        |
| `maestro_tool_selection`| Tool selection                | Intelligent tool recommendation      |
| `get_available_engines`| Engine information             | List engines and capabilities        |

For full tool documentation, see [Tool Reference](./documentation/TOOL_REFERENCE.md).

---

## Configuration and Tips

- **Environment Variables:**
  - `MAESTRO_PORT`, `MAESTRO_HOST`, `MAESTRO_LOG_LEVEL`, `MAESTRO_ENGINE_TIMEOUT`, `MAESTRO_MAX_ITERATIONS`, `MAESTRO_QUALITY_THRESHOLD`
  - API keys: `OPENAI_API_KEY`, `SERPAPI_KEY` (optional, for enhanced features)
- **Configuration File:**
  - Create `maestro.yaml` for advanced options (see example in docs)
- **Tips:**
  - Use Docker for easiest setup and isolation
  - For production, set higher quality thresholds and enable logging
  - Use the collaborative fallback for ambiguous or high-stakes tasks
  - Explore the [Quick Start Guide](./documentation/QUICK_START.md) and [Architecture Overview](./documentation/ARCHITECTURE.md)

---

## Development and Contribution

- **Development Setup:**
  - `pip install -e ".[dev]"`
  - Install pre-commit hooks: `pre-commit install`
- **Testing:**
  - Run all tests: `pytest`
  - Coverage: `pytest --cov=src tests/`
- **Code Quality:**
  - Format: `black src/ tests/`
  - Lint: `flake8 src/ tests/`
  - Type check: `mypy src/`
- **Contributing:**
  1. Fork the repo and create a feature branch
  2. Add tests and documentation for new features
  3. Ensure all tests pass and code is formatted
  4. Open a Pull Request

---

## License and Commercial Use

**Non-Commercial License** – MAESTRO is free for non-commercial use only. Commercial use requires explicit written permission.

- See [LICENSE](./LICENSE) for full terms
- For commercial licensing, contact: tanukimcp@gmail.com
- Subject: "MAESTRO Protocol Commercial License Request"

**Commercial use includes:** business operations, commercial services, revenue generation, corporate/organizational use, and any use that could reasonably generate financial benefit.

---

## Community and Support

- **GitHub:** https://github.com/tanukimcp/maestro
- **Issues:** https://github.com/tanukimcp/maestro/issues
- **Discussions:** https://github.com/tanukimcp/maestro/discussions
- **Documentation:** [Full Documentation](./documentation/README.md)

**Support:**
- Check the docs and FAQ first
- Report bugs and request features via GitHub Issues
- For questions, use GitHub Discussions or community forums

---

**TanukiMCP Maestro: Democratizing AI through Intelligence Amplification**
