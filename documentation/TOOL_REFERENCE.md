# Tool Reference

Complete reference for all MAESTRO tools with detailed parameters, examples, and use cases.

## 🎭 maestro_orchestrate

**Purpose**: Intelligent workflow orchestration for complex tasks using Mixture-of-Agents (MoA)

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task_description` | string | ✅ | - | The complex task to orchestrate |
| `context` | object | ❌ | `{}` | Additional context for the task |
| `success_criteria` | object | ❌ | `{}` | Success criteria for the task |
| `complexity_level` | string | ❌ | `"moderate"` | Complexity level: `simple`, `moderate`, `complex`, `expert` |

### Example Usage

```python
# Basic orchestration
result = await tools.handle_tool_call("maestro_orchestrate", {
    "task_description": "Analyze the competitive landscape for AI startups in 2024",
    "complexity_level": "complex"
})

# With context and success criteria
result = await tools.handle_tool_call("maestro_orchestrate", {
    "task_description": "Create a comprehensive marketing strategy",
    "context": {
        "industry": "SaaS",
        "target_market": "SMB",
        "budget": "$50k"
    },
    "success_criteria": {
        "deliverables": ["strategy document", "implementation plan"],
        "timeline": "2 weeks",
        "quality_threshold": 0.9
    },
    "complexity_level": "expert"
})
```

### Response Format

```json
{
  "workflow_id": "wf_12345",
  "status": "completed",
  "execution_time": 45.2,
  "quality_score": 0.94,
  "operator_profiles_used": ["specialist", "analyst", "critic"],
  "deliverables": {
    "primary_output": "...",
    "supporting_analysis": "...",
    "quality_verification": "..."
  },
  "metadata": {
    "steps_executed": 8,
    "verification_passes": 3,
    "early_stopping_triggered": false
  }
}
```

---

## ⚡ maestro_iae

**Purpose**: Integrated Analysis Engine for computational tasks with auto-engine selection

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `analysis_request` | string | ✅ | - | The analysis or computation request |
| `engine_type` | string | ❌ | `"auto"` | Engine type: `statistical`, `mathematical`, `quantum`, `auto` |
| `precision_level` | string | ❌ | `"standard"` | Precision: `standard`, `high`, `ultra` |
| `computational_context` | object | ❌ | `{}` | Additional computational context |

### Supported Computations

#### Mathematical Engine
- Matrix operations (eigenvalues, determinants, inversions)
- Calculus (derivatives, integrals, limits)
- Linear algebra and vector operations
- Statistical analysis and probability
- Optimization problems

#### Quantum Physics Engine
- Quantum state analysis
- Entanglement calculations
- Bell inequality tests
- Quantum circuit simulation
- Quantum information theory

#### Intelligence Amplification Engine
- Knowledge network analysis
- Concept clustering
- Cognitive load optimization
- Information synthesis
- Pattern recognition

### Example Usage

```python
# Mathematical computation
result = await tools.handle_tool_call("maestro_iae", {
    "analysis_request": "Calculate the eigenvalues and eigenvectors of matrix [[2,1],[1,2]]",
    "engine_type": "mathematical",
    "precision_level": "high"
})

# Quantum physics calculation
result = await tools.handle_tool_call("maestro_iae", {
    "analysis_request": "Calculate entanglement entropy for Bell state |00⟩ + |11⟩",
    "engine_type": "quantum",
    "precision_level": "ultra"
})

# Auto-selection with context
result = await tools.handle_tool_call("maestro_iae", {
    "analysis_request": "Analyze the correlation between variables X and Y in this dataset",
    "engine_type": "auto",
    "computational_context": {
        "data_type": "time_series",
        "sample_size": 1000,
        "variables": ["temperature", "sales"]
    }
})
```

---

## 🔎 maestro_search

**Purpose**: Enhanced search capabilities across multiple sources with LLM analysis

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Search query |
| `max_results` | integer | ❌ | `10` | Maximum number of results |
| `search_engine` | string | ❌ | `"duckduckgo"` | Engine: `duckduckgo`, `google`, `bing` |
| `temporal_filter` | string | ❌ | `"any"` | Time filter: `any`, `recent`, `week`, `month`, `year` |
| `result_format` | string | ❌ | `"structured"` | Format: `structured`, `summary`, `detailed` |

### Example Usage

```python
# Basic search
result = await tools.handle_tool_call("maestro_search", {
    "query": "latest AI research papers 2024",
    "max_results": 15
})

# Temporal search with specific engine
result = await tools.handle_tool_call("maestro_search", {
    "query": "quantum computing breakthroughs",
    "search_engine": "google",
    "temporal_filter": "month",
    "result_format": "detailed"
})

# Research-focused search
result = await tools.handle_tool_call("maestro_search", {
    "query": "machine learning interpretability methods",
    "max_results": 20,
    "temporal_filter": "year",
    "result_format": "summary"
})
```

### Response Format

```json
{
  "query": "AI research papers 2024",
  "enhanced_query": "artificial intelligence research publications 2024 peer-reviewed",
  "total_results": 15,
  "search_engine": "duckduckgo",
  "results": [
    {
      "title": "Advances in Large Language Models",
      "url": "https://example.com/paper1",
      "snippet": "Recent developments in...",
      "relevance_score": 0.95,
      "domain": "arxiv.org",
      "llm_analysis": "This paper discusses breakthrough techniques..."
    }
  ],
  "metadata": {
    "search_time": 2.3,
    "llm_enhanced": true,
    "temporal_filter_applied": "recent"
  }
}
```

---

## 📑 maestro_scrape

**Purpose**: Web scraping functionality with intelligent content extraction

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | ✅ | - | URL to scrape |
| `output_format` | string | ❌ | `"markdown"` | Format: `markdown`, `text`, `html`, `json` |
| `selectors` | array | ❌ | `[]` | CSS selectors for specific elements |
| `wait_time` | number | ❌ | `3` | Time to wait for page load (seconds) |
| `extract_links` | boolean | ❌ | `false` | Whether to extract links |

### Example Usage

```python
# Basic scraping
result = await tools.handle_tool_call("maestro_scrape", {
    "url": "https://example.com/article",
    "output_format": "markdown"
})

# Targeted scraping with selectors
result = await tools.handle_tool_call("maestro_scrape", {
    "url": "https://news.site.com/tech",
    "selectors": [".article-content", ".headline", ".author"],
    "extract_links": true,
    "wait_time": 5
})

# Structured data extraction
result = await tools.handle_tool_call("maestro_scrape", {
    "url": "https://research.site.com/papers",
    "output_format": "json",
    "extract_links": true
})
```

---

## ⚙️ maestro_execute

**Purpose**: Execute computational tasks with enhanced error handling

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `code` | string | ✅ | - | Code to execute |
| `language` | string | ❌ | `"python"` | Language: `python`, `javascript`, `bash` |
| `timeout` | number | ❌ | `30` | Execution timeout in seconds |
| `capture_output` | boolean | ❌ | `true` | Whether to capture stdout/stderr |
| `working_directory` | string | ❌ | `null` | Working directory for execution |
| `environment_vars` | object | ❌ | `{}` | Environment variables to set |

### Example Usage

```python
# Python execution
result = await tools.handle_tool_call("maestro_execute", {
    "code": """
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Calculate statistics
mean_y = np.mean(y)
std_y = np.std(y)

print(f"Mean: {mean_y:.4f}")
print(f"Std: {std_y:.4f}")
""",
    "language": "python",
    "timeout": 15
})

# JavaScript execution
result = await tools.handle_tool_call("maestro_execute", {
    "code": """
const data = [1, 2, 3, 4, 5];
const sum = data.reduce((a, b) => a + b, 0);
const avg = sum / data.length;
console.log(`Average: ${avg}`);
""",
    "language": "javascript"
})

# With environment variables
result = await tools.handle_tool_call("maestro_execute", {
    "code": "echo $API_KEY",
    "language": "bash",
    "environment_vars": {
        "API_KEY": "test-key-123"
    }
})
```

---

## 🚨 maestro_error_handler

**Purpose**: Advanced error handling and recovery suggestions

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `error_message` | string | ✅ | - | The error message to analyze |
| `error_context` | object | ❌ | `{}` | Context where the error occurred |
| `recovery_suggestions` | boolean | ❌ | `true` | Whether to provide recovery suggestions |

### Example Usage

```python
# Basic error analysis
result = await tools.handle_tool_call("maestro_error_handler", {
    "error_message": "ModuleNotFoundError: No module named 'pandas'",
    "error_context": {
        "operation": "data_analysis",
        "environment": "python3.9",
        "previous_steps": ["import numpy", "import matplotlib"]
    }
})

# Complex error with context
result = await tools.handle_tool_call("maestro_error_handler", {
    "error_message": "Connection timeout after 30 seconds",
    "error_context": {
        "operation": "web_scraping",
        "url": "https://slow-site.com",
        "network_conditions": "limited_bandwidth",
        "retry_count": 3
    },
    "recovery_suggestions": true
})
```

---

## 📅 maestro_temporal_context

**Purpose**: Temporal context awareness and time-based reasoning

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `temporal_query` | string | ✅ | - | Query requiring temporal reasoning |
| `time_range` | object | ❌ | `{}` | Time range with `start` and `end` |
| `temporal_precision` | string | ❌ | `"day"` | Precision: `year`, `month`, `day`, `hour`, `minute` |

### Example Usage

```python
# Time-aware analysis
result = await tools.handle_tool_call("maestro_temporal_context", {
    "temporal_query": "What were the major AI developments in Q1 2024?",
    "time_range": {
        "start": "2024-01-01",
        "end": "2024-03-31"
    },
    "temporal_precision": "month"
})
```

---

## 🔧 get_available_engines

**Purpose**: Get list of available computational engines and their capabilities

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `engine_type` | string | ❌ | `"all"` | Filter: `all`, `statistical`, `mathematical`, `quantum`, `enhanced` |
| `include_capabilities` | boolean | ❌ | `true` | Include detailed capabilities |

### Example Usage

```python
# List all engines
result = await tools.handle_tool_call("get_available_engines", {
    "engine_type": "all",
    "include_capabilities": true
})

# List only mathematical engines
result = await tools.handle_tool_call("get_available_engines", {
    "engine_type": "mathematical"
})
```

---

## 🎯 Best Practices

### Tool Selection Guidelines

1. **Complex Tasks**: Use `maestro_orchestrate` for multi-step workflows
2. **Calculations**: Use `maestro_iae` for mathematical/scientific computations
3. **Research**: Use `maestro_search` for information gathering
4. **Data Extraction**: Use `maestro_scrape` for web content
5. **Code Tasks**: Use `maestro_execute` for script execution
6. **Error Recovery**: Use `maestro_error_handler` for troubleshooting

### Performance Optimization

- Set appropriate `timeout` values for long-running operations
- Use `precision_level` wisely (higher precision = longer execution time)
- Leverage `temporal_filter` in searches for relevant results
- Use specific `selectors` in scraping for faster extraction

### Error Handling

- Always include error context for better analysis
- Use `maestro_error_handler` for complex error scenarios
- Set reasonable timeouts to prevent hanging operations
- Monitor execution logs for performance insights

---

## 📊 Tool Compatibility Matrix

| Tool | Python | JavaScript | Bash | Web | MCP | HTTP |
|------|--------|------------|------|-----|-----|------|
| `maestro_orchestrate` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `maestro_iae` | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| `maestro_search` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `maestro_scrape` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `maestro_execute` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `maestro_error_handler` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

For more examples and advanced usage patterns, see [Advanced Workflows](./ADVANCED_WORKFLOWS.md). 