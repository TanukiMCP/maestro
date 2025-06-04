# TanukiMCP Maestro Test Suite

Comprehensive testing suite for the production-ready MCP server that validates both Smithery.ai compatibility and real-world agentic IDE functionality.

## 🎯 What This Test Suite Validates

### ✅ Smithery.ai Compatibility
- **Instant Tool Discovery**: All tools register in <100ms for Smithery scanning
- **Static Tool Schemas**: No dynamic imports or side effects during tool listing
- **FastMCP Integration**: Proper MCP protocol compliance

### ✅ Real-World Functionality  
- **Natural Language Processing**: Tools work with natural language requests from agentic IDEs
- **Complex Workflows**: Multi-step orchestration and tool integration
- **Error Handling**: Graceful handling of edge cases and invalid inputs
- **Production Scenarios**: Code review, architecture guidance, debugging assistance

## 📁 Test Structure

```
tests/
├── __init__.py                 # Test suite initialization
├── test_tool_orchestrate.py    # Core orchestration functionality
├── test_tool_search.py         # Enhanced web search capabilities  
├── test_tool_iae.py           # Intelligence Amplification Engine
├── test_remaining_tools.py     # All other tools (collaboration, execution, etc.)
├── test_all_tools.py          # Integration testing and comprehensive workflows
├── run_all_tests.py           # Main test runner with detailed reporting
└── README.md                  # This documentation
```

## 🚀 Running Tests

### Quick Test (Individual Tools)
```bash
# Test individual tool
cd tests
python test_tool_orchestrate.py

# Test search functionality
python test_tool_search.py

# Test IAE capabilities
python test_tool_iae.py
```

### Comprehensive Test Suite
```bash
# Run all tests with detailed reporting
cd tests
python run_all_tests.py
```

### Using pytest (Optional)
```bash
# Install pytest if not available
pip install pytest pytest-asyncio

# Run with pytest
cd tests
pytest -v
```

## 📊 Test Categories

### 1. **Tool Registration Tests**
Validates that all 11 tools are properly registered:
- `maestro_orchestrate` - Core task orchestration
- `maestro_iae` - Intelligence Amplification Engine
- `get_available_engines` - Engine discovery
- `maestro_iae_discovery` - Agent discovery
- `maestro_tool_selection` - Intelligent tool routing
- `maestro_collaboration_response` - Multi-agent coordination
- `maestro_search` - Enhanced web search
- `maestro_scrape` - Web content extraction
- `maestro_execute` - Code/command execution
- `maestro_temporal_context` - Time-aware analysis
- `maestro_error_handler` - Error analysis and recovery

### 2. **Functional Tests**
Real-world scenarios that an agentic IDE would encounter:
- **Code Review**: "Help me review this Python function for issues"
- **Architecture Guidance**: "What database should I use for 1M users?"
- **Learning Assistance**: "Explain how OAuth 2.0 works"
- **Performance Analysis**: "Compare async/await vs callbacks in JavaScript"

### 3. **Integration Tests**
Complex workflows using multiple tools:
- Task orchestration → Engine discovery → Specialized analysis → Web research → Tool selection
- End-to-end validation of the full tool ecosystem

### 4. **Error Handling Tests**
Edge cases and boundary conditions:
- Empty inputs
- Invalid parameters  
- Network failures
- Graceful degradation

## 🎛️ Test Configuration

### Environment Variables
```bash
# Optional: Enable debug mode for detailed logging
export DEBUG_MODE=true

# Optional: Set test timeout
export TEST_TIMEOUT=30
```

### Test Customization
Modify test parameters in individual test files:
```python
# Adjust complexity levels
complexity_levels = ["minimal", "moderate", "high", "maximum"]

# Modify search parameters
max_results = 10
temporal_filter = "recent"

# Customize orchestration settings
quality_threshold = 0.8
enable_collaboration = True
```

## 📈 Production Readiness Criteria

Tests validate these production requirements:

### ✅ Performance
- Tool registration: <100ms (Smithery requirement)
- Response times: <30s for complex tasks
- Memory usage: Efficient lazy loading

### ✅ Reliability  
- 100% success rate on basic functionality
- Graceful error handling for all edge cases
- No crashes on invalid inputs

### ✅ Compatibility
- FastMCP protocol compliance
- Smithery.ai instant discovery
- Agentic IDE natural language support

### ✅ Functionality
- All 11 tools operational
- Real-world scenario handling
- Multi-tool integration workflows

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'server'**
```bash
# Ensure you're in the correct directory
cd tests
python test_all_tools.py
```

**Tool registration failures**
```bash
# Check server can be imported
python -c "import server; print('Server OK')"
```

**Async/await errors**
```bash
# Ensure Python 3.7+ is being used
python --version
```

### Debug Mode
Enable detailed logging:
```bash
export DEBUG_MODE=true
python run_all_tests.py
```

## 📋 Test Report Example

```
============================================================
  TanukiMCP Maestro - Production Readiness Test Suite
============================================================
🚀 Starting comprehensive testing at 2025-01-04 15:30:45

📋 Quick Import Validation
----------------------------------------
✅ Server module imports successfully
✅ FastMCP instance available  
✅ 11 tools registered

📋 Running test_tool_orchestrate
----------------------------------------
✅ maestro_orchestrate properly registered for Smithery scanning
✅ Simple orchestration completed: 1247 characters
✅ Complex orchestration completed: 2156 characters
✅ test_tool_orchestrate completed successfully in 3.45s

...

============================================================
  TEST EXECUTION SUMMARY
============================================================
📊 Total Tests Run: 5
✅ Successful: 5
❌ Failed: 0
⏱️  Total Duration: 15.67 seconds

============================================================
  PRODUCTION READINESS ASSESSMENT
============================================================
🎉 PRODUCTION READY!

✅ All tools properly registered for Smithery.ai scanning
✅ All tools functional with real-world scenarios
✅ Integration workflows working correctly
✅ Error handling working properly
✅ Natural language processing functional

🚀 TanukiMCP Maestro is ready for deployment!

📋 Production Metrics
----------------------------------------
📈 Tool Registration Time: <100ms (Smithery compatible)
📈 Average Response Time: 3.13s
📈 Success Rate: 100.0%
📈 Error Recovery: Graceful handling verified
```

## 🔗 Related Documentation

- [Server Implementation](../server.py)
- [MaestroTools](../src/maestro_tools.py)
- [Smithery Configuration](../smithery.yaml)
- [Production Validation](../validate_production.py)

## 🤝 Contributing

When adding new tools or features:

1. **Add tool registration test** in `test_all_tools.py`
2. **Create functional tests** with real-world scenarios
3. **Add integration test cases** if the tool interacts with others
4. **Update this README** with new test descriptions
5. **Run full test suite** to ensure compatibility

## 📄 License

Tests are covered under the same license as the main TanukiMCP Maestro project. 