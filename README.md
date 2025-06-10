# 🎭 Maestro MCP Server

> **Turn any LLM into a superintelligent AI assistant with advanced reasoning, web search, code execution, and multi-step workflows**

[![Docker Ready](https://img.shields.io/badge/Docker-Ready-green)](Dockerfile) [![MCP Protocol](https://img.shields.io/badge/MCP-2024--11--5-orange)](https://modelcontextprotocol.io)

## 🚀 What is Maestro?

Maestro is an **MCP server** that supercharges any LLM with powerful capabilities:

- 🔍 **Smart Web Search** - Research any topic with intelligent search
- 💻 **Code Execution** - Run Python, JavaScript, and other code safely  
- 🧠 **Advanced Reasoning** - Break down complex problems step-by-step
- 🔄 **Multi-Step Workflows** - Chain together multiple tasks automatically
- 🤝 **Human Collaboration** - Ask for help when needed

**Perfect for:** Research, data analysis, coding assistance, content creation, and complex problem-solving.

## 📋 Easy Setup for Cursor Users (Dad-Friendly!)

### Step 1: Install Python (One-Time Setup)

1. **Download Python**: Go to [python.org](https://python.org) and download Python 3.8 or newer
2. **Install Python**: 
   - ⚠️ **IMPORTANT**: During installation, check the box that says **"Add Python to PATH"**
   - This is crucial for the batch files to work!
3. **Verify Installation**: Open Command Prompt and type `python --version`

### Step 2: Get the Code

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/tanukimcp-maestro.git
   cd tanukimcp-maestro
   ```
   
   *Don't have git? Download the ZIP file from GitHub and extract it.*

### Step 3: Super Easy Installation

1. **Double-click `install.bat`** - This installs all dependencies automatically!
   - The batch file will check if Python is installed
   - It will install all required packages
   - It takes a few minutes the first time

### Step 4: Run the Server

**Option A: Double-Click Method (Easiest)**
- Double-click `start-server.bat`
- The server will start and show you the URL (usually http://localhost:8000)

**Option B: Command Line Method**
```bash
python run.py
```

### Step 5: Use with Cursor

1. **Configure Cursor**: Add this to your Cursor settings:
   ```json
   {
     "mcpServers": {
       "maestro": {
         "command": "python",
         "args": ["C:/path/to/tanukimcp-maestro/run.py", "--stdio"]
       }
     }
   }
   ```
   
   *Replace `C:/path/to/tanukimcp-maestro/` with the actual path where you downloaded this project*

2. **Restart Cursor** and start using Maestro tools!

### 🆘 Troubleshooting

**"Python is not recognized"**
- You forgot to check "Add Python to PATH" during installation
- Reinstall Python and make sure to check that box

**"pip is not available"**
- Usually fixed by reinstalling Python with pip included

**Dependencies fail to install**
- Check your internet connection
- Try running `install.bat` as Administrator

**Can't find the project folder**
- Right-click in the folder where you extracted/cloned the project
- Choose "Copy as path" and use that in your Cursor config

## ⚡ Quick Start (For Developers)

### Option 1: Use with Claude Desktop

1. **Clone and install:**
   ```bash
   git clone <this-repo-url>
   cd maestro-mcp
   pip install -r requirements.txt
   ```

2. **Add to Claude Desktop config:**
   ```json
   {
     "mcpServers": {
       "maestro": {
         "command": "python",
         "args": ["path/to/maestro-mcp/src/main.py", "--stdio"]
       }
     }
   }
   ```

3. **Restart Claude Desktop** and start using Maestro tools!

### Option 2: Run as HTTP Server

```bash
# Install and run
pip install -r requirements.txt
python run.py

# Test it works
curl http://localhost:8000/health
```

### Option 3: Docker

```bash
# Build and run with Docker
docker build -t maestro .
docker run -p 8000:8000 maestro
```

## 🛠️ What Can You Do?

### 🔍 Research & Analysis
Ask Claude: *"Research the latest developments in quantum computing and summarize the key breakthroughs"*

### 💻 Code & Execute
Ask Claude: *"Write a Python script to analyze this CSV data and create visualizations"*

### 🧠 Complex Problem Solving  
Ask Claude: *"Help me plan a marketing strategy for my startup, including market research and budget analysis"*

### 📊 Data Processing
Ask Claude: *"Download this dataset, clean it, and provide insights with charts"*

## 🎯 Available Tools

| Tool | What it does | Example Use |
|------|-------------|-------------|
| 🎭 **Orchestrate** | Plans and executes complex multi-step tasks | "Create a business plan with market research" |
| 🔍 **Web Search** | Intelligent web search with summarization | "What are the latest AI developments?" |
| 💻 **Code Execute** | Runs code safely in multiple languages | "Analyze this data with Python" |
| 🧠 **Intelligence Engine** | Advanced reasoning and analysis | "Solve this complex math problem" |
| ⚠️ **Error Handler** | Fixes problems and suggests solutions | Automatically handles errors |
| 🤝 **Collaboration** | Asks for human input when needed | "I need clarification on requirements" |

## 🔧 Configuration

Most users won't need to change anything, but you can customize:

- **Port**: Set `MAESTRO_PORT=8000`
- **Debug Mode**: Set `MAESTRO_MODE=development`  
- **Log Level**: Set `MAESTRO_LOG_LEVEL=INFO`

## 📦 Advanced Configuration

Environment variables you can set:
- `MAESTRO_PORT=8000` - Change the port
- `MAESTRO_MODE=development` - Enable debug mode  
- `MAESTRO_LOG_LEVEL=INFO` - Set logging level

## 🤝 Contributing

Want to add new capabilities? Check out:
- `src/maestro/tools.py` - Add new tools
- `src/engines/` - Add reasoning engines
- `docs/` - Documentation

## 📝 License

Non-commercial use is free. For commercial use, contact: tanukimcp@gmail.com

## 🆘 Support

- 🐛 **Issues**: [Open a GitHub issue](../../issues)
- 💬 **Questions**: [Start a discussion](../../discussions)  
- 📧 **Commercial**: tanukimcp@gmail.com

---

**Made with ❤️ for the AI community** 