# Installation Guide

Complete installation instructions for MAESTRO Protocol across different platforms and deployment scenarios.

## 🚨 License Compliance Notice

**MAESTRO Protocol is licensed for NON-COMMERCIAL use only.**

By installing and using MAESTRO, you agree to comply with the Non-Commercial License terms. Commercial use requires explicit written permission from TanukiMCP.

📋 **[Commercial License Information](../COMMERCIAL_LICENSE_INFO.md)**  
📧 **Commercial Licensing**: tanukimcp@gmail.com

---

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 2GB RAM (4GB recommended)
- **Storage**: 500MB free space
- **Network**: Internet connection for web tools

### Optional Dependencies
- **Docker**: For containerized deployment
- **Node.js**: For JavaScript code execution
- **Git**: For source installation

## 🚀 Installation Methods

### Method 1: Docker (Recommended)

#### Quick Start
```bash
# Pull and run the latest image
docker run -p 8000:8000 tanukimcp/maestro:latest
```

#### Custom Configuration
```bash
# Run with custom environment variables
docker run -p 8000:8000 \
  -e MAESTRO_LOG_LEVEL=DEBUG \
  -e MAESTRO_ENGINE_TIMEOUT=60 \
  tanukimcp/maestro:latest
```

#### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  maestro:
    image: tanukimcp/maestro:latest
    ports:
      - "8000:8000"
    environment:
      - MAESTRO_LOG_LEVEL=INFO
      - MAESTRO_PORT=8000
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

### Method 2: Python Package

#### From PyPI (Coming Soon)
```bash
# Install from PyPI
pip install tanukimcp-maestro

# Verify installation
python -c "import src; print('MAESTRO installed successfully')"
```

#### From Source
```bash
# Clone repository
git clone https://github.com/tanukimcp/maestro.git
cd maestro

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"
```

### Method 3: Smithery Deployment

#### Prerequisites
```bash
# Install Smithery CLI (if not already installed)
npm install -g @smithery/cli
```

#### Deploy to Smithery
```bash
# Deploy from local directory
smithery deploy

# Or deploy directly from GitHub
smithery deploy tanukimcp/maestro
```

#### Verify Smithery Deployment
```bash
# Check deployment status
smithery status maestro

# View logs
smithery logs maestro
```

## 🔧 Platform-Specific Installation

### Windows

#### Using PowerShell
```powershell
# Install Python (if not installed)
winget install Python.Python.3.11

# Clone and install MAESTRO
git clone https://github.com/tanukimcp/maestro.git
cd maestro
pip install -r requirements.txt
python -m src.main
```

#### Using Windows Subsystem for Linux (WSL)
```bash
# In WSL terminal
sudo apt update
sudo apt install python3 python3-pip git
git clone https://github.com/tanukimcp/maestro.git
cd maestro
pip3 install -r requirements.txt
python3 -m src.main
```

### macOS

#### Using Homebrew
```bash
# Install dependencies
brew install python@3.11 git

# Clone and install MAESTRO
git clone https://github.com/tanukimcp/maestro.git
cd maestro
pip3 install -r requirements.txt
python3 -m src.main
```

### Linux (Ubuntu/Debian)

#### System Installation
```bash
# Update system
sudo apt update && sudo apt upgrade

# Install dependencies
sudo apt install python3 python3-pip python3-venv git

# Clone and install MAESTRO
git clone https://github.com/tanukimcp/maestro.git
cd maestro

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install MAESTRO
pip install -r requirements.txt
python -m src.main
```

### Linux (CentOS/RHEL/Fedora)

#### System Installation
```bash
# Update system
sudo dnf update

# Install dependencies
sudo dnf install python3 python3-pip git

# Clone and install MAESTRO
git clone https://github.com/tanukimcp/maestro.git
cd maestro
pip3 install -r requirements.txt
python3 -m src.main
```

## 🌐 Deployment Configurations

### Development Setup
```bash
# Clone repository
git clone https://github.com/tanukimcp/maestro.git
cd maestro

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Run in development mode
python -m src.main
```

### Production Setup
```bash
# Create production directory
mkdir /opt/maestro
cd /opt/maestro

# Clone and install
git clone https://github.com/tanukimcp/maestro.git .
pip install -r requirements.txt

# Create systemd service (Linux)
sudo tee /etc/systemd/system/maestro.service > /dev/null <<EOF
[Unit]
Description=MAESTRO Protocol Server
After=network.target

[Service]
Type=simple
User=maestro
WorkingDirectory=/opt/maestro
ExecStart=/usr/bin/python3 -m src.main
Restart=always
RestartSec=10
Environment=MAESTRO_PORT=8000
Environment=MAESTRO_LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable maestro
sudo systemctl start maestro
```

## ✅ Verification

### Health Check
```bash
# Check if MAESTRO is running
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "version": "1.0.0",
  "tools_available": 9,
  "engines_loaded": 8
}
```

### Test Tool Call
```bash
# Test orchestration tool
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "maestro_orchestrate",
      "arguments": {
        "task_description": "Test MAESTRO installation",
        "complexity_level": "simple"
      }
    }
  }'
```

### Python Verification
```python
# test_installation.py
import asyncio
from src.maestro_tools import MaestroTools

async def test_maestro():
    tools = MaestroTools()
    
    # Test basic functionality
    result = await tools.handle_tool_call("maestro_iae", {
        "analysis_request": "Calculate 2 + 2",
        "engine_type": "mathematical"
    })
    
    print("✅ MAESTRO installation verified!")
    print(f"Result: {result[0].text[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_maestro())
```

## 🔧 Configuration

### Environment Variables
```bash
# Server configuration
export MAESTRO_PORT=8000
export MAESTRO_HOST=0.0.0.0
export MAESTRO_LOG_LEVEL=INFO

# Engine configuration
export MAESTRO_ENGINE_TIMEOUT=30
export MAESTRO_MAX_OPERATORS=5

# Web tools configuration
export MAESTRO_SEARCH_TIMEOUT=10
export MAESTRO_SCRAPE_TIMEOUT=15
```

### Configuration File
Create `maestro.yaml`:
```yaml
server:
  port: 8000
  host: "0.0.0.0"
  log_level: "INFO"

engines:
  timeout: 30
  mathematics:
    precision_levels: ["standard", "high", "ultra"]
  quantum:
    max_qubits: 20

tools:
  orchestrate:
    max_operators: 5
    quality_threshold: 0.8
  search:
    timeout: 10
    max_results: 20
```

## 🆘 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using port 8000
netstat -tulpn | grep 8000  # Linux
netstat -an | findstr 8000  # Windows

# Use different port
export MAESTRO_PORT=8001
python -m src.main
```

#### Missing Dependencies
```bash
# Install missing packages
pip install --upgrade pip
pip install -r requirements.txt

# For web tools
pip install playwright beautifulsoup4
playwright install
```

#### Permission Errors
```bash
# Linux/macOS: Use virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Windows: Run as administrator or use virtual environment
```

#### Memory Issues
```bash
# Increase available memory or use lighter configuration
export MAESTRO_ENGINE_TIMEOUT=15
export MAESTRO_MAX_OPERATORS=3
```

### Getting Help

- **Documentation**: [Complete Documentation](./README.md)
- **GitHub Issues**: [Report Issues](https://github.com/tanukimcp/maestro/issues)
- **Community**: [Discussions](https://github.com/tanukimcp/maestro/discussions)

---

## ⚖️ License Compliance Reminder

By completing this installation, you acknowledge that:

1. **Non-Commercial Use Only**: You will use MAESTRO only for non-commercial purposes
2. **Commercial License Required**: Any commercial use requires written permission from TanukiMCP
3. **Compliance Monitoring**: Usage may be monitored for license compliance
4. **Contact for Commercial Use**: tanukimcp@gmail.com

📋 **[Full License Terms](../LICENSE)** | 📋 **[Commercial License Info](../COMMERCIAL_LICENSE_INFO.md)**

---

**Enjoy using MAESTRO Protocol responsibly!** 🚀 