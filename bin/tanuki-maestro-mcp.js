#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Check if Python is available
function checkPython() {
    return new Promise((resolve) => {
        const python = spawn('python', ['--version'], { stdio: 'pipe' });
        python.on('close', (code) => {
            if (code === 0) {
                resolve('python');
            } else {
                const python3 = spawn('python3', ['--version'], { stdio: 'pipe' });
                python3.on('close', (code) => {
                    resolve(code === 0 ? 'python3' : null);
                });
            }
        });
    });
}

async function main() {
    console.log('🎭 Starting Tanuki Maestro MCP Server...');
    
    // Check Python availability
    const pythonCmd = await checkPython();
    if (!pythonCmd) {
        console.error('❌ Error: Python 3.9+ is required but not found.');
        console.error('Please install Python from https://python.org');
        process.exit(1);
    }
    
    // Get the package root directory
    const packageRoot = path.dirname(__dirname);
    const mainScript = path.join(packageRoot, 'src', 'main.py');
    
    if (!fs.existsSync(mainScript)) {
        console.error('❌ Error: Main script not found at', mainScript);
        process.exit(1);
    }
    
    // Install Python dependencies if requirements.txt exists
    const requirementsFile = path.join(packageRoot, 'requirements.txt');
    if (fs.existsSync(requirementsFile)) {
        console.log('📦 Installing Python dependencies...');
        const pip = spawn(pythonCmd, ['-m', 'pip', 'install', '-r', requirementsFile], {
            stdio: 'inherit',
            cwd: packageRoot
        });
        
        await new Promise((resolve) => {
            pip.on('close', resolve);
        });
    }
    
    // Parse command line arguments
    const args = process.argv.slice(2);
    
    // Check if user wants to start the HTTP server
    if (args.includes('--http') || args.includes('--server')) {
        console.log('🚀 Starting HTTP/SSE server...');
        const server = spawn('uvicorn', ['src.main:app', '--host', '0.0.0.0', '--port', '8001'], {
            stdio: 'inherit',
            cwd: packageRoot
        });
        
        process.on('SIGINT', () => {
            server.kill('SIGINT');
        });
        
        server.on('close', (code) => {
            process.exit(code);
        });
    } else {
        // Default: run as stdio MCP server
        console.log('🔌 Starting MCP server (stdio mode)...');
        const server = spawn(pythonCmd, [mainScript].concat(args), {
            stdio: 'inherit',
            cwd: packageRoot
        });
        
        process.on('SIGINT', () => {
            server.kill('SIGINT');
        });
        
        server.on('close', (code) => {
            process.exit(code);
        });
    }
}

main().catch((error) => {
    console.error('❌ Error:', error.message);
    process.exit(1);
}); 