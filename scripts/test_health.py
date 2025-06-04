import subprocess
import time

def main():
    proc = subprocess.Popen(
        ["python", "mcp_stdio_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    try:
        for _ in range(20):
            line = proc.stdout.readline()
            print(line, end="")
            if "Starting Maestro MCP Server" in line:
                print("Server started successfully.")
                break
            time.sleep(0.5)
    finally:
        proc.terminate()

if __name__ == "__main__":
    main() 