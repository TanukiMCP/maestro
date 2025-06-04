import os
import json
from pathlib import Path

def search_for_chat_data():
    """Search for actual chat conversation data in Cursor directories"""
    
    # Possible locations for chat data
    search_paths = [
        Path(r"C:\Users\ididi\AppData\Roaming\Cursor"),
        Path(r"C:\Users\ididi\AppData\Local\Cursor"),
        Path(r"C:\Users\ididi\.cursor"),
        Path(r"C:\Users\ididi\AppData\Roaming\Code\User\globalStorage"),
        Path(r"C:\Users\ididi\AppData\Roaming\Cursor\User\globalStorage")
    ]
    
    keywords_to_search = [
        "conversation", "chat", "messages", "assistant", "user", 
        "anthropic", "openai", "claude", "gpt", "ai", "copilot"
    ]
    
    for base_path in search_paths:
        if not base_path.exists():
            continue
            
        print(f"\n=== Searching in {base_path} ===")
        
        # Search for files that might contain chat data
        for file_path in base_path.rglob("*"):
            if file_path.is_file():
                # Check file name for relevant keywords
                file_name_lower = file_path.name.lower()
                if any(keyword in file_name_lower for keyword in keywords_to_search):
                    print(f"Found potential file: {file_path}")
                    examine_file(file_path)
                
                # Check JSON files for chat-like content
                if file_path.suffix.lower() == '.json':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read(1000)  # First 1000 chars
                            if any(keyword in content.lower() for keyword in keywords_to_search):
                                if '"role"' in content and ('"user"' in content or '"assistant"' in content):
                                    print(f"Found potential chat file: {file_path}")
                                    examine_file(file_path)
                    except:
                        continue
    
    # Also check for SQLite databases which might store chat data
    print(f"\n=== Looking for databases ===")
    for base_path in search_paths:
        if base_path.exists():
            for db_file in base_path.rglob("*.db"):
                print(f"Found database: {db_file}")
            for db_file in base_path.rglob("*.sqlite"):
                print(f"Found SQLite database: {db_file}")

def examine_file(file_path):
    """Examine a potentially relevant file"""
    try:
        size = file_path.stat().st_size
        print(f"  Size: {size} bytes")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(500)
            print(f"  Content preview: {content[:200]}...")
            
        # Try to parse as JSON
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    print(f"  JSON structure: {type(data)}")
                    if isinstance(data, dict):
                        print(f"  Keys: {list(data.keys())}")
                except:
                    pass
        print()
    except Exception as e:
        print(f"  Error examining file: {e}")

if __name__ == "__main__":
    search_for_chat_data() 