# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

import os
import json
from pathlib import Path
import random

def examine_cursor_files():
    """Examine the structure of Cursor files to understand the format"""
    cursor_path = Path(r"C:\Users\ididi\AppData\Roaming\Cursor")
    
    # Look in multiple locations
    paths_to_check = [
        cursor_path / "User" / "History",
        cursor_path / "logs", 
        cursor_path / "User"
    ]
    
    all_files = []
    for base_path in paths_to_check:
        if base_path.exists():
            print(f"\n=== Examining {base_path} ===")
            files = list(base_path.glob("**/*"))
            print(f"Found {len(files)} total files/folders")
            
            # Show directory structure
            for item in sorted(files)[:10]:  # Show first 10 items
                if item.is_file():
                    size = item.stat().st_size
                    print(f"FILE: {item.name} ({size} bytes)")
                    all_files.append(item)
                else:
                    print(f"DIR:  {item.name}/")
            
            if len(files) > 10:
                print(f"... and {len(files) - 10} more items")
    
    # Sample a few files to examine their content
    if all_files:
        print(f"\n=== EXAMINING SAMPLE FILES ===")
        sample_files = random.sample(all_files, min(5, len(all_files)))
        
        for file_path in sample_files:
            print(f"\n--- File: {file_path} ---")
            try:
                # Try to read as text first
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(500)  # First 500 characters
                    print(f"Content preview:\n{content}")
                    
                # Try to parse as JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        print(f"JSON structure: {type(data)}")
                        if isinstance(data, dict):
                            print(f"Keys: {list(data.keys())}")
                        elif isinstance(data, list):
                            print(f"List length: {len(data)}")
                            if data:
                                print(f"First item type: {type(data[0])}")
                                if isinstance(data[0], dict):
                                    print(f"First item keys: {list(data[0].keys())}")
                    except json.JSONDecodeError:
                        print("Not valid JSON")
                        
            except Exception as e:
                print(f"Error reading file: {e}")
    
    # Let's also check what specific file extensions we have
    print(f"\n=== FILE EXTENSIONS ===")
    extensions = {}
    for file_path in all_files:
        ext = file_path.suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    
    for ext, count in sorted(extensions.items()):
        print(f"{ext if ext else '(no extension)'}: {count} files")

if __name__ == "__main__":
    examine_cursor_files() 
