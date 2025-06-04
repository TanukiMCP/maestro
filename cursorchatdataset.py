# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

import os
import json
import re
import glob
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

# Constants
CURSOR_LOGS_PATH = r"C:\Users\ididi\AppData\Roaming\Cursor"
OUTPUT_PATH = r"C:\Users\ididi\tanukimcp-maestro\processed_logs"
ROLES = [
    "System Architect", 
    "Senior Developer", 
    "Frontend Specialist", 
    "Backend Engineer", 
    "DevOps Expert", 
    "Debug Specialist", 
    "Technical Writer"
]

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

def find_chat_files():
    """Find all potential chat log files in the Cursor directory"""
    # Look in multiple potential locations
    paths_to_check = [
        Path(CURSOR_LOGS_PATH) / "User" / "History",
        Path(CURSOR_LOGS_PATH) / "logs",
        Path(CURSOR_LOGS_PATH) / "User"
    ]
    
    found_files = []
    for base_path in paths_to_check:
        if base_path.exists():
            print(f"Searching in {base_path}...")
            # Look for JSON files in this directory and all subdirectories
            for json_file in base_path.glob("**/*.json"):
                found_files.append(json_file)
    
    print(f"Found {len(found_files)} potential chat files")
    return found_files

def extract_from_file(file_path):
    """Try to extract conversation data from a file with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            # Check if the file contains chat-like content
            if '"role"' in file_content and ('"user"' in file_content or '"assistant"' in file_content):
                # Clean the JSON if needed
                try:
                    chat_data = json.loads(file_content)
                    return chat_data
                except json.JSONDecodeError as e:
                    print(f"JSON error in {file_path}: {e}")
                    return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def extract_conversations(files):
    """Extract conversation data from files"""
    conversations = []
    
    for file_path in files:
        chat_data = extract_from_file(file_path)
        if chat_data:
            # Different file formats may have different structures
            if isinstance(chat_data, dict):
                if 'messages' in chat_data:
                    # Standard format
                    conversation = {
                        'id': str(file_path),
                        'messages': chat_data.get('messages', [])
                    }
                    conversations.append(conversation)
                elif 'history' in chat_data:
                    # Alternative format
                    conversation = {
                        'id': str(file_path),
                        'messages': chat_data.get('history', [])
                    }
                    conversations.append(conversation)
            elif isinstance(chat_data, list):
                # List format
                conversation = {
                    'id': str(file_path),
                    'messages': chat_data
                }
                conversations.append(conversation)
    
    print(f"Successfully extracted {len(conversations)} conversations")
    return conversations

def simple_categorize(conversation):
    """Assign a role based on simple keyword matching"""
    full_text = ""
    for msg in conversation.get('messages', []):
        if isinstance(msg, dict):
            content = msg.get('content', '')
            if isinstance(content, str):
                full_text += " " + content
    
    # Define keywords for each role
    role_keywords = {
        "System Architect": ["architecture", "system design", "infrastructure", "scalability"],
        "Senior Developer": ["algorithm", "code quality", "refactoring", "design pattern"],
        "Frontend Specialist": ["ui", "ux", "react", "css", "html", "javascript", "responsive"],
        "Backend Engineer": ["api", "database", "server", "endpoint", "authentication"],
        "DevOps Expert": ["deployment", "ci/cd", "kubernetes", "docker", "container", "aws"],
        "Debug Specialist": ["bug", "debug", "error", "exception", "fix", "test"],
        "Technical Writer": ["documentation", "readme", "comment", "explain"]
    }
    
    # Count keyword matches
    role_scores = {role: 0 for role in ROLES}
    for role, keywords in role_keywords.items():
        for keyword in keywords:
            if keyword.lower() in full_text.lower():
                role_scores[role] += 1
    
    # Assign the role with highest score, or default to Senior Developer
    max_score = max(role_scores.values()) if role_scores else 0
    if max_score > 0:
        for role, score in role_scores.items():
            if score == max_score:
                return role
    
    return "Senior Developer"  # Default role

def format_for_training(conversations):
    """Format conversations for training"""
    # Group by role
    role_data = {role.replace(" ", "_"): [] for role in ROLES}
    
    for conv in conversations:
        role = simple_categorize(conv)
        pairs = []
        
        messages = conv.get('messages', [])
        for i in range(len(messages) - 1):
            user_msg = None
            assistant_msg = None
            
            # Extract user and assistant messages
            if isinstance(messages[i], dict) and isinstance(messages[i+1], dict):
                if messages[i].get('role') == 'user' and messages[i+1].get('role') == 'assistant':
                    user_msg = messages[i].get('content', '')
                    assistant_msg = messages[i+1].get('content', '')
            
            if user_msg and assistant_msg:
                pairs.append({
                    'user': user_msg,
                    'assistant': assistant_msg
                })
        
        if pairs:
            role_data[role.replace(" ", "_")].extend(pairs)
    
    return role_data

def save_processed_data(role_data):
    """Save processed conversations to files by role"""
    for role, conversations in role_data.items():
        if conversations:
            # Create role directory
            role_dir = os.path.join(OUTPUT_PATH, role)
            os.makedirs(role_dir, exist_ok=True)
            
            # Save as JSON
            json_path = os.path.join(role_dir, f"{role}_data.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2)
            
            # Save as JSONL (one JSON object per line)
            jsonl_path = os.path.join(role_dir, f"{role}_data.jsonl")
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for conv in conversations:
                    f.write(json.dumps(conv) + '\n')
            
            print(f"Saved {len(conversations)} conversations for {role}")

def main():
    print("Searching for chat log files...")
    all_files = find_chat_files()
    
    print("Extracting conversations...")
    conversations = extract_conversations(all_files)
    
    if not conversations:
        print("No conversations found. Exiting.")
        return
    
    print("Categorizing conversations...")
    role_data = format_for_training(conversations)
    
    print("Saving processed data...")
    save_processed_data(role_data)
    
    print("Done!")

if __name__ == "__main__":
    main()
