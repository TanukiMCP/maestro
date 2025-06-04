#!/usr/bin/env python3
import requests
import json

# Test the 4 working tools
tools_to_test = [
    ('get_available_engines', {}),
    ('maestro_iae_discovery', {'task_type': 'data_analysis', 'complexity': 'medium'}),
    ('maestro_tool_selection', {'task_description': 'Analyze data trends'}),
    ('maestro_orchestrate', {'task_description': 'What is 2+2?', 'complexity_level': 'basic'})
]

print("🧪 Quick Tool Test")
print("=" * 40)

for tool_name, args in tools_to_test:
    payload = {
        'jsonrpc': '2.0',
        'id': f'test-{tool_name}',
        'method': 'tools/call',
        'params': {'name': tool_name, 'arguments': args}
    }
    
    try:
        response = requests.post('http://localhost:8000/mcp', json=payload, timeout=10)
        if response.status_code == 200:
            print(f'✅ {tool_name}: SUCCESS')
        else:
            print(f'❌ {tool_name}: ERROR {response.status_code}')
    except Exception as e:
        print(f'❌ {tool_name}: EXCEPTION {str(e)[:50]}')

print("=" * 40) 