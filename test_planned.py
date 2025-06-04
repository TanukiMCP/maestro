#!/usr/bin/env python3
import requests
import json

payload = {
    'jsonrpc': '2.0', 
    'id': 'test-planned', 
    'method': 'tools/call', 
    'params': {
        'name': 'maestro_search', 
        'arguments': {'query': 'test'}
    }
}

response = requests.post('http://localhost:8000/mcp', json=payload)
print('Status:', response.status_code)
if response.status_code == 200:
    result = response.json()['result']['content'][0]['text']
    print('Response:', result)
else:
    print('Error:', response.text) 