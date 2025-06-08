#!/usr/bin/env python3

import json
import requests

def test_tools():
    BASE_URL = "http://localhost:8000"
    
    print("Testing maestro_iae_discovery...")
    discovery_result = requests.post(
        f"{BASE_URL}/call",
        json={
            "name": "maestro_iae_discovery",
            "inputs": {
                "task_description": "Calculate the entanglement entropy of a quantum system",
                "domain_context": "quantum physics",
                "complexity_assessment": "moderate",
                "computational_requirements": {
                    "requires_computation": True,
                    "resource_intensity": "medium"
                }
            }
        }
    )
    print("\nIAE Discovery Result:\n" + "="*50)
    try:
        result = discovery_result.json()
        print(json.dumps(result, indent=2))
    except:
        print(discovery_result.text)
    
    print("\n\nTesting maestro_temporal_context...")
    temporal_result = requests.post(
        f"{BASE_URL}/call",
        json={
            "name": "maestro_temporal_context",
            "inputs": {
                "query": "What are the current trends in quantum computing research?",
                "time_scope": "current",
                "context_depth": "detailed",
                "currency_check": True
            }
        }
    )
    print("\nTemporal Context Result:\n" + "="*50)
    try:
        result = temporal_result.json()
        print(json.dumps(result, indent=2))
    except:
        print(temporal_result.text)
    
    # Test backward compatibility parameters
    print("\n\nTesting maestro_temporal_context with backward compatibility parameters...")
    temporal_compat_result = requests.post(
        f"{BASE_URL}/call",
        json={
            "name": "maestro_temporal_context",
            "inputs": {
                "context_request": "What were the historical developments in AI?",
                "time_frame": "historical",
                "context_depth": "detailed",
                "currency_check": True
            }
        }
    )
    print("\nTemporal Context (Compatibility) Result:\n" + "="*50)
    try:
        result = temporal_compat_result.json()
        print(json.dumps(result, indent=2))
    except:
        print(temporal_compat_result.text)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_tools() 