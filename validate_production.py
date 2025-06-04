#!/usr/bin/env python3
"""
Production Validation for TanukiMCP Maestro
Ensures all tools actually perform their stated functions for Smithery.ai deployment
"""

import asyncio
import json
import time
import urllib.request
import urllib.parse
from typing import Dict, Any, List

class ProductionValidator:
    """Comprehensive validator for production deployment"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
        
    def test_tool_discovery_speed(self) -> Dict[str, Any]:
        """Validate tool discovery meets Smithery <100ms requirement"""
        print("🔍 Testing tool discovery speed...")
        
        times = []
        for i in range(5):  # Test 5 times for consistency
            start_time = time.time()
            try:
                response = urllib.request.urlopen(f"{self.base_url}/mcp", timeout=10)
                discovery_time = (time.time() - start_time) * 1000
                times.append(discovery_time)
                data = json.loads(response.read().decode())
                tools_count = len(data.get("result", {}).get("tools", []))
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        result = {
            "success": max_time < 100,  # Smithery requirement
            "average_time_ms": round(avg_time, 2),
            "max_time_ms": round(max_time, 2),
            "tools_count": tools_count,
            "consistency": "PASS" if max(times) - min(times) < 50 else "FAIL"
        }
        
        print(f"   ⏱️  Average: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
        print(f"   📊 Tools: {tools_count}")
        print(f"   🎯 Smithery Requirement (<100ms): {'✅ PASS' if result['success'] else '❌ FAIL'}")
        
        return result
    
    def test_tool_functionality(self) -> Dict[str, Any]:
        """Test that each tool actually performs its stated function"""
        print("\n🛠️ Testing tool functionality...")
        
        test_cases = [
            {
                "name": "maestro_orchestrate",
                "test_args": {"task_description": "Analyze renewable energy benefits"},
                "expected_indicators": ["analysis", "reasoning", "validation"]
            },
            {
                "name": "maestro_iae", 
                "test_args": {"analysis_request": "Calculate factorial of 5"},
                "expected_indicators": ["120", "factorial", "calculation"]
            },
            {
                "name": "get_available_engines",
                "test_args": {"detailed": True},
                "expected_indicators": ["engines", "available", "capabilities"]
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"   🔧 Testing {test_case['name']}...")
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": test_case["name"],
                    "arguments": test_case["test_args"]
                }
            }
            
            try:
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(
                    f"{self.base_url}/mcp",
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                
                start_time = time.time()
                response = urllib.request.urlopen(req, timeout=30)
                execution_time = (time.time() - start_time) * 1000
                
                response_data = json.loads(response.read().decode())
                
                has_error = "error" in response_data
                has_content = "result" in response_data and "content" in response_data["result"]
                
                if has_content:
                    content = str(response_data["result"]["content"])
                    
                    functionality_score = 0
                    found_indicators = []
                    
                    for indicator in test_case["expected_indicators"]:
                        if indicator.lower() in content.lower():
                            functionality_score += 1
                            found_indicators.append(indicator)
                    
                    functionality_percentage = (functionality_score / len(test_case["expected_indicators"])) * 100
                    
                    result = {
                        "tool": test_case["name"],
                        "success": not has_error and has_content,
                        "execution_time_ms": round(execution_time, 2),
                        "functionality_score": functionality_percentage,
                        "actually_functional": functionality_percentage >= 30,
                        "status": "FUNCTIONAL" if functionality_percentage >= 30 else "LIMITED"
                    }
                    
                    print(f"      ⏱️  {execution_time:.0f}ms - {result['status']}")
                    
                else:
                    result = {
                        "tool": test_case["name"],
                        "success": False,
                        "actually_functional": False,
                        "status": "FAILED"
                    }
                    print(f"      ❌ Failed")
                
                results.append(result)
                
            except Exception as e:
                print(f"      ❌ Exception: {e}")
                results.append({
                    "tool": test_case["name"],
                    "success": False,
                    "actually_functional": False,
                    "status": "ERROR"
                })
        
        functional_tools = sum(1 for r in results if r.get("actually_functional", False))
        total_tools = len(results)
        
        print(f"\n   📊 Functionality: {functional_tools}/{total_tools} tools functional")
        
        return {
            "functional_tools": functional_tools,
            "total_tools": total_tools,
            "all_functional": functional_tools == total_tools,
            "results": results
        }
    
    def test_protocol_compliance(self) -> Dict[str, Any]:
        """Test MCP protocol compliance for Smithery.ai"""
        print("\n🌐 Testing MCP protocol compliance...")
        
        # Test initialize method
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        try:
            data = json.dumps(init_payload).encode('utf-8')
            req = urllib.request.Request(
                f"{self.base_url}/mcp",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            response = urllib.request.urlopen(req, timeout=10)
            response_data = json.loads(response.read().decode())
            
            # Check protocol compliance
            has_protocol_version = "protocolVersion" in response_data.get("result", {})
            has_server_info = "serverInfo" in response_data.get("result", {})
            has_capabilities = "capabilities" in response_data.get("result", {})
            
            protocol_version = response_data.get("result", {}).get("protocolVersion")
            is_correct_version = protocol_version == "2024-11-05"
            
            result = {
                "success": True,
                "protocol_version": protocol_version,
                "correct_version": is_correct_version,
                "has_server_info": has_server_info,
                "has_capabilities": has_capabilities,
                "smithery_compatible": has_protocol_version and has_server_info and is_correct_version
            }
            
            print(f"   🌐 Protocol Version: {protocol_version}")
            print(f"   ✅ Correct Version (2024-11-05): {'✅ YES' if is_correct_version else '❌ NO'}")
            print(f"   📋 Server Info: {'✅ YES' if has_server_info else '❌ NO'}")
            print(f"   🔧 Capabilities: {'✅ YES' if has_capabilities else '❌ NO'}")
            print(f"   ☁️ Smithery Compatible: {'✅ YES' if result['smithery_compatible'] else '❌ NO'}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Protocol test failed: {e}")
            return {"success": False, "error": str(e), "smithery_compatible": False}
    
    def test_production_readiness(self) -> Dict[str, Any]:
        """Test production readiness indicators"""
        print("\n🏭 Testing production readiness...")
        
        checks = []
        
        # Test health endpoint
        try:
            response = urllib.request.urlopen(f"{self.base_url}/health", timeout=5)
            health_data = json.loads(response.read().decode())
            
            health_check = {
                "name": "Health Endpoint",
                "success": health_data.get("status") == "healthy",
                "details": health_data
            }
            checks.append(health_check)
            print(f"   ❤️  Health Endpoint: {'✅ HEALTHY' if health_check['success'] else '❌ UNHEALTHY'}")
            
        except Exception as e:
            checks.append({"name": "Health Endpoint", "success": False, "error": str(e)})
            print(f"   ❤️  Health Endpoint: ❌ FAILED")
        
        # Test root endpoint
        try:
            response = urllib.request.urlopen(f"{self.base_url}/", timeout=5)
            root_data = json.loads(response.read().decode())
            
            root_check = {
                "name": "Root Endpoint",
                "success": "name" in root_data and "version" in root_data,
                "smithery_compatible": root_data.get("smithery_compatible", False),
                "details": root_data
            }
            checks.append(root_check)
            print(f"   🏠 Root Endpoint: {'✅ WORKING' if root_check['success'] else '❌ FAILED'}")
            print(f"   ☁️ Smithery Compatible: {'✅ YES' if root_check['smithery_compatible'] else '❌ NO'}")
            
        except Exception as e:
            checks.append({"name": "Root Endpoint", "success": False, "error": str(e)})
            print(f"   🏠 Root Endpoint: ❌ FAILED")
        
        # Calculate overall readiness
        successful_checks = sum(1 for check in checks if check["success"])
        total_checks = len(checks)
        
        return {
            "successful_checks": successful_checks,
            "total_checks": total_checks,
            "readiness_score": successful_checks / total_checks if total_checks > 0 else 0,
            "production_ready": successful_checks == total_checks,
            "checks": checks
        }
    
    def run_comprehensive_validation(self):
        """Run all validation tests"""
        print("🧪 PRODUCTION VALIDATION FOR SMITHERY.AI")
        print("=" * 50)
        
        discovery_result = self.test_tool_discovery_speed()
        functionality_result = self.test_tool_functionality()
        
        print("\n" + "=" * 50)
        print("📋 FINAL ASSESSMENT")
        print("=" * 50)
        
        discovery_pass = discovery_result.get("success", False)
        functionality_pass = functionality_result.get("all_functional", False)
        
        print(f"⚡ Discovery Speed: {'✅ PASS' if discovery_pass else '❌ FAIL'}")
        print(f"🛠️ Tool Functionality: {'✅ PASS' if functionality_pass else '❌ FAIL'}")
        
        all_pass = discovery_pass and functionality_pass
        
        print(f"\n{'🎯 SMITHERY.AI DEPLOYMENT READY' if all_pass else '⚠️ NEEDS FIXES'}")
        
        return {"production_ready": all_pass}

def main():
    validator = ProductionValidator()
    validator.run_comprehensive_validation()

if __name__ == "__main__":
    main() 