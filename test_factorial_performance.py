#!/usr/bin/env python3
"""
Test factorial calculation performance with MAESTRO Protocol optimizations
"""

import asyncio
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_factorial_performance():
    """Test factorial calculation with performance monitoring"""
    print("🎭 Testing MAESTRO Protocol Factorial Performance")
    print("=" * 60)
    
    try:
        from maestro.orchestrator import MAESTROOrchestrator
        
        orchestrator = MAESTROOrchestrator()
        print("✅ MAESTRO Orchestrator initialized")
        
        # Test with fast mode (new default)
        print("\n🚀 Testing Fast Mode:")
        start_time = time.time()
        
        result = await orchestrator.orchestrate_workflow(
            task_description="Calculate factorial of 5",
            verification_mode="fast"
        )
        
        fast_time = time.time() - start_time
        print(f"⏱️  Fast mode time: {fast_time:.3f} seconds")
        print(f"✅ Success: {result.success}")
        print(f"📄 Output preview: {result.detailed_output[:200]}...")
        
        # Test with comprehensive mode for comparison
        print("\n🔍 Testing Comprehensive Mode:")
        start_time = time.time()
        
        result = await orchestrator.orchestrate_workflow(
            task_description="Calculate factorial of 5",
            verification_mode="comprehensive"
        )
        
        comprehensive_time = time.time() - start_time
        print(f"⏱️  Comprehensive mode time: {comprehensive_time:.3f} seconds")
        print(f"✅ Success: {result.success}")
        print(f"📄 Output preview: {result.detailed_output[:200]}...")
        
        # Performance comparison
        print("\n📊 Performance Comparison:")
        print(f"Fast mode: {fast_time:.3f}s")
        print(f"Comprehensive mode: {comprehensive_time:.3f}s")
        if comprehensive_time > 0:
            speedup = comprehensive_time / fast_time
            print(f"🚀 Speedup: {speedup:.1f}x faster")
        
        # Success criteria
        if fast_time < 2.0:  # Target: under 2 seconds for simple math
            print("🎉 PERFORMANCE TARGET MET: Fast mode under 2 seconds!")
            return True
        else:
            print("⚠️  Performance target missed - may need more optimization")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Run performance tests"""
    success = asyncio.run(test_factorial_performance())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 