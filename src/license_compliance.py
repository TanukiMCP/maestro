# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
License Compliance Monitor - Detects potential commercial use violations
"""

import os
import socket
import platform
import hashlib
import time
import json
from typing import Dict, List, Optional
from pathlib import Path

class LicenseComplianceMonitor:
    """Monitor for license compliance and commercial use detection."""
    
    def __init__(self):
        self.compliance_file = Path.home() / ".maestro_compliance"
        self.commercial_indicators = [
            # Common commercial domains
            'amazonaws.com', 'googlecloud.com', 'azure.com', 'heroku.com',
            # Corporate network patterns
            'corp.', 'company.', 'enterprise.', 'inc.', 'llc.',
            # Production indicators
            'prod', 'production', 'live', 'api', 'service'
        ]
        
    def check_compliance(self) -> Dict:
        """Perform compliance check and return results."""
        results = {
            'timestamp': time.time(),
            'compliant': True,
            'warnings': [],
            'violations': [],
            'environment_score': 0,
            'risk_level': 'LOW'
        }
        
        # Check environment indicators
        env_score = self._analyze_environment()
        results['environment_score'] = env_score
        
        # Determine risk level
        if env_score > 70:
            results['risk_level'] = 'HIGH'
            results['violations'].append("High commercial environment score detected")
            results['compliant'] = False
        elif env_score > 40:
            results['risk_level'] = 'MEDIUM'
            results['warnings'].append("Potential commercial environment detected")
        
        # Check for commercial patterns
        commercial_patterns = self._detect_commercial_patterns()
        if commercial_patterns:
            results['violations'].extend(commercial_patterns)
            results['compliant'] = False
            
        # Log compliance check
        self._log_compliance_check(results)
        
        return results
    
    def _analyze_environment(self) -> int:
        """Analyze environment for commercial indicators. Returns score 0-100."""
        score = 0
        
        # Check hostname
        hostname = socket.gethostname().lower()
        for indicator in self.commercial_indicators:
            if indicator in hostname:
                score += 20
                
        # Check environment variables
        env_vars = os.environ
        commercial_env_vars = [
            'AWS_', 'AZURE_', 'GCP_', 'HEROKU_', 'KUBERNETES_',
            'DOCKER_', 'PROD_', 'PRODUCTION_'
        ]
        
        for env_var in env_vars:
            for commercial_var in commercial_env_vars:
                if env_var.startswith(commercial_var):
                    score += 10
                    break
                    
        # Check for containerization (often indicates production)
        if os.path.exists('/.dockerenv'):
            score += 15
            
        if os.environ.get('KUBERNETES_SERVICE_HOST'):
            score += 20
            
        # Check for cloud metadata endpoints
        try:
            import urllib.request
            cloud_endpoints = [
                'http://169.254.169.254/latest/meta-data/',  # AWS
                'http://metadata.google.internal/',           # GCP
                'http://169.254.169.254/metadata/instance'    # Azure
            ]
            
            for endpoint in cloud_endpoints:
                try:
                    urllib.request.urlopen(endpoint, timeout=1)
                    score += 25
                    break
                except:
                    pass
        except:
            pass
            
        return min(score, 100)
    
    def _detect_commercial_patterns(self) -> List[str]:
        """Detect specific commercial use patterns."""
        violations = []
        
        # Check for high-volume usage patterns
        if self._check_high_volume_usage():
            violations.append("High-volume usage pattern detected (>1000 requests/day)")
            
        # Check for API monetization patterns
        if self._check_api_monetization():
            violations.append("API monetization patterns detected")
            
        # Check for corporate file structures
        if self._check_corporate_structure():
            violations.append("Corporate development structure detected")
            
        return violations
    
    def _check_high_volume_usage(self) -> bool:
        """Check for high-volume usage indicating commercial use."""
        try:
            if self.compliance_file.exists():
                with open(self.compliance_file, 'r') as f:
                    data = json.load(f)
                    
                daily_requests = data.get('daily_requests', 0)
                return daily_requests > 1000
        except:
            pass
        return False
    
    def _check_api_monetization(self) -> bool:
        """Check for API monetization indicators."""
        # Look for common monetization files/directories
        monetization_indicators = [
            'billing/', 'payments/', 'pricing/', 'subscription/',
            'stripe/', 'paypal/', 'revenue/', 'customers/'
        ]
        
        for indicator in monetization_indicators:
            if Path(indicator).exists():
                return True
                
        return False
    
    def _check_corporate_structure(self) -> bool:
        """Check for corporate development structure."""
        corporate_indicators = [
            'docker-compose.prod.yml',
            'kubernetes/',
            'terraform/',
            'helm/',
            '.ci/',
            '.github/workflows/',
            'Jenkinsfile',
            'deployment/',
            'infrastructure/'
        ]
        
        for indicator in corporate_indicators:
            if Path(indicator).exists():
                return True
                
        return False
    
    def _log_compliance_check(self, results: Dict):
        """Log compliance check results."""
        try:
            log_data = {
                'timestamp': results['timestamp'],
                'compliant': results['compliant'],
                'risk_level': results['risk_level'],
                'environment_score': results['environment_score'],
                'machine_id': self._get_machine_id(),
                'platform': platform.platform()
            }
            
            # Append to compliance file
            compliance_data = []
            if self.compliance_file.exists():
                try:
                    with open(self.compliance_file, 'r') as f:
                        compliance_data = json.load(f)
                except:
                    compliance_data = []
                    
            compliance_data.append(log_data)
            
            # Keep only last 100 entries
            compliance_data = compliance_data[-100:]
            
            with open(self.compliance_file, 'w') as f:
                json.dump(compliance_data, f, indent=2)
                
        except Exception as e:
            # Fail silently to avoid breaking the main application
            pass
    
    def _get_machine_id(self) -> str:
        """Get unique machine identifier."""
        try:
            machine_info = f"{platform.node()}-{platform.machine()}"
            return hashlib.md5(machine_info.encode()).hexdigest()[:16]
        except:
            return "unknown"
    
    def get_compliance_report(self) -> Dict:
        """Get detailed compliance report."""
        results = self.check_compliance()
        
        report = {
            'license': 'Non-Commercial License',
            'compliance_status': 'COMPLIANT' if results['compliant'] else 'VIOLATION',
            'risk_assessment': results['risk_level'],
            'environment_analysis': {
                'score': results['environment_score'],
                'indicators': results.get('warnings', []) + results.get('violations', [])
            },
            'commercial_licensing_contact': 'tanukimcp@gmail.com',
            'license_terms': 'https://github.com/tanukimcp/maestro/blob/main/LICENSE',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['timestamp']))
        }
        
        if not results['compliant']:
            report['action_required'] = [
                "Commercial use detected - License violation",
                "Contact tanukimcp@gmail.com for commercial licensing",
                "Cease commercial use immediately or obtain proper license"
            ]
            
        return report

# Global compliance monitor instance
_compliance_monitor = None

def get_compliance_monitor() -> LicenseComplianceMonitor:
    """Get global compliance monitor instance."""
    global _compliance_monitor
    if _compliance_monitor is None:
        _compliance_monitor = LicenseComplianceMonitor()
    return _compliance_monitor

def check_license_compliance() -> bool:
    """Quick compliance check. Returns True if compliant."""
    monitor = get_compliance_monitor()
    results = monitor.check_compliance()
    
    if not results['compliant']:
        print("\n⚠️  LICENSE VIOLATION DETECTED ⚠️")
        print("This software is licensed for NON-COMMERCIAL use only.")
        print("Commercial use requires explicit permission from TanukiMCP.")
        print("Contact: tanukimcp@gmail.com")
        print("License: https://github.com/tanukimcp/maestro/blob/main/LICENSE")
        print()
        
    return results['compliant'] 