"""
Data Analysis Computational Engine

Provides computational data analysis through statistical algorithms, pattern recognition,
and data quality assessment using pandas, NumPy, SciPy, and other computational libraries.
"""

import json
import re
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Import computational libraries with graceful fallbacks
try:
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import pdist
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    DATA_ANALYSIS_LIBRARIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data analysis computation libraries not available: {e}")
    DATA_ANALYSIS_LIBRARIES_AVAILABLE = False
    pd = None
    np = None
    stats = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class DataAnalysisEngine:
    """
    Computational engine for data analysis through statistical algorithms and pattern recognition.
    
    LLM calls these methods with parameters to get actual computational results.
    """
    
    def __init__(self):
        self.name = "Data Analysis Engine"
        self.version = "1.0.0"
        self.supported_calculations = [
            "statistical_summary",
            "correlation_analysis", 
            "outlier_detection",
            "cluster_analysis",
            "distribution_fitting",
            "hypothesis_testing",
            "time_series_analysis",
            "data_quality_assessment"
        ]
    
    def calculate_statistical_summary(self, data: List[Union[float, int]], 
                                    variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive statistical summary using computational methods.
        
        Args:
            data: Numerical data for analysis
            variables: Variable names (optional)
            
        Returns:
            Dict with statistical measures and computational details
        """
        try:
            logger.info("📊 Computing statistical summary...")
            
            if not DATA_ANALYSIS_LIBRARIES_AVAILABLE:
                return {"error": "NumPy and SciPy libraries not available"}
            
            # Convert to numpy array
            data_array = np.array(data, dtype=float)
            
            # Remove NaN values
            clean_data = data_array[~np.isnan(data_array)]
            
            if len(clean_data) == 0:
                return {"error": "No valid numerical data provided"}
            
            # Calculate basic statistics
            mean_val = float(np.mean(clean_data))
            median_val = float(np.median(clean_data))
            std_val = float(np.std(clean_data, ddof=1))
            var_val = float(np.var(clean_data, ddof=1))
            min_val = float(np.min(clean_data))
            max_val = float(np.max(clean_data))
            
            # Calculate quartiles
            q1 = float(np.percentile(clean_data, 25))
            q3 = float(np.percentile(clean_data, 75))
            iqr = q3 - q1
            
            # Calculate distribution properties
            skewness = float(stats.skew(clean_data))
            kurt = float(stats.kurtosis(clean_data))
            
            # Confidence intervals
            sem = stats.sem(clean_data)
            ci_95 = stats.t.interval(0.95, len(clean_data)-1, loc=mean_val, scale=sem)
            
            # Normality test
            shapiro_stat, shapiro_p = stats.shapiro(clean_data[:5000])  # Limit for large datasets
            
            result = {
                "sample_size": len(clean_data),
                "missing_values": len(data_array) - len(clean_data),
                "central_tendency": {
                    "mean": mean_val,
                    "median": median_val,
                    "mode": self._calculate_mode(clean_data)
                },
                "dispersion": {
                    "standard_deviation": std_val,
                    "variance": var_val,
                    "range": max_val - min_val,
                    "interquartile_range": iqr
                },
                "distribution_shape": {
                    "skewness": skewness,
                    "kurtosis": kurt,
                    "min": min_val,
                    "max": max_val,
                    "q1": q1,
                    "q3": q3
                },
                "confidence_intervals": {
                    "mean_95_ci": [float(ci_95[0]), float(ci_95[1])],
                    "standard_error": float(sem)
                },
                "normality_test": {
                    "shapiro_wilk_statistic": float(shapiro_stat),
                    "shapiro_wilk_p_value": float(shapiro_p),
                    "is_normal_p05": shapiro_p > 0.05
                },
                "computation_method": "NumPy and SciPy statistical algorithms"
            }
            
            logger.info(f"✅ Statistical summary complete: {len(clean_data)} data points analyzed")
            return result
                    
        except Exception as e:
            logger.error(f"❌ Statistical summary failed: {str(e)}")
            return {"error": f"Calculation failed: {str(e)}"}
    
    def calculate_correlation_analysis(self, data_matrix: List[List[float]], 
                                     variable_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate correlation analysis using computational methods.
        
        Args:
            data_matrix: 2D array where each row is an observation, each column a variable
            variable_names: Names for variables (optional)
            
        Returns:
            Dict with correlation matrix and significance tests
        """
        try:
            logger.info("📊 Computing correlation analysis...")
            
            if not DATA_ANALYSIS_LIBRARIES_AVAILABLE:
                return {"error": "NumPy and SciPy libraries not available"}
            
            # Convert to numpy array
            data_array = np.array(data_matrix, dtype=float)
            
            if data_array.ndim != 2:
                return {"error": "Data must be 2-dimensional (observations x variables)"}
            
            n_obs, n_vars = data_array.shape
            
            if variable_names is None:
                variable_names = [f"var_{i+1}" for i in range(n_vars)]
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(data_array.T)
            
            # Calculate p-values for correlations
            p_values = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        corr_coef, p_val = stats.pearsonr(data_array[:, i], data_array[:, j])
                        p_values[i, j] = p_val
            
            # Find significant correlations (p < 0.05)
            significant_correlations = []
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    corr_val = corr_matrix[i, j]
                    p_val = p_values[i, j]
                    if p_val < 0.05:
                        significant_correlations.append({
                            "variable_1": variable_names[i],
                            "variable_2": variable_names[j],
                            "correlation": float(corr_val),
                            "p_value": float(p_val),
                            "strength": self._interpret_correlation_strength(abs(corr_val))
                        })
            
            # Calculate partial correlations (simplified)
            partial_correlations = self._calculate_partial_correlations(data_array, variable_names)
            
            result = {
                "correlation_matrix": {
                    variable_names[i]: {
                        variable_names[j]: float(corr_matrix[i, j]) 
                        for j in range(n_vars)
                    } for i in range(n_vars)
                },
                "p_value_matrix": {
                    variable_names[i]: {
                        variable_names[j]: float(p_values[i, j]) 
                        for j in range(n_vars)
                    } for i in range(n_vars)
                },
                "significant_correlations": significant_correlations,
                "partial_correlations": partial_correlations,
                "matrix_properties": {
                    "determinant": float(np.linalg.det(corr_matrix)),
                    "condition_number": float(np.linalg.cond(corr_matrix)),
                    "eigenvalues": np.linalg.eigvals(corr_matrix).tolist()
                },
                "sample_size": n_obs,
                "n_variables": n_vars,
                "computation_method": "Pearson correlation with significance testing"
            }
            
            logger.info(f"✅ Correlation analysis complete: {n_vars} variables, {len(significant_correlations)} significant correlations")
            return result
            
        except Exception as e:
            logger.error(f"❌ Correlation analysis failed: {str(e)}")
            return {"error": f"Calculation failed: {str(e)}"}
    
    def detect_outliers(self, data: List[float], method: str = "iqr") -> Dict[str, Any]:
        """
        Detect outliers using statistical methods.
        
        Args:
            data: Numerical data for outlier detection
            method: Detection method ("iqr", "zscore", "modified_zscore")
            
        Returns:
            Dict with outlier indices and statistics
        """
        try:
            logger.info(f"📊 Computing outlier detection using {method} method...")
            
            if not DATA_ANALYSIS_LIBRARIES_AVAILABLE:
                return {"error": "NumPy and SciPy libraries not available"}
            
            data_array = np.array(data, dtype=float)
            clean_data = data_array[~np.isnan(data_array)]
            
            outlier_indices = []
            outlier_values = []
            
            if method == "iqr":
                q1 = np.percentile(clean_data, 25)
                q3 = np.percentile(clean_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                for i, value in enumerate(clean_data):
                    if value < lower_bound or value > upper_bound:
                        outlier_indices.append(i)
                        outlier_values.append(float(value))
                        
                bounds = {"lower": float(lower_bound), "upper": float(upper_bound)}
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(clean_data))
                threshold = 3.0
                
                for i, z_score in enumerate(z_scores):
                    if z_score > threshold:
                        outlier_indices.append(i)
                        outlier_values.append(float(clean_data[i]))
                        
                bounds = {"threshold": threshold}
                
            elif method == "modified_zscore":
                median = np.median(clean_data)
                mad = np.median(np.abs(clean_data - median))
                modified_z_scores = 0.6745 * (clean_data - median) / mad
                threshold = 3.5
                
                for i, mod_z in enumerate(modified_z_scores):
                    if abs(mod_z) > threshold:
                        outlier_indices.append(i)
                        outlier_values.append(float(clean_data[i]))
                        
                bounds = {"threshold": threshold, "mad": float(mad)}
                
            else:
                return {"error": f"Unknown outlier detection method: {method}"}
            
            result = {
                "outlier_detection": {
                    "method": method,
                    "n_outliers": len(outlier_indices),
                    "outlier_indices": outlier_indices,
                    "outlier_values": outlier_values,
                    "outlier_percentage": float(len(outlier_indices) / len(clean_data) * 100)
                },
                "detection_parameters": bounds,
                "data_summary": {
                    "total_points": len(clean_data),
                    "mean": float(np.mean(clean_data)),
                    "std": float(np.std(clean_data)),
                    "median": float(np.median(clean_data))
                },
                "computation_method": f"Statistical outlier detection using {method} algorithm"
            }
            
            logger.info(f"✅ Outlier detection complete: {len(outlier_indices)} outliers found")
            return result
            
        except Exception as e:
            logger.error(f"❌ Outlier detection failed: {str(e)}")
            return {"error": f"Calculation failed: {str(e)}"}
    
    def _calculate_mode(self, data: np.ndarray) -> Optional[float]:
        """Calculate mode for numerical data."""
        try:
            mode_result = stats.mode(data, keepdims=True)
            return float(mode_result.mode[0])
        except Exception:
            return None
    
    def _interpret_correlation_strength(self, abs_corr: float) -> str:
        """Interpret correlation strength."""
        if abs_corr >= 0.9:
            return "very_strong"
        elif abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very_weak"
    
    def _calculate_partial_correlations(self, data: np.ndarray, var_names: List[str]) -> Dict[str, Any]:
        """Calculate partial correlations (simplified implementation)."""
        try:
            # This is a simplified version - full partial correlation requires matrix inversion
            corr_matrix = np.corrcoef(data.T)
            precision_matrix = np.linalg.inv(corr_matrix)
            
            partial_corrs = {}
            n_vars = len(var_names)
            
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    partial_corr = -precision_matrix[i, j] / np.sqrt(precision_matrix[i, i] * precision_matrix[j, j])
                    partial_corrs[f"{var_names[i]}__{var_names[j]}"] = float(partial_corr)
            
            return partial_corrs
            
        except np.linalg.LinAlgError:
            return {"error": "Cannot compute partial correlations - singular matrix"}
        except Exception as e:
            return {"error": f"Partial correlation calculation failed: {str(e)}"}
    
    # Keep some existing methods for backward compatibility
    def analyze_data(self, data: Any, analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        try:
            # Try to extract numerical data and run statistical summary
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                return self.calculate_statistical_summary(data)
            else:
                return {"error": "Legacy analyze_data method requires list of numbers. Use specific calculation methods instead."}
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            'success': False,
            'error': error_message,
            'computation_method': 'error_handling'
        } 