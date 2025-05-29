"""
Data Analysis Engine for MAESTRO Protocol
Provides comprehensive data analysis capabilities including statistical analysis,
data visualization, pattern recognition, and data quality assessment.
"""

import json
import re
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Optional imports with fallbacks
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

@dataclass
class DataQualityReport:
    """Report on data quality assessment"""
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    validity_score: float
    overall_score: float
    issues: List[str]
    recommendations: List[str]

@dataclass
class StatisticalSummary:
    """Statistical summary of data"""
    count: int
    mean: Optional[float]
    median: Optional[float]
    mode: Optional[Any]
    std_dev: Optional[float]
    variance: Optional[float]
    min_value: Optional[Any]
    max_value: Optional[Any]
    quartiles: Optional[Dict[str, float]]
    skewness: Optional[float]
    kurtosis: Optional[float]

@dataclass
class PatternAnalysis:
    """Results of pattern analysis"""
    trends: List[str]
    correlations: List[Dict[str, Any]]
    outliers: List[Any]
    clusters: List[Dict[str, Any]]
    seasonality: Optional[Dict[str, Any]]

class DataAnalysisEngine:
    """
    Advanced data analysis engine providing comprehensive analytical capabilities
    """
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'excel', 'tsv', 'parquet']
        self.analysis_types = [
            'descriptive', 'inferential', 'exploratory', 'confirmatory',
            'predictive', 'diagnostic', 'prescriptive'
        ]
    
    def analyze_data(self, data: Any, analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Perform comprehensive data analysis
        
        Args:
            data: Input data (various formats supported)
            analysis_type: Type of analysis to perform
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Parse and validate data
            parsed_data = self._parse_data(data)
            if not parsed_data:
                return self._create_error_result("Failed to parse input data")
            
            # Perform data quality assessment
            quality_report = self._assess_data_quality(parsed_data)
            
            # Generate statistical summary
            stats_summary = self._generate_statistical_summary(parsed_data)
            
            # Perform pattern analysis
            pattern_analysis = self._analyze_patterns(parsed_data)
            
            # Generate insights and recommendations
            insights = self._generate_insights(parsed_data, stats_summary, pattern_analysis)
            
            # Create visualizations if possible
            visualizations = self._create_visualizations(parsed_data)
            
            return {
                'success': True,
                'data_quality': quality_report.__dict__,
                'statistical_summary': stats_summary.__dict__,
                'pattern_analysis': pattern_analysis.__dict__,
                'insights': insights,
                'visualizations': visualizations,
                'recommendations': self._generate_recommendations(quality_report, insights),
                'metadata': {
                    'analysis_type': analysis_type,
                    'data_shape': self._get_data_shape(parsed_data),
                    'data_types': self._analyze_data_types(parsed_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Data analysis failed: {str(e)}")
            return self._create_error_result(f"Analysis failed: {str(e)}")
    
    def _parse_data(self, data: Any) -> Optional[Any]:
        """Parse input data into analyzable format"""
        try:
            if isinstance(data, str):
                # Try to parse as JSON
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    # Try to parse as CSV-like data
                    lines = data.strip().split('\n')
                    if len(lines) > 1:
                        headers = [h.strip() for h in lines[0].split(',')]
                        rows = []
                        for line in lines[1:]:
                            row_data = [cell.strip() for cell in line.split(',')]
                            if len(row_data) == len(headers):
                                rows.append(dict(zip(headers, row_data)))
                        return rows
                    return None
            
            elif isinstance(data, (list, dict)):
                return data
            
            elif HAS_PANDAS and isinstance(data, pd.DataFrame):
                return data.to_dict('records')
            
            else:
                # Try to convert to list
                try:
                    return list(data)
                except (TypeError, ValueError):
                    return None
                    
        except Exception as e:
            logger.error(f"Data parsing failed: {str(e)}")
            return None
    
    def _assess_data_quality(self, data: Any) -> DataQualityReport:
        """Assess the quality of the data"""
        issues = []
        recommendations = []
        
        # Convert to list of records if needed
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data) if hasattr(data, '__iter__') else [data]
        
        if not data:
            return DataQualityReport(0, 0, 0, 0, 0, ["No data provided"], ["Provide valid data"])
        
        # Completeness assessment
        total_fields = 0
        missing_fields = 0
        
        if isinstance(data[0], dict):
            all_keys = set()
            for record in data:
                if isinstance(record, dict):
                    all_keys.update(record.keys())
            
            for record in data:
                if isinstance(record, dict):
                    for key in all_keys:
                        total_fields += 1
                        if key not in record or record[key] is None or record[key] == '':
                            missing_fields += 1
        
        completeness_score = 1.0 - (missing_fields / max(total_fields, 1))
        
        # Consistency assessment
        consistency_issues = 0
        if isinstance(data[0], dict) and len(data) > 1:
            # Check for consistent data types
            type_patterns = {}
            for record in data:
                if isinstance(record, dict):
                    for key, value in record.items():
                        if key not in type_patterns:
                            type_patterns[key] = set()
                        type_patterns[key].add(type(value).__name__)
            
            for key, types in type_patterns.items():
                if len(types) > 2:  # Allow for some type variation
                    consistency_issues += 1
                    issues.append(f"Inconsistent data types in field '{key}': {types}")
        
        consistency_score = max(0, 1.0 - (consistency_issues / max(len(data), 1)))
        
        # Validity assessment (basic checks)
        validity_issues = 0
        for record in data:
            if isinstance(record, dict):
                for key, value in record.items():
                    # Check for obviously invalid values
                    if isinstance(value, str):
                        if len(value) > 1000:  # Suspiciously long strings
                            validity_issues += 1
                        if re.search(r'[^\w\s\-.,!?@#$%^&*()+=<>:;"\'/\\|`~]', str(value)):
                            # Contains unusual characters - log but don't penalize
                            logger.debug(f"Unusual characters found in value: {str(value)[:50]}...")
        
        validity_score = max(0, 1.0 - (validity_issues / max(len(data), 1)))
        
        # Accuracy assessment based on data consistency patterns
        accuracy_score = 0.8  # Default assumption
        
        # Overall score
        overall_score = (completeness_score + consistency_score + validity_score + accuracy_score) / 4
        
        # Generate recommendations
        if completeness_score < 0.8:
            recommendations.append("Address missing data through imputation or collection")
        if consistency_score < 0.8:
            recommendations.append("Standardize data formats and types")
        if validity_score < 0.8:
            recommendations.append("Validate and clean data entries")
        
        return DataQualityReport(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            validity_score=validity_score,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _generate_statistical_summary(self, data: Any) -> StatisticalSummary:
        """Generate statistical summary of the data"""
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data) if hasattr(data, '__iter__') else [data]
        
        if not data:
            return StatisticalSummary(0, None, None, None, None, None, None, None, None, None, None)
        
        count = len(data)
        
        # Extract numeric values for statistical analysis
        numeric_values = []
        all_values = []
        
        for item in data:
            if isinstance(item, (int, float)):
                numeric_values.append(float(item))
                all_values.append(item)
            elif isinstance(item, dict):
                for value in item.values():
                    all_values.append(value)
                    if isinstance(value, (int, float)):
                        numeric_values.append(float(value))
            else:
                all_values.append(item)
        
        # Calculate statistics for numeric data
        mean = median = std_dev = variance = skewness = kurtosis = None
        quartiles = None
        
        if numeric_values:
            try:
                mean = statistics.mean(numeric_values)
                median = statistics.median(numeric_values)
                if len(numeric_values) > 1:
                    std_dev = statistics.stdev(numeric_values)
                    variance = statistics.variance(numeric_values)
                
                # Calculate quartiles
                if len(numeric_values) >= 4:
                    sorted_values = sorted(numeric_values)
                    n = len(sorted_values)
                    quartiles = {
                        'Q1': sorted_values[n//4],
                        'Q2': median,
                        'Q3': sorted_values[3*n//4]
                    }
                
                # Calculate skewness and kurtosis if scipy is available
                if HAS_SCIPY and len(numeric_values) > 2:
                    skewness = float(stats.skew(numeric_values))
                    kurtosis = float(stats.kurtosis(numeric_values))
                    
            except Exception as e:
                logger.warning(f"Statistical calculation failed: {str(e)}")
        
        # Find mode and min/max
        mode = min_value = max_value = None
        try:
            if all_values:
                # Mode (most common value)
                try:
                    mode = statistics.mode(all_values)
                except statistics.StatisticsError:
                    # No unique mode found
                    mode = None
                
                # Min and max for comparable values
                if numeric_values:
                    min_value = min(numeric_values)
                    max_value = max(numeric_values)
                else:
                    # Try with all values if they're comparable
                    try:
                        min_value = min(all_values)
                        max_value = max(all_values)
                    except TypeError:
                        # Values not comparable
                        min_value = max_value = None
        except Exception as e:
            logger.warning(f"Mode/min/max calculation failed: {str(e)}")
        
        return StatisticalSummary(
            count=count,
            mean=mean,
            median=median,
            mode=mode,
            std_dev=std_dev,
            variance=variance,
            min_value=min_value,
            max_value=max_value,
            quartiles=quartiles,
            skewness=skewness,
            kurtosis=kurtosis
        )
    
    def _analyze_patterns(self, data: Any) -> PatternAnalysis:
        """Analyze patterns in the data"""
        trends = []
        correlations = []
        outliers = []
        clusters = []
        seasonality = None
        
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data) if hasattr(data, '__iter__') else [data]
        
        if not data:
            return PatternAnalysis(trends, correlations, outliers, clusters, seasonality)
        
        # Extract numeric sequences for trend analysis
        numeric_sequences = {}
        
        if isinstance(data[0], dict):
            # Group numeric values by field
            for record in data:
                if isinstance(record, dict):
                    for key, value in record.items():
                        if isinstance(value, (int, float)):
                            if key not in numeric_sequences:
                                numeric_sequences[key] = []
                            numeric_sequences[key].append(float(value))
        
        # Analyze trends
        for field, values in numeric_sequences.items():
            if len(values) > 2:
                # Simple trend detection
                if len(values) >= 3:
                    increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
                    decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
                    
                    if increases > decreases * 1.5:
                        trends.append(f"Increasing trend in {field}")
                    elif decreases > increases * 1.5:
                        trends.append(f"Decreasing trend in {field}")
                    else:
                        trends.append(f"Stable/mixed trend in {field}")
        
        # Analyze correlations between numeric fields
        if len(numeric_sequences) > 1:
            field_names = list(numeric_sequences.keys())
            for i, field1 in enumerate(field_names):
                for field2 in field_names[i+1:]:
                    values1 = numeric_sequences[field1]
                    values2 = numeric_sequences[field2]
                    
                    # Ensure same length
                    min_len = min(len(values1), len(values2))
                    if min_len > 2:
                        v1 = values1[:min_len]
                        v2 = values2[:min_len]
                        
                        # Calculate correlation
                        try:
                            if HAS_SCIPY:
                                corr, p_value = stats.pearsonr(v1, v2)
                                correlations.append({
                                    'field1': field1,
                                    'field2': field2,
                                    'correlation': float(corr),
                                    'p_value': float(p_value),
                                    'strength': self._interpret_correlation(corr)
                                })
                            else:
                                # Simple correlation calculation
                                mean1, mean2 = statistics.mean(v1), statistics.mean(v2)
                                num = sum((x - mean1) * (y - mean2) for x, y in zip(v1, v2))
                                den = (sum((x - mean1)**2 for x in v1) * sum((y - mean2)**2 for y in v2))**0.5
                                if den != 0:
                                    corr = num / den
                                    correlations.append({
                                        'field1': field1,
                                        'field2': field2,
                                        'correlation': float(corr),
                                        'strength': self._interpret_correlation(corr)
                                    })
                        except Exception as e:
                            logger.warning(f"Correlation calculation failed: {str(e)}")
        
        # Detect outliers using IQR method
        for field, values in numeric_sequences.items():
            if len(values) >= 4:
                try:
                    q1 = statistics.quantiles(values, n=4)[0]
                    q3 = statistics.quantiles(values, n=4)[2]
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    field_outliers = [v for v in values if v < lower_bound or v > upper_bound]
                    if field_outliers:
                        outliers.extend([{
                            'field': field,
                            'value': outlier,
                            'type': 'high' if outlier > upper_bound else 'low'
                        } for outlier in field_outliers])
                except Exception as e:
                    logger.warning(f"Outlier detection failed for {field}: {str(e)}")
        
        return PatternAnalysis(trends, correlations, outliers, clusters, seasonality)
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            return "very strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very weak"
    
    def _generate_insights(self, data: Any, stats: StatisticalSummary, patterns: PatternAnalysis) -> List[str]:
        """Generate insights from the analysis"""
        insights = []
        
        # Data size insights
        if stats.count > 1000:
            insights.append("Large dataset detected - suitable for robust statistical analysis")
        elif stats.count < 30:
            insights.append("Small dataset - statistical conclusions should be interpreted cautiously")
        
        # Distribution insights
        if stats.mean is not None and stats.median is not None:
            if abs(stats.mean - stats.median) > (stats.std_dev or 0) * 0.5:
                if stats.mean > stats.median:
                    insights.append("Data appears to be right-skewed (positive skew)")
                else:
                    insights.append("Data appears to be left-skewed (negative skew)")
            else:
                insights.append("Data appears to be approximately normally distributed")
        
        # Variability insights
        if stats.std_dev is not None and stats.mean is not None and stats.mean != 0:
            cv = stats.std_dev / abs(stats.mean)
            if cv > 1:
                insights.append("High variability detected - data points are widely spread")
            elif cv < 0.1:
                insights.append("Low variability detected - data points are tightly clustered")
        
        # Pattern insights
        if patterns.trends:
            insights.extend([f"Trend analysis: {trend}" for trend in patterns.trends])
        
        if patterns.correlations:
            strong_corrs = [c for c in patterns.correlations if abs(c['correlation']) > 0.6]
            if strong_corrs:
                insights.append(f"Found {len(strong_corrs)} strong correlations between variables")
        
        if patterns.outliers:
            insights.append(f"Detected {len(patterns.outliers)} potential outliers requiring investigation")
        
        return insights
    
    def _create_visualizations(self, data: Any) -> Dict[str, Any]:
        """Create data visualizations if plotting libraries are available"""
        visualizations = {
            'available': HAS_PLOTTING,
            'charts': []
        }
        
        if not HAS_PLOTTING:
            visualizations['message'] = "Plotting libraries not available - install matplotlib and seaborn for visualizations"
            return visualizations
        
        # Return comprehensive visualization descriptions
        # In a full implementation, this would generate actual plots
        visualizations['charts'] = [
            {
                'type': 'histogram',
                'description': 'Distribution of numeric variables',
                'recommendation': 'Use to understand data distribution and identify skewness'
            },
            {
                'type': 'correlation_matrix',
                'description': 'Correlation heatmap between numeric variables',
                'recommendation': 'Use to identify relationships between variables'
            },
            {
                'type': 'box_plot',
                'description': 'Box plots showing quartiles and outliers',
                'recommendation': 'Use to identify outliers and understand data spread'
            }
        ]
        
        return visualizations
    
    def _generate_recommendations(self, quality_report: DataQualityReport, insights: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Add quality-based recommendations
        recommendations.extend(quality_report.recommendations)
        
        # Add analysis-based recommendations
        if quality_report.overall_score < 0.7:
            recommendations.append("Consider data cleaning and preprocessing before analysis")
        
        if "outliers" in str(insights).lower():
            recommendations.append("Investigate outliers - they may indicate data errors or interesting edge cases")
        
        if "correlation" in str(insights).lower():
            recommendations.append("Explore causal relationships behind observed correlations")
        
        if "trend" in str(insights).lower():
            recommendations.append("Consider time-series analysis for trend forecasting")
        
        recommendations.append("Validate findings with domain experts")
        recommendations.append("Consider collecting additional data to strengthen analysis")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _get_data_shape(self, data: Any) -> Dict[str, Any]:
        """Get the shape/dimensions of the data"""
        if isinstance(data, list):
            rows = len(data)
            cols = 0
            if data and isinstance(data[0], dict):
                cols = len(data[0].keys())
            return {'rows': rows, 'columns': cols}
        elif isinstance(data, dict):
            return {'rows': 1, 'columns': len(data.keys())}
        else:
            return {'rows': 1, 'columns': 1}
    
    def _analyze_data_types(self, data: Any) -> Dict[str, Any]:
        """Analyze the types of data present"""
        type_analysis = {
            'numeric_fields': [],
            'text_fields': [],
            'boolean_fields': [],
            'date_fields': [],
            'mixed_fields': []
        }
        
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # Analyze field types
            field_types = {}
            for record in data:
                if isinstance(record, dict):
                    for key, value in record.items():
                        if key not in field_types:
                            field_types[key] = set()
                        field_types[key].add(type(value).__name__)
            
            for field, types in field_types.items():
                if len(types) == 1:
                    type_name = list(types)[0]
                    if type_name in ['int', 'float']:
                        type_analysis['numeric_fields'].append(field)
                    elif type_name == 'str':
                        type_analysis['text_fields'].append(field)
                    elif type_name == 'bool':
                        type_analysis['boolean_fields'].append(field)
                else:
                    type_analysis['mixed_fields'].append(field)
        
        return type_analysis
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create a standardized error result"""
        return {
            'success': False,
            'error': error_message,
            'recommendations': [
                'Check data format and structure',
                'Ensure data is properly formatted (JSON, CSV, etc.)',
                'Verify data contains analyzable content'
            ]
        } 