# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Scientific Computing Engine

Consolidated engine for mathematical computations, statistical analysis, and data science.
Provides high-precision numerical calculations to amplify LLM capabilities in STEM domains.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Union, Tuple
import json

logger = logging.getLogger(__name__)

# Import scientific computing libraries with graceful fallbacks
try:
    import scipy.stats as stats
    import scipy.optimize as optimize
    import scipy.integrate as integrate
    import scipy.linalg as linalg
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    SCIENTIFIC_LIBRARIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Scientific computing libraries not available: {e}")
    SCIENTIFIC_LIBRARIES_AVAILABLE = False
    stats = None
    optimize = None
    integrate = None
    linalg = None
    pd = None


class ScientificComputingEngine:
    """
    Consolidated computational engine for mathematics, statistics, and data analysis.
    
    Provides precise numerical computations for STEM applications where LLM reasoning
    alone is insufficient for accurate results.
    """
    
    def __init__(self):
        self.name = "Scientific Computing Engine"
        self.version = "1.0.0"
        self.supported_calculations = [
            # Mathematical computations
            "numerical_integration",
            "differential_equations", 
            "optimization",
            "linear_algebra",
            "fourier_transform",
            "polynomial_fitting",
            
            # Statistical analysis
            "hypothesis_testing",
            "regression_analysis", 
            "correlation_analysis",
            "distribution_analysis",
            "anova_test",
            "chi_square_test",
            
            # Data science
            "principal_component_analysis",
            "clustering_analysis",
            "outlier_detection",
            "time_series_analysis",
            "feature_selection",
            "model_validation"
        ]
    
    def numerical_integration(self, function_expr: str, bounds: Tuple[float, float], 
                            method: str = "quad") -> Dict[str, Any]:
        """
        Perform numerical integration using SciPy algorithms.
        
        Args:
            function_expr: String representation of function (e.g., "x**2 + 2*x")
            bounds: Tuple of (lower_bound, upper_bound)
            method: Integration method ("quad", "simpson", "trapz")
            
        Returns:
            Dict with integral value, error estimate, and computation details
        """
        try:
            logger.info(f"🔢 Computing numerical integration: {function_expr}")
            
            if not SCIENTIFIC_LIBRARIES_AVAILABLE:
                return {"error": "SciPy not available for numerical integration"}
            
            # Define function from string expression
            def func(x):
                return eval(function_expr, {"x": x, "np": np, "sin": np.sin, 
                           "cos": np.cos, "exp": np.exp, "log": np.log, "sqrt": np.sqrt})
            
            lower_bound, upper_bound = bounds
            
            if method == "quad":
                result, error = integrate.quad(func, lower_bound, upper_bound)
                method_info = "Adaptive quadrature (scipy.integrate.quad)"
            elif method == "simpson":
                x = np.linspace(lower_bound, upper_bound, 1001)  # Odd number for Simpson's rule
                y = func(x)
                result = integrate.simpson(y, x)
                error = None
                method_info = "Simpson's rule (scipy.integrate.simpson)"
            elif method == "trapz":
                x = np.linspace(lower_bound, upper_bound, 1001)
                y = func(x)
                result = integrate.trapz(y, x)
                error = None
                method_info = "Trapezoidal rule (scipy.integrate.trapz)"
            else:
                return {"error": f"Unknown integration method: {method}"}
            
            return {
                "integral_value": float(result),
                "estimated_error": float(error) if error else None,
                "function": function_expr,
                "bounds": bounds,
                "method": method_info,
                "computation_successful": True
            }
            
        except Exception as e:
            logger.error(f"❌ Numerical integration failed: {str(e)}")
            return {"error": f"Integration failed: {str(e)}"}
    
    def hypothesis_testing(self, data1: List[float], data2: List[float] = None, 
                          test_type: str = "ttest", alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform statistical hypothesis tests.
        
        Args:
            data1: Primary dataset
            data2: Secondary dataset (for two-sample tests)
            test_type: Type of test ("ttest", "wilcoxon", "ks_test", "shapiro")
            alpha: Significance level
            
        Returns:
            Dict with test statistic, p-value, and interpretation
        """
        try:
            logger.info(f"📊 Performing {test_type} hypothesis test")
            
            if not SCIENTIFIC_LIBRARIES_AVAILABLE:
                return {"error": "SciPy not available for statistical testing"}
            
            data1_array = np.array(data1)
            
            if test_type == "ttest":
                if data2 is None:
                    # One-sample t-test against zero
                    statistic, p_value = stats.ttest_1samp(data1_array, 0)
                    test_description = "One-sample t-test (H0: mean = 0)"
                else:
                    # Two-sample t-test
                    data2_array = np.array(data2)
                    statistic, p_value = stats.ttest_ind(data1_array, data2_array)
                    test_description = "Two-sample t-test (H0: means equal)"
                    
            elif test_type == "wilcoxon":
                if data2 is None:
                    # Wilcoxon signed-rank test
                    statistic, p_value = stats.wilcoxon(data1_array)
                    test_description = "Wilcoxon signed-rank test (H0: median = 0)"
                else:
                    # Mann-Whitney U test
                    data2_array = np.array(data2)
                    statistic, p_value = stats.mannwhitneyu(data1_array, data2_array)
                    test_description = "Mann-Whitney U test (H0: distributions equal)"
                    
            elif test_type == "ks_test":
                if data2 is None:
                    # Kolmogorov-Smirnov test against normal distribution
                    statistic, p_value = stats.kstest(data1_array, 'norm')
                    test_description = "Kolmogorov-Smirnov test (H0: data is normal)"
                else:
                    # Two-sample KS test
                    data2_array = np.array(data2)
                    statistic, p_value = stats.ks_2samp(data1_array, data2_array)
                    test_description = "Two-sample Kolmogorov-Smirnov test"
                    
            elif test_type == "shapiro":
                # Shapiro-Wilk test for normality
                statistic, p_value = stats.shapiro(data1_array)
                test_description = "Shapiro-Wilk test for normality (H0: data is normal)"
                
            else:
                return {"error": f"Unknown test type: {test_type}"}
            
            # Interpret results
            significant = p_value < alpha
            interpretation = (
                f"Reject null hypothesis (p < {alpha})" if significant 
                else f"Fail to reject null hypothesis (p >= {alpha})"
            )
            
            effect_size = None
            if test_type == "ttest" and data2 is not None:
                # Cohen's d for t-test
                pooled_std = np.sqrt(((len(data1_array)-1)*np.var(data1_array, ddof=1) + 
                                    (len(data2)-1)*np.var(data2, ddof=1)) / 
                                   (len(data1_array)+len(data2)-2))
                effect_size = (np.mean(data1_array) - np.mean(data2)) / pooled_std
            
            return {
                "test_statistic": float(statistic),
                "p_value": float(p_value),
                "alpha": alpha,
                "significant": significant,
                "interpretation": interpretation,
                "test_description": test_description,
                "effect_size": float(effect_size) if effect_size is not None else None,
                "sample_sizes": {"data1": len(data1_array), "data2": len(data2) if data2 else None}
            }
            
        except Exception as e:
            logger.error(f"❌ Hypothesis testing failed: {str(e)}")
            return {"error": f"Statistical test failed: {str(e)}"}
    
    def regression_analysis(self, x_data: List[List[float]], y_data: List[float], 
                           model_type: str = "linear") -> Dict[str, Any]:
        """
        Perform regression analysis with model validation.
        
        Args:
            x_data: Feature matrix (list of feature vectors)
            y_data: Target values
            model_type: Type of regression ("linear", "logistic", "random_forest")
            
        Returns:
            Dict with model metrics, coefficients, and predictions
        """
        try:
            logger.info(f"📈 Performing {model_type} regression analysis")
            
            if not SCIENTIFIC_LIBRARIES_AVAILABLE:
                return {"error": "Scikit-learn not available for regression analysis"}
            
            X = np.array(x_data)
            y = np.array(y_data)
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features for better performance
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Choose and fit model
            if model_type == "linear":
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)
                
                # Get coefficients and intercept
                coefficients = model.coef_.tolist()
                intercept = float(model.intercept_)
                feature_importance = None
                
            elif model_type == "logistic":
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train_scaled, y_train)
                
                coefficients = model.coef_[0].tolist()
                intercept = float(model.intercept_[0])
                feature_importance = None
                
            elif model_type == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                coefficients = None
                intercept = None
                feature_importance = model.feature_importances_.tolist()
                
            else:
                return {"error": f"Unknown model type: {model_type}"}
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Additional metrics for regression
            if model_type in ["linear", "random_forest"]:
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                train_mse = mean_squared_error(y_train, y_pred_train)
                test_mse = mean_squared_error(y_test, y_pred_test)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                additional_metrics = {
                    "train_mse": float(train_mse),
                    "test_mse": float(test_mse),
                    "train_mae": float(train_mae),
                    "test_mae": float(test_mae),
                    "train_rmse": float(np.sqrt(train_mse)),
                    "test_rmse": float(np.sqrt(test_mse))
                }
            else:
                # Classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                train_accuracy = accuracy_score(y_train, y_pred_train.round())
                test_accuracy = accuracy_score(y_test, y_pred_test.round())
                
                additional_metrics = {
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy)
                }
            
            return {
                "model_type": model_type,
                "train_score": float(train_score),
                "test_score": float(test_score),
                "coefficients": coefficients,
                "intercept": intercept,
                "feature_importance": feature_importance,
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                **additional_metrics,
                "overfitting_indicator": train_score - test_score > 0.1
            }
            
        except Exception as e:
            logger.error(f"❌ Regression analysis failed: {str(e)}")
            return {"error": f"Regression analysis failed: {str(e)}"}
    
    def principal_component_analysis(self, data: List[List[float]], 
                                   n_components: int = None) -> Dict[str, Any]:
        """
        Perform Principal Component Analysis for dimensionality reduction.
        
        Args:
            data: Feature matrix
            n_components: Number of components to keep (None for all)
            
        Returns:
            Dict with explained variance, components, and transformed data
        """
        try:
            logger.info("🔍 Performing Principal Component Analysis")
            
            if not SCIENTIFIC_LIBRARIES_AVAILABLE:
                return {"error": "Scikit-learn not available for PCA"}
            
            X = np.array(data)
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine number of components
            if n_components is None:
                n_components = min(X.shape[0], X.shape[1])
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            X_transformed = pca.fit_transform(X_scaled)
            
            # Calculate cumulative explained variance
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Find number of components for 95% variance
            components_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            return {
                "n_components": n_components,
                "original_dimensions": X.shape[1],
                "explained_variance_ratio": explained_variance_ratio.tolist(),
                "cumulative_explained_variance": cumulative_variance.tolist(),
                "components_for_95_variance": int(components_95),
                "principal_components": pca.components_.tolist(),
                "singular_values": pca.singular_values_.tolist(),
                "transformed_data": X_transformed.tolist(),
                "total_variance_explained": float(cumulative_variance[-1]),
                "dimensionality_reduction": f"{X.shape[1]} → {n_components} dimensions"
            }
            
        except Exception as e:
            logger.error(f"❌ PCA failed: {str(e)}")
            return {"error": f"PCA failed: {str(e)}"}
    
    def clustering_analysis(self, data: List[List[float]], n_clusters: int = 3, 
                          method: str = "kmeans") -> Dict[str, Any]:
        """
        Perform clustering analysis with quality metrics.
        
        Args:
            data: Feature matrix
            n_clusters: Number of clusters
            method: Clustering method ("kmeans", "hierarchical")
            
        Returns:
            Dict with cluster assignments, centroids, and quality metrics
        """
        try:
            logger.info(f"🔍 Performing {method} clustering analysis")
            
            if not SCIENTIFIC_LIBRARIES_AVAILABLE:
                return {"error": "Scikit-learn not available for clustering"}
            
            X = np.array(data)
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if method == "kmeans":
                # Perform K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
                inertia = kmeans.inertia_
                
            else:
                return {"error": f"Unknown clustering method: {method}"}
            
            # Calculate silhouette score
            if len(np.unique(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            else:
                silhouette_avg = 0.0
            
            # Calculate cluster statistics
            cluster_stats = []
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_data = X[cluster_mask]
                
                if len(cluster_data) > 0:
                    cluster_stats.append({
                        "cluster_id": i,
                        "size": int(np.sum(cluster_mask)),
                        "center": cluster_centers[i].tolist(),
                        "mean": np.mean(cluster_data, axis=0).tolist(),
                        "std": np.std(cluster_data, axis=0).tolist()
                    })
            
            return {
                "method": method,
                "n_clusters": n_clusters,
                "cluster_labels": cluster_labels.tolist(),
                "cluster_centers": cluster_centers.tolist(),
                "silhouette_score": float(silhouette_avg),
                "inertia": float(inertia) if method == "kmeans" else None,
                "cluster_statistics": cluster_stats,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "quality_assessment": (
                    "Excellent clustering" if silhouette_avg > 0.7 else
                    "Good clustering" if silhouette_avg > 0.5 else
                    "Moderate clustering" if silhouette_avg > 0.25 else
                    "Poor clustering"
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Clustering analysis failed: {str(e)}")
            return {"error": f"Clustering analysis failed: {str(e)}"} 