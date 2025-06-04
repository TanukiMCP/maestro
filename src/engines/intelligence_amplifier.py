"""
Intelligence Amplification Computational Engine

Provides computational amplification through network analysis, optimization algorithms,
and cognitive load analysis using NetworkX, SciPy, and other computational libraries.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Union
import json

logger = logging.getLogger(__name__)

# Import computational libraries with graceful fallbacks
try:
    import networkx as nx
    import scipy.optimize as opt
    import scipy.stats as stats
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    INTELLIGENCE_LIBRARIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Intelligence computation libraries not available: {e}")
    INTELLIGENCE_LIBRARIES_AVAILABLE = False
    nx = None
    opt = None
    stats = None


class IntelligenceAmplificationEngine:
    """
    Computational engine for intelligence amplification through network analysis,
    optimization, and cognitive modeling algorithms.
    
    LLM calls these methods with parameters to get actual computational results.
    """
    
    def __init__(self):
        self.name = "Intelligence Amplification Engine"
        self.version = "1.0.0"
        self.supported_calculations = [
            "knowledge_network_analysis",
            "cognitive_load_optimization", 
            "information_flow_analysis",
            "concept_clustering",
            "reasoning_path_optimization",
            "attention_allocation_analysis",
            "memory_consolidation_modeling"
        ]
    
    def analyze_knowledge_network(self, concepts: List[str], 
                                 relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze knowledge networks using graph theory algorithms.
        
        Args:
            concepts: List of concept nodes
            relationships: List of relationship edges with weights
            
        Returns:
            Dict with network metrics, centrality measures, and insights
        """
        try:
            logger.info("🧠 Computing knowledge network analysis...")
            
            if not INTELLIGENCE_LIBRARIES_AVAILABLE:
                return {"error": "NetworkX and related libraries not available"}
            
            # Create knowledge graph
            G = nx.DiGraph()
            
            # Add concept nodes
            for concept in concepts:
                G.add_node(concept)
            
            # Add relationship edges
            for rel in relationships:
                if 'source' in rel and 'target' in rel:
                    weight = rel.get('weight', 1.0)
                    G.add_edge(rel['source'], rel['target'], weight=weight)
            
            # Compute network metrics
            betweenness_centrality = nx.betweenness_centrality(G)
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            pagerank = nx.pagerank(G)
            
            # Find strongly connected components
            scc = list(nx.strongly_connected_components(G))
            
            # Calculate clustering coefficient
            clustering = nx.clustering(G.to_undirected())
            
            # Find shortest paths between all concept pairs
            shortest_paths = {}
            for source in concepts:
                for target in concepts:
                    if source != target and nx.has_path(G, source, target):
                        path_length = nx.shortest_path_length(G, source, target)
                        shortest_paths[f"{source} -> {target}"] = path_length
            
            # Identify central concepts (top 5 by betweenness centrality)
            central_concepts = sorted(betweenness_centrality.items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
            
            result = {
                "network_size": {
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                    "density": nx.density(G)
                },
                "centrality_measures": {
                    "betweenness": betweenness_centrality,
                    "eigenvector": eigenvector_centrality,
                    "pagerank": pagerank
                },
                "clustering_coefficients": clustering,
                "strongly_connected_components": len(scc),
                "central_concepts": [{"concept": concept, "centrality": score} 
                                   for concept, score in central_concepts],
                "shortest_paths": shortest_paths,
                "average_path_length": np.mean(list(shortest_paths.values())) if shortest_paths else 0,
                "computation_method": "NetworkX graph analysis algorithms"
            }
            
            logger.info(f"✅ Knowledge network analysis complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return result
            
        except Exception as e:
            logger.error(f"❌ Knowledge network analysis failed: {str(e)}")
            return {"error": f"Calculation failed: {str(e)}"}
    
    def optimize_cognitive_load(self, tasks: List[Dict[str, Any]], 
                               constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize cognitive load distribution using optimization algorithms.
        
        Args:
            tasks: List of tasks with complexity and priority metrics
            constraints: Resource and time constraints
            
        Returns:
            Dict with optimal task allocation and load distribution
        """
        try:
            logger.info("🧠 Computing cognitive load optimization...")
            
            if not INTELLIGENCE_LIBRARIES_AVAILABLE:
                return {"error": "SciPy optimization libraries not available"}
            
            # Extract task parameters
            n_tasks = len(tasks)
            complexities = np.array([task.get('complexity', 1.0) for task in tasks])
            priorities = np.array([task.get('priority', 1.0) for task in tasks])
            durations = np.array([task.get('duration', 1.0) for task in tasks])
            
            # Constraint parameters
            max_load = constraints.get('max_cognitive_load', 10.0)
            max_time = constraints.get('max_time', 8.0)
            
            # Objective function: maximize priority while minimizing cognitive load
            def objective(x):
                # x[i] = 1 if task i is selected, 0 otherwise
                total_priority = np.sum(x * priorities)
                total_load = np.sum(x * complexities)
                # Maximize priority, minimize load (negative for minimization)
                return -(total_priority - 0.1 * total_load)
            
            # Constraints
            constraints_list = [
                # Cognitive load constraint
                {'type': 'ineq', 'fun': lambda x: max_load - np.sum(x * complexities)},
                # Time constraint  
                {'type': 'ineq', 'fun': lambda x: max_time - np.sum(x * durations)},
            ]
            
            # Bounds: each task either selected (1) or not (0)
            bounds = [(0, 1) for _ in range(n_tasks)]
            
            # Initial guess
            x0 = np.ones(n_tasks) * 0.5
            
            # Solve optimization problem
            result_opt = opt.minimize(objective, x0, method='SLSQP', 
                                    bounds=bounds, constraints=constraints_list)
            
            # Round to binary selection
            selection = np.round(result_opt.x)
            
            # Calculate metrics for selected tasks
            selected_tasks = [i for i, selected in enumerate(selection) if selected == 1]
            total_complexity = np.sum(selection * complexities)
            total_priority = np.sum(selection * priorities)
            total_duration = np.sum(selection * durations)
            
            # Load distribution analysis
            load_per_hour = total_complexity / total_duration if total_duration > 0 else 0
            efficiency_score = total_priority / total_complexity if total_complexity > 0 else 0
            
            result = {
                "optimization_result": {
                    "success": result_opt.success,
                    "optimal_value": float(-result_opt.fun),
                    "iterations": result_opt.nit
                },
                "task_selection": {
                    "selected_tasks": selected_tasks,
                    "selection_vector": selection.tolist(),
                    "total_tasks_selected": int(np.sum(selection))
                },
                "load_metrics": {
                    "total_cognitive_load": float(total_complexity),
                    "total_priority_value": float(total_priority),
                    "total_duration": float(total_duration),
                    "load_per_hour": float(load_per_hour),
                    "efficiency_score": float(efficiency_score)
                },
                "constraint_satisfaction": {
                    "load_constraint_satisfied": bool(total_complexity <= max_load),
                    "time_constraint_satisfied": bool(total_duration <= max_time),
                    "load_utilization": float(total_complexity / max_load),
                    "time_utilization": float(total_duration / max_time)
                },
                "computation_method": "SciPy SLSQP constrained optimization"
            }
            
            logger.info(f"✅ Cognitive load optimization complete: {len(selected_tasks)} tasks selected")
            return result
                
        except Exception as e:
            logger.error(f"❌ Cognitive load optimization failed: {str(e)}")
            return {"error": f"Calculation failed: {str(e)}"}
    
    def analyze_concept_clustering(self, concept_features: List[List[float]], 
                                  concept_names: List[str], 
                                  n_clusters: int = 3) -> Dict[str, Any]:
        """
        Perform concept clustering using machine learning algorithms.
        
        Args:
            concept_features: Feature vectors for each concept
            concept_names: Names corresponding to each concept
            n_clusters: Number of clusters to form
            
        Returns:
            Dict with cluster assignments and analysis
        """
        try:
            logger.info("🧠 Computing concept clustering analysis...")
            
            if not INTELLIGENCE_LIBRARIES_AVAILABLE:
                return {"error": "Scikit-learn clustering libraries not available"}
            
            # Convert to numpy array
            X = np.array(concept_features)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate cluster statistics
            cluster_centers = kmeans.cluster_centers_
            inertia = kmeans.inertia_
            
            # Analyze clusters
            clusters = {}
            for i in range(n_clusters):
                cluster_indices = np.where(cluster_labels == i)[0]
                cluster_concepts = [concept_names[idx] for idx in cluster_indices]
                cluster_center = cluster_centers[i]
                
                # Calculate within-cluster variance
                cluster_points = X_scaled[cluster_indices]
                within_cluster_variance = np.mean(np.sum((cluster_points - cluster_center) ** 2, axis=1))
                
                clusters[f"cluster_{i}"] = {
                    "concepts": cluster_concepts,
                    "size": len(cluster_concepts),
                    "center_coordinates": cluster_center.tolist(),
                    "within_cluster_variance": float(within_cluster_variance)
                }
            
            # Calculate silhouette score approximation
            silhouette_approx = self._calculate_silhouette_approximation(X_scaled, cluster_labels)
            
            result = {
                "clustering_summary": {
                    "n_clusters": n_clusters,
                    "n_concepts": len(concept_names),
                    "total_inertia": float(inertia),
                    "silhouette_score_approx": float(silhouette_approx)
                },
                "clusters": clusters,
                "cluster_assignments": {concept_names[i]: int(cluster_labels[i]) 
                                      for i in range(len(concept_names))},
                "feature_scaling": {
                    "method": "StandardScaler (zero mean, unit variance)",
                    "original_feature_means": scaler.mean_.tolist(),
                    "original_feature_stds": scaler.scale_.tolist()
                },
                "computation_method": "K-means clustering with standardized features"
            }
            
            logger.info(f"✅ Concept clustering complete: {n_clusters} clusters formed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Concept clustering failed: {str(e)}")
            return {"error": f"Calculation failed: {str(e)}"}
    
    def _calculate_silhouette_approximation(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate an approximation of silhouette score."""
        try:
            n_samples = X.shape[0]
            if n_samples < 2:
                return 0.0
            
            silhouette_scores = []
            
            for i in range(n_samples):
                # Distance to points in same cluster
                same_cluster_mask = labels == labels[i]
                same_cluster_distances = np.linalg.norm(X[same_cluster_mask] - X[i], axis=1)
                a_i = np.mean(same_cluster_distances[same_cluster_distances > 0])  # Exclude self
                
                # Distance to points in nearest other cluster
                other_clusters = np.unique(labels[labels != labels[i]])
                if len(other_clusters) == 0:
                    silhouette_scores.append(0.0)
                    continue
                
                min_other_distance = float('inf')
                for cluster in other_clusters:
                    other_cluster_mask = labels == cluster
                    other_cluster_distances = np.linalg.norm(X[other_cluster_mask] - X[i], axis=1)
                    avg_distance = np.mean(other_cluster_distances)
                    min_other_distance = min(min_other_distance, avg_distance)
                
                b_i = min_other_distance
                
                # Silhouette score for point i
                if max(a_i, b_i) > 0:
                    s_i = (b_i - a_i) / max(a_i, b_i)
                else:
                    s_i = 0.0
                    
                silhouette_scores.append(s_i)
            
            return np.mean(silhouette_scores)
            
        except Exception:
            return 0.0 