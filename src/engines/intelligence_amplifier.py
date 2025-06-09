# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Intelligence Amplification Computational Engine

Provides computational amplification through network analysis, optimization algorithms,
and cognitive load analysis using NetworkX, SciPy, and other computational libraries.

This is a backend-only, headless engine designed to be called by external agentic IDEs.
No UI, no LLM client logic, just pure computational intelligence amplification.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Union, Optional, TypeVar, Tuple
import json
import asyncio
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
NetworkNode = TypeVar('NetworkNode')
NetworkEdge = TypeVar('NetworkEdge')

@dataclass
class ComputationResult:
    """Base class for computation results"""
    status: str
    timestamp: datetime
    computation_type: str
    result: Dict[str, Any]
    error: Optional[str] = None

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
    
    This is a backend-only engine that provides real computational results.
    No UI, no LLM client logic, just pure computational intelligence amplification.
    """
    
    def __init__(self):
        self.name = "Intelligence Amplification Engine"
        self.version = "2.0.0"
        self._validate_dependencies()
        
    def _validate_dependencies(self) -> None:
        """Validate that required computational libraries are available"""
        if not INTELLIGENCE_LIBRARIES_AVAILABLE:
            logger.error("Required computational libraries not available")
            raise ImportError(
                "Intelligence Amplification Engine requires: networkx, scipy, scikit-learn"
            )
    
    async def analyze_knowledge_network(
        self, 
        concepts: List[str],
        relationships: List[Dict[str, Any]]
    ) -> ComputationResult:
        """
        Analyze knowledge networks using graph theory algorithms.
        
        Args:
            concepts: List of concept nodes
            relationships: List of relationship edges with weights
            
        Returns:
            ComputationResult with network metrics, centrality measures, and insights
        """
        try:
            logger.info("🧠 Computing knowledge network analysis...")
            
            # Create knowledge graph
            G = nx.DiGraph()
            
            # Add concept nodes
            for concept in concepts:
                G.add_node(concept)
            
            # Add relationship edges with validation
            for rel in relationships:
                if not {'source', 'target'}.issubset(rel.keys()):
                    raise ValueError("Relationships must have 'source' and 'target' fields")
                weight = float(rel.get('weight', 1.0))
                G.add_edge(rel['source'], rel['target'], weight=weight)
            
            # Run computations asynchronously
            loop = asyncio.get_event_loop()
            metrics = await asyncio.gather(
                loop.run_in_executor(None, nx.betweenness_centrality, G),
                loop.run_in_executor(None, nx.eigenvector_centrality, G, 1000),
                loop.run_in_executor(None, nx.pagerank, G),
                loop.run_in_executor(None, lambda: list(nx.strongly_connected_components(G))),
                loop.run_in_executor(None, lambda: nx.clustering(G.to_undirected()))
            )
            
            betweenness_centrality, eigenvector_centrality, pagerank, scc, clustering = metrics
            
            # Calculate shortest paths asynchronously
            shortest_paths = {}
            path_tasks = []
            for source in concepts:
                for target in concepts:
                    if source != target:
                        task = loop.run_in_executor(
                            None,
                            lambda s=source, t=target: self._compute_shortest_path(G, s, t)
                        )
                        path_tasks.append(task)
            
            path_results = await asyncio.gather(*path_tasks)
            for result in path_results:
                if result:  # Filter out None results from unreachable paths
                    source, target, length = result
                    shortest_paths[f"{source} -> {target}"] = length
            
            # Identify central concepts
            central_concepts = sorted(
                betweenness_centrality.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Compile results
            result = {
                "network_size": len(G),
                "edge_count": G.number_of_edges(),
                "density": nx.density(G),
                "strongly_connected_components": len(scc),
                "average_clustering": np.mean(list(clustering.values())),
                "central_concepts": [
                    {
                        "concept": concept,
                        "centrality": score,
                        "eigenvector_centrality": eigenvector_centrality[concept],
                        "pagerank": pagerank[concept]
                    }
                    for concept, score in central_concepts
                ],
                "shortest_paths": shortest_paths
            }
            
            return ComputationResult(
                status="success",
                timestamp=datetime.now(),
                computation_type="knowledge_network_analysis",
                result=result
            )
            
        except Exception as e:
            logger.error(f"Knowledge network analysis failed: {e}")
            return ComputationResult(
                status="error",
                timestamp=datetime.now(),
                computation_type="knowledge_network_analysis",
                result={},
                error=str(e)
            )
    
    def _compute_shortest_path(
        self,
        G: nx.DiGraph,
        source: str,
        target: str
    ) -> Optional[Tuple[str, str, float]]:
        """Helper method to compute shortest path between two nodes"""
        try:
            if nx.has_path(G, source, target):
                length = nx.shortest_path_length(G, source, target)
                return (source, target, length)
        except nx.NetworkXError:
            pass
        return None
    
    async def analyze_concept_clustering(
        self,
        concept_features: List[List[float]],
        concept_names: List[str],
        n_clusters: int = 3
    ) -> ComputationResult:
        """
        Perform concept clustering using machine learning algorithms.
        
        Args:
            concept_features: Feature vectors for each concept
            concept_names: Names corresponding to each concept
            n_clusters: Number of clusters to form
            
        Returns:
            ComputationResult with cluster assignments and analysis
        """
        try:
            logger.info("🧠 Computing concept clustering analysis...")
            
            # Input validation
            if len(concept_features) != len(concept_names):
                raise ValueError("Number of features must match number of concept names")
            if n_clusters < 2:
                raise ValueError("Number of clusters must be at least 2")
            if n_clusters > len(concept_features):
                raise ValueError("Number of clusters cannot exceed number of concepts")
            
            # Convert to numpy array
            X = np.array(concept_features)
            
            # Run clustering asynchronously
            loop = asyncio.get_event_loop()
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = await loop.run_in_executor(None, scaler.fit_transform, X)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = await loop.run_in_executor(None, kmeans.fit_predict, X_scaled)
            
            # Calculate cluster statistics
            cluster_centers = kmeans.cluster_centers_
            inertia = kmeans.inertia_
            
            # Analyze clusters
            clusters = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_concepts = [
                    concept_names[j] 
                    for j in range(len(concept_names)) 
                    if cluster_labels[j] == i
                ]
                
                # Calculate cluster statistics
                cluster_features = X_scaled[cluster_mask]
                cluster_stats = {
                    "size": int(np.sum(cluster_mask)),
                    "concepts": cluster_concepts,
                    "center": cluster_centers[i].tolist(),
                    "variance": float(np.var(cluster_features, axis=0).mean()),
                    "max_distance": float(
                        np.max(
                            np.linalg.norm(
                                cluster_features - cluster_centers[i],
                                axis=1
                            )
                        )
                    )
                }
                clusters[f"cluster_{i}"] = cluster_stats
            
            # Compute silhouette score for cluster quality
            silhouette = float(
                await loop.run_in_executor(
                    None,
                    lambda: stats.silhouette_score(X_scaled, cluster_labels)
                )
            )
            
            result = {
                "n_clusters": n_clusters,
                "total_inertia": float(inertia),
                "silhouette_score": silhouette,
                "clusters": clusters,
                "cluster_labels": cluster_labels.tolist()
            }
            
            return ComputationResult(
                status="success",
                timestamp=datetime.now(),
                computation_type="concept_clustering",
                result=result
            )
            
        except Exception as e:
            logger.error(f"Concept clustering analysis failed: {e}")
            return ComputationResult(
                status="error",
                timestamp=datetime.now(),
                computation_type="concept_clustering",
                result={},
                error=str(e)
            )
