import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class DBSCANClusterer:
    def __init__(self, file_path):
        """
        Initialize the DBSCAN clusterer with the dataset
        
        Parameters:
        - file_path: Path to the CSV file
        """
        # Read the CSV file
        self.data = pd.read_csv(file_path)
        self.original_data = self.data.copy()
    
    def prepare_data(self, columns):
        """
        Prepare data for clustering by selecting and scaling specified columns
        
        Parameters:
        - columns: List of column names to use for clustering
        
        Returns:
        - Scaled data ready for DBSCAN
        """
        # Validate columns
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column {col} not found in the dataset")
        
        # Select columns and scale
        X = self.data[columns]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, X
    
    def cluster(self, columns, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering
        
        Parameters:
        - columns: List of column names to use for clustering
        - eps: Maximum distance between two samples to be considered in the same neighborhood
        - min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point
        
        Returns:
        - Dataframe with clustering results
        """
        # Prepare data
        X_scaled, X_original = self.prepare_data(columns)
        
        # Perform DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        
        # Add clusters to the dataframe
        result_df = X_original.copy()
        result_df['Cluster'] = clusters
        
        # Clustering summary
        self._print_clustering_summary(clusters)
        
        return result_df
    
    def _print_clustering_summary(self, clusters):
        """
        Print summary of clustering results
        
        Parameters:
        - clusters: Array of cluster labels
        """
        # Count unique clusters (excluding noise points)
        unique_clusters = set(clusters)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = list(clusters).count(-1)
        
        print("\n--- DBSCAN Clustering Summary ---")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print("\nCluster Distribution:")
        print(pd.Series(clusters).value_counts())
    
    def visualize_clusters(self, columns, eps=0.5, min_samples=5):
        """
        Visualize clustering results
        
        Parameters:
        - columns: List of two column names to use for visualization
        - eps, min_samples: DBSCAN parameters
        
        Returns:
        - Matplotlib figure
        """
        # Ensure only two columns are used for visualization
        if len(columns) != 2:
            raise ValueError("Visualization requires exactly 2 columns")
        
        # Perform clustering
        clustered_df = self.cluster(columns, eps, min_samples)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=columns[0], 
            y=columns[1], 
            hue='Cluster', 
            data=clustered_df, 
            palette='viridis', 
            legend='full'
        )
        
        plt.title(f'DBSCAN Clustering: {columns[0]} vs {columns[1]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        
        return plt

# Example usage
def main():
    # Path to the CSV file
    file_path = 'data_cuoiky.csv'
    
    # Create clusterer
    clusterer = DBSCANClusterer(file_path)
    
    # Example clustering scenarios
    clustering_scenarios = [
        {
            'columns': ['Quiz_count', 'GK'],
            'eps': 0.5,
            'min_samples': 3
        },
        {
            'columns': ['video_time', 'Diem_BT'],
            'eps': 0.4,
            'min_samples': 4
        },
        {
            'columns': ['Quiz_avg', 'Quiz_time_avg'],
            'eps': 0.6,
            'min_samples': 3
        }
    ]
    
    # Run and visualize each scenario
    for scenario in clustering_scenarios:
        print(f"\n=== Clustering {scenario['columns']} ===")
        try:
            # Perform clustering and visualize
            plt = clusterer.visualize_clusters(
                columns=scenario['columns'],
                eps=scenario['eps'],
                min_samples=scenario['min_samples']
            )
            plt.show()
        except Exception as e:
            print(f"Error in clustering: {e}")

# Uncomment to run
# main()

# Additional utility method for parameter tuning
def parameter_grid_search(file_path, columns, eps_range=None, min_samples_range=None):
    """
    Perform grid search to find optimal DBSCAN parameters
    
    Parameters:
    - file_path: Path to the CSV file
    - columns: Columns to use for clustering
    - eps_range: Range of eps values to test
    - min_samples_range: Range of min_samples values to test
    
    Returns:
    - DataFrame with parameter combinations and clustering results
    """
    # Default ranges if not provided
    eps_range = eps_range or np.linspace(0.1, 1.0, 10)
    min_samples_range = min_samples_range or range(2, 10)
    
    # Prepare results storage
    results = []
    
    # Create clusterer
    clusterer = DBSCANClusterer(file_path)
    
    # Prepare data
    X_scaled, _ = clusterer.prepare_data(columns)
    
    # Grid search
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_scaled)
            
            # Count clusters and noise
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            
            # Store results
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df