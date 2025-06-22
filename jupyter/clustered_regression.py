import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class ClusteredRegression:
    def __init__(self, X_columns, y_column, cluster_columns=None, fit_intercept=False, n_bootstraps=1000):
        self.X_columns = X_columns
        self.y_column = y_column
        self.cluster_columns = cluster_columns 
        self.fit_intercept = fit_intercept
        self.n_boostraps = n_bootstraps
        self.predicted_column = 'predicted_emissions'
        self.models = None
        self.standard_errors = None
        self.bootstrap_se = None

    def __check_data(self, df):
        assert not np.any(df[self.X_columns].isnull()), f'{self.X_columns} contain NaNs'
        assert not np.any(df[self.y_column].isnull()), f'{self.y_column} contains NaNs'
        return df
    
    def cluster_data(self, df): 
        df = self.__check_data(df)
        if self.cluster_columns is None:
            clusters = np.full(df.shape[0], 'all')
        else:
            clusters = df[self.cluster_columns].values.squeeze()
        return clusters
    
    def fit(self, df):
        df = self.__check_data(df)
        clusters = self.cluster_data(df)
        self.models = {}
        for cluster in np.unique(clusters):
            subset = df[clusters==cluster]
            if subset.empty:
                continue
            self.models[cluster] = self._fit_single_cluster(subset)
        return self.models

    def _fit_single_cluster(self, df_single_cluster):
        """
        Fit a linear regression on a single cluster's data.
        """
        if len(self.X_columns)==1:
            X = df_single_cluster[self.X_columns].values.reshape(-1, 1)
        else:
            X = df_single_cluster[self.X_columns].values
        y = df_single_cluster[self.y_column].values
        
        reg = LinearRegression(fit_intercept=self.fit_intercept)
        reg.fit(X, y)
        
        return reg
    
    def get_params(self):
        df = pd.DataFrame()
        for cluster in self.models.keys(): 
            df = pd.concat([df, pd.DataFrame({'cluster': cluster, 
                                              'slope': self.models[cluster].coef_, 
                                              'intercept': self.models[cluster].intercept_})])
        return df.set_index('cluster')

    def compute_se(self, df, method='bootstrap'):
        """
        Compute standard errors for each cluster using either:
          - 'direct': direct calculation (get_se)
          - 'bootstrap': via bootstrapping
        """
        df = self.__check_data(df)
        clusters = self.cluster_data(df)
        
        se_results = {}
        
        for cluster in np.unique(clusters):
            subset = df[clusters == cluster]
            
            if subset.empty or cluster not in self.models: # Skip if empty or no fitted model
                continue
            
            if method == 'direct':
                se_values = self._get_direct_se_for_cluster(subset, cluster)
                se_results[cluster] = se_values
            
            elif method == 'bootstrap':
                # Compute bootstrap SE for each predictor in cluster
                se_values = self._get_bootstrap_se_for_cluster(subset)
                se_results[cluster] = se_values
            else:
                raise ValueError("Unknown method. Choose either 'direct' or 'bootstrap'.")
        
        se_results = pd.DataFrame(se_results).T.rename_axis('cluster')

        if method == 'direct':
            self.standard_errors = se_results
        else:
            self.bootstrap_se = se_results
        
        return se_results
    
    def _get_direct_se_for_cluster(self, subset, cluster):
        """
        Not implemented.
        """
        raise NotImplementedError
        n_intercepts = 1 if self.fit_intercept else 0
        n_parameters = len(self.X_columns) + n_intercepts

    def _get_bootstrap_se_for_cluster(self, subset):
        """
        Helper function: compute bootstrap SE for each predictor in the cluster.
        """
        n = len(subset)
        
        bootstrap_coefs = []
        
        for _ in range(self.n_boostraps):
            sample_indices = np.random.choice(n, size=n, replace=True) # sample indices WITH replacement
            df_boot = subset.iloc[sample_indices]
            
            reg = self._fit_single_cluster(df_boot) # fit a new model
            coef = np.append(reg.coef_, reg.intercept_) if self.fit_intercept else reg.coef_
            bootstrap_coefs.append(coef)
        
        bootstrap_coefs = np.array(bootstrap_coefs)  # shape: (n_bootstraps, n_features)
        se_array = np.std(bootstrap_coefs, axis=0)
        
        return se_array
    
    def predict(self, df):
        df = self.__check_data(df)
        clusters = self.cluster_data(df)
        for cluster in np.unique(clusters):
            subset = df[clusters==cluster]
            if subset.empty or cluster not in self.models:
                continue
            if len(self.X_columns)==1:
                X = subset[self.X_columns].values.reshape(-1, 1)
            else:
                X = subset[self.X_columns].values
            subset[self.predicted_column] = self.models[cluster].predict(X)
            df.loc[subset.index, self.predicted_column] = subset[self.predicted_column]
        return df