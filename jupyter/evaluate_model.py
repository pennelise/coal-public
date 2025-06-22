import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from sklearn import metrics
from scipy import stats

def clean_data(df, model):
    return df.copy().dropna(subset=model.X_columns+[model.y_column])

def plot_by_state(pred_df, model, state_col, predicted_column=None, y_column=None):
    y_column = y_column if y_column is not None else model.y_column
    predicted_column = predicted_column if predicted_column is not None else model.predicted_column
    for state in pred_df[state_col].unique():
        subset = pred_df[pred_df[state_col]==state].dropna(subset=[y_column, predicted_column])
        if subset.empty:
            continue
        plt.plot(subset[y_column], subset[predicted_column], 'o', label=state, alpha=0.5)
    maxval = np.max([np.max(pred_df[y_column]), np.max(pred_df[predicted_column])])
    plt.plot([0, maxval], [0, maxval], 'k--', label='1:1')
    plt.xlabel('Actual emissions (metric tons)')
    plt.ylabel('Predicted emissions (metric tons)')
    plt.legend()
    plt.show()

def plot_2d_model(pred_df, model):
    if len(model.X_columns)>1:
        print('Skipping 2d model plot. Only one X column allowed for 2D plot')
        return
    fig, _ = plt.subplots(layout='constrained')
    for cluster in model.models.keys():
        subset = pred_df[model.cluster_data(pred_df)==cluster]
        if subset.empty:
            continue
        dots = plt.scatter(subset[model.X_columns], subset[model.y_column], alpha=0.5)
        plt.plot(subset[model.X_columns], subset[model.predicted_column], label=cluster, color=dots.get_facecolor())
    plt.xlabel('Coal Production (metric tons)')
    plt.ylabel('Emissions (metric tons)')
    fig.legend(loc="outside right upper", ncol=1)
    plt.show()

def format_2_sig_figs(i):
    return f'{float(f"{i:.2g}"):g}'

def aggregate_metrics(pred_df, model, true_column=None, predicted_column=None):
    true_column = true_column if true_column is not None else model.y_column
    predicted_column = predicted_column if predicted_column is not None else model.predicted_column
    y_pred = pred_df[model.predicted_column]
    return pd.DataFrame({'MAE': metrics.mean_absolute_error(pred_df[true_column], y_pred),
                          'RMSE': np.sqrt(metrics.mean_squared_error(pred_df[true_column], y_pred)),
                          'pearson r': stats.pearsonr(pred_df[true_column], y_pred)[0], 
                          'mean bias': np.mean(y_pred-pred_df[true_column]),
                          'total percent bias' : (np.sum(y_pred)-np.sum(pred_df[true_column]))/np.sum(pred_df[true_column])*100,},
                          index=[0]
                        )

def metrics_by_cluster(pred_df, model, true_column=None, predicted_column=None):
    true_column = true_column if true_column is not None else model.y_column
    predicted_column = predicted_column if predicted_column is not None else model.predicted_column
    cluster_metrics= pred_df.groupby(model.cluster_data(pred_df)).apply(lambda x: pd.Series({
                                        'MAE': metrics.mean_absolute_error(x[true_column], x[predicted_column]),
                                        'RMSE': np.sqrt(metrics.mean_squared_error(x[true_column], x[predicted_column])),
                                        'pearson r': stats.pearsonr(x[true_column], x[predicted_column])[0] if len(x)>1 else np.nan, 
                                        'mean bias': np.mean(x[predicted_column]-x[true_column]),
                                        'total percent bias' : (np.sum(x[predicted_column])-np.sum(x[true_column]))/np.sum(x[true_column])*100,})
                                        )
    model_EFs = pd.DataFrame({cluster: model.models[cluster].coef_[0] for cluster in model.models.keys()}, index=['slope']).T
    model_intercepts = pd.DataFrame({cluster: model.models[cluster].intercept_ for cluster in model.models.keys()}, index=['intercept']).T
    return pd.concat([cluster_metrics, model_EFs, model_intercepts], axis=1)

def metrics_by_state(pred_df, model, state_column='STATE', true_column=None, predicted_column=None):
    true_column = true_column if true_column is not None else model.y_column
    predicted_column = predicted_column if predicted_column is not None else model.predicted_column
    return pred_df.groupby(state_column).apply(lambda x: pd.Series({
                                    'MAE': metrics.mean_absolute_error(x[true_column], x[predicted_column]),
                                    'RMSE': np.sqrt(metrics.mean_squared_error(x[true_column], x[predicted_column])),
                                    'pearson r': stats.pearsonr(x[true_column], x[predicted_column])[0] if len(x)>1 else np.nan, 
                                    'mean bias': np.mean(x[predicted_column] - x[true_column]),
                                    'total percent bias' : (np.sum(x[predicted_column])-np.sum(x[true_column]))/np.sum(x[true_column])*100,})
                                    )


def evaluate_model(model, test_df, y_true_col=None, state_column='STATE'):
    y_true_col = y_true_col if y_true_col is not None else model.y_column
    pred_df = model.predict(test_df).dropna(subset=[model.predicted_column, y_true_col])
    plot_by_state(pred_df, model, 
                  state_col=state_column, 
                  predicted_column=model.predicted_column, 
                  y_column=y_true_col)
    plot_2d_model(pred_df, model)
    print('Aggregate metrics')
    display(aggregate_metrics(pred_df, model, true_column=y_true_col).applymap(lambda i: format_2_sig_figs(i)))
    print('Metrics by cluster')
    display(metrics_by_cluster(pred_df, model, true_column=y_true_col).applymap(lambda i: format_2_sig_figs(i)))
    if (model.cluster_columns is None) or (state_column not in model.cluster_columns):
        display('Metrics by state')
        display(metrics_by_state(pred_df, model, true_column=y_true_col).applymap(lambda i: format_2_sig_figs(i)))
    return aggregate_metrics, metrics_by_cluster, metrics_by_state

def get_residuals(pred_df, model):
    return pred_df[model.y_column] - pred_df[model.predicted_column]

# def get_model_params(model):
#     df = pd.DataFrame()
#     for cluster in model.models.keys(): 
#         df = pd.concat([df, pd.DataFrame({'cluster': cluster, 
#                                           'slope': model.models[cluster].coef_, 
#                                           'intercept': model.models[cluster].intercept_})])
#     return df