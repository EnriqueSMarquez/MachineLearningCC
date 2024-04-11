import argparse
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import yaml
from model import CustomKMeans

parser = argparse.ArgumentParser(description='App to illustrate KMEANS in 2D')
parser.add_argument('--data-config', type=str, default='./data_config.yaml')
args = parser.parse_args()

def prepare_data(data_config):
    data = {}
    with open(data_config) as stream:
        distributions_data = yaml.safe_load(stream)

    data['nb_clusters'] = len(distributions_data)
    
    X = []
    for _, distribution_info in distributions_data.items():
        mean = np.array(distribution_info['mean'])
        cov = np.array(distribution_info['cov'])
        length = int(distribution_info['size'])
        X += [np.random.multivariate_normal(mean, cov, size=length)]
    data['X'] = np.concatenate(X, axis=0)
    return data

data = prepare_data(args.data_config)
model = CustomKMeans(data['nb_clusters'])


app = dash.Dash(__name__)


app.layout = html.Div([
    html.H1("K-means Clustering Demo", id="h1"),
    
    html.Button("Initialize Data", id="init-button", n_clicks=0),
    html.Button("Next Iteration", id="next-button", n_clicks=0),
    
    dcc.Graph(id="cluster-graph", style={'height': '80vh', 'width': '100%'})
])

@app.callback(
    [Output("cluster-graph", "figure"),
     Output("h1", "children")],
    [Input("init-button", "n_clicks")],
    prevent_initial_call=True
)
def initialize_data(n_clicks):
    np.random.seed(0)
    X = np.random.rand(100, 2)

    data = prepare_data(args.data_config)
    model.initialise(data['nb_clusters'])
    centroids = model.centroids

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['X'][:,0], y=data['X'][:,1], mode='markers', name='data'))
    fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker=dict(color='red', size=10), name='centroids'))
    fig.update_layout(title="Initialized Data")
    
    return fig, "K-means Clustering Demo - Iteration 0"

@app.callback(
        [Output("cluster-graph", "figure", allow_duplicate=True),
         Output("h1", "children", allow_duplicate=True)],
    [Input("next-button", "n_clicks")],
    [State("cluster-graph", "figure")],
    prevent_initial_call=True,
)
def next_iteration(n_clicks, figure):
    if figure is None or "data" not in figure:
        return dash.no_update
    
    data = figure["data"][0]
    X = np.column_stack((data["x"], data["y"]))
    
    centroids, labels = model.next_iteration(X)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=labels), name='data'))
    fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker=dict(color='red', size=10), name='centroids'))
    
    return fig, f"K-means Clustering Demo - Iteration {model.current_iteration}"

if __name__ == "__main__":
    app.run_server(debug=True)
