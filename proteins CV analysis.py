import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import ast


def load_data(norm_type):
    df = pd.read_csv(f'top_20_proteins_{norm_type}.csv')
    df['Patient CVs'] = df['Patient CVs'].apply(ast.literal_eval)
    return df


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])


def create_data_table(norm_type):
    df = load_data(norm_type)
    return dash_table.DataTable(
        id=f'table-{norm_type}',
        columns=[
            {'name': 'Protein', 'id': 'protein names'},
            {'name': 'Average CV', 'id': 'Average CV'}
        ],
        data=df.to_dict('records'),
        row_selectable='single',
        style_table={'height': '300px', 'overflowY': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '10px'}
    )


app.layout = dbc.Container([
    html.H1("Protein CV Analysis", className="text-center my-4"),
    dbc.Row([
        dbc.Col([
            html.H3(norm_type, className="text-center"),
            create_data_table(norm_type),
            dcc.Graph(id=f'bar-{norm_type}'),
            dcc.Graph(id=f'hist-{norm_type}')
        ], width=4) for norm_type in ['Intensity', 'iBAQ', 'LFQ']
    ])
], fluid=True)

for norm_type in ['Intensity', 'iBAQ', 'LFQ']:
    @app.callback(
        [Output(f'bar-{norm_type}', 'figure'),
         Output(f'hist-{norm_type}', 'figure')],
        [Input(f'table-{norm_type}', 'selected_rows')]
    )
    def update_graphs(selected_rows, norm_type=norm_type):
        if not selected_rows:
            return {}, {}

        df = load_data(norm_type)
        protein = df.iloc[selected_rows[0]]

        cvs = protein['Patient CVs']
        patients = list(cvs.keys())
        values = list(cvs.values())

        bar_plot = go.Figure(data=[go.Bar(x=patients, y=values)])
        bar_plot.update_layout(
            title=f"{protein['protein names']} - CV per Patient",
            xaxis_title="Patient",
            yaxis_title="CV"
        )

        hist_plot = go.Figure(data=[go.Histogram(x=values)])
        hist_plot.update_layout(
            title=f"{protein['protein names']} - CV Distribution",
            xaxis_title="CV",
            yaxis_title="Count"
        )

        return bar_plot, hist_plot

if __name__ == '__main__':
    app.run_server(debug=True)