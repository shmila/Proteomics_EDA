import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])


def load_data(norm_type):
    df = pd.read_csv(f'tumor_top_20_proteins_{norm_type}.csv')
    df['Patient CVs'] = df['Patient CVs'].apply(eval)
    return df


def format_data_for_table(df):
    table_data = df.copy()
    table_data['Patient CVs'] = table_data['Patient CVs'].apply(lambda x: json.dumps(x, indent=2))
    return table_data


intensity_df = load_data('Intensity')
ibaq_df = load_data('iBAQ')
lfq_df = load_data('LFQ')


def create_barplot(df, norm_type):
    fig = px.bar(
        df,
        x='Protein names',
        y='Average CV',
        title=f'{norm_type} Average CV per Protein (Tumor Samples)',
        labels={'Protein names': 'Protein', 'Average CV': 'CV'}
    )
    fig.update_layout(
        showlegend=False,
        hovermode='x unified',
        height=300,
        xaxis_tickangle=-45,
        margin=dict(t=30, b=80)
    )
    return fig


def create_column(norm_type, df):
    table_data = format_data_for_table(df)

    return dbc.Col([
        html.H3(f"{norm_type} Analysis (Tumor)", className="text-center mb-4"),
        html.Div([
            dash.dash_table.DataTable(
                id=f'table-{norm_type}',
                columns=[
                    {'name': 'Protein', 'id': 'Protein names'},
                    {'name': 'Average CV', 'id': 'Average CV', 'type': 'numeric', 'format': {'specifier': '.2f'}}
                ],
                data=table_data.to_dict('records'),
                style_table={'height': '300px', 'overflowY': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px',
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                page_size=10
            )
        ], className="mb-4"),
        dcc.Graph(
            id=f'barplot-{norm_type}',
            figure=create_barplot(df, norm_type),
            className="mb-4"
        ),
        dcc.Graph(
            id=f'histogram-{norm_type}',
            className="mb-4"
        )
    ], width=4)


app.layout = dbc.Container([
    html.H1("Tumor Protein CV Analysis Dashboard", className="text-center my-4"),
    dbc.Row([
        create_column('Intensity', intensity_df),
        create_column('iBAQ', ibaq_df),
        create_column('LFQ', lfq_df)
    ], className="g-4")
], fluid=True)


def create_callback(norm_type):
    @app.callback(
        Output(f'histogram-{norm_type}', 'figure'),
        Input(f'barplot-{norm_type}', 'hoverData')
    )
    def update_histogram(hoverData):
        if not hoverData:
            return go.Figure()

        df = globals()[f'{norm_type.lower()}_df']
        protein_name = hoverData['points'][0]['x']
        protein_data = df[df['Protein names'] == protein_name].iloc[0]

        cv_values = list(protein_data['Patient CVs'].values())

        fig = px.histogram(
            x=cv_values,
            nbins=10,
            title=f'CV Distribution for {protein_name} (Tumor)',
            labels={'x': 'CV', 'count': 'Frequency'}
        )
        fig.update_layout(height=300, margin=dict(t=30, b=30))
        return fig


for norm_type in ['Intensity', 'iBAQ', 'LFQ']:
    create_callback(norm_type)

if __name__ == '__main__':
    app.run_server(debug=True)