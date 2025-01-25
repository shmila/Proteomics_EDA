import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])


def load_data(norm_type):
    top_20_df = pd.read_csv(f'tumor_top_20_proteins_{norm_type}.csv')
    top_20_df['Patient CVs'] = top_20_df['Patient CVs'].apply(eval)
    top_20_df['Patient Count'] = top_20_df['Patient CVs'].apply(len)

    all_proteins_df = pd.read_csv(f'tumor_protein_analysis_{norm_type}.csv')
    all_proteins_df['Patient CVs'] = all_proteins_df['Patient CVs'].apply(eval)
    all_proteins_df = all_proteins_df[all_proteins_df['Patient CVs'].apply(len) >= 2]

    return top_20_df.sort_values('Average CV'), all_proteins_df


def format_data_for_table(df):
    table_data = df.copy()
    table_data['Patient CVs'] = table_data['Patient CVs'].apply(lambda x: json.dumps(x, indent=2))
    return table_data


def create_distribution_plot(all_df, top_20_df, norm_type):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=all_df['Average CV'],
        name='Other Proteins',
        nbinsx=5800,
        marker_color='blue',
        opacity=0.7
    ))

    fig.add_trace(go.Bar(
        x=top_20_df['Average CV'],
        y=[1] * len(top_20_df),
        name='Top 20 Most Stable',
        marker_color='red',
        width=0.001,
        opacity=0.7
    ))

    fig.update_layout(
        title=f'Distribution of Protein Expression Variability ({norm_type})',
        xaxis_title='Average Coefficient of Variation (CV)',
        yaxis_title='Number of Proteins',
        height=400,
        showlegend=True,
        hovermode='x unified'
    )

    return fig


def create_barplot(df, norm_type, all_proteins_median):
    fig = px.bar(
        df.sort_values('Average CV'),
        x='Protein names',
        y='Average CV',
        title=f'{norm_type} Average CV per Protein (Tumor Samples)',
        labels={'Protein names': 'Protein', 'Average CV': 'CV'},
        hover_data=['Gene names', 'Patient Count']
    )

    fig.add_hline(
        y=all_proteins_median,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median CV of all proteins: {all_proteins_median:.3f}"
    )

    fig.update_layout(
        showlegend=False,
        hovermode='x unified',
        height=350,
        xaxis_ticktext=[],
        xaxis_tickvals=[],
        margin=dict(t=30, b=80)
    )

    return fig


def create_column(norm_type, top_20_df, all_proteins_df):
    table_data = format_data_for_table(top_20_df)

    return dbc.Col([
        html.H3(f"{norm_type} Analysis (Tumor)", className="text-center mb-4"),
        html.Div([
            dash.dash_table.DataTable(
                id=f'table-{norm_type}',
                columns=[
                    {'name': 'Protein', 'id': 'Protein names'},
                    {'name': 'Gene', 'id': 'Gene names'},
                    {'name': 'CV', 'id': 'Average CV', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                    {'name': '# Patients', 'id': 'Patient Count'}
                ],
                data=table_data.to_dict('records'),
                style_table={'height': '600px', 'overflowY': 'auto'},
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
                sort_action='native',
                page_action='none',
                page_size=None
            )
        ], className="mb-4"),
        dcc.Graph(
            id=f'distribution-{norm_type}',
            figure=create_distribution_plot(all_proteins_df, top_20_df, norm_type),
            className="mb-4",
            style={'height': '400px'}
        ),
        dcc.Graph(
            id=f'barplot-{norm_type}',
            figure=create_barplot(top_20_df, norm_type, all_proteins_df['Average CV'].median()),
            className="mb-4",
            style={'height': '350px'}
        ),
        dcc.Graph(
            id=f'histogram-{norm_type}',
            className="mb-4"
        )
    ], width=4)


intensity_top20_df, intensity_all_df = load_data('Intensity')
ibaq_top20_df, ibaq_all_df = load_data('iBAQ')
lfq_top20_df, lfq_all_df = load_data('LFQ')

app.layout = dbc.Container([
    html.H1("Tumor Protein CV Analysis Dashboard", className="text-center my-4"),
    dbc.Row([
        create_column('Intensity', intensity_top20_df, intensity_all_df),
        create_column('iBAQ', ibaq_top20_df, ibaq_all_df),
        create_column('LFQ', lfq_top20_df, lfq_all_df)
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

        df_name = f'{norm_type.lower()}_top20_df'
        df = globals()[df_name]
        protein_name = hoverData['points'][0]['x']
        protein_data = df[df['Protein names'] == protein_name].iloc[0]

        cv_values = list(protein_data['Patient CVs'].values())

        fig = px.histogram(
            x=cv_values,
            nbins=10,
            title=f'CV Distribution for {protein_data["Gene names"]} ({protein_name})',
            labels={'x': 'CV', 'count': 'Frequency'}
        )
        fig.update_layout(height=300, margin=dict(t=30, b=30))
        return fig


for norm_type in ['Intensity', 'iBAQ', 'LFQ']:
    create_callback(norm_type)

if __name__ == '__main__':
    app.run_server(debug=True)