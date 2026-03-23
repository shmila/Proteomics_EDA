from os.path import join

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np

base_dir = r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])


def load_data(norm_type):
    csv_path = join(base_dir, "CSVs", "top_20_proteins_per_norm_type", f'top_20_proteins_{norm_type}_raw_cv.csv')
    df = pd.read_csv(csv_path)
    df['Patient CVs'] = df['Patient CVs'].apply(eval)
    df['Patient Measurements'] = df['Patient Measurements'].apply(eval)
    return df


def format_data_for_table(df):
    table_data = df.copy()
    table_data['Patient Count'] = table_data['Patient CVs'].apply(len)
    # Convert dictionaries to strings for display
    table_data['Patient CVs'] = table_data['Patient CVs'].apply(lambda x: json.dumps(x, indent=2))
    table_data['Patient Measurements'] = table_data['Patient Measurements'].apply(lambda x: json.dumps(x, indent=2))
    return table_data


def create_barplot(df, norm_type):
    df_with_count = df.copy()
    df_with_count['Patient Count'] = df_with_count['Patient CVs'].apply(len)

    fig = px.bar(
        df_with_count,
        x='Protein names',
        y='Average CV',
        title=f'{norm_type} Average CV per Protein',
        labels={'Protein names': 'Protein', 'Average CV': 'CV'},
        custom_data=['Patient Count']
    )

    fig.update_traces(
        hovertemplate="<br>".join([
            "Protein: %{x}",
            "CV: %{y:.3f}",
            "Patient Count: %{customdata[0]}",
            "<extra></extra>"
        ])
    )

    fig.update_layout(
        showlegend=False,
        hovermode='x unified',
        height=300,
        xaxis_tickangle=-45,
        margin=dict(t=30, b=80)
    )
    return fig


def get_patient_stats(row):
    patient_measurements = row['Patient Measurements']

    # Calculate mean for EVERY patient (matching the plot's calculation)
    patient_means = []
    for patient, measurements in patient_measurements.items():
        if measurements:  # Include all patients with measurements
            patient_mean = np.mean(measurements)
            patient_means.append(patient_mean)

    if patient_means:
        cv_of_means = (np.std(patient_means) / np.mean(patient_means)) * 100
        overall_cv = (np.std([m for p in patient_measurements.keys() for m in patient_measurements[p]]) /
                      np.mean([m for p in patient_measurements.keys() for m in patient_measurements[p]])) * 100

        return {
            'Overall_CV': overall_cv,
            'CV_of_Means': cv_of_means  # This should now match the plot's value
        }
    return {
        'Overall_CV': np.nan,
        'CV_of_Means': np.nan
    }


def create_column(norm_type, df):
    extended_df = df.copy()
    stats = pd.DataFrame([get_patient_stats(row) for _, row in df.iterrows()])
    extended_df = pd.concat([extended_df, stats], axis=1)

    table_data = format_data_for_table(extended_df)

    return dbc.Col([
        html.H3(f"{norm_type} Analysis", className="text-center mb-4"),
        html.Div([
            dash.dash_table.DataTable(
                id=f'table-{norm_type}',
                columns=[
                    {'name': 'Protein', 'id': 'Protein names'},
                    {'name': 'Average CV', 'id': 'Average CV', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Patient Count', 'id': 'Patient Count', 'type': 'numeric'},
                    {'name': 'Overall CV', 'id': 'Overall_CV', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Inter-Patient CV', 'id': 'Inter_Patient_CV', 'type': 'numeric',
                     'format': {'specifier': '.2f'}},
                    {'name': 'CV of Means', 'id': 'CV_of_Means', 'type': 'numeric', 'format': {'specifier': '.2f'}},
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


# Load data
intensity_df = load_data('Intensity')
ibaq_df = load_data('iBAQ')
lfq_df = load_data('LFQ')

app.layout = dbc.Container([
    html.H1("Protein CV Analysis Dashboard", className="text-center my-4"),
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
        patient_count = len(cv_values)

        fig = px.histogram(
            x=cv_values,
            nbins=10,
            title=f'CV Distribution for {protein_name}<br>(n={patient_count} patients)',
            labels={'x': 'CV', 'count': 'Frequency'}
        )
        fig.update_layout(height=300, margin=dict(t=40, b=30))
        return fig


# Create callbacks
for norm_type in ['Intensity', 'iBAQ', 'LFQ']:
    create_callback(norm_type)

if __name__ == '__main__':
    app.run_server(debug=True)
