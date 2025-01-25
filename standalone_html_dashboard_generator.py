import dash
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import dash_bootstrap_components as dbc


def load_data(norm_type):
    df = pd.read_csv(f'tumor_top_20_proteins_{norm_type}.csv')
    df['Patient CVs'] = df['Patient CVs'].apply(eval)
    df['Patient Count'] = df['Patient CVs'].apply(len)
    all_df = pd.read_csv(f'tumor_protein_analysis_{norm_type}.csv')
    all_df['Patient CVs'] = all_df['Patient CVs'].apply(eval)
    return df, all_df


def create_distribution_fig(all_df, top_20_df, norm_type):
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
        xaxis_title='Average CV',
        yaxis_title='Number of Proteins',
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    return fig.to_json()


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
    return fig.to_json()


def create_dashboard_html():
    # Load data
    intensity_df, intensity_all = load_data('Intensity')
    ibaq_df, ibaq_all = load_data('iBAQ')
    lfq_df, lfq_all = load_data('LFQ')

    # Generate plots
    plots = {
        'intensity_dist': create_distribution_fig(intensity_all, intensity_df, 'Intensity'),
        'ibaq_dist': create_distribution_fig(ibaq_all, ibaq_df, 'iBAQ'),
        'lfq_dist': create_distribution_fig(lfq_all, lfq_df, 'LFQ'),
        'intensity_bar': create_barplot(intensity_df, 'Intensity', intensity_all['Average CV'].median()),
        'ibaq_bar': create_barplot(ibaq_df, 'iBAQ', ibaq_all['Average CV'].median()),
        'lfq_bar': create_barplot(lfq_df, 'LFQ', lfq_all['Average CV'].median())
    }

    # Create HTML template with embedded data
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Protein CV Analysis Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .plot-container {
                margin-bottom: 20px;
            }
            .table-container {
                height: 400px;
                overflow-y: auto;
                margin-bottom: 20px;
            }
            table {
                width: 100%;
            }
            th {
                position: sticky;
                top: 0;
                background: white;
            }
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h1 class="text-center my-4">Tumor Protein CV Analysis Dashboard</h1>
            <div class="row">
                <!-- Intensity Column -->
                <div class="col-md-4">
                    <h3 class="text-center">Intensity Analysis (Tumor)</h3>
                    <div class="table-container">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Protein</th>
                                    <th>Gene</th>
                                    <th>CV</th>
                                    <th>Patients</th>
                                </tr>
                            </thead>
                            <tbody id="intensity-table">
                            </tbody>
                        </table>
                    </div>
                    <div id="intensity-dist" class="plot-container"></div>
                    <div id="intensity-bar" class="plot-container"></div>
                    <div id="intensity-hist" class="plot-container"></div>
                </div>

                <!-- iBAQ Column -->
                <div class="col-md-4">
                    <h3 class="text-center">iBAQ Analysis (Tumor)</h3>
                    <div class="table-container">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Protein</th>
                                    <th>Gene</th>
                                    <th>CV</th>
                                    <th>Patients</th>
                                </tr>
                            </thead>
                            <tbody id="ibaq-table">
                            </tbody>
                        </table>
                    </div>
                    <div id="ibaq-dist" class="plot-container"></div>
                    <div id="ibaq-bar" class="plot-container"></div>
                    <div id="ibaq-hist" class="plot-container"></div>
                </div>

                <!-- LFQ Column -->
                <div class="col-md-4">
                    <h3 class="text-center">LFQ Analysis (Tumor)</h3>
                    <div class="table-container">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Protein</th>
                                    <th>Gene</th>
                                    <th>CV</th>
                                    <th>Patients</th>
                                </tr>
                            </thead>
                            <tbody id="lfq-table">
                            </tbody>
                        </table>
                    </div>
                    <div id="lfq-dist" class="plot-container"></div>
                    <div id="lfq-bar" class="plot-container"></div>
                    <div id="lfq-hist" class="plot-container"></div>
                </div>
            </div>
        </div>

        <script>
            // Plot data
            const plots = ''' + json.dumps(plots) + ''';

            // Table data with custom serialization to preserve Patient CVs structure
            const tableData = {
                intensity: ''' + json.dumps(intensity_df.to_dict('records'), ensure_ascii=False) + ''',
                ibaq: ''' + json.dumps(ibaq_df.to_dict('records'), ensure_ascii=False) + ''',
                lfq: ''' + json.dumps(lfq_df.to_dict('records'), ensure_ascii=False) + '''
            };

            // Create plots
            Plotly.newPlot('intensity-dist', JSON.parse(plots.intensity_dist));
            Plotly.newPlot('ibaq-dist', JSON.parse(plots.ibaq_dist));
            Plotly.newPlot('lfq-dist', JSON.parse(plots.lfq_dist));
            Plotly.newPlot('intensity-bar', JSON.parse(plots.intensity_bar));
            Plotly.newPlot('ibaq-bar', JSON.parse(plots.ibaq_bar));
            Plotly.newPlot('lfq-bar', JSON.parse(plots.lfq_bar));

            // Create empty histogram plots
            const createEmptyHist = (divId) => {
                Plotly.newPlot(divId, [{
                    type: 'histogram',
                    x: []
                }], {
                    title: 'Hover over bars to see protein CV distribution',
                    height: 300,
                    margin: {t: 30, b: 30}
                });
            };

            createEmptyHist('intensity-hist');
            createEmptyHist('ibaq-hist');
            createEmptyHist('lfq-hist');

            // Populate tables
            function populateTable(data, tableId) {
                const tbody = document.getElementById(tableId);
                data.forEach(row => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${row['Protein names']}</td>
                        <td>${row['Gene names']}</td>
                        <td>${row['Average CV'].toFixed(3)}</td>
                        <td>${row['Patient Count']}</td>
                    `;
                    tbody.appendChild(tr);
                });
            }

            populateTable(tableData.intensity, 'intensity-table');
            populateTable(tableData.ibaq, 'ibaq-table');
            populateTable(tableData.lfq, 'lfq-table');

            // Add hover callbacks for bar plots
            const createHistogram = (data, divId, protein) => {
                const proteinData = data.find(p => p['Protein names'] === protein);
                if (!proteinData) return;

                // Parse the CV values - they are stored as an object in the Patient CVs field
                const cvValues = [];
                for (let patientId in proteinData['Patient CVs']) {
                    cvValues.push(proteinData['Patient CVs'][patientId]);
                }

                // Create histogram trace with exact bins
                const trace = {
                    type: 'histogram',
                    x: cvValues,
                    xbins: {
                        start: Math.min(...cvValues) - 0.001,
                        end: Math.max(...cvValues) + 0.001,
                        size: (Math.max(...cvValues) - Math.min(...cvValues)) / 5
                    },
                    autobinx: false,
                    marker: {
                        color: 'rgb(65, 105, 225)'
                    },
                    opacity: 1
                };

                Plotly.newPlot(divId, [trace], {
                    title: `CV Distribution for ${proteinData['Gene names']} (${protein})`,
                    xaxis: {
                        title: 'CV',
                        showgrid: true,
                        gridcolor: 'rgb(240, 240, 240)',
                        range: [Math.min(...cvValues) - 0.01, Math.max(...cvValues) + 0.01]
                    },
                    yaxis: {
                        title: 'count',
                        showgrid: true,
                        gridcolor: 'rgb(240, 240, 240)',
                        range: [0, 1]
                    },
                    plot_bgcolor: 'rgb(240, 240, 250)',
                    height: 300,
                    margin: {t: 30, b: 30}
                });
            };

            document.getElementById('intensity-bar').on('plotly_hover', function(data) {
                if (!data.points || !data.points[0]) return;
                createHistogram(tableData.intensity, 'intensity-hist', data.points[0].x);
            });

            document.getElementById('ibaq-bar').on('plotly_hover', function(data) {
                if (!data.points || !data.points[0]) return;
                createHistogram(tableData.ibaq, 'ibaq-hist', data.points[0].x);
            });

            document.getElementById('lfq-bar').on('plotly_hover', function(data) {
                if (!data.points || !data.points[0]) return;
                createHistogram(tableData.lfq, 'lfq-hist', data.points[0].x);
            });
        </script>
    </body>
    </html>
    '''

    with open('protein_cv_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == "__main__":
    create_dashboard_html()