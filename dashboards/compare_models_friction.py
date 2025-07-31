import os
import re
import pickle
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from utils import model_prediction, load_friction_data, impulse_response
from save_result import Model_Result
import traceback

# ==========================================
# Step 1. Build dataframe from your folder
# ==========================================

folder = "trained_models"
entries = []
loss_histories = {}
val_histories = {}

pattern = re.compile(
    r"(friction)_FNO1d_"
    r"in_channels(?P<in_channels>\d+)_"
    r"out_channels(?P<out_channels>\d+)_"
    r"modes(?P<modes>\d+)_"
    r"width(?P<width>\d+)_"
    r"activation(?P<activation>[^_]+)_"
    r"n_blocks(?P<n_blocks>\d+)_"
)

for filename in os.listdir(folder):
    if filename.endswith("_data.pkl"):
        match = pattern.match(filename)
        if match:
            entry = match.groupdict()
            entry['filename'] = filename

            for key in ['in_channels', 'out_channels', 'modes', 'width', 'n_blocks']:
                entry[key] = int(entry[key])

            with open(os.path.join(folder, filename), "rb") as f:
                data = pickle.load(f)
                loss_histories[filename] = [float(x) for x in data['loss_history']]
                val_histories[filename] = [float(x) for x in data['val_history']]
                entry['final_val_loss'] = float(data['val_history'][-1])

            entries.append(entry)

df = pd.DataFrame(entries)
df = df.drop(columns=['in_channels', 'out_channels'])

color_sequence = px.colors.qualitative.Dark24
df['color'] = [color_sequence[i % len(color_sequence)] for i in range(len(df))]

# ==========================================
# Prepare dropdown filter options
# ==========================================

def generate_dropdown_options(column):
    unique_vals = df[column].unique()
    return [{'label': str(val), 'value': val} for val in sorted(unique_vals, key=lambda x: str(x))]

# ==========================================
# Step 2. Initialize Dash app with tabs
# ==========================================

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Model Catalogue"),

    dcc.Store(id='current-page', data=0),  # Pagination tracking

    # Filters and sorting controls
    html.Div([
        html.Div([
            html.Div([
                html.Label(col, style={'font-size': '12px', 'margin-bottom': '2px'}),
                dcc.Dropdown(
                    id=f'filter-{col}',
                    options=generate_dropdown_options(col),
                    multi=True,
                    placeholder=f'{col}...',
                    clearable=True,
                    style={
                        'width': '100%',
                        'font-size': '12px',
                        'min-width': '100px',
                        'margin-bottom': '5px'
                    }
                )
            ], style={
                'width': '16%',
                'display': 'inline-block',
                'margin-right': '5px',
                'vertical-align': 'top'
            })
            for col in df.columns if col not in ['filename', 'color', 'final_val_loss']
        ], style={
            'display': 'inline-block',
            'width': '78%',
            'vertical-align': 'top'
        }),

        html.Div([
            html.Label("Sort by:", style={'font-size': '12px'}),
            dcc.Dropdown(
                id='sort-by',
                options=[{'label': col, 'value': col} for col in df.columns if col not in ['filename', 'color']] + 
                        [{'label': 'Final Validation Loss', 'value': 'final_val_loss'}],
                placeholder='Column...',
                clearable=True,
                style={'width': '100%', 'font-size': '12px', 'margin-bottom': '5px'}
            ),
            dcc.RadioItems(
                id='sort-direction',
                options=[
                    {'label': 'Asc', 'value': 'asc'},
                    {'label': 'Desc', 'value': 'desc'}
                ],
                value='asc',
                labelStyle={'display': 'inline-block', 'margin-right': '5px', 'font-size': '12px'}
            )
        ], style={
            'display': 'inline-block',
            'width': '20%',
            'vertical-align': 'top',
            'margin-left': '10px'
        })
    ], style={'width': '100%', 'margin-bottom': '15px'}),

    dash_table.DataTable(
        id='model-table',
        columns=[
            {"name": col, "id": col} for col in df.columns if col not in ['color', 'final_val_loss']
        ] + [
            {"name": "Final Validation Loss", "id": "final_val_loss", "type": "numeric", "format": {"specifier": ".2e"}}
        ],
        data=df.to_dict('records'),
        row_selectable='multi',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=20,
        style_data_conditional=[],
        sort_action='custom',
        sort_mode='single',
        sort_by=[],
        hidden_columns=['filename'],  # hide filename but keep it in data for unique ID tracking
    ),

    # Tab layout for different plots
    dcc.Tabs([
        dcc.Tab(label='Training History', children=[
            dcc.Graph(id='loss-val-plot')
        ]),
        dcc.Tab(label='Impulse Response', children=[
            dcc.Graph(id='impulse-response-plot')
        ]),
        dcc.Tab(label='Signal Response', children=[
            html.Div([
                html.Label("Input Signal Type:", style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='signal-type',
                    options=[
                        {'label': 'Training', 'value': 'training'},
                        {'label': 'Testing', 'value': 'testing'}
                    ],
                    value='training',
                    style={'width': '150px', 'display': 'inline-block'}
                ),
                html.Button('Generate Responses', id='generate-button', style={'margin-left': '20px'}),
                html.Button('Previous', id='prev-button', n_clicks=0, style={'margin-left': '20px'}),
                html.Button('Next', id='next-button', n_clicks=0, style={'margin-left': '10px'}),
            ], style={'margin': '20px 0'}),
            dcc.Graph(id='signal-response-plot')
        ])
    ])
])

# ==========================================
# Helper functions
# ==========================================

def apply_filters(df, filters):
    filtered_df = df.copy()
    for col, values in filters.items():
        if values:
            filtered_df = filtered_df[filtered_df[col].isin(values)]
    return filtered_df

def apply_sorting(df, sort_by, sort_direction):
    if sort_by and sort_direction:
        return df.sort_values(by=sort_by, ascending=(sort_direction == 'asc'))
    return df

def get_selected_filenames(virtual_data, selected_rows_indices):
    selected = []
    if selected_rows_indices and virtual_data:
        for idx in selected_rows_indices:
            if idx < len(virtual_data):
                selected.append(virtual_data[idx]['filename'])
    return selected

train_x_norm, train_y_norm, test_x_norm, test_y_norm = load_friction_data()

# ==========================================
# Step 3. Callback
# ==========================================

@app.callback(
    [Output('model-table', 'data'),
     Output('loss-val-plot', 'figure'),
     Output('impulse-response-plot', 'figure'),
     Output('signal-response-plot', 'figure'),
     Output('model-table', 'style_data_conditional'),
     Output('current-page', 'data')],
    [Input(f'filter-{col}', 'value') for col in df.columns if col not in ['filename', 'color', 'final_val_loss']] +
    [Input('sort-by', 'value'),
     Input('sort-direction', 'value'),
     Input('model-table', 'derived_virtual_selected_rows'),
     Input('generate-button', 'n_clicks'),
     Input('next-button', 'n_clicks'),
     Input('prev-button', 'n_clicks')],
    [State('model-table', 'derived_virtual_data'),
     State('signal-type', 'value'),
     State('current-page', 'data')]
)
def update_table_and_plots(*args):
    filter_inputs = args[:4]
    sort_by = args[4]
    sort_direction = args[5]
    selected_rows_indices = args[6]
    generate_clicks = args[7]
    next_clicks = args[8]
    prev_clicks = args[9]
    virtual_data = args[10]
    signal_type = args[11]
    current_page = args[12]

    filter_cols = [col for col in df.columns if col not in ['filename', 'color', 'final_val_loss']]
    filters = {col: values for col, values in zip(filter_cols, filter_inputs) if values}

    filtered_df = apply_filters(df, filters) if filters else df.copy()
    filtered_df = apply_sorting(filtered_df, sort_by, sort_direction)
    data_to_use = filtered_df.to_dict('records')

    # Map selected indices to filenames
    selected_filenames = get_selected_filenames(virtual_data, selected_rows_indices)

    loss_traces = []
    impulse_traces = []
    style_data_conditional = []

    # Pagination setup
    samples_per_page = 9
    ctx = dash.callback_context
    if ctx.triggered:
        if 'next-button' in ctx.triggered[0]['prop_id']:
            current_page += 1
        elif 'prev-button' in ctx.triggered[0]['prop_id']:
            current_page -= 1

    current_page = max(0, current_page)

    inputs, outputs = None, None
    total_samples = 0
    if generate_clicks and generate_clicks > 0:
        if signal_type == 'training':
            inputs, outputs = train_x_norm, train_y_norm
        elif signal_type == 'testing':
            inputs, outputs = test_x_norm, test_y_norm

        if outputs is not None:
            total_samples = outputs.shape[0]

    total_pages = max((total_samples + samples_per_page - 1) // samples_per_page, 1)
    current_page = min(current_page, total_pages - 1)

    start_idx = current_page * samples_per_page
    end_idx = min(start_idx + samples_per_page, total_samples)

    # Initialize 3x3 subplots
    signal_fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f"Sample {i+1}" for i in range(start_idx, end_idx)],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    signal_fig.update_layout(
        height=900,
        title_text=f"{signal_type.capitalize()} Signal Responses (Samples {start_idx+1}-{end_idx})",
        showlegend=False
    )

    for i, row in enumerate(data_to_use):
        filename = row['filename']
        color = row['color']
        if filename in selected_filenames:
            model_filename_prefix = filename.replace('_data.pkl', '')

            loss_traces.append(go.Scatter(
                y=loss_histories[filename],
                mode='lines',
                line=dict(color=color, width=2),
                opacity=0.5,
                showlegend=False
            ))
            loss_traces.append(go.Scatter(
                y=val_histories[filename],
                mode='lines',
                line=dict(color=color, dash='dash', width=3),
                showlegend=False
            ))

            try:
                model_result = Model_Result.load(model_filename_prefix)
                response = impulse_response(model_result.model)
                impulse_traces.append(go.Scatter(
                    y=response,
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=model_filename_prefix,
                    showlegend=False
                ))

                if generate_clicks and generate_clicks > 0 and inputs is not None:
                    # Slice inputs to only the current page samples
                    inputs_slice = inputs[start_idx:end_idx]

                    # Get predictions for just these samples
                    predictions_slice = model_prediction(model_result.model, inputs_slice)

                    for idx in range(end_idx - start_idx):
                        sample_idx = start_idx + idx
                        if sample_idx >= outputs.shape[0]:
                            continue
                        row_idx = idx // 3 + 1
                        col_idx = idx % 3 + 1

                        # Ground truth for this sample
                        signal_fig.add_trace(
                            go.Scatter(
                                y=outputs[sample_idx,0,:],
                                mode='lines',
                                line=dict(color='black', dash='dash', width=4),
                                showlegend=False
                            ),
                            row=row_idx,
                            col=col_idx
                        )

                        # Prediction from sliced prediction output
                        signal_fig.add_trace(
                            go.Scatter(
                                y=predictions_slice[idx,0,:],
                                mode='lines',
                                line=dict(color=color, width=2),
                                showlegend=False
                            ),
                            row=row_idx,
                            col=col_idx
                        )

            except Exception as e:
                print(f"Error processing model {model_filename_prefix}: {str(e)}")
                traceback.print_exc()
                continue

            style_data_conditional.append({
                'if': {'row_index': i},
                'backgroundColor': color,
                'color': 'white'
            })

    loss_fig = go.Figure(data=loss_traces)
    loss_fig.update_layout(
        title="Loss and Validation History",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",
        yaxis=dict(tickformat=".1e"),
        showlegend=False
    )

    impulse_fig = go.Figure(data=impulse_traces)
    impulse_fig.update_layout(
        title="Impulse Response",
        xaxis_title="Time Step",
        yaxis_title="Response",
        showlegend=False
    )

    return data_to_use, loss_fig, impulse_fig, signal_fig, style_data_conditional, current_page

# ==========================================
# Step 4. Run server
# ==========================================

if __name__ == '__main__':
    app.run(debug=True)
