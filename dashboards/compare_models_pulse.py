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
from utils import impulse_response, model_prediction, generate_pulse_response
from save_result import Model_Result
import equinox as eqx
import jax
import traceback

# ==========================================
# Step 1. Build dataframe from your folder
# ==========================================

folder = "trained_models"
entries = []
loss_histories = {}
val_histories = {}

pattern = re.compile(
    r"(pulse)_FNO1d_"
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
        columns=[{"name": col, "id": col} for col in df.columns if col not in ['filename', 'color', 'final_val_loss']] +
                [{"name": "Final Validation Loss", "id": "final_val_loss", "type": "numeric", "format": {"specifier": ".2e"}}],
        data=df.to_dict('records'),
        row_selectable='multi',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=20,
        style_data_conditional=[],
        sort_action='custom',
        sort_mode='single',
        sort_by=[]
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
                        {'label': 'Pulse', 'value': 'pulse'},
                        {'label': 'Fourier', 'value': 'fourier'}
                    ],
                    value='pulse',
                    style={'width': '150px', 'display': 'inline-block'}
                ),
                html.Button('Generate Responses', id='generate-button', style={'margin-left': '20px'})
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

# ==========================================
# Step 3. Callback
# ==========================================

@app.callback(
    [Output('model-table', 'data'),
     Output('loss-val-plot', 'figure'),
     Output('impulse-response-plot', 'figure'),
     Output('signal-response-plot', 'figure'),
     Output('model-table', 'style_data_conditional')],
    [Input(f'filter-{col}', 'value') for col in df.columns if col not in ['filename', 'color', 'final_val_loss']] +
    [Input('sort-by', 'value'),
     Input('sort-direction', 'value'),
     Input('model-table', 'derived_virtual_selected_rows'),
     Input('generate-button', 'n_clicks')],
    [State('model-table', 'derived_virtual_data'),
     State('signal-type', 'value')]
)
def update_table_and_plots(*args):
    filter_inputs = args[:-6]
    sort_by = args[-6]
    sort_direction = args[-5]
    selected_rows = args[-4]
    generate_clicks = args[-3]
    virtual_data = args[-2]
    signal_type = args[-1]
    
    filter_cols = [col for col in df.columns if col not in ['filename', 'color', 'final_val_loss']]
    filters = {col: values for col, values in zip(filter_cols, filter_inputs) if values}
    
    filtered_df = apply_filters(df, filters) if filters else df.copy()
    filtered_df = apply_sorting(filtered_df, sort_by, sort_direction)
    data_to_use = filtered_df.to_dict('records')
    
    loss_traces = []
    impulse_traces = []
    style_data_conditional = []
    selected_rows = selected_rows or []

    # Initialize signal response figure with subplots
    signal_fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=[f"Sample {i+1}" for i in range(5)],
        vertical_spacing=0.1
    )
    signal_fig.update_layout(
        height=1000,
        title_text=f"{signal_type.capitalize()} Signal Responses",
        showlegend=False
    )

    # Generate inputs and outputs once if button clicked
    time = np.linspace(0, 200, 300)
    inputs, outputs = None, None
    if generate_clicks and generate_clicks > 0:
        if len(data_to_use) > 0:
            inputs, outputs = generate_pulse_response(
                signal_type, 
                time,
                samples=5
            )

    for i, row in enumerate(data_to_use):
        if i in selected_rows:
            filename = row['filename']
            color = row['color']
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
                    predictions = model_prediction(model_result.model, inputs)
                    
                    for sample_idx in range(5):
                        # Plot output only once (thick black dashed)
                        signal_fig.add_trace(
                            go.Scatter(
                                y=outputs[sample_idx,:],
                                mode='lines',
                                line=dict(color='black', dash='dash', width=4),
                                showlegend=False
                            ),
                            row=sample_idx+1,
                            col=1
                        )
                        
                        # Model prediction (solid color line)
                        signal_fig.add_trace(
                            go.Scatter(
                                y=predictions[sample_idx,0,:],
                                mode='lines',
                                line=dict(color=color, width=2),
                                showlegend=False
                            ),
                            row=sample_idx+1,
                            col=1
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

    return data_to_use, loss_fig, impulse_fig, signal_fig, style_data_conditional

# ==========================================
# Step 4. Run server
# ==========================================

if __name__ == '__main__':
    app.run(debug=True)