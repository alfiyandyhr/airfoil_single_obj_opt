# -------------------------------------
# Dash app to explore the results of
# single-objective airfoiloptimization
# by alfiyandyhr
# -------------------------------------
from dash import Dash, dcc, html
import dash
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64

# Importing summary
df_summary = pd.read_csv('summary/summary_all.csv', index_col=0)

df_base_min_max = df_summary[:3]
df_lhs = df_summary[df_summary['Method'] == 'LHS']
df_dcgan = df_summary[df_summary['Method'] == 'DCGAN']
df_mixed = df_summary[df_summary['Method'] == 'DCGAN+GF'][100:]

names = []
for G in range(1,502):
    if G == 1:
        for S in range(1,101):
            names.append(f'G{G}S{S}')
    else:
        names.append(f'G{G}S1')

# Importing airfoil coordinates
folders = ['lhs', 'dcgan', 'mixed']
coord_list = {}
for folder in folders:
    if folder == 'lhs': method = 'LHS'
    elif folder == 'dcgan': method = 'DCGAN'
    else: method = 'DCGAN+GF'
    for G in range(1,502):
        if G == 1:
            if folder != 'mixed':
                for S in range(1,101):
                    coord_list[f'{method} G{G}S{S}'] = np.genfromtxt(f'database/{folder}/G{G}/G{G}S{S}_coords.dat')
        else:
            coord_list[f'{method} G{G}S1'] = np.genfromtxt(f'database/{folder}/G{G}/G{G}S1_coords.dat')

# Importing baseline
coord_list['baseline'] = np.genfromtxt('database/baseline/baseline_coords.dat')
coord_list['dv_min'] = np.genfromtxt('database/dv_min/dv_min_coords.dat')
coord_list['dv_max'] = np.genfromtxt('database/dv_max/dv_max_coords.dat')

# Instantiating Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Create server variable with Flask server object for use with gunicorn
server = app.server

# Main Layout
app.layout = html.Div([
    html.H4(children='Single-Objective Airfoil Optimization',
            style={'text-align': 'center'}),
    
    html.H6(children='''by @alfiyandyhr at the Institute of Fluid Science, Tohoku University.''',
             style={'text-align': 'center'}),

    html.P(children=['Minimize CD; subject to CL >= 0.5 and A_FFD >= 0.9*A_FFD_base', html.Br(),
                     'Re: 7.04E6, Mach = 0.73, AoA = 2', html.Br()],
           style={'text-align': 'center'}),
    
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            clickData={'points': [{'text': 'DCGAN+GF G430S1'}]}
        )
    ], style={'width': '60%', 'display': 'block', 'margin': 'auto'}),

    html.Div([
            html.Div([
                html.Img(id='cfd-image',
                         src='',
                         style={'width':'60%',
                                'display':'inline-block',
                                'margin-left':'60px',
                                'margin-top':'20px',
                                'margin-bottom':'20px'}),
                html.Div([
                        html.P(id='cfd-report',
                        children=[]),
                ], style={'display':'inline-block', 'width':'30%','position':'absolute',
                          'margin-left':'20px', 'margin-top':'50px'}),

            ], style={'display': 'inline-block', 'width': '48%', 'position':'absolute'}),

            html.Div([
                html.H6(children='''You can pinpoint an airfoil from the list below.''',
                        style={'text-align': 'center', 'margin-top': '20px'}),
                dcc.Dropdown(
                    id='crossfilter-airfoil-candidates',
                    options=[{'label': i, 'value': i} for i in df_summary['Name']],
                    value='DCGAN+GF G430S1'
                ),
                dcc.Graph(id='airfoil-plot',
                          style={'margin-top': '80px'}),
            ], style={'display': 'inline-block', 'width': '48%', 'float':'right'})
    ], style={'width': '98%'}),
    
])

@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [
     dash.dependencies.Input('crossfilter-airfoil-candidates', 'value'),
    ])

def update_graph(airfoil_candidate_name):

    fig = go.Figure()
    
    # LHS plots
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(1,601)],
            y=df_lhs['Penalized_obj'],
            name='LHS',
            mode='markers',
            marker={
                'size': 4,
                'color': 'red',
                'symbol': 'circle'
            },
            text=df_lhs['Name'].to_numpy(),
            customdata=np.stack((df_lhs['CD'], df_lhs['CL'], df_lhs['Feasibility']), axis=-1),
            hovertemplate='<b>%{text}</b><br>' +
                          'Penalized = %{y:.2f}<br>' +
                          'C<sub>D</sub> = %{customdata[0]:.6f}<br>' +
                          'C<sub>L</sub> = %{customdata[1]:.6f}<br>' +
                          '%{customdata[2]} design'
        )
    )

    # DCGAN plots
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(1,601)],
            y=df_dcgan['Penalized_obj'],
            name='DCGAN',
            mode='markers',
            marker={
                'size': 4,
                'color': 'blue',
                'symbol': 'circle'
            },
            text=df_dcgan['Name'].to_numpy(),
            customdata=np.stack((df_dcgan['CD'], df_dcgan['CL'], df_dcgan['Feasibility']), axis=-1),
            hovertemplate='<b>%{text}</b><br>' +
                          'Penalized = %{y:.2f}<br>' +
                          'C<sub>D</sub> = %{customdata[0]:.6f}<br>' +
                          'C<sub>L</sub> = %{customdata[1]:.6f}<br>' +
                          '%{customdata[2]} design'
        )
    )

    # DCGAN+GF plots
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(101,601)],
            y=df_mixed['Penalized_obj'],
            name='DCGAN+GF',
            mode='markers',
            marker={
                'size': 4,
                'color': 'green',
                'symbol': 'circle'
            },
            text=df_mixed['Name'].to_numpy(),
            customdata=np.stack((df_mixed['CD'], df_mixed['CL'], df_mixed['Feasibility']), axis=-1),
            hovertemplate='<b>%{text}</b><br>' +
                          'Penalized = %{y:.2f}<br>' +
                          'C<sub>D</sub> = %{customdata[0]:.6f}<br>' +
                          'C<sub>L</sub> = %{customdata[1]:.6f}<br>' +
                          '%{customdata[2]} design'
        )
    )

    # The selected airfoil
    fig.add_trace(
        go.Scatter(
            x=df_summary[df_summary['Name'] == airfoil_candidate_name]['n_CFD'].to_numpy(),
            y=df_summary[df_summary['Name'] == airfoil_candidate_name]['Penalized_obj'].to_numpy(),
            name='Pinpointed Airfoil',
            mode='markers',
            marker={
                'size': 12,
                'opacity': 1.0,
                'color': 'black',
                'symbol': 'x'
            },
            text=[airfoil_candidate_name],
            customdata=np.array([df_summary[df_summary['Name'] == airfoil_candidate_name]['CD'],
                                 df_summary[df_summary['Name'] == airfoil_candidate_name]['CL'],
                                 df_summary[df_summary['Name'] == airfoil_candidate_name]['Feasibility']]).reshape(1,-1),
            hovertemplate='<b>%{text}</b><br>' +
                          'Penalized = %{y:.2f}<br>' +
                          'C<sub>D</sub> = %{customdata[0]:.6f}<br>' +
                          'C<sub>L</sub> = %{customdata[1]:.6f}<br>' +
                          '%{customdata[2]} design'
        )
    )
    
    # The best feasible solution (LHS G485S1)
    fig.add_trace(
        go.Scatter(
            x=[584],
            y=[108.105934],
            name='Best feasible LHS - G485S1',
            mode='markers',
            marker={
                'size': 12,
                'color': 'red',
                'symbol': 'circle-open'
            },
            text=['LHS G485S1'],
            customdata=np.array([0.010811, 0.729375, 'Feasible']).reshape(1,-1),
            hovertemplate='<b>%{text}</b><br>' +
                          'Penalized = %{y:.2f}<br>' +
                          'C<sub>D</sub> = %{customdata[0]:.6f}<br>' +
                          'C<sub>L</sub> = %{customdata[1]:.6f}<br>' +
                          '%{customdata[2]} design'
        )
    )

    # The best feasible solution (DCGAN G432S1)
    fig.add_trace(
        go.Scatter(
            x=[531],
            y=[103.780839],
            name='Best feasible DCGAN - G432S1',
            mode='markers',
            marker={
                'size': 12,
                'color': 'blue',
                'symbol': 'circle-open'
            },
            text=['DCGAN G432S1'],
            customdata=np.array([0.010378, 0.543137, 'Feasible']).reshape(1,-1),
            hovertemplate='<b>%{text}</b><br>' +
                          'Penalized = %{y:.2f}<br>' +
                          'C<sub>D</sub> = %{customdata[0]:.6f}<br>' +
                          'C<sub>L</sub> = %{customdata[1]:.6f}<br>' +
                          '%{customdata[2]} design'
        )
    )

    # The best feasible solution (DCGAN+GF G430S1)
    fig.add_trace(
        go.Scatter(
            x=[529],
            y=[102.893823],
            name='Best feasible DCGAN+GF - G430S1',
            mode='markers',
            marker={
                'size': 12,
                'color': 'green',
                'symbol': 'circle-open'
            },
            text=['DCGAN+GF G430S1'],
            customdata=np.array([0.010289, 0.537815, 'Feasible']).reshape(1,-1),
            hovertemplate='<b>%{text}</b><br>' +
                          'Penalized = %{y:.2f}<br>' +
                          'C<sub>D</sub> = %{customdata[0]:.6f}<br>' +
                          'C<sub>L</sub> = %{customdata[1]:.6f}<br>' +
                          '%{customdata[2]} design'
        )
    )

    # Baseline RAE2822
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[df_base_min_max['Penalized_obj'][0]],
            name='Baseline - RAE2822',
            mode='markers',
            marker={
                'size': 10,
                'color': 'grey',
                'symbol': 'circle'
            },
            text=['baseline'],
            customdata=np.stack((df_base_min_max['CD'][0], df_base_min_max['CL'][0], df_base_min_max['Feasibility'][0]), axis=-1).reshape(1,-1),
            hovertemplate='<b>%{text}</b><br>' +
                          'Penalized = %{y:.2f}<br>' +
                          'C<sub>D</sub> = %{customdata[0]:.6f}<br>' +
                          'C<sub>L</sub> = %{customdata[1]:.6f}<br>' +
                          '%{customdata[2]} design'
        )
    )
    
    fig.add_shape(type='line',
                  x0=0, y0=df_base_min_max['Penalized_obj'][0],
                  x1=600, y1=df_base_min_max['Penalized_obj'][0],
                  line=dict(color='grey',dash='dash'),
                  xref='x', yref='y')
    
    fig.update_layout(
        xaxis_title='Number of CFD evaluations',
        yaxis_title='Penalized objective (count)',
        margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
        height=450,
        hovermode='closest'
    )
    
    return fig

@app.callback(
    dash.dependencies.Output('cfd-image', 'src'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')])
def update_cfd_image(clickData):
    airfoil_name = clickData['points'][0]['text']
    
    if airfoil_name == 'baseline':
        encoded = base64.b64encode(open(f'database/baseline/baseline.png', 'rb').read())
        src_link = f'data:image/png;base64,{encoded.decode()}'
    
    else:
        airfoil_name = airfoil_name.split(' ')

        folder = 'mixed' if airfoil_name[0] == 'DCGAN+GF' else airfoil_name[0].lower()
        G = airfoil_name[1].split('S')[0]
        encoded = base64.b64encode(open(f'database/{folder}/{G}/{airfoil_name[1]}.png', 'rb').read())
        src_link = f'data:image/png;base64,{encoded.decode()}'
    
    return src_link

@app.callback(
    dash.dependencies.Output('cfd-report', 'children'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')])
def update_cfd_report(clickData):
    airfoil_name = clickData['points'][0]['text']
    child = [f'{airfoil_name}', html.Br(),
             f"Penalized_obj = {df_summary[df_summary['Name']==airfoil_name]['Penalized_obj'].to_numpy()[0]:.2f}", html.Br(),
             f"CD = {df_summary[df_summary['Name']==airfoil_name]['CD'].to_numpy()[0]:.6f}", html.Br(),
             f"CL = {df_summary[df_summary['Name']==airfoil_name]['CL'].to_numpy()[0]:.6f}", html.Br(),
             f"A_constr = {df_summary[df_summary['Name']==airfoil_name]['A_constr'].to_numpy()[0]:.6f}", html.Br(),
             f"GF_Score= {0.4-df_summary[df_summary['Name']==airfoil_name]['GF_score'].to_numpy()[0]:.6f}", html.Br(),
             f"Constr_viol = {df_summary[df_summary['Name']==airfoil_name]['Constr_viol'].to_numpy()[0]:.6f}", html.Br(),
             f"{df_summary[df_summary['Name']==airfoil_name]['Feasibility'].to_numpy()[0]} design", html.Br(),]
    
    return child

def airfoil_plot(coord_np, title):
    return {
        'data': [dict(
            x=pd.Series(coord_np[:,0]),
            y=pd.Series(coord_np[:,1]),
            mode='lines'
        )],
        'layout': {
            'height': 225,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': title
            }],
            'yaxis': {'type': 'linear'},
            'xaxis': {'showgrid': False},
        }
    }

@app.callback(
    dash.dependencies.Output('airfoil-plot', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')])
def update_airfoil_plot(clickData):
    airfoil_name = clickData['points'][0]['text']
    coord_np = coord_list[airfoil_name]
    return airfoil_plot(coord_np, airfoil_name)

if __name__ == '__main__':
    app.run_server(debug=True)