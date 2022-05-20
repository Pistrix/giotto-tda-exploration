import pickle
import numpy as np
import base64
import dash
from PIL import Image
from dash import html, dcc
from dash.dependencies import Input, Output
from dash_canvas.utils import array_to_data_url, parse_jsonstring
from dash_canvas import DashCanvas
from io import BytesIO

from skimage import io
from gtda.images import Binarizer, RadialFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, HeatKernel

import plotly.graph_objects as go
import plotly.express as px

app = dash.Dash(__name__)
application = app.server

canvas_width = 300
canvas_height = 300
filename_canvas = 'data/canvas_background.jpg'
canvas_image = io.imread(filename_canvas)


def get_rawimage_layout():
    return go.Layout(
                title=go.layout.Title(
                    text="Raw Image",
                    x=0.5
                ),
                margin=go.layout.Margin(
                    t=100
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=500,
                width=500
            )


def get_binarized_layout():
    return go.Layout(
                title=go.layout.Title(
                    text="Binarized Image",
                    x=0.5
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=500,
                width=500
            )


def get_radial_layout():
    return go.Layout(
                title=go.layout.Title(
                    text="Radial Filtration Image",
                    x=0.5
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=500,
                width=500
            )


def get_persistence_layout():
    return go.Layout(
                title=go.layout.Title(
                    text="Persistence Diagram of Homology Groups",
                    x=0.5
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=500,
                width=500
            )


def get_scaled_layout():
    return go.Layout(
                title=go.layout.Title(
                    text="Persistence Diagram of Homology Groups Scaled",
                    x=0.5
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=500,
                width=500
            )


def get_heatkernel_layout():
    return go.Layout(
                title=go.layout.Title(
                    text="Heat Kernel Image",
                    x=0.5
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=500,
                width=500
            )


app.layout = html.Div(
    [
        # Canvas
        html.Div(
            [
                html.Table(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    [
                                                html.Div(
                                                    [
                                                        html.P(
                                                            "Use your mouse to write inside the canvas",
                                                            className="canvas-text",
                                                            ),
                                                        html.Div(
                                                            [
                                                                DashCanvas(
                                                                    id="canvas",
                                                                    lineWidth=15,
                                                                    image_content=array_to_data_url(
                                                                        canvas_image),
                                                                    width=canvas_width,
                                                                    height=canvas_height,
                                                                    hide_buttons=[
                                                                        "zoom",
                                                                        "pan",
                                                                        "line",
                                                                        "pencil",
                                                                        "rectangle",
                                                                        "select",
                                                                        ],
                                                                    lineColor="black",
                                                                    goButtonTitle="Process"
                                                                    )
                                                                ],
                                                            className="canvas"
                                                            )
                                                    ]
                                                )
                                    ]
                                ),
                                html.Td(
                                    [
                                        html.Div(
                                            [
                                                html.P(
                                                    "Estimator Result",
                                                    id="result-above-text",
                                                    className="result-above-text"
                                                ),
                                                dcc.Loading(
                                                    [
                                                        html.Div(
                                                            html.P(
                                                                "",
                                                                id="result-below-text",
                                                                className="result-below-text"
                                                            )
                                                        )
                                                    ],
                                                    id="loading-anim",
                                                    type="circle"
                                                )
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    className="upper-table"
                )
            ],
            className=""
        ),
        html.Div(
            [
                html.Table(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    [html.Div(
                                        [dcc.Graph(id='raw-image',
                                                   figure=go.Figure(
                                                        layout=get_rawimage_layout()
                                                    ))],
                                        className="plot-center"
                                    )]
                                ),
                                html.Td(
                                    [html.Div(
                                        [dcc.Graph(id='bini-image',
                                                   figure=go.Figure(
                                                        layout=get_binarized_layout()
                                                    ))],
                                        className="plot-center"
                                    )]
                                ),
                                html.Td(
                                    [html.Div(
                                        [dcc.Graph(id='radial-image',
                                                   figure=go.Figure(
                                                        layout=get_radial_layout()
                                                    ))],
                                        className="plot-center"
                                    )]
                                )
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(
                                    [html.Div(
                                        [dcc.Graph(id='cubical-image',
                                                   figure=go.Figure(
                                                        layout=get_persistence_layout()
                                                    ))],
                                        className="plot-center"
                                    )]
                                ),
                                html.Td(
                                    [html.Div(
                                        [dcc.Graph(id='scaled-image',
                                                   figure=go.Figure(
                                                        layout=get_scaled_layout()
                                                    ))],
                                        className="plot-center"
                                    )]
                                ),
                                html.Td(
                                    [html.Div(
                                        [dcc.Graph(id='heat-image',
                                                   figure=go.Figure(
                                                        layout=get_heatkernel_layout()
                                                    ))],
                                        className="plot-center"
                                    )]
                                )
                            ]
                        )
                    ],
                    className="lower-table"
                )
            ]
        )
    ]
)


def preprocess_mask(string):
    '''
    Use PIL to reshape the canvas data to 28x28 (aligning with MNIST).
    Returns a PIL friendly version and numpy friendly version for backend processing
    '''
    # Get mask from json data
    mask = parse_jsonstring(string, shape=(300, 300))
    mask = (~mask.astype(bool)).astype(int)

    # Convert image mask to pillow object for resizing.
    image_string = array_to_data_url((255 * mask).astype(np.uint8))
    image_pillow = Image.open(BytesIO(base64.b64decode(image_string[22:])))
    image_pillow = image_pillow.resize((28, 28))

    # Convert image pillow to normalized Numpy array
    image_numpy = reverse_image_array(np.array(image_pillow))
    image_numpy = np.reshape(image_numpy, (1, 28, 28))

    return image_pillow, image_numpy


@app.callback(
    Output('raw-image', 'figure'),
    Output('bini-image', 'figure'),
    Output('radial-image', 'figure'),
    Output('cubical-image', 'figure'),
    Output('scaled-image', 'figure'),
    Output('heat-image', 'figure'),
    Output('result-below-text', 'children'),
    Input('canvas', 'json_data'),
    prevent_initial_call=True)
def process_image(mask_as_string):
    image_pillow, image_numpy = preprocess_mask(mask_as_string)

    # Images at each step
    image_raw = px.imshow(image_pillow, color_continuous_scale='gray')
    image_raw.update_layout(get_rawimage_layout())

    binarizer = Binarizer(threshold=0.4)
    image_binarized = binarizer.fit_transform(np.array(image_numpy))
    plot_biniarized = binarizer.plot(
        image_binarized, plotly_params={"layout": get_binarized_layout()})

    radial_filtration = RadialFiltration(center=np.array([20, 6]))
    image_radial = radial_filtration.fit_transform(image_binarized)
    plot_radial = radial_filtration.plot(
        image_radial, colorscale="jet", plotly_params={"layout": get_radial_layout()})

    cubical_persistence = CubicalPersistence(
        n_jobs=-1, reduced_homology=False)
    image_cubical = cubical_persistence.fit_transform(image_radial)
    plot_cubical = cubical_persistence.plot(
        image_cubical, plotly_params={"layout": get_persistence_layout()})

    scaler = Scaler()
    image_scaled = scaler.fit_transform(image_cubical)
    plot_scaled = scaler.plot(
        image_scaled, plotly_params={"layout": get_scaled_layout()})

    heat = HeatKernel(sigma=.15, n_bins=60, n_jobs=-1)
    image_heat = heat.fit_transform(image_scaled)
    plot_heat = heat.plot(
        image_heat, homology_dimension_idx=1, colorscale="jet", plotly_params={"layout": get_heatkernel_layout()}
        )

    # Load TDA pipeline and digit model
    model_filename = 'data/model_pipeline.sav'
    model = pickle.load(open(model_filename, 'rb'))

    # Run models on input image
    result = model.predict(image_numpy)
    result = str(result[0])

    return image_raw, plot_biniarized, plot_radial, plot_cubical, plot_scaled, plot_heat, result


def reverse_image_array(array):
    '''Reverse array values so 0->255 and 255->0 for use with giotto-tda backend'''
    array = 255. - array
    return array


if __name__ == '__main__':
    app.run_server(debug=True)
