import pickle
import numpy as np
import sys
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
np.set_printoptions(threshold=sys.maxsize)

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

canvas_width = 300
canvas_height = 300
filename_canvas = 'canvas_background.jpg'
canvas_image = io.imread(filename_canvas)

app.layout = html.Div(
    [
        # Canvas
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Write inside the canvas with your stylus and press Sign",
                            className="canvas",
                        ),
                        html.Div(
                            DashCanvas(
                                id="canvas",
                                lineWidth=15,
                                image_content=array_to_data_url(canvas_image),
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
                                goButtonTitle="Analyze",
                            ),
                            className="canvas",
                            style={"margin-top": "1em"},
                        ),
                        html.P(
                            "",
                            id="ml-result",
                        ),
                    ],
                    className="canvas",
                )
            ],
            className="canvas"
        ),
        html.Div(
            [
                html.Table(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    [html.Div(
                                        [html.Img(id='raw-image', width=300)]
                                    )]
                                ),
                                html.Td(
                                    [html.Div(
                                        [dcc.Graph(id='bini-image')]
                                    )]
                                ),
                                html.Td(
                                    [html.Div(
                                        [dcc.Graph(id='radial-image')]
                                    )]
                                )
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(
                                    [html.Div(
                                        [dcc.Graph(id='cubical-image')]
                                    )]
                                ),
                                html.Td(
                                    [html.Div(
                                        [dcc.Graph(id='scaled-image')]
                                    )]
                                ),
                                html.Td(
                                    [html.Div(
                                        [dcc.Graph(id='heat-image')]
                                    )]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)


def preprocess_mask(string):
    '''
    Use PIL to reshape the canvas data to 28x28 (aligning with MNST).
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
    Output('raw-image', 'src'),
    Output('bini-image', 'figure'),
    Output('radial-image', 'figure'),
    Output('cubical-image', 'figure'),
    Output('scaled-image', 'figure'),
    Output('heat-image', 'figure'),
    Output('ml-result', 'children'),
    Input('canvas', 'json_data'),
    prevent_initial_call=True)
def process_image(mask_as_string):
    image_pillow, image_numpy = preprocess_mask(mask_as_string)

    # Images at each step
    binarizer = Binarizer(threshold=0.4)
    image_binarized = binarizer.fit_transform(np.array(image_numpy))
    plot_biniarized = binarizer.plot(image_binarized)

    radial_filtration = RadialFiltration(center=np.array([20, 6]))
    image_radial = radial_filtration.fit_transform(image_binarized)
    plot_radial = radial_filtration.plot(image_radial, colorscale="jet")

    cubical_persistence = CubicalPersistence(
        n_jobs=-1, reduced_homology=(False))
    image_cubical = cubical_persistence.fit_transform(image_radial)
    #if np.array_equal(image_cubical, [[[0., 0., 0.], [0., 0., 1.]]]):
    #    print("No homology features found, set plots to default")
    #    return image_pillow, plot_biniarized, {}, {}, {}, {}, "Unknown"
    plot_cubical = cubical_persistence.plot(image_cubical)

    scaler = Scaler()
    image_scaled = scaler.fit_transform(image_cubical)
    plot_scaled = scaler.plot(image_scaled)

    heat = HeatKernel(sigma=.15, n_bins=60, n_jobs=-1)
    image_heat = heat.fit_transform(image_scaled)
    plot_heat = heat.plot(
        image_heat, homology_dimension_idx=1, colorscale="jet")

    # Load TDA pipeline and digit model
    model_filename = 'model_pipeline.sav'
    model = pickle.load(open(model_filename, 'rb'))

    # Run models on input image
    result = model.predict(image_numpy)
    result = str(result[0])

    return image_pillow, plot_biniarized, plot_radial, plot_cubical, plot_scaled, plot_heat, result


def reverse_image_array(array):
    '''Reverse array values so 0->255 and 255->0 for use with giotto-tda backend'''
    array = 255. - array
    return array


def denormalize_image_array(array):
    '''Converts normalized array back to [0,255] for displaying in html'''
    array = (1-array)*255
    return array


if __name__ == '__main__':
    app.run_server(debug=True)
