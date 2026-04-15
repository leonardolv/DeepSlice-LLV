from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import pandas as pd
import numpy as np
import os
from skimage.color import rgb2gray
import warnings
import h5py
from PIL import Image


VALID_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class PredictionProgressCallback(tf.keras.callbacks.Callback):
    """Keras callback used to expose prediction progress to the GUI."""

    def __init__(self, total_images, phase, progress_callback):
        super().__init__()
        self.total_images = total_images
        self.phase = phase
        self.progress_callback = progress_callback

    def on_predict_batch_end(self, batch, logs=None):
        if self.progress_callback is None:
            return
        completed = min(batch + 1, self.total_images)
        self.progress_callback(completed, self.total_images, self.phase)


def gray_scale(img: np.ndarray) -> np.ndarray:
    """
    Convert the image to grayscale

    :param img: The image to convert
    :type img: numpy.ndarray
    :return: The converted image
    :rtype: numpy.ndarray
    """
    img = rgb2gray(img).reshape(299, 299, 1)
    return img


def initialise_network(xception_weights: str, weights: str, species: str) -> Sequential:
    """
    Initialise a neural network with the given weights

    :param weights: The weights for the network
    :type weights: list
    :param species: The species of the animal, this is necessary because of a previous error where the models are slightly different for different species
    :return: The initialised neural network
    :rtype: keras.models.Sequential
    """
    base_model = Xception(include_top=True, weights=xception_weights)

    if species == "rat":
        inputs = Input(shape=(299, 299, 3))
        base_model_layer = base_model(inputs, training=True)
        dense1_layer = Dense(256, activation="relu")(base_model_layer)
        dense2_layer = Dense(256, activation="relu")(dense1_layer)
        output_layer = Dense(9, activation="linear")(dense2_layer)
        model = Model(inputs=inputs, outputs=output_layer)
    else:
        model = Sequential()
        model.add(base_model)
        model.add(Dense(256, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(9, activation="linear"))

    if weights is not None:
        model = load_xception_weights(model, weights, species)
    return model


def load_xception_weights(model, weights, species="mouse"):
    with h5py.File(weights, "r") as new:
        # set weight of each layer manually
        if species == "mouse":
            xception_idx = 0
            dense_idx = 1
        elif species == "rat":
            # RatModelInProgress.h5 has an "input_2" layer at index 0, so we need to adjust the indices<
            xception_idx = 1
            dense_idx = 2

        model.layers[dense_idx].set_weights(
            [new["dense"]["dense"]["kernel:0"], new["dense"]["dense"]["bias:0"]]
        )
        model.layers[dense_idx + 1].set_weights(
            [new["dense_1"]["dense_1"]["kernel:0"], new["dense_1"]["dense_1"]["bias:0"]]
        )
        model.layers[dense_idx + 2].set_weights(
            [new["dense_2"]["dense_2"]["kernel:0"], new["dense_2"]["dense_2"]["bias:0"]]
        )

        # Set the weights of the xception model
        weight_names = new["xception"].attrs["weight_names"].tolist()
        weight_names_layers = [
            name.decode("utf-8").split("/")[0] for name in weight_names
        ]

        for i in range(len(model.layers[xception_idx].layers)):
            name_of_layer = model.layers[xception_idx].layers[i].name
            # if layer name is in the weight names, then we will set weights
            if name_of_layer in weight_names_layers:
                # Get name of weights in the layer
                layer_weight_names = []
                for weight in model.layers[xception_idx].layers[i].weights:
                    try:
                        layer_weight_names.append(weight.name.split("/")[1])
                    except IndexError:
                        layer_weight_names.append(f"{weight.name}:0")

                h5_group = new["xception"][name_of_layer]
                weights_list = [np.array(h5_group[kk]) for kk in layer_weight_names]
                model.layers[xception_idx].layers[i].set_weights(weights_list)
    return model


def _create_image_generator(images: list, batch_size: int = 16) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    images = [i for i in images if os.path.splitext(i)[1].lower() in VALID_IMAGE_FORMATS]
    sizes = [get_image_size(i) for i in images]
    width = [i[0] for i in sizes]
    height = [i[1] for i in sizes]
    if len(images) == 0:
        raise ValueError(
            "No images found. Ensure files are one of: "
            + ", ".join(VALID_IMAGE_FORMATS)
        )

    image_df = pd.DataFrame({"Filenames": images})
    with warnings.catch_warnings():
        # throws warning about samplewise_std_normalization conflicting with
        # samplewise_center, which is not used here.
        warnings.simplefilter("ignore")
        image_generator = ImageDataGenerator(
            preprocessing_function=gray_scale, samplewise_std_normalization=True
        ).flow_from_dataframe(
            image_df,
            x_col="Filenames",
            y_col=None,
            target_size=(299, 299),
            batch_size=batch_size,
            colormode="rgb",
            shuffle=False,
            class_mode=None,
        )
    return image_generator, width, height


def load_images_from_path(image_path: str, batch_size: int = 16) -> np.ndarray:
    """
    Load the images from the given path
    :param image_path: The path to the images
    :type image_path: str
    :return: an Image generator for the found images
    :rtype: keras.preprocessing.image.ImageDataGenerator
    """
    if image_path is None or not os.path.isdir(image_path):
        raise ValueError("The path provided is not a directory")
    images = glob(os.path.join(image_path, "*"))
    return _create_image_generator(images, batch_size=batch_size)


def load_images_from_list(image_list: list, batch_size: int = 16) -> np.ndarray:
    """
    Load the images from the given list
    :param image_list: The list of images
    :type image_list: list
    :return: an Image generator for the found images
    :rtype: keras.preprocessing.image.ImageDataGenerator
    """
    if image_list is None:
        raise ValueError("image_list must not be None")
    return _create_image_generator(image_list, batch_size=batch_size)


def predictions_util(
    model: Sequential,
    image_generator: ImageDataGenerator,
    primary_weights: str,
    secondary_weights: str,
    ensemble: bool = False,
    species: str = "mouse",
    progress_callback=None,
    log_callback=None,
):
    """
    Predict the image alignments

    :param model: The model to use for prediction
    :param image_generator: The image generator to use for prediction
    :type model: keras.models.Sequential
    :type image_generator: keras.preprocessing.image.ImageDataGenerator
    :return: The predicted alignments
    :rtype: list
    """
    model = load_xception_weights(model, primary_weights, species)
    steps = int(np.ceil(image_generator.n / image_generator.batch_size))
    callbacks = None
    if progress_callback is not None:
        callbacks = [
            PredictionProgressCallback(
                total_images=image_generator.n,
                phase="primary",
                progress_callback=progress_callback,
            )
        ]
    if log_callback is not None:
        log_callback("Running primary inference pass")

    try:
        predictions = model.predict(
            image_generator,
            steps=steps,
            verbose=1,
            callbacks=callbacks,
        )
    except Exception as exc:
        raise RuntimeError(
            "Primary inference failed. Check image integrity, TensorFlow runtime, and available memory."
        ) from exc

    predictions = predictions.astype(np.float64)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Primary inference produced non-finite values (NaN/Inf)")

    if ensemble:
        if secondary_weights is None:
            raise ValueError(
                "secondary_weights is required when ensemble=True"
            )
        image_generator.reset()
        model = load_xception_weights(model, secondary_weights, species)
        callbacks = None
        if progress_callback is not None:
            callbacks = [
                PredictionProgressCallback(
                    total_images=image_generator.n,
                    phase="secondary",
                    progress_callback=progress_callback,
                )
            ]
        if log_callback is not None:
            log_callback("Running ensemble secondary inference pass")

        try:
            secondary_predictions = model.predict(
                image_generator,
                steps=steps,
                verbose=1,
                callbacks=callbacks,
            )
        except Exception as exc:
            raise RuntimeError(
                "Secondary inference failed during ensemble pass. Check model weights and runtime memory."
            ) from exc

        secondary_predictions = secondary_predictions.astype(np.float64)
        if not np.isfinite(secondary_predictions).all():
            raise RuntimeError(
                "Secondary inference produced non-finite values (NaN/Inf)"
            )
        predictions = np.mean([predictions, secondary_predictions], axis=0)

    if not np.isfinite(predictions).all():
        raise RuntimeError("Inference produced non-finite coordinates (NaN/Inf)")

    filenames = image_generator.filenames
    filenames = [os.path.basename(i) for i in filenames]
    predictions_df = pd.DataFrame(
        {
            "Filenames": filenames,
            "ox": predictions[:, 0],
            "oy": predictions[:, 1],
            "oz": predictions[:, 2],
            "ux": predictions[:, 3],
            "uy": predictions[:, 4],
            "uz": predictions[:, 5],
            "vx": predictions[:, 6],
            "vy": predictions[:, 7],
            "vz": predictions[:, 8],
        }
    )

    return predictions_df


def get_image_size(fname):
    """Return width and height for an image file using Pillow."""
    try:
        with Image.open(fname) as image:
            width, height = image.size
    except Exception as exc:
        raise ValueError(f"Unable to read image dimensions for '{fname}': {exc}") from exc
    return width, height
