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
from ..diagnostics import monitored


VALID_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class EnsemblePartialResultError(RuntimeError):
    """Raised when ensemble secondary inference fails but primary predictions are available."""

    def __init__(self, message: str, partial_predictions: pd.DataFrame):
        super().__init__(message)
        self.partial_predictions = partial_predictions


class PredictionProgressCallback(tf.keras.callbacks.Callback):
    """Keras callback used to expose prediction progress to the GUI."""

    def __init__(self, total_images, phase, progress_callback, cancel_check=None, batch_size: int = 1):
        super().__init__()
        self.total_images = total_images
        self.phase = phase
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.batch_size = max(int(batch_size), 1)

    def _raise_if_cancelled(self):
        if self.cancel_check is not None and bool(self.cancel_check()):
            raise RuntimeError("Prediction cancelled by user")

    def on_predict_batch_begin(self, batch, logs=None):
        self._raise_if_cancelled()

    def on_predict_batch_end(self, batch, logs=None):
        self._raise_if_cancelled()
        if self.progress_callback is None:
            return
        completed = min((batch + 1) * self.batch_size, self.total_images)
        self.progress_callback(completed, self.total_images, self.phase)


def gray_scale(img: np.ndarray) -> np.ndarray:
    """
    Convert the image to grayscale

    :param img: The image to convert
    :type img: numpy.ndarray
    :return: The converted image
    :rtype: numpy.ndarray
    """
    h, w = img.shape[:2]
    img = rgb2gray(img).reshape(h, w, 1)
    return img


@monitored("DS-006")
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
        base_model_layer = base_model(inputs, training=False)
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
        xception_layer = None
        dense_layers = []
        for layer in model.layers:
            if layer.name == "xception":
                xception_layer = layer
            elif "dense" in layer.name:
                dense_layers.append(layer)

        if xception_layer is None:
            raise RuntimeError("xception layer not found in model")

        if len(dense_layers) != 3:
            raise RuntimeError(f"Expected 3 dense layers, found {len(dense_layers)}")

        dense_layers[0].set_weights([new["dense"]["dense"]["kernel:0"], new["dense"]["dense"]["bias:0"]])
        dense_layers[1].set_weights([new["dense_1"]["dense_1"]["kernel:0"], new["dense_1"]["dense_1"]["bias:0"]])
        dense_layers[2].set_weights([new["dense_2"]["dense_2"]["kernel:0"], new["dense_2"]["dense_2"]["bias:0"]])

        # Set the weights of the xception model
        weight_names = new["xception"].attrs["weight_names"].tolist()
        weight_names_layers = {
            name.decode("utf-8").split("/")[0] for name in weight_names
        }
        updated_xception_layers = set()

        for i in range(len(xception_layer.layers)):
            name_of_layer = xception_layer.layers[i].name
            # if layer name is in the weight names, then we will set weights
            if name_of_layer in weight_names_layers:
                # Get name of weights in the layer
                layer_weight_names = []
                for weight in xception_layer.layers[i].weights:
                    try:
                        # Find the matching name without caring about index
                        raw_name = weight.name.split("/")[-1]
                        # E.g. 'kernel' from 'kernel:0' or 'kernel'
                        layer_weight_names.append(raw_name.split(":")[0])
                    except IndexError:
                        layer_weight_names.append(weight.name)

                # Get weights from the new model
                new_weights = []
                for j in range(len(layer_weight_names)):
                    # try exact name then name + :0
                    weight_val = None
                    if layer_weight_names[j] in new["xception"][name_of_layer]:
                        weight_val = new["xception"][name_of_layer][layer_weight_names[j]]
                    elif f"{layer_weight_names[j]}:0" in new["xception"][name_of_layer]:
                        weight_val = new["xception"][name_of_layer][f"{layer_weight_names[j]}:0"]

                    if weight_val is not None:
                        new_weights.append(weight_val)

                if len(new_weights) > 0:
                    xception_layer.layers[i].set_weights(new_weights)
                    updated_xception_layers.add(name_of_layer)

        if not updated_xception_layers:
            raise RuntimeError("No xception layers were updated. Weights file may be incompatible.")
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
    image_generator.deepslice_width = width
    image_generator.deepslice_height = height
    image_generator.deepslice_paths = list(images)
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


def _resolve_generator_metadata(image_generator: ImageDataGenerator):
    source_paths = getattr(image_generator, "deepslice_paths", None)
    if source_paths is None or len(source_paths) != len(image_generator.filenames):
        source_paths = getattr(image_generator, "filepaths", None)
    if source_paths is None or len(source_paths) != len(image_generator.filenames):
        directory = getattr(image_generator, "directory", None)
        if directory:
            source_paths = [
                os.path.join(directory, relative_path)
                for relative_path in image_generator.filenames
            ]
        else:
            source_paths = list(image_generator.filenames)
    else:
        source_paths = list(source_paths)

    width = getattr(image_generator, "deepslice_width", None)
    height = getattr(image_generator, "deepslice_height", None)
    if (
        width is None
        or height is None
        or len(width) != len(source_paths)
        or len(height) != len(source_paths)
    ):
        sizes = [get_image_size(path) for path in source_paths]
        width = [size[0] for size in sizes]
        height = [size[1] for size in sizes]

    return source_paths, width, height


def _build_predictions_dataframe(
    image_generator: ImageDataGenerator,
    predictions: np.ndarray,
    ensemble_delta_mean_abs: np.ndarray = None,
    ensemble_delta_max_abs: np.ndarray = None,
) -> pd.DataFrame:
    source_paths, width, height = _resolve_generator_metadata(image_generator)
    filenames = [os.path.basename(path) for path in source_paths]
    dataframe = pd.DataFrame(
        {
            "Filenames": filenames,
            "width": width,
            "height": height,
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

    if ensemble_delta_mean_abs is not None:
        dataframe["ensemble_delta_mean_abs"] = np.asarray(ensemble_delta_mean_abs, dtype=float)
    if ensemble_delta_max_abs is not None:
        dataframe["ensemble_delta_max_abs"] = np.asarray(ensemble_delta_max_abs, dtype=float)

    return dataframe


def _validate_prediction_matrix(predictions: np.ndarray, phase: str):
    matrix = np.asarray(predictions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 9:
        raise RuntimeError(
            f"{phase} inference produced invalid coordinate shape {matrix.shape}; expected (N, 9)"
        )
    if matrix.shape[0] == 0:
        raise RuntimeError(f"{phase} inference produced no predictions")
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{phase} inference produced non-finite values (NaN/Inf)")
    return matrix


def predictions_util(
    model: Sequential,
    image_generator: ImageDataGenerator,
    primary_weights: str,
    secondary_weights: str,
    ensemble: bool = False,
    species: str = "mouse",
    progress_callback=None,
    log_callback=None,
    cancel_check=None,
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
                cancel_check=cancel_check,
                batch_size=image_generator.batch_size,
            )
        ]
    if cancel_check is not None and bool(cancel_check()):
        raise RuntimeError("Prediction cancelled by user")
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

    predictions = _validate_prediction_matrix(predictions, phase="Primary")

    ensemble_delta_mean_abs = None
    ensemble_delta_max_abs = None

    if ensemble:
        if secondary_weights is None:
            raise ValueError(
                "secondary_weights is required when ensemble=True"
            )
        if cancel_check is not None and bool(cancel_check()):
            raise RuntimeError("Prediction cancelled by user")
        image_generator.reset()
        model = load_xception_weights(model, secondary_weights, species)
        callbacks = None
        if progress_callback is not None:
            callbacks = [
                PredictionProgressCallback(
                    total_images=image_generator.n,
                    phase="secondary",
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    batch_size=image_generator.batch_size,
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
            partial_predictions = _build_predictions_dataframe(image_generator, predictions)
            raise EnsemblePartialResultError(
                "Secondary inference failed during ensemble pass. Primary predictions are available for recovery.",
                partial_predictions,
            ) from exc

        try:
            secondary_predictions = _validate_prediction_matrix(secondary_predictions, phase="Secondary")
        except Exception as exc:
            partial_predictions = _build_predictions_dataframe(image_generator, predictions)
            raise EnsemblePartialResultError(
                "Secondary inference produced invalid outputs. Primary predictions are available for recovery.",
                partial_predictions,
            ) from exc

        ensemble_abs_delta = np.abs(predictions - secondary_predictions)
        ensemble_delta_mean_abs = np.mean(ensemble_abs_delta, axis=1)
        ensemble_delta_max_abs = np.max(ensemble_abs_delta, axis=1)
        predictions = np.mean([predictions, secondary_predictions], axis=0)

    predictions = _validate_prediction_matrix(predictions, phase="Final")

    predictions_df = _build_predictions_dataframe(
        image_generator,
        predictions,
        ensemble_delta_mean_abs=ensemble_delta_mean_abs,
        ensemble_delta_max_abs=ensemble_delta_max_abs,
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
