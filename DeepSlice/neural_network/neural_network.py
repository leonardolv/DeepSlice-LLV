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


DENSE_HEAD_LAYER_NAMES = ("dense", "dense_1", "dense_2")
XCEPTION_BASE_LAYER_NAME = "xception"
VALID_SPECIES = ("mouse", "rat")


@monitored("DS-006")
def initialise_network(xception_weights: str, weights: str, species: str) -> Model:
    """
    Initialise a neural network with the given weights.

    Both species expose the same named layers (``xception``, ``dense``,
    ``dense_1``, ``dense_2``) so weight loading is species-agnostic.

    :param xception_weights: Path to the Xception ImageNet weights, or None.
    :param weights: Path to the DeepSlice head weights, or None.
    :param species: "mouse" or "rat".
    :return: The initialised neural network.
    """
    if species not in VALID_SPECIES:
        raise ValueError(
            f"species must be one of {VALID_SPECIES!r}, got {species!r}"
        )

    base_model = Xception(
        include_top=True, weights=xception_weights, name=XCEPTION_BASE_LAYER_NAME
    )

    if species == "rat":
        inputs = Input(shape=(299, 299, 3), name="image_in")
        x = base_model(inputs, training=False)
        x = Dense(256, activation="relu", name="dense")(x)
        x = Dense(256, activation="relu", name="dense_1")(x)
        out = Dense(9, activation="linear", name="dense_2")(x)
        model = Model(inputs=inputs, outputs=out, name="deepslice_rat")
    else:
        model = Sequential(name="deepslice_mouse")
        model.add(base_model)
        model.add(Dense(256, activation="relu", name="dense"))
        model.add(Dense(256, activation="relu", name="dense_1"))
        model.add(Dense(9, activation="linear", name="dense_2"))

    if weights is not None:
        model = load_xception_weights(model, weights, species)
    return model


def _get_named_layer(model, name: str):
    """Resolve a layer by name with a clear error if it is missing."""
    try:
        return model.get_layer(name)
    except ValueError as exc:
        available = ", ".join(layer.name for layer in model.layers)
        raise RuntimeError(
            f"Model is missing expected layer '{name}'. "
            f"Rebuild the model with initialise_network(). "
            f"Available layers: {available}"
        ) from exc


def load_xception_weights(model, weights, species=None):
    """
    Load the Xception base and dense-head weights from an HDF5 file into
    ``model``. Layers are resolved by name, so mouse (Sequential) and rat
    (Functional) builds share a single code path.

    The ``species`` parameter is accepted for API compatibility but no longer
    influences loading. It is validated if provided.
    """
    if species is not None and species not in VALID_SPECIES:
        raise ValueError(
            f"species must be one of {VALID_SPECIES!r} or None, got {species!r}"
        )

    with h5py.File(weights, "r") as new:
        for group_name in DENSE_HEAD_LAYER_NAMES:
            layer = _get_named_layer(model, group_name)
            kernel = np.array(new[group_name][group_name]["kernel:0"])
            bias = np.array(new[group_name][group_name]["bias:0"])
            layer.set_weights([kernel, bias])

        xception_layer = _get_named_layer(model, XCEPTION_BASE_LAYER_NAME)

        weight_names = new[XCEPTION_BASE_LAYER_NAME].attrs["weight_names"].tolist()
        weight_names_layers = {
            name.decode("utf-8").split("/")[0] for name in weight_names
        }
        updated_xception_layers = set()

        for sub_layer in xception_layer.layers:
            name_of_layer = sub_layer.name
            if name_of_layer not in weight_names_layers:
                continue

            # Normalise live weight names (e.g. 'kernel:0' -> 'kernel') so that
            # HDF5 files saved with either naming convention are supported.
            layer_weight_names = []
            for weight in sub_layer.weights:
                raw_name = weight.name.split("/")[-1]
                layer_weight_names.append(raw_name.split(":")[0])

            h5_group = new[XCEPTION_BASE_LAYER_NAME][name_of_layer]
            new_weights = []
            for key in layer_weight_names:
                if key in h5_group:
                    new_weights.append(np.array(h5_group[key]))
                elif f"{key}:0" in h5_group:
                    new_weights.append(np.array(h5_group[f"{key}:0"]))

            if len(new_weights) == len(layer_weight_names):
                sub_layer.set_weights(new_weights)
                updated_xception_layers.add(name_of_layer)

        if len(updated_xception_layers) != len(weight_names_layers):
            missing_layers = sorted(weight_names_layers - updated_xception_layers)
            missing_preview = ", ".join(missing_layers[:10])
            if len(missing_layers) > 10:
                missing_preview += f", ... (+{len(missing_layers) - 10} more)"
            raise RuntimeError(
                "Xception weight loading incomplete. "
                f"Updated {len(updated_xception_layers)}/{len(weight_names_layers)} layers. "
                f"Missing layers: {missing_preview}"
            )
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
