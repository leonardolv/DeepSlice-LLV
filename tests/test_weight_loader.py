"""Tests for the neural network builder and the name-based weight loader."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def tf_module():
    """Skip this module when TensorFlow is unavailable (e.g., CPU-only CI)."""
    return pytest.importorskip("tensorflow")


@pytest.mark.parametrize("species", ["mouse", "rat"])
def test_initialise_network_produces_named_layers(tf_module, species):
    from DeepSlice.neural_network.neural_network import (
        DENSE_HEAD_LAYER_NAMES,
        XCEPTION_BASE_LAYER_NAME,
        initialise_network,
    )

    model = initialise_network(xception_weights=None, weights=None, species=species)

    for expected in (XCEPTION_BASE_LAYER_NAME, *DENSE_HEAD_LAYER_NAMES):
        model.get_layer(expected)


@pytest.mark.parametrize("species", ["mouse", "rat"])
def test_forward_pass_produces_9_vector(tf_module, species):
    import numpy as np

    from DeepSlice.neural_network.neural_network import initialise_network

    model = initialise_network(xception_weights=None, weights=None, species=species)
    dummy = np.zeros((1, 299, 299, 3), dtype=np.float32)
    output = model.predict(dummy, verbose=0)

    assert output.shape == (1, 9)
    assert np.isfinite(output).all()


def test_initialise_network_rejects_unknown_species(tf_module):
    from DeepSlice.neural_network.neural_network import initialise_network

    with pytest.raises(ValueError, match="species must be one of"):
        initialise_network(xception_weights=None, weights=None, species="hamster")


def test_load_xception_weights_rejects_unknown_species(tf_module):
    from DeepSlice.neural_network.neural_network import load_xception_weights

    with pytest.raises(ValueError, match="species must be one of"):
        load_xception_weights(model=None, weights="/tmp/ignored.h5", species="hamster")


def test_load_xception_weights_missing_dense_layer_raises(tf_module, tmp_path):
    """The loader must emit a clear error when the model lacks expected layers."""
    import h5py
    import numpy as np
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Input

    from DeepSlice.neural_network.neural_network import load_xception_weights

    inputs = Input(shape=(4,))
    outputs = Dense(2, name="unrelated")(inputs)
    bad_model = Model(inputs, outputs)

    weights_path = tmp_path / "weights.h5"
    with h5py.File(weights_path, "w") as file_handle:
        group = file_handle.create_group("dense/dense")
        group.create_dataset("kernel:0", data=np.zeros((2048, 256), dtype=np.float32))
        group.create_dataset("bias:0", data=np.zeros((256,), dtype=np.float32))

    with pytest.raises(RuntimeError, match="missing expected layer 'dense'"):
        load_xception_weights(bad_model, str(weights_path))
