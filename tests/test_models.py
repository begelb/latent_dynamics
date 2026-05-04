"""Tests for the autoencoder components."""

from __future__ import annotations

import torch
from torch import nn

from latentdynamics.config.schema import ArchConfig
from latentdynamics.models import (
    Decoder,
    Encoder,
    LatentDynamicsAutoencoder,
    LatentMap,
    build_autoencoder,
    hidden_activation,
    terminal_activation,
)


def _arch(activation: str = "relu", **kwargs) -> ArchConfig:
    base = {
        "num_layers": 2,
        "hidden_shape": 8,
        "high_dims": 4,
        "low_dims": 2,
        "activation": activation,
        "encoder_out_activation": "tanh",
        "decoder_out_activation": "sigmoid",
    }
    base.update(kwargs)
    return ArchConfig(**base)


class TestActivations:
    def test_hidden_relu(self):
        assert isinstance(hidden_activation("relu"), nn.ReLU)

    def test_hidden_tanh(self):
        assert isinstance(hidden_activation("tanh"), nn.Tanh)

    def test_hidden_unknown_raises(self):
        try:
            hidden_activation("swish")
        except KeyError:
            return
        raise AssertionError("expected KeyError")

    def test_terminal_none_returns_none(self):
        assert terminal_activation("none") is None

    def test_terminal_sigmoid(self):
        assert isinstance(terminal_activation("sigmoid"), nn.Sigmoid)


class TestEncoderDecoderShapes:
    def test_encoder_output_shape(self):
        enc = Encoder(_arch())
        x = torch.randn(7, 4)
        assert enc(x).shape == (7, 2)

    def test_decoder_output_shape(self):
        dec = Decoder(_arch())
        z = torch.randn(7, 2)
        assert dec(z).shape == (7, 4)

    def test_latent_map_output_shape(self):
        lm = LatentMap(_arch())
        z = torch.randn(7, 2)
        assert lm(z).shape == (7, 2)


class TestActivationHonoured:
    def test_relu_present_when_relu_selected(self):
        m = build_autoencoder(_arch(activation="relu"))
        assert any(isinstance(layer, nn.ReLU) for layer in m.encoder.net)
        assert not any(isinstance(layer, nn.Tanh) and idx < len(m.encoder.net) - 1
                       for idx, layer in enumerate(m.encoder.net))

    def test_tanh_present_when_tanh_selected(self):
        m = build_autoencoder(_arch(activation="tanh"))
        # Hidden activations should be Tanh, not ReLU.
        assert not any(isinstance(layer, nn.ReLU) for layer in m.encoder.net)
        # At least one Tanh hidden layer.
        hidden_tanhs = [
            layer for idx, layer in enumerate(m.encoder.net)
            if isinstance(layer, nn.Tanh) and idx < len(m.encoder.net) - 1
        ]
        assert len(hidden_tanhs) >= 1

    def test_decoder_terminal_none(self):
        m = build_autoencoder(_arch(decoder_out_activation="none"))
        # Final layer is Linear (no terminal activation).
        assert isinstance(m.decoder.net[-1], nn.Linear)

    def test_decoder_terminal_sigmoid(self):
        m = build_autoencoder(_arch(decoder_out_activation="sigmoid"))
        assert isinstance(m.decoder.net[-1], nn.Sigmoid)

    def test_encoder_terminal_tanh(self):
        m = build_autoencoder(_arch(encoder_out_activation="tanh"))
        assert isinstance(m.encoder.net[-1], nn.Tanh)

    def test_component_hidden_shapes_and_terminals_are_honoured(self):
        arch = _arch(
            num_layers=1,
            hidden_shape=4,
            latent_out_activation="none",
            encoder={"hidden_shapes": [7, 5], "out_activation": "none"},
            latent_map={"hidden_shapes": [3, 3], "activation": "tanh"},
            decoder={"hidden_shapes": [6], "out_activation": "none"},
        )
        m = build_autoencoder(arch)

        encoder_linears = [layer for layer in m.encoder.net if isinstance(layer, nn.Linear)]
        latent_linears = [layer for layer in m.latent_map.net if isinstance(layer, nn.Linear)]
        decoder_linears = [layer for layer in m.decoder.net if isinstance(layer, nn.Linear)]

        assert [layer.out_features for layer in encoder_linears[:-1]] == [7, 5]
        assert [layer.out_features for layer in latent_linears[:-1]] == [3, 3]
        assert [layer.out_features for layer in decoder_linears[:-1]] == [6]
        assert isinstance(m.encoder.net[-1], nn.Linear)
        assert isinstance(m.latent_map.net[-1], nn.Linear)
        assert isinstance(m.decoder.net[-1], nn.Linear)
        assert any(isinstance(layer, nn.Tanh) for layer in m.latent_map.net)


class TestForwardPass:
    def test_forward_returns_all_intermediate_tensors(self):
        m = build_autoencoder(_arch())
        x_t = torch.randn(5, 4)
        x_tau = torch.randn(5, 4)
        fp = m(x_t, x_tau)
        assert fp.x_t.shape == (5, 4)
        assert fp.x_t_hat.shape == (5, 4)
        assert fp.z_t.shape == (5, 2)
        assert fp.z_tau.shape == (5, 2)
        assert fp.z_tau_pred.shape == (5, 2)
        assert fp.x_tau_hat.shape == (5, 4)


class TestBuildAutoencoder:
    def test_returns_full_module(self):
        m = build_autoencoder(_arch())
        assert isinstance(m, LatentDynamicsAutoencoder)
        assert isinstance(m.encoder, Encoder)
        assert isinstance(m.latent_map, LatentMap)
        assert isinstance(m.decoder, Decoder)

    def test_state_dict_roundtrip(self, tmp_path):
        a = build_autoencoder(_arch())
        b = build_autoencoder(_arch())
        # Different random init.
        x = torch.randn(3, 4)
        assert not torch.allclose(a.encoder(x), b.encoder(x))
        b.load_state_dict(a.state_dict())
        torch.testing.assert_close(a.encoder(x), b.encoder(x))
