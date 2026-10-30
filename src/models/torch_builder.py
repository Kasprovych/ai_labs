"""PyTorch model builders for Laboratory Work #2, mirroring the Keras builders of Lab 1.

The same Builder abstraction is kept so both frameworks are compared on identical architectures:
- Baseline FFN: Linear -> act (x2) -> Linear(1).
- Regularized: [Linear(no bias) -> BatchNorm1d -> act -> Dropout] (x2) -> Linear(1).

Keras defaults are reproduced explicitly so that differences in results come from the framework,
not from silently different hyperparameters:
- weights: Glorot (Xavier) uniform, biases: zeros (PyTorch default is Kaiming uniform);
- BatchNorm: momentum 0.99 in Keras == momentum 0.01 in PyTorch, epsilon 1e-3;
- LeakyReLU: negative slope 0.3 (Keras 3 default; PyTorch default is 0.01);
- GELU: exact erf formulation (default in both frameworks);
- L2: Keras adds l2 * sum(W^2) of the kernels to the loss. PyTorch has no layer-level regularizer,
  so the network exposes `regularization_loss()` and the trainer adds it to the loss.
  (Adam's `weight_decay` is not equivalent: it also decays BN/bias params and skips the loss value.)
"""

from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import torch
from torch import nn

ActivationName = Literal["relu", "leaky_relu", "gelu"]

KERAS_LEAKY_RELU_SLOPE = 0.3
KERAS_BN_MOMENTUM = 0.01  # PyTorch convention: 1 - Keras momentum (0.99)
KERAS_BN_EPSILON = 1e-3


def make_activation(name: ActivationName) -> nn.Module:
    """Factory for the hidden-layer activation functions under study."""
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=KERAS_LEAKY_RELU_SLOPE)
    if name == "gelu":
        return nn.GELU(approximate="none")
    return nn.ReLU()


class FeedForwardNet(nn.Module):
    """Binary classifier MLP that outputs logits (sigmoid is applied in the loss / at inference).

    Returning logits and using BCEWithLogitsLoss is the numerically stable PyTorch idiom,
    equivalent to Keras' sigmoid output + binary_crossentropy.
    """

    def __init__(self, layers: nn.Sequential, l2_factor: float = 0.0) -> None:
        super().__init__()
        self.layers = layers
        self._l2_factor = l2_factor
        self._init_keras_like()

    def _init_keras_like(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)

    def regularization_loss(self) -> torch.Tensor:
        """L2 penalty on hidden Linear kernels, identical to Keras kernel_regularizer=l2(factor)."""
        hidden = [m for m in self.layers if isinstance(m, nn.Linear)][:-1]
        if self._l2_factor == 0.0 or not hidden:
            return torch.zeros(())
        return self._l2_factor * sum(m.weight.pow(2).sum() for m in hidden)


class ITorchModelBuilder(ABC):
    """Abstract Builder interface for PyTorch networks (counterpart of IModelBuilder)."""

    @abstractmethod
    def build(self, input_dim: int) -> FeedForwardNet:
        """Constructs a freshly initialized network."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this architecture and activation variant."""
        pass

    @property
    @abstractmethod
    def activation(self) -> str:
        """Activation function name used in hidden layers."""
        pass


class TorchBaselineModelBuilder(ITorchModelBuilder):
    """Baseline feedforward network: at least 2 hidden layers, no regularization."""

    def __init__(
        self,
        activation: ActivationName = "relu",
        hidden_units: Optional[List[int]] = None,
        model_name: Optional[str] = None
    ) -> None:
        self._activation = activation
        self._hidden_units = hidden_units or [64, 32]
        self._custom_name = model_name or f"Baseline_FFN_{activation}"

    @property
    def name(self) -> str:
        return self._custom_name

    @property
    def activation(self) -> str:
        return self._activation

    def build(self, input_dim: int) -> FeedForwardNet:
        modules: List[nn.Module] = []
        in_features = input_dim
        for units in self._hidden_units:
            modules += [nn.Linear(in_features, units), make_activation(self._activation)]
            in_features = units
        modules.append(nn.Linear(in_features, 1))
        return FeedForwardNet(nn.Sequential(*modules))


class TorchRegularizedModelBuilder(ITorchModelBuilder):
    """Regularized network with Batch Normalization, Dropout and L2 weight penalty."""

    def __init__(
        self,
        activation: ActivationName = "relu",
        hidden_units: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        l2_factor: float = 1e-4,
        model_name: Optional[str] = None
    ) -> None:
        self._activation = activation
        self._hidden_units = hidden_units or [64, 32]
        self._dropout_rate = dropout_rate
        self._l2_factor = l2_factor
        self._custom_name = model_name or f"Regularized_BN_Dropout_L2_{activation}"

    @property
    def name(self) -> str:
        return self._custom_name

    @property
    def activation(self) -> str:
        return self._activation

    def build(self, input_dim: int) -> FeedForwardNet:
        """Linear(no bias) -> BatchNorm -> Activation -> Dropout per hidden block."""
        modules: List[nn.Module] = []
        in_features = input_dim
        for units in self._hidden_units:
            modules += [
                # bias is redundant before BatchNorm, which has its own shift (beta)
                nn.Linear(in_features, units, bias=False),
                nn.BatchNorm1d(units, momentum=KERAS_BN_MOMENTUM, eps=KERAS_BN_EPSILON),
                make_activation(self._activation),
                nn.Dropout(self._dropout_rate),
            ]
            in_features = units
        modules.append(nn.Linear(in_features, 1))
        return FeedForwardNet(nn.Sequential(*modules), l2_factor=self._l2_factor)
