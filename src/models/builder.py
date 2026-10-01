"""Model building abstractions implementing Builder & Abstract Factory patterns (GoF).

Adheres to:
- Open/Closed Principle (OCP): New model architectures can be added without modifying existing code.
- Liskov Substitution Principle (LSP): Any builder produces a standard `tf.keras.Model` interchangeable in training.
- Interface Segregation Principle (ISP): Clean, minimal contract for model construction.
"""

from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


class IModelBuilder(ABC):
    """Abstract Builder interface defining the contract for neural network creation."""

    @abstractmethod
    def build(self, input_dim: int) -> tf.keras.Model:
        """Constructs and returns an uncompiled or compiled Keras model."""
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


class BaselineModelBuilder(IModelBuilder):
    """Builder for Baseline Feedforward Neural Network (Multi-Layer Perceptron).

    Features:
    - Minimum 2 hidden layers (as required by specification).
    - No Batch Normalization, Dropout, or L2 regularization.
    - Demonstrates baseline learning capability and potential overfitting risk.
    """

    def __init__(
        self,
        activation: Literal["relu", "leaky_relu", "gelu"] = "relu",
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

    def build(self, input_dim: int) -> tf.keras.Model:
        """Builds a sequential baseline network."""
        model = models.Sequential(name=self.name)
        model.add(layers.Input(shape=(input_dim,), name="input_layer"))

        for idx, units in enumerate(self._hidden_units, start=1):
            if self._activation == "leaky_relu":
                model.add(layers.Dense(units, name=f"dense_{idx}"))
                model.add(layers.LeakyReLU(name=f"leaky_relu_{idx}"))
            elif self._activation == "gelu":
                model.add(layers.Dense(units, activation="gelu", name=f"dense_{idx}"))
            else:
                model.add(layers.Dense(units, activation="relu", name=f"dense_{idx}"))

        # Output layer for binary classification
        model.add(layers.Dense(1, activation="sigmoid", name="output_layer"))
        return model


class RegularizedModelBuilder(IModelBuilder):
    """Builder for Advanced Regularized Neural Network.

    Features (as required by specification):
    - Batch Normalization (stabilizes hidden layer activation distributions).
    - Dropout (stochastic regularization to prevent co-adaptation of neurons).
    - L2 Kernel Regularization (weight decay penalty on high weights).
    """

    def __init__(
        self,
        activation: Literal["relu", "leaky_relu", "gelu"] = "relu",
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

    def build(self, input_dim: int) -> tf.keras.Model:
        """Builds advanced network with Dense -> BatchNorm -> Activation -> Dropout."""
        model = models.Sequential(name=self.name)
        model.add(layers.Input(shape=(input_dim,), name="input_layer"))

        l2_reg = regularizers.l2(self._l2_factor)

        for idx, units in enumerate(self._hidden_units, start=1):
            # Dense layer with L2 weight regularization
            model.add(layers.Dense(units, kernel_regularizer=l2_reg, use_bias=False, name=f"dense_{idx}"))
            # Batch Normalization layer
            model.add(layers.BatchNormalization(name=f"batch_norm_{idx}"))
            # Activation function
            if self._activation == "leaky_relu":
                model.add(layers.LeakyReLU(name=f"leaky_relu_{idx}"))
            elif self._activation == "gelu":
                model.add(layers.Activation("gelu", name=f"gelu_{idx}"))
            else:
                model.add(layers.Activation("relu", name=f"relu_{idx}"))
            # Dropout layer
            model.add(layers.Dropout(self._dropout_rate, name=f"dropout_{idx}"))

        # Output layer with sigmoid for binary fraud probability
        model.add(layers.Dense(1, activation="sigmoid", name="output_layer"))
        return model
