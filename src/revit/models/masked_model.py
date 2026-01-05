import torch
from torch.nn.utils.parametrize import (
    is_parametrized,
    register_parametrization,
    remove_parametrizations,
)

from .base_model import BaseLLMModel
from .gpt2_xl import GPT2XLModel
from .llama import LLamaModel
from .model_enum import ModelName
from .weight_mask import ElementwiseMask

MODEL_NAME = ModelName.LLAMA3_3B
TARGET_LAYER = 5


class MaskedModel(BaseLLMModel):
    def __init__(self, model_name: ModelName, target_layer: int = TARGET_LAYER):
        self.model_name = model_name
        super().__init__(model_name, target_layer)

    @staticmethod
    def _init_model(target_layer: int = TARGET_LAYER):
        if MODEL_NAME == ModelName.LLAMA3_3B:
            return LLamaModel._init_model(target_layer)
        return GPT2XLModel._init_model(target_layer)

    def _get_target_weight(self, layer: int = None) -> torch.nn.Parameter:
        """Return the target MLP weight parameter."""
        if layer is None:
            layer = self.target_layer
        if MODEL_NAME == ModelName.LLAMA3_3B:
            return getattr(self.llm.model.layers[layer].mlp.down_proj, "weight")
        else:
            return getattr(self.llm.transformer.h[layer].mlp.c_proj, "weight")

    def _set_weight_to_layer(self, weight: torch.Tensor, layer: int):
        """Replace the MLP weight at the given layer."""
        if MODEL_NAME == ModelName.LLAMA3_3B:
            mod = self.llm.model.layers[layer].mlp.down_proj
        else:
            mod = self.llm.transformer.h[layer].mlp.c_proj
        with torch.no_grad():
            if is_parametrized(mod, "weight"):
                mod.parametrizations.weight.original.copy_(weight)
            else:
                mod.weight.copy_(weight)

    def attach_mask_parametrization(self, mask_getter, *, use_weight_getter=True):
        target = self.mlp_out_attr
        if is_parametrized(target, "weight"):
            remove_parametrizations(target, "weight", leave_parametrized=False)

        if use_weight_getter:
            self._current_weight = (
                self._get_target_weight().detach().clone().to(self.device)
            )

            def weight_getter():
                return self._current_weight

            register_parametrization(
                target,
                "weight",
                ElementwiseMask(mask_getter, weight_getter=weight_getter),
            )
        else:
            register_parametrization(target, "weight", ElementwiseMask(mask_getter))

        target.parametrizations.weight.original.requires_grad_(False)

    def set_current_weight(self, weight: torch.Tensor):
        self._current_weight = weight.detach()
