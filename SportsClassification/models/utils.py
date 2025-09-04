from collections import OrderedDict
import glob
import os
from typing import Literal
import torch as T
from torch import nn


class ModelHandler:
    LOAD_SOURCE_DICT = {
        "local": "_load_state_dict_from_local"
    }

    def load_weights(
        self,
        model: nn.Module,
        source: Literal["local", "hugging_face", "s3"] | None = None,
        version_name: str = "",
        model_part: Literal["all", "backbone", "detection_head"] = "all",
        num_layers: int = 0,
        *args,
        **kwargs
    ):
        if source is None:
            print("Loading weights criteria are not defined.")
            return
        fn_name = self.LOAD_SOURCE_DICT[source]
        fn = getattr(self, fn_name)
        state_dict = fn(version_name=version_name, **kwargs)
        self._load_weights(model=model, state_dict=state_dict, model_part=model_part)
        
    
    @staticmethod
    def freeze_weights(model: nn.Module, state_dict: OrderedDict | dict):
        for name, param in model.named_parameters():
            if name in state_dict:
                param.requires_grad = False
        print("Loaded weights were frozen")

    @staticmethod
    def unfreeze_weights(model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True
        print("Weights were unfrozen")

    @staticmethod
    def _load_weights_from_s3(s3_bucket_name: str, type: str = "latest", version_name: str = "") -> OrderedDict | None:
        pass

    @staticmethod
    def _load_state_dict_from_local(
        checkpoint_dir: str,
        project_name: str,
        version_type: str = "latest",
        version_name: str = "",
        *args,
        **kwargs
    ) -> OrderedDict | None:
        files = glob.glob(os.path.join(checkpoint_dir, project_name, f"*{version_name}*", "*.pth"))
        
        # Make sure the directory is not empty
        if not files:
            print("No files found")
            return None
        else:
            # Get the newest file
            newest_state_dict = max(files, key=os.path.getctime)

        # load selected state_dict
        loaded_state_dict = T.load(newest_state_dict)
        print(f"State dict loaded from {newest_state_dict}")
        return loaded_state_dict

    def _load_weights(
        self,
        model: nn.Module,
        state_dict: OrderedDict,
        model_part: Literal["all", "backbone", "detection_head"],
        if_freeze: bool = True
    ):
        new_state_dict = {k: v for k, v in state_dict.items() if ( model_part in k or model_part == "all")}
        model.load_state_dict(new_state_dict, strict=False)
        print("Weights loaded")

        if if_freeze:
            self.freeze_weights(model=model, state_dict=new_state_dict)
