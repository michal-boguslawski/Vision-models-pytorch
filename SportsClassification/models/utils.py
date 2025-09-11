from collections import OrderedDict
import glob
import os
from typing import Literal, Any
import torch as T
from torch import nn
from utils.aws_handler import AWSHandler
from utils.helpers import filter_kwargs
from utils.logger import SingletonLogger


logger_instance = SingletonLogger()


class ModelHandler:
    LOAD_SOURCE_DICT = {
        "local": "_load_state_dict_from_local",
        "s3": "_load_state_dict_from_s3"
    }

    def load_weights(
        self,
        model: nn.Module,
        source: Literal["local", "hugging_face", "s3"] | None = None,
        version_name: str = "",
        model_part: Literal["all", "backbone", "detection_head"] = "all",
        checkpoint_dir: str | None = None,
        project_name: str | None = None,
        device: str = "cpu",
        num_layers: int = 0,
        **kwargs: dict[str, Any]
    ):
        if source is None:
            print("Loading weights criteria are not defined.")
            return
        fn_name = self.LOAD_SOURCE_DICT[source]
        fn = getattr(self, fn_name)
        state_dict = fn(version_name=version_name, checkpoint_dir=checkpoint_dir, project_name=project_name, device=device, **filter_kwargs(fn, kwargs))
        self._load_weights(model=model, state_dict=state_dict, model_part=model_part)
        
    
    @staticmethod
    def freeze_weights(model: nn.Module, state_dict: OrderedDict[str, T.Tensor] | dict[str, T.Tensor]):
        for name, param in model.named_parameters():
            if name in state_dict:
                param.requires_grad = False
        logger_instance.logger.info("Loaded weights were frozen")

    @staticmethod
    def unfreeze_weights(model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True
        logger_instance.logger.info("Weights were unfrozen")

    @staticmethod
    def _load_state_dict_from_local(
        checkpoint_dir: str,
        project_name: str,
        version_type: str = "latest",
        version_name: str = "",
        device: str = "cpu"
    ) -> OrderedDict[str, T.Tensor] | None:
        files = glob.glob(os.path.join(checkpoint_dir, project_name, f"*{version_name}*", "*.pth"))
        
        # Make sure the directory is not empty
        if not files:
            logger_instance.logger.info("State dict not found")
            return None
        else:
            # Get the newest file
            newest_state_dict = max(files, key=os.path.getctime)

        # load selected state_dict
        loaded_state_dict = T.load(newest_state_dict, map_location=T.device(device))
        logger_instance.logger.info(f"State dict loaded from {newest_state_dict}")
        return loaded_state_dict

    def _load_weights(
        self,
        model: nn.Module,
        state_dict: OrderedDict[str, T.Tensor] | None,
        model_part: Literal["all", "backbone", "detection_head"],
        if_freeze: bool = True
    ):
        if state_dict is None:
            logger_instance.logger.info("No weights available")
            return

        new_state_dict = {k: v for k, v in state_dict.items() if ( model_part in k or model_part == "all")}
        model.load_state_dict(new_state_dict, strict=False)
        logger_instance.logger.info("Weights loaded")

        if if_freeze:
            self.freeze_weights(model=model, state_dict=new_state_dict)

    def _load_state_dict_from_s3(
        self,
        s3_bucket_name: str,
        version_name: str,
        checkpoint_dir: str,
        project_name: str,
        device: str = "cpu",
    ) -> OrderedDict[str, T.Tensor]:
        pass
