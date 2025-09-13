from abc import ABC, abstractmethod
from collections import OrderedDict
import glob
from io import BytesIO
import os
from typing import Literal, Any
import torch as T
from torch import nn
from utils.aws_handler import AWSHandler
from utils.helpers import filter_kwargs
from utils.metaclass import SingletonMeta


class AbstractModelHandler(ABC):
    from utils.logger import SingletonLogger
    logger_instance = SingletonLogger()

    def __init__(self):
        self._next: AbstractModelHandler | None = None
        self._place: str | None = None

    def set_next(self, handler: "AbstractModelHandler") -> "AbstractModelHandler":
        self._next = handler
        return handler

    def load_state_dict(
        self,
        project_name: str | None = None,
        experiment_name: str | None = None,
        source: str | None = None,
        save_locally: bool = False,
        **kwargs
    ) -> OrderedDict[str, T.Tensor] | None:
        result = None
        if source is None or source == self._place:
            result = self._load_state_dict(project_name=project_name, experiment_name=experiment_name, save_locally=save_locally)
        if result is None and self._next:
            return self._next._load_state_dict(project_name=project_name, experiment_name=experiment_name, save_locally=save_locally)
        if result is None:
            self.logger_instance.logger.info(f"[{self._place}] No model weights found.")
        return result

    def save_state_dict(
        self,
        state_dict: OrderedDict[str, T.Tensor] | dict[str, T.Tensor] | None = None,
        filename: str = "best_model",
        target: str | None = None
    ) -> bool:
        success = False
        if target is None or target == self._place:
            success = self._save_state_dict(state_dict=state_dict, filename=filename)
        if not success and self._next:
            return self._next._save_state_dict(state_dict=state_dict, filename=filename)
        if not success:
            self.logger_instance.logger.info(f"Model weights were not saved")
        return success

    @abstractmethod
    def _load_state_dict(
        self,
        project_name: str | None = None,
        experiment_name: str | None = None,
        **kwargs
    ) -> OrderedDict[str, T.Tensor] | None:
        pass

    @abstractmethod
    def _save_state_dict(
        self,
        state_dict: OrderedDict[str, T.Tensor] | dict[str, T.Tensor] | None = None,
        filename: str = "best_model"
    ) -> bool:
        pass


class LocalModelHandler(AbstractModelHandler):
    def __init__(self, checkpoints_dir: str = "checkpoints", project_name: str | None = None, experiment_name: str | None = None, **kwargs):
        super().__init__()
        self.checkpoints_dir = checkpoints_dir
        self.project_name = project_name
        self.experiment_name = experiment_name
        self._place = "local"

    def _load_state_dict(
        self,
        project_name: str | None = None,
        experiment_name: str | None = None,
        **kwargs
    ) -> OrderedDict[str, T.Tensor] | None:
        project_name = project_name or self.project_name
        experiment_name = experiment_name or experiment_name
        if self.checkpoints_dir and project_name and experiment_name:
            dir_path = os.path.join(self.checkpoints_dir, project_name, experiment_name)
            checkpoints = glob.glob(os.path.join(dir_path, "*.pth"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                state_dict = T.load(latest_checkpoint)
                self.logger_instance.logger.info(f"[Local] Loaded model weights from {latest_checkpoint}")
                return state_dict
        return None

    def _save_state_dict(self, state_dict: OrderedDict[str, T.Tensor] | dict[str, T.Tensor] | None = None, filename: str = "best_model") -> bool:
        if self.checkpoints_dir and self.project_name and self.experiment_name and state_dict:
            file_path = os.path.join(self.checkpoints_dir, self.project_name, self.experiment_name, filename + ".pth")
            T.save(state_dict, file_path)
            self.logger_instance.logger.info(f"[Local] Saved model weights to {file_path}")
            return True
        return False


class S3ModelHandler(AbstractModelHandler):
    aws_handler = AWSHandler()

    def __init__(self, checkpoints_dir: str = "checkpoints", project_name: str | None = None, experiment_name: str | None = None):
        super().__init__()
        self.checkpoints_dir = checkpoints_dir
        self.project_name = project_name
        self.experiment_name = experiment_name
        self._place = "s3"

    def _load_state_dict(
        self,
        project_name: str | None = None,
        experiment_name: str | None = None,
        save_locally: bool = False,
        **kwargs
    ) -> OrderedDict[str, T.Tensor] | None:
        """Tu powinno coś iść do AWSHandler""" #########################################
        project_name = project_name or self.project_name
        experiment_name = experiment_name or experiment_name
        if project_name and experiment_name:
            dir_path = os.path.join(self.checkpoints_dir, project_name, experiment_name)
            objects_summary = self.aws_handler.list_files_in_s3(dir_path)
            if objects_summary:
                latest_object = max(objects_summary, key=lambda x: x.last_modified)
                self.logger_instance.logger.info(f"[S3] Loaded model weights from {latest_object.key}")
                if save_locally:
                    local_path = os.path.join(dir_path, latest_object.key.split("/")[-1])
                    self.aws_handler.download_file_from_s3(s3_path=latest_object.key, local_path=local_path)
                    self.logger_instance.logger.info(f"[S3] Saved model weights to {local_path}")
                    return T.load(local_path)
                else:
                    file_stream = BytesIO(latest_object.Object().get()["Body"].read())
                    return T.load(file_stream)
        return None

    def _save_state_dict(self, state_dict: OrderedDict[str, T.Tensor] | dict[str, T.Tensor] | None = None, filename: str = "best_model") -> bool:
        """Tu powinno coś iść do AWSHandler""" #########################################
        if self.project_name and self.experiment_name:
            file_path = os.path.join(self.checkpoints_dir, self.project_name, self.experiment_name, filename + ".pth")
            if state_dict:
                file_stream = BytesIO()
                T.save(state_dict, file_stream)
                file_stream.seek(0)
                if self.aws_handler.s3_bucket:
                    self.aws_handler.s3_bucket.upload_fileobj(file_stream, file_path)
            else:
                self.aws_handler.upload_file_to_s3(local_path=file_path, s3_path=file_path)
            self.logger_instance.logger.info(f"[S3] Saved model weights to {file_path}")
            return True
        return False


class ModelHandler(metaclass=SingletonMeta):
    from utils.logger import SingletonLogger
    logger_instance = SingletonLogger()

    def __init__(self):
        self._initialized = False

    def initialize(
        self,
        checkpoints_dir: str = "checkpoints",
        project_name: str | None = None,
        experiment_name: str | None = None,
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.checkpoints_dir = checkpoints_dir
        self._initialized = True
        
        local_model_handler = LocalModelHandler(checkpoints_dir=checkpoints_dir, project_name=project_name, experiment_name=experiment_name)
        s3_model_handler = S3ModelHandler(checkpoints_dir=checkpoints_dir, project_name=project_name, experiment_name=experiment_name)
        
        local_model_handler.set_next(s3_model_handler)
        self._head = local_model_handler

    def load_weights(
        self,
        model: nn.Module,
        source: Literal["local", "hugging_face", "s3"] | None = None,
        model_part: Literal["all", "backbone", "detection_head"] = "all",
        project_name: str | None = None,
        experiment_name: str | None = None,
        save_locally: bool = False,
        **kwargs: dict[str, Any]
    ):
        state_dict = self._head.load_state_dict(project_name=project_name, experiment_name=experiment_name, source=source, save_locally=save_locally)
        
        if state_dict:
            new_state_dict = {k: v for k, v in state_dict.items() if ( model_part in k or model_part == "all")}
            model.load_state_dict(new_state_dict, strict=False)
            self._freeze_weights(model=model, state_dict_keys=list(new_state_dict.keys()))

    def _freeze_weights(self, model: nn.Module, state_dict_keys: list[str]):
        for name, param in model.named_parameters():
            if name in state_dict_keys:
                param.requires_grad = False
        self.logger_instance.logger.info("Loaded weights were frozen")

    def unfreeze_weights(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True
        self.logger_instance.logger.info("Loaded weights were unfrozen")

    def save_weights(self, model: nn.Module | None = None, filename: str = "best_model", target: str | None = None):
        self._head.save_state_dict(state_dict=model.state_dict() if model else None, filename=filename, target=target)
