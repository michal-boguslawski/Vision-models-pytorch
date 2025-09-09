
from PIL import Image
import torch as T
from datasets.dataset_utils import DatasetHandler
from models.model_factory import BuildModel
from models.utils import ModelHandler
from utils.config_parser import ConfigParser
from utils.helpers import filter_kwargs


class InferenceRunner:
    def __init__(self, config_path: str, device: str | None = None):
        self.config = ConfigParser(config_path)
        self.device = device or self.config["misc"].get("device")
        self.model_handler = ModelHandler()
        self.data_handler = DatasetHandler(config=self.config["dataset"])

        self._get_names()
        self._build_model()
        self._load_weights()

    def _get_names(self):
        self.project_name = self.config["project_name"]
        self.experiment_name = self.config["experiment_name"]
        self.checkpoint_dir = self.config["checkpoint_dir"]

    def _build_model(self):
        self.model = BuildModel(**filter_kwargs(BuildModel, self.config["model"])).to(self.device)
        self.model.eval()

    def _load_weights(self):
        self.model_handler.load_weights(
            model=self.model,
            source="local",
            version_name=self.experiment_name,
            model_part="all",
            checkpoint_dir=self.checkpoint_dir,
            project_name=self.project_name,
            device=self.device,
            s3_bucket_name=self.config["s3_bucket_name"]
        )

    def _preprocess_image(self, image: Image.Image | T.Tensor) -> T.Tensor:
        return self.data_handler.preprocess_single_image(image=image, size=self.config["dataset"].get("input_size"))

    def predict(self, image: Image.Image | T.Tensor):
        predictions: dict[str, float] = {}
        image = self._preprocess_image(image=image)
        image = image.unsqueeze(0).to(self.device)
        with T.no_grad():
            outputs = self.model(image)
            outputs = outputs.softmax(dim=1)
            outputs = outputs.cpu().numpy().tolist()[0]
        for key, label in self.data_handler.id_to_label.items():
            predictions[label] = outputs[key]
        return predictions
