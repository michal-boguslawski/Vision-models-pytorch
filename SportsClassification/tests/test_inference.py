from inference.infer import InferenceRunner
from PIL import Image


if __name__ == "__main__":
    path = "logs/sports_cv_project_classification/vgg11-paper-pretrained-w-augmenations-cosine_annealing_warm_restarts_with_decay/config.yaml"
    runner = InferenceRunner(config_path=path, device="cpu")
    image = Image.open("data/processed/val/tennis/1.jpg").convert("RGB")
    outputs = runner.predict(image=image)
    print(outputs)
