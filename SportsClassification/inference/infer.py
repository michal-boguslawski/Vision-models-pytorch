class InferenceRunner:
    def __init__(self, checkpoint_path, device="cuda"):
        self.model = load_model(checkpoint_path, device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        x = self.transform(image).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(x)
        return decode_predictions(outputs)