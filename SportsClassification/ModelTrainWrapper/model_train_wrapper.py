import torch as T
from torch.optim import Adam
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


class ModelTrainWrapper:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module = nn.CrossEntropyLoss,
        optimizer: T.optim.Optimizer = Adam,
        learning_rate: float = 0.0001,
        device: T.device = T.device("cuda" if T.cuda.is_available() else "cpu"),
    ):
        self.model = model(n_classes=100).to(device)
        self.loss_fn = loss_fn()
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.class_to_idx = None
        self.class_names = None
        
    def _prepare_dataloader(
        self,
        data_path: str,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),     # Resize to a fixed size
            transforms.ToTensor(),             # Convert PIL image to Tensor
        ])
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        if self.class_to_idx:
            dataset.class_to_idx = self.class_to_idx
        else:
            self.class_to_idx = dataset.class_to_idx
            self.class_names = dataset.classes
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            pin_memory=T.cuda.is_available()
        )
        return dataloader
        
    def predict(self, image: T.Tensor) -> tuple:
        self.model.eval()
        with T.no_grad():
            pred_probs = self.model(image).to(self.device)
        pred_indicies = pred_probs.argmax(dim=-1)
        pred_classes = [self.class_names[i] for i in pred_indicies]
        return pred_classes, pred_indicies, pred_probs
        
    def evaluate_accuracy(
        self,
        data_path: str = None,
        batch_size: int = 32,
        dataloader: DataLoader = None,
        n: int = 10
    ) -> tuple:
        all_preds = []
        all_labels = []
        if not dataloader:
            dataloader = self._prepare_dataloader(data_path=data_path, batch_size=batch_size)
        
        i = 0
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, pred_indicies, _ = self.predict(images)
            all_preds.extend(pred_indicies.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if i == 0:
                sample = (images, pred_indicies, labels)
            i += 1
            if i >= n:
                break
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy, sample, all_labels, all_preds
    
    @staticmethod
    def plot_images(
        images: T.Tensor,
        preds_classes: list[str],
        labels_classes: list[str],
        n: int = 16
    ) -> None:
        fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
        axs = axs.flatten()
        for i in range(n):
            axs[i].imshow(images[i].permute(1, 2, 0).cpu().numpy())
            axs[i].axis("off")
            axs[i].set_title(f"{preds_classes[i]} vs {labels_classes[i]}", fontsize=6)
        plt.tight_layout()
        plt.show()
            
    @staticmethod
    def plot_loss(
        epoch: int,
        losses: list,
        step: int = 100
    ):
        plt.figure(figsize=(10, 5))

        # Plot raw loss
        plt.plot(losses, label="Loss")

        # Compute and plot 10-step moving average
        if len(losses) >= step:
            moving_avg = np.convolve(losses, np.ones(step)/step, mode='valid')
            plt.plot(range(step-1, len(losses)), moving_avg, label=f"{step}-step Mean", color="orange")

        plt.title(f"Loss for epoch {epoch}")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def index_to_class(self, labels):
        return [self.class_names[i] for i in labels]
    
    def confusion_matrix(
        self,
        labels: T.Tensor,
        preds: T.Tensor,
        set_type: str | None = None
    ):
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true=labels,
            y_pred=preds,
            labels=range(100),
            display_labels=self.class_names,
            normalize="true"
        )
        disp.ax_.set_title(f"Confusion Matrix {set_type}")
        plt.show()
    
    def on_epoch_end(
        self,
        epoch: int,
        losses: list,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader | None = None
    ):
        '''
        Actions taken at the end of each epoch.
        Calculate accuracy for both training and test datasets, show sample of images with predictions, confusion matrix and plot loss function
        '''
        
        print(50 * "=")
        # Calculate accuracy and get sample image data and predictions for train set
        train_accuracy, train_sample, all_train_labels, all_train_preds = self.evaluate_accuracy(dataloader=train_dataloader)
        train_images, train_preds, train_labels = train_sample
        train_preds_classes = self.index_to_class(train_preds)
        train_labels_classes = self.index_to_class(train_labels)
        
        # Calculate accuracy and get sample image data and predictions for test set
        if test_dataloader:
            test_accuracy, test_sample, all_test_labels, all_test_preds = self.evaluate_accuracy(dataloader=test_dataloader)
            test_images, test_preds, test_labels = test_sample
            test_preds_classes = self.index_to_class(test_preds)
            test_labels_classes = self.index_to_class(test_labels)
            
        # Print results together with sample images with predicted classification and confusion matricies
        print(f"Epoch: {epoch}, train set accuracy {train_accuracy:.4f} test set accuracy {test_accuracy:.4f}")
        print("Train sample")
        self.plot_images(train_images, train_preds_classes, train_labels_classes)
        # self.confusion_matrix(all_train_labels, all_train_preds, "train")
        if test_dataloader:
            print("Test sample")
            self.plot_images(test_images, test_preds_classes, test_labels_classes)
            # self.confusion_matrix(all_test_labels, all_test_preds, "test")
            
        print("Loss plot")
        self.plot_loss(epoch=epoch, losses=losses)
    
    def train_step(
        self,
        images: T.Tensor,
        labels: T.Tensor
    ):
        '''
        Performs one training step:
        - Moves inputs to the correct device
        - Performs forward and backward passes
        - Updates model weights
        - Returns scalar loss value

        Parameters:
            images (torch.Tensor): A batch of input images (e.g., shape [B, C, H, W])
            labels (torch.Tensor): Corresponding labels for the batch (e.g., shape [B])

        Returns:
            float: The loss value for this training step
        '''
        
        # Move input data and labels to the specified device (e.g., GPU or CPU)
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Clear gradients from previous step
        self.optimizer.zero_grad()
        
        # Forward pass: compute model predictions
        preds = self.model(images)
        
        # Compute the loss between predictions and true labels
        loss = self.loss_fn(preds, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update model weights
        self.optimizer.step()
        
        # Return loss value as a Python float (for logging or plotting)
        return loss.item()
        
    def train(
        self,
        data_path: str,
        batch_size: int = 32,
        epochs: int = 10,
        test_data_path: str = None
    ):
        test_dataloader = None
        loss_list = []
        
        dataloader = self._prepare_dataloader(data_path, batch_size=batch_size)
        if test_data_path:
            test_dataloader = self._prepare_dataloader(test_data_path, batch_size=batch_size, shuffle=True)
            
        for epoch in range(epochs):
            self.model.train()
            for images, labels in dataloader:
                loss = self.train_step(images=images, labels=labels)
                loss_list.append(loss)
            self.on_epoch_end(epoch=epoch, losses=loss_list, train_dataloader=dataloader, test_dataloader=test_dataloader)
