# importing libraries
from image_dataset import create_dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import transformers
from torchvision import transforms
import numpy as np

# my own functions
from helper_functions import *

# getting the data to dataloader

final_shape = 16
patch_size = 32
new_size = (512, 512)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img.repeat(3//img.shape[0], 1, 1)),
    transforms.Resize(new_size),
                       ]
                              )

print('Starting create training dataloader')

dataloader = create_dataloader(
    data_type = 'train',
    selected_categories = ['person', ],
    selected_output = ['bbox', ],
    new_size = new_size,
    patch_size = patch_size,
    transform = transform,
    batch_size = 32,
    #dataset_size = 1000
    )

print('Training dataloader created')

val_dataloader = create_dataloader(
    data_type = 'val',
    selected_categories = ['person', ],
    selected_output = ['bbox', ],
    new_size = new_size,
    patch_size = patch_size,
    transform = transform,
    batch_size = 32,
    #dataset_size = 1000
    )

print('Validation dataloader created')

val_iter = iter(val_dataloader)

# defining anchors
anchor_sizes = [16, 32, 64, 128, 256]
anchor_ratios = [1, 2, 3/2, 5, 5/2, 4/3, 6/5, 4, 5/3]
anchors = generate_anchors(anchor_sizes, anchor_ratios, new_size=new_size)
print(anchors.shape, anchors[:5])

possible_anchors = generate_possible_anchors((final_shape, final_shape), anchors, patch_size, new_size)
print(possible_anchors.shape)

# defining model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_config = transformers.ResNetConfig(num_channels=3,
                                          #embedding_size=64,
                                          #hidden_sizes=(128, 256, 512, 1024, 2048),
                                          hidden_act='gelu',
                                          #depths = [3, 3, 3, 3, 3, 3],
                                          device = device)
model = ResNetModel(resnet_config,
                           output_dim=1,
                           anchor_size=len(anchors),
                           device = device)
'''
model = InceptionModel(output_dim=1,
                       anchor_size=len(anchors),
                       if_pretrained=False,
                       device = device)
'''

# creating functions to train
loss_fn_rpn = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = optim.AdamW(model.parameters(), lr=2e-4)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                num_warmup_steps=3000,
                                                                num_training_steps=40000)

# loading weights
model.load_state_dict(torch.load("model_weights.pth"))
optimizer.load_state_dict(torch.load("optimizer_weights.pth"))
scheduler.load_state_dict(torch.load("scheduler_weights.pth"))

# training
print('Training has started')
epochs = 10000
losses = []
step = 0
step_show = 150
converted_anchors = convert_bbox(possible_anchors)
torch.cuda.empty_cache()
for epoch in range(epochs):
    for data in dataloader:
        optimizer.zero_grad()
        loss = 0
        image_data = data['images'].to(device)
        temp_bs = len(image_data)
        bboxs = data['bboxs']
        converted_bboxs = convert_bbox(bboxs)
        img_ids = data['img_ids']

        ones, zeros = generate_target_iou_per_anchor(converted_anchors, converted_bboxs, img_ids, temp_bs)

        rpn_output, _ = model(image_data)

        ones_output = rpn_output[ones]
        zeros_output = rpn_output[zeros]
        zeros_select = torch.multinomial(zeros_output - zeros_output.min(), len(ones_output) * 2 + len(anchors) * 2)
        zeros_output = zeros_output[zeros_select]

        if len(ones_output) > 0:
            loss += loss_fn_rpn(ones_output, torch.ones_like(ones_output))
        if len(zeros_output) > 0:
            loss += loss_fn_rpn(zeros_output, torch.zeros_like(zeros_output))
        loss /= (ones_output.numel() + zeros_output.numel())

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if step % step_show == 0:
            try:
                val_data = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dataloader)
                val_data = next(val_iter)
            with torch.no_grad():
                print_log(val_data, model, losses, epoch, step, possible_anchors, device)
            torch.save(model.state_dict(), "model_weights.pth")
            torch.save(optimizer.state_dict(), "optimizer_weights.pth")
            torch.save(scheduler.state_dict(), "scheduler_weights.pth")
            print('Weights saved')
        step += 1
