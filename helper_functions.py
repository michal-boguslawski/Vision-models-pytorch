import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.functional.detection import intersection_over_union
from matplotlib.patches import Rectangle
import torch.nn.functional as F
import transformers
import torch.nn as nn
import timm


def show_image(img, num=4, rescale=None):
    img = img.cpu()
    if rescale:
        img += rescale[0]
        img /= rescale[1]
    img = torchvision.utils.make_grid(img, num)
    plt.imshow(img.permute(1, 2, 0).clip(0, 1))
    plt.show()


def calc_mov_average(arr, window_len=300):
    # Handle edge case where window_len is greater than the length of the array
    if window_len <= 1:
        return arr  # No moving average needed if window length is 1 or less

    # Array to store the moving averages
    result = np.zeros_like(arr)

    # Loop over the array and compute the moving average
    for i in range(len(arr)):
        # Define the window start and end indices
        start = max(0, i - window_len + 1)
        end = i + 1

        # Compute the mean for the window, handles fewer elements than window_len
        result[i] = np.mean(arr[start:end])

    return result


def show_step_function(epoch, latent_dim, step, show_step, generator, disc_loss_list, gen_loss_list, if_quantile=True,
                       default_max=None, default_min=None,
                       device='cpu'):
    print(f'Epoch: {epoch}, Step: {step}, Discriminator loss last {show_step}: {np.mean(disc_loss_list[-show_step:])}, \
    Generator loss last {show_step}: {np.mean(gen_loss_list[-show_step:])}')
    noise = torch.randn(16, latent_dim, device=device)
    fake_image = generator(noise)
    show_image(fake_image, rescale=(1, 2))
    plt.plot(
        range(len(gen_loss_list)),
        torch.Tensor(gen_loss_list),
        label="Generator Loss"
    )

    x = calc_mov_average(gen_loss_list)
    plt.plot(
        range(len(x)),
        x,
        label="Generator Loss Moving Average"
    )

    plt.plot(
        range(len(disc_loss_list)),
        torch.Tensor(disc_loss_list),
        label="Critic Loss"
    )

    x = calc_mov_average(disc_loss_list)
    plt.plot(
        range(len(x)),
        x,
        label="Critic Loss Moving Average"
    )

    if default_max:
        max_max = default_max
    elif if_quantile:
        quantile1 = np.quantile(gen_loss_list, 0.95)
        quantile2 = np.quantile(disc_loss_list, 0.95)
        max_max = np.max([quantile1, quantile2])
    else:
        quantile1 = np.max(gen_loss_list)
        quantile2 = np.max(disc_loss_list)
        max_max = np.max([quantile1, quantile2])

    if default_min:
        min_min = default_min
    elif if_quantile:
        quantile1 = np.quantile(gen_loss_list, 0.05)
        quantile2 = np.quantile(disc_loss_list, 0.05)
        min_min = np.min([quantile1, quantile2])
    else:
        quantile1 = np.min(gen_loss_list)
        quantile2 = np.min(disc_loss_list)
        min_min = np.min([quantile1, quantile2])
    plt.ylim((min_min - np.abs(min_min) * 0.05),
             (max_max + np.abs(max_max) * 0.05))
    plt.legend()
    plt.show()


def training_step_discriminator(true_image, generator, discriminator, loss_fn, latent_dim, device='cpu'):
    noise = torch.randn(true_image.shape[0], latent_dim, device=device)
    fake_image = generator(noise)
    fake_output = discriminator(fake_image.detach())
    true_output = discriminator(true_image)
    ones = torch.ones_like(true_output)
    zeros = torch.zeros_like(fake_output)
    arguments = {'fake_image': fake_image,
                 'fake_output': fake_output,
                 'true_image': true_image,
                 'true_output': true_output,
                 'discriminator': discriminator,
                 'if_disc': True,
                 'ones': ones,
                 'zeros': zeros,
                 'device': device}
    loss, args = loss_fn(**arguments)
    return loss, args


def training_step_generator(batch_size, generator, discriminator, loss_fn, latent_dim, device='cpu'):
    noise = torch.randn(batch_size, latent_dim, device=device)
    fake_image = generator(noise)
    fake_output = discriminator(fake_image)
    ones = torch.ones_like(fake_output)
    arguments = {'fake_image': fake_image,
                 'fake_output': fake_output,
                 'discriminator': discriminator,
                 'if_disc': False,
                 'ones': ones,
                 'device': device}
    loss = loss_fn(**arguments)
    return loss


def get_gp(real, fake, crit, alpha, gamma=10):
    mix_images = real * alpha + fake * (1 - alpha)  # 128 x 3 x 128 x 128
    mix_scores = crit(mix_images)  # 128 x 1

    gradient = torch.autograd.grad(
        inputs=mix_images,
        outputs=mix_scores,
        grad_outputs=torch.ones_like(mix_scores),
        retain_graph=True,
        create_graph=True,
    )[0]  # 128 x 3 x 128 x 128

    gradient = gradient.reshape(len(gradient), -1)  # 128 x 49152
    gradient_norm = gradient.norm(2, dim=1)
    gp = gamma * ((gradient_norm - 1) ** 2).mean()

    return gp


def WGANLoss(fake_image, fake_output, true_image=None, true_output=None, discriminator=None, if_disc=True, device='cpu',
             **kwargs):
    if if_disc:
        alpha = torch.rand(len(true_image), 1, 1, 1, device=device, requires_grad=True)  # 128 x 1 x 1 x 1
        gp = get_gp(true_image, fake_image.detach(), discriminator, alpha)
        return fake_output.mean() - true_output.mean() + gp, (
            fake_output.mean().item(), true_output.mean().item(), gp.item())
    else:
        return -fake_output.mean()


class ResNetModel(nn.Module):
    def __init__(self, resnet_config, output_dim, anchor_size, device='cpu'):
        super(ResNetModel, self).__init__()
        self.baseline = transformers.ResNetModel(resnet_config).to(device)
        self.output_dim = output_dim
        self.anchor_size = anchor_size
        # self.lin1 = nn.Linear(resnet_config.hidden_sizes[-1],
        #                            1024,
        #                            device=device)
        self.lin1 = nn.Conv2d(resnet_config.hidden_sizes[-1],
                              out_channels=1024,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              device=device)
        self.output_rpn = nn.Linear(1024,
                                    anchor_size,
                                    device=device)

    def forward(self, x):
        last_hidden_state, _ = self.baseline(x, return_dict=False)
        # last_hidden_state = last_hidden_state.permute(0, 2, 3, 1)
        # last_hidden_state = output['last_hidden_state']
        x = self.lin1(last_hidden_state)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1)
        rpn_output = self.output_rpn(x)
        return rpn_output, last_hidden_state


class InceptionModel(nn.Module):
    def __init__(self, output_dim, anchor_size, if_pretrained, device='cpu'):
        super(InceptionModel, self).__init__()
        model = timm.create_model('inception_v3', pretrained=if_pretrained).to(device)
        baseline = nn.Sequential(*list(model.children())[:-3])
        self.baseline = baseline
        self.output_dim = output_dim
        self.anchor_size = anchor_size
        self.lin1 = nn.Linear(2048,
                              1024,
                              device=device)
        self.output_rpn = nn.Linear(1024,
                                    anchor_size,
                                    device=device)

    def forward(self, x):
        last_hidden_state = self.baseline(x)
        last_hidden_state = last_hidden_state.permute(0, 2, 3, 1)
        # last_hidden_state = output['last_hidden_state']
        x = self.lin1(last_hidden_state)
        x = F.gelu(x)
        rpn_output = self.output_rpn(x)
        return rpn_output, last_hidden_state


def generate_anchors(anchor_sizes, anchor_ratios, new_size):
    anchors = []
    for size in anchor_sizes:
        for ratio in anchor_ratios:
            anchors.append((min(size, new_size[0]),
                            min(size, new_size[1])))
            size1 = size * ratio
            size2 = size
            # if size1 < new_size[0] and size2 < new_size[1]:
            anchors.append((min(size1, new_size[0]),
                            min(size2, new_size[1])))
            size1 = size
            size2 = size * ratio
            # if size1 < new_size[0] and size2 < new_size[1]:
            anchors.append((min(size1, new_size[0]),
                            min(size2, new_size[1])))
    anchors.append(new_size)
    return torch.tensor(list(set(anchors)))


def generate_possible_anchors(shape, anchors, patch_size, new_size):
    '''
    test = generate_possible_anchors((4, 16, 16), anchors, patch_size, new_size)
    '''
    anchor_length = len(anchors)
    x_pom = torch.arange(shape[0])
    y_pom = torch.arange(shape[1])
    anchor_pom = torch.arange(anchor_length)

    tensor = torch.zeros(shape + (anchor_length, 4))
    tensor[x_pom, :, :, 0] = (x_pom * patch_size + patch_size / 2).view(1, shape[0], 1, 1)
    tensor[:, y_pom, :, 1] = (y_pom * patch_size + patch_size / 2).view(1, 1, shape[1], 1)
    tensor[:, :, anchor_pom, 0] -= anchors[:, 0] / 2
    tensor[:, :, anchor_pom, 1] -= anchors[:, 1] / 2
    tensor[:, :, anchor_pom, 2] = anchors[:, 0].to(torch.float)
    tensor[:, :, anchor_pom, 3] = anchors[:, 1].to(torch.float)
    tensor[:, :, :, 2] += tensor[:, :, :, 0].clip(-torch.inf, 0)
    tensor[:, :, :, 3] += tensor[:, :, :, 1].clip(-torch.inf, 0)
    tensor[:, :, :, 2] -= (tensor[:, :, :, 0] + tensor[:, :, :, 2] - float(new_size[0])).clip(0, torch.inf)
    tensor[:, :, :, 3] -= (tensor[:, :, :, 1] + tensor[:, :, :, 3] - float(new_size[1])).clip(0, torch.inf)
    tensor[..., :2] = tensor[..., :2].clip(0, 512)
    return tensor


def convert_bbox(tensor, type_from='xywh', type_to='xyxy'):
    tensor_copy = tensor.clone()
    if type_from == 'xywh' and type_to == 'xyxy':
        tensor_copy[..., 2:] += tensor_copy[..., :2]
    elif type_from == 'xyxy' and type_to == 'xywh':
        tensor_copy[..., 2:] -= tensor_copy[..., :2]
    return tensor_copy


def calculate_target_anchor_iou(anchors, bboxs, patch_size=0, lower_bound=0.3, upper_bound=0.9, approx=0.9):
    anchor_shape = anchors.shape
    ones_indices_list = []
    zeros_indices_list = []
    for x in bboxs:
        iou = intersection_over_union(anchors.view(-1, 4), x.unsqueeze(0), aggregate=False).view(anchor_shape[:-1])
        # area_part = torch.min(x[2] / patch_size, torch.tensor(1)) * torch.min(x[3] / patch_size, torch.tensor(1))
        quant = torch.quantile(iou, 1 - 2 / iou.numel())
        # final_upper_bound = torch.max(quant, torch.tensor(upper_bound))
        final_upper_bound = min(quant, upper_bound)  # upper_bound
        # final_upper_bound = torch.min(torch.tensor(upper_bound), area_part * upper_bound * approx)
        # index = (iou > final_upper_bound) | (iou < lower_bound) #| (lower_bound > final_upper_bound
        ones_indices_list.append((iou > final_upper_bound))
        zeros_indices_list.append((iou < min(lower_bound, final_upper_bound)))
    ones_tensor = torch.stack(ones_indices_list, dim=-1)
    ones_tensor = torch.max(ones_tensor, dim=-1)
    zeros_tensor = torch.stack(zeros_indices_list, dim=-1)
    zeros_tensor = torch.max(zeros_tensor, dim=-1)
    return ones_tensor.values, zeros_tensor.values


def generate_target_iou_per_anchor(anchors, bboxs, img_ids, batch_size):
    batch_ones = []
    batch_zeros = []
    for i in range(batch_size):
        bbox = bboxs[img_ids == i]
        ones, zeros = calculate_target_anchor_iou(anchors, bbox)
        batch_ones.append(ones)
        batch_zeros.append(zeros)
    return torch.stack(batch_ones, dim=0), torch.stack(batch_zeros, dim=0)


def print_log(data, model, losses, epoch, step, possible_anchors, device):
    image_data = data['images'].to(device)
    temp_bs = len(image_data)
    rpn_output, _ = model(image_data)
    pred_iou = F.sigmoid(rpn_output.detach()).cpu()
    bboxs = data['bboxs']
    converted_bboxs = convert_bbox(bboxs)
    img_ids = data['img_ids']

    ones, zeros = generate_target_iou_per_anchor(convert_bbox(possible_anchors), converted_bboxs, img_ids, temp_bs)
    top_5_value = torch.quantile(pred_iou[0], 1 - 5 / pred_iou[0].numel())
    chosen_anchors = (pred_iou[0] > min(0.99, top_5_value))
    # anchors_to_plot = possible_anchors[chosen_anchors]
    anchors_to_plot = nms(pred_iou[0][chosen_anchors], possible_anchors[chosen_anchors], treshold=0.5)

    print(f'Epoch: {epoch}, step: {step}, \
    num positives: {ones[0].sum()}, \
    max pred iou: {pred_iou[0].max().item()}, \
    mean pred iou: {pred_iou[0].mean().item()} \
    count highs: {len(anchors_to_plot)}')
    fig, axs = plt.subplots(1, 4, figsize=(12, 5))
    axs[0].plot(losses)
    axs[0].plot(calc_mov_average(losses))
    axs[1].hist(pred_iou[0].view(-1, 1).cpu(), bins=20)

    axs[2].imshow(image_data[0].cpu().permute(1, 2, 0).clip(0, 1))
    max_i = 0
    for i, row in enumerate(anchors_to_plot):
        rect = Rectangle(row[:2], row[2], row[3], edgecolor=(0.73 * i % 1, 0.53 * i % 1, 0.37 * i % 1),
                         facecolor='none', linewidth=1)
        axs[2].add_patch(rect)
        max_i += 1
        if max_i > 100:
            break
    axs[3].imshow(image_data[0].cpu().permute(1, 2, 0).clip(0, 1))
    max_i = 0
    bboxs_to_plot = possible_anchors[ones[0]]
    # bboxs_to_plot = bboxs[img_ids==0]
    for i, row in enumerate(bboxs_to_plot):
        rect = Rectangle(row[:2], row[2], row[3], edgecolor=(0.73 * i % 1, 0.53 * i % 1, 0.37 * i % 1),
                         facecolor='none', linewidth=1)
        axs[3].add_patch(rect)
        max_i += 1
        if max_i > 100:
            break
    plt.savefig(f'training_images/Image_step_{step}.jpg')


def calc_mov_average(arr, window_len=300):
    # Handle edge case where window_len is greater than the length of the array
    if window_len <= 1:
        return arr  # No moving average needed if window length is 1 or less

    # Array to store the moving averages
    result = np.zeros_like(arr)

    # Loop over the array and compute the moving average
    for i in range(len(arr)):
        # Define the window start and end indices
        start = max(0, i - window_len + 1)
        end = i + 1

        # Compute the mean for the window, handles fewer elements than window_len
        result[i] = np.mean(arr[start:end])

    return result


def nms(preds, bboxs, treshold=0.5):
    # preds = pred_iou[0][chosen_anchors]
    # bboxs = possible_anchors[chosen_anchors]
    sorted_indices = preds.sort(descending=True).indices
    sorted_bboxs = convert_bbox(bboxs[sorted_indices])
    bboxs_to_print = []
    i = 0
    while len(sorted_bboxs) > 0 and i < 100:
        i += 1
        bbox = sorted_bboxs[0]
        bboxs_to_print.append(bbox)
        sorted_bboxs = sorted_bboxs[1:]
        if len(sorted_bboxs) > 1:
            ious = intersection_over_union(bbox.unsqueeze(0), sorted_bboxs, aggregate=False)
            kept_bboxs = (ious < treshold)[0]
            sorted_bboxs = sorted_bboxs[kept_bboxs]

    return convert_bbox(torch.stack(bboxs_to_print, dim=0), type_from='xyxy', type_to='xywh')
