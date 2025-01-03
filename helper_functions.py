import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


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
