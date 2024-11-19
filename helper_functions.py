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


def show_step_function(epoch, latent_dim, step, show_step, generator, disc_loss_list, gen_loss_list, device='cpu'):
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

  plt.plot(
    range(len(disc_loss_list)),
    torch.Tensor(disc_loss_list),
    label="Critic Loss"
  )
  max_max = np.max([np.max(gen_loss_list), np.max(disc_loss_list)])
  min_min = np.min([np.min(gen_loss_list), np.min(disc_loss_list)])
  plt.ylim((min_min - np.abs(min_min) * 0.05),
           (max_max + np.abs(max_max)*0.05))
  plt.legend()
  plt.show()

def training_step_discriminator(true_image, generator, discriminator, loss_fn, latent_dim, device='cpu'):
  noise = torch.randn(true_image.shape[0], latent_dim, device=device)
  fake_image = generator(noise)
  fake_output = discriminator(fake_image.detach())
  true_output = discriminator(true_image)
  ones = torch.ones_like(true_output)
  zeros = torch.zeros_like(fake_output)
  arguments ={'fake_image': fake_image,
              'fake_output': fake_output,
              'true_image': true_image,
              'true_output': true_output,
              'discriminator': discriminator,
              'if_disc': True,
              'ones': ones,
              'zeros': zeros,
              'device': device}
  loss = loss_fn(**arguments)
  return loss

def training_step_generator(batch_size, generator, discriminator, loss_fn, latent_dim, device='cpu'):
  noise = torch.randn(batch_size, latent_dim, device=device)
  fake_image = generator(noise)
  fake_output = discriminator(fake_image)
  ones = torch.ones_like(fake_output)
  arguments ={'fake_image': fake_image,
              'fake_output': fake_output,
              'discriminator': discriminator,
              'if_disc': False,
              'ones': ones,
              'device': device}
  loss = loss_fn(**arguments)
  return loss

def get_gp(real, fake, crit, alpha, gamma=10):
  mix_images = real * alpha + fake * (1-alpha) # 128 x 3 x 128 x 128
  mix_scores = crit(mix_images) # 128 x 1

  gradient = torch.autograd.grad(
      inputs = mix_images,
      outputs = mix_scores,
      grad_outputs=torch.ones_like(mix_scores),
      retain_graph=True,
      create_graph=True,
  )[0] # 128 x 3 x 128 x 128

  gradient = gradient.reshape(len(gradient), -1)   # 128 x 49152
  gradient_norm = gradient.norm(2, dim=1)
  gp = gamma * ((gradient_norm-1)**2).mean()

  return gp

def WGANLoss(fake_image, fake_output, true_image=None, true_output=None, discriminator=None, if_disc=True, device='cpu', **kwargs):
  if if_disc:
    alpha=torch.rand(len(true_image),1,1,1,device=device, requires_grad=True) # 128 x 1 x 1 x 1
    gp = get_gp(true_image, fake_image.detach(), discriminator, alpha)
    return fake_output.mean() - true_output.mean() + gp
  else:
    return -fake_output.mean()