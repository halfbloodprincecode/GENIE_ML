import torch.nn.functional as F

def vqvae_loss(x, quantized_x, lambda_e_latent=1):
    e_latent_loss = F.mse_loss(quantized_x.detach(), x)
    q_latent_loss = F.mse_loss(quantized_x, x.detach())
    loss = q_latent_loss + lambda_e_latent * e_latent_loss
    return loss