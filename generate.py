import torch 
import torchvision
import os
import argparse

from model import Generator
from utils import load_model
from model import Discriminator
from utils import load_model, load_discriminator, discriminator_rejection_sampling,estimate_r_and_cK,optimal_budgeted_rejection_sampling


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()




    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784
    device='cuda' if torch.cuda.is_available() else 'cpu'
    device='mps' if torch.backends.mps.is_available() else device
    model = Generator(g_output_dim = mnist_dim).to(device)
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
#DRS
    D = Discriminator(mnist_dim).to(device)
    D = load_discriminator(D, 'checkpoints')
    D = torch.nn.DataParallel(D).to(device)
    D.eval()

    print('Model loaded.')

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)
    n_samples = 10000  # nombre d'images à générer

""""
    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, 100).to(device)
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1

"""

#DRS
# Génération avec DRS
samples = discriminator_rejection_sampling(model, D, n_samples, batch_size=args.batch_size)

samples = samples.view(-1, 1, 28, 28)
# Sauvegarde
for i in range(samples.shape[0]):
    torchvision.utils.save_image(samples[i:i+1], os.path.join('samples', f'{i}.png'))
"""
#OBRS
# Définir ton budget K (ex : 5 tirages pour 1 accepté)
K = 2

# Étape 1 : estimer c_K
c_K= estimate_r_and_cK(model, D, K)

# Étape 2 : générer des échantillons filtrés OBRS
samples = optimal_budgeted_rejection_sampling(model, D, n_samples=n_samples, c_K=c_K)
samples = (samples + 1) / 2
samples = samples.view(-1, 1, 28, 28)
# Sauvegarde
for i in range(samples.shape[0]):
    torchvision.utils.save_image(samples[i:i+1], os.path.join('samples', f'{i}.png'))

"""