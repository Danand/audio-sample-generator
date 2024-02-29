import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import streamlit as st

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)

# Hyperparameters etc.
device = "mps"
#lr = 3e-4
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 1
num_epochs = 3

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

model_output_dir="./temp/models"
model_output_path=f"{model_output_dir}/mnist.pth"

step = 0
dataset_size = 30000

placeholder_status = st.empty()
placeholder_image_fake = st.empty()
placeholder_image_real = st.empty()

if st.button(
    label="Train",
    use_container_width=True,
    type="primary",
):
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            if batch_idx > dataset_size:
                break

            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # where the second option of maximizing doesn't suffer from
            # saturating gradients
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                placeholder_status.text(f"Epoch [{epoch + 1}/{num_epochs}] Batch {batch_idx}/{dataset_size} Loss D: {lossD:.4f}, loss G: {lossG:.4f}")

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    placeholder_image_fake.image(
                        image=img_grid_fake.permute(1, 2, 0).cpu().numpy(),
                        use_column_width=True,
                    )

                    placeholder_image_real.image(
                        image=img_grid_real.permute(1, 2, 0).cpu().numpy(),
                        use_column_width=True,
                    )

            step += 1

    torch.save(
        obj=gen.state_dict(),
        f=model_output_path,
    )

    st.text("Training finished.")

if st.button(
    label="Generate",
    use_container_width=True,
    type="primary",
):
    gen.load_state_dict(torch.load(model_output_path))

    fixed_noise = torch.randn((batch_size, z_dim)).to(device)

    with torch.no_grad():
        fake_flat = gen(fixed_noise)
        fake = fake_flat.reshape(-1, 1, 28, 28)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

        st.image(
            image=img_grid_fake.permute(1, 2, 0).cpu().numpy(),
            use_column_width=True,
        )
