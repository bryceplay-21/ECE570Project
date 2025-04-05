import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import timm  # For pretrained classifiers
import torch.nn.functional as F
from collections import Counter

class SefaExtension:
    def __init__(self, generator, device='cuda'):
        self.device = device
        self.generator = generator.to(device)
        self.latent_dim = self.generator.z_dim
        self.num_layers = self.generator.num_ws
        self.directions = None

    def extract_affine_weights(self):
        weight_list = []
        for name, mod in self.generator.named_modules():
            if 'affine' in name and hasattr(mod, 'weight'):
                weight_list.append(mod.weight.data.cpu().numpy())
        shapes = [w.shape[0] for w in weight_list]
        common_shape = Counter(shapes).most_common(1)[0][0]
        weight_list = [w for w in weight_list if w.shape[0] == common_shape]
        W = np.concatenate(weight_list, axis=1)
        return W

    def perform_svd(self, num_components=10):
        W = self.extract_affine_weights()
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        self.directions = Vh[:num_components]
        return self.directions

    def cluster_directions(self, num_clusters=3):
        if self.directions is None:
            raise ValueError("Call perform_svd() first.")
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.directions)
        return kmeans.labels_

    def edit_latent_code(self, z, direction_weights):
        w = self.generator.mapping(z, None)
        batch, num_layers, latent_dim = w.shape
        flattened_w = w.view(batch, -1)  # e.g., [1, 9216]
    
        for idx, weight in direction_weights.items():
            direction = torch.tensor(self.directions[idx], device=self.device).float()
    
            # Ensure both tensors match in size
            min_len = min(flattened_w.shape[1], direction.shape[0])
            direction = direction[:min_len]
            flattened_w[:, :min_len] += weight * direction.unsqueeze(0)
    
        w = flattened_w.view(batch, num_layers, latent_dim)
        return w

    def generate_images(self, num_samples=5, direction_weights={0: 5.0}):
        self.generator.eval()
        z_samples = torch.randn(num_samples, self.latent_dim).to(self.device)
        all_images = []
        for z in z_samples:
            z = z.unsqueeze(0)
            ws = self.edit_latent_code(z, direction_weights)
            img = self.generator.synthesis(ws, noise_mode='const')
            all_images.append(img)
        return torch.cat(all_images)

    def visualize_latent_variation(self, direction_idx=0, num_steps=5, scale_range=(-5, 5)):
        scales = np.linspace(scale_range[0], scale_range[1], num_steps)
        z = torch.randn(1, self.latent_dim).to(self.device)
        imgs = []
        for alpha in scales:
            w = self.edit_latent_code(z, {direction_idx: alpha})
            img = self.generator.synthesis(w, noise_mode='const')
            imgs.append(img)
        grid = torch.cat(imgs)
        save_image(grid, f'sefa_direction_{direction_idx}.png', nrow=num_steps, normalize=True, value_range=(-1, 1))

    def evaluate_attribute_change(self, attribute_classifier, direction_idx=0, steps=[-5, 0, 5]):
        results = {}
        z = torch.randn(1, self.latent_dim).to(self.device)
        for alpha in steps:
            w = self.edit_latent_code(z, {direction_idx: alpha})
            img = self.generator.synthesis(w, noise_mode='const')
            img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            logits = attribute_classifier(img_resized)
            results[alpha] = logits.detach().cpu().numpy()
        return results

    def launch_interactive_ui(self, num_directions=3):
        sliders = [widgets.FloatSlider(value=0.0, min=-10.0, max=10.0, step=0.5, description=f'Dir {i}') for i in range(num_directions)]
        button = widgets.Button(description="Generate Image")
        output = widgets.Output()

        def on_button_clicked(b):
            with output:
                output.clear_output()
                z = torch.randn(1, self.latent_dim).to(self.device)
                direction_weights = {i: sliders[i].value for i in range(num_directions)}
                w = self.edit_latent_code(z, direction_weights)
                img = self.generator.synthesis(w, noise_mode='const')
                img = (img.clamp(-1, 1) + 1) / 2.0
                img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
                plt.imshow(img[0].permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                plt.show()

        button.on_click(on_button_clicked)
        display(widgets.VBox(sliders + [button, output]))

class SimpleAttributeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=2)

    def forward(self, x):
        return self.model(x)
