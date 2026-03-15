import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
import gradio as gr

# --- CONFIGURATION ---
DATA_DIR = "./mc_dataset_rgba"  # Your folder from the previous step
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50

import os

os.makedirs("models", exist_ok=True)


# --- DATASET ---
class MCDataset(Dataset):
    def __init__(self, folder):
        self.images = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")
        ]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts 0-255 to 0.0-1.0
                transforms.Normalize(
                    [0.5] * 4, [0.5] * 4
                ),  # Scales to [-1, 1] for the AI
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGBA")
        return self.transform(img)


# --- MODEL SETUP ---
# Native 16x16 with 4 channels (RGBA)
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# 1. Load CLIP to process text
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)

# 2. Upgrade the U-Net to a "Conditional" version
model = UNet2DConditionModel(
    sample_size=16,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(128, 256, 512),  # Increased power for text understanding
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=768,  # Matches CLIP's output
).to(DEVICE)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# --- TRAINING FUNCTION ---
def train():
    dataset = MCDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()

    print(f"Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        for step, batch in enumerate(loader):
            clean_images = batch.to(DEVICE)
            noise = torch.randn(clean_images.shape).to(DEVICE)
            timesteps = torch.randint(
                0, 1000, (clean_images.shape[0],), device=DEVICE
            ).long()

            # Add noise to the images
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict the noise
            noise_pred = model(noisy_images, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), "mc_16x16_gen.pth")


# --- GENERATION (INFERENCE) ---
@torch.no_grad()
def generate(seed):
    model.eval()
    generator = torch.manual_seed(int(seed))
    # Start with pure random RGBA noise
    image = torch.randn((1, 4, 16, 16), generator=generator).to(DEVICE)

    # Loop to "denoise" the image back to a texture
    for t in noise_scheduler.timesteps:
        model_output = model(image, t).sample
        image = noise_scheduler.step(model_output, t, image).prev_sample

    # Process back to a viewable image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((image * 255).astype("uint8"), "RGBA")


# --- WEB UI ---
def launch_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# ⛏️ Minecraft 16x16 AI Generator")
        with gr.Row():
            seed_input = gr.Number(label="Random Seed", value=42)
            btn = gr.Button("Generate Texture")
        output_img = gr.Image(label="Result", image_mode="RGBA")

        btn.click(fn=generate, inputs=seed_input, outputs=output_img)
    demo.launch()


if __name__ == "__main__":
    # If model doesn't exist, train it
    if not os.path.exists("models/mc_16x16_gen.pth"):
        train()
    else:
        model.load_state_dict(
            torch.load("models/mc_16x16_gen.pth", map_location=DEVICE)
        )

    launch_ui()
