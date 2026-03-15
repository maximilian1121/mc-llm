import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import gradio as gr

# --- CONFIG ---
DATA_DIR = "./mc_dataset_rgba"
MODEL_PATH = "models/mc_text_item_gen.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 100

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# 1. Load CLIP Components
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
text_encoder.requires_grad_(False)

# 2. Setup Conditional U-Net (4 channels for RGBA)
model = UNet2DConditionModel(
    sample_size=16,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(128, 256, 512),
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=768,
).to(DEVICE)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# --- DATASET HANDLING ---
class MinecraftDataset(Dataset):
    def __init__(self, root_dir, tokenizer):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.images = [f for f in os.listdir(root_dir) if f.endswith(".png")]
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (16, 16), interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert("RGBA")
        # Generate caption from filename: "golden_apple.png" -> "golden apple"
        caption = self.images[idx].replace(".png", "").replace("_", " ")

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        return self.transform(image), tokens


# --- TRAINING LOGIC ---
def run_training():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found. Add your sprites there!")
        return

    dataset = MinecraftDataset(DATA_DIR, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"✨ Starting training on {DEVICE}...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for images, input_ids in loader:
            images, input_ids = images.to(DEVICE), input_ids.to(DEVICE)

            # Get text embeddings
            encoder_hidden_states = text_encoder(input_ids)[0]

            # Add noise
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, 1000, (images.shape[0],), device=DEVICE).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # Predict & Backprop
            noise_pred = model(noisy_images, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
            torch.save(model.state_dict(), MODEL_PATH)

    print("✅ Training complete. Model saved.")


# --- GENERATION LOGIC ---
@torch.no_grad()
def generate_batch(prompt, base_seed, num_images):
    model.eval()
    results = []

    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    encoder_hidden_states = text_encoder(inputs.input_ids.to(DEVICE))[0]

    for i in range(int(num_images)):
        generator = torch.manual_seed(int(base_seed) + i)
        latents = torch.randn((1, 4, 16, 16), generator=generator).to(DEVICE)
        noise_scheduler.set_timesteps(50)

        for t in noise_scheduler.timesteps:
            output = model(latents, t, encoder_hidden_states).sample
            latents = noise_scheduler.step(output, t, latents).prev_sample

        img_np = (latents / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
        results.append(Image.fromarray((img_np * 255).astype("uint8"), "RGBA"))
    return results


# --- WEB UI ---
custom_css = """
.gradio-container img { image-rendering: pixelated; }
#pixel-gallery img { width: 100% !important; height: auto !important; }
"""


def launch_interface():
    with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as demo:
        gr.Markdown(
            "# ⛏️ PixelArt Diffusion Lab\nGenerate custom 16x16 Minecraft items."
        )
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="enchanted blaze rod")
                seed = gr.Number(label="Seed", value=42)
                num = gr.Slider(label="Count", minimum=1, maximum=16, step=1, value=4)
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out_gallery = gr.Gallery(
                    label="Output", columns=4, object_fit="contain"
                )

        btn.click(generate_batch, [prompt, seed, num], out_gallery)
    demo.launch()


if __name__ == "__main__":
    import sys

    # To train, run: python script.py train
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        run_training()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        launch_interface()
    else:
        print("No model found. Run 'python your_script.py train' first.")
