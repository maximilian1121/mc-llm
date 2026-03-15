url = "https://piston-data.mojang.com/v1/objects/ba2df812c2d12e0219c489c4cd9a5e1f0760f5bd/client.jar"
import requests

print("Fetching latest minecraft jar file!")
content = requests.get(url=url).content

print("Creating directories")
import os

os.makedirs("models", exist_ok=True)

os.makedirs("temp", exist_ok=True)

print("Writing client.jar data to file")
with open("temp/client.jar", "wb") as f:
    f.write(content)

import os
import zipfile
import shutil
from PIL import Image


def extract_items_only(jar_path, output_dir="mc_dataset_rgba"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    count = 0
    if not os.path.exists(jar_path):
        print(f"Error: Could not find {jar_path}")
        return

    print(f"Opening {jar_path}...")
    with zipfile.ZipFile(jar_path, "r") as jar:
        for file_path in jar.namelist():
            if file_path.startswith(
                "assets/minecraft/textures/item/"
            ) and file_path.endswith(".png"):

                with jar.open(file_path) as file:
                    try:
                        img = Image.open(file)

                        if img.size == (16, 16):
                            clean_name = file_path.split("/")[-1]

                            rgba_img = img.convert("RGBA")
                            rgba_img.save(os.path.join(output_dir, clean_name))
                            count += 1
                    except Exception as e:
                        continue

    print(f"Success! {count} 16x16 item textures extracted to '{output_dir}'.")


print("Extracting all minecraft item textures")

extract_items_only("temp/client.jar")

print("Adding context to all dataset images")

for filename in os.listdir("./mc_dataset_rgba"):
    if filename.endswith(".png"):
        item_name = filename.replace(".png", "").replace("_", " ")
        caption = f"a 16x16 minecraft item texture of {item_name}, pixel art, transparent background"

        with open(
            os.path.join("./mc_dataset_rgba", filename.replace(".png", ".txt")), "w"
        ) as f:
            f.write(caption)

print("Done! Every item now has a matching text caption.")
print("Success! Please now run either seed/text_based.py to train/run the ai model!")
