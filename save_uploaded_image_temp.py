import os
import time
from PIL import Image

def save_image(uploaded_file, temp_dir="temp"):
    """Safely saves uploaded image to temp folder, handling locked files on Windows."""

    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # Before saving, try to remove any old files safely
    possible_old_files = [
        os.path.join(temp_dir, "delete.mp4"),
        os.path.join(temp_dir, "delete.mov"),
        os.path.join(temp_dir, "delete.jpg"),
        os.path.join(temp_dir, "delete.png"),
    ]

    for old_file in possible_old_files:
        if os.path.exists(old_file):
            for attempt in range(3):
                try:
                    os.remove(old_file)
                    print(f"🧹 Removed old file: {old_file}")
                    break
                except PermissionError:
                    print(f"⚠️ File {old_file} locked (attempt {attempt+1}), retrying...")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"⚠️ Error removing {old_file}: {e}")
                    break

    # Save new image
    save_path = os.path.join(temp_dir, "delete.jpg")
    try:
        image = Image.open(uploaded_file)
        image.save(save_path)
        print(f"✅ Image saved to {save_path}")
    except Exception as e:
        print(f"❌ Error saving uploaded image: {e}")
