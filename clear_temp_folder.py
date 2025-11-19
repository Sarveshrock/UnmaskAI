import os
import shutil
import time

def clean(temp_dir="temp"):
    """Safely clear the temp folder without crashing on Windows."""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        return

    for attempt in range(3):  # Retry up to 3 times if files are locked
        try:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except PermissionError:
                        time.sleep(0.3)
                        try:
                            os.remove(file_path)
                        except:
                            pass
            # Remove and recreate the folder
            shutil.rmtree(temp_dir, ignore_errors=True)
            os.makedirs(temp_dir, exist_ok=True)
            print(f"🧹 Temp folder cleared successfully (attempt {attempt+1})")
            return
        except Exception as e:
            print(f"⚠️ Cleanup attempt {attempt+1} failed: {e}")
            time.sleep(0.5)
    print("❌ Could not completely clear temp folder.")
