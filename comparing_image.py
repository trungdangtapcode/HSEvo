from PIL import Image
import imageio
import os
from tqdm import tqdm

def create_side_by_side_gif(folder1_path, folder2_path, output_gif_path):
    # Get list of image files from both folders and sort them
    folder1_images = sorted([f for f in os.listdir(folder1_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    folder2_images = sorted([f for f in os.listdir(folder2_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Get lengths and maximum iterations
    len1, len2 = len(folder1_images), len(folder2_images)
    max_length = max(len1, len2)
    
    # List to store frames for GIF
    frames = []
    
    for i in tqdm(range(max_length)):
        # Get image paths (use last image if index exceeds length)
        img1_path = os.path.join(folder1_path, folder1_images[min(i, len1-1)])
        img2_path = os.path.join(folder2_path, folder2_images[min(i, len2-1)])
        
        # Open images
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # Ensure images have same height (resize if needed)
        if img1.height != img2.height:
            new_height = min(img1.height, img2.height)
            img1 = img1.resize((int(img1.width * new_height/img1.height), new_height))
            img2 = img2.resize((int(img2.width * new_height/img2.height), new_height))
        
        # Create new image with combined width
        combined_width = img1.width + img2.width
        new_img = Image.new('RGB', (combined_width, img1.height))
        
        # Paste images side by side
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))
        
        # Add frame to list
        frames.append(new_img)
    
    # Save as GIF
    imageio.mimsave(output_gif_path, frames, duration=0.15)  # 0.5 seconds per frame

# Example usage
folder2_path = 'outputs/plots/offline/20250409_005158 (44iter me eval2 hmean)'
folder1_path = 'outputs/plots/offline/20250408_234401 (kaggle 80i mean)'
output_gif_path = 'output.gif'

create_side_by_side_gif(folder1_path, folder2_path, output_gif_path)