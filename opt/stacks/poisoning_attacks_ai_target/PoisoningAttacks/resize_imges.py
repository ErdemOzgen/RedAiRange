from PIL import Image, ImageOps
import os
import numpy as np

def pad_image_to_square(img, pad_color=(255, 255, 255)):
    # Ensure the image is in RGB mode
    img = img.convert('RGB')
    
    # Determine the larger dimension to set the size of the square
    max_dim = max(img.size)
    
    # Calculate padding
    pad_width = max_dim - img.size[0]
    pad_height = max_dim - img.size[1]
    
    # Add padding to make the image square
    padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))
    img = ImageOps.expand(img, padding, pad_color)
    
    return img

# Example usage
input_dir = 'rutan-bumerang'
output_dir = 'resized-images-rb'
final_size = (32, 32)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List to store image data
image_data = []

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        img = Image.open(os.path.join(input_dir, filename))
        img = pad_image_to_square(img)
        img_resized = img.resize(final_size, Image.ANTIALIAS)
        img_resized.save(os.path.join(output_dir, filename))
        
        # Convert the image to a NumPy array and add it to the list
        img_array = np.array(img_resized)
        image_data.append(img_array)

# Convert the list of image data to a NumPy array
image_data = np.array(image_data)

# Save the NumPy array to a file
np.save(os.path.join(output_dir, 'images.npy'), image_data)
    
# Print the shape of the saved NumPy array to inspect it
print('Saved images.npy with shape:', image_data.shape)
custom_images = np.load(os.path.join(output_dir, 'images.npy'))
print(custom_images.shape)

