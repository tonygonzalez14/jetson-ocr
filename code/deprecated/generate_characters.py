"""
This script generates PNG images of uppercase and lowercase English letters, as well as digits 0-9.
It uses the PIL library to create and save images with the specified font and size to be used in the overlay.
Note: This script is no longer in use and was part of the development process.

Dependencies:
- PIL (Pillow)

Usage:
1. Ensure you have the PIL library installed.
2. Run the script to generate PNG images in the specified output folder.
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Define parameters
output_folder = 'letters_png/'
font_size = 80
font_path = 'arial.ttf'  
image_size = (128, 128)  

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through English alphabet (uppercase and lowercase)
for char_code in range(ord('A'), ord('Z')+1):
    char_upper = chr(char_code)
    char_lower = chr(char_code + 32) 
    
    # Create a new transparent image for uppercase
    img_upper = Image.new('RGBA', image_size, (0, 0, 0, 0))
    
    # Draw the uppercase letter onto the image
    draw_upper = ImageDraw.Draw(img_upper)
    font = ImageFont.truetype(font_path, font_size)
    
    # Get the bounding box of the uppercase text
    text_bbox_upper = draw_upper.textbbox((0, 0), char_upper, font=font)
    
    # Calculate the size of the uppercase text
    text_width_upper = text_bbox_upper[2] - text_bbox_upper[0]
    text_height_upper = text_bbox_upper[3] - text_bbox_upper[1]
    
    # Calculate the position to center the uppercase text
    text_position_upper = ((image_size[0] - text_width_upper) // 2, (image_size[1] - text_height_upper) // 2)
    
    # Draw the uppercase text onto the image
    draw_upper.text(text_position_upper, char_upper, font=font, fill=(0, 255, 0, 255))
    
    # Save the uppercase image
    img_upper.save(os.path.join(output_folder, f'{char_upper}_U.png'), 'PNG')
    
    # Create a new transparent image for lowercase
    img_lower = Image.new('RGBA', image_size, (0, 0, 0, 0))
    
    # Draw the lowercase letter onto the image
    draw_lower = ImageDraw.Draw(img_lower)
    text_bbox_lower = draw_lower.textbbox((0, 0), char_lower, font=font)
    text_width_lower = text_bbox_lower[2] - text_bbox_lower[0]
    text_height_lower = text_bbox_lower[3] - text_bbox_lower[1]
    text_position_lower = ((image_size[0] - text_width_lower) // 2, (image_size[1] - text_height_lower) // 2)
    draw_lower.text(text_position_lower, char_lower, font=font, fill=(0, 255, 0, 255))
    
    # Save the lowercase image
    img_lower.save(os.path.join(output_folder, f'{char_lower}_L.png'), 'PNG')

# Loop through numbers 0-9
for num in range(10):
    num_char = str(num)
    
    # Create a new transparent image for the number
    img_num = Image.new('RGBA', image_size, (0, 0, 0, 0))
    
    # Draw the number onto the image
    draw_num = ImageDraw.Draw(img_num)
    text_bbox_num = draw_num.textbbox((0, 0), num_char, font=font)
    text_width_num = text_bbox_num[2] - text_bbox_num[0]
    text_height_num = text_bbox_num[3] - text_bbox_num[1]
    text_position_num = ((image_size[0] - text_width_num) // 2, (image_size[1] - text_height_num) // 2)
    draw_num.text(text_position_num, num_char, font=font, fill=(0, 255, 0, 255))
    
    # Save the number image with correct filename
    img_num.save(os.path.join(output_folder, f'{num_char}.png'), 'PNG')
