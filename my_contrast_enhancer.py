from PIL import Image, ImageEnhance, ImageFilter

input_file_name = "3e1e34c7c1ac4d-foto-while.png"
output_file_name= "3e1e34c7c1ac4d-foto-while-contrast-enhanced.png"

im = Image.open(input_file_name) #input image
im = im.filter(ImageFilter.MedianFilter())
enhancer = ImageEnhance.Contrast(im)
im = enhancer.enhance(2)
im = im.convert('1')
im.save(output_file_name) #ouput image