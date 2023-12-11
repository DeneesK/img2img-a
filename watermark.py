from PIL import Image


def watermark_with_transparency(input_image_path,
                                watermark_image_path='./wm.png',
                                position=(0, 0)):
    base_image = Image.open(input_image_path).convert('RGBA')
    watermark = Image.open(watermark_image_path).convert('RGBA')
    width, height = base_image.size
    w_w = int(width / 2)
    w_h = int(w_w / 5.9)
    watermark = watermark.resize((w_w, w_h))
    x_watermark = int(width * 0.05)
    y_watermark = int(height * 0.9)
    position = (x_watermark, y_watermark)

    transparent = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    transparent.paste(base_image, (0, 0))
    transparent.paste(watermark, position, mask=watermark)
    transparent.save(input_image_path)


watermark_with_transparency('output (10).png')
