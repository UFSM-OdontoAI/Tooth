from PIL import Image, ImageDraw, ImageFont

def process_image(pil_image):
    img = pil_image.convert('RGB')
    gray = img.convert('L').convert('RGB')
    draw = ImageDraw.Draw(gray)
    w, h = gray.size
    text = "Processado"
    try:
        font = ImageFont.truetype("arial.ttf", size=max(20, w//20))
    except:
        font = ImageFont.load_default()
    text_w, text_h = draw.textsize(text, font=font)
    draw.rectangle(((0, h - text_h - 10), (text_w + 10, h)), fill=(0,0,0))
    draw.text((5, h - text_h - 5), text, fill=(255,255,255), font=font)
    return gray
