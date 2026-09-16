#!/usr/bin/env python3
"""Generate a deterministic vision test image: a 4-digit number plus a colored shape."""
from PIL import Image, ImageDraw, ImageFont

NUMBER = "4782"
FONT = "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"

img = Image.new("RGB", (800, 360), "white")
d = ImageDraw.Draw(img)
d.text((40, 70), NUMBER, font=ImageFont.truetype(FONT, 200), fill="black")
d.ellipse((600, 90, 760, 250), fill=(220, 30, 30))
d.rectangle((0, 0, 799, 359), outline="black", width=4)
img.save("vision-test.png")
print(f"wrote vision-test.png (number={NUMBER}, circle=red)")
