#!/usr/bin/env python
from PIL import Image,ImageFont, ImageDraw
from emoji import unicode_codes
import os

def main():
  os.makedirs('data/emoji/training')
  font = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', 72, encoding='unic')
  for alias, ch in unicode_codes.EMOJI_UNICODE.items():
    width, height = font.getsize(ch)
    side = max(width, height) + 1
    image = Image.new('RGB', (side, side), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    x = (side - width) / 2
    y = (side - height) / 2
    draw.text((x, y), ch, font=font, fill=(0, 0, 0))
    draw.text((x + 1, y + 1), ch, font=font, fill=(0, 0, 0))
    image = image.resize((32, 32), resample=Image.ANTIALIAS)
    alias = alias.replace(':', '')
    image.save('data/emoji/training/%s.png' % alias)



if __name__ == '__main__':
  main()





