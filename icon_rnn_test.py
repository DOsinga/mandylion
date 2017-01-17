import unittest
from icon_rnn import encode_icon, decode_icon

class FakeBM(object):
  def __init__(self, pixel_size):
    self.pixel_size = pixel_size
    self.pixels = [[0 for x in range(pixel_size)] for y in range(pixel_size)]
    self.mode = 'RGB'

  def setpixel(self, x, y):
    self.pixels[x][y] = 1

  def getpixel(self, x, y=None):
    if y is None:
      x, y = x
    if self.pixels[x][y] == 1:
      return 0, 0, 0
    else:
      return 255, 255, 255

  def setpixels(self, pixels):
    for x, y in pixels:
      self.setpixel(x, y)

class TestIconParsing(unittest.TestCase):

    def test_parse_icon(self):
      icon = FakeBM(32)
      icon.setpixel(16,1)
      icon.setpixel(16,30)
      encoded = encode_icon(icon, 32)
      decoded = list(decode_icon(encoded, 32))
      self.assertEqual(len(decoded), 2)
      self.assertEqual(decoded, [(16, 1), (16, 30)])

      for encoded, icon_size in (
          [[32, 4, 5, 32, 4, 5], 32],
          [[16, 4, 5, 16, 4, 5], 16],
          [[8, 4, 8, 8, 4], 8],
        ):
        icon = FakeBM(icon_size)
        icon.setpixels(decode_icon(encoded, icon_size))
        self.assertEqual(encoded, encode_icon(icon, icon_size))


if __name__ == '__main__':
    unittest.main()
