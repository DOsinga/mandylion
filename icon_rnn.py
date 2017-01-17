from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import os
import time
from collections import Counter
from PIL import Image, ImageDraw
import argparse
import json
import scipy.misc
import colorsys
from skimage.measure import label

NUM_TRAIN_BATCHES = 10000
MODEL_FILE_NAME = 'tensorflow_inception_graph.pb'
LABELS_FILE_NAME = 'imagenet_comp_graph_label_strings.txt'


# The model below is roughly from:
#   https://github.com/spiglerg/RNN_Text_Generation_Tensorflow
class ModelNetwork:
  def __init__(self, in_size, lstm_size, num_layers, out_size, session, learning_rate=0.003, name="rnn", drop_out=0):
    self.scope = name

    self.in_size = in_size
    self.lstm_size = lstm_size
    self.num_layers = num_layers
    self.out_size = out_size

    self.session = session

    self.learning_rate = tf.constant(learning_rate)

    # Last state of LSTM, used when running the network in TEST mode
    self.lstm_last_state = np.zeros((self.num_layers * 2 * self.lstm_size,))

    with tf.variable_scope(self.scope):
      ## (batch_size, timesteps, in_size)
      self.xinput = tf.placeholder(tf.float32, shape=(None, None, self.in_size), name="xinput")
      self.lstm_init_value = tf.placeholder(tf.float32, shape=(None, self.num_layers * 2 * self.lstm_size),
                                            name="lstm_init_value")


      self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False)
      self.keep_prob = tf.placeholder(tf.float32)
      self.lstm_cell = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell, self.keep_prob)
      self.lstm = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell] * self.num_layers, state_is_tuple=False)

      # LSTM
      #cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, forget_bias=1.0)
      #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers)
      #if drop_out:
      #  cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - drop_out)
      #self.cell = cell

      # Iteratively compute output of recurrent network
      outputs, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, self.xinput, initial_state=self.lstm_init_value,
                                                       dtype=tf.float32)

      # Linear activation (FC layer on top of the LSTM net)
      self.rnn_out_W = tf.Variable(tf.random_normal((self.lstm_size, self.out_size), stddev=0.01))
      self.rnn_out_B = tf.Variable(tf.random_normal((self.out_size,), stddev=0.01))

      outputs_reshaped = tf.reshape(outputs, [-1, self.lstm_size])
      network_output = (tf.matmul(outputs_reshaped, self.rnn_out_W) + self.rnn_out_B)

      batch_time_shape = tf.shape(outputs)
      self.final_outputs = tf.reshape(tf.nn.softmax(network_output),
                                      (batch_time_shape[0], batch_time_shape[1], self.out_size))

      ## Training: provide target outputs for supervised training.
      self.y_batch = tf.placeholder(tf.float32, (None, None, self.out_size))
      y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])

      self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network_output, y_batch_long))
      self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.cost)

  ## Input: X is a single element, not a list!
  def run_step(self, x, init_zero_state=True):
    ## Reset the initial state of the network.
    if init_zero_state:
      init_value = np.zeros((self.num_layers * 2 * self.lstm_size,))
    else:
      init_value = self.lstm_last_state

    out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state],
                                            feed_dict={self.xinput: [x],
                                                       self.lstm_init_value: [init_value],
                                                       self.keep_prob: 1.0})

    self.lstm_last_state = next_lstm_state[0]

    return out[0][0]

  ## xbatch must be (batch_size, timesteps, input_size)
  ## ybatch must be (batch_size, timesteps, output_size)
  def train_batch(self, xbatch, ybatch, keep_prop=1.0):
    init_value = np.zeros((xbatch.shape[0], self.num_layers * 2 * self.lstm_size))

    cost, _ = self.session.run([self.cost, self.train_op],
                               feed_dict={self.xinput: xbatch,
                                          self.y_batch: ybatch,
                                          self.lstm_init_value: init_value,
                                          self.keep_prob: keep_prop})

    return cost


def embed(data_, width):
  data = np.zeros((len(data_), width))
  for row, col in enumerate(data_):
    data[row, col] = 1
  return data


def decode_embed(array):
  return array.index(1)


def encode_icon(img, icon_size):
  if img.mode == 'P' or img.mode == '1':
    img = img.convert('RGBA')
  size_last_x = 0
  encoded = []
  for y in range(icon_size):
    for x in range(icon_size):
      p = img.getpixel((x, y))
      if sum(img.getpixel((x, y))[:3]) < 700 and (len(p) == 3 or p[3] > 50):
        encoded.append(x)
        size_last_x = len(encoded)
    encoded.append(icon_size)
  return encoded[:size_last_x]


def decode_icon(encoded, icon_size, rows=None, stop_token=None):
  y = 0
  for idx in encoded:
    if idx == icon_size:
      y += 1
    elif idx == icon_size + 1:
      break
    else:
      x = idx
      yield x, y


def load_icons(image_path, icon_size):
  print('scanning images', image_path)
  icon_count = 0
  res = [icon_size + 1]
  for fn_img in os.listdir(image_path):
    img = Image.open(os.path.join(image_path, fn_img))
    if sum(img.size) != icon_size * 2:
      continue
    res.extend(encode_icon(img, icon_size))
    res.append(icon_size + 1)
    icon_count += 1
  print('done: %s icons, %s total_size, %2.2f points per icon' % (icon_count, len(res), len(res) / float(icon_count)))
  return res


def train_model(net, sess, data, model_path, saver, in_size, icon_size, batch_size=128, time_steps=500, newline_starts=False, keep_prob=1.0):
  last_time = time.time()

  batch = np.zeros((batch_size, time_steps, in_size))
  batch_y = np.zeros((batch_size, time_steps, in_size))

  min_loss = None
  if newline_starts:
    possible_batch_ids = [batch_id for batch_id in range(data.shape[0] - time_steps - 1) if data[batch_id][icon_size] or data[batch_id][icon_size + 1]]
  else:
   possible_batch_ids = range(data.shape[0] - time_steps - 1)

  print('%d number of training samples' % (len(possible_batch_ids),))
  for i in range(NUM_TRAIN_BATCHES):
    batch_ids = random.sample(possible_batch_ids, batch_size)

    for j in range(time_steps):
      ind1 = [k + j for k in batch_ids]
      ind2 = [k + j + 1 for k in batch_ids]

      batch[:, j, :] = data[ind1, :]
      batch_y[:, j, :] = data[ind2, :]

    cst = net.train_batch(batch, batch_y, keep_prop=keep_prob)

    if i > 0 and (i % 100) == 0:
      new_time = time.time()
      diff = new_time - last_time
      last_time = new_time

      print('Batch: %d, loss: %2.2f, speed: %2.2f sec per batch' % (i, cst, diff / 100))
      if min_loss is None or cst < min_loss:
        min_loss = cst
        saver.save(sess, model_path)

  saver.save(sess, model_path)


def classify(net, inception_model, rnn_icons, icon_size):
  print('loading model')
  graph_def = tf.GraphDef()
  with open(os.path.join(inception_model, MODEL_FILE_NAME)) as f:
      graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def)
  inception_session = tf.Session()
  labels = open(os.path.join(inception_model, LABELS_FILE_NAME), 'r').read().splitlines()
  block_size = 224 / icon_size

  json_path = os.path.join(rnn_icons, 'weights.json')
  res = {}
  if os.path.isfile(json_path):
    res = json.load(open(json_path))
  net.run_step(embed([icon_size + 1], icon_size + 2), True)

  while True:
    out = net.run_step(embed([icon_size + 1], icon_size + 2), False)
    matrix = [' ' for _ in range(icon_size * icon_size)]
    img1 = np.full(shape=(224, 224, 3), fill_value=255)
    encoded = []
    while True:
      idx = np.random.choice(range(icon_size + 2), p=out)
      if idx == icon_size + 1:
        break
      encoded.append(idx)
      out = net.run_step(embed([idx], icon_size + 2), False)

    for x, y in decode_icon(encoded, icon_size):
      if y >= icon_size:
        print('invalid icon')
        break
      matrix[x + y * icon_size] = 'X'
      for x1 in range(x * block_size, (x + 1) * block_size):
        for y1 in range(y * block_size, (y + 1) * block_size):
          img1[y1][x1] = [0, 0, 0]
    else:
      result = inception_session.run('import/output2:0', feed_dict={'import/input:0': img1.reshape((1,) + img1.shape)})
      result = np.squeeze(result)
      top_index, weight = max(enumerate(result), key=lambda t:t[1])
      label = labels[top_index]
      if weight > res.get(label, 0.25):
        scipy.misc.imsave('%s/%s.png' % (rnn_icons, label), img1)
        print(label, weight)
        res[label] = float(weight)
        json.dump(res, file(json_path, 'w'), indent=2)


def generate_icon(net, icon_size):
  while True:
    out = net.run_step(embed([icon_size + 1], icon_size + 2), False)
    encoded = []
    while True:
      idx = np.random.choice(range(icon_size + 2),
                                 p=out)  # Sample character from the network according to the generated output probabilities
      if idx == icon_size + 1:
        if encoded:
          break
        continue
      encoded.append(idx)
      out = net.run_step(embed([idx], icon_size + 2), False)

    icon_coos = list(decode_icon(encoded, icon_size))
    if all(x < icon_size for x, y in icon_coos) and all(y < icon_size for x, y in icon_coos):
      return icon_coos


def create_poster(net, icon_size, poster_path):
  icon_size_2 = (icon_size + 2)
  poster = Image.new('RGB', (icon_size_2 * 20, icon_size_2 * 15), color=(255, 255, 255))

  net.run_step(embed([icon_size + 1], icon_size + 2), True)

  for x in range(20):
    for y in range(15):
      icon_coos = generate_icon(net, icon_size)
      rgb = colorsys.hls_to_rgb(random.random(), 0.30, 0.9)
      rgb = tuple(int(v * 255) for v in rgb)
      for x1, y1 in icon_coos:
        poster.putpixel((icon_size_2 * x + x1, icon_size_2 * y + y1), rgb)
  poster.save(poster_path)


def is_boundary_color(labeled, fill_center_label):
  arr_w, arr_h = labeled.shape
  for x1 in range(arr_w):
    if labeled[x1][0] == fill_center_label or labeled[x1][arr_h - 1] == fill_center_label:
      return True
  for y1 in range(arr_h):
    if labeled[0][y1] == fill_center_label or labeled[arr_w - 1][y1] == fill_center_label:
      return True
  return False


def create_stela(net, icon_size, stela_path):
  padding = 5
  columns = 20
  column_width = (icon_size + 4)
  height = icon_size * 15
  background_color = (205, 193, 167)
  poster = Image.new('RGB', (padding * 2 + column_width * columns, height + padding * 2), color=background_color)
  poster_draw = ImageDraw.ImageDraw(poster)

  light_line_color = (108, 68, 38)
  line_color = (25, 34, 43)
  fill_color = (161, 126, 44)

  for column in range(columns + 1):
    x = column * column_width + 1
    poster_draw.line((x, padding, x, height - padding), fill=line_color, width=padding - 2)

  net.run_step(embed([icon_size + 1], icon_size + 2), True)

  for x in range(columns):
    y = 0
    previous_width = None
    previous_min_x = None
    delta_y = 0
    x_offset = padding + column_width * x
    while True:
      icon_coos = generate_icon(net, icon_size)
      y_min = min(y for _, y in icon_coos)
      y_max = max(y for _, y in icon_coos)
      x_min = min(x for x, _ in icon_coos)
      x_max = max(x for x, _ in icon_coos)
      if y_max - y_min + y + 5 + padding > height:
        break

      # put two next to each other, move the old one to the side
      # and adjust the coordinates of the new one:
      if previous_width and previous_width + x_max - x_min < icon_size - 3:
          y -= delta_y
          new_left = (icon_size - previous_width - x_max + x_min) / 2 - 1
          if new_left < previous_min_x:
            for y1 in range(y + padding + 1, y + delta_y + padding + 1):
              for x1 in range(0, previous_width):
                poster.putpixel((x_offset + new_left + x1, y1), poster.getpixel((x_offset + x1 + previous_min_x, y1)))
              for x1 in range(new_left + previous_width, icon_size - 1):
                poster.putpixel((x_offset + x1, y1), background_color)
          delta_y = max(delta_y, y_max - y_min + 4)
          x_shift = new_left + previous_width + 1 - x_min
          x_min += x_shift
          x_max += x_shift
          icon_coos = [(x1 + x_shift, y1) for x1, y1 in icon_coos]
          previous_width = None
      else:
        delta_y = y_max - y_min + 4
        previous_min_x = x_min
        previous_width = x_max - x_min

      arr = np.zeros((x_max - x_min + 1, y_max - y_min + 1))
      for x1, y1 in icon_coos:
        arr[x1 - x_min][y1 - y_min] = 1
      labeled = label(arr, background=2, connectivity=1)
      arr_w, arr_h = arr.shape
      fill_center_label = labeled[arr_w / 2, arr_h / 2]
      if not is_boundary_color(labeled, fill_center_label):
        left_center_label = labeled[0, arr_h / 2]
        for x1 in range(arr_w):
          l = labeled[x1, arr_h / 2]
          if l != left_center_label:
            if l != fill_center_label:
              for x2 in range(arr_w):
                for y2 in range(arr_h):
                  if labeled[x2, y2] == fill_center_label:
                    poster.putpixel((x_offset + x2 + x_min,
                                     padding + y + y2 + 1),
                                    fill_color)

      if random.randint(0, 1) == 1:
        rgb = line_color
      else:
        rgb = light_line_color
      rgb = tuple(x + random.randint(-2, 2) for x in rgb)
      for x1, y1 in icon_coos:
        poster.putpixel((x_offset + x1,
                         padding + y + y1 - y_min + 1),
                         rgb)
      y += delta_y
  poster.save(stela_path)


def main(data_set, icon_size, mode, inception_model):
  base_path = os.path.join('data', data_set)
  in_size = out_size = icon_size + 2

  ## Initialize the network
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.InteractiveSession(config=config)
  net = ModelNetwork(in_size=in_size,
                     lstm_size=192,
                     num_layers=2,
                     out_size=out_size,
                     session=sess,
                     learning_rate=0.002,
                     name="icon_rnn_network")
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())

  rnn_model_path = os.path.join(base_path, 'model.ckpt')
  if mode == 'train':
    icons = load_icons(os.path.join(base_path, 'training'), icon_size)
    train_model(net, sess, embed(icons, in_size), rnn_model_path, saver, in_size, icon_size)
  else:
    saver.restore(sess, rnn_model_path)
    poster_path = os.path.join(base_path, 'poster.png')
    if mode == 'poster':
      create_poster(net, icon_size, poster_path)
    elif mode == 'stela':
      create_stela(net, icon_size, poster_path)
    else:
      classify(net, inception_model, os.path.join(base_path, 'classified'), icon_size)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='RNN Icon generation')
  parser.add_argument('--dataset', type=str, default='32x32',
                      help='Which data set to run on. Dataset should be a subdirectory of the data directory. ')
  parser.add_argument('--size', type=int, default=32,
                      help='Size of the icons.')
  parser.add_argument('--mode', choices=['train', 'classify', 'poster', 'stella'],
                      help='What to do')
  parser.add_argument('--inception_model', type=str, default='inception5h',
                      help='Inception model for recognizing results')
  args = parser.parse_args()
  main(args.dataset, args.size, args.mode, args.inception_model)
