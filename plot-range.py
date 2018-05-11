#!/usr/bin/python2 -tt

import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("Index", help="Display pictures starting with Index", type=int, nargs='?', default=50)
parser.add_argument("Rows", help="Number of Rows to display", type=int, nargs='?', default=4)
parser.add_argument("Columns", help="Number of Columns to display", type=int, nargs='?', default=10)
args = parser.parse_args()

start_index = args.Index
subplot_row = args.Rows
subplot_col = args.Columns

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
traning_data_len = len(training_data)

def decode_to_int( dc ):
	for i in range(0,len(dc)):
		if dc[i] == 1:
			return i

plt.style.use('grayscale')
for subplot_index in range(0, subplot_row*subplot_col):
	current_index = (start_index+subplot_index) % traning_data_len
	digit_classification = training_data[current_index][1]
	pixels = training_data[current_index][0]
	glyph = np.empty((28, 28))
	for i in range(0,784):
		glyph[i/28][i%28]=pixels[i]

	plt.subplot(subplot_row,subplot_col,subplot_index+1)
	plt.imshow(glyph, interpolation='none')
	plt.title(str(current_index) + "=" + str(decode_to_int(digit_classification)), fontsize=10)
	plt.axis('off')

plt.show()

