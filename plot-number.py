#!/usr/bin/python2 -tt

import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("Index", help="Search from index", type=int, nargs='?', default=50)
parser.add_argument("Target", help="Search for Target number", type=int, nargs='?', choices=range(0, 10), metavar="[0-9]", default=7)
parser.add_argument("Count", help="Display Count occurences", type=int, nargs='?', default=40)
args = parser.parse_args()

start_index = args.Index
search_target = args.Target
search_count = args.Count
subplot_col = 10
subplot_row=search_count/subplot_col
if search_count%subplot_col != 0:
    subplot_row += 1

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print(np.shape(training_data))		# (50000, 2)
print(len(training_data[0][0]))		# 784 - Pixels
print(len(training_data[0][1]))		# 10 - Digit encoded in bit position

traning_data_len = len(training_data)

def decode_to_int( dc ):
	for i in range(0,len(dc)):
		if dc[i] == 1:
			return i

plt.style.use('grayscale')
current_index = start_index
subplot_index = 0
while(subplot_index < search_count):
	digit_classification = training_data[current_index][1]
	if decode_to_int(digit_classification) == search_target:
		pixels = training_data[current_index][0]
		glyph = np.empty((28, 28))
		for i in range(0,784):
			glyph[i/28][i%28]=pixels[i]

		plt.subplot(subplot_row,subplot_col,subplot_index+1)
		plt.imshow(glyph, interpolation='none')
		plt.title(str(current_index) + "=" + str(search_target), fontsize=10)
		plt.axis('off')
		subplot_index += 1

	current_index = (current_index+1) % traning_data_len

plt.show()

