# Artificial Neural Network Library for C

This repository contains a library for building and training artificial neural networks (**annl.c**). The library is written in ANSI C and is designed as a simple, general, and fast platform for supervised machine learning for C programmers. It is a work in progress.

# License and Copyright

All files in this repository are released under the [GNU General Public License, Version 3](https://www.gnu.org/licenses/gpl-3.0.en.html) as per the included [license](https://github.com/jolejarz/neural-network-c/blob/main/LICENSE.txt) and [copyright](https://github.com/jolejarz/neural-network-c/blob/main/COPYRIGHT.txt) files.

# Examples

Several examples demonstrating the use of the library have been completed.

## XOR Logic Gate

In the file **xor.c**, we train the XOR logic gate using a simple network with one hidden layer consisting of three units. All layers are fully connected.

## Seven-Segment Display

The seven-segment display is a ubiquitous device for rendering the ten decimal digits 0 through 9. For the purposes of this example, consider a seven-segment display as rendering the sixteen hexadecimal digits 0 through 9 and A through F. Since there are seven segments to the display, and since each segment can be either lit or not lit, each hexadecimal digit's representation on the display can be specified as a sequence of seven bits. We would like to train a neural network to convert the seven-bit representation of the hexadecimal digit from the display to the same digit's four-bit binary representation. The network that we use is a four-layer perceptron, where each of the three hidden layers contains thirteen units. The output layer contains five units, where one unit is an indicator and is equal to zero if the input vector maps onto one of the sixteen hexadecimal digits and one otherwise. If the indicator unit equals zero, then the remaining four units of the output vector encode the hexadecimal digit's binary representation. The construction and training of the network is done in the file **seven-segment_display.c**.

## LeNet5 Architecture

As a simple example that uses the library for computer vision, in the file **LeNet5.c**, we construct ten input-output pairs, where each input matrix is a 32x32 image representing one of the ten decimal digits. We then train the model to classify each manually written image as one of the ten decimal digits. The file **LeNet5_omp.c** performs the same training with multithreaded parallelization using OpenMP.

## One-Unit Time Delay

As a simple example of a recurrent neural network, suppose that each input-output pair consists of a single input unit and a single output unit. We would like to train the recurrent network so that the output unit at time _t_ is equal to the input unit at time _t-1_. This is done in the file **bit_delay.c**.
