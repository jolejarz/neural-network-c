# Artificial Neural Network Library for C

This repository contains a library for building and training artificial neural networks (**annl.c**). The library is written from scratch in C and provides an application programming interface (API) for supervised machine learning using C. It is a work in progress.

# License and Copyright

All files in this repository are released under the [GNU General Public License, Version 3](https://www.gnu.org/licenses/gpl-3.0.en.html) as per the included [license](https://github.com/jolejarz/neural-network-c/blob/main/LICENSE.txt) and [copyright](https://github.com/jolejarz/neural-network-c/blob/main/COPYRIGHT.txt) files.

# Files

The library is contained in the following source files. The functions that are included in each file are briefly described.

* **activation.c**: Setting Activation Functions
* **bias.c**: Setting Bias Parameters
* **connection.c**: Connecting Layers and Setting Weight Parameters
* **gradient.c**: Calculating the Gradient of the Loss Function with Respect to the Weight and Bias Parameters
* **integration.c**: Performing Gradient Descent
* **layer.c**: Creating Layers
* **loss.c**: Calculating the Loss Function
* **output.c**: Calculating the Set of Outputs for a Given Set of Inputs
* **randomization.c**: Randomizing the Weight and Bias Parameters Before Training Begins
* **sequence.c**: Setting the Execution Sequence for Calculating the Outputs

# Functions

The API consists of the following functions.

## Activation Functions

```
void annlActivateLogistic (annlLayer *layer_current, int derivative)
```

This is the logistic activation function. `annlLayer *layer_current` is the layer that the activation is applied to. For calculating outputs, `int derivative` is set to NO_DERIVATIVE, and the activation function is returned. For performing backpropagation, `int derivative` is set to DERIVATIVE, and the derivative of the activation function with respect to its argument is returned.

```
void annlActivateReLU (annlLayer *layer_current, int derivative)
```

This is the rectified linear unit activation function. `annlLayer *layer_current` is the layer that the activation is applied to. For calculating outputs, `int derivative` is set to NO_DERIVATIVE, and the activation function is returned. For performing backpropagation, `int derivative` is set to DERIVATIVE, and the derivative of the activation function with respect to its argument is returned.

```
void annlActivateSoftmax (annlLayer *layer_current, int derivative)
```

This is the softmax activation function. `annlLayer *layer_current` is the layer that the activation is applied to. For calculating outputs, `int derivative` is set to NO_DERIVATIVE, and the activation function is returned. For performing backpropagation, `int derivative` is set to DERIVATIVE, and the derivative of the activation function with respect to its argument is returned.

```
void annlActivateTanh (annlLayer *layer_current, int derivative)
```

This is the hyperbolic tangent activation function. `annlLayer *layer_current` is the layer that the activation is applied to. For calculating outputs, `int derivative` is set to NO_DERIVATIVE, and the activation function is returned. For performing backpropagation, `int derivative` is set to DERIVATIVE, and the derivative of the activation function with respect to its argument is returned.

```
int annlHeavisideTheta (double x)
```

This is the Heaviside step function. It returns 1 if `double x` is greater than 0, and it returns 0 otherwise.

## Bias Functions

```
void annlSetBiasFull (annlLayer *layer_current, int train)
```

This function sets up the bias parameters for the fully connected layer `annlLayer *layer_current`. `int train` is set to TRAIN_BASIC for basic gradient descent or TRAIN_ADAM for the Adam optimizer.

```
void annlSetBiasFullExisting (annlLayer *layer_current, double *b, double *db)
```

This function sets up the bias parameters for the fully connected layer `annlLayer *layer_current`. It is used for building recurrent networks, where, after the network is unfolded in time, `annlLayer *layer_current` represents a subsequent instance of a layer that was previously set up. `double b` and `double db` are the vectors of bias parameters and their gradients, respectively, from the corresponding existing layer.

```
void annlSetBiasFullExisting_b (annlLayer *layer_current, double *b)
```

This function sets up the bias parameters for the fully connected layer `annlLayer *layer_current`. It is used for multithreaded parallelization. `double b` is the vector of bias parameters from the corresponding existing layer in an execution sequence that was previously set up.

```
void annlSetBiasConvolution (annlLayer *layer_current, int L, int n, int train)
```

This function sets up the bias parameters for the convolutional layer `annlLayer *layer_current`. `int L` is the linear size of the square grid of units, and `int n` is the number of feature maps. `int train` is set to TRAIN_BASIC for basic gradient descent or TRAIN_ADAM for the Adam optimizer.

```
void annlSetBiasConvolutionExisting_b (annlLayer *layer_current, int L, int n, double *b)
```

This function sets up the bias parameters for the convolutional layer `annlLayer *layer_current`. It is used for multithreaded parallelization. `double b` is the vector of bias parameters from the corresponding existing layer in an execution sequence that was previously set up. `int L` is the linear size of the square grid of units, and `int n` is the number of feature maps.

## Layer Functions

```
annlLayer* annlCreateLayer (int size, int num_layer_w, void (*activation)(annlLayer*,int))
```

This function creates a new layer and returns a pointer to the corresponding layer structure. `int size` is the number of units in the layer, `int num_layer_w` is the number of layers that the new layer will be connected to via weight parameters, and `void (*activation)(annlLayer*,int)` is a pointer to an activation function.

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
