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

This function sets up the bias parameters for the fully connected layer `annlLayer *layer_current`. It is used for building recurrent networks, where, after the network is unfolded in time, `annlLayer *layer_current` represents a subsequent instance of a layer that was previously set up. `double *b` and `double *db` are the vectors of bias parameters and their gradients, respectively, from the corresponding existing layer.

```
void annlSetBiasFullExisting_b (annlLayer *layer_current, double *b)
```

This function sets up the bias parameters for the fully connected layer `annlLayer *layer_current`. It is used for multithreaded parallelization. `double *b` is the vector of bias parameters from the corresponding existing layer in an execution sequence that was previously set up.

```
void annlSetBiasConvolution (annlLayer *layer_current, int L, int n, int train)
```

This function sets up the bias parameters for the convolutional layer `annlLayer *layer_current`. `int L` is the linear size of the square grid of units, and `int n` is the number of feature maps. `int train` is set to TRAIN_BASIC for basic gradient descent or TRAIN_ADAM for the Adam optimizer.

```
void annlSetBiasConvolutionExisting_b (annlLayer *layer_current, int L, int n, double *b)
```

This function sets up the bias parameters for the convolutional layer `annlLayer *layer_current`. It is used for multithreaded parallelization. `double *b` is the vector of bias parameters from the corresponding existing layer in an execution sequence that was previously set up. `int L` is the linear size of the square grid of units, and `int n` is the number of feature maps.

## Connection Functions

```
void annlConnectFull (annlLayer *layer_previous, annlLayer *layer_current, int train)
```

This function sets up `annlLayer *layer_current` as a fully connected layer. It is connected to layer `annlLayer *layer_previous` with weight parameters. `int train` is set to TRAIN_BASIC for basic gradient descent or TRAIN_ADAM for the Adam optimizer.

```
void annlConnectFullExisting (annlLayer *layer_previous, annlLayer *layer_current, double *w, double *dw)
```

This function sets up `annlLayer *layer_current` as a fully connected layer. It is connected to layer `annlLayer *layer_previous` with weight parameters. This function is used for building recurrent networks, where, after the network is unfolded in time, `annlLayer *layer_current` represents a subsequent instance of layer `annlLayer *layer_previous`. `double *w` and `double *dw` are the corresponding vectors of weight parameters and their gradients, respectively, from layer `annlLayer *layer_previous`.

```
void annlConnectFullExisting_w (annlLayer *layer_previous, annlLayer *layer_current, double *w)
```

This function sets up `annlLayer *layer_current` as a fully connected layer. It is connected to layer `annlLayer *layer_previous` with weight parameters. This function is used for multithreaded parallelization. `double *w` is the vector of weight parameters from the corresponding existing layer in an execution sequence that was previously set up.

```
void annlConnectConvolution (annlLayer *layer_previous, annlLayer *layer_current, int L, int n, int (*a)[][2], int train)
```

This function sets up `annlLayer *layer_current` as a convolutional layer. It is connected to layer `annlLayer *layer_previous` with weight parameters. `int L` is the linear size of the square grid of units, and `int n` is the number of connections between feature maps in `annlLayer *layer_current` and the units in each feature map of `annlLayer *layer_previous`. `int (*a)[][2]` specifies how the feature maps in `annlLayer *layer_current` are connected to the units in each feature map in `annlLayer *layer_previous`. `int train` is set to TRAIN_BASIC for basic gradient descent or TRAIN_ADAM for the Adam optimizer.

```
void annlConnectConvolutionExisting_w (annlLayer *layer_previous, annlLayer *layer_current, int L, int n, int (*a)[][2], double *w)
```

This function sets up `annlLayer *layer_current` as a convolutional layer. It is connected to layer `annlLayer *layer_previous` with weight parameters. This function is used for multithreaded parallelization. `int L` is the linear size of the square grid of units, and `int n` is the number of connections between feature maps in `annlLayer *layer_current` and the units in each feature map of `annlLayer *layer_previous`. `int (*a)[][2]` specifies how the feature maps in `annlLayer *layer_current` are connected to the units in each feature map in `annlLayer *layer_previous`. `double *w` is the vector of weight parameters from the corresponding existing layer in an execution sequence that was previously set up.

```
void annlConnectPool (annlLayer *layer_previous, annlLayer *layer_current, int L, int n, int train)
```

This function sets up `annlLayer *layer_current` as a pooling layer. It is connected to layer `annlLayer *layer_previous` with weight parameters. `int L` is the linear size of the square grid of units, and `int n` is the number of feature maps. `int train` is set to TRAIN_BASIC for basic gradient descent or TRAIN_ADAM for the Adam optimizer.

```
void annlConnectPoolExisting_w (annlLayer *layer_previous, annlLayer *layer_current, int L, int n, double *w)
```

This function sets up `annlLayer *layer_current` as a pooling layer. It is connected to layer `annlLayer *layer_previous` with weight parameters. This function is used for multithreaded parallelization. `int L` is the linear size of the square grid of units, and `int n` is the number of feature maps. `double *w` is the vector of weight parameters from the corresponding existing layer in an execution sequence that was previously set up.

## Gradient Functions

```
void annlCalculateGradient (annlSequence sequence)
```

This function calculates the gradient of the loss function. `annlSequence sequence` is a structure specifying all execution sequences.

```
void annlCalculateGradient_omp (annlSequence sequence)
```

This function calculates the gradient of the loss function. It is used for multithreaded parallelization. `annlSequence sequence` is a structure specifying all execution sequences.

```
void annlCalcFull_db (annlLayer *layer_current)
```

**(for internal use)** This function calculates the derivative of the loss function with respect to each bias parameter in the fully connected layer `annlLayer *layer_current`.

```
void annlCalcFull_dw (annlLayer *layer_current, int layer_w_index)
```

**(for internal use)** This function calculates the derivative of the loss function with respect to each weight parameter connecting the fully connected layer `annlLayer *layer_current` to the layer specified by `int layer_w_index`.

```
void annlCalcFull_dxj (annlLayer *layer_current, int layer_w_index)
```

**(for internal use)** For each unit in the layer specified by `int layer_w_index`, this function calculates a portion of the derivative of the loss function with respect to that unit. The portion that is calculated comes from that unit's connections to the fully connected layer `annlLayer *layer_current`.

```
void annlCalcConvolution_db (annlLayer *layer_current)
```

**(for internal use)** This function calculates the derivative of the loss function with respect to each bias parameter in the convolutional layer `annlLayer *layer_current`.

```
void annlCalcConvolution_dw (annlLayer *layer_current, int layer_w_index)
```

**(for internal use)** This function calculates the derivative of the loss function with respect to each weight parameter connecting the convolutional layer `annlLayer *layer_current` to the layer specified by `int layer_w_index`.

```
void annlCalcConvolution_dxj (annlLayer *layer_current, int layer_w_index)
```

**(for internal use)** For each unit in the layer specified by `int layer_w_index`, this function calculates a portion of the derivative of the loss function with respect to that unit. The portion that is calculated comes from that unit's connections to the convolutional layer `annlLayer *layer_current`.

## Integration Functions

```
void annlUpdateParameters (annlLayer *layer_input, double step)
```

This function updates the weight and bias parameters after the gradient of the loss function is calculated. `annlLayer *layer_input` is the starting layer in the execution sequence, and `double step` is the step size.

```
void annlUpdateParameters_omp (annlSequence sequence, double step)
```

This function updates the weight and bias parameters after the gradient of the loss function is calculated. It is used for multithreaded parallelization. `annlSequence sequence` is a structure specifying all execution sequences, and `double step` is the step size.

```
void annlIntegrateFull_db (annlLayer *layer_current, double step)
```

**(for internal use)** This function updates the bias parameters for the fully connected layer `annlLayer *layer_current`. `double step` is the step size.

```
void annlIntegrateFull_db_Adam (annlLayer *layer_current, double step)
```

**(for internal use)** This function updates the bias parameters for the fully connected layer `annlLayer *layer_current` using the Adam optimizer. `double step` is the step size.

```
void annlIntegrateConvolution_db (annlLayer *layer_current, double step)
```

**(for internal use)** This function updates the bias parameters for the convolutional layer `annlLayer *layer_current`. `double step` is the step size.

```
void annlIntegrateConvolution_db_Adam (annlLayer *layer_current, double step)
```

**(for internal use)** This function updates the bias parameters for the convolutional layer `annlLayer *layer_current` using the Adam optimizer. `double step` is the step size.

```
void annlIntegrateFull_dw (annlLayer *layer_current, int layer_w_index, double step)
```

**(for internal use)** This function updates the weight parameters for the fully connected layer `annlLayer *layer_current`. `double step` is the step size.

```
void annlIntegrateFull_dw_Adam (annlLayer *layer_current, int layer_w_index, double step)
```

**(for internal use)** This function updates the weight parameters for the fully connected layer `annlLayer *layer_current` using the Adam optimizer. `double step` is the step size.

```
void annlIntegrateConvolution_dw (annlLayer *layer_current, int layer_w_index, double step)
```

**(for internal use)** This function updates the weight parameters for the convolutional layer `annlLayer *layer_current`. `double step` is the step size.

```
void annlIntegrateConvolution_dw_Adam (annlLayer *layer_current, int layer_w_index, double step)
```

**(for internal use)** This function updates the weight parameters for the convolutional layer `annlLayer *layer_current` using the Adam optimizer. `double step` is the step size.

## Layer Functions

```
annlLayer* annlCreateLayer (int size, int num_layer_w, void (*activation)(annlLayer*,int))
```

This function creates a new layer and returns a pointer to the corresponding layer structure. `int size` is the number of units in the layer, `int num_layer_w` is the number of layers that the new layer will be connected to via weight parameters, and `void (*activation)(annlLayer*,int)` is an activation function.

## Loss Functions

```
double annlCalculateLoss (int output_size, double *output, double *output_target, double *output_target_fit, int derivative, int derivative_index)
```

This is the squared error loss function. `int output_size` is the number of units in the output layer, `double *output` is the output layer, and `double *output_target` is the target output layer. A particular unit in the output layer is included in the calculation of loss if and only if the corresponding element in `double *output_target_fit` is set to 1. For calculating outputs, `int derivative` is set to NO_DERIVATIVE, and the loss function is returned. For performing backpropagation, `int derivative` is set to DERIVATIVE, and the derivative of the loss function with respect to the output unit specified by `int derivative_index` is returned.

```
double annlCalculateLossTotal (annlSequence sequence)
```

This function calculates the total loss from all execution sequences specified in `annlSequence sequence`.

```
double annlCalculateLossTotal_omp (annlSequence sequence)
```

This function calculates the total loss from all execution sequences specified in `annlSequence sequence`. It is used for multithreaded parallelization.

## Output Functions

```
annlLayer* annlCalculateOutput (annlLayer *layer_input)
```

This function calculates the output of the network, beginning at layer `annlLayer *layer_input` and following the execution sequence until the end. It returns a pointer to the final layer in the execution sequence.

```
void annlCalcFull_z_w (annlLayer *layer_current, int layer_w_index)
```

**(for internal use)** This function computes the weighted sum of the units in the layer specified by `int layer_w_index` when calculating the outputs for the fully connected layer `annlLayer *layer_current`.

```
void annlCalcConvolution_z_w (annlLayer *layer_current, int layer_w_index)
```

**(for internal use)** This function computes the weighted sum of the units in the layer specified by `int layer_w_index` when calculating the outputs for the convolutional layer `annlLayer *layer_current`.

```
void annlCalcFull_z_b (annlLayer *layer_current)
```

**(for internal use)** This function adds the bias parameters when calculating the outputs for the fully connected layer `annlLayer *layer_current`.

```
void annlCalcConvolution_z_b (annlLayer *layer_current)
```

**(for internal use)** This function adds the bias parameters when calculating the outputs for the convolutional layer `annlLayer *layer_current`.

## Randomization Functions

```
void annlRandomizeParameters (annlLayer *layer_current, gsl_rng *rng)
```

This function randomizes the weight and bias parameters independently according to a uniform distribution between -1 and 1. `annlLayer *layer_current` is the fully connected layer to be initialized, and `gsl_rng *rng` is a GSL random number generator structure.

```
void annlRandomizeParametersConvolution (annlLayer *layer_current, gsl_rng *rng)
```

This function randomizes the weight and bias parameters independently according to a uniform distribution between -1 and 1. `annlLayer *layer_current` is the convolutional layer to be initialized, and `gsl_rng *rng` is a GSL random number generator structure.

## Sequence Functions

```
void annlLinkSequence (annlLayer *layer_previous, annlLayer *layer_next)
```

This function connects two layers and is used for setting the execution sequence. Once the outputs in layer `annlLayer *layer_previous` have been calculated, the execution proceeds to layer `annlLayer *layer_next`.

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
