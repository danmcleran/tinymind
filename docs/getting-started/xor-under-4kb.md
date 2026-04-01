---
title: Neural Network in Under 4KB
layout: default
parent: Getting Started
nav_order: 1
---

# A Neural Network in Under 4KB

To demonstrate what a tinymind neural network can do, I chose to implement a classic "hello world" kind of program for neural networks ([xor.cpp](https://github.com/danmcleran/tinymind/blob/master/examples/xor/xor.cpp)). Here I will demonstrate how to instantiate and train a neural network to predict the outcome of the mathematical XOR function. We will generate and feed known data into the neural network and train it to become an XOR predictor. By inspecting the generated output file during compilation, we can prove to ourselves that the whole neural network and associated code and data occupies a bit under 4KB in the final image, assuming we turn on compiler optimizations.

## What 4KB Means for Embedded

On a microcontroller like the STM32L0 (ARM Cortex-M0+) with 16 KB of flash and 2 KB of RAM, a 4KB neural network consumes 25% of flash -- leaving room for the application, peripherals, and communication stack. With training disabled (`IsTrainable=false`), the data footprint drops further: a non-trainable Q8.8 MLP (2->3->1) takes just 144 bytes of RAM.

For comparison, the minimum footprint of TensorFlow Lite Micro is ~50-100 KB -- it won't fit on this device at all, let alone leave room for anything else. Tinymind makes neural networks feasible on hardware where no other framework can run.

# Q Format

To make the neural network small and fast, we can't afford the luxury of floating point numbers. Also, when I was developing tinymind the hardware I had didn't have a floating point unit, which makes using floating point even more difficult and expensive. We need to use fixed-point numbers to solve this problem. [Tinymind](https://github.com/danmcleran/tinymind) contains a C++ template library to help us here, Q Format. See the [Q-Format]({{ site.baseurl }}/q-format) page for a full explanation of how it works.

We declare the size and resolution of our Q-Format type like this:

```cpp
// Q-Format value type
static const size_t NUMBER_OF_FIXED_BITS = 8;
static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
```

We have now declared a signed (the template parameter true means it is signed) fixed-point type which can represent the values -128 to 127.99609375. Our fractional part resolution is 2^(-8) or 0.00390625, since we are using 8 bits as the fractional part of our Q-format type. You'll see from this example that we'll have plenty enough dynamic range and resolution to solve this problem without floating point numbers. The fixed and fractional portion of the value are concatenated into a 16-bit field. As an example, the floating point number 1.5 will be represented as 0x180 in the q-format chosen above. This is because the integer portion (1) is shifted up by the number of fractional bits (8), resulting in 0x100. We then OR in the number which represents half of the dynamic range for 8 bits (128 or 0x80 in hex).

# Neural Network

We need to specify our neural network architecture. We do it as follows:

```cpp
// Neural network architecture
static const size_t NUMBER_OF_INPUTS = 2;
static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
static const size_t NUMBER_OF_OUTPUTS = 1;
```

We'll define a neural network with 2 input neurons, 1 hidden layer with 3 neurons in it, and 1 output neuron. The 2 input neurons will receive the streams of 1s and 0s from the training data. The output neuron outputs the predicted result of the operation. The hidden layer is the transfer function which learns to predict the result, given the input values and the known result.

![image](https://user-images.githubusercontent.com/1591721/215128190-f54e5c6a-121e-46f7-91f3-21c66a79098a.png)

We typedef the neural network transfer functions policy as well as the neural network itself:

```cpp
// Typedef of transfer functions for the fixed-point neural network
typedef tinymind::FixedPointTransferFunctions<  ValueType,
RandomNumberGenerator,
tinymind::TanhActivationPolicy<ValueType>,
tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
// typedef the neural network itself
typedef tinymind::MultilayerPerceptron< ValueType,
NUMBER_OF_INPUTS,
NUMBER_OF_HIDDEN_LAYERS,
NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
NUMBER_OF_OUTPUTS,
TransferFunctionsType> NeuralNetworkType;
```

The TransferFunctionsType specifies the ValueType (our Q-format type), the random number generator for the neural network, the activation policy for Input->Hidden layers, as well as the activation policy for the Hidden->Output layer. The neural network code within tinymind makes no assumptions about its environment. It needs to be provided with template parameters which encapsulate policy.

# Lookup Tables

Since the hardware I was using didn't have a floating point unit, I also did not have the luxury of the built in math functions for tanh, sigmoid, etc. To get around this limitation, tinymind generates and utilizes lookup tables (LUTs) for these activation functions ([activationTableGenerator.cpp](https://github.com/danmcleran/tinymind/blob/master/apps/activation/activationTableGenerator.cpp)). This generator generates header files for every supported activation function and Q-Format resolution. Because we don't want to generate code we're not going to use, we need to define preprocessor symbols to compile in the LUT(s) we need. If you look at the compile command you'll see the following:

```
-DTINYMIND_USE_TANH_8_8=1
```

This instructs the compiler to use the tanh activation LUT for the Q-Format type Q8.8, which is what we're using in this example. We need to do this for every LUT we plan to use.

# Generating Training Data

We'll generate the training data programmatically. The function generateXorTrainingValue is called to generate a single training sample. We use the built-in random number generator to generate the 2 inputs:

```cpp
static void generateXorTrainingValue(ValueType& x, ValueType& y, ValueType& z)
{
    const int randX = rand() & 0x1;
    const int randY = rand() & 0x1;
    const int result = (randX ^ randY);
    x = ValueType(randX, 0);
    y = ValueType(randY, 0);
    z = ValueType(result, 0);
}
```

We feed the data into the neural network, predict an output, and then train if we have an error outside of our zero-tolerance:

```cpp
for (unsigned i = 0; i < TRAINING_ITERATIONS; ++i)
{
    generateXorTrainingValue(values[0], values[1], output[0]);
    testNeuralNet.feedForward(&values[0]);
    error = testNeuralNet.calculateError(&output[0]);
    if (!NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
    {
        testNeuralNet.trainNetwork(&output[0]);
    }
    testNeuralNet.getLearnedValues(&learnedValues[0]);
```

# Compiling The Example

To compile the example, cd to the example dir and type:

```bash
make
```

To make the optimized target:

```bash
make release
```

To clean out old outputs and force re-compile and link:

```bash
make clean
```

# Running the Program

You can run the program by invoking:

```bash
make run
```

# Plotting Results

After running the example, you will see a text file output, nn_fixed_xor.txt. To plot the results:

```bash
make plot
```

![fixed_8_8](https://user-images.githubusercontent.com/1591721/200387686-75043947-8bbe-4162-b889-e9bcf368408a.png)

You can see the weights training through time, as well as the error reducing through time, as the predicted value approximates the expected value. Notice the magnitude of the numbers. Expected values are varying between 0 and 256. This maps to the representation of 0 and 1 for our Q-Format type. The representation of 1 in a signed Q8.8 format is 0x100.

# Determining Neural Network Size

To determine how much space the neural network code and data occupies:

```bash
make release
size output/lookupTables.o
```

```
text   data    bss    dec    hex filename
 224      0      0    224     e0 output/lookupTables.o
```

```bash
size output/xornet.o
```

```
text   data    bss    dec    hex filename
3272      8    388   3668    e54 output/xornet.o
```

The total space occupied by this neural network, Q-format support, and LUTs was just under 4KB. The neural network itself occupies 3,668 bytes. The LUT occupies 224 bytes for a grand total of **3,892 bytes**.

# Conclusion

Using tinymind, neural networks can be instantiated using fixed-point numbers so that they take up a very small amount of code and data space. I hope that this inspires you to clone tinymind and give the unit tests and examples a look. Maybe even generate one of your own.
