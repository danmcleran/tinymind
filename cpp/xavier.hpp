/**
* Copyright (c) 2025 Dan McLeran
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#pragma once

#include <cmath>

namespace tinymind {

enum layer_e
{
    INVALID = 0,
    INPUT_LAYER,
    HIDDEN_LAYER,
    OUTPUT_LAYER
};

/**
 * The XavierWeightInitializer class implements the Xavier weight initialization algorithm.
 * It generates weights for neural network connections based on the number of inputs and outputs
 * of each neuron, ensuring that the weights are initialized in a way that helps maintain
 * the variance of activations across layers.
 * 
 * This is very tied to the neural network initializtion order, so be careful if changing that.
 * It was done this way to minimize the touch to existing code.
 */
template<
            size_t NumberOfInputs,
            size_t NumberOfHiddenLayers,
            size_t NumberOfNeuronsInHiddenLayers,
            size_t NumberOfOutputs>
struct XavierWeightInitializer
{
private:
    static const unsigned NumberOfNeurons = (NumberOfInputs + (NumberOfHiddenLayers * NumberOfNeuronsInHiddenLayers) + NumberOfOutputs);
    static const unsigned FirstHiddenNeuron = NumberOfInputs;
    static const unsigned FirstOuputNeuron = (NumberOfInputs + (NumberOfHiddenLayers * NumberOfNeuronsInHiddenLayers));

    unsigned neuron;
    layer_e previousLayer;
    layer_e currentLayer;
    layer_e nextLayer;
    unsigned numInputs;
    unsigned numOutputs;

    void advanceNeuron()
    {
        ++neuron;
        if (neuron >= NumberOfNeurons)
        {
            // reset for next call
            neuron = 0;
            previousLayer = layer_e::INVALID;
            currentLayer = layer_e::INPUT_LAYER;
            nextLayer = layer_e::HIDDEN_LAYER;
            numInputs = NumberOfInputs;
            numOutputs = NumberOfNeuronsInHiddenLayers;
        }
        else
        {
            if (neuron >= FirstOuputNeuron)
            {
                currentLayer = layer_e::OUTPUT_LAYER;
                previousLayer = layer_e::HIDDEN_LAYER;
                nextLayer = layer_e::INVALID;
            }
            else
            {
                if ((neuron >= FirstHiddenNeuron) && (neuron < FirstOuputNeuron))
                {
                    currentLayer = layer_e::HIDDEN_LAYER;
                    
                    if (neuron < (NumberOfInputs + NumberOfNeuronsInHiddenLayers))
                    {
                        previousLayer = layer_e::INPUT_LAYER;
                    }
                    else
                    {
                        previousLayer = layer_e::HIDDEN_LAYER;
                    }

                    if (neuron + NumberOfNeuronsInHiddenLayers >= FirstOuputNeuron)
                    {
                        nextLayer = layer_e::OUTPUT_LAYER;
                    }
                    else
                    {
                        nextLayer = layer_e::HIDDEN_LAYER;
                    }
                }
            }
        }
    }

    void calculateInputsAndOutputs()
    {
        if (currentLayer == layer_e::INPUT_LAYER)
        {
            numInputs = NumberOfInputs;
            numOutputs = NumberOfNeuronsInHiddenLayers;
        }
        else if (currentLayer == layer_e::HIDDEN_LAYER)
        {
            if (previousLayer == layer_e::INPUT_LAYER)
            {
                numInputs = NumberOfInputs;
            }
            else
            {
                numInputs = NumberOfNeuronsInHiddenLayers;
            }

            if (nextLayer == layer_e::OUTPUT_LAYER)
            {
                numOutputs = NumberOfOutputs;
            }
            else
            {
                numOutputs = NumberOfNeuronsInHiddenLayers;
            }
        }
        else
        {
            numInputs = NumberOfNeuronsInHiddenLayers;
            numOutputs = NumberOfOutputs;
        }
    }

public:
    XavierWeightInitializer() : neuron(0),
                                 previousLayer(layer_e::INVALID),
                                 currentLayer(layer_e::INPUT_LAYER),
                                 nextLayer(layer_e::HIDDEN_LAYER),
                                 numInputs(0),
                                 numOutputs(0)
    {
    }

    double generateUniformWeight()
    {
        calculateInputsAndOutputs();

        const double limit = std::sqrt(6.0 / (static_cast<double>(numInputs + numOutputs)));
        const double randomValue = ((static_cast<double>(rand()) / RAND_MAX) * 2.0 * limit) - limit;

        advanceNeuron();

        return randomValue;
    }

    double generateNormalWeight()
    {
        calculateInputsAndOutputs();

        const double limit = std::sqrt(2.0 / (static_cast<double>(numInputs + numOutputs)));
        const double randomValue = ((static_cast<double>(rand()) / RAND_MAX) * 2.0 * limit) - limit;

        advanceNeuron();

        return randomValue;
    }
};
}