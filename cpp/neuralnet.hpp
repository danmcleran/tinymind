/**
* Copyright (c) 2020 Intel Corporation
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

#include <cstddef>

#include "activationFunctions.hpp"
#include "gradientClipping.hpp"
#include "weightDecay.hpp"
#include "learningRateSchedule.hpp"

namespace tinymind {
    namespace detail {
        template<typename...>
        struct make_void { typedef void type; };

        template<typename... Ts>
        using void_t = typename make_void<Ts...>::type;

        template<typename T, typename = void>
        struct gradient_clipping_policy_of
        {
            typedef NullGradientClippingPolicy<typename T::TransferFunctionsValueType> type;
        };

        template<typename T>
        struct gradient_clipping_policy_of<T, void_t<typename T::GradientClippingPolicyType> >
        {
            typedef typename T::GradientClippingPolicyType type;
        };

        template<typename T, typename = void>
        struct weight_decay_policy_of
        {
            typedef NullWeightDecayPolicy<typename T::TransferFunctionsValueType> type;
        };

        template<typename T>
        struct weight_decay_policy_of<T, void_t<typename T::WeightDecayPolicyType> >
        {
            typedef typename T::WeightDecayPolicyType type;
        };

        template<typename T, typename = void>
        struct learning_rate_schedule_policy_of
        {
            typedef FixedLearningRatePolicy<typename T::TransferFunctionsValueType> type;
        };

        template<typename T>
        struct learning_rate_schedule_policy_of<T, void_t<typename T::LearningRateSchedulePolicyType> >
        {
            typedef typename T::LearningRateSchedulePolicyType type;
        };
    } // namespace detail

    typedef enum
    {
        FeedForwardOutputLayerConfiguration,
        ClassifierOutputLayerConfiguration,
    } outputLayerConfiguration_e;

    typedef enum
    {
        NonRecurrentHiddenLayerConfig,
        RecurrentHiddenLayerConfig,
        GRUHiddenLayerConfig,
        LSTMHiddenLayerConfig,
    } hiddenLayerConfiguration_e;

    static const size_t GRU_NUMBER_OF_GATES = 3;
    static const size_t LSTM_NUMBER_OF_GATES = 4;

    template<size_t LayerSize, hiddenLayerConfiguration_e Config>
    struct GateConnectionCount
    {
        static const size_t value = LayerSize;
    };

    template<size_t LayerSize>
    struct GateConnectionCount<LayerSize, GRUHiddenLayerConfig>
    {
        static const size_t value = LayerSize * GRU_NUMBER_OF_GATES;
    };

    template<size_t LayerSize>
    struct GateConnectionCount<LayerSize, LSTMHiddenLayerConfig>
    {
        static const size_t value = LayerSize * LSTM_NUMBER_OF_GATES;
    };

    template<typename ValueType, size_t NumberOfGradients, size_t BatchSize>
    struct GradientsHolder
    {
        static_assert(BatchSize > 0, "Invalid batch size.");

        GradientsHolder() : mIndex(0), mPass(0)
        {
            size_t bufferIndex;
            for(size_t gradient = 0;gradient < NumberOfGradients;++gradient)
            {
                bufferIndex = gradient * sizeof(ValueType);
                new (&this->mGradientsBuffer[bufferIndex]) ValueType();
            }

            this->resetGradients();
        }

        template<typename LayerType>
        void updateBiasGradients(LayerType& layer, const size_t nextNeuron, const ValueType& gradient)
        {
            ValueType averageGradient;
            ValueType* pValue;
            size_t bufferIndex;

            if(BatchSize == this->mPass)
            {
                bufferIndex = this->mIndex * sizeof(ValueType);
                pValue = reinterpret_cast<ValueType*>(&this->mGradientsBuffer[bufferIndex]);
                averageGradient = (*pValue / BatchSize);

                layer.setBiasNeuronGradientForConnection(nextNeuron, gradient);

                *pValue = 0;
            }

            this->updateGradients(gradient);
        }

        template<typename LayerType>
        void updateGradients(LayerType& layer, const size_t neuron, const size_t nextNeuron, const ValueType& gradient)
        {
            ValueType averageGradient;
            ValueType* pValue;
            size_t bufferIndex;
            
            if(BatchSize == this->mPass)
            {
                bufferIndex = this->mIndex * sizeof(ValueType);
                pValue = reinterpret_cast<ValueType*>(&this->mGradientsBuffer[bufferIndex]);
                averageGradient = (*pValue / BatchSize);
                
                layer.setGradientForNeuronAndConnection(neuron, nextNeuron, averageGradient);

                *pValue = 0;
            }

            this->updateGradients(gradient);
        }
    private:
        void resetGradients()
        {
            ValueType* pValue;
            size_t bufferIndex;
            for(size_t gradient = 0;gradient < NumberOfGradients;++gradient)
            {
                bufferIndex = gradient * sizeof(ValueType);
                pValue = reinterpret_cast<ValueType*>(&this->mGradientsBuffer[bufferIndex]);
                *pValue = 0;
            }
        }

        void updateGradients(const ValueType& gradient)
        {
            ValueType oldGradient;
            ValueType* pValue;
            size_t bufferIndex;

            bufferIndex = this->mIndex * sizeof(ValueType);
            pValue = reinterpret_cast<ValueType*>(&this->mGradientsBuffer[bufferIndex]);
            oldGradient = *pValue;
            *pValue = (oldGradient + gradient);
            ++this->mIndex;
            if(NumberOfGradients == this->mIndex)
            {
                this->mIndex = 0;
                ++this->mPass;
                if(this->mPass > BatchSize)
                {
                    this->mPass = 1;
                }
            }
        }

        size_t mIndex;
        size_t mPass;
        unsigned char mGradientsBuffer[NumberOfGradients * sizeof(ValueType)];
    };

    template<typename ValueType, size_t NumberOfGradients>
    struct GradientsHolder<ValueType, NumberOfGradients, 1>
    {
        template<typename LayerType>
        void updateBiasGradients(LayerType& layer, const size_t nextNeuron, const ValueType& gradient)
        {
            layer.setBiasNeuronGradientForConnection(nextNeuron, gradient);
        }

        template<typename LayerType>
        void updateGradients(LayerType& layer, const size_t neuron, const size_t nextNeuron, const ValueType& gradient)
        {
            layer.setGradientForNeuronAndConnection(neuron, nextNeuron, gradient);
        }
    };

    template<typename NeuralNetworkType, size_t NumberOfInnerHiddenLayers>
    struct NullGradientsManager
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;

        template<typename LayerType>
        void updateBiasGradients(LayerType& layer, const size_t nextNeuron, const ValueType& gradient)
        {
        }

        template<typename LayerType>
        void updateGradients(LayerType& layer, const size_t neuron, const size_t nextNeuron, const ValueType& gradient)
        {
        }
    };

    template<typename NeuralNetworkType, size_t NumberOfInnerHiddenLayers>
    struct GradientsManager
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerType InnerHiddenLayerType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;

        static const size_t NumberOfInputLayerNeurons = InputLayerType::NumberOfNeuronsInLayer;
        static const size_t NumberOfInnerHiddenLayerNeurons = InnerHiddenLayerType::NumberOfNeuronsInLayer;
        static const size_t NumberOfLastHiddenLayerNeurons = LastHiddenLayerType::NumberOfNeuronsInLayer;
        static const size_t NumberOfOutputLayerNeurons = OutputLayerType::NumberOfNeuronsInLayer;

        static const size_t LastHiddenToOutputNumGradients = (NumberOfLastHiddenLayerNeurons * NumberOfOutputLayerNeurons) + NumberOfOutputLayerNeurons;
        static const size_t InnerHiddenToLastHiddenNumGradients = (NumberOfInnerHiddenLayerNeurons * NumberOfLastHiddenLayerNeurons) + NumberOfLastHiddenLayerNeurons;
        static const size_t InnerToInnerNumGradients = ((NumberOfInnerHiddenLayers - 1) * NumberOfInnerHiddenLayerNeurons * NumberOfInnerHiddenLayerNeurons) + NumberOfInnerHiddenLayerNeurons;
        static const size_t InputToHiddenNumGradients = (NumberOfInputLayerNeurons * NumberOfInnerHiddenLayerNeurons) + NumberOfInnerHiddenLayerNeurons;
        static const size_t NumberOfGradients = (LastHiddenToOutputNumGradients + InnerHiddenToLastHiddenNumGradients + InnerToInnerNumGradients + InputToHiddenNumGradients);

        template<typename LayerType>
        void updateBiasGradients(LayerType& layer, const size_t nextNeuron, const ValueType& gradient)
        {
            this->gradientsHolder.updateBiasGradients(layer, nextNeuron, gradient);
        }

        template<typename LayerType>
        void updateGradients(LayerType& layer, const size_t neuron, const size_t nextNeuron, const ValueType& gradient)
        {
            this->gradientsHolder.updateGradients(layer, neuron, nextNeuron, gradient);
        }
    private:
        GradientsHolder<ValueType, NumberOfGradients, NeuralNetworkType::NeuralNetworkBatchSize> gradientsHolder;
    };

    template<typename NeuralNetworkType>
    struct GradientsManager<NeuralNetworkType, 1>
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerType InnerHiddenLayerType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;

        static const size_t NumberOfInputLayerNeurons = InputLayerType::NumberOfNeuronsInLayer;
        static const size_t NumberOfInnerHiddenLayerNeurons = InnerHiddenLayerType::NumberOfNeuronsInLayer;
        static const size_t NumberOfLastHiddenLayerNeurons = LastHiddenLayerType::NumberOfNeuronsInLayer;
        static const size_t NumberOfOutputLayerNeurons = OutputLayerType::NumberOfNeuronsInLayer;

        static const size_t LastHiddenToOutputNumGradients = (NumberOfLastHiddenLayerNeurons * NumberOfOutputLayerNeurons) + NumberOfOutputLayerNeurons;
        static const size_t InnerHiddenToLastHiddenNumGradients = (NumberOfInnerHiddenLayerNeurons * NumberOfLastHiddenLayerNeurons) + NumberOfLastHiddenLayerNeurons;
        static const size_t InputToHiddenNumGradients = (NumberOfInputLayerNeurons * NumberOfInnerHiddenLayerNeurons) + NumberOfInnerHiddenLayerNeurons;        
        static const size_t NumberOfGradients = (LastHiddenToOutputNumGradients + InnerHiddenToLastHiddenNumGradients + InputToHiddenNumGradients);

        template<typename LayerType>
        void updateBiasGradients(LayerType& layer, const size_t nextNeuron, const ValueType& gradient)
        {
            this->gradientsHolder.updateBiasGradients(layer, nextNeuron, gradient);
        }

        template<typename LayerType>
        void updateGradients(LayerType& layer, const size_t neuron, const size_t nextNeuron, const ValueType& gradient)
        {
            this->gradientsHolder.updateGradients(layer, neuron, nextNeuron, gradient);
        }
    private:
        GradientsHolder<ValueType, NumberOfGradients, NeuralNetworkType::NeuralNetworkBatchSize> gradientsHolder;
    };

    template<typename NeuralNetworkType>
    struct GradientsManager<NeuralNetworkType, 0>
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        static const size_t NumberOfInputLayerNeurons = InputLayerType::NumberOfNeuronsInLayer;
        static const size_t NumberOfHiddenLayerNeurons = LastHiddenLayerType::NumberOfNeuronsInLayer;
        static const size_t NumberOfOutputLayerNeurons = OutputLayerType::NumberOfNeuronsInLayer;
        
        static const size_t HiddenToOutputNumGradients = (NumberOfHiddenLayerNeurons * NumberOfOutputLayerNeurons) + NumberOfOutputLayerNeurons;
        static const size_t InputToHiddenNumGradients = (NumberOfInputLayerNeurons * NumberOfHiddenLayerNeurons) + NumberOfHiddenLayerNeurons;
        static const size_t NumberOfGradients = HiddenToOutputNumGradients + InputToHiddenNumGradients;

        template<typename LayerType>
        void updateBiasGradients(LayerType& layer, const size_t nextNeuron, const ValueType& gradient)
        {
            this->gradientsHolder.updateBiasGradients(layer, nextNeuron, gradient);
        }

        template<typename LayerType>
        void updateGradients(LayerType& layer, const size_t neuron, const size_t nextNeuron, const ValueType& gradient)
        {
            this->gradientsHolder.updateGradients(layer, neuron, nextNeuron, gradient);
        }
    private:
        GradientsHolder<ValueType, NumberOfGradients, NeuralNetworkType::NeuralNetworkBatchSize> gradientsHolder;
    };

    template<typename NeuralNetworkType, size_t NumberOfInnerHiddenLayers, bool IsTrainable>
    struct GradientsManagerSelector
    {
    };

    template<typename NeuralNetworkType, size_t NumberOfInnerHiddenLayers>
    struct GradientsManagerSelector<NeuralNetworkType, NumberOfInnerHiddenLayers, true>
    {
        typedef GradientsManager<NeuralNetworkType, NumberOfInnerHiddenLayers> GradientsManagerType;
    };

    template<typename NeuralNetworkType, size_t NumberOfInnerHiddenLayers>
    struct GradientsManagerSelector<NeuralNetworkType, NumberOfInnerHiddenLayers, false>
    {
        typedef NullGradientsManager<NeuralNetworkType, NumberOfInnerHiddenLayers> GradientsManagerType;
    };

    template<typename NeuralNetworkType, size_t NumberOfInnerHiddenLayers>
    struct BackPropConnectionWeightUpdater
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerType InnerHiddenLayerType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;

        template<typename TrainingPolicyType>
        static void updateConnectionWeights(TrainingPolicyType& trainingPolicy, NeuralNetworkType& nn)
        {
            OutputLayerType& outputLayer = nn.getOutputLayer();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            InnerHiddenLayerType* pInnerHiddenLayers = nn.getPointerToInnerHiddenLayers();
            InputLayerType& inputLayer = nn.getInputLayer();

            trainingPolicy.updateConnectionWeights(outputLayer, lastHiddenLayer);

            trainingPolicy.updateConnectionWeights(lastHiddenLayer, pInnerHiddenLayers[NumberOfInnerHiddenLayers - 1]);

            for (int hiddenLayer = static_cast<int>(NumberOfInnerHiddenLayers) - 1; hiddenLayer > 0; --hiddenLayer)
            {
                trainingPolicy.updateConnectionWeights(pInnerHiddenLayers[hiddenLayer], pInnerHiddenLayers[hiddenLayer - 1]);
            }

            trainingPolicy.updateConnectionWeights(pInnerHiddenLayers[0], inputLayer);
        }

    private:
        static_assert(NeuralNetworkType::NeuralNetworkRecurrentConnectionDepth == 0, "Invalid use of BackPropConnectionWeightUpdater.");
    };

    template<typename NeuralNetworkType>
    struct BackPropConnectionWeightUpdater<NeuralNetworkType, 1>
    {
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerType InnerHiddenLayerType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;

        template<typename TrainingPolicyType>
        static void updateConnectionWeights(TrainingPolicyType& trainingPolicy, NeuralNetworkType& nn)
        {
            OutputLayerType& outputLayer = nn.getOutputLayer();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            InnerHiddenLayerType* pInnerHiddenLayers = nn.getPointerToInnerHiddenLayers();
            InputLayerType& inputLayer = nn.getInputLayer();

            trainingPolicy.updateConnectionWeights(outputLayer, lastHiddenLayer);

            trainingPolicy.updateConnectionWeights(lastHiddenLayer, pInnerHiddenLayers[0]);

            trainingPolicy.updateConnectionWeights(pInnerHiddenLayers[0], inputLayer);
        }

    private:
        static_assert(NeuralNetworkType::NeuralNetworkRecurrentConnectionDepth == 0, "Invalid use of BackPropConnectionWeightUpdater.");
    };

    template<typename NeuralNetworkType>
    struct BackPropConnectionWeightUpdater<NeuralNetworkType, 0>
    {
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerType InnerHiddenLayerType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;

        template<typename TrainingPolicyType>
        static void updateConnectionWeights(TrainingPolicyType& trainingPolicy, NeuralNetworkType& nn)
        {
            OutputLayerType& outputLayer = nn.getOutputLayer();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            InputLayerType& inputLayer = nn.getInputLayer();

            trainingPolicy.updateConnectionWeights(outputLayer, lastHiddenLayer);

            trainingPolicy.updateConnectionWeights(lastHiddenLayer, inputLayer);
        }

    private:
        static_assert(NeuralNetworkType::NeuralNetworkRecurrentConnectionDepth == 0, "Invalid use of BackPropConnectionWeightUpdater.");
    };

    template<typename NeuralNetworkType>
    struct BackPropThruTimeConnectionWeightUpdater
    {
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkRecurrentLayerType RecurrentLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerType InnerHiddenLayerType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;

        static const size_t GateMultiplier = NeuralNetworkType::HiddenLayerGateMultiplier;

        template<typename TrainingPolicyType>
        static void updateConnectionWeights(TrainingPolicyType& trainingPolicy, NeuralNetworkType& nn)
        {
            OutputLayerType& outputLayer = nn.getOutputLayer();
            RecurrentLayerType& recurrentLayer = nn.getRecurrentLayer();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            InputLayerType& inputLayer = nn.getInputLayer();

            trainingPolicy.updateConnectionWeights(outputLayer, lastHiddenLayer);

            trainingPolicy.template updateConnectionWeightsGated<LastHiddenLayerType, RecurrentLayerType, GateMultiplier>(lastHiddenLayer, recurrentLayer);

            trainingPolicy.template updateConnectionWeightsGated<LastHiddenLayerType, InputLayerType, GateMultiplier>(lastHiddenLayer, inputLayer);
        }

    private:
        static_assert(NeuralNetworkType::NeuralNetworkRecurrentConnectionDepth > 0, "Invalid use of BackPropThruTimeConnectionWeightUpdater.");
    };

    template<typename ValueType, size_t NumberOfBiasNeurons>
    struct BiasNeuronConnectionWeightUpdater
    {

    };

    template<typename ValueType>
    struct BiasNeuronConnectionWeightUpdater<ValueType, 1>
    {
        template<typename LayerType>
        static void updateBiasConnectionWeights(LayerType& previousLayer, const size_t neuron, const ValueType& learningRate, const ValueType& momentumRate, const ValueType& accelerationRate)
        {
            const ValueType previousDeltaWeight = previousLayer.getBiasNeuronPreviousDeltaWeightForConnection(neuron);
            const ValueType currentDeltaWeight = previousLayer.getBiasNeuronDeltaWeightForConnection(neuron);
            const ValueType newDeltaWeight = (learningRate * previousLayer.getBiasNeuronGradientForConnection(neuron)) + (momentumRate * currentDeltaWeight) + (accelerationRate * previousDeltaWeight);
            const ValueType currentWeight = previousLayer.getBiasNeuronWeightForConnection(neuron);

            previousLayer.setBiasNeuronDeltaWeightForConnection(neuron, newDeltaWeight);
            previousLayer.setBiasNeuronWeightForConnection(neuron, (currentWeight + newDeltaWeight));
        }
    };

    template<typename ValueType>
    struct BiasNeuronConnectionWeightUpdater<ValueType, 0>
    {
        template<typename LayerType>
        static void updateBiasConnectionWeights(LayerType& layer, const size_t neuron, const ValueType& learningRate, const ValueType& momentumRate, const ValueType& accelerationRate)
        {
            (void)layer; // Suppress unused parameter warning
            (void)neuron; // Suppress unused parameter warning
            (void)learningRate; // Suppress unused parameter warning
            (void)momentumRate; // Suppress unused parameter warning
            (void)accelerationRate; // Suppress unused parameter warning
        }
    };

    template<typename TransferFunctionsPolicy>
    struct NodeDeltasCalculator
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        template<typename LayerType, typename NextLayerType>
        static void calculateAndSetNodeDeltas(LayerType& layer, const NextLayerType& nextLayer)
        {
            ValueType sum;
            ValueType nodeDelta;

            for(size_t neuron = 0; neuron < LayerType::NumberOfNeuronsInLayer;++neuron)
            {
                sum = 0;
                for (size_t nextNeuron = 0; nextNeuron < NextLayerType::NumberOfNeuronsInLayer; ++nextNeuron)
                {
                    sum += (layer.getWeightForNeuronAndConnection(neuron, nextNeuron) * nextLayer.getNodeDeltaForNeuron(nextNeuron));
                }

                nodeDelta = sum * TransferFunctionsPolicy::hiddenNeuronActivationFunctionDerivative(layer.getOutputValueForNeuron(neuron));

                layer.setNodeDeltaForNeuron(neuron, nodeDelta);
            }
        }

        // Compute raw delta for LSTM hidden layer: just the weighted sum of output
        // deltas, WITHOUT multiplying by the activation derivative. The LSTM gate
        // derivatives are handled separately by computeGateDeltas().
        template<typename LayerType, typename NextLayerType>
        static void calculateAndSetRawNodeDeltas(LayerType& layer, const NextLayerType& nextLayer)
        {
            ValueType sum;

            for(size_t neuron = 0; neuron < LayerType::NumberOfNeuronsInLayer;++neuron)
            {
                sum = 0;
                for (size_t nextNeuron = 0; nextNeuron < NextLayerType::NumberOfNeuronsInLayer; ++nextNeuron)
                {
                    sum += (layer.getWeightForNeuronAndConnection(neuron, nextNeuron) * nextLayer.getNodeDeltaForNeuron(nextNeuron));
                }
                layer.setNodeDeltaForNeuron(neuron, sum);
            }
        }

        template<typename LayerType, typename NextLayerType, size_t GateMultiplier>
        static void calculateAndSetNodeDeltasGated(LayerType& layer, const NextLayerType& nextLayer)
        {
            ValueType sum;
            ValueType nodeDelta;

            for(size_t neuron = 0; neuron < LayerType::NumberOfNeuronsInLayer;++neuron)
            {
                sum = 0;
                for (size_t nextNeuron = 0; nextNeuron < NextLayerType::NumberOfNeuronsInLayer; ++nextNeuron)
                {
                    const ValueType nextDelta = nextLayer.getNodeDeltaForNeuron(nextNeuron);
                    for (size_t gate = 0; gate < GateMultiplier; ++gate)
                    {
                        sum += (layer.getWeightForNeuronAndConnection(neuron, nextNeuron * GateMultiplier + gate) * nextDelta);
                    }
                }

                nodeDelta = sum * TransferFunctionsPolicy::hiddenNeuronActivationFunctionDerivative(layer.getOutputValueForNeuron(neuron));

                layer.setNodeDeltaForNeuron(neuron, nodeDelta);
            }
        }
    };

    template<typename TransferFunctionsPolicy, typename OutputLayerType>
    struct OutputLayerNodeDeltasCalculator
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static void calculateOutputLayerNodeDeltas(OutputLayerType& outputLayer, ValueType const* const targetValues)
        {
            ValueType error;
            ValueType outputValueForNeuron;
            ValueType activationFunctionDerivative;
            ValueType nodeDelta;

            for(size_t neuron = 0; neuron < OutputLayerType::NumberOfNeuronsInLayer;++neuron)
            {
                outputValueForNeuron = outputLayer.getOutputValueForNeuron(neuron);
                error = (targetValues[neuron] - outputValueForNeuron);
                activationFunctionDerivative = TransferFunctionsPolicy::outputNeuronActivationFunctionDerivative(outputValueForNeuron);
                nodeDelta = (error * activationFunctionDerivative);
                
                outputLayer.setNodeDeltaForNeuron(neuron, nodeDelta);
            }
        }
    };

    template<typename TransferFunctionsPolicy, typename OutputLayerType>
    struct ClassificationOutputLayerNodeDeltasCalculator
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;
        
        static void calculateOutputLayerNodeDeltas(OutputLayerType& outputLayer, ValueType const* const targetValues)
        {
            ValueType outputValues[OutputLayerType::NumberOfNeuronsInLayer];
            ValueType results[OutputLayerType::NumberOfNeuronsInLayer];

            for(size_t neuron = 0; neuron < OutputLayerType::NumberOfNeuronsInLayer;++neuron)
            {
                outputValues[neuron] = outputLayer.getOutputValueForNeuron(neuron);
            }

            TransferFunctionsPolicy::outputNeuronActivationFunctionDerivative(&outputValues[0], targetValues, &results[0], OutputLayerType::NumberOfNeuronsInLayer);

            for(size_t neuron = 0; neuron < OutputLayerType::NumberOfNeuronsInLayer;++neuron)
            {
                outputLayer.setNodeDeltaForNeuron(neuron, results[neuron]);
            }
        }
    };

    template<typename TransferFunctionsPolicy, typename OutputLayerType, outputLayerConfiguration_e OutputLayerConfiguration>
    struct OutputLayerNodeDeltasCalculatorChooser
    {
    };

    template<typename TransferFunctionsPolicy, typename OutputLayerType>
    struct OutputLayerNodeDeltasCalculatorChooser<TransferFunctionsPolicy, OutputLayerType, FeedForwardOutputLayerConfiguration>
    {
        typedef OutputLayerNodeDeltasCalculator<TransferFunctionsPolicy, OutputLayerType> OutputLayerNodeDeltasCalculatorType;
    };

    template<typename TransferFunctionsPolicy, typename OutputLayerType>
    struct OutputLayerNodeDeltasCalculatorChooser<TransferFunctionsPolicy, OutputLayerType, ClassifierOutputLayerConfiguration>
    {
        typedef ClassificationOutputLayerNodeDeltasCalculator<TransferFunctionsPolicy, OutputLayerType> OutputLayerNodeDeltasCalculatorType;
    };

    template<typename NeuralNetworkType, size_t NumberOfInnerHiddenLayers>
    struct NetworkDeltasCalculator
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerType InnerHiddenLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef NodeDeltasCalculator<TransferFunctionsPolicy> NodeDeltasCalculatorType;
        typedef typename OutputLayerNodeDeltasCalculatorChooser<
                        TransferFunctionsPolicy,
                        OutputLayerType,
                        NeuralNetworkType::NeuralNetworkOutputLayerConfiguration>::OutputLayerNodeDeltasCalculatorType OutputLayerNodeDeltasCalculatorType;
        
        static void calculateNetworkDeltas(NeuralNetworkType& nn, ValueType const* const targetValues)
        {
            InputLayerType& inputLayer = nn.getInputLayer();
            InnerHiddenLayerType* pInnerHiddenLayers = nn.getPointerToInnerHiddenLayers();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            OutputLayerType& outputLayer = nn.getOutputLayer();

            OutputLayerNodeDeltasCalculatorType::calculateOutputLayerNodeDeltas(outputLayer, targetValues);

            NodeDeltasCalculatorType::calculateAndSetNodeDeltas(lastHiddenLayer, outputLayer);

            NodeDeltasCalculatorType::calculateAndSetNodeDeltas(pInnerHiddenLayers[NumberOfInnerHiddenLayers - 1], lastHiddenLayer);

            for (int hiddenLayer = static_cast<int>(NumberOfInnerHiddenLayers - 2); hiddenLayer >= 0; --hiddenLayer)
            {
                NodeDeltasCalculatorType::calculateAndSetNodeDeltas(pInnerHiddenLayers[hiddenLayer], pInnerHiddenLayers[hiddenLayer + 1]);
            }

            NodeDeltasCalculatorType::calculateAndSetNodeDeltas(inputLayer, pInnerHiddenLayers[0]);
        }
    };

    template<typename NeuralNetworkType>
    struct NetworkDeltasCalculator<NeuralNetworkType, 1>
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerType InnerHiddenLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef NodeDeltasCalculator<TransferFunctionsPolicy> NodeDeltasCalculatorType;
        typedef typename OutputLayerNodeDeltasCalculatorChooser<
                        TransferFunctionsPolicy,
                        OutputLayerType,
                        NeuralNetworkType::NeuralNetworkOutputLayerConfiguration>::OutputLayerNodeDeltasCalculatorType OutputLayerNodeDeltasCalculatorType;
        
        static void calculateNetworkDeltas(NeuralNetworkType& nn, ValueType const* const targetValues)
        {
            InputLayerType& inputLayer = nn.getInputLayer();
            InnerHiddenLayerType* pInnerHiddenLayers = nn.getPointerToInnerHiddenLayers();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            OutputLayerType& outputLayer = nn.getOutputLayer();

            OutputLayerNodeDeltasCalculatorType::calculateOutputLayerNodeDeltas(outputLayer, targetValues);

            NodeDeltasCalculatorType::calculateAndSetNodeDeltas(lastHiddenLayer, outputLayer);

            NodeDeltasCalculatorType::calculateAndSetNodeDeltas(pInnerHiddenLayers[0], lastHiddenLayer);

            NodeDeltasCalculatorType::calculateAndSetNodeDeltas(inputLayer, pInnerHiddenLayers[0]);
        }
    };

    template<typename NeuralNetworkType>
    struct NetworkDeltasCalculator<NeuralNetworkType, 0>
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef NodeDeltasCalculator<TransferFunctionsPolicy> NodeDeltasCalculatorType;
        typedef typename OutputLayerNodeDeltasCalculatorChooser<
                        TransferFunctionsPolicy,
                        OutputLayerType,
                        NeuralNetworkType::NeuralNetworkOutputLayerConfiguration>::OutputLayerNodeDeltasCalculatorType OutputLayerNodeDeltasCalculatorType;
        
        static void calculateNetworkDeltas(NeuralNetworkType& nn, ValueType const* const targetValues)
        {
            InputLayerType& inputLayer = nn.getInputLayer();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            OutputLayerType& outputLayer = nn.getOutputLayer();

            OutputLayerNodeDeltasCalculatorType::calculateOutputLayerNodeDeltas(outputLayer, targetValues);

            NodeDeltasCalculatorType::calculateAndSetNodeDeltas(lastHiddenLayer, outputLayer);

            NodeDeltasCalculatorType::calculateAndSetNodeDeltas(inputLayer, lastHiddenLayer);
        }
    };
    
    template<typename NeuralNetworkType>
    struct RecurrentNetworkDeltasCalculator
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkRecurrentLayerType RecurrentLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef NodeDeltasCalculator<TransferFunctionsPolicy> NodeDeltasCalculatorType;
        typedef typename OutputLayerNodeDeltasCalculatorChooser<
                        TransferFunctionsPolicy,
                        OutputLayerType,
                        NeuralNetworkType::NeuralNetworkOutputLayerConfiguration>::OutputLayerNodeDeltasCalculatorType OutputLayerNodeDeltasCalculatorType;

        static const size_t RecurrentConnectionDepth = RecurrentLayerType::RecurrentLayerRecurrentConnectionDepth;
        static const size_t GateMultiplier = NeuralNetworkType::HiddenLayerGateMultiplier;

        static void calculateNetworkDeltas(NeuralNetworkType& nn, ValueType const* const targetValues)
        {
            InputLayerType& inputLayer = nn.getInputLayer();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            RecurrentLayerType& recurrentLayer = nn.getRecurrentLayer();
            OutputLayerType& outputLayer = nn.getOutputLayer();

            OutputLayerNodeDeltasCalculatorType::calculateOutputLayerNodeDeltas(outputLayer, targetValues);

            // Compute raw dL/dh_t for each hidden neuron (WITHOUT activation derivative,
            // since LSTM gate derivatives are handled by computeGateDeltas)
            NodeDeltasCalculatorType::calculateAndSetRawNodeDeltas(lastHiddenLayer, outputLayer);

            // Decompose dL/dh_t into per-gate deltas through LSTM cell equations
            lastHiddenLayer.computeGateDeltas();

            // Backpropagate per-gate deltas to recurrent and input layers
            calculateNodeDeltasFromGateDeltas(recurrentLayer, lastHiddenLayer);
            calculateNodeDeltasFromGateDeltas(inputLayer, lastHiddenLayer);
        }

        template<typename LayerType, typename LstmLayerType>
        static void calculateNodeDeltasFromGateDeltas(LayerType& layer, const LstmLayerType& lstmLayer)
        {
            ValueType sum;
            ValueType nodeDelta;

            for (size_t neuron = 0; neuron < LayerType::NumberOfNeuronsInLayer; ++neuron)
            {
                sum = 0;
                for (size_t hiddenNeuron = 0; hiddenNeuron < LstmLayerType::NumberOfNeuronsInLayer; ++hiddenNeuron)
                {
                    for (size_t gate = 0; gate < GateMultiplier; ++gate)
                    {
                        const size_t conn = hiddenNeuron * GateMultiplier + gate;
                        sum += (layer.getWeightForNeuronAndConnection(neuron, conn) * lstmLayer.getGateDeltaForNeuron(hiddenNeuron, gate));
                    }
                }

                nodeDelta = sum * TransferFunctionsPolicy::hiddenNeuronActivationFunctionDerivative(layer.getOutputValueForNeuron(neuron));
                layer.setNodeDeltaForNeuron(neuron, nodeDelta);
            }
        }
    };    

    template<typename ValueType, typename GradientsManagerType, size_t NumberOfBiasNeurons>
    struct BiasGradientsCalculator
    {
    };
    
    template<typename ValueType, typename GradientsManagerType>
    struct BiasGradientsCalculator<ValueType, GradientsManagerType, 1>
    {
        template<typename LayerType, typename NextLayerType>
        static void calculateAndUpdateGradients(LayerType& layer, const NextLayerType& nextLayer, GradientsManagerType& gradientsManager)
        {
            const ValueType biasNeuronOutputValue = layer.getBiasNeuronOutputValue();
            ValueType nodeDelta;
            ValueType gradient;

            for (size_t nextNeuron = 0; nextNeuron < NextLayerType::NumberOfNeuronsInLayer; ++nextNeuron)
            {
                nodeDelta = nextLayer.getNodeDeltaForNeuron(nextNeuron);
                gradient = (biasNeuronOutputValue * nodeDelta);
                gradientsManager.updateBiasGradients(layer, nextNeuron, gradient);
            }
        }
    };
    
    template<typename ValueType, typename GradientsManagerType>
    struct BiasGradientsCalculator<ValueType, GradientsManagerType, 0>
    {
        template<typename LayerType, typename NextLayerType>
        static void calculateAndUpdateGradients(LayerType& layer, const NextLayerType& nextLayer, GradientsManagerType& gradientsManager)
        {
            (void)layer; // Suppress unused parameter warning
            (void)nextLayer; // Suppress unused parameter warning
            (void)gradientsManager; // Suppress unused parameter warning
        }
    };

    template<typename ValueType, typename GradientsManagerType, outputLayerConfiguration_e OutputLayerConfiguration, size_t NumberOfBiasNeurons>
    struct OuputLayerBiasGradientsCalculator
    {
    };
    
    template<typename ValueType, typename GradientsManagerType>
    struct OuputLayerBiasGradientsCalculator<ValueType, GradientsManagerType, FeedForwardOutputLayerConfiguration, 1>
    {
        template<typename LayerType, typename NextLayerType>
        static void calculateAndUpdateGradients(LayerType& layer, const NextLayerType& nextLayer, GradientsManagerType& gradientsManager)
        {
            const ValueType biasNeuronOutputValue = layer.getBiasNeuronOutputValue();
            ValueType nodeDelta;
            ValueType gradient;

            for (size_t nextNeuron = 0; nextNeuron < NextLayerType::NumberOfNeuronsInLayer; ++nextNeuron)
            {
                nodeDelta = nextLayer.getNodeDeltaForNeuron(nextNeuron);
                gradient = (biasNeuronOutputValue * nodeDelta);
                gradientsManager.updateBiasGradients(layer, nextNeuron, gradient);
            }
        }
    };
    
    template<typename ValueType, typename GradientsManagerType>
    struct OuputLayerBiasGradientsCalculator<ValueType, GradientsManagerType, ClassifierOutputLayerConfiguration, 1>
    {
        template<typename LayerType, typename NextLayerType>
        static void calculateAndUpdateGradients(LayerType& layer, const NextLayerType& nextLayer, GradientsManagerType& gradientsManager)
        {
            const ValueType biasNeuronOutputValue = layer.getBiasNeuronOutputValue();
            ValueType nodeDelta;
            ValueType gradient;

            for (size_t nextNeuron = 0; nextNeuron < NextLayerType::NumberOfNeuronsInLayer; ++nextNeuron)
            {
                nodeDelta = nextLayer.getNodeDeltaForNeuron(nextNeuron);
                gradient = (biasNeuronOutputValue * nodeDelta);
                gradientsManager.updateBiasGradients(layer, nextNeuron, gradient);
            }
        }
    };
    
    template<typename ValueType, typename GradientsManagerType, outputLayerConfiguration_e OutputLayerConfiguration>
    struct OuputLayerBiasGradientsCalculator<ValueType, GradientsManagerType, OutputLayerConfiguration, 0>
    {
        template<typename LayerType, typename NextLayerType>
        static void calculateAndUpdateGradients(LayerType& layer, const NextLayerType& nextLayer, GradientsManagerType& gradientsManager)
        {
            (void)layer;
            (void)nextLayer;
            (void)gradientsManager;
        }
    };

    template<typename ValueType, typename GradientsManagerType, size_t NumberOfBiasNeurons>
    struct GatedBiasGradientsHelper
    {
    };

    template<typename ValueType, typename GradientsManagerType>
    struct GatedBiasGradientsHelper<ValueType, GradientsManagerType, 1>
    {
        template<typename LayerType, typename LstmLayerType, size_t GateMultiplier>
        static void calculate(LayerType& layer, const LstmLayerType& lstmLayer, GradientsManagerType& gradientsManager)
        {
            const ValueType biasOutput = layer.getBiasNeuronOutputValue();
            for (size_t hiddenNeuron = 0; hiddenNeuron < LstmLayerType::NumberOfNeuronsInLayer; ++hiddenNeuron)
            {
                for (size_t gate = 0; gate < GateMultiplier; ++gate)
                {
                    const size_t conn = hiddenNeuron * GateMultiplier + gate;
                    const ValueType gradient = (biasOutput * lstmLayer.getGateDeltaForNeuron(hiddenNeuron, gate));
                    gradientsManager.updateBiasGradients(layer, conn, gradient);
                }
            }
        }
    };

    template<typename ValueType, typename GradientsManagerType>
    struct GatedBiasGradientsHelper<ValueType, GradientsManagerType, 0>
    {
        template<typename LayerType, typename LstmLayerType, size_t GateMultiplier>
        static void calculate(LayerType& layer, const LstmLayerType& lstmLayer, GradientsManagerType& gradientsManager)
        {
            (void)layer;
            (void)lstmLayer;
            (void)gradientsManager;
        }
    };

    template<typename TransferFunctionsPolicy, typename GradientsManagerType>
    struct GradientsCalculator
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        template<typename LayerType, typename NextLayerType>
        static void calculateAndUpdateGradients(LayerType& layer, const NextLayerType& nextLayer, GradientsManagerType& gradientsManager)
        {
            typedef BiasGradientsCalculator<ValueType, GradientsManagerType, LayerType::NumberOfBiasNeuronsInLayer> BiasGradientsCalculatorType;
            ValueType outputValue;
            ValueType nodeDelta;
            ValueType gradient;

            for (size_t neuron = 0; neuron < LayerType::NumberOfNeuronsInLayer; ++neuron)
            {
                outputValue = layer.getOutputValueForNeuron(neuron);
                for (size_t nextNeuron = 0; nextNeuron < NextLayerType::NumberOfNeuronsInLayer; ++nextNeuron)
                {
                    nodeDelta = nextLayer.getNodeDeltaForNeuron(nextNeuron);
                    gradient = (outputValue * nodeDelta);
                    gradientsManager.updateGradients(layer, neuron, nextNeuron, gradient);
                }
            }

            BiasGradientsCalculatorType::calculateAndUpdateGradients(layer, nextLayer, gradientsManager);
        }

        template<typename LayerType, typename LstmLayerType, size_t GateMultiplier>
        static void calculateAndUpdateGradientsGated(LayerType& layer, const LstmLayerType& lstmLayer, GradientsManagerType& gradientsManager)
        {
            ValueType outputValue;
            ValueType gradient;

            for (size_t neuron = 0; neuron < LayerType::NumberOfNeuronsInLayer; ++neuron)
            {
                outputValue = layer.getOutputValueForNeuron(neuron);
                for (size_t hiddenNeuron = 0; hiddenNeuron < LstmLayerType::NumberOfNeuronsInLayer; ++hiddenNeuron)
                {
                    for (size_t gate = 0; gate < GateMultiplier; ++gate)
                    {
                        const size_t conn = hiddenNeuron * GateMultiplier + gate;
                        gradient = (outputValue * lstmLayer.getGateDeltaForNeuron(hiddenNeuron, gate));
                        gradientsManager.updateGradients(layer, neuron, conn, gradient);
                    }
                }
            }

            // Update bias gradients for all gate connections
            GatedBiasGradientsHelper<ValueType, GradientsManagerType, LayerType::NumberOfBiasNeuronsInLayer>::template
                calculate<LayerType, LstmLayerType, GateMultiplier>(layer, lstmLayer, gradientsManager);
        }

        template<typename LayerType, typename NextLayerType>
        static void calculateAndUpdateOutputLayerGradients(LayerType& hiddenLayer, const NextLayerType& outputLayer, GradientsManagerType& gradientsManager)
        {
            typedef OuputLayerBiasGradientsCalculator<ValueType, GradientsManagerType, NextLayerType::OutputLayerConfiguration, LayerType::NumberOfBiasNeuronsInLayer> BiasGradientsCalculatorType;
            ValueType outputValue;
            ValueType nodeDelta;
            ValueType gradient;

            for (size_t neuron = 0; neuron < LayerType::NumberOfNeuronsInLayer; ++neuron)
            {
                outputValue = hiddenLayer.getOutputValueForNeuron(neuron);
                
                for (size_t nextNeuron = 0; nextNeuron < NextLayerType::NumberOfNeuronsInLayer; ++nextNeuron)
                {
                    nodeDelta = outputLayer.getNodeDeltaForNeuron(nextNeuron);
                    gradient = (outputValue * nodeDelta);
                    gradientsManager.updateGradients(hiddenLayer, neuron, nextNeuron, gradient);
                }
            }

            BiasGradientsCalculatorType::calculateAndUpdateGradients(hiddenLayer, outputLayer, gradientsManager);
        }
    };

    template<typename NeuralNetworkType, size_t NumberOfInnerHiddenLayers>
    struct NetworkGradientsCalculator
    {
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerType InnerHiddenLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef typename NeuralNetworkType::GradientsManagerType GradientsManagerType;
        typedef GradientsCalculator<TransferFunctionsPolicy, GradientsManagerType> GradientsCalculatorType;
        
        static void calculateNetworkGradients(NeuralNetworkType& nn)
        {
            InputLayerType& inputLayer = nn.getInputLayer();
            InnerHiddenLayerType* pInnerHiddenLayers = nn.getPointerToInnerHiddenLayers();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            OutputLayerType& outputLayer = nn.getOutputLayer();
            GradientsManagerType& gradientsManager = nn.getGradientsManager();

            GradientsCalculatorType::calculateAndUpdateOutputLayerGradients(lastHiddenLayer, outputLayer, gradientsManager);

            GradientsCalculatorType::calculateAndUpdateGradients(pInnerHiddenLayers[NumberOfInnerHiddenLayers - 1], lastHiddenLayer, gradientsManager);

            for (int hiddenLayer = static_cast<int>(NumberOfInnerHiddenLayers - 2); hiddenLayer >= 0; --hiddenLayer)
            {
                GradientsCalculatorType::calculateAndUpdateGradients(pInnerHiddenLayers[static_cast<size_t>(hiddenLayer)], pInnerHiddenLayers[static_cast<size_t>(hiddenLayer + 1)], gradientsManager);
            }

            GradientsCalculatorType::calculateAndUpdateGradients(inputLayer, pInnerHiddenLayers[0], gradientsManager);
        }
    };

    template<typename NeuralNetworkType>
    struct NetworkGradientsCalculator<NeuralNetworkType, 1>
    {
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerType InnerHiddenLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef typename NeuralNetworkType::GradientsManagerType GradientsManagerType;
        typedef GradientsCalculator<TransferFunctionsPolicy, GradientsManagerType> GradientsCalculatorType;
        
        static void calculateNetworkGradients(NeuralNetworkType& nn)
        {
            InputLayerType& inputLayer = nn.getInputLayer();
            InnerHiddenLayerType* pInnerHiddenLayers = nn.getPointerToInnerHiddenLayers();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            OutputLayerType& outputLayer = nn.getOutputLayer();
            GradientsManagerType& gradientsManager = nn.getGradientsManager();

            GradientsCalculatorType::calculateAndUpdateOutputLayerGradients(lastHiddenLayer, outputLayer, gradientsManager);

            GradientsCalculatorType::calculateAndUpdateGradients(pInnerHiddenLayers[0], lastHiddenLayer, gradientsManager);

            GradientsCalculatorType::calculateAndUpdateGradients(inputLayer, pInnerHiddenLayers[0], gradientsManager);
        }
    };

    template<typename NeuralNetworkType>
    struct NetworkGradientsCalculator<NeuralNetworkType, 0>
    {
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef typename NeuralNetworkType::GradientsManagerType GradientsManagerType;
        typedef GradientsCalculator<TransferFunctionsPolicy, GradientsManagerType> GradientsCalculatorType;

        static void calculateNetworkGradients(NeuralNetworkType& nn)
        {
            InputLayerType& inputLayer = nn.getInputLayer();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            OutputLayerType& outputLayer = nn.getOutputLayer();
            GradientsManagerType& gradientsManager = nn.getGradientsManager();

            GradientsCalculatorType::calculateAndUpdateOutputLayerGradients(lastHiddenLayer, outputLayer, gradientsManager);

            GradientsCalculatorType::calculateAndUpdateGradients(inputLayer, lastHiddenLayer, gradientsManager);
        }
    };

    template<typename NeuralNetworkType>
    struct RecurrentNetworkGradientsCalculator
    {
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkRecurrentLayerType RecurrentLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef typename NeuralNetworkType::GradientsManagerType GradientsManagerType;
        typedef GradientsCalculator<TransferFunctionsPolicy, GradientsManagerType> GradientsCalculatorType;

        static const size_t GateMultiplier = NeuralNetworkType::HiddenLayerGateMultiplier;

        static void calculateNetworkGradients(NeuralNetworkType& nn)
        {
            InputLayerType& inputLayer = nn.getInputLayer();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            RecurrentLayerType& recurrentLayer = nn.getRecurrentLayer();
            OutputLayerType& outputLayer = nn.getOutputLayer();
            GradientsManagerType& gradientsManager = nn.getGradientsManager();

            GradientsCalculatorType::calculateAndUpdateGradients(lastHiddenLayer, outputLayer, gradientsManager);

            GradientsCalculatorType::template calculateAndUpdateGradientsGated<RecurrentLayerType, LastHiddenLayerType, GateMultiplier>(recurrentLayer, lastHiddenLayer, gradientsManager);

            GradientsCalculatorType::template calculateAndUpdateGradientsGated<InputLayerType, LastHiddenLayerType, GateMultiplier>(inputLayer, lastHiddenLayer, gradientsManager);
        }
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct BackPropagationParent
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;
        typedef typename detail::gradient_clipping_policy_of<TransferFunctionsPolicy>::type GradientClippingPolicy;
        typedef typename detail::weight_decay_policy_of<TransferFunctionsPolicy>::type WeightDecayPolicy;
        typedef typename detail::learning_rate_schedule_policy_of<TransferFunctionsPolicy>::type LearningRateSchedulePolicy;

        ValueType getAccelerationRate() const
        {
            return this->mAccelerationRate;
        }

        ValueType getLearningRate() const
        {
            return this->mLearningRate;
        }

        ValueType getMomentumRate() const
        {
            return this->mMomentumRate;
        }

        void initialize()
        {
            this->mLearningRate = TransferFunctionsPolicy::initialLearningRate();
            this->mMomentumRate = TransferFunctionsPolicy::initialMomentumRate();
            this->mAccelerationRate = TransferFunctionsPolicy::initialAccelerationRate();
            this->mLearningRateSchedule.initialize(this->mLearningRate);
        }

        void setAccelerationRate(const ValueType& value)
        {
            this->mAccelerationRate = value;
        }

        void setLearningRate(const ValueType& value)
        {
            this->mLearningRate = value;
        }

        void setMomentumRate(const ValueType& value)
        {
            this->mMomentumRate = value;
        }

        void stepLearningRate()
        {
            this->mLearningRate = this->mLearningRateSchedule.step(this->mLearningRate);
        }

        template<typename LayerType, typename PreviousLayerType>
        void updateConnectionWeights(LayerType& layer, PreviousLayerType& previousLayer)
        {
            (void)layer; // Suppress unused parameter warning
            typedef BiasNeuronConnectionWeightUpdater<ValueType, PreviousLayerType::NumberOfBiasNeuronsInLayer> BiasNeuronConnectionWeightUpdaterType;
            ValueType previousDeltaWeight;
            ValueType currentDeltaWeight;
            ValueType newDeltaWeight;
            ValueType currentWeight;
            ValueType gradient;

            for (size_t neuron = 0; neuron < LayerType::NumberOfNeuronsInLayer; ++neuron)
            {
                for (size_t previousNeuron = 0; previousNeuron < PreviousLayerType::NumberOfNeuronsInLayer; ++previousNeuron)
                {
                    previousDeltaWeight = previousLayer.getPreviousDeltaWeightForNeuronAndConnection(previousNeuron, neuron);
                    currentDeltaWeight = previousLayer.getDeltaWeightForNeuronAndConnection(previousNeuron, neuron);
                    gradient = GradientClippingPolicy::clip(previousLayer.getGradientForNeuronAndConnection(previousNeuron, neuron));
                    newDeltaWeight =    (this->mLearningRate * gradient) +
                                        (this->mMomentumRate * currentDeltaWeight) +
                                        (this->mAccelerationRate * previousDeltaWeight);
                    currentWeight = WeightDecayPolicy::applyDecay(previousLayer.getWeightForNeuronAndConnection(previousNeuron, neuron), this->mLearningRate);

                    previousLayer.setDeltaWeightForNeuronAndConnection(previousNeuron, neuron, newDeltaWeight);
                    previousLayer.setWeightForNeuronAndConnection(previousNeuron, neuron, (currentWeight + newDeltaWeight));
                }

                //Update bias values
                BiasNeuronConnectionWeightUpdaterType::updateBiasConnectionWeights(previousLayer, neuron, this->mLearningRate, this->mMomentumRate, this->mAccelerationRate);
            }
        }

        template<typename LayerType, typename PreviousLayerType, size_t GateMultiplier>
        void updateConnectionWeightsGated(LayerType& layer, PreviousLayerType& previousLayer)
        {
            (void)layer; // Suppress unused parameter warning
            typedef BiasNeuronConnectionWeightUpdater<ValueType, PreviousLayerType::NumberOfBiasNeuronsInLayer> BiasNeuronConnectionWeightUpdaterType;
            ValueType previousDeltaWeight;
            ValueType currentDeltaWeight;
            ValueType newDeltaWeight;
            ValueType currentWeight;
            ValueType gradient;

            for (size_t neuron = 0; neuron < LayerType::NumberOfNeuronsInLayer; ++neuron)
            {
                for (size_t previousNeuron = 0; previousNeuron < PreviousLayerType::NumberOfNeuronsInLayer; ++previousNeuron)
                {
                    for (size_t gate = 0; gate < GateMultiplier; ++gate)
                    {
                        const size_t conn = neuron * GateMultiplier + gate;
                        previousDeltaWeight = previousLayer.getPreviousDeltaWeightForNeuronAndConnection(previousNeuron, conn);
                        currentDeltaWeight = previousLayer.getDeltaWeightForNeuronAndConnection(previousNeuron, conn);

                        gradient = GradientClippingPolicy::clip(previousLayer.getGradientForNeuronAndConnection(previousNeuron, conn));

                        newDeltaWeight =    (this->mLearningRate * gradient) +
                                            (this->mMomentumRate * currentDeltaWeight) +
                                            (this->mAccelerationRate * previousDeltaWeight);
                        currentWeight = WeightDecayPolicy::applyDecay(previousLayer.getWeightForNeuronAndConnection(previousNeuron, conn), this->mLearningRate);

                        previousLayer.setDeltaWeightForNeuronAndConnection(previousNeuron, conn, newDeltaWeight);
                        previousLayer.setWeightForNeuronAndConnection(previousNeuron, conn, (currentWeight + newDeltaWeight));
                    }
                }

                //Update bias values for each gate connection
                for (size_t gate = 0; gate < GateMultiplier; ++gate)
                {
                    BiasNeuronConnectionWeightUpdaterType::updateBiasConnectionWeights(previousLayer, neuron * GateMultiplier + gate, this->mLearningRate, this->mMomentumRate, this->mAccelerationRate);
                }
            }
        }
    protected:
        BackPropagationParent() : mLearningRate(0), mMomentumRate(0), mAccelerationRate(0)
        {
        }

        ~BackPropagationParent(){}

        ValueType mLearningRate;
        ValueType mMomentumRate;
        ValueType mAccelerationRate;
        LearningRateSchedulePolicy mLearningRateSchedule;

    private:
        BackPropagationParent(const BackPropagationParent&) {} // hide copy constructor
        BackPropagationParent& operator=(const BackPropagationParent&) {} // hide assignment operator
        static_assert(BatchSize > 0, "Invalid batch size.");
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct BackPropagationPolicy : public BackPropagationParent<TransferFunctionsPolicy, BatchSize>
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        template<typename NNType>
        void trainNetwork(NNType& nn, ValueType const* const targetValues)
        {
            typedef NetworkDeltasCalculator<NNType, NNType::NumberOfInnerHiddenLayers> NetworkDeltasCalculatorType;
            typedef NetworkGradientsCalculator<NNType, NNType::NumberOfInnerHiddenLayers> NetworkGradientsCalculatorType;
            typedef BackPropConnectionWeightUpdater<NNType,NNType::NumberOfInnerHiddenLayers> BackPropConnectionWeightUpdaterType;

            NetworkDeltasCalculatorType::calculateNetworkDeltas(nn, targetValues);

            NetworkGradientsCalculatorType::calculateNetworkGradients(nn);

            BackPropConnectionWeightUpdaterType::updateConnectionWeights(*this, nn);

            this->stepLearningRate();
        }
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct BackPropagationThruTimePolicy : public BackPropagationParent<TransferFunctionsPolicy, BatchSize>
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        template<typename NNType>
        void trainNetwork(NNType& nn, ValueType const* const targetValues)
        {
            typedef RecurrentNetworkDeltasCalculator<NNType> NetworkDeltasCalculatorType;
            typedef RecurrentNetworkGradientsCalculator<NNType> NetworkGradientsCalculatorType;
            typedef BackPropThruTimeConnectionWeightUpdater<NNType> BackPropConnectionWeightUpdaterType;

            NetworkDeltasCalculatorType::calculateNetworkDeltas(nn, targetValues);

            NetworkGradientsCalculatorType::calculateNetworkGradients(nn);

            BackPropConnectionWeightUpdaterType::updateConnectionWeights(*this, nn);

            this->stepLearningRate();
        }
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ClassifierBackPropagationPolicy : public BackPropagationParent<TransferFunctionsPolicy, BatchSize>
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        template<typename NNType>
        void trainNetwork(NNType& nn, ValueType const* const targetValues)
        {
            typedef NetworkDeltasCalculator<NNType, NNType::NumberOfInnerHiddenLayers> NetworkDeltasCalculatorType;
            typedef NetworkGradientsCalculator<NNType, NNType::NumberOfInnerHiddenLayers> NetworkGradientsCalculatorType;
            typedef BackPropConnectionWeightUpdater<NNType,NNType::NumberOfInnerHiddenLayers> BackPropConnectionWeightUpdaterType;

            NetworkDeltasCalculatorType::calculateNetworkDeltas(nn, targetValues);

            NetworkGradientsCalculatorType::calculateNetworkGradients(nn);

            BackPropConnectionWeightUpdaterType::updateConnectionWeights(*this, nn);

            this->stepLearningRate();
        }
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ClassifierBackPropagationThruTimePolicy : public BackPropagationParent<TransferFunctionsPolicy, BatchSize>
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        template<typename NNType>
        void trainNetwork(NNType& nn, ValueType const* const targetValues)
        {
            typedef RecurrentNetworkDeltasCalculator<NNType> NetworkDeltasCalculatorType;
            typedef RecurrentNetworkGradientsCalculator<NNType> NetworkGradientsCalculatorType;
            typedef BackPropThruTimeConnectionWeightUpdater<NNType> BackPropConnectionWeightUpdaterType;

            NetworkDeltasCalculatorType::calculateNetworkDeltas(nn, targetValues);

            NetworkGradientsCalculatorType::calculateNetworkGradients(nn);

            BackPropConnectionWeightUpdaterType::updateConnectionWeights(*this, nn);

            this->stepLearningRate();
        }
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct NullTrainingPolicy
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        ValueType getAccelerationRate() const
        {
            return 0;
        }
        
        ValueType getLearningRate() const
        {
            return 0;
        }

        ValueType getMomentumRate() const
        {
            return 0;
        }

        void initialize()
        {
        }
        
        void setAccelerationRate(const ValueType& value)
        {
        }
        
        void setLearningRate(const ValueType& value)
        {
        }
        
        void setMomentumRate(const ValueType& value)
        {
        }

        template<typename NNType>
        void trainNetwork(NNType& nn, ValueType const* const targetValues)
        {
            (void)nn; // Suppress unused parameter warning
            (void)targetValues; // Suppress unused parameter warning
        }
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize, bool HasRecurrentLayer, bool IsTrainable, outputLayerConfiguration_e OutputLayerConfiguration>
    struct BackPropTrainingPolicySelector
    {
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct BackPropTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, false, true, FeedForwardOutputLayerConfiguration>
    {
        typedef BackPropagationPolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct BackPropTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, false, true, ClassifierOutputLayerConfiguration>
    {
        typedef ClassifierBackPropagationPolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct BackPropTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, true, true, FeedForwardOutputLayerConfiguration>
    {
        typedef BackPropagationThruTimePolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct BackPropTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, true, true, ClassifierOutputLayerConfiguration>
    {
        typedef ClassifierBackPropagationThruTimePolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct BackPropTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, false, false, FeedForwardOutputLayerConfiguration>
    {
        typedef NullTrainingPolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct BackPropTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, false, false, ClassifierOutputLayerConfiguration>
    {
        typedef NullTrainingPolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct BackPropTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, true, false, FeedForwardOutputLayerConfiguration>
    {
        typedef NullTrainingPolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct BackPropTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, true, false, ClassifierOutputLayerConfiguration>
    {
        typedef NullTrainingPolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename ValueType>
    struct Connection
    {
        typedef ValueType ConnectionValueType;

        static const bool IsTrainable = false;

        Connection() : mWeight(0)
        {
        }

        ValueType getDeltaWeight() const
        {
            return 0;
        }

        ValueType getGradient() const
        {
            return 0;
        }

        ValueType getPreviousDeltaWeight() const
        {
            return 0;
        }

        ValueType getWeight() const
        {
            return this->mWeight;
        }

        void setDeltaWeight(const ValueType& value)
        {
        }

        void setGradient(const ValueType& value)
        {
        }
        
        void setWeight(const ValueType& value)
        {
            this->mWeight = value;
        }

        void * operator new(size_t, void *p)
        {
            return p;
        }
    protected:
        ValueType mWeight;
    };

    template<typename ValueType>
    struct TrainableConnection : public Connection<ValueType>
    {
        typedef ValueType ConnectionValueType;
        
        static const bool IsTrainable = true;

        TrainableConnection() : mDeltaWeight(0), mPreviousDeltaWeight(0), mGradient(0)
        {
        }

        ValueType getDeltaWeight() const
        {
            return this->mDeltaWeight;
        }

        ValueType getGradient() const
        {
            return this->mGradient;
        }

        ValueType getPreviousDeltaWeight() const
        {
            return this->mPreviousDeltaWeight;
        }

        void setDeltaWeight(const ValueType& value)
        {
            this->mPreviousDeltaWeight = this->mDeltaWeight;
            this->mDeltaWeight = value;
        }

        void setGradient(const ValueType& value)
        {
            this->mGradient = value;
        }
    protected:
        ValueType mDeltaWeight;
        ValueType mPreviousDeltaWeight;
        ValueType mGradient;
    };

    template<typename ValueType, bool IsTrainable>
    struct ConnectionTypeSelector
    {
    };

    template<typename ValueType>
    struct ConnectionTypeSelector<ValueType, true>
    {
        typedef TrainableConnection<ValueType> ConnectionType;
    };

    template<typename ValueType>
    struct ConnectionTypeSelector<ValueType, false>
    {
        typedef Connection<ValueType> ConnectionType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct Neuron
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;

        ValueType getOutputValue() const
        {
            return this->mOutputValue;
        }

        ValueType getOutputValueForConnection(const size_t connection) const
        {
            const size_t bufferIndex = connection * sizeof(ConnectionType);
            const ConnectionType* pConnection = reinterpret_cast<const ConnectionType*>(&this->mOutgoingConnectionsBuffer[bufferIndex]);
            const ValueType result = this->mOutputValue * pConnection->getWeight();

            return result;
        }
        
        ValueType getWeightForConnection(const size_t connection) const
        {
            const size_t bufferIndex = connection * sizeof(ConnectionType);
            const ConnectionType* pConnection = reinterpret_cast<const ConnectionType*>(&this->mOutgoingConnectionsBuffer[bufferIndex]);
            return pConnection->getWeight();
        }

        void initializeWeights()
        {
            for (size_t nextNeuron = 0; nextNeuron < NumberOfOutgoingConnections; ++nextNeuron)
            {
                this->setWeightForConnection(nextNeuron, TransferFunctionsPolicy::generateRandomWeight());
            }
        }

        void setWeightForConnection(const size_t connection, const ValueType& weight)
        {
            const size_t bufferIndex = connection * sizeof(ConnectionType);
            ConnectionType* pConnection = reinterpret_cast<ConnectionType*>(&this->mOutgoingConnectionsBuffer[bufferIndex]);
            pConnection->setWeight(weight);
        }

        void setIndex(const size_t index)
        {
            this->mIndex = index;
        }

        void setOutputValue(const ValueType& value)
        {
            this->mOutputValue = value;
        }

        void * operator new(size_t, void *p)
        {
            return p;
        }

    protected: // Don't instantiate class. Only for use by child classses
        Neuron()
        {
            size_t bufferIndex;
            for(size_t index = 0;index < NumberOfOutgoingConnections;++index)
            {
                bufferIndex = index * sizeof(ConnectionType);
                new (&this->mOutgoingConnectionsBuffer[bufferIndex]) ConnectionType();
            }

            this->mOutputValue = static_cast<ValueType>(0);
            this->mIndex = 0;
        }
        
        unsigned char mOutgoingConnectionsBuffer[NumberOfOutgoingConnections * sizeof(ConnectionType)];
        ValueType mOutputValue;
        size_t mIndex;
        
        static_assert(NumberOfOutgoingConnections > 0, "Invalid number of outgoing connections.");
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct TrainableNeuron : public Neuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef Neuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> ParentType;
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;

        ValueType getDeltaWeightForConnection(const size_t connection) const
        {
            const size_t bufferIndex = connection * sizeof(ConnectionType);
            const ConnectionType* pConnection = reinterpret_cast<const ConnectionType*>(&this->mOutgoingConnectionsBuffer[bufferIndex]);
            return pConnection->getDeltaWeight();
        }

        ValueType getGradientForConnection(const size_t connection) const
        {
            const size_t bufferIndex = connection * sizeof(ConnectionType);
            const ConnectionType* pConnection = reinterpret_cast<const ConnectionType*>(&this->mOutgoingConnectionsBuffer[bufferIndex]);
            return pConnection->getGradient();
        }

        ValueType getNodeDelta() const
        {
            return this->mNodeDelta;
        }

        ValueType getPreviousDeltaWeightForConnection(const size_t connection) const
        {
            const size_t bufferIndex = connection * sizeof(ConnectionType);
            const ConnectionType* pConnection = reinterpret_cast<const ConnectionType*>(&this->mOutgoingConnectionsBuffer[bufferIndex]);
            return pConnection->getPreviousDeltaWeight();
        }

        void initializeWeights()
        {
            ParentType::initializeWeights();

            for (size_t nextNeuron = 0; nextNeuron < NumberOfOutgoingConnections; ++nextNeuron)
            {
                this->setDeltaWeightForConnection(nextNeuron, TransferFunctionsPolicy::initialDeltaWeight());
            }
        }

        void setDeltaWeightForConnection(const size_t connection, const ValueType& deltaWeight)
        {
            const size_t bufferIndex = connection * sizeof(ConnectionType);
            ConnectionType* pConnection = reinterpret_cast<ConnectionType*>(&this->mOutgoingConnectionsBuffer[bufferIndex]);
            pConnection->setDeltaWeight(deltaWeight);
        }

        void setGradientForConnection(const size_t connection, const ValueType& gradient)
        {
            const size_t bufferIndex = connection * sizeof(ConnectionType);
            ConnectionType* pConnection = reinterpret_cast<ConnectionType*>(&this->mOutgoingConnectionsBuffer[bufferIndex]);
            pConnection->setGradient(gradient);
        }

        void setNodeDelta(const ValueType& value)
        {
            this->mNodeDelta = value;
        }
        
    protected: // Don't instantiate class. Only for use by child classses
        TrainableNeuron()
        {
            this->mNodeDelta = static_cast<ValueType>(0);
        }
        
        ValueType mNodeDelta;

        static_assert(NumberOfOutgoingConnections > 0, "Invalid number of outgoing connections.");
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct InputLayerNeuron : public Neuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename NeuronTransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct TrainableInputLayerNeuron : public TrainableNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename NeuronTransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy,
            bool IsTrainable
            >
    struct InputLayerNeuronTypeSelector
    {
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct InputLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, true>
    {
        typedef TrainableInputLayerNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> InputLayerNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct InputLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, false>
    {
        typedef InputLayerNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> InputLayerNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct HiddenLayerNeuron : public Neuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct TrainableHiddenLayerNeuron : public TrainableNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct GruHiddenLayerNeuron : public Neuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename NeuronTransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;

        ValueType getState(void) const { return this->mPreviousOutput; }
        void setState(const ValueType& state) { this->mPreviousOutput = state; }

        void setGateActivations(const ValueType& updateGate, const ValueType& resetGate,
                                const ValueType& candidateActivation, const ValueType& prevOutput,
                                const ValueType& recurrentSum)
        {
            mUpdateGateActivation = updateGate;
            mResetGateActivation = resetGate;
            mCandidateActivation = candidateActivation;
            mPreviousOutput = prevOutput;
            mRecurrentSum = recurrentSum;
        }

        ValueType getUpdateGateActivation() const { return mUpdateGateActivation; }
        ValueType getResetGateActivation() const { return mResetGateActivation; }
        ValueType getCandidateActivation() const { return mCandidateActivation; }
        ValueType getPreviousOutput() const { return mPreviousOutput; }
        ValueType getRecurrentSum() const { return mRecurrentSum; }

    private:
        ValueType mUpdateGateActivation;
        ValueType mResetGateActivation;
        ValueType mCandidateActivation;
        ValueType mPreviousOutput;
        ValueType mRecurrentSum;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct TrainableGruHiddenLayerNeuron : public TrainableNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename NeuronTransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;

        ValueType getState(void) const { return this->mPreviousOutput; }
        void setState(const ValueType& state) { this->mPreviousOutput = state; }

        void setGateActivations(const ValueType& updateGate, const ValueType& resetGate,
                                const ValueType& candidateActivation, const ValueType& prevOutput,
                                const ValueType& recurrentSum)
        {
            mUpdateGateActivation = updateGate;
            mResetGateActivation = resetGate;
            mCandidateActivation = candidateActivation;
            mPreviousOutput = prevOutput;
            mRecurrentSum = recurrentSum;
        }

        ValueType getUpdateGateActivation() const { return mUpdateGateActivation; }
        ValueType getResetGateActivation() const { return mResetGateActivation; }
        ValueType getCandidateActivation() const { return mCandidateActivation; }
        ValueType getPreviousOutput() const { return mPreviousOutput; }
        ValueType getRecurrentSum() const { return mRecurrentSum; }

        ValueType getGateDelta(const size_t gate) const { return mGateDeltas[gate]; }
        void setGateDelta(const size_t gate, const ValueType& delta) { mGateDeltas[gate] = delta; }

    private:
        ValueType mUpdateGateActivation;
        ValueType mResetGateActivation;
        ValueType mCandidateActivation;
        ValueType mPreviousOutput;
        ValueType mRecurrentSum;
        ValueType mGateDeltas[GRU_NUMBER_OF_GATES];
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct LstmHiddenLayerNeuron : public Neuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename NeuronTransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;

        ValueType getState(void) const { return this->mState; }
        void setState(const ValueType& state) { this->mState = state; }

        void setGateActivations(const ValueType& cellCandidate, const ValueType& inputGate,
                                const ValueType& forgetGate, const ValueType& outputGate,
                                const ValueType& prevCellState)
        {
            mCellCandidateActivation = cellCandidate;
            mInputGateActivation = inputGate;
            mForgetGateActivation = forgetGate;
            mOutputGateActivation = outputGate;
            mPreviousCellState = prevCellState;
        }

        ValueType getCellCandidateActivation() const { return mCellCandidateActivation; }
        ValueType getInputGateActivation() const { return mInputGateActivation; }
        ValueType getForgetGateActivation() const { return mForgetGateActivation; }
        ValueType getOutputGateActivation() const { return mOutputGateActivation; }
        ValueType getPreviousCellState() const { return mPreviousCellState; }

    private:
        ValueType mState;
        ValueType mCellCandidateActivation;
        ValueType mInputGateActivation;
        ValueType mForgetGateActivation;
        ValueType mOutputGateActivation;
        ValueType mPreviousCellState;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct TrainableLstmHiddenLayerNeuron : public TrainableNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename NeuronTransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;

        ValueType getState(void) const { return this->mState; }
        void setState(const ValueType& state) { this->mState = state; }

        void setGateActivations(const ValueType& cellCandidate, const ValueType& inputGate,
                                const ValueType& forgetGate, const ValueType& outputGate,
                                const ValueType& prevCellState)
        {
            mCellCandidateActivation = cellCandidate;
            mInputGateActivation = inputGate;
            mForgetGateActivation = forgetGate;
            mOutputGateActivation = outputGate;
            mPreviousCellState = prevCellState;
        }

        ValueType getCellCandidateActivation() const { return mCellCandidateActivation; }
        ValueType getInputGateActivation() const { return mInputGateActivation; }
        ValueType getForgetGateActivation() const { return mForgetGateActivation; }
        ValueType getOutputGateActivation() const { return mOutputGateActivation; }
        ValueType getPreviousCellState() const { return mPreviousCellState; }

        ValueType getGateDelta(const size_t gate) const { return mGateDeltas[gate]; }
        void setGateDelta(const size_t gate, const ValueType& delta) { mGateDeltas[gate] = delta; }

    private:
        ValueType mState;
        ValueType mCellCandidateActivation;
        ValueType mInputGateActivation;
        ValueType mForgetGateActivation;
        ValueType mOutputGateActivation;
        ValueType mPreviousCellState;
        ValueType mGateDeltas[LSTM_NUMBER_OF_GATES];
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy,
            bool IsTrainable,
            hiddenLayerConfiguration_e HiddenLayerConfig = NonRecurrentHiddenLayerConfig
            >
    struct HiddenLayerNeuronTypeSelector
    {
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, true, NonRecurrentHiddenLayerConfig>
    {
        typedef TrainableHiddenLayerNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> HiddenLayerNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, false, NonRecurrentHiddenLayerConfig>
    {
        typedef HiddenLayerNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> HiddenLayerNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, true, RecurrentHiddenLayerConfig>
    {
        typedef TrainableHiddenLayerNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> HiddenLayerNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, false, RecurrentHiddenLayerConfig>
    {
        typedef HiddenLayerNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> HiddenLayerNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, true, GRUHiddenLayerConfig>
    {
        typedef TrainableGruHiddenLayerNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> HiddenLayerNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, false, GRUHiddenLayerConfig>
    {
        typedef GruHiddenLayerNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> HiddenLayerNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, true, LSTMHiddenLayerConfig>
    {
        typedef TrainableLstmHiddenLayerNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> HiddenLayerNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, false, LSTMHiddenLayerConfig>
    {
        typedef LstmHiddenLayerNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> HiddenLayerNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct RecurrentLayerNeuron : public Neuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename NeuronTransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct TrainableRecurrentLayerNeuron : public TrainableNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename NeuronTransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy,
            bool IsTrainable
            >
    struct RecurrentLayerNeuronTypeSelector
    {
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct RecurrentLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, true>
    {
        typedef TrainableRecurrentLayerNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> RecurrentLayerNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct RecurrentLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, false>
    {
        typedef RecurrentLayerNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> RecurrentLayerNeuronType;
    };

    template<
            typename ConnectionType,
            typename TransferFunctionsPolicy
            >
    struct OutputLayerNeuron : public Neuron<ConnectionType, 1, TransferFunctionsPolicy>
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = 1;
    };

    template<
            typename ConnectionType,
            typename TransferFunctionsPolicy
            >
    struct TrainableOutputLayerNeuron : public TrainableNeuron<ConnectionType, 1, TransferFunctionsPolicy>
    {
        typedef ConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = 1;
    };

    template<
            typename ConnectionType,
            typename TransferFunctionsPolicy,
            bool IsTrainable
            >
    struct OutputLayerNeuronTypeSelector
    {
    };

    template<
            typename ConnectionType,
            typename TransferFunctionsPolicy
            >
    struct OutputLayerNeuronTypeSelector<ConnectionType, TransferFunctionsPolicy, true>
    {
        typedef TrainableOutputLayerNeuron<ConnectionType, TransferFunctionsPolicy> OutputLayerNeuronType;
    };

    template<
            typename ConnectionType,
            typename TransferFunctionsPolicy
            >
    struct OutputLayerNeuronTypeSelector<ConnectionType, TransferFunctionsPolicy, false>
    {
        typedef OutputLayerNeuron<ConnectionType, TransferFunctionsPolicy> OutputLayerNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct BiasNeuron : public Neuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct TrainableBiasNeuron : public TrainableNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy,
            bool IsTrainable
            >
    struct BiasNeuronTypeSelector
    {
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct BiasNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, true>
    {
        typedef TrainableBiasNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> BiasNeuronType;
    };

    template<
            typename ConnectionType,
            size_t NumberOfOutgoingConnections,
            typename TransferFunctionsPolicy
            >
    struct BiasNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, false>
    {
        typedef BiasNeuron<ConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> BiasNeuronType;
    };

    template<typename NeuronType, size_t NumberOfNeurons>
    struct Layer
    {
        typedef typename NeuronType::ValueType ValueType;
        typedef typename NeuronType::NeuronTransferFunctionsPolicy TransferFunctionsPolicy;

        static const size_t NumberOfNeuronsInLayer = NumberOfNeurons;

        ValueType getGradientForNeuronAndConnection(const size_t neuron, const size_t connection) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            return pNeuron->getGradientForConnection(connection);
        }

        ValueType getDeltaWeightForNeuronAndConnection(const size_t neuron, const size_t connection) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            return pNeuron->getDeltaWeightForConnection(connection);
        }

        ValueType getNodeDeltaForNeuron(const size_t neuron) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            return pNeuron->getNodeDelta();
        }

        ValueType getOutputValueForNeuron(const size_t neuron) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            return pNeuron->getOutputValue();
        }

        ValueType getOutputValueForOutgoingConnection(const size_t connection) const
        {
            size_t bufferIndex;
            ValueType sum(0);

            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                bufferIndex = neuron * sizeof(NeuronType);
                const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
                sum += pNeuron->getOutputValueForConnection(connection);
            }

            return sum;
        }

        ValueType getPreviousDeltaWeightForNeuronAndConnection(const size_t neuron, const size_t connection) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            return pNeuron->getPreviousDeltaWeightForConnection(connection);
        }

        ValueType getWeightForNeuronAndConnection(const size_t neuron, const size_t connection) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            return pNeuron->getWeightForConnection(connection);
        }

        void initializeNeurons()
        {
            size_t bufferIndex;
            NeuronType* pNeuron;
            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                bufferIndex = neuron * sizeof(NeuronType);
                pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
                pNeuron->setIndex(neuron);
            }
        }

        void initializeWeights()
        {
            size_t bufferIndex;
            NeuronType* pNeuron;
            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                bufferIndex = neuron * sizeof(NeuronType);
                pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
                pNeuron->initializeWeights();
            }
        }

        void setDeltaWeightForNeuronAndConnection(const size_t neuron, const size_t connection, const ValueType& value)
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            NeuronType* pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            pNeuron->setDeltaWeightForConnection(connection, value);
        }
        
        void setGradientForNeuronAndConnection(const size_t neuron, const size_t connection, const ValueType& value)
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            NeuronType* pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            pNeuron->setGradientForConnection(connection, value);
        }

        void setNodeDeltaForNeuron(const size_t neuron, const ValueType& value)
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            NeuronType* pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            pNeuron->setNodeDelta(value);
        }
        
        void setWeightForNeuronAndConnection(const size_t neuron, const size_t connection, const ValueType& weight)
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            NeuronType* pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            pNeuron->setWeightForConnection(connection, weight);
        }

        void * operator new(size_t, void *p)
        {
            return p;
        }
        NeuronType* getNeuron(const size_t neuron)
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            return reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
        }

    protected:
        Layer()
        {
            size_t bufferIndex;

            for(size_t neuronIndex = 0;neuronIndex < NumberOfNeurons;++neuronIndex)
            {
                bufferIndex = neuronIndex * sizeof(NeuronType);
                new (&this->mNeuronsBuffer[bufferIndex]) NeuronType();
            }
        }
        unsigned char mNeuronsBuffer[NumberOfNeurons * sizeof(NeuronType)];
        static_assert(NumberOfNeurons > 0, "Number neurons must be > 0");
    };

    template<typename NeuronType, size_t NumberOfNeurons>
    struct LayerWithBias : public Layer<NeuronType, NumberOfNeurons>
    {
        typedef typename NeuronType::NeuronConnectionType ConnectionType;
        typedef typename NeuronType::ValueType ValueType;
        typedef typename NeuronType::NeuronTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef typename BiasNeuronTypeSelector<ConnectionType, NeuronType::NumberOfOutgoingConnectionsFromNeuron, TransferFunctionsPolicy, ConnectionType::IsTrainable>::BiasNeuronType BiasNeuronType;

        static const size_t NumberOfBiasNeuronsInLayer = 1;

        ValueType getBiasNeuronDeltaWeightForConnection(const size_t connection) const
        {
            return this->mBiasNeuron.getDeltaWeightForConnection(connection);
        }

        ValueType getBiasNeuronGradientForConnection(const size_t connection) const
        {
            return this->mBiasNeuron.getGradientForConnection(connection);
        }

        ValueType getBiasNeuronOutputValue() const
        {
            return this->mBiasNeuron.getOutputValue();
        }

        ValueType getBiasNeuronPreviousDeltaWeightForConnection(const size_t connection) const
        {
            return this->mBiasNeuron.getPreviousDeltaWeightForConnection(connection);
        }

        ValueType getBiasNeuronWeightForConnection(const size_t connection) const
        {
            return this->mBiasNeuron.getWeightForConnection(connection);
        }

        ValueType getBiasNeuronValueForOutgoingConnection(const size_t connection) const
        {
            return this->mBiasNeuron.getOutputValueForConnection(connection);
        }

        void initializeNeurons()
        {
            Layer<NeuronType, NumberOfNeurons>::initializeNeurons();

            this->mBiasNeuron.setOutputValue(TransferFunctionsPolicy::initialBiasOutputValue());
        }

        void initializeWeights()
        {
            Layer<NeuronType, NumberOfNeurons>::initializeWeights();

            this->mBiasNeuron.initializeWeights();
        }

        void setBiasNeuronDeltaWeightForConnection(const size_t connection, const ValueType& deltaWeight)
        {
            this->mBiasNeuron.setDeltaWeightForConnection(connection, deltaWeight);
        }

        void setBiasNeuronGradientForConnection(const size_t connection, const ValueType& value)
        {
            this->mBiasNeuron.setGradientForConnection(connection, value);
        }

        void setBiasNeuronWeightForConnection(const size_t connection, const ValueType& weight)
        {
            this->mBiasNeuron.setWeightForConnection(connection, weight);
        }

    protected:
        LayerWithBias()
        {
        }

        ~LayerWithBias()
        {
        }

        BiasNeuronType mBiasNeuron;
    };

    template<typename NeuronType, size_t NumberOfNeurons>
    struct InputLayer : public LayerWithBias<NeuronType, NumberOfNeurons>
    {
        typedef typename NeuronType::NeuronTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfNeuronsInLayer = NumberOfNeurons;
        static const size_t NumberOfBiasNeuronsInLayer = 1;

        /**
         * Feed forward in the InputLayer simply latches the current value for each neuron.
         */
        void feedForward(ValueType const* const values)
        {
            size_t bufferIndex;
            NeuronType* pNeuron;
            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                bufferIndex = neuron * sizeof(NeuronType);
                pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
                pNeuron->setOutputValue(values[neuron]);
            }
        }
    };

    template<typename NeuronType, size_t NumberOfNeurons>
    struct HiddenLayer : public LayerWithBias<NeuronType, NumberOfNeurons>
    {
        typedef typename NeuronType::ValueType ValueType;
        typedef typename NeuronType::NeuronTransferFunctionsPolicy TransferFunctionsPolicy;

        static const size_t NumberOfNeuronsInLayer = NumberOfNeurons;
        static const size_t NumberOfBiasNeuronsInLayer = 1;

        // Fallback for non-LSTM hidden layers: gate deltas equal the node delta
        void computeGateDeltas() {}

        ValueType getGateDeltaForNeuron(const size_t neuron, const size_t gate) const
        {
            (void)gate;
            return this->getNodeDeltaForNeuron(neuron);
        }

        template<typename PreviousLayerType>
        void feedForward(const PreviousLayerType& previousLayer)
        {
            size_t bufferIndex;
            NeuronType* pNeuron;
            ValueType activation;
            ValueType sum;

            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                sum = previousLayer.getBiasNeuronValueForOutgoingConnection(neuron);
                sum += previousLayer.getOutputValueForOutgoingConnection(neuron);

                activation = TransferFunctionsPolicy::hiddenNeuronActivationFunction(sum);
                
                bufferIndex = neuron * sizeof(NeuronType);
                pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
                pNeuron->setOutputValue(activation);
            }
        }

        template<typename PreviousLayerType, typename RecurrentLayerType>
        void feedForward(const PreviousLayerType& previousLayer, RecurrentLayerType& recurrentLayer)
        {
            size_t bufferIndex;
            NeuronType* pNeuron;
            ValueType activation;
            ValueType sum;

            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                sum = previousLayer.getBiasNeuronValueForOutgoingConnection(neuron);

                sum += previousLayer.getOutputValueForOutgoingConnection(neuron);
                
                sum += recurrentLayer.getOutputValueForOutgoingConnection(neuron);

                activation = TransferFunctionsPolicy::hiddenNeuronActivationFunction(sum);
                
                bufferIndex = neuron * sizeof(NeuronType);
                pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
                pNeuron->setOutputValue(activation);
                
                recurrentLayer.setOutputValueForOutgoingConnection(neuron, activation);
            }
        }

        ValueType getRecurrentConnectionDeltaWeightForNeuronAtDepth(const size_t neuron, const size_t depth) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            return pNeuron->getRecurrentConnectionDeltaWeightAtDepth(depth);
        }
        
        ValueType getRecurrentConnectionGradientForNeuronAtDepth(const size_t neuron, const size_t depth) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            return pNeuron->getRecurrentConnectionGradientAtDepth(depth);
        }
        
        ValueType getRecurrentConnectionWeightForNeuronAtDepth(const size_t neuron, const size_t depth) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            return pNeuron->getRecurrentConnectionWeightAtDepth(depth);
        }
        
        void setRecurrentConnectionDeltaWeightForNeuronAtDepth(const size_t neuron, const size_t depth, const ValueType& value)
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            NeuronType* pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            pNeuron->setRecurrentConnectionDeltaWeightAtDepth(depth, value);
        }
        
        void setRecurrentConnectionWeightForNeuronAtDepth(const size_t neuron, const size_t depth, const ValueType& value)
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            NeuronType* pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            pNeuron->setRecurrentConnectionWeightAtDepth(depth, value);
        }
    };

    /**
     * Hidden neural net layer with GRU neurons.
     *
     * GRU equations (reset-after variant, compatible with PyTorch default):
     *   z_t = sigmoid(W_z * x_t + U_z * h_{t-1})       -- update gate
     *   r_t = sigmoid(W_r * x_t + U_r * h_{t-1})       -- reset gate
     *   h_hat_t = tanh(W_h * x_t + r_t * (U_h * h_{t-1}))  -- candidate
     *   h_t = (1 - z_t) * h_{t-1} + z_t * h_hat_t      -- output
     *
     * Each gate has independent weights via GateConnectionCount (3x connections):
     *   n*3 + 0 = update gate
     *   n*3 + 1 = reset gate
     *   n*3 + 2 = candidate
     */
    template<typename NeuronType, size_t NumberOfNeurons>
    struct GruHiddenLayer : public HiddenLayer<NeuronType, NumberOfNeurons>
    {
        typedef typename NeuronType::ValueType ValueType;
        typedef typename NeuronType::NeuronTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef SigmoidActivationPolicy<ValueType> UpdateGateActivationPolicy;
        typedef SigmoidActivationPolicy<ValueType> ResetGateActivationPolicy;
        typedef TanhActivationPolicy<ValueType> CandidateActivationPolicy;

        static const size_t NumberOfNeuronsInLayer = NumberOfNeurons;
        static const size_t NumberOfBiasNeuronsInLayer = 0;

        template<typename PreviousLayerType, typename RecurrentLayerType>
        void feedForward(const PreviousLayerType& previousLayer, RecurrentLayerType& recurrentLayer)
        {
            size_t bufferIndex;
            NeuronType* pNeuron;
            ValueType updateGateInput;
            ValueType resetGateInput;
            ValueType candidateInput;
            ValueType updateGateActivation;
            ValueType resetGateActivation;
            ValueType candidateActivation;
            ValueType recurrentSum;
            ValueType output;
            ValueType previousOutput;

            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                bufferIndex = neuron * sizeof(NeuronType);
                pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);

                previousOutput = pNeuron->getOutputValue();

                const size_t updateGateConn   = neuron * GRU_NUMBER_OF_GATES + 0;
                const size_t resetGateConn    = neuron * GRU_NUMBER_OF_GATES + 1;
                const size_t candidateConn    = neuron * GRU_NUMBER_OF_GATES + 2;

                // Update gate: z_t = sigmoid(W_z * x_t + U_z * h_{t-1})
                updateGateInput = previousLayer.getBiasNeuronValueForOutgoingConnection(updateGateConn);
                updateGateInput += previousLayer.getOutputValueForOutgoingConnection(updateGateConn);
                updateGateInput += recurrentLayer.getOutputValueForOutgoingConnection(updateGateConn);

                // Reset gate: r_t = sigmoid(W_r * x_t + U_r * h_{t-1})
                resetGateInput = previousLayer.getBiasNeuronValueForOutgoingConnection(resetGateConn);
                resetGateInput += previousLayer.getOutputValueForOutgoingConnection(resetGateConn);
                resetGateInput += recurrentLayer.getOutputValueForOutgoingConnection(resetGateConn);

                // Candidate: h_hat_t = tanh(W_h * x_t + r_t * (U_h * h_{t-1}))
                // First compute the recurrent sum for the candidate gate
                recurrentSum = recurrentLayer.getOutputValueForOutgoingConnection(candidateConn);
                candidateInput = previousLayer.getBiasNeuronValueForOutgoingConnection(candidateConn);
                candidateInput += previousLayer.getOutputValueForOutgoingConnection(candidateConn);

                // Apply gate activations
                updateGateActivation = UpdateGateActivationPolicy::activationFunction(updateGateInput);
                resetGateActivation = ResetGateActivationPolicy::activationFunction(resetGateInput);

                // Modulate recurrent contribution by reset gate (reset-after variant)
                candidateInput += resetGateActivation * recurrentSum;
                candidateActivation = CandidateActivationPolicy::activationFunction(candidateInput);

                // Save gate activations for backpropagation
                pNeuron->setGateActivations(updateGateActivation, resetGateActivation,
                                            candidateActivation, previousOutput, recurrentSum);

                // h_t = (1 - z_t) * h_{t-1} + z_t * h_hat_t
                output = (ValueType(1) - updateGateActivation) * previousOutput + updateGateActivation * candidateActivation;

                pNeuron->setOutputValue(output);

                recurrentLayer.setOutputValueForOutgoingConnection(neuron, output);
            }
        }

        /**
         * Compute per-gate deltas for GRU backpropagation.
         * Must be called after the standard hidden layer delta (dL/dh_t)
         * has been computed and stored in each neuron's mNodeDelta.
         *
         * Produces 3 gate-input deltas stored in the neuron's mGateDeltas[]:
         *   [0] = dL/d(updateGateInput) = dL/dh_t * (hHat - prevH) * z * (1-z)
         *   [1] = dL/d(resetGateInput)  = dL/d(candidateInput) * recurrentSum * r * (1-r)
         *   [2] = dL/d(candidateInput)  = dL/dh_t * z * (1 - hHat^2)
         */
        void computeGateDeltas()
        {
            size_t bufferIndex;
            NeuronType* pNeuron;

            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                bufferIndex = neuron * sizeof(NeuronType);
                pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);

                const ValueType dLdh = pNeuron->getNodeDelta();

                // Retrieve stored gate activations
                const ValueType z = pNeuron->getUpdateGateActivation();
                const ValueType r = pNeuron->getResetGateActivation();
                const ValueType hHat = pNeuron->getCandidateActivation();
                const ValueType prevH = pNeuron->getPreviousOutput();
                const ValueType recurrentSum = pNeuron->getRecurrentSum();

                // dL/d(candidateInput) = dL/dh_t * z_t * (1 - hHat^2)
                const ValueType dLdCandidateInput = dLdh * z * (ValueType(1) - hHat * hHat);

                // Update gate: dL/d(updateGateInput) = dL/dh_t * (hHat - prevH) * z * (1-z)
                pNeuron->setGateDelta(0, dLdh * (hHat - prevH) * z * (ValueType(1) - z));

                // Reset gate: dL/d(resetGateInput) = dL/d(candidateInput) * recurrentSum * r * (1-r)
                pNeuron->setGateDelta(1, dLdCandidateInput * recurrentSum * r * (ValueType(1) - r));

                // Candidate: dL/d(candidateInput)
                pNeuron->setGateDelta(2, dLdCandidateInput);
            }
        }

        ValueType getGateDeltaForNeuron(const size_t neuron, const size_t gate) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            return pNeuron->getGateDelta(gate);
        }
    };

    /**
     * Hidden neural net layer with LSTM neurons.
     */
    template<typename NeuronType, size_t NumberOfNeurons>
    struct LstmHiddenLayer : public HiddenLayer<NeuronType, NumberOfNeurons>
    {
        typedef typename NeuronType::ValueType ValueType;
        typedef typename NeuronType::NeuronTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef SigmoidActivationPolicy<ValueType> InputGateActivationPolicy;
        typedef SigmoidActivationPolicy<ValueType> ForgetGateActivationPolicy;
        typedef SigmoidActivationPolicy<ValueType> OutputGateActivationPolicy;
        typedef TanhActivationPolicy<ValueType> CellStateActivationPolicy;
        
        static const size_t NumberOfNeuronsInLayer = NumberOfNeurons;
        static const size_t NumberOfBiasNeuronsInLayer = 0;

        template<typename PreviousLayerType, typename RecurrentLayerType>
        void feedForward(const PreviousLayerType& previousLayer, RecurrentLayerType& recurrentLayer)
        {
            size_t bufferIndex;
            NeuronType* pNeuron;
            ValueType cellCandidateInput;
            ValueType inputGateInput;
            ValueType forgetGateInput;
            ValueType outputGateInput;
            ValueType inputActivation;
            ValueType inputGateActivation;
            ValueType forgetGateActivation;
            ValueType outputGateActivation;
            ValueType output;
            ValueType cellState;
            ValueType previousCellState;

            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                bufferIndex = neuron * sizeof(NeuronType);
                pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);

                previousCellState = pNeuron->getState();

                // Each gate has independent weights via separate connection indices.
                // For hidden neuron n, connections are:
                //   n*4 + 0 = cell candidate
                //   n*4 + 1 = input gate
                //   n*4 + 2 = forget gate
                //   n*4 + 3 = output gate
                const size_t cellCandidateConn = neuron * LSTM_NUMBER_OF_GATES + 0;
                const size_t inputGateConn     = neuron * LSTM_NUMBER_OF_GATES + 1;
                const size_t forgetGateConn    = neuron * LSTM_NUMBER_OF_GATES + 2;
                const size_t outputGateConn    = neuron * LSTM_NUMBER_OF_GATES + 3;

                // Cell candidate: independent weighted sum
                cellCandidateInput = previousLayer.getBiasNeuronValueForOutgoingConnection(cellCandidateConn);
                cellCandidateInput += previousLayer.getOutputValueForOutgoingConnection(cellCandidateConn);
                cellCandidateInput += recurrentLayer.getOutputValueForOutgoingConnection(cellCandidateConn);

                // Input gate: independent weighted sum
                inputGateInput = previousLayer.getBiasNeuronValueForOutgoingConnection(inputGateConn);
                inputGateInput += previousLayer.getOutputValueForOutgoingConnection(inputGateConn);
                inputGateInput += recurrentLayer.getOutputValueForOutgoingConnection(inputGateConn);

                // Forget gate: independent weighted sum
                forgetGateInput = previousLayer.getBiasNeuronValueForOutgoingConnection(forgetGateConn);
                forgetGateInput += previousLayer.getOutputValueForOutgoingConnection(forgetGateConn);
                forgetGateInput += recurrentLayer.getOutputValueForOutgoingConnection(forgetGateConn);

                // Output gate: independent weighted sum
                outputGateInput = previousLayer.getBiasNeuronValueForOutgoingConnection(outputGateConn);
                outputGateInput += previousLayer.getOutputValueForOutgoingConnection(outputGateConn);
                outputGateInput += recurrentLayer.getOutputValueForOutgoingConnection(outputGateConn);

                // Apply gate activations
                inputActivation = TransferFunctionsPolicy::hiddenNeuronActivationFunction(cellCandidateInput);
                inputGateActivation = InputGateActivationPolicy::activationFunction(inputGateInput);
                forgetGateActivation = ForgetGateActivationPolicy::activationFunction(forgetGateInput);
                outputGateActivation = OutputGateActivationPolicy::activationFunction(outputGateInput);

                // Save gate activations for backpropagation
                pNeuron->setGateActivations(inputActivation, inputGateActivation,
                                            forgetGateActivation, outputGateActivation,
                                            previousCellState);

                // Update cell state: c_t = f_t * c_{t-1} + i_t * c_hat_t
                cellState = (forgetGateActivation * previousCellState) + (inputActivation * inputGateActivation);
                pNeuron->setState(cellState);

                // Compute output: o_t * tanh(c_t)
                output = CellStateActivationPolicy::activationFunction(cellState);
                output *= outputGateActivation;

                pNeuron->setOutputValue(output);

                recurrentLayer.setOutputValueForOutgoingConnection(neuron, output);
            }
        }

        // NOTE: getOutputValueForOutgoingConnection is inherited from Layer,
        // which correctly iterates over ALL neurons to compute the weighted sum.
        // A previous override here only accessed a single neuron, which was a bug.

        /**
         * Compute per-gate deltas for LSTM backpropagation.
         * Must be called after the standard hidden layer delta (dL/dh_t) has been
         * computed and stored in each neuron's mNodeDelta.
         *
         * For each neuron, decomposes dL/dh_t through the LSTM cell equations:
         *   h_t = o_t * tanh(c_t)
         *   c_t = f_t * c_{t-1} + i_t * c_hat_t
         *
         * Produces 4 gate-input deltas stored in the neuron's mGateDeltas[]:
         *   [0] = dL/d(cellCandidateInput)  = dL/dc_t * i_t * tanh'(cellCandidateInput)
         *   [1] = dL/d(inputGateInput)      = dL/dc_t * c_hat_t * sigmoid'(inputGateInput)
         *   [2] = dL/d(forgetGateInput)     = dL/dc_t * c_{t-1} * sigmoid'(forgetGateInput)
         *   [3] = dL/d(outputGateInput)     = dL/dh_t * tanh(c_t) * sigmoid'(outputGateInput)
         */
        void computeGateDeltas()
        {
            size_t bufferIndex;
            NeuronType* pNeuron;

            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                bufferIndex = neuron * sizeof(NeuronType);
                pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);

                const ValueType dLdh = pNeuron->getNodeDelta();

                // Retrieve stored gate activations
                const ValueType cHat = pNeuron->getCellCandidateActivation();  // tanh(cellCandidateInput)
                const ValueType iGate = pNeuron->getInputGateActivation();     // sigmoid(inputGateInput)
                const ValueType fGate = pNeuron->getForgetGateActivation();    // sigmoid(forgetGateInput)
                const ValueType oGate = pNeuron->getOutputGateActivation();    // sigmoid(outputGateInput)
                const ValueType prevC = pNeuron->getPreviousCellState();
                const ValueType cellState = pNeuron->getState();

                // h_t = o_t * tanh(c_t)
                const ValueType tanhC = CellStateActivationPolicy::activationFunction(cellState);

                // dL/do_t = dL/dh_t * tanh(c_t)
                const ValueType dLdo = dLdh * tanhC;

                // dL/dc_t = dL/dh_t * o_t * (1 - tanh(c_t)^2)
                const ValueType dLdc = dLdh * oGate * (ValueType(1) - tanhC * tanhC);

                // Gate-input deltas through activation derivatives:
                // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)), where sigmoid(x) is the stored activation
                // tanh'(x) = 1 - tanh(x)^2, where tanh(x) is the stored activation

                // Cell candidate: dL/d(cellCandidateInput) = dL/dc_t * i_t * (1 - cHat^2)
                pNeuron->setGateDelta(0, dLdc * iGate * (ValueType(1) - cHat * cHat));

                // Input gate: dL/d(inputGateInput) = dL/dc_t * cHat * i_t * (1 - i_t)
                pNeuron->setGateDelta(1, dLdc * cHat * iGate * (ValueType(1) - iGate));

                // Forget gate: dL/d(forgetGateInput) = dL/dc_t * c_{t-1} * f_t * (1 - f_t)
                pNeuron->setGateDelta(2, dLdc * prevC * fGate * (ValueType(1) - fGate));

                // Output gate: dL/d(outputGateInput) = dL/do_t * o_t * (1 - o_t)
                pNeuron->setGateDelta(3, dLdo * oGate * (ValueType(1) - oGate));
            }
        }

        ValueType getGateDeltaForNeuron(const size_t neuron, const size_t gate) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            return pNeuron->getGateDelta(gate);
        }
    };

    template<typename ValueType>
    struct NullHiddenLayer
    {
        ValueType getBiasNeuronWeightForConnection(const size_t connection) const
        {
            (void)connection; // Suppress unused parameter warning
            return 0;
        }

        ValueType getWeightForNeuronAndConnection(const size_t neuron, const size_t connection) const
        {
            (void)neuron;   // Suppress unused parameter warning
            (void)connection; // Suppress unused parameter warning
            return 0;
        }

        void setBiasNeuronDeltaWeightForConnection(const size_t connection, const ValueType& deltaWeight)
        {
            (void)connection; // Suppress unused parameter warning
            (void)deltaWeight; // Suppress unused parameter warning
        }

        void setBiasNeuronGradientForConnection(const size_t connection, const ValueType& value)
        {
            (void)connection; // Suppress unused parameter warning
            (void)value; // Suppress unused parameter warning
        }

        void setBiasNeuronWeightForConnection(const size_t connection, const ValueType& weight)
        {
            (void)connection; // Suppress unused parameter warning
            (void)weight; // Suppress unused parameter warning
        }

        void setWeightForNeuronAndConnection(const size_t neuron, const size_t connection, const ValueType& weight)
        {
            (void)neuron;   // Suppress unused parameter warning
            (void)connection; // Suppress unused parameter warning
            (void)weight; // Suppress unused parameter warning
        }
    };

    /**
     * Handle recurrent values.
     */
    template<typename NeuronType, size_t NumberOfNeurons, size_t RecurrentConnectionDepth>
    struct RecurrentLayer : public Layer<NeuronType, NumberOfNeurons>
    {
        typedef typename NeuronType::ValueType ValueType;

        static const size_t NumberOfNeuronsInLayer = NumberOfNeurons;
        static const size_t NumberOfBiasNeuronsInLayer = 0;
        static const size_t RecurrentLayerRecurrentConnectionDepth = RecurrentConnectionDepth;

        ValueType getOutputValueForOutgoingConnection(const size_t connection) const
        {
            size_t bufferIndex;
            ValueType sum(0);

            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                bufferIndex = neuron * sizeof(NeuronType);
                const NeuronType* pNeuron = reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
                sum += pNeuron->getOutputValueForConnection(connection);
            }

            return sum;
        }
        
        void setOutputValueForOutgoingConnection(const size_t connection, const ValueType& value)
        {
            const size_t bufferIndex = connection * sizeof(NeuronType);
            NeuronType* pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
            pNeuron->setOutputValue(value);
        }
    };

    template<typename NeuronType, size_t NumberOfNeurons>
    struct OutputLayer : public Layer<NeuronType, NumberOfNeurons>
    {
        typedef typename NeuronType::NeuronTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfNeuronsInLayer = NumberOfNeurons;
        static const size_t NumberOfBiasNeuronsInLayer = 0;
        static const outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration;

        template<typename PreviousLayerType>
        void feedForward(const PreviousLayerType& previousLayer)
        {
            size_t bufferIndex;
            NeuronType* pNeuron;
            ValueType activation;
            ValueType sum;

            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                sum = previousLayer.getBiasNeuronValueForOutgoingConnection(neuron);
                sum += previousLayer.getOutputValueForOutgoingConnection(neuron);

                activation = TransferFunctionsPolicy::outputNeuronActivationFunction(sum);

                bufferIndex = neuron * sizeof(NeuronType);
                pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
                pNeuron->setOutputValue(activation);
            }
        }
    };

    template<typename NeuronType, size_t NumberOfNeurons>
    struct ClassifierOutputLayer : public Layer<NeuronType, NumberOfNeurons>
    {
        typedef typename NeuronType::NeuronTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfNeuronsInLayer = NumberOfNeurons;
        static const size_t NumberOfBiasNeuronsInLayer = 0;
        static const outputLayerConfiguration_e OutputLayerConfiguration = ClassifierOutputLayerConfiguration;

        template<typename PreviousLayerType>
        void feedForward(const PreviousLayerType& previousLayer)
        {
            size_t bufferIndex;
            ValueType values[NumberOfNeurons];
            ValueType results[NumberOfNeurons];
            NeuronType* pNeuron;

            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                values[neuron] = previousLayer.getBiasNeuronValueForOutgoingConnection(neuron);
                values[neuron] += previousLayer.getOutputValueForOutgoingConnection(neuron);
            }

            TransferFunctionsPolicy::outputNeuronActivationFunction(&values[0], &results[0], NumberOfNeurons);

            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                bufferIndex = neuron * sizeof(NeuronType);
                pNeuron = reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
                pNeuron->setOutputValue(results[neuron]);
            }
        }
    };

    template<typename NeuronType, size_t NumberOfNeurons, outputLayerConfiguration_e outputLayerConfiguration>
    struct OutputLayerTypeSelector
    {
    };

    template<typename NeuronType, size_t NumberOfNeurons>
    struct OutputLayerTypeSelector<NeuronType, NumberOfNeurons, FeedForwardOutputLayerConfiguration>
    {
        typedef OutputLayer<NeuronType, NumberOfNeurons> OutputLayerType;
    };

    template<typename NeuronType, size_t NumberOfNeurons>
    struct OutputLayerTypeSelector<NeuronType, NumberOfNeurons, ClassifierOutputLayerConfiguration>
    {
        typedef ClassifierOutputLayer<NeuronType, NumberOfNeurons> OutputLayerType;
    };

    template<typename InnerHiddenLayerType, size_t NumberOfInnerHiddenLayers>
    struct InnerHiddenLayerManager
    {
        InnerHiddenLayerManager()
        {
            size_t bufferIndex;
            for(size_t innerHiddenLayerIndex = 0;innerHiddenLayerIndex < NumberOfInnerHiddenLayers;++innerHiddenLayerIndex)
            {
                bufferIndex = innerHiddenLayerIndex * sizeof(InnerHiddenLayerType);
                new (&this->mInnerHiddenLayersBuffer[bufferIndex]) InnerHiddenLayerType();
            }

            this->pInnerHiddenLayers = reinterpret_cast<InnerHiddenLayerType*>(&this->mInnerHiddenLayersBuffer[0]);
        }
        
        void initializeInnerHiddenLayerWeights()
        {
            InnerHiddenLayerType* pInnerHiddenLayer;
            size_t bufferIndex;
            for (size_t hiddenLayer = 0; hiddenLayer < NumberOfInnerHiddenLayers; ++hiddenLayer)
            {
                bufferIndex = hiddenLayer * sizeof(InnerHiddenLayerType);
                pInnerHiddenLayer = reinterpret_cast<InnerHiddenLayerType*>(&this->mInnerHiddenLayersBuffer[bufferIndex]);
                pInnerHiddenLayer->initializeWeights();
            }
        }

        InnerHiddenLayerType* getPointerToInnerHiddenLayers()
        {
            return this->pInnerHiddenLayers;
        }

        void initializeInnerHiddenLayerNeurons()
        {
            InnerHiddenLayerType* pInnerHiddenLayer;
            size_t bufferIndex;
            for (size_t hiddenLayer = 0; hiddenLayer < NumberOfInnerHiddenLayers; ++hiddenLayer)
            {
                bufferIndex = hiddenLayer * sizeof(InnerHiddenLayerType);
                pInnerHiddenLayer = reinterpret_cast<InnerHiddenLayerType*>(&this->mInnerHiddenLayersBuffer[bufferIndex]);
                pInnerHiddenLayer->initializeNeurons();
            }
        }
    private:
        unsigned char mInnerHiddenLayersBuffer[NumberOfInnerHiddenLayers * sizeof(InnerHiddenLayerType)];
        InnerHiddenLayerType* pInnerHiddenLayers;
    };

    template<typename InnerHiddenLayerType>
    struct InnerHiddenLayerManager<InnerHiddenLayerType, 0>
    {
        InnerHiddenLayerManager()
        {
#if __cplusplus >= 201103L
            this->pInnerHiddenLayers = nullptr;
#else
            this->pInnerHiddenLayers = NULL;
#endif
        }

        static void initializeInnerHiddenLayerWeights()
        {
        }

        InnerHiddenLayerType* getPointerToInnerHiddenLayers()
        {
            return this->pInnerHiddenLayers;
        }
        
        static void initializeInnerHiddenLayerNeurons()
        {
        }
    private:
        InnerHiddenLayerType* pInnerHiddenLayers;
    };

    template<typename InputLayerType, typename InnerHiddenLayerType, typename LastHiddenLayerType, size_t NumberOfInnerHiddenLayers>
    struct HiddenLayerFeedForwardManager
    {
        static void feedForward(InputLayerType& inputLayer, InnerHiddenLayerType* pInnerHiddenLayers, LastHiddenLayerType& lastHiddenLayer)
        {
            pInnerHiddenLayers[0].feedForward(inputLayer);

            for (size_t hiddenLayer = 1; hiddenLayer < NumberOfInnerHiddenLayers; ++hiddenLayer)
            {
                pInnerHiddenLayers[hiddenLayer].feedForward(pInnerHiddenLayers[hiddenLayer - 1]);
            }

            lastHiddenLayer.feedForward(pInnerHiddenLayers[NumberOfInnerHiddenLayers - 1]);
        }
    };

    template<typename InputLayerType, typename InnerHiddenLayerType, typename LastHiddenLayerType>
    struct HiddenLayerFeedForwardManager<InputLayerType, InnerHiddenLayerType, LastHiddenLayerType, 1>
    {
        static void feedForward(InputLayerType& inputLayer, InnerHiddenLayerType* pInnerHiddenLayers, LastHiddenLayerType& lastHiddenLayer)
        {
            pInnerHiddenLayers[0].feedForward(inputLayer);

            lastHiddenLayer.feedForward(pInnerHiddenLayers[0]);
        }
    };

    template<typename InputLayerType, typename InnerHiddenLayerType, typename LastHiddenLayerType>
    struct HiddenLayerFeedForwardManager<InputLayerType, InnerHiddenLayerType, LastHiddenLayerType, 0>
    {
        static void feedForward(InputLayerType& inputLayer, InnerHiddenLayerType* pInnerHiddenLayers, LastHiddenLayerType& lastHiddenLayer)
        {
            (void)pInnerHiddenLayers; // Suppress unused parameter warning
            lastHiddenLayer.feedForward(inputLayer);
        }
    };

    template<typename ConnectionType, size_t NumberOfHiddenLayers, size_t NumberOfNeuronsInHiddenLayers, size_t NumberOfOutputs, typename TransferFunctionsPolicy, hiddenLayerConfiguration_e HiddenLayerConfig>
    struct HiddenLayerTypeSelector
    {
    };

    template<typename ConnectionType, size_t NumberOfHiddenLayers, size_t NumberOfNeuronsInHiddenLayers, size_t NumberOfOutputs, typename TransferFunctionsPolicy>
    struct HiddenLayerTypeSelector<ConnectionType, NumberOfHiddenLayers, NumberOfNeuronsInHiddenLayers, NumberOfOutputs, TransferFunctionsPolicy, NonRecurrentHiddenLayerConfig>
    {
        typedef typename HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfNeuronsInHiddenLayers, TransferFunctionsPolicy, ConnectionType::IsTrainable, NonRecurrentHiddenLayerConfig>::HiddenLayerNeuronType InnerHiddenLayerNeuronType;
        typedef HiddenLayer<InnerHiddenLayerNeuronType, NumberOfNeuronsInHiddenLayers> InnerHiddenLayerType;
        typedef typename HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutputs, TransferFunctionsPolicy, ConnectionType::IsTrainable, NonRecurrentHiddenLayerConfig>::HiddenLayerNeuronType LastHiddenLayerNeuronType;
        typedef HiddenLayer<LastHiddenLayerNeuronType, NumberOfNeuronsInHiddenLayers> LastHiddenLayerType;
    };

    template<typename ConnectionType, size_t NumberOfHiddenLayers, size_t NumberOfNeuronsInHiddenLayers, size_t NumberOfOutputs, typename TransferFunctionsPolicy>
    struct HiddenLayerTypeSelector<ConnectionType, NumberOfHiddenLayers, NumberOfNeuronsInHiddenLayers, NumberOfOutputs, TransferFunctionsPolicy, RecurrentHiddenLayerConfig>
    {
        // TODO: Need to add support for recurrent NNs with > 1 hidden layer
    };

    template<typename ConnectionType, size_t NumberOfHiddenLayers, size_t NumberOfNeuronsInHiddenLayers, size_t NumberOfOutputs, typename TransferFunctionsPolicy>
    struct HiddenLayerTypeSelector<ConnectionType, NumberOfHiddenLayers, NumberOfNeuronsInHiddenLayers, NumberOfOutputs, TransferFunctionsPolicy, GRUHiddenLayerConfig>
    {
        // TODO: Need to add support for recurrent NNs with > 1 hidden layer
    };

    template<typename ConnectionType, size_t NumberOfHiddenLayers, size_t NumberOfNeuronsInHiddenLayers, size_t NumberOfOutputs, typename TransferFunctionsPolicy>
    struct HiddenLayerTypeSelector<ConnectionType, NumberOfHiddenLayers, NumberOfNeuronsInHiddenLayers, NumberOfOutputs, TransferFunctionsPolicy, LSTMHiddenLayerConfig>
    {
        // TODO: Need to add support for recurrent NNs with > 1 hidden layer
    };

    template<typename ConnectionType, size_t NumberOfNeuronsInHiddenLayers, size_t NumberOfOutputs, typename TransferFunctionsPolicy>
    struct HiddenLayerTypeSelector<ConnectionType, 1, NumberOfNeuronsInHiddenLayers, NumberOfOutputs, TransferFunctionsPolicy, NonRecurrentHiddenLayerConfig>
    {
        typedef typename ConnectionType::ConnectionValueType ValueType;
        typedef NullHiddenLayer<ValueType> InnerHiddenLayerType;
        typedef typename HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutputs, TransferFunctionsPolicy, ConnectionType::IsTrainable, NonRecurrentHiddenLayerConfig>::HiddenLayerNeuronType LastHiddenLayerNeuronType;
        typedef HiddenLayer<LastHiddenLayerNeuronType, NumberOfNeuronsInHiddenLayers> LastHiddenLayerType;
    };

    template<typename ConnectionType, size_t NumberOfNeuronsInHiddenLayers, size_t NumberOfOutputs, typename TransferFunctionsPolicy>
    struct HiddenLayerTypeSelector<ConnectionType, 1, NumberOfNeuronsInHiddenLayers, NumberOfOutputs, TransferFunctionsPolicy, RecurrentHiddenLayerConfig>
    {
        typedef typename ConnectionType::ConnectionValueType ValueType;
        typedef NullHiddenLayer<ValueType> InnerHiddenLayerType;
        typedef typename HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutputs, TransferFunctionsPolicy, ConnectionType::IsTrainable, RecurrentHiddenLayerConfig>::HiddenLayerNeuronType LastHiddenLayerNeuronType;
        typedef HiddenLayer<LastHiddenLayerNeuronType, NumberOfNeuronsInHiddenLayers> LastHiddenLayerType;
    };

    template<typename ConnectionType, size_t NumberOfNeuronsInHiddenLayers, size_t NumberOfOutputs, typename TransferFunctionsPolicy>
    struct HiddenLayerTypeSelector<ConnectionType, 1, NumberOfNeuronsInHiddenLayers, NumberOfOutputs, TransferFunctionsPolicy, GRUHiddenLayerConfig>
    {
        typedef typename ConnectionType::ConnectionValueType ValueType;
        typedef NullHiddenLayer<ValueType> InnerHiddenLayerType;
        typedef typename HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutputs, TransferFunctionsPolicy, ConnectionType::IsTrainable, GRUHiddenLayerConfig>::HiddenLayerNeuronType LastHiddenLayerNeuronType;
        typedef GruHiddenLayer<LastHiddenLayerNeuronType, NumberOfNeuronsInHiddenLayers> LastHiddenLayerType;
    };

    template<typename ConnectionType, size_t NumberOfNeuronsInHiddenLayers, size_t NumberOfOutputs, typename TransferFunctionsPolicy>
    struct HiddenLayerTypeSelector<ConnectionType, 1, NumberOfNeuronsInHiddenLayers, NumberOfOutputs, TransferFunctionsPolicy, LSTMHiddenLayerConfig>
    {
        typedef typename ConnectionType::ConnectionValueType ValueType;
        typedef NullHiddenLayer<ValueType> InnerHiddenLayerType;
        typedef typename HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutputs, TransferFunctionsPolicy, ConnectionType::IsTrainable, LSTMHiddenLayerConfig>::HiddenLayerNeuronType LastHiddenLayerNeuronType;
        typedef LstmHiddenLayer<LastHiddenLayerNeuronType, NumberOfNeuronsInHiddenLayers> LastHiddenLayerType;
    };

    /**
     * Selects the correct last hidden layer type based on hidden layer configuration.
     */
    template<typename NeuronType, size_t NumberOfNeurons, hiddenLayerConfiguration_e HiddenLayerConfig>
    struct LastHiddenLayerTypeSelector
    {
        typedef HiddenLayer<NeuronType, NumberOfNeurons> LastHiddenLayerType;
    };

    template<typename NeuronType, size_t NumberOfNeurons>
    struct LastHiddenLayerTypeSelector<NeuronType, NumberOfNeurons, GRUHiddenLayerConfig>
    {
        typedef GruHiddenLayer<NeuronType, NumberOfNeurons> LastHiddenLayerType;
    };

    template<typename NeuronType, size_t NumberOfNeurons>
    struct LastHiddenLayerTypeSelector<NeuronType, NumberOfNeurons, LSTMHiddenLayerConfig>
    {
        typedef LstmHiddenLayer<NeuronType, NumberOfNeurons> LastHiddenLayerType;
    };

    struct NullRecurrentLayer
    {
        static const size_t RecurrentLayerRecurrentConnectionDepth = 0;
        static const size_t NumberOfBiasNeuronsInLayer = 0;
        
        void initializeNeurons()
        {
        }

        void initializeWeights()
        {
        }
    };

    template<typename ConnectionType, size_t NumberOfNeuronsInHiddenLayers, size_t NumberOfOutgoingConnectionsPerNeuron, typename TransferFunctionsPolicy, size_t RecurrentConnectionDepth, bool HasRecurrentLayer>
    struct RecurrentLayerTypeSelector
    {
    };

    template<typename ConnectionType, size_t NumberOfNeuronsInHiddenLayers, size_t NumberOfOutgoingConnectionsPerNeuron, typename TransferFunctionsPolicy, size_t RecurrentConnectionDepth>
    struct RecurrentLayerTypeSelector<ConnectionType, NumberOfNeuronsInHiddenLayers, NumberOfOutgoingConnectionsPerNeuron, TransferFunctionsPolicy, RecurrentConnectionDepth, false>
    {
        typedef NullRecurrentLayer RecurrentLayerType;
    };

    template<typename ConnectionType, size_t NumberOfNeuronsInHiddenLayers, size_t NumberOfOutgoingConnectionsPerNeuron, typename TransferFunctionsPolicy, size_t RecurrentConnectionDepth>
    struct RecurrentLayerTypeSelector<ConnectionType, NumberOfNeuronsInHiddenLayers, NumberOfOutgoingConnectionsPerNeuron, TransferFunctionsPolicy, RecurrentConnectionDepth, true>
    {
        typedef typename RecurrentLayerNeuronTypeSelector<ConnectionType, NumberOfOutgoingConnectionsPerNeuron, TransferFunctionsPolicy, ConnectionType::IsTrainable>::RecurrentLayerNeuronType RecurrentLayerNeuronType;
        typedef RecurrentLayer<RecurrentLayerNeuronType, NumberOfNeuronsInHiddenLayers, RecurrentConnectionDepth> RecurrentLayerType;
    };

    template<typename NeuralNetworkType, bool IsTrainable>
    struct WeightInitPolicy
    {
    };

    template<typename NeuralNetworkType>
    struct WeightInitPolicy<NeuralNetworkType, true>
    {
        static void initializeWeights(NeuralNetworkType& neuralNetwork)
        {
            neuralNetwork.initializeWeights();
        }
    };

    template<typename NeuralNetworkType>
    struct WeightInitPolicy<NeuralNetworkType, false>
    {
        static void initializeWeights(NeuralNetworkType& neuralNetwork)
        {
            (void)neuralNetwork; // Suppress unused parameter warning
        }
    };

    /**
     * MLP Neural Network
     */
    template<
            typename ValueType,
            size_t NumberOfInputs,
            size_t NumberOfHiddenLayers,
            size_t NumberOfNeuronsInHiddenLayers,
            size_t NumberOfOutputs,
            typename TransferFunctionsPolicy,
            bool IsTrainable = true,
            size_t BatchSize = 1,
            bool HasRecurrentLayer = false,
            hiddenLayerConfiguration_e HiddenLayerConfig = NonRecurrentHiddenLayerConfig,
            size_t RecurrentConnectionDepth = 0,
            outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration
            >
    class MultilayerPerceptron
    {
    public:
        typedef MultilayerPerceptron<   ValueType,
                                        NumberOfInputs,
                                        NumberOfHiddenLayers,
                                        NumberOfNeuronsInHiddenLayers,
                                        NumberOfOutputs,
                                        TransferFunctionsPolicy,
                                        IsTrainable,
                                        BatchSize,
                                        HasRecurrentLayer,
                                        HiddenLayerConfig,
                                        RecurrentConnectionDepth,
                                        OutputLayerConfiguration> NeuralNetworkType;

        typedef ValueType NeuralNetworkValueType;
        typedef typename ConnectionTypeSelector<ValueType, IsTrainable>::ConnectionType ConnectionType;
        typedef HiddenLayerTypeSelector<ConnectionType,
                                        NumberOfHiddenLayers,
                                        NumberOfNeuronsInHiddenLayers,
                                        NumberOfOutputs,
                                        TransferFunctionsPolicy,
                                        HiddenLayerConfig> HiddenLayerTypeSelectorType;
        typedef typename HiddenLayerTypeSelectorType::InnerHiddenLayerType InnerHiddenLayerType;
        typedef typename HiddenLayerTypeSelectorType::LastHiddenLayerType LastHiddenLayerType;
        static const size_t MlpInputToHiddenConnections = GateConnectionCount<NumberOfNeuronsInHiddenLayers, HiddenLayerConfig>::value;
        static const size_t HiddenLayerGateMultiplier = GateConnectionCount<1, HiddenLayerConfig>::value;
        typedef RecurrentLayerTypeSelector< ConnectionType,
                                            NumberOfNeuronsInHiddenLayers,
                                            MlpInputToHiddenConnections,
                                            TransferFunctionsPolicy,
                                            RecurrentConnectionDepth,
                                            HasRecurrentLayer> RecurrentLayerTypeSelectorType;
        typedef typename RecurrentLayerTypeSelectorType::RecurrentLayerType NeuralNetworkRecurrentLayerType;
        typedef typename InputLayerNeuronTypeSelector<ConnectionType, MlpInputToHiddenConnections, TransferFunctionsPolicy, IsTrainable>::InputLayerNeuronType InputLayerNeuronType;
        typedef typename OutputLayerNeuronTypeSelector<ConnectionType, TransferFunctionsPolicy, IsTrainable>::OutputLayerNeuronType OutputLayerNeuronType;
        typedef InputLayer<InputLayerNeuronType, NumberOfInputs> InputLayerType;
        typedef OutputLayerTypeSelector<OutputLayerNeuronType,
                                        NumberOfOutputs,
                                        OutputLayerConfiguration> OutputLayerTypeSelectorType;
        typedef typename OutputLayerTypeSelectorType::OutputLayerType NeuralNetworkOutputLayerType;
        typedef TransferFunctionsPolicy NeuralNetworkTransferFunctionsPolicy;
        typedef typename GradientsManagerSelector<
                                                    NeuralNetworkType,
                                                    NumberOfHiddenLayers - 1,
                                                    IsTrainable>::GradientsManagerType GradientsManagerType;
        typedef typename BackPropTrainingPolicySelector<
                                                        TransferFunctionsPolicy,
                                                        BatchSize,
                                                        HasRecurrentLayer,
                                                        IsTrainable,
                                                        OutputLayerConfiguration>::TrainingPolicyType TrainingPolicyType;

        typedef WeightInitPolicy<NeuralNetworkType, IsTrainable> WeightInitPolicyType;

        static const size_t NeuralNetworkNumberOfHiddenLayers = NumberOfHiddenLayers;
        static const size_t NumberOfInnerHiddenLayers = NumberOfHiddenLayers - 1;
        static const size_t NumberOfInputLayerNeurons = InputLayerType::NumberOfNeuronsInLayer;
        static const size_t NumberOfHiddenLayerNeurons = LastHiddenLayerType::NumberOfNeuronsInLayer;
        static const size_t NumberOfOutputLayerNeurons = NeuralNetworkOutputLayerType::NumberOfNeuronsInLayer;
        static const size_t NeuralNetworkRecurrentConnectionDepth = NeuralNetworkRecurrentLayerType::RecurrentLayerRecurrentConnectionDepth;
        static const size_t NeuralNetworkBatchSize = BatchSize;
        static const outputLayerConfiguration_e NeuralNetworkOutputLayerConfiguration = OutputLayerConfiguration;

        MultilayerPerceptron()
        {
            size_t bufferIndex;

            this->mInputLayer.initializeNeurons();

            this->mInnerHiddenLayerManager.initializeInnerHiddenLayerNeurons();

            this->mLastHiddenLayer.initializeNeurons();

            this->mOutputLayer.initializeNeurons();

            this->mRecurrentLayer.initializeNeurons();

            this->mTrainingPolicy.initialize();

            WeightInitPolicyType::initializeWeights(*this);

            for(size_t learnedValue = 0;learnedValue < NumberOfOutputLayerNeurons;++learnedValue)
            {
                bufferIndex = learnedValue * sizeof(ValueType);
                new (&this->mLearnedValuesBuffer[bufferIndex]) ValueType();
            }
        }

        void initializeWeights()
        {
            this->mInputLayer.initializeWeights();

            this->mInnerHiddenLayerManager.initializeInnerHiddenLayerWeights();

            this->mLastHiddenLayer.initializeWeights();

            this->mOutputLayer.initializeWeights();

            this->mRecurrentLayer.initializeWeights();
        }

        ValueType calculateError(ValueType const* const targetValues)
        {
            ValueType* pLearnedValues = reinterpret_cast<ValueType*>(&this->mLearnedValuesBuffer[0]);

            getLearnedValues(pLearnedValues);

            return TransferFunctionsPolicy::calculateError(targetValues, pLearnedValues);
        }

        void feedForward(ValueType const* const values)
        {
            this->feedForwardInputLayer(values);
            
            this->feedForwardHiddenLayers();

            this->feedForwardOutputLayer();
        }

        ValueType getAccelerationRate() const
        {
            return this->mTrainingPolicy.getAccelerationRate();
        }
        
        ValueType getLearningRate() const
        {
            return this->mTrainingPolicy.getLearningRate();
        }

        ValueType getMomentumRate() const
        {
            return this->mTrainingPolicy.getMomentumRate();
        }

        NeuralNetworkRecurrentLayerType& getRecurrentLayer()
        {
            return this->mRecurrentLayer;
        }
        
        GradientsManagerType& getGradientsManager()
        {
            return this->mGradientsManager;
        }

        ValueType getHiddenLayerBiasNeuronWeightForConnection(const size_t hiddenLayer, const size_t connection)
        {
            if((NumberOfHiddenLayers - 1) == hiddenLayer)
            {
                return this->mLastHiddenLayer.getBiasNeuronWeightForConnection(connection);
            }
            else
            {
                return this->mInnerHiddenLayerManager.getPointerToInnerHiddenLayers()[hiddenLayer].getBiasNeuronWeightForConnection(connection);
            }
        }

        ValueType getHiddenLayerWeightForNeuronAndConnection(const size_t hiddenLayer, const size_t neuron, const size_t connection)
        {
            if((NumberOfHiddenLayers - 1) == hiddenLayer)
            {
                return this->mLastHiddenLayer.getWeightForNeuronAndConnection(neuron, connection);
            }
            else
            {
                return this->mInnerHiddenLayerManager.getPointerToInnerHiddenLayers()[hiddenLayer].getWeightForNeuronAndConnection(neuron, connection);
            }
        }

        InputLayerType& getInputLayer()
        {
            return this->mInputLayer;
        }

        ValueType getInputLayerBiasNeuronWeightForConnection(const size_t connection) const
        {
            return this->mInputLayer.getBiasNeuronWeightForConnection(connection);
        }

        ValueType getInputLayerWeightForNeuronAndConnection(const size_t neuron, const size_t connection) const
        {
            return this->mInputLayer.getWeightForNeuronAndConnection(neuron, connection);
        }

        LastHiddenLayerType& getLastHiddenLayer()
        {
            return this->mLastHiddenLayer;
        }

        void getLearnedValues(ValueType* output) const
        {
            for (size_t outputNeuron = 0; outputNeuron < NumberOfOutputLayerNeurons; ++outputNeuron)
            {
                output[outputNeuron] = mOutputLayer.getOutputValueForNeuron(outputNeuron);
            }
        }

        NeuralNetworkOutputLayerType& getOutputLayer()
        {
            return this->mOutputLayer;
        }

        InnerHiddenLayerType* getPointerToInnerHiddenLayers()
        {
            return this->mInnerHiddenLayerManager.getPointerToInnerHiddenLayers();
        }

        void setAccelerationRate(const ValueType& value)
        {
            this->mTrainingPolicy.setAccelerationRate(value);
        }
        
        void setLearningRate(const ValueType& value)
        {
            this->mTrainingPolicy.setLearningRate(value);
        }
        
        void setMomentumRate(const ValueType& value)
        {
            this->mTrainingPolicy.setMomentumRate(value);
        }

        void setHiddenLayerBiasNeuronWeightForConnection(const size_t hiddenLayer, const size_t connection, const ValueType& weight)
        {
            if((NumberOfHiddenLayers - 1) == hiddenLayer)
            {
                this->mLastHiddenLayer.setBiasNeuronWeightForConnection(connection, weight);
            }
            else
            {
                this->mInnerHiddenLayerManager.getPointerToInnerHiddenLayers()[hiddenLayer].setBiasNeuronWeightForConnection(connection, weight);
            }
        }

        void setHiddenLayerBiasDeltaWeightForConnection(const size_t hiddenLayer, const size_t connection, const ValueType& deltaWeight)
        {
            if((NumberOfHiddenLayers - 1) == hiddenLayer)
            {
                this->mLastHiddenLayer.setBiasNeuronDeltaWeightForConnection(connection, deltaWeight);
            }
            else
            {
                this->mInnerHiddenLayerManager.getPointerToInnerHiddenLayers()[hiddenLayer].setBiasNeuronDeltaWeightForConnection(connection, deltaWeight);
            }
        }
        
        void setHiddenLayerDeltaWeightForNeuronAndConnection(const size_t hiddenLayer, const size_t neuron, const size_t connection, const ValueType& deltaWeight)
        {
            if((NumberOfHiddenLayers - 1) == hiddenLayer)
            {
                this->mLastHiddenLayer.setDeltaWeightForNeuronAndConnection(neuron, connection, deltaWeight);
            }
            else
            {
                this->mInnerHiddenLayerManager.getPointerToInnerHiddenLayers()[hiddenLayer].setDeltaWeightForNeuronAndConnection(neuron, connection, deltaWeight);
            }
        }
        
        void setHiddenLayerWeightForNeuronAndConnection(const size_t hiddenLayer, const size_t neuron, const size_t connection, const ValueType& weight)
        {
            if((NumberOfHiddenLayers - 1) == hiddenLayer)
            {
                this->mLastHiddenLayer.setWeightForNeuronAndConnection(neuron, connection, weight);
            }
            else
            {
                this->mInnerHiddenLayerManager.getPointerToInnerHiddenLayers()[hiddenLayer].setWeightForNeuronAndConnection(neuron, connection, weight);
            }
        }

        void setInputLayerBiasWeightForConnection(const size_t connection, const ValueType& weight)
        {
            this->mInputLayer.setBiasNeuronWeightForConnection(connection, weight);
        }
        
        void setInputLayerBiasDeltaWeightForConnection(const size_t connection, const ValueType& deltaWeight)
        {
            this->mInputLayer.setBiasNeuronDeltaWeightForConnection(connection, deltaWeight);
        }
        
        void setInputLayerDeltaWeightForNeuronAndConnection(const size_t neuron, const size_t connection, const ValueType& deltaWeight)
        {
            this->mInputLayer.setDeltaWeightForNeuronAndConnection(neuron, connection, deltaWeight);
        }

        void setInputLayerWeightForNeuronAndConnection(const size_t neuron, const size_t connection, const ValueType& weight)
        {
            this->mInputLayer.setWeightForNeuronAndConnection(neuron, connection, weight);
        }

        void setOutputLayerDeltaWeightForNeuronAndConnection(const size_t neuron, const size_t connection, const ValueType& deltaWeight)
        {
            this->mOutputLayer.setDeltaWeightForNeuronAndConnection(neuron, connection, deltaWeight);
        }

        void setOutputLayerWeightForNeuronAndConnection(const size_t neuron, const size_t connection, const ValueType& weight)
        {
            this->mOutputLayer.setWeightForNeuronAndConnection(neuron, connection, weight);
        }

        void setWeights(NeuralNetworkType& other)
        {
            ValueType weightValue;
            size_t    hiddenLayer = 0;

            for(size_t i = 0; i < NumberOfInputLayerNeurons; ++i)
            {
                for(size_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
                {
                    weightValue = other.getInputLayerWeightForNeuronAndConnection(i, h);
                    this->setInputLayerWeightForNeuronAndConnection(i, h, weightValue);
                }
            }

            for(size_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
            {
                weightValue = other.getInputLayerBiasNeuronWeightForConnection(h);
                this->setInputLayerBiasWeightForConnection(h, weightValue);
            }

            for(;hiddenLayer < (NumberOfHiddenLayers - 1);++hiddenLayer)
            {
                for(size_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
                {
                    for(size_t h1 = 0; h1 < NumberOfHiddenLayerNeurons; ++h1)
                    {
                        weightValue = other.getHiddenLayerWeightForNeuronAndConnection(hiddenLayer, h, h1);
                        this->setHiddenLayerWeightForNeuronAndConnection(hiddenLayer, h, h1, weightValue);
                    }
                }

                for(size_t h1 = 0; h1 < NumberOfHiddenLayerNeurons; ++h1)
                {
                    weightValue = other.getHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, h1);
                    this->setHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, h1, weightValue);
                }
            }

            for(size_t hiddenNeuron = 0; hiddenNeuron < NumberOfHiddenLayerNeurons; ++hiddenNeuron)
            {
                for(size_t outputNeuron = 0; outputNeuron < NumberOfOutputLayerNeurons; ++outputNeuron)
                {
                    weightValue = other.getHiddenLayerWeightForNeuronAndConnection(hiddenLayer, hiddenNeuron, outputNeuron);
                    this->setHiddenLayerWeightForNeuronAndConnection(hiddenLayer, hiddenNeuron, outputNeuron, weightValue);
                }
            }

            for(size_t outputNeuron = 0; outputNeuron < NumberOfOutputLayerNeurons; ++outputNeuron)
            {
                weightValue = other.getHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, outputNeuron);
                this->setHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, outputNeuron, weightValue);
            }
        }

        void trainNetwork(ValueType const* const targetValues)
        {
            this->mTrainingPolicy.trainNetwork(*this, targetValues);
        }
    protected:
        void feedForwardHiddenLayers()
        {
            typedef HiddenLayerFeedForwardManager<InputLayerType, InnerHiddenLayerType, LastHiddenLayerType, NumberOfInnerHiddenLayers> HiddenLayerFeedForwardManagerType;

            HiddenLayerFeedForwardManagerType::feedForward(this->mInputLayer, this->mInnerHiddenLayerManager.getPointerToInnerHiddenLayers(), this->mLastHiddenLayer);
        }
        
        void feedForwardInputLayer(ValueType const* const values)
        {
            this->mInputLayer.feedForward(values);
        }
        
        void feedForwardOutputLayer()
        {
            this->mOutputLayer.feedForward(this->mLastHiddenLayer);
        }

    protected:
        InputLayerType mInputLayer;
        InnerHiddenLayerManager<InnerHiddenLayerType, NumberOfInnerHiddenLayers> mInnerHiddenLayerManager;
        LastHiddenLayerType mLastHiddenLayer;
        NeuralNetworkOutputLayerType mOutputLayer;
        NeuralNetworkRecurrentLayerType mRecurrentLayer;
        TrainingPolicyType mTrainingPolicy;
        GradientsManagerType mGradientsManager;
    private:
        unsigned char mLearnedValuesBuffer[NumberOfOutputLayerNeurons * sizeof(ValueType)];
        unsigned char alignmentPads[1];
    private:
        MultilayerPerceptron(const MultilayerPerceptron&) {} // hide copy constructor
        MultilayerPerceptron& operator=(const MultilayerPerceptron&) {} // hide assignment operator

        static_assert(NumberOfInputs > 0, "Invalid number of inputs.");
        static_assert(NumberOfHiddenLayers > 0, "Invalid number of hidden layers.");
        static_assert(NumberOfNeuronsInHiddenLayers > 0, "Invalid number of neurons in hidden layers layers.");
        static_assert(NumberOfOutputs > 0, "Invalid number of outputs.");
        static_assert(NumberOfOutputLayerNeurons == TransferFunctionsPolicy::NumberOfTransferFunctionsOutputNeurons, "TransferFunctionPolicy NumberOfOutputNeurons is incorrect.");
    };

    /**
     * Recurrent MLP
     */
    template<
            typename ValueType,
            size_t NumberOfInputs,
            size_t NumberOfHiddenLayers,
            size_t NumberOfNeuronsInHiddenLayers,
            size_t NumberOfOutputs,
            typename TransferFunctionsPolicy,
            bool IsTrainable = true,
            size_t BatchSize = 1,
            bool HasRecurrentLayer = true,
            hiddenLayerConfiguration_e HiddenLayerConfig = RecurrentHiddenLayerConfig,
            size_t RecurrentConnectionDepth = 1,
            outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration
            >
    class RecurrentMultilayerPerceptron : public MultilayerPerceptron<  ValueType,
                                                                        NumberOfInputs,
                                                                        NumberOfHiddenLayers,
                                                                        NumberOfNeuronsInHiddenLayers,
                                                                        NumberOfOutputs, 
                                                                        TransferFunctionsPolicy,
                                                                        IsTrainable,
                                                                        BatchSize,
                                                                        HasRecurrentLayer,
                                                                        HiddenLayerConfig,
                                                                        RecurrentConnectionDepth,
                                                                        OutputLayerConfiguration
                                                                        >
    {
    public:
        void feedForward(ValueType const* const values)
        {
            this->feedForwardInputLayer(values);
            
            this->mLastHiddenLayer.feedForward(this->mInputLayer, this->mRecurrentLayer);
            
            this->feedForwardOutputLayer();
        }
    private:
        static_assert(RecurrentConnectionDepth > 0, "Invalid recurrent connection depth.");
    };

    /**
     * Elman Network
     */
    template<
            typename ValueType,
            size_t NumberOfInputs,
            size_t NumberOfNeuronsInHiddenLayers,
            size_t NumberOfOutputs,
            typename TransferFunctionsPolicy,
            bool IsTrainable = true,
            size_t BatchSize = 1,
            outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration
            >
    class ElmanNetwork : public RecurrentMultilayerPerceptron<  ValueType,
                                                                NumberOfInputs,
                                                                1,
                                                                NumberOfNeuronsInHiddenLayers,
                                                                NumberOfOutputs,
                                                                TransferFunctionsPolicy,
                                                                IsTrainable,
                                                                BatchSize
                                                                >
    {
    };

    // =========================================================================
    // Heterogeneous Hidden Layer Support
    //
    // The types and templates below enable neural networks where each hidden
    // layer can have a different number of neurons. The existing
    // MultilayerPerceptron class (above) is fully preserved for backward
    // compatibility.
    // =========================================================================

    // Forward declaration
    template<size_t... Sizes>
    struct HiddenLayers;

    namespace detail {
        template<size_t First, size_t... Rest>
        struct FirstOf
        {
            static const size_t value = First;
        };

        template<size_t... Sizes>
        struct LastOf;

        template<size_t S>
        struct LastOf<S>
        {
            static const size_t value = S;
        };

        template<size_t S, size_t... Rest>
        struct LastOf<S, Rest...> : LastOf<Rest...>
        {
        };

        template<size_t Count, size_t Size, size_t... Accumulated>
        struct UniformHiddenLayersHelper
        {
            typedef typename UniformHiddenLayersHelper<Count - 1, Size, Accumulated..., Size>::type type;
        };

        template<size_t Size, size_t... Accumulated>
        struct UniformHiddenLayersHelper<0, Size, Accumulated...>
        {
            typedef HiddenLayers<Accumulated...> type;
        };

        template<size_t A, size_t B>
        struct PairwiseGradients
        {
            static const size_t value = A * B + B;
        };

        template<size_t A, size_t B, size_t... Rest>
        struct PairwiseGradientsVar
        {
            static const size_t value = A * B + B + PairwiseGradientsVar<B, Rest...>::value;
        };

        template<size_t A, size_t B>
        struct PairwiseGradientsVar<A, B>
        {
            static const size_t value = A * B + B;
        };
        // Computes total gradients for the full chain: Input, HiddenSizes..., Output
        template<size_t NumberOfInputs, size_t NumberOfOutputs, typename HiddenLayersDescriptor>
        struct TotalGradientsForNetwork;

        template<size_t NumberOfInputs, size_t NumberOfOutputs, size_t... Sizes>
        struct TotalGradientsForNetwork<NumberOfInputs, NumberOfOutputs, HiddenLayers<Sizes...> >
        {
            static const size_t value = PairwiseGradientsVar<NumberOfInputs, Sizes..., NumberOfOutputs>::value;
        };
    } // namespace detail

    /**
     * Descriptor type for specifying per-layer hidden neuron counts.
     * Example: HiddenLayers<10, 5, 3> means 3 hidden layers with 10, 5, and 3 neurons.
     */
    template<size_t... Sizes>
    struct HiddenLayers
    {
        static const size_t Count = sizeof...(Sizes);
        static const size_t FirstLayerSize = detail::FirstOf<Sizes...>::value;
        static const size_t LastLayerSize = detail::LastOf<Sizes...>::value;
    };

    /**
     * Helper alias: generates a HiddenLayers<Size, Size, ..., Size> with Count copies.
     * Example: UniformHiddenLayers<3, 5> = HiddenLayers<5, 5, 5>
     */
    template<size_t Count, size_t Size>
    struct UniformHiddenLayersAlias
    {
        typedef typename detail::UniformHiddenLayersHelper<Count, Size>::type type;
    };

    // =========================================================================
    // Layer Chain: recursive heterogeneous storage for inner hidden layers
    // =========================================================================

    struct EmptyLayerChain
    {
        void initializeWeights() {}
        void initializeNeurons() {}
    };

    template<typename LayerType, typename RestChainType>
    struct LayerChain
    {
        LayerType layer;
        RestChainType rest;

        void initializeWeights()
        {
            layer.initializeWeights();
            rest.initializeWeights();
        }

        void initializeNeurons()
        {
            layer.initializeNeurons();
            rest.initializeNeurons();
        }
    };

    template<typename ChainType>
    struct ChainFirstLayer;

    template<typename LayerType, typename RestType>
    struct ChainFirstLayer<LayerChain<LayerType, RestType> >
    {
        typedef LayerType type;

        static LayerType& get(LayerChain<LayerType, RestType>& chain)
        {
            return chain.layer;
        }
    };

    // =========================================================================
    // InnerLayerChainBuilder: builds the chain of inner hidden layers from
    // a HiddenLayers descriptor. All layers except the last become "inner"
    // layers stored in the chain.
    // =========================================================================

    template<typename ConnectionType, typename TransferFunctionsPolicy, bool IsTrainable,
             hiddenLayerConfiguration_e HLConfig, typename HiddenLayerSizes>
    struct InnerLayerChainBuilder;

    // Single hidden layer: no inner layers
    template<typename CT, typename TF, bool IT, hiddenLayerConfiguration_e HLC, size_t S0>
    struct InnerLayerChainBuilder<CT, TF, IT, HLC, HiddenLayers<S0> >
    {
        typedef EmptyLayerChain ChainType;
    };

    // Two or more hidden layers: recursive construction
    // Inner layers use the proper layer type based on HLC (e.g., LstmHiddenLayer for LSTM)
    template<typename CT, typename TF, bool IT, hiddenLayerConfiguration_e HLC, size_t S0, size_t S1, size_t... Rest>
    struct InnerLayerChainBuilder<CT, TF, IT, HLC, HiddenLayers<S0, S1, Rest...> >
    {
        typedef typename HiddenLayerNeuronTypeSelector<CT, S1, TF, CT::IsTrainable, HLC>::HiddenLayerNeuronType NeuronType;
        typedef typename LastHiddenLayerTypeSelector<NeuronType, S0, HLC>::LastHiddenLayerType CurrentLayerType;
        typedef typename InnerLayerChainBuilder<CT, TF, IT, HLC, HiddenLayers<S1, Rest...> >::ChainType RestChainType;
        typedef LayerChain<CurrentLayerType, RestChainType> ChainType;
    };

    // =========================================================================
    // RecurrentLayerChainBuilder: builds a parallel chain of recurrent layers
    // for multi-layer recurrent/LSTM networks. Each inner hidden layer gets
    // its own recurrent connection.
    // =========================================================================

    template<typename ConnectionType, typename TransferFunctionsPolicy,
             size_t RecurrentConnectionDepth, bool HasRecurrentLayer,
             typename HiddenLayerSizes>
    struct RecurrentLayerChainBuilder;

    // Single hidden layer: no inner recurrent layers needed
    template<typename CT, typename TF, size_t RCD, bool HRL, size_t S0>
    struct RecurrentLayerChainBuilder<CT, TF, RCD, HRL, HiddenLayers<S0> >
    {
        typedef EmptyLayerChain ChainType;
    };

    // Two or more hidden layers: create recurrent layer for each inner layer
    template<typename CT, typename TF, size_t RCD, size_t S0, size_t S1, size_t... Rest>
    struct RecurrentLayerChainBuilder<CT, TF, RCD, true, HiddenLayers<S0, S1, Rest...> >
    {
        typedef typename RecurrentLayerNeuronTypeSelector<CT, S0, TF, CT::IsTrainable>::RecurrentLayerNeuronType RecurrentNeuronType;
        typedef RecurrentLayer<RecurrentNeuronType, S0, RCD> CurrentRecurrentLayerType;
        typedef typename RecurrentLayerChainBuilder<CT, TF, RCD, true, HiddenLayers<S1, Rest...> >::ChainType RestChainType;
        typedef LayerChain<CurrentRecurrentLayerType, RestChainType> ChainType;
    };

    // No recurrent layer: empty chain
    template<typename CT, typename TF, size_t RCD, size_t S0, size_t S1, size_t... Rest>
    struct RecurrentLayerChainBuilder<CT, TF, RCD, false, HiddenLayers<S0, S1, Rest...> >
    {
        typedef EmptyLayerChain ChainType;
    };

    // =========================================================================
    // Chain-based feed-forward helper
    // =========================================================================

    template<typename ChainType>
    struct ChainFeedForwardHelper;

    template<>
    struct ChainFeedForwardHelper<EmptyLayerChain>
    {
        template<typename PrevLayerType, typename LastHiddenLayerType>
        static void feedForward(PrevLayerType& prev, EmptyLayerChain& chain, LastHiddenLayerType& lastHidden)
        {
            (void)chain;
            lastHidden.feedForward(prev);
        }
    };

    template<typename LayerType, typename RestChainType>
    struct ChainFeedForwardHelper<LayerChain<LayerType, RestChainType> >
    {
        template<typename PrevLayerType, typename LastHiddenLayerType>
        static void feedForward(PrevLayerType& prev, LayerChain<LayerType, RestChainType>& chain, LastHiddenLayerType& lastHidden)
        {
            chain.layer.feedForward(prev);
            ChainFeedForwardHelper<RestChainType>::feedForward(chain.layer, chain.rest, lastHidden);
        }
    };

    // =========================================================================
    // Chain-based recurrent feed-forward helper
    // Walks the inner hidden layer chain and recurrent layer chain in
    // parallel, passing each inner layer its corresponding recurrent
    // connection. The last hidden layer gets the main recurrent layer.
    // =========================================================================

    template<typename HiddenChainType, typename RecurrentChainType>
    struct ChainRecurrentFeedForwardHelper;

    // Base case: empty inner chains, feed forward to last hidden layer
    template<>
    struct ChainRecurrentFeedForwardHelper<EmptyLayerChain, EmptyLayerChain>
    {
        template<typename PrevLayerType, typename LastHiddenLayerType, typename RecurrentLayerType>
        static void feedForward(PrevLayerType& prev, EmptyLayerChain& hiddenChain, EmptyLayerChain& recurrentChain, LastHiddenLayerType& lastHidden, RecurrentLayerType& recurrentLayer)
        {
            (void)hiddenChain;
            (void)recurrentChain;
            lastHidden.feedForward(prev, recurrentLayer);
        }
    };

    // Recursive case: feed forward inner LSTM layer with its recurrent connection
    template<typename HiddenLayerType, typename HiddenRestType, typename RecurrentLayerType, typename RecurrentRestType>
    struct ChainRecurrentFeedForwardHelper<LayerChain<HiddenLayerType, HiddenRestType>, LayerChain<RecurrentLayerType, RecurrentRestType> >
    {
        template<typename PrevLayerType, typename LastHiddenLayerType, typename LastRecurrentLayerType>
        static void feedForward(PrevLayerType& prev, LayerChain<HiddenLayerType, HiddenRestType>& hiddenChain, LayerChain<RecurrentLayerType, RecurrentRestType>& recurrentChain, LastHiddenLayerType& lastHidden, LastRecurrentLayerType& lastRecurrentLayer)
        {
            hiddenChain.layer.feedForward(prev, recurrentChain.layer);
            ChainRecurrentFeedForwardHelper<HiddenRestType, RecurrentRestType>::feedForward(hiddenChain.layer, hiddenChain.rest, recurrentChain.rest, lastHidden, lastRecurrentLayer);
        }
    };

    // Mixed case: inner hidden chain but no recurrent chain (non-recurrent inner layers)
    template<typename HiddenLayerType, typename HiddenRestType>
    struct ChainRecurrentFeedForwardHelper<LayerChain<HiddenLayerType, HiddenRestType>, EmptyLayerChain>
    {
        template<typename PrevLayerType, typename LastHiddenLayerType, typename RecurrentLayerType>
        static void feedForward(PrevLayerType& prev, LayerChain<HiddenLayerType, HiddenRestType>& hiddenChain, EmptyLayerChain& recurrentChain, LastHiddenLayerType& lastHidden, RecurrentLayerType& recurrentLayer)
        {
            (void)recurrentChain;
            hiddenChain.layer.feedForward(prev);
            ChainRecurrentFeedForwardHelper<HiddenRestType, EmptyLayerChain>::feedForward(hiddenChain.layer, hiddenChain.rest, recurrentChain, lastHidden, recurrentLayer);
        }
    };

    // =========================================================================
    // Compile-time dispatch for feed-forward: recurrent vs non-recurrent
    // =========================================================================

    template<bool HasRecurrentLayer>
    struct ChainFeedForwardDispatcher;

    template<>
    struct ChainFeedForwardDispatcher<false>
    {
        template<typename ChainType, typename RecurrentChainType, typename InputType, typename InnerChainType, typename InnerRecurrentChainType, typename LastHiddenType, typename RecurrentType>
        static void feedForward(InputType& input, InnerChainType& chain, InnerRecurrentChainType&, LastHiddenType& lastHidden, RecurrentType&)
        {
            ChainFeedForwardHelper<ChainType>::feedForward(input, chain, lastHidden);
        }
    };

    template<>
    struct ChainFeedForwardDispatcher<true>
    {
        template<typename ChainType, typename RecurrentChainType, typename InputType, typename InnerChainType, typename InnerRecurrentChainType, typename LastHiddenType, typename RecurrentType>
        static void feedForward(InputType& input, InnerChainType& chain, InnerRecurrentChainType& recurrentChain, LastHiddenType& lastHidden, RecurrentType& recurrentLayer)
        {
            ChainRecurrentFeedForwardHelper<ChainType, RecurrentChainType>::feedForward(input, chain, recurrentChain, lastHidden, recurrentLayer);
        }
    };

    // =========================================================================
    // Chain-based backward delta calculation
    // Processes from output toward input (right-to-left through the chain).
    // =========================================================================

    template<typename ChainType>
    struct ChainBackwardDeltaCalculator;

    template<>
    struct ChainBackwardDeltaCalculator<EmptyLayerChain>
    {
        template<typename NodeDeltasCalcType, typename NextLayerType>
        static void calculate(EmptyLayerChain& chain, NextLayerType& nextTowardOutput)
        {
            (void)chain;
            (void)nextTowardOutput;
        }
    };

    // Single-element chain
    template<typename LayerType>
    struct ChainBackwardDeltaCalculator<LayerChain<LayerType, EmptyLayerChain> >
    {
        template<typename NodeDeltasCalcType, typename NextLayerType>
        static void calculate(LayerChain<LayerType, EmptyLayerChain>& chain, NextLayerType& nextTowardOutput)
        {
            NodeDeltasCalcType::calculateAndSetNodeDeltas(chain.layer, nextTowardOutput);
        }

        static LayerType& getFirstLayer(LayerChain<LayerType, EmptyLayerChain>& chain)
        {
            return chain.layer;
        }
    };

    // Multi-element chain
    template<typename LayerType, typename RestChainType>
    struct ChainBackwardDeltaCalculator<LayerChain<LayerType, RestChainType> >
    {
        template<typename NodeDeltasCalcType, typename NextLayerType>
        static void calculate(LayerChain<LayerType, RestChainType>& chain, NextLayerType& nextTowardOutput)
        {
            // Process the rest of the chain first (closer to output)
            ChainBackwardDeltaCalculator<RestChainType>::template calculate<NodeDeltasCalcType>(chain.rest, nextTowardOutput);
            // Then calculate this layer's deltas against the first layer of the rest
            typedef typename ChainFirstLayer<RestChainType>::type RestFirstLayerType;
            RestFirstLayerType& firstOfRest = ChainFirstLayer<RestChainType>::get(chain.rest);
            NodeDeltasCalcType::calculateAndSetNodeDeltas(chain.layer, firstOfRest);
        }

        static LayerType& getFirstLayer(LayerChain<LayerType, RestChainType>& chain)
        {
            return chain.layer;
        }
    };

    // =========================================================================
    // Chain-based backward gradient calculation
    // Same right-to-left traversal pattern as delta calculation.
    // =========================================================================

    template<typename ChainType>
    struct ChainBackwardGradientCalculator;

    template<>
    struct ChainBackwardGradientCalculator<EmptyLayerChain>
    {
        template<typename GradCalcType, typename GradientsManagerType, typename NextLayerType>
        static void calculate(EmptyLayerChain& chain, NextLayerType& nextTowardOutput, GradientsManagerType& gm)
        {
            (void)chain;
            (void)nextTowardOutput;
            (void)gm;
        }
    };

    template<typename LayerType>
    struct ChainBackwardGradientCalculator<LayerChain<LayerType, EmptyLayerChain> >
    {
        template<typename GradCalcType, typename GradientsManagerType, typename NextLayerType>
        static void calculate(LayerChain<LayerType, EmptyLayerChain>& chain, NextLayerType& nextTowardOutput, GradientsManagerType& gm)
        {
            GradCalcType::calculateAndUpdateGradients(chain.layer, nextTowardOutput, gm);
        }

        static LayerType& getFirstLayer(LayerChain<LayerType, EmptyLayerChain>& chain)
        {
            return chain.layer;
        }
    };

    template<typename LayerType, typename RestChainType>
    struct ChainBackwardGradientCalculator<LayerChain<LayerType, RestChainType> >
    {
        template<typename GradCalcType, typename GradientsManagerType, typename NextLayerType>
        static void calculate(LayerChain<LayerType, RestChainType>& chain, NextLayerType& nextTowardOutput, GradientsManagerType& gm)
        {
            ChainBackwardGradientCalculator<RestChainType>::template calculate<GradCalcType>(chain.rest, nextTowardOutput, gm);
            typedef typename ChainFirstLayer<RestChainType>::type RestFirstLayerType;
            RestFirstLayerType& firstOfRest = ChainFirstLayer<RestChainType>::get(chain.rest);
            GradCalcType::calculateAndUpdateGradients(chain.layer, firstOfRest, gm);
        }

        static LayerType& getFirstLayer(LayerChain<LayerType, RestChainType>& chain)
        {
            return chain.layer;
        }
    };

    // =========================================================================
    // Chain-based backward weight update
    // =========================================================================

    template<typename ChainType>
    struct ChainBackwardWeightUpdater;

    template<>
    struct ChainBackwardWeightUpdater<EmptyLayerChain>
    {
        template<typename TrainingPolicyType, typename NextLayerType>
        static void update(TrainingPolicyType& tp, EmptyLayerChain& chain, NextLayerType& nextTowardOutput)
        {
            (void)tp;
            (void)chain;
            (void)nextTowardOutput;
        }
    };

    template<typename LayerType>
    struct ChainBackwardWeightUpdater<LayerChain<LayerType, EmptyLayerChain> >
    {
        template<typename TrainingPolicyType, typename NextLayerType>
        static void update(TrainingPolicyType& tp, LayerChain<LayerType, EmptyLayerChain>& chain, NextLayerType& nextTowardOutput)
        {
            tp.updateConnectionWeights(nextTowardOutput, chain.layer);
        }

        static LayerType& getFirstLayer(LayerChain<LayerType, EmptyLayerChain>& chain)
        {
            return chain.layer;
        }
    };

    template<typename LayerType, typename RestChainType>
    struct ChainBackwardWeightUpdater<LayerChain<LayerType, RestChainType> >
    {
        template<typename TrainingPolicyType, typename NextLayerType>
        static void update(TrainingPolicyType& tp, LayerChain<LayerType, RestChainType>& chain, NextLayerType& nextTowardOutput)
        {
            ChainBackwardWeightUpdater<RestChainType>::update(tp, chain.rest, nextTowardOutput);
            typedef typename ChainFirstLayer<RestChainType>::type RestFirstLayerType;
            RestFirstLayerType& firstOfRest = ChainFirstLayer<RestChainType>::get(chain.rest);
            tp.updateConnectionWeights(firstOfRest, chain.layer);
        }

        static LayerType& getFirstLayer(LayerChain<LayerType, RestChainType>& chain)
        {
            return chain.layer;
        }
    };

    // =========================================================================
    // Input layer connectors: handle the input-to-first-hidden connection
    // for delta, gradient, and weight-update passes.
    // =========================================================================

    template<typename InnerChainType>
    struct InputLayerDeltaConnector;

    template<>
    struct InputLayerDeltaConnector<EmptyLayerChain>
    {
        template<typename NodeDeltasCalcType, typename InputLayerType, typename LastHiddenLayerType>
        static void connect(InputLayerType& input, EmptyLayerChain& chain, LastHiddenLayerType& lastHidden)
        {
            (void)chain;
            NodeDeltasCalcType::calculateAndSetNodeDeltas(input, lastHidden);
        }
    };

    template<typename LayerType, typename RestChainType>
    struct InputLayerDeltaConnector<LayerChain<LayerType, RestChainType> >
    {
        template<typename NodeDeltasCalcType, typename InputLayerType, typename LastHiddenLayerType>
        static void connect(InputLayerType& input, LayerChain<LayerType, RestChainType>& chain, LastHiddenLayerType& lastHidden)
        {
            (void)lastHidden;
            NodeDeltasCalcType::calculateAndSetNodeDeltas(input, chain.layer);
        }
    };

    template<typename InnerChainType>
    struct InputLayerGradientConnector;

    template<>
    struct InputLayerGradientConnector<EmptyLayerChain>
    {
        template<typename GradCalcType, typename GradientsManagerType, typename InputLayerType, typename LastHiddenLayerType>
        static void connect(InputLayerType& input, EmptyLayerChain& chain, LastHiddenLayerType& lastHidden, GradientsManagerType& gm)
        {
            (void)chain;
            GradCalcType::calculateAndUpdateGradients(input, lastHidden, gm);
        }
    };

    template<typename LayerType, typename RestChainType>
    struct InputLayerGradientConnector<LayerChain<LayerType, RestChainType> >
    {
        template<typename GradCalcType, typename GradientsManagerType, typename InputLayerType, typename LastHiddenLayerType>
        static void connect(InputLayerType& input, LayerChain<LayerType, RestChainType>& chain, LastHiddenLayerType& lastHidden, GradientsManagerType& gm)
        {
            (void)lastHidden;
            GradCalcType::calculateAndUpdateGradients(input, chain.layer, gm);
        }
    };

    template<typename InnerChainType>
    struct InputLayerWeightUpdateConnector;

    template<>
    struct InputLayerWeightUpdateConnector<EmptyLayerChain>
    {
        template<typename TrainingPolicyType, typename InputLayerType, typename LastHiddenLayerType>
        static void connect(TrainingPolicyType& tp, InputLayerType& input, EmptyLayerChain& chain, LastHiddenLayerType& lastHidden)
        {
            (void)chain;
            tp.updateConnectionWeights(lastHidden, input);
        }
    };

    template<typename LayerType, typename RestChainType>
    struct InputLayerWeightUpdateConnector<LayerChain<LayerType, RestChainType> >
    {
        template<typename TrainingPolicyType, typename InputLayerType, typename LastHiddenLayerType>
        static void connect(TrainingPolicyType& tp, InputLayerType& input, LayerChain<LayerType, RestChainType>& chain, LastHiddenLayerType& lastHidden)
        {
            (void)lastHidden;
            tp.updateConnectionWeights(chain.layer, input);
        }
    };

    // =========================================================================
    // Network-level chain calculators
    // =========================================================================

    template<typename NeuralNetworkType>
    struct ChainNetworkDeltasCalculator
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerChainType InnerChainType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef NodeDeltasCalculator<TransferFunctionsPolicy> NodeDeltasCalculatorType;
        typedef typename OutputLayerNodeDeltasCalculatorChooser<
                        TransferFunctionsPolicy,
                        OutputLayerType,
                        NeuralNetworkType::NeuralNetworkOutputLayerConfiguration>::OutputLayerNodeDeltasCalculatorType OutputLayerNodeDeltasCalculatorType;

        static void calculateNetworkDeltas(NeuralNetworkType& nn, ValueType const* const targetValues)
        {
            InputLayerType& inputLayer = nn.getInputLayer();
            InnerChainType& innerChain = nn.getInnerHiddenLayerChain();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            OutputLayerType& outputLayer = nn.getOutputLayer();

            OutputLayerNodeDeltasCalculatorType::calculateOutputLayerNodeDeltas(outputLayer, targetValues);

            NodeDeltasCalculatorType::calculateAndSetNodeDeltas(lastHiddenLayer, outputLayer);

            ChainBackwardDeltaCalculator<InnerChainType>::template calculate<NodeDeltasCalculatorType>(innerChain, lastHiddenLayer);

            InputLayerDeltaConnector<InnerChainType>::template connect<NodeDeltasCalculatorType>(inputLayer, innerChain, lastHiddenLayer);
        }
    };

    template<typename NeuralNetworkType>
    struct ChainNetworkGradientsCalculator
    {
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerChainType InnerChainType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef typename NeuralNetworkType::GradientsManagerType GradientsManagerType;
        typedef GradientsCalculator<TransferFunctionsPolicy, GradientsManagerType> GradientsCalculatorType;

        static void calculateNetworkGradients(NeuralNetworkType& nn)
        {
            InputLayerType& inputLayer = nn.getInputLayer();
            InnerChainType& innerChain = nn.getInnerHiddenLayerChain();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            OutputLayerType& outputLayer = nn.getOutputLayer();
            GradientsManagerType& gradientsManager = nn.getGradientsManager();

            GradientsCalculatorType::calculateAndUpdateOutputLayerGradients(lastHiddenLayer, outputLayer, gradientsManager);

            ChainBackwardGradientCalculator<InnerChainType>::template calculate<GradientsCalculatorType>(innerChain, lastHiddenLayer, gradientsManager);

            InputLayerGradientConnector<InnerChainType>::template connect<GradientsCalculatorType>(inputLayer, innerChain, lastHiddenLayer, gradientsManager);
        }
    };

    template<typename NeuralNetworkType>
    struct ChainBackPropConnectionWeightUpdater
    {
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerChainType InnerChainType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;

        template<typename TrainingPolicyType>
        static void updateConnectionWeights(TrainingPolicyType& trainingPolicy, NeuralNetworkType& nn)
        {
            OutputLayerType& outputLayer = nn.getOutputLayer();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            InnerChainType& innerChain = nn.getInnerHiddenLayerChain();
            InputLayerType& inputLayer = nn.getInputLayer();

            trainingPolicy.updateConnectionWeights(outputLayer, lastHiddenLayer);

            ChainBackwardWeightUpdater<InnerChainType>::update(trainingPolicy, innerChain, lastHiddenLayer);

            InputLayerWeightUpdateConnector<InnerChainType>::connect(trainingPolicy, inputLayer, innerChain, lastHiddenLayer);
        }
    };

    // =========================================================================
    // Chain-based recurrent training calculators
    // These mirror the chain-based non-recurrent calculators but include
    // recurrent layer processing for backpropagation through time.
    // =========================================================================

    template<typename NeuralNetworkType>
    struct ChainRecurrentNetworkDeltasCalculator
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerChainType InnerChainType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkRecurrentLayerType RecurrentLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef NodeDeltasCalculator<TransferFunctionsPolicy> NodeDeltasCalculatorType;
        typedef typename OutputLayerNodeDeltasCalculatorChooser<
                        TransferFunctionsPolicy,
                        OutputLayerType,
                        NeuralNetworkType::NeuralNetworkOutputLayerConfiguration>::OutputLayerNodeDeltasCalculatorType OutputLayerNodeDeltasCalculatorType;

        static void calculateNetworkDeltas(NeuralNetworkType& nn, ValueType const* const targetValues)
        {
            InputLayerType& inputLayer = nn.getInputLayer();
            InnerChainType& innerChain = nn.getInnerHiddenLayerChain();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            RecurrentLayerType& recurrentLayer = nn.getRecurrentLayer();
            OutputLayerType& outputLayer = nn.getOutputLayer();

            OutputLayerNodeDeltasCalculatorType::calculateOutputLayerNodeDeltas(outputLayer, targetValues);

            NodeDeltasCalculatorType::calculateAndSetNodeDeltas(lastHiddenLayer, outputLayer);

            NodeDeltasCalculatorType::calculateAndSetNodeDeltas(recurrentLayer, lastHiddenLayer);

            ChainBackwardDeltaCalculator<InnerChainType>::template calculate<NodeDeltasCalculatorType>(innerChain, lastHiddenLayer);

            InputLayerDeltaConnector<InnerChainType>::template connect<NodeDeltasCalculatorType>(inputLayer, innerChain, lastHiddenLayer);
        }
    };

    // Compile-time dispatcher: use gated gradient calculation when GateMultiplier > 1
    template<size_t GateMultiplier>
    struct GatedGradientDispatcher
    {
        template<typename GradCalcType, typename LayerType, typename HiddenLayerType, typename GradientsManagerType>
        static void calculate(LayerType& layer, const HiddenLayerType& hiddenLayer, GradientsManagerType& gradientsManager)
        {
            GradCalcType::template calculateAndUpdateGradientsGated<LayerType, HiddenLayerType, GateMultiplier>(layer, hiddenLayer, gradientsManager);
        }
    };

    template<>
    struct GatedGradientDispatcher<1>
    {
        template<typename GradCalcType, typename LayerType, typename HiddenLayerType, typename GradientsManagerType>
        static void calculate(LayerType& layer, const HiddenLayerType& hiddenLayer, GradientsManagerType& gradientsManager)
        {
            GradCalcType::calculateAndUpdateGradients(layer, hiddenLayer, gradientsManager);
        }
    };

    // Compile-time dispatcher: use gated weight update when GateMultiplier > 1
    template<size_t GateMultiplier>
    struct GatedWeightUpdateDispatcher
    {
        template<typename TrainingPolicyType, typename LayerType, typename PreviousLayerType>
        static void update(TrainingPolicyType& trainingPolicy, LayerType& layer, PreviousLayerType& previousLayer)
        {
            trainingPolicy.template updateConnectionWeightsGated<LayerType, PreviousLayerType, GateMultiplier>(layer, previousLayer);
        }
    };

    template<>
    struct GatedWeightUpdateDispatcher<1>
    {
        template<typename TrainingPolicyType, typename LayerType, typename PreviousLayerType>
        static void update(TrainingPolicyType& trainingPolicy, LayerType& layer, PreviousLayerType& previousLayer)
        {
            trainingPolicy.updateConnectionWeights(layer, previousLayer);
        }
    };

    // Gated-aware input layer gradient connector
    template<typename InnerChainType, size_t GateMultiplier>
    struct GatedInputLayerGradientConnector;

    template<size_t GateMultiplier>
    struct GatedInputLayerGradientConnector<EmptyLayerChain, GateMultiplier>
    {
        template<typename GradCalcType, typename GradientsManagerType, typename InputLayerType, typename LastHiddenLayerType>
        static void connect(InputLayerType& input, EmptyLayerChain& chain, LastHiddenLayerType& lastHidden, GradientsManagerType& gm)
        {
            (void)chain;
            GatedGradientDispatcher<GateMultiplier>::template calculate<GradCalcType>(input, lastHidden, gm);
        }
    };

    template<typename LayerType, typename RestChainType, size_t GateMultiplier>
    struct GatedInputLayerGradientConnector<LayerChain<LayerType, RestChainType>, GateMultiplier>
    {
        template<typename GradCalcType, typename GradientsManagerType, typename InputLayerType, typename LastHiddenLayerType>
        static void connect(InputLayerType& input, LayerChain<LayerType, RestChainType>& chain, LastHiddenLayerType& lastHidden, GradientsManagerType& gm)
        {
            (void)lastHidden;
            GatedGradientDispatcher<GateMultiplier>::template calculate<GradCalcType>(input, chain.layer, gm);
        }
    };

    // Gated-aware input layer weight update connector
    template<typename InnerChainType, size_t GateMultiplier>
    struct GatedInputLayerWeightUpdateConnector;

    template<size_t GateMultiplier>
    struct GatedInputLayerWeightUpdateConnector<EmptyLayerChain, GateMultiplier>
    {
        template<typename TrainingPolicyType, typename InputLayerType, typename LastHiddenLayerType>
        static void connect(TrainingPolicyType& tp, InputLayerType& input, EmptyLayerChain& chain, LastHiddenLayerType& lastHidden)
        {
            (void)chain;
            GatedWeightUpdateDispatcher<GateMultiplier>::update(tp, lastHidden, input);
        }
    };

    template<typename LayerType, typename RestChainType, size_t GateMultiplier>
    struct GatedInputLayerWeightUpdateConnector<LayerChain<LayerType, RestChainType>, GateMultiplier>
    {
        template<typename TrainingPolicyType, typename InputLayerType, typename LastHiddenLayerType>
        static void connect(TrainingPolicyType& tp, InputLayerType& input, LayerChain<LayerType, RestChainType>& chain, LastHiddenLayerType& lastHidden)
        {
            (void)lastHidden;
            GatedWeightUpdateDispatcher<GateMultiplier>::update(tp, chain.layer, input);
        }
    };

    template<typename NeuralNetworkType>
    struct ChainRecurrentNetworkGradientsCalculator
    {
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerChainType InnerChainType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkRecurrentLayerType RecurrentLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy TransferFunctionsPolicy;
        typedef typename NeuralNetworkType::GradientsManagerType GradientsManagerType;
        typedef GradientsCalculator<TransferFunctionsPolicy, GradientsManagerType> GradientsCalculatorType;

        static const size_t GateMultiplier = NeuralNetworkType::HiddenLayerGateMultiplier;

        static void calculateNetworkGradients(NeuralNetworkType& nn)
        {
            InputLayerType& inputLayer = nn.getInputLayer();
            InnerChainType& innerChain = nn.getInnerHiddenLayerChain();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            RecurrentLayerType& recurrentLayer = nn.getRecurrentLayer();
            OutputLayerType& outputLayer = nn.getOutputLayer();
            GradientsManagerType& gradientsManager = nn.getGradientsManager();

            GradientsCalculatorType::calculateAndUpdateOutputLayerGradients(lastHiddenLayer, outputLayer, gradientsManager);

            GatedGradientDispatcher<GateMultiplier>::template calculate<GradientsCalculatorType>(recurrentLayer, lastHiddenLayer, gradientsManager);

            ChainBackwardGradientCalculator<InnerChainType>::template calculate<GradientsCalculatorType>(innerChain, lastHiddenLayer, gradientsManager);

            GatedInputLayerGradientConnector<InnerChainType, GateMultiplier>::template connect<GradientsCalculatorType>(inputLayer, innerChain, lastHiddenLayer, gradientsManager);
        }
    };

    template<typename NeuralNetworkType>
    struct ChainRecurrentBackPropConnectionWeightUpdater
    {
        typedef typename NeuralNetworkType::NeuralNetworkOutputLayerType OutputLayerType;
        typedef typename NeuralNetworkType::LastHiddenLayerType LastHiddenLayerType;
        typedef typename NeuralNetworkType::NeuralNetworkRecurrentLayerType RecurrentLayerType;
        typedef typename NeuralNetworkType::InnerHiddenLayerChainType InnerChainType;
        typedef typename NeuralNetworkType::InputLayerType InputLayerType;

        static const size_t GateMultiplier = NeuralNetworkType::HiddenLayerGateMultiplier;

        template<typename TrainingPolicyType>
        static void updateConnectionWeights(TrainingPolicyType& trainingPolicy, NeuralNetworkType& nn)
        {
            OutputLayerType& outputLayer = nn.getOutputLayer();
            LastHiddenLayerType& lastHiddenLayer = nn.getLastHiddenLayer();
            RecurrentLayerType& recurrentLayer = nn.getRecurrentLayer();
            InnerChainType& innerChain = nn.getInnerHiddenLayerChain();
            InputLayerType& inputLayer = nn.getInputLayer();

            trainingPolicy.updateConnectionWeights(outputLayer, lastHiddenLayer);

            GatedWeightUpdateDispatcher<GateMultiplier>::update(trainingPolicy, lastHiddenLayer, recurrentLayer);

            ChainBackwardWeightUpdater<InnerChainType>::update(trainingPolicy, innerChain, lastHiddenLayer);

            GatedInputLayerWeightUpdateConnector<InnerChainType, GateMultiplier>::connect(trainingPolicy, inputLayer, innerChain, lastHiddenLayer);
        }
    };

    // =========================================================================
    // Chain-based gradients manager
    // =========================================================================

    template<typename NeuralNetworkType, bool IsTrainable>
    struct ChainGradientsManagerSelector;

    template<typename NeuralNetworkType>
    struct ChainGradientsManagerSelector<NeuralNetworkType, true>
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        static const size_t NumberOfGradients = NeuralNetworkType::TotalNumberOfGradients;

        struct type
        {
            template<typename LayerType>
            void updateBiasGradients(LayerType& layer, const size_t nextNeuron, const ValueType& gradient)
            {
                this->gradientsHolder.updateBiasGradients(layer, nextNeuron, gradient);
            }

            template<typename LayerType>
            void updateGradients(LayerType& layer, const size_t neuron, const size_t nextNeuron, const ValueType& gradient)
            {
                this->gradientsHolder.updateGradients(layer, neuron, nextNeuron, gradient);
            }
        private:
            GradientsHolder<ValueType, NumberOfGradients, NeuralNetworkType::NeuralNetworkBatchSize> gradientsHolder;
        };
    };

    template<typename NeuralNetworkType>
    struct ChainGradientsManagerSelector<NeuralNetworkType, false>
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;

        struct type
        {
            template<typename LayerType>
            void updateBiasGradients(LayerType& layer, const size_t nextNeuron, const ValueType& gradient)
            {
                (void)layer;
                (void)nextNeuron;
                (void)gradient;
            }

            template<typename LayerType>
            void updateGradients(LayerType& layer, const size_t neuron, const size_t nextNeuron, const ValueType& gradient)
            {
                (void)layer;
                (void)neuron;
                (void)nextNeuron;
                (void)gradient;
            }
        };
    };

    // =========================================================================
    // Chain-based training policy
    // =========================================================================

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ChainBackPropagationPolicy : public BackPropagationParent<TransferFunctionsPolicy, BatchSize>
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        template<typename NNType>
        void trainNetwork(NNType& nn, ValueType const* const targetValues)
        {
            ChainNetworkDeltasCalculator<NNType>::calculateNetworkDeltas(nn, targetValues);

            ChainNetworkGradientsCalculator<NNType>::calculateNetworkGradients(nn);

            ChainBackPropConnectionWeightUpdater<NNType>::updateConnectionWeights(*this, nn);

            this->stepLearningRate();
        }
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ChainClassifierBackPropagationPolicy : public BackPropagationParent<TransferFunctionsPolicy, BatchSize>
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        template<typename NNType>
        void trainNetwork(NNType& nn, ValueType const* const targetValues)
        {
            ChainNetworkDeltasCalculator<NNType>::calculateNetworkDeltas(nn, targetValues);

            ChainNetworkGradientsCalculator<NNType>::calculateNetworkGradients(nn);

            ChainBackPropConnectionWeightUpdater<NNType>::updateConnectionWeights(*this, nn);

            this->stepLearningRate();
        }
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ChainBackPropagationThruTimePolicy : public BackPropagationParent<TransferFunctionsPolicy, BatchSize>
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        template<typename NNType>
        void trainNetwork(NNType& nn, ValueType const* const targetValues)
        {
            ChainRecurrentNetworkDeltasCalculator<NNType>::calculateNetworkDeltas(nn, targetValues);

            ChainRecurrentNetworkGradientsCalculator<NNType>::calculateNetworkGradients(nn);

            ChainRecurrentBackPropConnectionWeightUpdater<NNType>::updateConnectionWeights(*this, nn);

            this->stepLearningRate();
        }
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ChainClassifierBackPropagationThruTimePolicy : public BackPropagationParent<TransferFunctionsPolicy, BatchSize>
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        template<typename NNType>
        void trainNetwork(NNType& nn, ValueType const* const targetValues)
        {
            ChainRecurrentNetworkDeltasCalculator<NNType>::calculateNetworkDeltas(nn, targetValues);

            ChainRecurrentNetworkGradientsCalculator<NNType>::calculateNetworkGradients(nn);

            ChainRecurrentBackPropConnectionWeightUpdater<NNType>::updateConnectionWeights(*this, nn);

            this->stepLearningRate();
        }
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize, bool HasRecurrentLayer, bool IsTrainable, outputLayerConfiguration_e OutputLayerConfiguration>
    struct ChainTrainingPolicySelector
    {
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ChainTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, false, true, FeedForwardOutputLayerConfiguration>
    {
        typedef ChainBackPropagationPolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ChainTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, false, true, ClassifierOutputLayerConfiguration>
    {
        typedef ChainClassifierBackPropagationPolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ChainTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, true, true, FeedForwardOutputLayerConfiguration>
    {
        typedef ChainBackPropagationThruTimePolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ChainTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, true, true, ClassifierOutputLayerConfiguration>
    {
        typedef ChainClassifierBackPropagationThruTimePolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ChainTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, false, false, FeedForwardOutputLayerConfiguration>
    {
        typedef NullTrainingPolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ChainTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, false, false, ClassifierOutputLayerConfiguration>
    {
        typedef NullTrainingPolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ChainTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, true, false, FeedForwardOutputLayerConfiguration>
    {
        typedef NullTrainingPolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize>
    struct ChainTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, true, false, ClassifierOutputLayerConfiguration>
    {
        typedef NullTrainingPolicy<TransferFunctionsPolicy, BatchSize> TrainingPolicyType;
    };

    // =========================================================================
    // Chain-based hidden layer accessor
    // Provides runtime-indexed access to hidden layers stored in a chain.
    // Used by NeuralNetwork to implement layer-indexed weight get/set methods.
    // =========================================================================

    template<typename ChainType>
    struct ChainHiddenLayerAccessor;

    template<>
    struct ChainHiddenLayerAccessor<EmptyLayerChain>
    {
        template<typename ValueType>
        static ValueType getBiasNeuronWeightForConnection(EmptyLayerChain&, size_t, size_t)
        {
            return ValueType();
        }

        template<typename ValueType>
        static ValueType getWeightForNeuronAndConnection(EmptyLayerChain&, size_t, size_t, size_t)
        {
            return ValueType();
        }

        template<typename ValueType>
        static void setBiasNeuronWeightForConnection(EmptyLayerChain&, size_t, size_t, const ValueType&)
        {
        }

        template<typename ValueType>
        static void setBiasNeuronDeltaWeightForConnection(EmptyLayerChain&, size_t, size_t, const ValueType&)
        {
        }

        template<typename ValueType>
        static void setWeightForNeuronAndConnection(EmptyLayerChain&, size_t, size_t, size_t, const ValueType&)
        {
        }

        template<typename ValueType>
        static void setDeltaWeightForNeuronAndConnection(EmptyLayerChain&, size_t, size_t, size_t, const ValueType&)
        {
        }
    };

    template<typename LayerType, typename RestChainType>
    struct ChainHiddenLayerAccessor<LayerChain<LayerType, RestChainType> >
    {
        typedef typename LayerType::ValueType ValueType;

        template<typename VT = ValueType>
        static VT getBiasNeuronWeightForConnection(LayerChain<LayerType, RestChainType>& chain, size_t layerIndex, size_t connection)
        {
            if(layerIndex == 0)
            {
                return chain.layer.getBiasNeuronWeightForConnection(connection);
            }
            return ChainHiddenLayerAccessor<RestChainType>::template getBiasNeuronWeightForConnection<ValueType>(chain.rest, layerIndex - 1, connection);
        }

        template<typename VT = ValueType>
        static VT getWeightForNeuronAndConnection(LayerChain<LayerType, RestChainType>& chain, size_t layerIndex, size_t neuron, size_t connection)
        {
            if(layerIndex == 0)
            {
                return chain.layer.getWeightForNeuronAndConnection(neuron, connection);
            }
            return ChainHiddenLayerAccessor<RestChainType>::template getWeightForNeuronAndConnection<ValueType>(chain.rest, layerIndex - 1, neuron, connection);
        }

        static void setBiasNeuronWeightForConnection(LayerChain<LayerType, RestChainType>& chain, size_t layerIndex, size_t connection, const ValueType& weight)
        {
            if(layerIndex == 0)
            {
                chain.layer.setBiasNeuronWeightForConnection(connection, weight);
                return;
            }
            ChainHiddenLayerAccessor<RestChainType>::template setBiasNeuronWeightForConnection<ValueType>(chain.rest, layerIndex - 1, connection, weight);
        }

        static void setBiasNeuronDeltaWeightForConnection(LayerChain<LayerType, RestChainType>& chain, size_t layerIndex, size_t connection, const ValueType& deltaWeight)
        {
            if(layerIndex == 0)
            {
                chain.layer.setBiasNeuronDeltaWeightForConnection(connection, deltaWeight);
                return;
            }
            ChainHiddenLayerAccessor<RestChainType>::template setBiasNeuronDeltaWeightForConnection<ValueType>(chain.rest, layerIndex - 1, connection, deltaWeight);
        }

        static void setWeightForNeuronAndConnection(LayerChain<LayerType, RestChainType>& chain, size_t layerIndex, size_t neuron, size_t connection, const ValueType& weight)
        {
            if(layerIndex == 0)
            {
                chain.layer.setWeightForNeuronAndConnection(neuron, connection, weight);
                return;
            }
            ChainHiddenLayerAccessor<RestChainType>::template setWeightForNeuronAndConnection<ValueType>(chain.rest, layerIndex - 1, neuron, connection, weight);
        }

        static void setDeltaWeightForNeuronAndConnection(LayerChain<LayerType, RestChainType>& chain, size_t layerIndex, size_t neuron, size_t connection, const ValueType& deltaWeight)
        {
            if(layerIndex == 0)
            {
                chain.layer.setDeltaWeightForNeuronAndConnection(neuron, connection, deltaWeight);
                return;
            }
            ChainHiddenLayerAccessor<RestChainType>::template setDeltaWeightForNeuronAndConnection<ValueType>(chain.rest, layerIndex - 1, neuron, connection, deltaWeight);
        }
    };

    // =========================================================================
    // Hidden layer size accessor
    // Provides runtime-indexed access to hidden layer sizes from
    // a HiddenLayers descriptor. Used by setWeights to determine
    // iteration bounds for each layer.
    // =========================================================================

    template<typename HiddenLayersDescriptorType>
    struct HiddenLayerSizeAccessor;

    template<size_t S>
    struct HiddenLayerSizeAccessor<HiddenLayers<S> >
    {
        static size_t getSize(size_t)
        {
            return S;
        }
    };

    template<size_t S, size_t... Rest>
    struct HiddenLayerSizeAccessor<HiddenLayers<S, Rest...> >
    {
        static size_t getSize(size_t index)
        {
            if(index == 0)
            {
                return S;
            }
            return HiddenLayerSizeAccessor<HiddenLayers<Rest...> >::getSize(index - 1);
        }
    };

    // =========================================================================
    // NeuralNetwork: MLP with heterogeneous hidden layers
    //
    // Usage:
    //   // Different sizes per hidden layer
    //   NeuralNetwork<double, 2, HiddenLayers<10, 5, 3>, 1, TransferFunctions>
    //
    //   // Uniform hidden layers (equivalent to MultilayerPerceptron<..., 2, 5, ...>)
    //   NeuralNetwork<double, 2, UniformHiddenLayersAlias<2, 5>::type, 1, TransferFunctions>
    // =========================================================================

    template<
            typename ValueType,
            size_t NumberOfInputs,
            typename HiddenLayersDescriptor,
            size_t NumberOfOutputs,
            typename TransferFunctionsPolicy,
            bool IsTrainable = true,
            size_t BatchSize = 1,
            bool HasRecurrentLayer = false,
            hiddenLayerConfiguration_e HiddenLayerConfig = NonRecurrentHiddenLayerConfig,
            size_t RecurrentConnectionDepth = 0,
            outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration
            >
    class NeuralNetwork
    {
    public:
        typedef NeuralNetwork<  ValueType,
                                NumberOfInputs,
                                HiddenLayersDescriptor,
                                NumberOfOutputs,
                                TransferFunctionsPolicy,
                                IsTrainable,
                                BatchSize,
                                HasRecurrentLayer,
                                HiddenLayerConfig,
                                RecurrentConnectionDepth,
                                OutputLayerConfiguration> NeuralNetworkType;

        typedef ValueType NeuralNetworkValueType;
        typedef typename ConnectionTypeSelector<ValueType, IsTrainable>::ConnectionType ConnectionType;
        typedef TransferFunctionsPolicy NeuralNetworkTransferFunctionsPolicy;

        // Gate multiplier: 4 for LSTM (one weight set per gate), 1 for non-LSTM
        static const size_t HiddenLayerGateMultiplier = GateConnectionCount<1, HiddenLayerConfig>::value;

        // Input layer: neurons have outgoing connections to the first hidden layer
        // For LSTM networks, each hidden neuron needs 4 connections (one per gate)
        static const size_t InputToHiddenConnections = GateConnectionCount<HiddenLayersDescriptor::FirstLayerSize, HiddenLayerConfig>::value;
        typedef typename InputLayerNeuronTypeSelector<ConnectionType, InputToHiddenConnections, TransferFunctionsPolicy, IsTrainable>::InputLayerNeuronType InputLayerNeuronType;
        typedef InputLayer<InputLayerNeuronType, NumberOfInputs> InputLayerType;

        // Inner hidden layer chain (all hidden layers except the last)
        typedef typename InnerLayerChainBuilder<ConnectionType, TransferFunctionsPolicy, IsTrainable, HiddenLayerConfig, HiddenLayersDescriptor>::ChainType InnerHiddenLayerChainType;

        // Last hidden layer: neurons connect to the output layer
        typedef typename HiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutputs, TransferFunctionsPolicy, ConnectionType::IsTrainable, HiddenLayerConfig>::HiddenLayerNeuronType LastHiddenLayerNeuronType;
        typedef typename LastHiddenLayerTypeSelector<LastHiddenLayerNeuronType, HiddenLayersDescriptor::LastLayerSize, HiddenLayerConfig>::LastHiddenLayerType LastHiddenLayerType;

        // Recurrent layer for the last hidden layer
        // For LSTM, recurrent neurons need 4x connections (one per gate)
        static const size_t RecurrentToHiddenConnections = GateConnectionCount<HiddenLayersDescriptor::LastLayerSize, HiddenLayerConfig>::value;
        typedef RecurrentLayerTypeSelector< ConnectionType,
                                            HiddenLayersDescriptor::LastLayerSize,
                                            RecurrentToHiddenConnections,
                                            TransferFunctionsPolicy,
                                            RecurrentConnectionDepth,
                                            HasRecurrentLayer> RecurrentLayerTypeSelectorType;
        typedef typename RecurrentLayerTypeSelectorType::RecurrentLayerType NeuralNetworkRecurrentLayerType;

        // Recurrent layer chain for inner hidden layers (multi-layer LSTM)
        typedef typename RecurrentLayerChainBuilder<ConnectionType, TransferFunctionsPolicy, RecurrentConnectionDepth, HasRecurrentLayer, HiddenLayersDescriptor>::ChainType InnerRecurrentLayerChainType;

        // Output layer
        typedef typename OutputLayerNeuronTypeSelector<ConnectionType, TransferFunctionsPolicy, IsTrainable>::OutputLayerNeuronType OutputLayerNeuronType;
        typedef typename OutputLayerTypeSelector<OutputLayerNeuronType, NumberOfOutputs, OutputLayerConfiguration>::OutputLayerType NeuralNetworkOutputLayerType;

        // Gradient count: computed over the full chain Input, S0, S1, ..., Sk, Output
        static const size_t TotalNumberOfGradients = detail::TotalGradientsForNetwork<NumberOfInputs, NumberOfOutputs, HiddenLayersDescriptor>::value;

        // Gradients manager
        typedef typename ChainGradientsManagerSelector<NeuralNetworkType, IsTrainable>::type GradientsManagerType;

        // Training policy
        typedef typename ChainTrainingPolicySelector<
                            TransferFunctionsPolicy,
                            BatchSize,
                            HasRecurrentLayer,
                            IsTrainable,
                            OutputLayerConfiguration>::TrainingPolicyType TrainingPolicyType;

        static const size_t NeuralNetworkNumberOfHiddenLayers = HiddenLayersDescriptor::Count;
        static const size_t NumberOfInnerHiddenLayers = HiddenLayersDescriptor::Count - 1;
        static const size_t NumberOfInputLayerNeurons = InputLayerType::NumberOfNeuronsInLayer;
        static const size_t NumberOfHiddenLayerNeurons = LastHiddenLayerType::NumberOfNeuronsInLayer;
        static const size_t NumberOfOutputLayerNeurons = NeuralNetworkOutputLayerType::NumberOfNeuronsInLayer;
        static const size_t NeuralNetworkRecurrentConnectionDepth = NeuralNetworkRecurrentLayerType::RecurrentLayerRecurrentConnectionDepth;
        static const size_t NeuralNetworkBatchSize = BatchSize;
        static const outputLayerConfiguration_e NeuralNetworkOutputLayerConfiguration = OutputLayerConfiguration;

        NeuralNetwork()
        {
            size_t bufferIndex;

            this->mInputLayer.initializeNeurons();

            this->mInnerHiddenLayerChain.initializeNeurons();

            this->mLastHiddenLayer.initializeNeurons();

            this->mOutputLayer.initializeNeurons();

            this->mRecurrentLayer.initializeNeurons();

            this->mTrainingPolicy.initialize();

            if(IsTrainable)
            {
                this->initializeWeights();
            }

            for(size_t learnedValue = 0;learnedValue < NumberOfOutputLayerNeurons;++learnedValue)
            {
                bufferIndex = learnedValue * sizeof(ValueType);
                new (&this->mLearnedValuesBuffer[bufferIndex]) ValueType();
            }
        }

        void initializeWeights()
        {
            this->mInputLayer.initializeWeights();

            this->mInnerHiddenLayerChain.initializeWeights();

            this->mLastHiddenLayer.initializeWeights();

            this->mOutputLayer.initializeWeights();

            this->mRecurrentLayer.initializeWeights();
        }

        ValueType calculateError(ValueType const* const targetValues)
        {
            ValueType* pLearnedValues = reinterpret_cast<ValueType*>(&this->mLearnedValuesBuffer[0]);

            getLearnedValues(pLearnedValues);

            return TransferFunctionsPolicy::calculateError(targetValues, pLearnedValues);
        }

        void feedForward(ValueType const* const values)
        {
            this->mInputLayer.feedForward(values);

            ChainFeedForwardDispatcher<HasRecurrentLayer>::template feedForward<InnerHiddenLayerChainType, InnerRecurrentLayerChainType>(this->mInputLayer, this->mInnerHiddenLayerChain, this->mInnerRecurrentLayerChain, this->mLastHiddenLayer, this->mRecurrentLayer);

            this->mOutputLayer.feedForward(this->mLastHiddenLayer);
        }

        ValueType getAccelerationRate() const
        {
            return this->mTrainingPolicy.getAccelerationRate();
        }

        ValueType getLearningRate() const
        {
            return this->mTrainingPolicy.getLearningRate();
        }

        ValueType getMomentumRate() const
        {
            return this->mTrainingPolicy.getMomentumRate();
        }

        NeuralNetworkRecurrentLayerType& getRecurrentLayer()
        {
            return this->mRecurrentLayer;
        }

        GradientsManagerType& getGradientsManager()
        {
            return this->mGradientsManager;
        }

        ValueType getHiddenLayerBiasNeuronWeightForConnection(const size_t hiddenLayer, const size_t connection)
        {
            if((NeuralNetworkNumberOfHiddenLayers - 1) == hiddenLayer)
            {
                return this->mLastHiddenLayer.getBiasNeuronWeightForConnection(connection);
            }
            else
            {
                return ChainHiddenLayerAccessor<InnerHiddenLayerChainType>::template getBiasNeuronWeightForConnection<ValueType>(this->mInnerHiddenLayerChain, hiddenLayer, connection);
            }
        }

        ValueType getHiddenLayerWeightForNeuronAndConnection(const size_t hiddenLayer, const size_t neuron, const size_t connection)
        {
            if((NeuralNetworkNumberOfHiddenLayers - 1) == hiddenLayer)
            {
                return this->mLastHiddenLayer.getWeightForNeuronAndConnection(neuron, connection);
            }
            else
            {
                return ChainHiddenLayerAccessor<InnerHiddenLayerChainType>::template getWeightForNeuronAndConnection<ValueType>(this->mInnerHiddenLayerChain, hiddenLayer, neuron, connection);
            }
        }

        InputLayerType& getInputLayer()
        {
            return this->mInputLayer;
        }

        InnerHiddenLayerChainType& getInnerHiddenLayerChain()
        {
            return this->mInnerHiddenLayerChain;
        }

        LastHiddenLayerType& getLastHiddenLayer()
        {
            return this->mLastHiddenLayer;
        }

        void getLearnedValues(ValueType* output) const
        {
            for (size_t outputNeuron = 0; outputNeuron < NumberOfOutputLayerNeurons; ++outputNeuron)
            {
                output[outputNeuron] = mOutputLayer.getOutputValueForNeuron(outputNeuron);
            }
        }

        NeuralNetworkOutputLayerType& getOutputLayer()
        {
            return this->mOutputLayer;
        }

        ValueType getInputLayerBiasNeuronWeightForConnection(const size_t connection) const
        {
            return this->mInputLayer.getBiasNeuronWeightForConnection(connection);
        }

        ValueType getInputLayerWeightForNeuronAndConnection(const size_t neuron, const size_t connection) const
        {
            return this->mInputLayer.getWeightForNeuronAndConnection(neuron, connection);
        }

        void setAccelerationRate(const ValueType& value)
        {
            this->mTrainingPolicy.setAccelerationRate(value);
        }

        void setLearningRate(const ValueType& value)
        {
            this->mTrainingPolicy.setLearningRate(value);
        }

        void setMomentumRate(const ValueType& value)
        {
            this->mTrainingPolicy.setMomentumRate(value);
        }

        void setHiddenLayerBiasNeuronWeightForConnection(const size_t hiddenLayer, const size_t connection, const ValueType& weight)
        {
            if((NeuralNetworkNumberOfHiddenLayers - 1) == hiddenLayer)
            {
                this->mLastHiddenLayer.setBiasNeuronWeightForConnection(connection, weight);
            }
            else
            {
                ChainHiddenLayerAccessor<InnerHiddenLayerChainType>::setBiasNeuronWeightForConnection(this->mInnerHiddenLayerChain, hiddenLayer, connection, weight);
            }
        }

        void setHiddenLayerBiasDeltaWeightForConnection(const size_t hiddenLayer, const size_t connection, const ValueType& deltaWeight)
        {
            if((NeuralNetworkNumberOfHiddenLayers - 1) == hiddenLayer)
            {
                this->mLastHiddenLayer.setBiasNeuronDeltaWeightForConnection(connection, deltaWeight);
            }
            else
            {
                ChainHiddenLayerAccessor<InnerHiddenLayerChainType>::setBiasNeuronDeltaWeightForConnection(this->mInnerHiddenLayerChain, hiddenLayer, connection, deltaWeight);
            }
        }

        void setHiddenLayerDeltaWeightForNeuronAndConnection(const size_t hiddenLayer, const size_t neuron, const size_t connection, const ValueType& deltaWeight)
        {
            if((NeuralNetworkNumberOfHiddenLayers - 1) == hiddenLayer)
            {
                this->mLastHiddenLayer.setDeltaWeightForNeuronAndConnection(neuron, connection, deltaWeight);
            }
            else
            {
                ChainHiddenLayerAccessor<InnerHiddenLayerChainType>::setDeltaWeightForNeuronAndConnection(this->mInnerHiddenLayerChain, hiddenLayer, neuron, connection, deltaWeight);
            }
        }

        void setHiddenLayerWeightForNeuronAndConnection(const size_t hiddenLayer, const size_t neuron, const size_t connection, const ValueType& weight)
        {
            if((NeuralNetworkNumberOfHiddenLayers - 1) == hiddenLayer)
            {
                this->mLastHiddenLayer.setWeightForNeuronAndConnection(neuron, connection, weight);
            }
            else
            {
                ChainHiddenLayerAccessor<InnerHiddenLayerChainType>::setWeightForNeuronAndConnection(this->mInnerHiddenLayerChain, hiddenLayer, neuron, connection, weight);
            }
        }

        void setInputLayerBiasWeightForConnection(const size_t connection, const ValueType& weight)
        {
            this->mInputLayer.setBiasNeuronWeightForConnection(connection, weight);
        }

        void setInputLayerBiasDeltaWeightForConnection(const size_t connection, const ValueType& deltaWeight)
        {
            this->mInputLayer.setBiasNeuronDeltaWeightForConnection(connection, deltaWeight);
        }

        void setInputLayerDeltaWeightForNeuronAndConnection(const size_t neuron, const size_t connection, const ValueType& deltaWeight)
        {
            this->mInputLayer.setDeltaWeightForNeuronAndConnection(neuron, connection, deltaWeight);
        }

        void setInputLayerWeightForNeuronAndConnection(const size_t neuron, const size_t connection, const ValueType& weight)
        {
            this->mInputLayer.setWeightForNeuronAndConnection(neuron, connection, weight);
        }

        void setOutputLayerDeltaWeightForNeuronAndConnection(const size_t neuron, const size_t connection, const ValueType& deltaWeight)
        {
            this->mOutputLayer.setDeltaWeightForNeuronAndConnection(neuron, connection, deltaWeight);
        }

        void setOutputLayerWeightForNeuronAndConnection(const size_t neuron, const size_t connection, const ValueType& weight)
        {
            this->mOutputLayer.setWeightForNeuronAndConnection(neuron, connection, weight);
        }

        void setWeights(NeuralNetworkType& other)
        {
            typedef HiddenLayerSizeAccessor<HiddenLayersDescriptor> SizeAccessor;
            ValueType weightValue;

            const size_t firstHiddenLayerSize = SizeAccessor::getSize(0);

            for(size_t i = 0;i < NumberOfInputLayerNeurons;++i)
            {
                for(size_t h = 0;h < firstHiddenLayerSize;++h)
                {
                    weightValue = other.getInputLayerWeightForNeuronAndConnection(i, h);
                    this->setInputLayerWeightForNeuronAndConnection(i, h, weightValue);
                }
            }

            for(size_t h = 0;h < firstHiddenLayerSize;++h)
            {
                weightValue = other.getInputLayerBiasNeuronWeightForConnection(h);
                this->setInputLayerBiasWeightForConnection(h, weightValue);
            }

            for(size_t hiddenLayer = 0;hiddenLayer < (NeuralNetworkNumberOfHiddenLayers - 1);++hiddenLayer)
            {
                const size_t neuronsInThisLayer = SizeAccessor::getSize(hiddenLayer);
                const size_t neuronsInNextLayer = SizeAccessor::getSize(hiddenLayer + 1);

                for(size_t h = 0;h < neuronsInThisLayer;++h)
                {
                    for(size_t h1 = 0;h1 < neuronsInNextLayer;++h1)
                    {
                        weightValue = other.getHiddenLayerWeightForNeuronAndConnection(hiddenLayer, h, h1);
                        this->setHiddenLayerWeightForNeuronAndConnection(hiddenLayer, h, h1, weightValue);
                    }
                }

                for(size_t h1 = 0;h1 < neuronsInNextLayer;++h1)
                {
                    weightValue = other.getHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, h1);
                    this->setHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, h1, weightValue);
                }
            }

            const size_t lastHiddenLayer = NeuralNetworkNumberOfHiddenLayers - 1;

            for(size_t hiddenNeuron = 0;hiddenNeuron < NumberOfHiddenLayerNeurons;++hiddenNeuron)
            {
                for(size_t outputNeuron = 0;outputNeuron < NumberOfOutputLayerNeurons;++outputNeuron)
                {
                    weightValue = other.getHiddenLayerWeightForNeuronAndConnection(lastHiddenLayer, hiddenNeuron, outputNeuron);
                    this->setHiddenLayerWeightForNeuronAndConnection(lastHiddenLayer, hiddenNeuron, outputNeuron, weightValue);
                }
            }

            for(size_t outputNeuron = 0;outputNeuron < NumberOfOutputLayerNeurons;++outputNeuron)
            {
                weightValue = other.getHiddenLayerBiasNeuronWeightForConnection(lastHiddenLayer, outputNeuron);
                this->setHiddenLayerBiasNeuronWeightForConnection(lastHiddenLayer, outputNeuron, weightValue);
            }
        }

        void trainNetwork(ValueType const* const targetValues)
        {
            this->mTrainingPolicy.trainNetwork(*this, targetValues);
        }

    protected:
        void feedForwardHiddenLayers()
        {
            ChainFeedForwardDispatcher<HasRecurrentLayer>::template feedForward<InnerHiddenLayerChainType>(this->mInputLayer, this->mInnerHiddenLayerChain, this->mLastHiddenLayer, this->mRecurrentLayer);
        }

        void feedForwardInputLayer(ValueType const* const values)
        {
            this->mInputLayer.feedForward(values);
        }

        void feedForwardOutputLayer()
        {
            this->mOutputLayer.feedForward(this->mLastHiddenLayer);
        }

    protected:
        InputLayerType mInputLayer;
        InnerHiddenLayerChainType mInnerHiddenLayerChain;
        InnerRecurrentLayerChainType mInnerRecurrentLayerChain;
        LastHiddenLayerType mLastHiddenLayer;
        NeuralNetworkOutputLayerType mOutputLayer;
        NeuralNetworkRecurrentLayerType mRecurrentLayer;
        TrainingPolicyType mTrainingPolicy;
        GradientsManagerType mGradientsManager;
    private:
        unsigned char mLearnedValuesBuffer[NumberOfOutputLayerNeurons * sizeof(ValueType)];

        NeuralNetwork(const NeuralNetwork&) {} // hide copy constructor
        NeuralNetwork& operator=(const NeuralNetwork&) {} // hide assignment operator

        static_assert(NumberOfInputs > 0, "Invalid number of inputs.");
        static_assert(HiddenLayersDescriptor::Count > 0, "Must have at least one hidden layer.");
        static_assert(NumberOfOutputs > 0, "Invalid number of outputs.");
        static_assert(NumberOfOutputLayerNeurons == TransferFunctionsPolicy::NumberOfTransferFunctionsOutputNeurons, "TransferFunctionPolicy NumberOfOutputNeurons is incorrect.");
    };

    /**
     * Recurrent Neural Network
     *
     * Equivalent of RecurrentMultilayerPerceptron for the NeuralNetwork class.
     * Supports heterogeneous hidden layers via HiddenLayers<S0, S1, ...>.
     */
    template<
            typename ValueType,
            size_t NumberOfInputs,
            typename HiddenLayersDescriptor,
            size_t NumberOfOutputs,
            typename TransferFunctionsPolicy,
            bool IsTrainable = true,
            size_t BatchSize = 1,
            size_t RecurrentConnectionDepth = 1,
            outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration
            >
    class RecurrentNeuralNetwork : public NeuralNetwork< ValueType,
                                                          NumberOfInputs,
                                                          HiddenLayersDescriptor,
                                                          NumberOfOutputs,
                                                          TransferFunctionsPolicy,
                                                          IsTrainable,
                                                          BatchSize,
                                                          true,
                                                          RecurrentHiddenLayerConfig,
                                                          RecurrentConnectionDepth,
                                                          OutputLayerConfiguration>
    {
    private:
        static_assert(RecurrentConnectionDepth > 0, "Invalid recurrent connection depth.");
    };

    /**
     * Elman Neural Network
     *
     * Equivalent of ElmanNetwork for the NeuralNetwork class.
     * Single hidden layer with recurrent connection depth of 1.
     */
    template<
            typename ValueType,
            size_t NumberOfInputs,
            size_t NumberOfNeuronsInHiddenLayer,
            size_t NumberOfOutputs,
            typename TransferFunctionsPolicy,
            bool IsTrainable = true,
            size_t BatchSize = 1,
            outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration
            >
    class ElmanNeuralNetwork : public RecurrentNeuralNetwork< ValueType,
                                                               NumberOfInputs,
                                                               HiddenLayers<NumberOfNeuronsInHiddenLayer>,
                                                               NumberOfOutputs,
                                                               TransferFunctionsPolicy,
                                                               IsTrainable,
                                                               BatchSize,
                                                               1,
                                                               OutputLayerConfiguration>
    {
    };

    /**
     * LSTM Neural Network
     *
     * Recurrent neural network with LSTM hidden layer.
     * Supports heterogeneous hidden layers via HiddenLayers<S0, S1, ...>.
     */
    template<
            typename ValueType,
            size_t NumberOfInputs,
            typename HiddenLayersDescriptor,
            size_t NumberOfOutputs,
            typename TransferFunctionsPolicy,
            bool IsTrainable = true,
            size_t BatchSize = 1,
            size_t RecurrentConnectionDepth = 1,
            outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration
            >
    class LstmNeuralNetwork : public NeuralNetwork< ValueType,
                                                      NumberOfInputs,
                                                      HiddenLayersDescriptor,
                                                      NumberOfOutputs,
                                                      TransferFunctionsPolicy,
                                                      IsTrainable,
                                                      BatchSize,
                                                      true,
                                                      LSTMHiddenLayerConfig,
                                                      RecurrentConnectionDepth,
                                                      OutputLayerConfiguration>
    {
    public:
        typedef NeuralNetwork<  ValueType,
                                NumberOfInputs,
                                HiddenLayersDescriptor,
                                NumberOfOutputs,
                                TransferFunctionsPolicy,
                                IsTrainable,
                                BatchSize,
                                true,
                                LSTMHiddenLayerConfig,
                                RecurrentConnectionDepth,
                                OutputLayerConfiguration> BaseType;
        typedef typename BaseType::LastHiddenLayerNeuronType LastHiddenLayerNeuronType;

        void resetState()
        {
            for (size_t neuron = 0; neuron < BaseType::NumberOfHiddenLayerNeurons; ++neuron)
            {
                this->getLastHiddenLayer().getNeuron(neuron)->setState(ValueType{});
            }
        }

    private:
        static_assert(RecurrentConnectionDepth > 0, "Invalid recurrent connection depth.");
    };

    /**
     * GRU Neural Network
     *
     * Recurrent neural network with GRU hidden layer.
     * Uses 3 gates (update, reset, candidate) instead of LSTM's 4,
     * resulting in ~25% less memory and compute per hidden neuron.
     * Supports heterogeneous hidden layers via HiddenLayers<S0, S1, ...>.
     */
    template<
            typename ValueType,
            size_t NumberOfInputs,
            typename HiddenLayersDescriptor,
            size_t NumberOfOutputs,
            typename TransferFunctionsPolicy,
            bool IsTrainable = true,
            size_t BatchSize = 1,
            size_t RecurrentConnectionDepth = 1,
            outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration
            >
    class GruNeuralNetwork : public NeuralNetwork< ValueType,
                                                     NumberOfInputs,
                                                     HiddenLayersDescriptor,
                                                     NumberOfOutputs,
                                                     TransferFunctionsPolicy,
                                                     IsTrainable,
                                                     BatchSize,
                                                     true,
                                                     GRUHiddenLayerConfig,
                                                     RecurrentConnectionDepth,
                                                     OutputLayerConfiguration>
    {
    public:
        typedef NeuralNetwork<  ValueType,
                                NumberOfInputs,
                                HiddenLayersDescriptor,
                                NumberOfOutputs,
                                TransferFunctionsPolicy,
                                IsTrainable,
                                BatchSize,
                                true,
                                GRUHiddenLayerConfig,
                                RecurrentConnectionDepth,
                                OutputLayerConfiguration> BaseType;
        typedef typename BaseType::LastHiddenLayerNeuronType LastHiddenLayerNeuronType;

        void resetState()
        {
            for (size_t neuron = 0; neuron < BaseType::NumberOfHiddenLayerNeurons; ++neuron)
            {
                this->getLastHiddenLayer().getNeuron(neuron)->setState(ValueType{});
                this->getLastHiddenLayer().getNeuron(neuron)->setOutputValue(ValueType{});
                this->getRecurrentLayer().setOutputValueForOutgoingConnection(neuron, ValueType{});
            }
        }

    private:
        static_assert(RecurrentConnectionDepth > 0, "Invalid recurrent connection depth.");
    };
}
