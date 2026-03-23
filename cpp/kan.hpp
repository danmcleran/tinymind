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

#include <cstddef>
#include <new>

#include "bspline.hpp"
#include "kanTransferFunctions.hpp"

namespace tinymind {

    // =========================================================================
    // KAN Connections
    // =========================================================================

    /**
     * Non-trainable KAN connection (inference only).
     *
     * Each KAN edge stores:
     * - B-spline coefficients (NumberOfCoefficients = GridSize + SplineDegree)
     * - Base weight w_b (for SiLU residual path)
     * - Spline weight w_s (scaling the spline output)
     *
     * Edge function: phi(x) = w_b * SiLU(x) + w_s * spline(x)
     */
    template<typename ValueType, size_t GridSize, size_t SplineDegree>
    struct KanConnection
    {
        typedef ValueType ConnectionValueType;

        static const bool IsTrainable = false;
        static const size_t NumberOfCoefficients = GridSize + SplineDegree;
        static const size_t KanGridSize = GridSize;
        static const size_t KanSplineDegree = SplineDegree;

        KanConnection()
        {
            for (size_t i = 0; i < NumberOfCoefficients; ++i)
            {
                mCoefficients[i] = static_cast<ValueType>(0);
            }
            mBaseWeight = static_cast<ValueType>(0);
            mSplineWeight = static_cast<ValueType>(0);
        }

        ValueType getCoefficient(const size_t index) const
        {
            return this->mCoefficients[index];
        }

        const ValueType* getCoefficients() const
        {
            return this->mCoefficients;
        }

        void setCoefficient(const size_t index, const ValueType& value)
        {
            this->mCoefficients[index] = value;
        }

        ValueType getBaseWeight() const
        {
            return this->mBaseWeight;
        }

        void setBaseWeight(const ValueType& value)
        {
            this->mBaseWeight = value;
        }

        ValueType getSplineWeight() const
        {
            return this->mSplineWeight;
        }

        void setSplineWeight(const ValueType& value)
        {
            this->mSplineWeight = value;
        }

        /**
         * Evaluate the KAN edge function: phi(x) = w_b * SiLU(x) + w_s * spline(x)
         */
        ValueType evaluate(const ValueType& x, const ValueType* knots, const size_t numberOfBasisFunctions) const
        {
            const ValueType siluValue = SiLUActivationPolicy<ValueType>::activationFunction(x);
            const ValueType splineValue = DeBoorEvaluator<ValueType, SplineDegree>::evaluateSpline(
                this->mCoefficients, knots, numberOfBasisFunctions, x);

            return this->mBaseWeight * siluValue + this->mSplineWeight * splineValue;
        }

        // Stub methods for compatibility with training infrastructure
        ValueType getCoefficientGradient(const size_t index) const { (void)index; return static_cast<ValueType>(0); }
        void setCoefficientGradient(const size_t index, const ValueType& value) { (void)index; (void)value; }
        ValueType getCoefficientDeltaWeight(const size_t index) const { (void)index; return static_cast<ValueType>(0); }
        void setCoefficientDeltaWeight(const size_t index, const ValueType& value) { (void)index; (void)value; }
        ValueType getCoefficientPreviousDeltaWeight(const size_t index) const { (void)index; return static_cast<ValueType>(0); }
        ValueType getBaseWeightGradient() const { return static_cast<ValueType>(0); }
        void setBaseWeightGradient(const ValueType& value) { (void)value; }
        ValueType getBaseWeightDeltaWeight() const { return static_cast<ValueType>(0); }
        void setBaseWeightDeltaWeight(const ValueType& value) { (void)value; }
        ValueType getBaseWeightPreviousDeltaWeight() const { return static_cast<ValueType>(0); }
        ValueType getSplineWeightGradient() const { return static_cast<ValueType>(0); }
        void setSplineWeightGradient(const ValueType& value) { (void)value; }
        ValueType getSplineWeightDeltaWeight() const { return static_cast<ValueType>(0); }
        void setSplineWeightDeltaWeight(const ValueType& value) { (void)value; }
        ValueType getSplineWeightPreviousDeltaWeight() const { return static_cast<ValueType>(0); }

        void * operator new(size_t, void *p)
        {
            return p;
        }

    protected:
        ValueType mCoefficients[NumberOfCoefficients];
        ValueType mBaseWeight;
        ValueType mSplineWeight;
    };

    /**
     * Trainable KAN connection.
     *
     * Extends KanConnection with gradient and delta weight storage
     * for each learnable parameter (all coefficients, base weight, spline weight).
     */
    template<typename ValueType, size_t GridSize, size_t SplineDegree>
    struct TrainableKanConnection : public KanConnection<ValueType, GridSize, SplineDegree>
    {
        typedef ValueType ConnectionValueType;

        static const bool IsTrainable = true;
        static const size_t NumberOfCoefficients = GridSize + SplineDegree;

        TrainableKanConnection()
        {
            for (size_t i = 0; i < NumberOfCoefficients; ++i)
            {
                mCoefficientGradients[i] = static_cast<ValueType>(0);
                mCoefficientDeltaWeights[i] = static_cast<ValueType>(0);
                mCoefficientPreviousDeltaWeights[i] = static_cast<ValueType>(0);
            }
            mBaseWeightGradient = static_cast<ValueType>(0);
            mBaseWeightDeltaWeight = static_cast<ValueType>(0);
            mBaseWeightPreviousDeltaWeight = static_cast<ValueType>(0);
            mSplineWeightGradient = static_cast<ValueType>(0);
            mSplineWeightDeltaWeight = static_cast<ValueType>(0);
            mSplineWeightPreviousDeltaWeight = static_cast<ValueType>(0);
        }

        ValueType getCoefficientGradient(const size_t index) const
        {
            return this->mCoefficientGradients[index];
        }

        void setCoefficientGradient(const size_t index, const ValueType& value)
        {
            this->mCoefficientGradients[index] = value;
        }

        ValueType getCoefficientDeltaWeight(const size_t index) const
        {
            return this->mCoefficientDeltaWeights[index];
        }

        void setCoefficientDeltaWeight(const size_t index, const ValueType& value)
        {
            this->mCoefficientPreviousDeltaWeights[index] = this->mCoefficientDeltaWeights[index];
            this->mCoefficientDeltaWeights[index] = value;
        }

        ValueType getCoefficientPreviousDeltaWeight(const size_t index) const
        {
            return this->mCoefficientPreviousDeltaWeights[index];
        }

        ValueType getBaseWeightGradient() const { return this->mBaseWeightGradient; }
        void setBaseWeightGradient(const ValueType& value) { this->mBaseWeightGradient = value; }
        ValueType getBaseWeightDeltaWeight() const { return this->mBaseWeightDeltaWeight; }
        void setBaseWeightDeltaWeight(const ValueType& value)
        {
            this->mBaseWeightPreviousDeltaWeight = this->mBaseWeightDeltaWeight;
            this->mBaseWeightDeltaWeight = value;
        }
        ValueType getBaseWeightPreviousDeltaWeight() const { return this->mBaseWeightPreviousDeltaWeight; }

        ValueType getSplineWeightGradient() const { return this->mSplineWeightGradient; }
        void setSplineWeightGradient(const ValueType& value) { this->mSplineWeightGradient = value; }
        ValueType getSplineWeightDeltaWeight() const { return this->mSplineWeightDeltaWeight; }
        void setSplineWeightDeltaWeight(const ValueType& value)
        {
            this->mSplineWeightPreviousDeltaWeight = this->mSplineWeightDeltaWeight;
            this->mSplineWeightDeltaWeight = value;
        }
        ValueType getSplineWeightPreviousDeltaWeight() const { return this->mSplineWeightPreviousDeltaWeight; }

    protected:
        ValueType mCoefficientGradients[NumberOfCoefficients];
        ValueType mCoefficientDeltaWeights[NumberOfCoefficients];
        ValueType mCoefficientPreviousDeltaWeights[NumberOfCoefficients];
        ValueType mBaseWeightGradient;
        ValueType mBaseWeightDeltaWeight;
        ValueType mBaseWeightPreviousDeltaWeight;
        ValueType mSplineWeightGradient;
        ValueType mSplineWeightDeltaWeight;
        ValueType mSplineWeightPreviousDeltaWeight;
    };

    /**
     * Selects trainable or non-trainable KAN connection based on IsTrainable.
     */
    template<typename ValueType, size_t GridSize, size_t SplineDegree, bool IsTrainable>
    struct KanConnectionTypeSelector
    {
    };

    template<typename ValueType, size_t GridSize, size_t SplineDegree>
    struct KanConnectionTypeSelector<ValueType, GridSize, SplineDegree, true>
    {
        typedef TrainableKanConnection<ValueType, GridSize, SplineDegree> ConnectionType;
    };

    template<typename ValueType, size_t GridSize, size_t SplineDegree>
    struct KanConnectionTypeSelector<ValueType, GridSize, SplineDegree, false>
    {
        typedef KanConnection<ValueType, GridSize, SplineDegree> ConnectionType;
    };

    // =========================================================================
    // KAN Neurons
    // =========================================================================

    /**
     * KAN neuron base.
     *
     * Each neuron has outgoing KAN connections to the next layer.
     * The neuron's output is set externally by the layer (sum of incoming
     * spline evaluations). The neuron evaluates its outgoing connections
     * when requested.
     */
    template<typename KanConnectionType, size_t NumberOfOutgoingConnections, typename TransferFunctionsPolicy>
    struct KanNeuron
    {
        typedef KanConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;
        static const size_t NumberOfCoefficientsPerConnection = KanConnectionType::NumberOfCoefficients;

        ValueType getOutputValue() const
        {
            return this->mOutputValue;
        }

        void setOutputValue(const ValueType& value)
        {
            this->mOutputValue = value;
        }

        /**
         * Evaluate the KAN edge function phi(x) for the specified outgoing connection.
         * x is this neuron's output value.
         */
        ValueType evaluateConnectionOutput(const size_t connection, const ValueType* knots, const size_t numberOfBasisFunctions) const
        {
            const size_t bufferIndex = connection * sizeof(KanConnectionType);
            const KanConnectionType* pConnection = reinterpret_cast<const KanConnectionType*>(&this->mOutgoingConnectionsBuffer[bufferIndex]);
            return pConnection->evaluate(this->mOutputValue, knots, numberOfBasisFunctions);
        }

        KanConnectionType* getConnection(const size_t connection)
        {
            const size_t bufferIndex = connection * sizeof(KanConnectionType);
            return reinterpret_cast<KanConnectionType*>(&this->mOutgoingConnectionsBuffer[bufferIndex]);
        }

        const KanConnectionType* getConnection(const size_t connection) const
        {
            const size_t bufferIndex = connection * sizeof(KanConnectionType);
            return reinterpret_cast<const KanConnectionType*>(&this->mOutgoingConnectionsBuffer[bufferIndex]);
        }

        void initializeWeights()
        {
            for (size_t conn = 0; conn < NumberOfOutgoingConnections; ++conn)
            {
                KanConnectionType* pConnection = this->getConnection(conn);
                // Initialize spline coefficients with small random values
                for (size_t c = 0; c < NumberOfCoefficientsPerConnection; ++c)
                {
                    pConnection->setCoefficient(c, TransferFunctionsPolicy::generateRandomWeight());
                }
                // Base weight starts at 1.0 (SiLU residual)
                pConnection->setBaseWeight(Constants<ValueType>::one());
                // Spline weight starts at 1.0
                pConnection->setSplineWeight(Constants<ValueType>::one());
            }
        }

        void setIndex(const size_t index)
        {
            this->mIndex = index;
        }

        void * operator new(size_t, void *p)
        {
            return p;
        }

    protected:
        KanNeuron()
        {
            size_t bufferIndex;
            for (size_t index = 0; index < NumberOfOutgoingConnections; ++index)
            {
                bufferIndex = index * sizeof(KanConnectionType);
                new (&this->mOutgoingConnectionsBuffer[bufferIndex]) KanConnectionType();
            }
            this->mOutputValue = static_cast<ValueType>(0);
            this->mIndex = 0;
        }

        unsigned char mOutgoingConnectionsBuffer[NumberOfOutgoingConnections * sizeof(KanConnectionType)];
        ValueType mOutputValue;
        size_t mIndex;

        static_assert(NumberOfOutgoingConnections > 0, "Invalid number of outgoing connections.");
    };

    /**
     * Trainable KAN neuron - adds node delta storage for backpropagation.
     */
    template<typename KanConnectionType, size_t NumberOfOutgoingConnections, typename TransferFunctionsPolicy>
    struct TrainableKanNeuron : public KanNeuron<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef KanConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;

        ValueType getNodeDelta() const
        {
            return this->mNodeDelta;
        }

        void setNodeDelta(const ValueType& value)
        {
            this->mNodeDelta = value;
        }

    protected:
        TrainableKanNeuron() : mNodeDelta(static_cast<ValueType>(0))
        {
        }

        ValueType mNodeDelta;
    };

    // Neuron type aliases for different layers
    template<typename KanConnectionType, size_t NumberOfOutgoingConnections, typename TransferFunctionsPolicy>
    struct KanInputLayerNeuron : public KanNeuron<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef KanConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;
        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;
    };

    template<typename KanConnectionType, size_t NumberOfOutgoingConnections, typename TransferFunctionsPolicy>
    struct TrainableKanInputLayerNeuron : public TrainableKanNeuron<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef KanConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;
        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;
    };

    template<typename KanConnectionType, size_t NumberOfOutgoingConnections, typename TransferFunctionsPolicy, bool IsTrainable>
    struct KanInputLayerNeuronTypeSelector
    {
    };

    template<typename KanConnectionType, size_t NumberOfOutgoingConnections, typename TransferFunctionsPolicy>
    struct KanInputLayerNeuronTypeSelector<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, true>
    {
        typedef TrainableKanInputLayerNeuron<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> NeuronType;
    };

    template<typename KanConnectionType, size_t NumberOfOutgoingConnections, typename TransferFunctionsPolicy>
    struct KanInputLayerNeuronTypeSelector<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, false>
    {
        typedef KanInputLayerNeuron<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> NeuronType;
    };

    template<typename KanConnectionType, size_t NumberOfOutgoingConnections, typename TransferFunctionsPolicy>
    struct KanHiddenLayerNeuron : public KanNeuron<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef KanConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;
        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;
    };

    template<typename KanConnectionType, size_t NumberOfOutgoingConnections, typename TransferFunctionsPolicy>
    struct TrainableKanHiddenLayerNeuron : public TrainableKanNeuron<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy>
    {
        typedef KanConnectionType NeuronConnectionType;
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;
        static const size_t NumberOfOutgoingConnectionsFromNeuron = NumberOfOutgoingConnections;
    };

    template<typename KanConnectionType, size_t NumberOfOutgoingConnections, typename TransferFunctionsPolicy, bool IsTrainable>
    struct KanHiddenLayerNeuronTypeSelector
    {
    };

    template<typename KanConnectionType, size_t NumberOfOutgoingConnections, typename TransferFunctionsPolicy>
    struct KanHiddenLayerNeuronTypeSelector<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, true>
    {
        typedef TrainableKanHiddenLayerNeuron<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> NeuronType;
    };

    template<typename KanConnectionType, size_t NumberOfOutgoingConnections, typename TransferFunctionsPolicy>
    struct KanHiddenLayerNeuronTypeSelector<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy, false>
    {
        typedef KanHiddenLayerNeuron<KanConnectionType, NumberOfOutgoingConnections, TransferFunctionsPolicy> NeuronType;
    };

    /**
     * KAN output neuron - no outgoing connections, just stores output and delta.
     */
    template<typename TransferFunctionsPolicy>
    struct KanOutputNeuron
    {
        typedef TransferFunctionsPolicy NeuronTransferFunctionsPolicy;
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        KanOutputNeuron() : mOutputValue(static_cast<ValueType>(0)), mNodeDelta(static_cast<ValueType>(0))
        {
        }

        ValueType getOutputValue() const { return this->mOutputValue; }
        void setOutputValue(const ValueType& value) { this->mOutputValue = value; }

        ValueType getNodeDelta() const { return this->mNodeDelta; }
        void setNodeDelta(const ValueType& value) { this->mNodeDelta = value; }

        void setIndex(const size_t index) { (void)index; }

        void * operator new(size_t, void *p)
        {
            return p;
        }

    protected:
        ValueType mOutputValue;
        ValueType mNodeDelta;
    };

    // =========================================================================
    // KAN Layers
    // =========================================================================

    /**
     * KAN layer base - stores neurons in a buffer and owns a shared knot vector.
     */
    template<typename NeuronType, size_t NumberOfNeurons, size_t GridSize, size_t SplineDegree>
    struct KanLayerBase
    {
        typedef NeuronType LayerNeuronType;
        typedef typename NeuronType::ValueType ValueType;
        typedef UniformKnotVector<ValueType, GridSize, SplineDegree> KnotVectorType;

        static const size_t NumberOfNeuronsInLayer = NumberOfNeurons;
        static const size_t NumberOfBasisFunctions = KnotVectorType::NumberOfBasisFunctions;

        NeuronType* getNeuron(const size_t neuron)
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            return reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
        }

        const NeuronType* getNeuron(const size_t neuron) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            return reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
        }

        ValueType getOutputValueForNeuron(const size_t neuron) const
        {
            return this->getNeuron(neuron)->getOutputValue();
        }

        void setOutputValueForNeuron(const size_t neuron, const ValueType& value)
        {
            this->getNeuron(neuron)->setOutputValue(value);
        }

        ValueType getNodeDeltaForNeuron(const size_t neuron) const
        {
            return this->getNeuron(neuron)->getNodeDelta();
        }

        void setNodeDeltaForNeuron(const size_t neuron, const ValueType& value)
        {
            this->getNeuron(neuron)->setNodeDelta(value);
        }

        const KnotVectorType& getKnotVector() const
        {
            return this->mKnotVector;
        }

        void initializeNeurons()
        {
            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                this->getNeuron(neuron)->setIndex(neuron);
            }
        }

        void initializeKnots(const ValueType& gridMin, const ValueType& gridMax)
        {
            this->mKnotVector.initialize(gridMin, gridMax);
        }

        void initializeWeights()
        {
            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                this->getNeuron(neuron)->initializeWeights();
            }
        }

        /**
         * Get the sum of all phi_{i,j}(x_i) for outgoing connection index j,
         * summed across all neurons i in this layer.
         */
        ValueType getSplineOutputForOutgoingConnection(const size_t connection) const
        {
            ValueType sum(0);
            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                sum += this->getNeuron(neuron)->evaluateConnectionOutput(
                    connection, this->mKnotVector.knots, NumberOfBasisFunctions);
            }
            return sum;
        }

        void * operator new(size_t, void *p)
        {
            return p;
        }

    protected:
        KanLayerBase()
        {
            size_t bufferIndex;
            for (size_t neuronIndex = 0; neuronIndex < NumberOfNeurons; ++neuronIndex)
            {
                bufferIndex = neuronIndex * sizeof(NeuronType);
                new (&this->mNeuronsBuffer[bufferIndex]) NeuronType();
            }
        }

        unsigned char mNeuronsBuffer[NumberOfNeurons * sizeof(NeuronType)];
        KnotVectorType mKnotVector;

        static_assert(NumberOfNeurons > 0, "Number of neurons must be > 0.");
    };

    /**
     * KAN Input Layer - latches input values, has outgoing KAN connections.
     */
    template<typename NeuronType, size_t NumberOfNeurons, size_t GridSize, size_t SplineDegree>
    struct KanInputLayer : public KanLayerBase<NeuronType, NumberOfNeurons, GridSize, SplineDegree>
    {
        typedef typename NeuronType::ValueType ValueType;

        static const size_t NumberOfNeuronsInLayer = NumberOfNeurons;

        void feedForward(ValueType const* const values)
        {
            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                this->getNeuron(neuron)->setOutputValue(values[neuron]);
            }
        }
    };

    /**
     * KAN Hidden Layer - receives spline outputs from previous layer, has outgoing KAN connections.
     */
    template<typename NeuronType, size_t NumberOfNeurons, size_t GridSize, size_t SplineDegree>
    struct KanHiddenLayer : public KanLayerBase<NeuronType, NumberOfNeurons, GridSize, SplineDegree>
    {
        typedef typename NeuronType::ValueType ValueType;
        typedef typename KanLayerBase<NeuronType, NumberOfNeurons, GridSize, SplineDegree>::KnotVectorType KnotVectorType;

        static const size_t NumberOfNeuronsInLayer = NumberOfNeurons;

        /**
         * Feed forward: for each neuron j, sum phi_{i,j}(x_i) across all
         * neurons i in the previous layer.
         */
        template<typename PreviousLayerType>
        void feedForward(const PreviousLayerType& previousLayer)
        {
            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                const ValueType sum = previousLayer.getSplineOutputForOutgoingConnection(neuron);
                this->getNeuron(neuron)->setOutputValue(sum);
            }
        }
    };

    /**
     * KAN Output Layer - receives spline outputs, no outgoing connections.
     */
    template<typename NeuronType, size_t NumberOfNeurons>
    struct KanOutputLayer
    {
        typedef typename NeuronType::ValueType ValueType;

        static const size_t NumberOfNeuronsInLayer = NumberOfNeurons;

        NeuronType* getNeuron(const size_t neuron)
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            return reinterpret_cast<NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
        }

        const NeuronType* getNeuron(const size_t neuron) const
        {
            const size_t bufferIndex = neuron * sizeof(NeuronType);
            return reinterpret_cast<const NeuronType*>(&this->mNeuronsBuffer[bufferIndex]);
        }

        ValueType getOutputValueForNeuron(const size_t neuron) const
        {
            return this->getNeuron(neuron)->getOutputValue();
        }

        ValueType getNodeDeltaForNeuron(const size_t neuron) const
        {
            return this->getNeuron(neuron)->getNodeDelta();
        }

        void setNodeDeltaForNeuron(const size_t neuron, const ValueType& value)
        {
            this->getNeuron(neuron)->setNodeDelta(value);
        }

        void initializeNeurons()
        {
            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                this->getNeuron(neuron)->setIndex(neuron);
            }
        }

        template<typename PreviousLayerType>
        void feedForward(const PreviousLayerType& previousLayer)
        {
            for (size_t neuron = 0; neuron < NumberOfNeurons; ++neuron)
            {
                const ValueType sum = previousLayer.getSplineOutputForOutgoingConnection(neuron);
                this->getNeuron(neuron)->setOutputValue(sum);
            }
        }

        void * operator new(size_t, void *p)
        {
            return p;
        }

        KanOutputLayer()
        {
            size_t bufferIndex;
            for (size_t neuronIndex = 0; neuronIndex < NumberOfNeurons; ++neuronIndex)
            {
                bufferIndex = neuronIndex * sizeof(NeuronType);
                new (&this->mNeuronsBuffer[bufferIndex]) NeuronType();
            }
        }

    private:
        unsigned char mNeuronsBuffer[NumberOfNeurons * sizeof(NeuronType)];

        static_assert(NumberOfNeurons > 0, "Number of output neurons must be > 0.");
    };

    // =========================================================================
    // Inner Hidden Layer Manager (for multi-hidden-layer support)
    // =========================================================================

    template<typename InnerHiddenLayerType, size_t NumberOfInnerHiddenLayers>
    struct KanInnerHiddenLayerManager
    {
        KanInnerHiddenLayerManager()
        {
            size_t bufferIndex;
            for (size_t i = 0; i < NumberOfInnerHiddenLayers; ++i)
            {
                bufferIndex = i * sizeof(InnerHiddenLayerType);
                new (&this->mBuffer[bufferIndex]) InnerHiddenLayerType();
            }
            this->pLayers = reinterpret_cast<InnerHiddenLayerType*>(&this->mBuffer[0]);
        }

        void initializeNeurons()
        {
            for (size_t i = 0; i < NumberOfInnerHiddenLayers; ++i)
            {
                this->pLayers[i].initializeNeurons();
            }
        }

        void initializeWeights()
        {
            for (size_t i = 0; i < NumberOfInnerHiddenLayers; ++i)
            {
                this->pLayers[i].initializeWeights();
            }
        }

        template<typename ValueType>
        void initializeKnots(const ValueType& gridMin, const ValueType& gridMax)
        {
            for (size_t i = 0; i < NumberOfInnerHiddenLayers; ++i)
            {
                this->pLayers[i].initializeKnots(gridMin, gridMax);
            }
        }

        InnerHiddenLayerType* getPointerToLayers()
        {
            return this->pLayers;
        }

    private:
        unsigned char mBuffer[NumberOfInnerHiddenLayers * sizeof(InnerHiddenLayerType)];
        InnerHiddenLayerType* pLayers;
    };

    template<typename InnerHiddenLayerType>
    struct KanInnerHiddenLayerManager<InnerHiddenLayerType, 0>
    {
        KanInnerHiddenLayerManager()
        {
#if __cplusplus >= 201103L
            this->pLayers = nullptr;
#else
            this->pLayers = NULL;
#endif
        }

        static void initializeNeurons() {}
        static void initializeWeights() {}

        template<typename ValueType>
        static void initializeKnots(const ValueType& gridMin, const ValueType& gridMax) { (void)gridMin; (void)gridMax; }

        InnerHiddenLayerType* getPointerToLayers()
        {
            return this->pLayers;
        }

    private:
        InnerHiddenLayerType* pLayers;
    };

    // =========================================================================
    // KAN Feed Forward Manager
    // =========================================================================

    template<typename InputLayerType, typename InnerHiddenLayerType, typename LastHiddenLayerType, size_t NumberOfInnerHiddenLayers>
    struct KanHiddenLayerFeedForwardManager
    {
        static void feedForward(InputLayerType& inputLayer, InnerHiddenLayerType* pInnerLayers, LastHiddenLayerType& lastHiddenLayer)
        {
            pInnerLayers[0].feedForward(inputLayer);

            for (size_t i = 1; i < NumberOfInnerHiddenLayers; ++i)
            {
                pInnerLayers[i].feedForward(pInnerLayers[i - 1]);
            }

            lastHiddenLayer.feedForward(pInnerLayers[NumberOfInnerHiddenLayers - 1]);
        }
    };

    template<typename InputLayerType, typename InnerHiddenLayerType, typename LastHiddenLayerType>
    struct KanHiddenLayerFeedForwardManager<InputLayerType, InnerHiddenLayerType, LastHiddenLayerType, 1>
    {
        static void feedForward(InputLayerType& inputLayer, InnerHiddenLayerType* pInnerLayers, LastHiddenLayerType& lastHiddenLayer)
        {
            pInnerLayers[0].feedForward(inputLayer);
            lastHiddenLayer.feedForward(pInnerLayers[0]);
        }
    };

    template<typename InputLayerType, typename InnerHiddenLayerType, typename LastHiddenLayerType>
    struct KanHiddenLayerFeedForwardManager<InputLayerType, InnerHiddenLayerType, LastHiddenLayerType, 0>
    {
        static void feedForward(InputLayerType& inputLayer, InnerHiddenLayerType* pInnerLayers, LastHiddenLayerType& lastHiddenLayer)
        {
            (void)pInnerLayers;
            lastHiddenLayer.feedForward(inputLayer);
        }
    };

    // =========================================================================
    // KAN Training: Deltas, Gradients, Weight Updates
    // =========================================================================

    /**
     * Calculate output layer node deltas.
     * For KAN, output is a linear sum (no activation), so delta = target - output.
     */
    template<typename TransferFunctionsPolicy, typename OutputLayerType>
    struct KanOutputLayerDeltasCalculator
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        static void calculate(OutputLayerType& outputLayer, ValueType const* const targetValues)
        {
            for (size_t neuron = 0; neuron < OutputLayerType::NumberOfNeuronsInLayer; ++neuron)
            {
                const ValueType outputValue = outputLayer.getOutputValueForNeuron(neuron);
                const ValueType delta = targetValues[neuron] - outputValue;
                outputLayer.setNodeDeltaForNeuron(neuron, delta);
            }
        }
    };

    /**
     * Calculate hidden layer node deltas for KAN.
     *
     * For a KAN hidden neuron j with output x_j:
     *   delta_j = sum_k( d(phi_{j,k})/d(x_j) * delta_k )
     *
     * where d(phi_{j,k})/d(x_j) = w_b * SiLU'(x_j) + w_s * spline'(x_j)
     * and delta_k is the next layer's node delta for neuron k.
     */
    template<typename ValueType, size_t SplineDegree>
    struct KanHiddenLayerDeltasCalculator
    {
        template<typename LayerType, typename NextLayerType>
        static void calculate(LayerType& layer, const NextLayerType& nextLayer)
        {
            for (size_t neuron = 0; neuron < LayerType::NumberOfNeuronsInLayer; ++neuron)
            {
                ValueType sum(0);
                const ValueType x = layer.getOutputValueForNeuron(neuron);
                const ValueType siluDeriv = SiLUActivationPolicy<ValueType>::activationFunctionDerivative(x);

                for (size_t nextNeuron = 0; nextNeuron < NextLayerType::NumberOfNeuronsInLayer; ++nextNeuron)
                {
                    const ValueType nextDelta = nextLayer.getNodeDeltaForNeuron(nextNeuron);
                    const typename LayerType::LayerNeuronType* pNeuron = layer.getNeuron(neuron);
                    const typename LayerType::LayerNeuronType::NeuronConnectionType* pConn = pNeuron->getConnection(nextNeuron);

                    const ValueType splineDeriv = DeBoorEvaluator<ValueType, SplineDegree>::evaluateSplineDerivative(
                        pConn->getCoefficients(),
                        layer.getKnotVector().knots,
                        LayerType::NumberOfBasisFunctions,
                        x);

                    const ValueType edgeDeriv = pConn->getBaseWeight() * siluDeriv + pConn->getSplineWeight() * splineDeriv;
                    sum += edgeDeriv * nextDelta;
                }

                layer.setNodeDeltaForNeuron(neuron, sum);
            }
        }
    };

    /**
     * Calculate and update gradients for KAN connections.
     *
     * For each connection from neuron i in layer to neuron j in nextLayer:
     *   gradient_c_m = w_s * B_{m,k}(x_i) * delta_j   (for each coefficient m)
     *   gradient_w_b = SiLU(x_i) * delta_j
     *   gradient_w_s = spline(x_i) * delta_j
     */
    template<typename ValueType, size_t SplineDegree>
    struct KanGradientsCalculator
    {
        template<typename LayerType, typename NextLayerType>
        static void calculate(LayerType& layer, const NextLayerType& nextLayer)
        {
            for (size_t neuron = 0; neuron < LayerType::NumberOfNeuronsInLayer; ++neuron)
            {
                const ValueType x = layer.getOutputValueForNeuron(neuron);
                const ValueType siluValue = SiLUActivationPolicy<ValueType>::activationFunction(x);

                for (size_t nextNeuron = 0; nextNeuron < NextLayerType::NumberOfNeuronsInLayer; ++nextNeuron)
                {
                    const ValueType nextDelta = nextLayer.getNodeDeltaForNeuron(nextNeuron);
                    typename LayerType::LayerNeuronType* pNeuron = layer.getNeuron(neuron);
                    typename LayerType::LayerNeuronType::NeuronConnectionType* pConn = pNeuron->getConnection(nextNeuron);

                    // Gradient for spline coefficients: w_s * B_{m,k}(x_i) * delta_j
                    // We compute all basis function values at x and multiply by w_s * delta_j
                    const ValueType splineWeight = pConn->getSplineWeight();
                    const ValueType splineValue = DeBoorEvaluator<ValueType, SplineDegree>::evaluateSpline(
                        pConn->getCoefficients(),
                        layer.getKnotVector().knots,
                        LayerType::NumberOfBasisFunctions,
                        x);

                    // For coefficient gradients, we use a numerical approach:
                    // Perturb each coefficient and measure the change in spline output.
                    // This avoids needing to evaluate individual basis functions.
                    // gradient_c_m ≈ delta_j * w_s * B_{m,k}(x_i)
                    // We compute B_{m,k}(x_i) by evaluating a spline with unit coefficient at m.
                    for (size_t c = 0; c < LayerType::LayerNeuronType::NumberOfCoefficientsPerConnection; ++c)
                    {
                        // Create a unit coefficient vector: all zeros except index c = 1
                        ValueType unitCoeffs[LayerType::LayerNeuronType::NumberOfCoefficientsPerConnection];
                        for (size_t i = 0; i < LayerType::LayerNeuronType::NumberOfCoefficientsPerConnection; ++i)
                        {
                            unitCoeffs[i] = static_cast<ValueType>(0);
                        }
                        unitCoeffs[c] = Constants<ValueType>::one();

                        const ValueType basisValue = DeBoorEvaluator<ValueType, SplineDegree>::evaluateSpline(
                            unitCoeffs,
                            layer.getKnotVector().knots,
                            LayerType::NumberOfBasisFunctions,
                            x);

                        const ValueType coeffGradient = splineWeight * basisValue * nextDelta;
                        pConn->setCoefficientGradient(c, coeffGradient);
                    }

                    // Gradient for base weight: SiLU(x_i) * delta_j
                    pConn->setBaseWeightGradient(siluValue * nextDelta);

                    // Gradient for spline weight: spline(x_i) * delta_j
                    pConn->setSplineWeightGradient(splineValue * nextDelta);
                }
            }
        }
    };

    /**
     * Update KAN connection weights using gradient descent with momentum.
     *
     * For each parameter p: p_new = p_old + lr * gradient + momentum * delta + acceleration * prev_delta
     */
    template<typename ValueType>
    struct KanWeightUpdater
    {
        template<typename LayerType, typename NextLayerType>
        static void update(LayerType& layer, const NextLayerType& nextLayer,
                          const ValueType& learningRate, const ValueType& momentumRate, const ValueType& accelerationRate)
        {
            (void)nextLayer;

            for (size_t neuron = 0; neuron < LayerType::NumberOfNeuronsInLayer; ++neuron)
            {
                typename LayerType::LayerNeuronType* pNeuron = layer.getNeuron(neuron);

                for (size_t conn = 0; conn < LayerType::LayerNeuronType::NumberOfOutgoingConnectionsFromNeuron; ++conn)
                {
                    typename LayerType::LayerNeuronType::NeuronConnectionType* pConn = pNeuron->getConnection(conn);

                    // Update spline coefficients
                    for (size_t c = 0; c < LayerType::LayerNeuronType::NumberOfCoefficientsPerConnection; ++c)
                    {
                        const ValueType prevDelta = pConn->getCoefficientPreviousDeltaWeight(c);
                        const ValueType currDelta = pConn->getCoefficientDeltaWeight(c);
                        const ValueType gradient = pConn->getCoefficientGradient(c);

                        const ValueType newDelta = (learningRate * gradient) + (momentumRate * currDelta) + (accelerationRate * prevDelta);

                        const ValueType currentCoeff = pConn->getCoefficient(c);
                        pConn->setCoefficientDeltaWeight(c, newDelta);
                        pConn->setCoefficient(c, currentCoeff + newDelta);
                    }

                    // Update base weight
                    {
                        const ValueType prevDelta = pConn->getBaseWeightPreviousDeltaWeight();
                        const ValueType currDelta = pConn->getBaseWeightDeltaWeight();
                        const ValueType gradient = pConn->getBaseWeightGradient();

                        const ValueType newDelta = (learningRate * gradient) + (momentumRate * currDelta) + (accelerationRate * prevDelta);

                        const ValueType currentWeight = pConn->getBaseWeight();
                        pConn->setBaseWeightDeltaWeight(newDelta);
                        pConn->setBaseWeight(currentWeight + newDelta);
                    }

                    // Update spline weight
                    {
                        const ValueType prevDelta = pConn->getSplineWeightPreviousDeltaWeight();
                        const ValueType currDelta = pConn->getSplineWeightDeltaWeight();
                        const ValueType gradient = pConn->getSplineWeightGradient();

                        const ValueType newDelta = (learningRate * gradient) + (momentumRate * currDelta) + (accelerationRate * prevDelta);

                        const ValueType currentWeight = pConn->getSplineWeight();
                        pConn->setSplineWeightDeltaWeight(newDelta);
                        pConn->setSplineWeight(currentWeight + newDelta);
                    }
                }
            }
        }
    };

    /**
     * KAN network deltas calculator - specialized on NumberOfInnerHiddenLayers.
     */
    template<typename NNType, size_t NumberOfInnerHiddenLayers, size_t SplineDegree>
    struct KanNetworkDeltasCalculator
    {
        typedef typename NNType::KanValueType ValueType;

        static void calculate(NNType& nn, ValueType const* const targetValues)
        {
            KanOutputLayerDeltasCalculator<typename NNType::KanTransferFunctionsPolicy, typename NNType::OutputLayerType>::calculate(
                nn.getOutputLayer(), targetValues);

            KanHiddenLayerDeltasCalculator<ValueType, SplineDegree>::calculate(
                nn.getLastHiddenLayer(), nn.getOutputLayer());

            typename NNType::InnerHiddenLayerType* pInnerLayers = nn.getPointerToInnerHiddenLayers();

            KanHiddenLayerDeltasCalculator<ValueType, SplineDegree>::calculate(
                pInnerLayers[NumberOfInnerHiddenLayers - 1], nn.getLastHiddenLayer());

            for (int i = static_cast<int>(NumberOfInnerHiddenLayers) - 2; i >= 0; --i)
            {
                KanHiddenLayerDeltasCalculator<ValueType, SplineDegree>::calculate(
                    pInnerLayers[i], pInnerLayers[i + 1]);
            }

            KanHiddenLayerDeltasCalculator<ValueType, SplineDegree>::calculate(
                nn.getInputLayer(), pInnerLayers[0]);
        }
    };

    template<typename NNType, size_t SplineDegree>
    struct KanNetworkDeltasCalculator<NNType, 0, SplineDegree>
    {
        typedef typename NNType::KanValueType ValueType;

        static void calculate(NNType& nn, ValueType const* const targetValues)
        {
            KanOutputLayerDeltasCalculator<typename NNType::KanTransferFunctionsPolicy, typename NNType::OutputLayerType>::calculate(
                nn.getOutputLayer(), targetValues);

            KanHiddenLayerDeltasCalculator<ValueType, SplineDegree>::calculate(
                nn.getLastHiddenLayer(), nn.getOutputLayer());

            KanHiddenLayerDeltasCalculator<ValueType, SplineDegree>::calculate(
                nn.getInputLayer(), nn.getLastHiddenLayer());
        }
    };

    /**
     * KAN network gradients calculator - specialized on NumberOfInnerHiddenLayers.
     */
    template<typename NNType, size_t NumberOfInnerHiddenLayers, size_t SplineDegree>
    struct KanNetworkGradientsCalculator
    {
        typedef typename NNType::KanValueType ValueType;

        static void calculate(NNType& nn)
        {
            typename NNType::InnerHiddenLayerType* pInnerLayers = nn.getPointerToInnerHiddenLayers();

            KanGradientsCalculator<ValueType, SplineDegree>::calculate(
                nn.getInputLayer(), pInnerLayers[0]);

            for (size_t i = 0; i < NumberOfInnerHiddenLayers - 1; ++i)
            {
                KanGradientsCalculator<ValueType, SplineDegree>::calculate(
                    pInnerLayers[i], pInnerLayers[i + 1]);
            }

            KanGradientsCalculator<ValueType, SplineDegree>::calculate(
                pInnerLayers[NumberOfInnerHiddenLayers - 1], nn.getLastHiddenLayer());

            KanGradientsCalculator<ValueType, SplineDegree>::calculate(
                nn.getLastHiddenLayer(), nn.getOutputLayer());
        }
    };

    template<typename NNType, size_t SplineDegree>
    struct KanNetworkGradientsCalculator<NNType, 0, SplineDegree>
    {
        typedef typename NNType::KanValueType ValueType;

        static void calculate(NNType& nn)
        {
            KanGradientsCalculator<ValueType, SplineDegree>::calculate(
                nn.getInputLayer(), nn.getLastHiddenLayer());

            KanGradientsCalculator<ValueType, SplineDegree>::calculate(
                nn.getLastHiddenLayer(), nn.getOutputLayer());
        }
    };

    /**
     * KAN network weight updater - specialized on NumberOfInnerHiddenLayers.
     */
    template<typename NNType, size_t NumberOfInnerHiddenLayers>
    struct KanNetworkWeightUpdater
    {
        typedef typename NNType::KanValueType ValueType;

        static void update(NNType& nn, const ValueType& lr, const ValueType& momentum, const ValueType& acceleration)
        {
            typename NNType::InnerHiddenLayerType* pInnerLayers = nn.getPointerToInnerHiddenLayers();

            KanWeightUpdater<ValueType>::update(nn.getInputLayer(), pInnerLayers[0], lr, momentum, acceleration);

            for (size_t i = 0; i < NumberOfInnerHiddenLayers - 1; ++i)
            {
                KanWeightUpdater<ValueType>::update(pInnerLayers[i], pInnerLayers[i + 1], lr, momentum, acceleration);
            }

            KanWeightUpdater<ValueType>::update(pInnerLayers[NumberOfInnerHiddenLayers - 1], nn.getLastHiddenLayer(), lr, momentum, acceleration);

            KanWeightUpdater<ValueType>::update(nn.getLastHiddenLayer(), nn.getOutputLayer(), lr, momentum, acceleration);
        }
    };

    template<typename NNType>
    struct KanNetworkWeightUpdater<NNType, 0>
    {
        typedef typename NNType::KanValueType ValueType;

        static void update(NNType& nn, const ValueType& lr, const ValueType& momentum, const ValueType& acceleration)
        {
            KanWeightUpdater<ValueType>::update(nn.getInputLayer(), nn.getLastHiddenLayer(), lr, momentum, acceleration);

            KanWeightUpdater<ValueType>::update(nn.getLastHiddenLayer(), nn.getOutputLayer(), lr, momentum, acceleration);
        }
    };

    /**
     * KAN backpropagation training policy.
     */
    template<typename TransferFunctionsPolicy, size_t BatchSize, size_t SplineDegree>
    struct KanBackPropagationPolicy
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        void initialize()
        {
            this->mLearningRate = TransferFunctionsPolicy::initialLearningRate();
            this->mMomentumRate = TransferFunctionsPolicy::initialMomentumRate();
            this->mAccelerationRate = TransferFunctionsPolicy::initialAccelerationRate();
        }

        ValueType getLearningRate() const { return this->mLearningRate; }
        ValueType getMomentumRate() const { return this->mMomentumRate; }
        ValueType getAccelerationRate() const { return this->mAccelerationRate; }

        void setLearningRate(const ValueType& value) { this->mLearningRate = value; }
        void setMomentumRate(const ValueType& value) { this->mMomentumRate = value; }
        void setAccelerationRate(const ValueType& value) { this->mAccelerationRate = value; }

        template<typename NNType>
        void trainNetwork(NNType& nn, ValueType const* const targetValues)
        {
            KanNetworkDeltasCalculator<NNType, NNType::NumberOfInnerHiddenLayers, SplineDegree>::calculate(nn, targetValues);

            KanNetworkGradientsCalculator<NNType, NNType::NumberOfInnerHiddenLayers, SplineDegree>::calculate(nn);

            KanNetworkWeightUpdater<NNType, NNType::NumberOfInnerHiddenLayers>::update(
                nn, this->mLearningRate, this->mMomentumRate, this->mAccelerationRate);
        }

    private:
        ValueType mLearningRate;
        ValueType mMomentumRate;
        ValueType mAccelerationRate;

        static_assert(BatchSize > 0, "Invalid batch size.");
    };

    /**
     * Null training policy for inference-only KAN.
     */
    template<typename TransferFunctionsPolicy, size_t BatchSize, size_t SplineDegree>
    struct KanNullTrainingPolicy
    {
        typedef typename TransferFunctionsPolicy::TransferFunctionsValueType ValueType;

        void initialize() {}
        ValueType getLearningRate() const { return static_cast<ValueType>(0); }
        ValueType getMomentumRate() const { return static_cast<ValueType>(0); }
        ValueType getAccelerationRate() const { return static_cast<ValueType>(0); }
        void setLearningRate(const ValueType& value) { (void)value; }
        void setMomentumRate(const ValueType& value) { (void)value; }
        void setAccelerationRate(const ValueType& value) { (void)value; }

        template<typename NNType>
        void trainNetwork(NNType& nn, ValueType const* const targetValues)
        {
            (void)nn;
            (void)targetValues;
        }
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize, size_t SplineDegree, bool IsTrainable>
    struct KanTrainingPolicySelector
    {
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize, size_t SplineDegree>
    struct KanTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, SplineDegree, true>
    {
        typedef KanBackPropagationPolicy<TransferFunctionsPolicy, BatchSize, SplineDegree> PolicyType;
    };

    template<typename TransferFunctionsPolicy, size_t BatchSize, size_t SplineDegree>
    struct KanTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, SplineDegree, false>
    {
        typedef KanNullTrainingPolicy<TransferFunctionsPolicy, BatchSize, SplineDegree> PolicyType;
    };

    // =========================================================================
    // Top-Level KAN Network Class
    // =========================================================================

    /**
     * Kolmogorov-Arnold Network.
     *
     * A neural network where learnable activation functions (B-splines) are
     * on the edges and nodes are pure summation. This is fundamentally different
     * from MLPs which have fixed activations on nodes and scalar weights on edges.
     *
     * Template parameters:
     *   ValueType                   - QValue<...> or float/double
     *   NumberOfInputs              - Number of input neurons
     *   NumberOfHiddenLayers        - Number of hidden layers (>= 1)
     *   NumberOfNeuronsInHiddenLayers - Neurons per hidden layer
     *   NumberOfOutputs             - Number of output neurons
     *   TransferFunctionsPolicy     - KanTransferFunctions<...>
     *   IsTrainable                 - Enable training (default true)
     *   BatchSize                   - Gradient accumulation batch size (default 1)
     *   GridSize                    - B-spline grid intervals (default 5)
     *   SplineDegree                - B-spline polynomial degree (default 3)
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
            size_t GridSize = 5,
            size_t SplineDegree = 3
            >
    class KolmogorovArnoldNetwork
    {
    public:
        typedef KolmogorovArnoldNetwork<ValueType, NumberOfInputs, NumberOfHiddenLayers,
                NumberOfNeuronsInHiddenLayers, NumberOfOutputs, TransferFunctionsPolicy,
                IsTrainable, BatchSize, GridSize, SplineDegree> KanNetworkType;

        typedef ValueType KanValueType;
        typedef TransferFunctionsPolicy KanTransferFunctionsPolicy;

        // Connection type
        typedef typename KanConnectionTypeSelector<ValueType, GridSize, SplineDegree, IsTrainable>::ConnectionType ConnectionType;

        // Neuron types
        typedef typename KanInputLayerNeuronTypeSelector<ConnectionType, NumberOfNeuronsInHiddenLayers, TransferFunctionsPolicy, IsTrainable>::NeuronType InputLayerNeuronType;

        // Inner hidden layer neurons connect to next hidden layer (same width)
        typedef typename KanHiddenLayerNeuronTypeSelector<ConnectionType, NumberOfNeuronsInHiddenLayers, TransferFunctionsPolicy, IsTrainable>::NeuronType InnerHiddenLayerNeuronType;

        // Last hidden layer neurons connect to output layer
        typedef typename KanHiddenLayerNeuronTypeSelector<ConnectionType, NumberOfOutputs, TransferFunctionsPolicy, IsTrainable>::NeuronType LastHiddenLayerNeuronType;

        typedef KanOutputNeuron<TransferFunctionsPolicy> OutputNeuronType;

        // Layer types
        typedef KanInputLayer<InputLayerNeuronType, NumberOfInputs, GridSize, SplineDegree> InputLayerType;
        typedef KanHiddenLayer<InnerHiddenLayerNeuronType, NumberOfNeuronsInHiddenLayers, GridSize, SplineDegree> InnerHiddenLayerType;
        typedef KanHiddenLayer<LastHiddenLayerNeuronType, NumberOfNeuronsInHiddenLayers, GridSize, SplineDegree> LastHiddenLayerType;
        typedef KanOutputLayer<OutputNeuronType, NumberOfOutputs> OutputLayerType;

        // Training policy
        typedef typename KanTrainingPolicySelector<TransferFunctionsPolicy, BatchSize, SplineDegree, IsTrainable>::PolicyType TrainingPolicyType;

        // Constants
        static const size_t NumberOfInputLayerNeurons = NumberOfInputs;
        static const size_t NumberOfHiddenLayerNeurons = NumberOfNeuronsInHiddenLayers;
        static const size_t NumberOfOutputLayerNeurons = NumberOfOutputs;
        static const size_t NumberOfInnerHiddenLayers = NumberOfHiddenLayers - 1;
        static const size_t KanGridSize = GridSize;
        static const size_t KanSplineDegree = SplineDegree;

        KolmogorovArnoldNetwork()
        {
            // Initialize layers
            this->mInputLayer.initializeNeurons();
            this->mInnerHiddenLayerManager.initializeNeurons();
            this->mLastHiddenLayer.initializeNeurons();
            this->mOutputLayer.initializeNeurons();

            // Initialize knot vectors for all layers
            const ValueType gridMin = Constants<ValueType>::negativeOne();
            const ValueType gridMax = Constants<ValueType>::one();
            this->mInputLayer.initializeKnots(gridMin, gridMax);
            this->mInnerHiddenLayerManager.initializeKnots(gridMin, gridMax);
            this->mLastHiddenLayer.initializeKnots(gridMin, gridMax);

            // Initialize weights
            this->mInputLayer.initializeWeights();
            this->mInnerHiddenLayerManager.initializeWeights();
            this->mLastHiddenLayer.initializeWeights();

            // Initialize training policy
            this->mTrainingPolicy.initialize();

            // Initialize learned values buffer
            size_t bufferIndex;
            for (size_t i = 0; i < NumberOfOutputLayerNeurons; ++i)
            {
                bufferIndex = i * sizeof(ValueType);
                new (&this->mLearnedValuesBuffer[bufferIndex]) ValueType();
            }
        }

        void feedForward(ValueType const* const values)
        {
            this->mInputLayer.feedForward(values);
            this->feedForwardHiddenLayers();
            this->mOutputLayer.feedForward(this->mLastHiddenLayer);
        }

        ValueType calculateError(ValueType const* const targetValues)
        {
            ValueType* pLearnedValues = reinterpret_cast<ValueType*>(&this->mLearnedValuesBuffer[0]);
            getLearnedValues(pLearnedValues);
            return TransferFunctionsPolicy::calculateError(targetValues, pLearnedValues);
        }

        void trainNetwork(ValueType const* const targetValues)
        {
            this->mTrainingPolicy.trainNetwork(*this, targetValues);
        }

        void getLearnedValues(ValueType* output) const
        {
            for (size_t neuron = 0; neuron < NumberOfOutputLayerNeurons; ++neuron)
            {
                output[neuron] = this->mOutputLayer.getOutputValueForNeuron(neuron);
            }
        }

        // Layer accessors
        InputLayerType& getInputLayer() { return this->mInputLayer; }
        LastHiddenLayerType& getLastHiddenLayer() { return this->mLastHiddenLayer; }
        OutputLayerType& getOutputLayer() { return this->mOutputLayer; }

        InnerHiddenLayerType* getPointerToInnerHiddenLayers()
        {
            return this->mInnerHiddenLayerManager.getPointerToLayers();
        }

        // Training rate accessors
        ValueType getLearningRate() const { return this->mTrainingPolicy.getLearningRate(); }
        ValueType getMomentumRate() const { return this->mTrainingPolicy.getMomentumRate(); }
        ValueType getAccelerationRate() const { return this->mTrainingPolicy.getAccelerationRate(); }
        void setLearningRate(const ValueType& value) { this->mTrainingPolicy.setLearningRate(value); }
        void setMomentumRate(const ValueType& value) { this->mTrainingPolicy.setMomentumRate(value); }
        void setAccelerationRate(const ValueType& value) { this->mTrainingPolicy.setAccelerationRate(value); }

    protected:
        void feedForwardHiddenLayers()
        {
            KanHiddenLayerFeedForwardManager<InputLayerType, InnerHiddenLayerType, LastHiddenLayerType, NumberOfInnerHiddenLayers>::feedForward(
                this->mInputLayer, this->mInnerHiddenLayerManager.getPointerToLayers(), this->mLastHiddenLayer);
        }

        InputLayerType mInputLayer;
        KanInnerHiddenLayerManager<InnerHiddenLayerType, NumberOfInnerHiddenLayers> mInnerHiddenLayerManager;
        LastHiddenLayerType mLastHiddenLayer;
        OutputLayerType mOutputLayer;
        TrainingPolicyType mTrainingPolicy;

    private:
        unsigned char mLearnedValuesBuffer[NumberOfOutputLayerNeurons * sizeof(ValueType)];

        KolmogorovArnoldNetwork(const KolmogorovArnoldNetwork&) {}
        KolmogorovArnoldNetwork& operator=(const KolmogorovArnoldNetwork&) { return *this; }

        static_assert(NumberOfInputs > 0, "Invalid number of inputs.");
        static_assert(NumberOfHiddenLayers > 0, "Invalid number of hidden layers.");
        static_assert(NumberOfNeuronsInHiddenLayers > 0, "Invalid number of neurons in hidden layers.");
        static_assert(NumberOfOutputs > 0, "Invalid number of outputs.");
        static_assert(NumberOfOutputLayerNeurons == TransferFunctionsPolicy::NumberOfTransferFunctionsOutputNeurons, "TransferFunctionPolicy NumberOfOutputNeurons is incorrect.");
    };
}
