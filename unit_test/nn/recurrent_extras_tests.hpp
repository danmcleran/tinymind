/**
* Copyright (c) 2026 Dan McLeran
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

// Included from nn_unit_test.cpp, after its helpers (FloatingPointTransferFunctions,
// UniformRealRandomNumberGenerator, CellStateAccess, RecurrentGradientCheckTransferFunctions).
//
// Stacked recurrent layers, BatchSize > 1, setWeights, weight files and
// resetState: every trainable weight of the network, enumerated, is checked
// against central finite differences, copied, saved and restored.
//
// This file is included inside a test suite (a namespace), so it includes no
// headers itself: <functional>, <map>, <string>, <vector>, <cmath> and
// <algorithm> come from nn_unit_test.cpp and Boost.Test.

namespace recurrent_extras
{
    struct Param
    {
        std::string group;
        std::function<double()> get;
        std::function<void(double)> set;
    };

    template<typename T>
    inline double toDouble(const T& value)
    {
        return static_cast<double>(value.getValue()) / static_cast<double>(1ULL << T::NumberOfFractionalBits);
    }

    template<>
    inline double toDouble<double>(const double& value)
    {
        return value;
    }

    template<typename T>
    inline T fromDouble(const double value)
    {
        return tinymind::ValueConverter<double, T>::convertToDestinationType(value);
    }

    template<>
    inline double fromDouble<double>(const double value)
    {
        return value;
    }

    // Recurrent weights exist only on a recurrent network, and inner recurrent
    // weights only on a stacked one. Choosing at compile time keeps every
    // library accessor that gets instantiated also executed (coverage gate).
    template<bool HasInnerRecurrentLayers>
    struct InnerRecurrentParams
    {
        template<typename NN>
        static void add(NN& nn, std::vector<Param>& params, const size_t layer, const size_t gates)
        {
            typedef typename NN::NeuralNetworkValueType V;
            const size_t size = NN::getHiddenLayerSize(layer);
            for (size_t n = 0; n < size; ++n)
            {
                for (size_t c = 0; c < gates * size; ++c)
                {
                    params.push_back({"inner_rec",
                                      [&nn, layer, n, c]() { return toDouble(nn.getInnerRecurrentLayerWeightForNeuronAndConnection(layer, n, c)); },
                                      [&nn, layer, n, c](double v) { nn.setInnerRecurrentLayerWeightForNeuronAndConnection(layer, n, c, fromDouble<V>(v)); }});
                }
            }
        }
    };

    template<>
    struct InnerRecurrentParams<false>
    {
        template<typename NN>
        static void add(NN&, std::vector<Param>&, const size_t, const size_t)
        {
        }
    };

    template<bool HasRecurrentLayer>
    struct LastRecurrentParams
    {
        template<typename NN>
        static void add(NN& nn, std::vector<Param>& params, const size_t gates)
        {
            typedef typename NN::NeuralNetworkValueType V;
            const size_t size = NN::getHiddenLayerSize(NN::NeuralNetworkNumberOfHiddenLayers - 1);
            for (size_t n = 0; n < size; ++n)
            {
                for (size_t c = 0; c < gates * size; ++c)
                {
                    params.push_back({"rec",
                                      [&nn, n, c]() { return toDouble(nn.getRecurrentLayer().getWeightForNeuronAndConnection(n, c)); },
                                      [&nn, n, c](double v) { nn.getRecurrentLayer().setWeightForNeuronAndConnection(n, c, fromDouble<V>(v)); }});
                }
            }
        }
    };

    template<>
    struct LastRecurrentParams<false>
    {
        template<typename NN>
        static void add(NN&, std::vector<Param>&, const size_t)
        {
        }
    };

    // Every trainable weight of the network, grouped by role.
    template<typename NN>
    std::vector<Param> enumerateParams(NN& nn)
    {
        typedef typename NN::NeuralNetworkValueType V;
        typedef InnerRecurrentParams<NN::NeuralNetworkHasRecurrentLayer && (NN::NeuralNetworkNumberOfHiddenLayers > 1)> InnerRec;
        typedef LastRecurrentParams<NN::NeuralNetworkHasRecurrentLayer> LastRec;
        static const size_t G = NN::HiddenLayerGateMultiplier;
        static const size_t L = NN::NeuralNetworkNumberOfHiddenLayers;
        std::vector<Param> params;

        const size_t first = G * NN::getHiddenLayerSize(0);
        for (size_t i = 0; i < NN::NumberOfInputLayerNeurons; ++i)
        {
            for (size_t c = 0; c < first; ++c)
            {
                params.push_back({"in_w",
                                  [&nn, i, c]() { return toDouble(nn.getInputLayerWeightForNeuronAndConnection(i, c)); },
                                  [&nn, i, c](double v) { nn.setInputLayerWeightForNeuronAndConnection(i, c, fromDouble<V>(v)); }});
            }
        }
        for (size_t c = 0; c < first; ++c)
        {
            params.push_back({"in_b",
                              [&nn, c]() { return toDouble(nn.getInputLayerBiasNeuronWeightForConnection(c)); },
                              [&nn, c](double v) { nn.setInputLayerBiasWeightForConnection(c, fromDouble<V>(v)); }});
        }
        for (size_t l = 0; l + 1 < L; ++l)
        {
            const size_t size = NN::getHiddenLayerSize(l);
            const size_t next = G * NN::getHiddenLayerSize(l + 1);
            for (size_t n = 0; n < size; ++n)
            {
                for (size_t c = 0; c < next; ++c)
                {
                    params.push_back({"inner_w",
                                      [&nn, l, n, c]() { return toDouble(nn.getHiddenLayerWeightForNeuronAndConnection(l, n, c)); },
                                      [&nn, l, n, c](double v) { nn.setHiddenLayerWeightForNeuronAndConnection(l, n, c, fromDouble<V>(v)); }});
                }
            }
            for (size_t c = 0; c < next; ++c)
            {
                params.push_back({"inner_b",
                                  [&nn, l, c]() { return toDouble(nn.getHiddenLayerBiasNeuronWeightForConnection(l, c)); },
                                  [&nn, l, c](double v) { nn.setHiddenLayerBiasNeuronWeightForConnection(l, c, fromDouble<V>(v)); }});
            }
            InnerRec::add(nn, params, l, G);
        }
        LastRec::add(nn, params, G);
        for (size_t n = 0; n < NN::NumberOfHiddenLayerNeurons; ++n)
        {
            for (size_t o = 0; o < NN::NumberOfOutputLayerNeurons; ++o)
            {
                params.push_back({"out_w",
                                  [&nn, n, o]() { return toDouble(nn.getHiddenLayerWeightForNeuronAndConnection(L - 1, n, o)); },
                                  [&nn, n, o](double v) { nn.setHiddenLayerWeightForNeuronAndConnection(L - 1, n, o, fromDouble<V>(v)); }});
            }
        }
        for (size_t o = 0; o < NN::NumberOfOutputLayerNeurons; ++o)
        {
            params.push_back({"out_b",
                              [&nn, o]() { return toDouble(nn.getHiddenLayerBiasNeuronWeightForConnection(L - 1, o)); },
                              [&nn, o](double v) { nn.setHiddenLayerBiasNeuronWeightForConnection(L - 1, o, fromDouble<V>(v)); }});
        }
        return params;
    }

    inline std::vector<double> values(const std::vector<Param>& params)
    {
        std::vector<double> result;
        for (size_t p = 0; p < params.size(); ++p)
        {
            result.push_back(params[p].get());
        }
        return result;
    }

    // h (and an LSTM's cell state) of every hidden layer.
    template<size_t Gates>
    struct HiddenState
    {
        std::vector<double> h;
        std::vector<double> c;

        template<typename LayerType>
        void saveLayer(LayerType& layer)
        {
            for (size_t n = 0; n < LayerType::NumberOfNeuronsInLayer; ++n)
            {
                h.push_back(layer.getNeuron(n)->getOutputValue());
                c.push_back(CellStateAccess<Gates>::get(layer.getNeuron(n)));
            }
        }

        template<typename LayerType>
        void loadLayer(LayerType& layer, size_t& cursor) const
        {
            for (size_t n = 0; n < LayerType::NumberOfNeuronsInLayer; ++n, ++cursor)
            {
                layer.getNeuron(n)->setOutputValue(h[cursor]);
                CellStateAccess<Gates>::set(layer.getNeuron(n), c[cursor]);
            }
        }

        void saveChain(tinymind::EmptyLayerChain&)
        {
        }

        template<typename LayerType, typename RestType>
        void saveChain(tinymind::LayerChain<LayerType, RestType>& chain)
        {
            saveLayer(chain.layer);
            saveChain(chain.rest);
        }

        void loadChain(tinymind::EmptyLayerChain&, size_t&) const
        {
        }

        template<typename LayerType, typename RestType>
        void loadChain(tinymind::LayerChain<LayerType, RestType>& chain, size_t& cursor) const
        {
            loadLayer(chain.layer, cursor);
            loadChain(chain.rest, cursor);
        }

        template<typename NN>
        void save(NN& nn)
        {
            h.clear();
            c.clear();
            saveChain(nn.getInnerHiddenLayerChain());
            saveLayer(nn.getLastHiddenLayer());
        }

        template<typename NN>
        void load(NN& nn) const
        {
            size_t cursor = 0;
            nn.resetState();
            loadChain(nn.getInnerHiddenLayerChain(), cursor);
            loadLayer(nn.getLastHiddenLayer(), cursor);
        }
    };

    inline double randomUnit()
    {
        return (rand() % 2001 - 1000) / 1000.0;
    }

    // One training step from a warm state against central finite differences
    // of the same step, every weight of every layer (one-step truncation:
    // each layer's h_{t-1} and c_{t-1} are inputs). Every group must move and
    // match to a relative 1e-5.
    template<typename NN>
    void checkGradients(NN& nn)
    {
        static const size_t I = NN::NumberOfInputLayerNeurons;
        static const size_t O = NN::NumberOfOutputLayerNeurons;
        static const size_t G = NN::HiddenLayerGateMultiplier;
        HiddenState<G> state;
        double warm[I], x[I], target[O];

        for (int step = 0; step < 3; ++step)
        {
            for (size_t i = 0; i < I; ++i) warm[i] = randomUnit();
            nn.feedForward(&warm[0]);
        }
        state.save(nn);
        for (size_t i = 0; i < I; ++i) x[i] = randomUnit();
        for (size_t o = 0; o < O; ++o) target[o] = 0.9 * randomUnit();

        auto loss = [&]()
        {
            double out[O];
            state.load(nn);
            nn.feedForward(&x[0]);
            nn.getLearnedValues(&out[0]);
            double l = 0;
            for (size_t o = 0; o < O; ++o) l += 0.5 * (target[o] - out[o]) * (target[o] - out[o]);
            return l;
        };

        std::vector<Param> params = enumerateParams(nn);
        const double lr = 1e-3;
        const double eps = 1e-6;
        std::vector<double> before(params.size());
        std::vector<double> expected(params.size());
        double largest = 0;
        for (size_t p = 0; p < params.size(); ++p)
        {
            const double w = params[p].get();
            before[p] = w;
            params[p].set(w + eps);
            const double lp = loss();
            params[p].set(w - eps);
            const double lm = loss();
            params[p].set(w);
            expected[p] = -lr * (lp - lm) / (2.0 * eps);
            largest = std::max(largest, std::fabs(expected[p]));
        }

        nn.setLearningRate(lr);
        nn.setMomentumRate(0.0);
        nn.setAccelerationRate(0.0);
        state.load(nn);
        nn.feedForward(&x[0]);
        nn.trainNetwork(&target[0]);

        std::map<std::string, double> worst;
        std::map<std::string, size_t> moved;
        std::map<std::string, size_t> count;
        const double floor = 1e-3 * largest;
        for (size_t p = 0; p < params.size(); ++p)
        {
            const double actual = params[p].get() - before[p];
            const double rel = std::fabs(actual - expected[p]) / std::max(std::fabs(expected[p]), floor);
            worst[params[p].group] = std::max(worst[params[p].group], rel);
            ++count[params[p].group];
            if (actual != 0.0)
            {
                ++moved[params[p].group];
            }
        }
        for (std::map<std::string, size_t>::const_iterator it = count.begin(); it != count.end(); ++it)
        {
            BOOST_TEST_INFO("group " << it->first);
            BOOST_TEST(worst[it->first] < 1e-5);
            BOOST_TEST_INFO("group " << it->first);
            BOOST_TEST(moved[it->first] == it->second);
        }
    }

    // Outputs of a fixed input sequence.
    template<typename NN>
    std::vector<double> runSequence(NN& nn, const std::vector<std::vector<double> >& inputs)
    {
        typedef typename NN::NeuralNetworkValueType V;
        std::vector<double> result;
        for (size_t t = 0; t < inputs.size(); ++t)
        {
            V x[NN::NumberOfInputLayerNeurons];
            V y[NN::NumberOfOutputLayerNeurons];
            for (size_t i = 0; i < NN::NumberOfInputLayerNeurons; ++i) x[i] = fromDouble<V>(inputs[t][i]);
            nn.feedForward(&x[0]);
            nn.getLearnedValues(&y[0]);
            for (size_t o = 0; o < NN::NumberOfOutputLayerNeurons; ++o) result.push_back(toDouble(y[o]));
        }
        return result;
    }

    inline std::vector<std::vector<double> > randomSequence(const size_t length, const size_t width)
    {
        std::vector<std::vector<double> > seq(length, std::vector<double>(width));
        for (size_t t = 0; t < length; ++t)
            for (size_t i = 0; i < width; ++i)
                seq[t][i] = randomUnit();
        return seq;
    }

    // resetState must return every layer to the freshly constructed state.
    template<typename NN>
    void checkReset(NN& nn)
    {
        const std::vector<std::vector<double> > seq = randomSequence(5, NN::NumberOfInputLayerNeurons);
        const std::vector<double> fresh = runSequence(nn, seq);
        runSequence(nn, randomSequence(7, NN::NumberOfInputLayerNeurons));
        nn.resetState();
        const std::vector<double> again = runSequence(nn, seq);
        BOOST_TEST(fresh == again);
    }

    // setWeights must copy every weight: same weights, same outputs.
    template<typename NN>
    void checkSetWeights(NN& a, NN& b)
    {
        typedef typename NN::NeuralNetworkValueType V;
        V x[NN::NumberOfInputLayerNeurons];
        V t[NN::NumberOfOutputLayerNeurons];
        for (int step = 0; step < 20; ++step)
        {
            for (size_t i = 0; i < NN::NumberOfInputLayerNeurons; ++i) x[i] = fromDouble<V>(randomUnit());
            for (size_t o = 0; o < NN::NumberOfOutputLayerNeurons; ++o) t[o] = fromDouble<V>(0.5 * randomUnit());
            a.feedForward(&x[0]);
            a.trainNetwork(&t[0]);
        }
        b.setWeights(a);

        std::vector<Param> pa = enumerateParams(a);
        std::vector<Param> pb = enumerateParams(b);
        size_t differ = 0;
        for (size_t p = 0; p < pa.size(); ++p)
        {
            if (pa[p].get() != pb[p].get())
            {
                ++differ;
            }
        }
        BOOST_TEST(differ == 0u);

        const std::vector<std::vector<double> > seq = randomSequence(6, NN::NumberOfInputLayerNeurons);
        a.resetState();
        b.resetState();
        BOOST_TEST(runSequence(a, seq) == runSequence(b, seq));
    }

    // A stored then loaded weight file must reproduce every weight exactly.
    template<typename FileManager, typename NN>
    void checkWeightFile(NN& a, NN& b, char const* const path)
    {
        typedef typename NN::NeuralNetworkValueType V;
        {
            std::ofstream out(path);
            FileManager::storeNetworkWeights(a, out);
        }
        {
            std::ifstream in(path);
            FileManager::template loadNetworkWeights<V, V>(b, in);
        }
        std::vector<Param> pa = enumerateParams(a);
        std::vector<Param> pb = enumerateParams(b);
        size_t differ = 0;
        for (size_t p = 0; p < pa.size(); ++p)
        {
            if (pa[p].get() != pb[p].get())
            {
                ++differ;
            }
        }
        BOOST_TEST(differ == 0u);
    }

    // BatchSize B: after B samples the weights must equal w0 plus the mean of
    // the B one-step updates a BatchSize-1 network makes from w0 with the same
    // states, and must not move before then. Two batches in a row.
    template<typename BatchNN, typename SingleNN>
    void checkBatchMatchesMeanUpdate(BatchNN& batched, SingleNN& single)
    {
        static const size_t B = BatchNN::NeuralNetworkBatchSize;
        static const size_t I = BatchNN::NumberOfInputLayerNeurons;
        static const size_t O = BatchNN::NumberOfOutputLayerNeurons;
        std::vector<Param> pb = enumerateParams(batched);
        std::vector<Param> ps = enumerateParams(single);
        BOOST_REQUIRE(pb.size() == ps.size());
        // Both networks start from the same weights.
        for (size_t p = 0; p < ps.size(); ++p)
        {
            const double w = pb[p].get();
            pb[p].set(w);
            ps[p].set(w);
        }
        batched.setLearningRate(0.1);
        batched.setMomentumRate(0.0);
        batched.setAccelerationRate(0.0);
        single.setLearningRate(0.1);
        single.setMomentumRate(0.0);
        single.setAccelerationRate(0.0);

        for (int batch = 0; batch < 2; ++batch)
        {
            const std::vector<double> w0 = values(pb);
            std::vector<double> sum(w0.size(), 0.0);
            for (size_t s = 0; s < B; ++s)
            {
                double x[I];
                double t[O];
                for (size_t i = 0; i < I; ++i) x[i] = randomUnit();
                for (size_t o = 0; o < O; ++o) t[o] = 0.9 * randomUnit();

                single.feedForward(&x[0]);
                single.trainNetwork(&t[0]);
                for (size_t p = 0; p < ps.size(); ++p)
                {
                    sum[p] += ps[p].get() - w0[p];
                    ps[p].set(w0[p]);
                }

                batched.feedForward(&x[0]);
                batched.trainNetwork(&t[0]);
                double worst = 0;
                for (size_t p = 0; p < pb.size(); ++p)
                {
                    const double expected = (s + 1 < B) ? w0[p] : (w0[p] + sum[p] / static_cast<double>(B));
                    worst = std::max(worst, std::fabs(pb[p].get() - expected));
                }
                BOOST_TEST_INFO("batch " << batch << " sample " << s);
                BOOST_TEST(worst < 1e-12);
            }
            for (size_t p = 0; p < ps.size(); ++p)
            {
                ps[p].set(pb[p].get());
            }
        }
    }

    typedef RecurrentGradientCheckTransferFunctions DoubleTF;

    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> Q16;
    typedef tinymind::FixedPointTransferFunctions<
                                                    Q16,
                                                    UniformRealRandomNumberGenerator<Q16>,
                                                    tinymind::TanhActivationPolicy<Q16>,
                                                    tinymind::TanhActivationPolicy<Q16> > Q16TF;
} // namespace recurrent_extras

BOOST_AUTO_TEST_SUITE(test_suite_recurrent_extras)

BOOST_AUTO_TEST_CASE(test_case_stacked_lstm_gradients_match_finite_differences)
{
    srand(RANDOM_SEED);
    static tinymind::LstmNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> nn;
    recurrent_extras::checkGradients(nn);
}

BOOST_AUTO_TEST_CASE(test_case_stacked_gru_gradients_match_finite_differences)
{
    srand(RANDOM_SEED);
    static tinymind::GruNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> nn;
    recurrent_extras::checkGradients(nn);
}

BOOST_AUTO_TEST_CASE(test_case_stacked_elman_gradients_match_finite_differences)
{
    srand(RANDOM_SEED);
    static tinymind::RecurrentNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> nn;
    recurrent_extras::checkGradients(nn);
}

BOOST_AUTO_TEST_CASE(test_case_three_layer_lstm_gradients_match_finite_differences)
{
    srand(RANDOM_SEED);
    static tinymind::LstmNeuralNetwork<double, 3, tinymind::HiddenLayers<3, 4, 2>, 2, recurrent_extras::DoubleTF> nn;
    recurrent_extras::checkGradients(nn);
}

BOOST_AUTO_TEST_CASE(test_case_stacked_inner_recurrent_weights_initialized)
{
    // The inner layers' recurrent weights get random initial values like
    // every other weight, so the inner layers are recurrent from the start.
    srand(RANDOM_SEED);
    static tinymind::LstmNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> nn;
    std::vector<recurrent_extras::Param> params = recurrent_extras::enumerateParams(nn);
    size_t inner = 0;
    size_t nonZero = 0;
    for (size_t p = 0; p < params.size(); ++p)
    {
        if (params[p].group == "inner_rec")
        {
            ++inner;
            if (params[p].get() != 0.0)
            {
                ++nonZero;
            }
        }
    }
    BOOST_TEST(inner == 64u);
    BOOST_TEST(nonZero == inner);
}

BOOST_AUTO_TEST_CASE(test_case_stacked_reset_state)
{
    srand(RANDOM_SEED);
    static tinymind::LstmNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> lstm;
    recurrent_extras::checkReset(lstm);
    static tinymind::GruNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> gru;
    recurrent_extras::checkReset(gru);
    static tinymind::RecurrentNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> elman;
    recurrent_extras::checkReset(elman);
}

BOOST_AUTO_TEST_CASE(test_case_set_weights_copies_every_weight)
{
    srand(RANDOM_SEED);
    {
        typedef tinymind::LstmNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> NN;
        static NN a;
        static NN b;
        recurrent_extras::checkSetWeights(a, b);
    }
    {
        typedef tinymind::GruNeuralNetwork<double, 3, tinymind::HiddenLayers<4>, 2, recurrent_extras::DoubleTF> NN;
        static NN a;
        static NN b;
        recurrent_extras::checkSetWeights(a, b);
    }
    {
        typedef tinymind::RecurrentNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> NN;
        static NN a;
        static NN b;
        recurrent_extras::checkSetWeights(a, b);
    }
    {
        typedef tinymind::NeuralNetwork<double, 3, tinymind::HiddenLayers<5, 3>, 2, recurrent_extras::DoubleTF> NN;
        static NN a;
        static NN b;
        recurrent_extras::checkSetWeights(a, b);
    }
}

BOOST_AUTO_TEST_CASE(test_case_weight_files_round_trip)
{
    srand(RANDOM_SEED);
    {
        typedef tinymind::LstmNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> NN;
        static NN a;
        static NN b;
        recurrent_extras::checkWeightFile<tinymind::RecurrentNetworkPropertiesFileManager<NN> >(a, b, "output/recurrent_extras_lstm_weights.txt");
    }
    {
        typedef tinymind::RecurrentNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> NN;
        static NN a;
        static NN b;
        recurrent_extras::checkWeightFile<tinymind::NetworkPropertiesFileManager<NN> >(a, b, "output/recurrent_extras_elman_weights.txt");
    }
    {
        typedef tinymind::NeuralNetwork<double, 3, tinymind::HiddenLayers<5, 3>, 2, recurrent_extras::DoubleTF> NN;
        static NN a;
        static NN b;
        recurrent_extras::checkWeightFile<tinymind::NetworkPropertiesFileManager<NN> >(a, b, "output/recurrent_extras_mlp_weights.txt");
    }
    {
        typedef tinymind::GruNeuralNetwork<recurrent_extras::Q16, 2, tinymind::HiddenLayers<4, 3>, 1, recurrent_extras::Q16TF> NN;
        static NN a;
        static NN b;
        recurrent_extras::checkWeightFile<tinymind::RecurrentNetworkPropertiesFileManager<NN> >(a, b, "output/recurrent_extras_gru_q16_weights.txt");
    }
}

BOOST_AUTO_TEST_CASE(test_case_batch_size_applies_mean_gradient_once_per_batch)
{
    srand(RANDOM_SEED);
    {
        static tinymind::NeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF, true, 3> batched;
        static tinymind::NeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> single;
        recurrent_extras::checkBatchMatchesMeanUpdate(batched, single);
    }
    {
        static tinymind::LstmNeuralNetwork<double, 3, tinymind::HiddenLayers<4>, 2, recurrent_extras::DoubleTF, true, 2> batched;
        static tinymind::LstmNeuralNetwork<double, 3, tinymind::HiddenLayers<4>, 2, recurrent_extras::DoubleTF> single;
        recurrent_extras::checkBatchMatchesMeanUpdate(batched, single);
    }
    {
        static tinymind::GruNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF, true, 2> batched;
        static tinymind::GruNeuralNetwork<double, 3, tinymind::HiddenLayers<4, 3>, 2, recurrent_extras::DoubleTF> single;
        recurrent_extras::checkBatchMatchesMeanUpdate(batched, single);
    }
    {
        static tinymind::RecurrentNeuralNetwork<double, 3, tinymind::HiddenLayers<4>, 2, recurrent_extras::DoubleTF, true, 2> batched;
        static tinymind::RecurrentNeuralNetwork<double, 3, tinymind::HiddenLayers<4>, 2, recurrent_extras::DoubleTF> single;
        recurrent_extras::checkBatchMatchesMeanUpdate(batched, single);
    }
}

BOOST_AUTO_TEST_SUITE_END()
