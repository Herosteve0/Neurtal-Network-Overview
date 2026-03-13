using System;
using System.Security.Cryptography.X509Certificates;

namespace NeuralNetworkSystem {
    public class Layer {
        float WeightScaler(int previousLength) {
            return (float)Math.Sqrt(6f / previousLength);
        }

        public Layer(int index, int size, LayerFunctions functions, INetworkFunctions network_functions) {
            this.index = index;
            NeuronNum = size;
            Bias = new Vector(size);

            NetworkFunctions = network_functions;
            _NormalizeInput = functions.normalization;
            _OutputActivation = functions.output_activation;

            ActivationFunction = functions.activation_function;
            ActivationFunctionDerivative = functions.activation_function_derivative;
        }
        public Layer(int index, int size, Layer previousLayer, LayerFunctions functions, INetworkFunctions network_functions) : this(index, size, functions, network_functions) { // [to, from]
            float value = WeightScaler(previousLayer.NeuronNum);
            Weights = Matrix.Random(size, previousLayer.NeuronNum, -value, value);
            WeightsT = new Matrix(previousLayer.NeuronNum, size);
            Weights.Transpose(WeightsT);
            PreviousLayer = previousLayer;
            previousLayer.NextLayer = this;
        }

        public int index { get; }
        public int NeuronNum { get; }
        public Vector Bias { get; set; }
        public Matrix Weights { get; set; }
        public Matrix WeightsT { get; set; }

        public Layer PreviousLayer { get; protected set; }
        public Layer NextLayer { get; protected set; }

        public INetworkFunctions NetworkFunctions;

        protected InputNormalization _NormalizeInput;
        protected OutputActivation _OutputActivation;

        Func<float, float> ActivationFunction;
        Func<float, float> ActivationFunctionDerivative;


        public virtual void Forward(Vector input, Vector ValuesOut, Vector ActivationOut) { NetworkFunctions.CalculateValue(Weights, Bias, input, ActivationFunction, ValuesOut, ActivationOut); }

        // input only used in output layer
        public virtual void Backward(Vector ValuesIn, Vector DeltaIn, Vector DeltaOut) { NetworkFunctions.Backward(NextLayer.WeightsT, DeltaIn, ValuesIn, ActivationFunctionDerivative, DeltaOut); }
        public virtual void BackwardWeights(Vector DeltaIn, Vector ActivationIn, Matrix WeightDelta) { NetworkFunctions.BackwardWeights(DeltaIn, ActivationIn, WeightDelta); }
        public virtual void BackwardBias(Vector DeltaIn, Vector BiasDelta) { NetworkFunctions.BackwardBias(DeltaIn, BiasDelta); }

        public virtual void AdjustWeight(Matrix WeightsDelta, float scale) { NetworkFunctions.AdjustWeights(WeightsDelta, scale, Weights, WeightsT); }
        public virtual void AdjustBias(Vector BiasDelta, float scale) { NetworkFunctions.AdjustBias(BiasDelta, scale, Bias); }
    }

    class InputLayer : Layer {
        public InputLayer(int size, LayerFunctions functions, INetworkFunctions network_functions) : base(0, size, functions, network_functions) { }

        public override void Forward(Vector input, Vector ValuesOut, Vector ActivationOut) {
            ActivationOut.Data = _NormalizeInput(input.Data);
        }
    }

    class OutputLayer : Layer {
        public OutputLayer(int index, int size, Layer previousLayer, LayerFunctions functions, INetworkFunctions network_functions) : base(index, size, previousLayer, functions, network_functions) { }

        public override void Forward(Vector input, Vector ValuesOut, Vector ActivationOut) {
            base.Forward(input, ValuesOut, ActivationOut);
            ActivationOut.Data = _OutputActivation(ValuesOut.Data);
        }

        public override void Backward(Vector ActivationIn, Vector CorrectValues, Vector DeltaOut) { NetworkFunctions.BackwardOutput(ActivationIn, CorrectValues, DeltaOut); }
    }

    public class NeuralNetwork {
        public NeuralNetwork(int[] layers, LayerFunctions functions) {
            NetworkFunctions = functions.NetworkFunctions;
            
            LayerAmount = layers.Length;
            Layers = new Layer[LayerAmount];
            LayerLength = new int[LayerAmount];

            for (int i = 0; i < LayerAmount; i++) {
                LayerLength[i] = layers[i];
                if (i == 0) {
                    Layers[0] = new InputLayer(layers[i], functions, NetworkFunctions);
                } else if (i == LayerAmount - 1) {
                    Layers[LayerAmount - 1] = new OutputLayer(i, layers[i], Layers[i - 1], functions, NetworkFunctions);
                } else {
                    Layers[i] = new Layer(i, layers[i], Layers[i - 1], functions, NetworkFunctions);
                }
            }
        }

        public INetworkFunctions NetworkFunctions;
        public int LayerAmount { get; }
        public int[] LayerLength { get; }
        public Layer[] Layers { get; }

        public Vector Calculate(Vector input, VirtualNetwork network) {
            network.Forward(Layers[0], input);
            for (int i = 1; i < LayerAmount; i++) {
                network.Forward(Layers[i], network.Activations[i - 1]);
            }

            return network.Activations[LayerAmount - 1];
        }
    }

    public class VirtualNetwork {
        public float loss;
        public INetworkFunctions NetworkFunctions;

        public Vector[] Values;
        public Vector[] Activations;
        public Vector[] Delta;
        public Matrix[] WeightDelta;
        public Vector[] BiasDelta;

        public VirtualNetwork(NeuralNetwork Network) {
            NetworkFunctions = Network.NetworkFunctions;

            Values = new Vector[Network.LayerAmount];
            Activations = new Vector[Network.LayerAmount];

            Delta = new Vector[Network.LayerAmount - 1];
            WeightDelta = new Matrix[Network.LayerAmount - 1];
            BiasDelta = new Vector[Network.LayerAmount - 1];

            for (int i = 0; i < Network.LayerAmount; i++) {
                Values[i] = new Vector(Network.LayerLength[i]);
                Activations[i] = new Vector(Network.LayerLength[i]);

                if (i == Network.LayerAmount - 1) continue;

                Delta[i] = new Vector(Network.LayerLength[i + 1]);
                WeightDelta[i] = new Matrix(Network.LayerLength[i + 1], Network.LayerLength[i]);
                BiasDelta[i] = new Vector(Network.LayerLength[i + 1]);
            }
        }

        public void Forward(Layer layer, Vector input) {
            layer.Forward(input, Values[layer.index], Activations[layer.index]);
        }
        public void Backward(Layer layer, Vector CorrectValues) {
            int l = layer.index - 1; // Delta arrays are i-th element for (i+1)th layer
            if (layer.NextLayer == null) {
                layer.Backward(Activations[layer.index], CorrectValues, Delta[l]);
            } else {
                layer.Backward(Values[layer.index], Delta[l + 1], Delta[l]);
            }

            layer.BackwardWeights(Delta[l], Activations[layer.index - 1], WeightDelta[l]);
            layer.BackwardBias(Delta[l], BiasDelta[l]);
        }

        public void Adjust(Layer layer, float scale) {
            int l = layer.index - 1;

            layer.AdjustWeight(WeightDelta[l], scale);
            layer.AdjustBias(BiasDelta[l], scale);
        }
    }
}