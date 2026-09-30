using System.Runtime.InteropServices.Swift;
using Neurtal_Network_Overview.Data;

namespace Neurtal_Network_Overview.System {
    public record NetworkInstance(NeuralNetwork Network, int Accuracy) {
        public void Dispose() {
            Network.Dispose();
        }
    }
    
    public class NeuralNetwork : IDisposable{
        
        private BaseLayer[] Layers { get; }
        
        public IBackend Backend { get; }
        
        public NeuralNetwork(int[] layers, IBackend backend, Random random) {
            Backend = backend;
            LayerAmount = layers.Length;
            LayerLength = layers.ToArray();
            
            Layers = new BaseLayer[LayerAmount];
            for (int i = 0; i < LayerAmount; i++) {
                int length = layers[i];
                Layers[i] = i == LayerAmount - 1
                    ? new OutputLayer(length, backend, Layers[i - 1], random)
                    : i == 0
                        ? new InputLayer(length, backend)
                        : new Layer(length, backend, Layers[i - 1], random);
            }
        }
        public NeuralNetwork(int[] layers, IBackend backend, float[][] weights, float[][] bias) {
            Backend = backend;
            LayerAmount = layers.Length;
            LayerLength = layers.ToArray();
            
            Layers = new BaseLayer[LayerAmount];
            for (int i = 0; i < LayerAmount; i++) {
                int length = layers[i];
                Layers[i] = i == LayerAmount - 1
                    ? new OutputLayer(length, backend, Layers[i - 1], weights[i - 1], bias[i - 1])
                    : i == 0
                        ? new InputLayer(length, backend)
                        : new Layer(length, backend, Layers[i - 1], weights[i - 1], bias[i - 1]);
            }
        }
        
        public BaseLayer this[int index] => Layers[index];


        public int LayerAmount { get; }
        public int[] LayerLength { get; }

        public DataGuess<float> QuickCalculate(DataSample<float> input) {
            VirtualNetwork network = new VirtualNetwork(this);
            DataGuess<float> result = Calculate(input, network);
            network.Dispose();
            return result;
        }
        public DataGuess<float> Calculate(DataSample<float> input, VirtualNetwork network) {
            network.Load(input);
            network.IterateForward(this);
            return network.Guess();
        }

        public void UpdateTranpose() {
            for (int i = 1; i < LayerAmount; i++) {
                ((Layer)this[i]).UpdateTranpose();
            }
        }
        
        public (float[][] weights, float[][] bias) GetData() {
            float[][] weights = new float[LayerAmount - 1][];
            float[][] bias = new float[LayerAmount - 1][];
            
            for (int i = 1; i < LayerAmount; i++) {
                weights[i - 1] = new float[LayerLength[i] * LayerLength[i - 1]];
               Backend.Read(((Layer)this[i]).Weights).CopyTo(weights[i - 1]);
                
                bias[i - 1] = new float[LayerLength[i]];
                Backend.Read(((Layer)this[i]).Bias).CopyTo(bias[i - 1]);
            }
            
            return (weights, bias);
        }
        public NeuralNetwork Copy() {
            var variables = GetData();
            return new NeuralNetwork(LayerLength, Backend, variables.weights, variables.bias);
        }

        public void Dispose() {
            foreach (BaseLayer layer in Layers) {
                layer.Dispose();
            }
        }
    }

    public class VirtualNetwork : IDisposable {
        private DataSample<float> _rawInput;
        
        private readonly ITensor _input;
        private readonly ITensor _expectedOutput;
        private readonly ITensor[] _values;
        private readonly ITensor[] _activations;
        
        private readonly ITensor[] _delta;
        private readonly ITensor[] _weightDelta;
        private readonly ITensor[] _biasDelta;

        public readonly IBackend Backend;
        public readonly int Length;
        public float Loss;
        
        public VirtualNetwork(NeuralNetwork network) : this(network.LayerLength.ToArray(), network.Backend) { }
        public VirtualNetwork(int[] layers, IBackend backend) {
            Length = layers.Length;
            Backend = backend;
            
            _values = new ITensor[Length - 1];
            _activations = new ITensor[Length];
            
            _delta = new ITensor[Length - 1];
            _weightDelta = new ITensor[Length - 1];
            _biasDelta = new ITensor[Length - 1];
            
            
            _input = Backend.CreateTensor(layers[0]);
            _expectedOutput = Backend.CreateTensor(layers[Length - 1]);
            
            for (int i = 0; i < Length; i++) {
                _activations[i] = Backend.CreateTensor(layers[i]);

                if (i == Length - 1) continue;
                
                _values[i] = Backend.CreateTensor(layers[i + 1]);

                _delta[i] = Backend.CreateTensor(layers[i + 1]);
                _weightDelta[i] = Backend.CreateTensor(layers[i + 1], layers[i]);
                _biasDelta[i] = Backend.CreateTensor(layers[i + 1]);
            }
        }
        

        public void Load(DataSample<float> input) {
            _rawInput = input;
            Backend.Set(_input, input.Values);
        }
        public void LoadAnswer(int label) {
            float[] expectedOutput = new float[_expectedOutput.Length];
            expectedOutput[label] = 1;
            Backend.Set(_expectedOutput, expectedOutput);
        }
        public DataGuess<float> Guess() {
            return new DataGuess<float>(
                    _rawInput,
                    _activations[Length - 1]
                ); 
        }

        public void IterateForward(NeuralNetwork network) {
            for (int i = 0; i < network.LayerAmount; i++) {
                Forward(i, network[i]);
            }
        }
        public void IterateBackward(NeuralNetwork network) {
            for (int i = network.LayerAmount - 1; i > 0; i--) {
                Backward(i, network[i]);
            }
        }

        public void Forward(int index, BaseLayer layer) {
            // Layer 0 has no parameters, so index is shifted by -1
            int l = index - 1;

            if (index == 0) {
                layer.Forward(_input, null, _activations[index]);
            } else {
                layer.Forward(_activations[index - 1], _values[l], _activations[index]);
            }
        }

        public void Backward(int index, BaseLayer layer) {
            // Layer 0 has no parameters, so index is shifted by -1
            int l = index - 1;
            
            if (index == Length - 1) {
                layer.Backward(_activations[index], _expectedOutput, _delta[l]);
            } else {
                layer.Backward(_values[l], _delta[l + 1] , _delta[l]);
            }
            
            layer.BackPropagation(_delta[l], _activations[index - 1], _weightDelta[l], _biasDelta[l]);
        }

        public void Adjust(int index, BaseLayer layer, float scale) {
            // Layer 0 has no parameters, so index is shifted by -1
            int l = index - 1;
            
            layer.Adjust(_weightDelta[l], _biasDelta[l], scale);
        }

        public void Dispose() {
            _input.Dispose();
            _expectedOutput.Dispose();
            foreach (ITensor tensor in _values) {
                tensor.Dispose();
            }
            foreach (ITensor tensor in _activations) {
                tensor.Dispose();
            }
            foreach (ITensor tensor in _delta) {
                tensor.Dispose();
            }
            foreach (ITensor tensor in _weightDelta) {
                tensor.Dispose();
            }
            foreach (ITensor tensor in _biasDelta) {
                tensor.Dispose();
            }
        }
    }
}