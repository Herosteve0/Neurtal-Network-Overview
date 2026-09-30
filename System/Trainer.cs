using System.Numerics;
using System.Runtime.InteropServices;
using Neurtal_Network_Overview.Data;

namespace Neurtal_Network_Overview.System {
    public class TrainerLearningInfo {
        public TrainerLearningInfo(float learningRate) {
            BaseLearningRate = learningRate;
            ResetLearningRate();

            _hasDecay = false;
        }
        public TrainerLearningInfo(float learningRate, float decayMultiplier, int decayPatience) : this(learningRate) {
            _hasDecay = true;
            
            _decayMultiplier = decayMultiplier;
            _decayPatience = decayPatience;
        }

        private float BaseLearningRate { get; }
        public float LearningRate { get; private set; }

        private readonly bool _hasDecay;
        private readonly float _decayMultiplier;
        private readonly float _decayPatience;

        private int _patienceCounter = 0;
        private float _minLoss = float.PositiveInfinity;

        public static implicit operator float(TrainerLearningInfo info) {
            if (info == null) return 0;
            return info.LearningRate;
        }

        public bool HasDecayed => LearningRate == 0f;
        
        public void ResetLearningRate() {
            LearningRate = BaseLearningRate;
        }
        
        public void CheckLoss(float loss) {
            if (!_hasDecay) return;
            
            if (loss < _minLoss) {
                _minLoss = loss;
                _patienceCounter = 0;
                return;
            }
            
            _patienceCounter++;
            if (_patienceCounter <= _decayPatience) return;
            
            _patienceCounter = 0;
            ApplyDecay();
        }

        public void ApplyDecay() {
            LearningRate *= _decayMultiplier;
        }
    }
    
    public class Trainer : NetworkScanner {
        public Trainer(NeuralNetwork network, int batchSize, int delayTicks, TrainerLearningInfo learningInfo, bool isSilent = false) : base("Trainer", network, batchSize, delayTicks, isSilent) {
            _learningRate = learningInfo;
        }

        private readonly TrainerLearningInfo _learningRate;
        
        private float _lossMemory;
        private int _lossCount;

        public float LearningRate => _learningRate;

        public async Task Train(DataBatch<float> dataBatch, int repeats, Random random) {
            _lossMemory = 0f;
            _lossCount = 0;
            await Scan(dataBatch, repeats, random);
        }

        protected override void BreathFunction() {
            if (!IsSilent) {
                Console.WriteLine(
                    $"Training Loss: {_lossMemory / _lossCount} | Learning Rate: {_learningRate.LearningRate}");
            }
            _lossMemory = 0f;
            _lossCount = 0;
        }

        protected override void Process(DataBatch<float> trainingData) {
            float scale = _learningRate / trainingData.Size;

            float averageLoss = Backend is CpuBackend && ProgramExecutor.MultiThread
                ? ExecuteMultiThread(trainingData, scale)
                : Execute(trainingData, scale);
            averageLoss /= trainingData.Size;
            
            _learningRate.CheckLoss(averageLoss);
            if (_learningRate.HasDecayed) {
                ForceStop("LearningRate Decayed.");
            }

            _lossMemory += averageLoss;
            _lossCount++;
        }

        private float Execute(DataBatch<float> dataBatch, float scale) {
            VirtualNetwork network = new VirtualNetwork(Network);

            float totalLoss = dataBatch.Data.Sum(
                trainingData => TrainVirtualNetwork(network, trainingData)
                );

            for (int i = 1; i < Network.LayerAmount; i++) {
                network.Adjust(i, Network[i], scale);
            }

            network.Dispose();
            return totalLoss;
        }

        private float ExecuteMultiThread(DataBatch<float> dataBatch, float scale) {
            float totalLoss = 0f;
            
            object lockObj = new object();
            Parallel.For(0L, dataBatch.Size, () => new VirtualNetwork(Network), (i, state, local) => {
                DataSample<float> data = dataBatch[(int)i];

                local.Loss += TrainVirtualNetwork(local, data);

                return local;
            }, local => {
                lock (lockObj) {
                    totalLoss += local.Loss;
                    for (int i = 1; i < Network.LayerAmount; i++) {
                        local.Adjust(i, Network[i], scale);
                    }
                    local.Dispose();
                }
            });
            
            return totalLoss;
        }

        private float TrainVirtualNetwork(VirtualNetwork network, DataSample<float> trainingData) {
            DataGuess<float> output = Network.Calculate(trainingData, network);
            // all layers of the network have the values we want (input, value, activation)

            network.LoadAnswer(trainingData.Label);
            network.IterateBackward(Network);

            return Backend.Loss(output);
        }
    }
}