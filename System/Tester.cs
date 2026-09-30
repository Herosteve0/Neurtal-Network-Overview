
using System.Drawing;
using Neurtal_Network_Overview.Data;

namespace Neurtal_Network_Overview.System {
    public class Tester : NetworkScanner {
        private const int DelayTicks = 5000;
        
        public Tester(NeuralNetwork network, int batchSize, bool isSilent = false) : base("Tester", network, batchSize, DelayTicks, isSilent) { }
        
        private List<DataGuess<float>> _mistakes;

        public async Task<int> Test(DataBatch<float> dataBatch, bool showMistakes = false) {
            _mistakes = [];
            
            await Scan(dataBatch);
            
            int testingAccuracy = dataBatch.Size - _mistakes.Count;
            if (!IsSilent) {
                Console.WriteLine(
                    $"Testing completed with {(double)testingAccuracy / dataBatch.Size * 100}% accuracy. [{testingAccuracy}/{dataBatch.Size}]");
            }                
            if (showMistakes) {
                DrawImages(_mistakes.ToArray());
            }

            return testingAccuracy;
        }

        protected override void Process(DataBatch<float> testingData) {
            if (Backend is CpuBackend && ProgramExecutor.MultiThread) {
                ExecuteMultiThread(testingData);
            }
            else {
                Execute(testingData);
            }
        }

        private void Execute(DataBatch<float> dataBatch) {
            VirtualNetwork network = new VirtualNetwork(Network);

            foreach (DataSample<float> testingData in dataBatch.Data) {
                TestingCalculations(testingData, network, _mistakes);
            }

            network.Dispose();
        }

        private void ExecuteMultiThread(DataBatch<float> dataBatch) {
            object lockObj = new object();

            Parallel.For(
                0L, dataBatch.Size,

                () => new LocalThread(Network),
                (i, state, local) => {
                    DataSample<float> data = dataBatch.Data[i];
                    
                    TestingCalculations(data, local.VirtualNetwork, local.Wrongs);

                    return local;
                },

                local => {
                    lock (lockObj) {
                        _mistakes.AddRange(local.Wrongs);
                    }
                    local.VirtualNetwork.Dispose();
                }

            );
        }
        private class LocalThread(NeuralNetwork network) {
            public readonly VirtualNetwork VirtualNetwork = new(network);
            public readonly List<DataGuess<float>> Wrongs = [];
        }

        private void TestingCalculations(DataSample<float> testingData, VirtualNetwork network, List<DataGuess<float>> wrongs) {
            DataGuess<float> result = Network.Calculate(testingData, network);

            result.EvaluateGuess(Network.Backend);
            if (!result.IsCorrect) {
                wrongs.Add(result);
            }
        }
        
        public static void DrawImages(DataGuess<float>[] wrongs){
            const string filepath = "../../../WrongGuesses";

            if (Path.Exists(filepath)) Directory.Delete(filepath, true);
            Directory.CreateDirectory(filepath);

            for (int i = 0; i < wrongs.Length; i++) {
                DataSample<float> data = wrongs[i].Data;
                int guess = wrongs[i].Guess;
            
                CreateImage(data, $"{filepath}/{i}. {guess} instead of {data.Label}.png");
            }
        }

        private static void CreateImage(DataSample<float> data, string filepath) {
            const int width = 28;
            const int height = 28;

            Bitmap image = new Bitmap(width, height);

            int i = 0;
            for (int y = 0; y < height; y++) { 
                for (int x = 0; x < width; x++) {
                    byte value = (byte)(Math.Clamp((int)(data.Values[i++] * 255f), 0, 255));
                    image.SetPixel(x, y, Color.FromArgb(value, value, value));
                }
            }

            image.Save(filepath);
        }
    }
}