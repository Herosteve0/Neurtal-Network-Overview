
using System.Runtime.Intrinsics;

namespace NeuralNetworkSystem {
    public struct Data {
        public Data(Vector data, int label) {
            this.data = data;
            this.label = label;
        }

        public Vector data { get; }
        public int label { get; }
    }

    public class DataBatch {
        public DataBatch(Data[] data) {
            Data = data;
            Size = data.Length;
        }

        public Data[] Data { get; }
        public int Size { get; }

        public DataBatch GetSmallBatch(int index, int size) {
            Data[] newdata = new Data[size];
            Array.Copy(Data, index, newdata, 0, size);
            return new DataBatch(newdata);
        }

        public void Shuffle() {
            for (int i = Size - 1; i > 0; i--) {
                int r = ProgramManager.random.Next(0, i + 1);
                (Data[i], Data[r]) = (Data[r], Data[i]);
            }
        }
    }

    public class NeuralNetworkTrainer {
        public NeuralNetworkTrainer(NeuralNetwork network, LossCalculation loss_function, float learning_rate = 0.075f, bool lr_decay = true, int lr_decay_patience = 10, int batchSize = 100, int cycles = 5) {
            Network = network;
            LossFunction = loss_function;
            this.batchSize = batchSize;
            this.cycles = cycles;
            base_learning_rate = learning_rate;
            LearningRate = base_learning_rate;
            learning_rate_decay = lr_decay;
            learning_rate_decay_patience = lr_decay_patience;

            timeDelta = 0;
            isTraining = false;
            PausedTraining = false;
            StepTraining = false;
        }

        NeuralNetwork Network { get; }
        LossCalculation LossFunction { get; }
        
        public float base_learning_rate { get; }
        public float LearningRate { get; private set; }
        public bool learning_rate_decay {  get; }
        public int learning_rate_decay_patience {  get; }
        float min_loss = -1;
        int loss_counter = 0;

        public int batchSize { get; }
        public int cycles { get; }
        public int Seed { get; }

        public bool isTraining { get; private set; }
        public bool PausedTraining { get; private set; }
        public bool StepTraining { get; private set; }
        public int TrainingProgress { get; private set; }
        public int TrainingAmount { get; private set; }

        public double timeDelta { get; private set; }
        DateTime timeTemp;

        CancellationTokenSource canceltoken;
        
        public float TrainingCalculations(Data TrainingData, VirtualNetwork network) {
            Vector output = Network.Calculate(TrainingData.data, network); // all layers of the network have the values we want (inupt, value, activation)

            int length = Network.LayerAmount - 1;

            Vector Y = Vector.SingleValue(Network.LayerLength[length], TrainingData.label);
            float Loss = LossFunction(output.Data, TrainingData.label);

            for (int i = length; i > 0; i--) {
                network.Backward(Network.Layers[i], Y);
            }

            return Loss;
        }

        public void SingleExampleTraining(Data TrainingData) {
            BatchTraining(new DataBatch(new Data[] { TrainingData }));
        }
        public void BatchTraining(DataBatch DataBatch) {
            float avg_loss = 0f;
            float scale = LearningRate / DataBatch.Size;
            switch (ProgramManager.ProgramType) {
                case ParallelType.None:
                    avg_loss = NormalCalculateGradient(DataBatch, scale);
                    break;
                case ParallelType.CPU: 
                    avg_loss = CPUParallelCalculateGradient(DataBatch, scale);
                    break;
            }
            avg_loss /= DataBatch.Size;


            //DetailVisualization.StoreLoss(avg_loss);

            if (learning_rate_decay) {
                if (min_loss == -1f) min_loss = avg_loss;
                if (avg_loss >= min_loss) {
                    loss_counter++;
                    if (loss_counter > learning_rate_decay_patience) {
                        loss_counter = 0;
                        LearningRate *= 0.5f;
                    }
                } else {
                    min_loss = avg_loss;
                    loss_counter = 0;
                }
            }
        }
        public float NormalCalculateGradient(DataBatch DataBatch, float scale) {
            float total_loss = 0f;
            VirtualNetwork network = new VirtualNetwork(Network);
            foreach (Data TrainingData in DataBatch.Data) {
                total_loss += TrainingCalculations(TrainingData, network);
            }

            for (int i = 1; i < Network.LayerAmount; i++) {
                network.Adjust(Network.Layers[i], scale);
            }

            return total_loss;
        }
        public float CPUParallelCalculateGradient(DataBatch DataBatch, float scale) {
            float total_loss = 0f;
            object lockObj = new object();
            Parallel.For(
                0, DataBatch.Size,

                () => new VirtualNetwork(Network),
                (i, state, local) => {
                    Data data = DataBatch.Data[i];

                    local.loss += TrainingCalculations(data, local);

                    return local;
                },

                local => {
                    lock (lockObj) {
                        total_loss += local.loss;
                        for (int i = 1; i < Network.LayerAmount; i++) {
                            local.Adjust(Network.Layers[i], scale);
                        }
                    }
                }

            );
            return total_loss;
        }


        public void ForceStopTraining() {
            if (!isTraining) return;
            isTraining = false;
            PrintMessage(ConsoleMessages.ForceStop);
        }
        public void TogglePause() {
            PausedTraining = !PausedTraining;
            if (PausedTraining) canceltoken = new CancellationTokenSource();
            else canceltoken.Cancel();
            PrintMessage(ConsoleMessages.Pause);
        }
        public void ToggleStep() {
            StepTraining = !StepTraining;
            if (StepTraining) canceltoken = new CancellationTokenSource();
            else canceltoken.Cancel();
            PrintMessage(ConsoleMessages.Step);
        }
        public void DoStep() {
            if (!StepTraining) return;
            PrintMessage(ConsoleMessages.DoStep);
            canceltoken.Cancel();
        }

        enum ConsoleMessages {
            Start,
            Progress,
            Finish,
            ForceStop,
            Pause,
            Step,
            DoStep,

            FinishEarly,

            ErrorTesting
        }
        void PrintMessage(ConsoleMessages type) {
            if (type == ConsoleMessages.Start) Console.WriteLine($"Started training on {TrainingAmount} examples.");
            else if (type == ConsoleMessages.Progress) {
                if (ProgramManager.disableMessages) return;
                Console.WriteLine($"Training is {100 * (double)TrainingProgress / TrainingAmount:F2}% Complete [{TrainingProgress}/{TrainingAmount}]");
            } else if (type == ConsoleMessages.Finish) Console.WriteLine($"Training Complete.");
            else if (type == ConsoleMessages.ForceStop) Console.WriteLine($"Force stopped training.");
            else if (type == ConsoleMessages.Pause) Console.WriteLine((PausedTraining ? "Paused" : "Unpaused") + " training.");
            else if (type == ConsoleMessages.Step) Console.WriteLine((StepTraining ? "Enabled" : "Disabled") + " step training.");
            else if (type == ConsoleMessages.DoStep) Console.WriteLine("Did one training step.");

            else if (type == ConsoleMessages.FinishEarly) Console.WriteLine($"Training finished early. (Learning Rate reached 0)");

            else if (type == ConsoleMessages.ErrorTesting) Console.WriteLine($"Wait until the testing is complete before starting the training process.");
        }

        int delay_ticks = 2500;

        async Task Train(DataBatch batch, bool breath) {
            if (PausedTraining) await WaitFor(-1, canceltoken.Token);

            BatchTraining(batch);
            TrainingProgress += batchSize;
            if (breath) {
                PrintMessage(ConsoleMessages.Progress);
                //DetailVisualization.Refresh();
                timeDelta = (DateTime.Now - timeTemp).TotalSeconds;
                await Task.Delay(1);
            }
            timeTemp = DateTime.Now;
        }

        async Task WaitFor(int ms, CancellationToken token) {
            try {
                await Task.Delay(ms, token);
            } catch (TaskCanceledException) { }
        }

        public async Task MNIST_RandomTraining() { await MNIST_RandomTraining(cycles); }
        public async Task MNIST_RandomTraining(int loops) {
            if (ProgramManager.Tester.isTesting) {
                PrintMessage(ConsoleMessages.ErrorTesting);
                return;
            }
            DataBatch training_data = new DataBatch(MNISTDatabase.LoadAllTrainingData());

            TrainingProgress = 0;
            TrainingAmount = training_data.Size * loops;
            isTraining = true;

            PrintMessage(ConsoleMessages.Start);
            int counter = 0;
            timeTemp = DateTime.Now;
            for (int cycle = 0; cycle < loops; cycle++) {
                training_data.Shuffle();
                for (int i = 0; i < training_data.Size; i += batchSize) {
                    if (!isTraining) return;

                    counter += batchSize;
                    bool breath = counter >= delay_ticks;
                    await Train(training_data.GetSmallBatch(i, batchSize), breath);
                    if (breath) counter = 0;

                    if (LearningRate <= 0f) {
                        PrintMessage(ConsoleMessages.FinishEarly);
                        cycle = loops;
                        break;
                    }
                }
            }
            isTraining = false;
            
            PrintMessage(ConsoleMessages.Finish);
            //DetailVisualization.Refresh();
        }
    }
}