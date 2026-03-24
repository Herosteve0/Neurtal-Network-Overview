
namespace NeuralNetworkSystem {
    public class NeuralNetworkTester {
        public NeuralNetworkTester(NeuralNetwork network, int batchSize) {
            Network = network;
            this.batchSize = batchSize;

            timeDelta = 0;
            isTesting = false;
        }

        NeuralNetwork Network { get; }
        
        public int batchSize { get; }

        public bool isTesting { get; private set; }
        public int TestingProgress { get; private set; }
        public int TestingAccuracy { get; private set; }
        public int TestingAmount { get; private set; }

        public double timeDelta { get; private set; }
        DateTime timeTemp;


        public int TestingCalculations(Data TestingData, VirtualNetwork network) {
            Vector result = Network.Calculate(TestingData.data, network);
            return result.MaxIndex();
        }

        public void BatchTesting(DataBatch DataBatch, List<Data> wrongs, List<int> wrong_labels) {
            switch (ProgramManager.ProgramType) {
                case ParallelType.None:
                    TestingAccuracy += NormalCalculateGradient(DataBatch, wrongs, wrong_labels);
                    break;
                case ParallelType.CPU:
                    TestingAccuracy += CPUParallelCalculateGradient(DataBatch, wrongs, wrong_labels);
                    break;
                case ParallelType.GPU:
                    TestingAccuracy += GPUParallelCalculateGradient(DataBatch, wrongs, wrong_labels);
                    break;
            }
        }

        public int NormalCalculateGradient(DataBatch DataBatch, List<Data> wrongs, List<int> wrong_labels) {
            int correct = 0;
            VirtualNetwork network = new VirtualNetwork(Network);

            foreach (Data TestingData in DataBatch.Data) {
                int guess = TestingCalculations(TestingData, network);
                if (guess == TestingData.label) {
                    correct++;
                } else {
                    wrongs.Add(TestingData);
                    wrong_labels.Add(guess);
                }
            }

            return correct;
        }
        public int CPUParallelCalculateGradient(DataBatch DataBatch, List<Data> wrongs, List<int> wrong_labels) {
            int correct = 0;
            object lockObj = new object();

            Parallel.For(
                0, DataBatch.Size,

                () => new LocalThread(Network),
                (i, state, local) => {
                    Data data = DataBatch.Data[i];
                    int guess = TestingCalculations(data, local.network);

                    if (guess == data.label) {
                        local.correct++;
                    } else {
                        local.wrongs.Add(data);
                        local.wrong_labels.Add(guess);
                    }

                    return local;
                },

                local => {
                    lock (lockObj) {
                        correct += local.correct;
                        wrongs.AddRange(local.wrongs);
                        wrong_labels.AddRange(local.wrong_labels);
                    }
                }

            );
            return correct;
        }
        public int GPUParallelCalculateGradient(DataBatch DataBatch, List<Data> wrongs, List<int> wrong_labels) {
            return 0;
        }


        class LocalThread {
            public VirtualNetwork network;
            public int correct;
            public List<Data> wrongs;
            public List<int> wrong_labels;

            public LocalThread(NeuralNetwork network) {
                this.network = new VirtualNetwork(network);
                correct = 0;
                wrongs = new List<Data>();
                wrong_labels = new List<int>();
            }
        }



        public void ForceStopTesting() {
            if (!isTesting) return;
            isTesting = false;
            PrintMessage(ConsoleMessages.ForceStop);
        }

        enum ConsoleMessages {
            Start,
            Progress,
            Finish,
            ForceStop,

            ErrorTraining
        }
        void PrintMessage(ConsoleMessages type) {
            if (type == ConsoleMessages.Start) Console.WriteLine($"Started testing on {TestingAmount} test samples.");
            else if (type == ConsoleMessages.Progress) {
                if (ProgramManager.disableMessages) return;
                Console.WriteLine($"Testing is {100 * (double)TestingProgress / TestingAmount:F2}% Complete [{TestingProgress}/{TestingAmount}]");
            } else if (type == ConsoleMessages.Finish) Console.WriteLine($"Testing completed with {(double)TestingAccuracy / TestingAmount * 100}% accuracy. [{TestingAccuracy}/{TestingAmount}]");
            else if (type == ConsoleMessages.ForceStop) Console.WriteLine($"Force stopped testing.");
            else if (type == ConsoleMessages.ErrorTraining) Console.WriteLine($"Wait until the training is complete before starting the testing process.");
        }

        int delay_ticks = 2500;




        async Task Test(DataBatch batch, List<Data> wrongs, List<int> wrong_labels, bool breath) {
            BatchTesting(batch, wrongs, wrong_labels);
            TestingProgress += batchSize;
            if (breath) {
                PrintMessage(ConsoleMessages.Progress);
                //DetailVisualization.Refresh();
                timeDelta = (DateTime.Now - timeTemp).TotalSeconds;
                await Task.Delay(1);
            }
            timeTemp = DateTime.Now;
        }

        public async Task MINST_Test(bool show_wrongs) {
            if (ProgramManager.Trainer.isTraining) {
                PrintMessage(ConsoleMessages.ErrorTraining);
                return;
            }
            DataBatch batch = new DataBatch(MNISTDatabase.LoadAllTestingData());

            List<Data> wrongs = new List<Data>();
            List<int> wrong_labels = new List<int>();

            TestingAccuracy = 0;
            TestingProgress = 0;
            TestingAmount = batch.Size;
            isTesting = true;

            PrintMessage(ConsoleMessages.Start);
            int counter = 0;
            timeTemp = DateTime.Now;
            for (int i = 0; i < TestingAmount; i += batchSize) {
                if (!isTesting) return;

                counter += batchSize;
                bool breath = counter >= delay_ticks;
                await Test(batch.GetSmallBatch(i, batchSize), wrongs, wrong_labels, breath);
                if (breath) counter = 0;
            }
            isTesting = false;

            PrintMessage(ConsoleMessages.Finish);
            //DetailVisualization.Refresh();
            if (show_wrongs) ProgramManager.DrawImages(wrongs.ToArray(), wrong_labels.ToArray());
        }
    }
}