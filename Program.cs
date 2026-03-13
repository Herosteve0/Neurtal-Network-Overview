using NeuralNetworkSystem;

ProgramManager.InitializeProgram();

DateTime time = DateTime.Now;
await ProgramManager.Trainer.MNIST_RandomTraining();
Console.WriteLine(DateTime.Now - time);
await ProgramManager.Trainer.MINST_Test();

public enum ParallelType {
    None, CPU, GPU
}

public static class ProgramManager {

    public static Random random;
    public static bool disableMessages = false;
    public static ParallelType ProgramType = ParallelType.CPU;
    public static bool useAVX512 = true;

    public static NeuralNetwork Network;
    public static NeuralNetworkTrainer Trainer;

    public static void InitializeProgram() {
        random = new Random(5000);

        int[] layers = { 784, 256, 256, 128, 10 };
        LayerFunctions funcs = FunctionManager.GetFunctions(
            InputNormalizationFunctionsType.NormalizeMeadian,
            OutputFunctionsType.SoftMax,
            ActivationFunctionsTypes.ReLU
            );

        Network = new NeuralNetwork(layers, funcs);
        Trainer = new NeuralNetworkTrainer(Network, LossFunctions.SoftMax, 0.085f, true, 500, 250, 1);
    }
}