using NeuralNetworkSystem;
using System.Drawing;


ProgramManager.Network = NeuralNetworkStoring.Load("save.nn");
int old_accuracy;
if (ProgramManager.Network != null) {
    await ProgramManager.Tester.MINST_Test(false);
    old_accuracy = ProgramManager.Tester.TestingAccuracy;
} else {
    old_accuracy = 0;
}

ProgramManager.InitializeProgram();

DateTime time = DateTime.Now;
await ProgramManager.Trainer.MNIST_RandomTraining();
Console.WriteLine(DateTime.Now - time);

await ProgramManager.Tester.MINST_Test(true);

int delta_accuracy = ProgramManager.Tester.TestingAccuracy - old_accuracy;
Console.WriteLine($"Accuracy changed by {delta_accuracy}");
if (delta_accuracy > 0) {
    Console.WriteLine($"Neural Network saved.");
    NeuralNetworkStoring.Save(ProgramManager.Network, "save.nn");
}


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
    public static NeuralNetworkTester Tester;

    public static void InitializeProgram() {
        int[] layers = { 784, 512, 512, 512, 512, 10 };
        CreateNetwork(layers);
    }

    public static NeuralNetwork CreateNetwork(int[] layers) {
        random = new Random(5000);

        LayerFunctions funcs = FunctionManager.GetFunctions(
            InputNormalizationFunctionsType.NormalizeMeadian,
            OutputFunctionsType.SoftMax,
            ActivationFunctionsTypes.ReLU
            );

        Network = new NeuralNetwork(layers, funcs);
        Trainer = new NeuralNetworkTrainer(Network, LossFunctions.SoftMax, 0.085f, true, 500, 250, 450);
        Tester = new NeuralNetworkTester(Network, 2500);
        return Network;
    }

    public static void DrawImages(Data[] wrongs, int[] guess){
        string filepath = @".\WrongGuesses";

        if (Path.Exists(filepath)) Directory.Delete(filepath, true);
        Directory.CreateDirectory(filepath);

        for (int i = 0; i < wrongs.Length; i++) {
            CreateImage(wrongs[i], $"{filepath}/{i}. {guess[i]} instead of {wrongs[i].label}.png");
        }
    }

    static void CreateImage(Data data, string filepath) {
        int width = 28;
        int height = 28;

        var image = new Bitmap(width, height);

        int i = 0;
        for (int y = 0; y < height; y++) { 
            for (int x = 0; x < width; x++) {
                byte value = (byte)(Math.Clamp((int)(data.data[i++] * 255f), 0, 255));
                image.SetPixel(x, y, Color.FromArgb(value, value, value));
            }
        }

        image.Save(filepath);
    }
}