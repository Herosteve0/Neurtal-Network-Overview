using System.Drawing;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using Neurtal_Network_Overview.Data;
using Neurtal_Network_Overview.System;

Context context = Context.Create(builder => builder
    .Cuda()
    // .OpenCL()
    .EnableAlgorithms()
    );

Console.WriteLine("Available devices:");
foreach (Device device in context) {
    Console.WriteLine($"- {device.Name}");
}
Accelerator accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);


DataBatch<float> trainingData = new DataBatch<float>(MnistDatabase.LoadAllTrainingData());
DataBatch<float> testingData = new DataBatch<float>(MnistDatabase.LoadAllTestingData());

/*
 * Gets the score of the previous network.
 */
NetworkFile storer = new NetworkFile("save.nn");
int oldAccuracy = storer.LoadAccuracy();


GpuMultiTrainer multiTrainer = new GpuMultiTrainer(4, accelerator, [784, 8128, 8128, 10], trainingData, testingData);
NetworkInstance instance = await multiTrainer.TrainProcess(5000, 5, 10);


int deltaAccuracy = instance.Accuracy - oldAccuracy;
Console.WriteLine($"Accuracy changed by {deltaAccuracy}");
if (deltaAccuracy > 0) {
    Console.WriteLine($"Neural Network saved.");
    storer.Save(instance);
}

instance.Dispose();

public abstract class ProgramExecutor {
    public static bool MultiThread;
    
    public ProgramExecutor(bool multiThread) {
        MultiThread = multiThread;
    }
}