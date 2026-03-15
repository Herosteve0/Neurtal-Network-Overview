using System.IO;

namespace NeuralNetworkSystem {
    public class NeuralNetworkStoring {
        public static void Save(NeuralNetwork network, string filepath) {
            using var writer = new BinaryWriter(File.Open(filepath, FileMode.Create));

            writer.Write("NeuralNetworkEduProject");
            //writer.Write(ProgramHandler.version);
            
            writer.Write(network.LayerAmount);
            for (int i = 0; i < network.LayerAmount; i++) {
                writer.Write(network.LayerLength[i]);
            }

            for (int i = 1; i < network.LayerAmount; i++) {
                
                Layer layer = network.Layers[i];

                foreach (float v in layer.Weights.Data) { writer.Write(v); }
                foreach (float v in layer.Bias.Data) { writer.Write(v); }
            }
        }

        public static NeuralNetwork Load(string filepath) {
            if (!Path.Exists(filepath)) {
                return null;
            }

            using var reader = new BinaryReader(File.OpenRead(filepath));

            string header = reader.ReadString();
            //int version = reader.ReadInt32();

            int layer_amount = reader.ReadInt32();
            int[] layers = new int[layer_amount];
            for (int i = 0; i < layer_amount; i++)
            {
                layers[i] = reader.ReadInt32();
            }

            NeuralNetwork network = ProgramManager.CreateNetwork(layers);

            for (int i = 1; i < layer_amount; i++)
            {
                Layer layer = network.Layers[i];

                for (int row = 0; row < layers[i]; row++)
                {
                    for (int col = 0; col < layers[i - 1]; col++)
                    {
                        layer.Weights[row, col] = reader.ReadSingle();
                    }
                }

                for (int row = 0; row < layers[i]; row++)
                {
                    layer.Bias[row] = reader.ReadSingle();
                }
            }

            return network;
        }
    }
}