using System;
using System.IO;

using NeuralNetworkSystem;

public class MNISTDatabase {
    public static int ReadBigEndianInt(BinaryReader br) {
        byte[] bytes = br.ReadBytes(4);
        if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }

    public static float[][] LoadImages(string path) {

        using BinaryReader br = new BinaryReader(File.OpenRead(path));

        int magic = ReadBigEndianInt(br); // 2051
        int count = ReadBigEndianInt(br);
        int rows = ReadBigEndianInt(br); // 28
        int columns = ReadBigEndianInt(br); // 28
        
        int imageSize = rows * columns;
        float[][] images = new float[columns][];

        for (int i = 0; i < count; i++) {
            images[i] = new float[imageSize];
            for (int p = 0; p < imageSize; p++) {
                images[i][p] = br.ReadByte() / 255f;
            }
        }

        return images;
    }

    public static int[] LoadLabels(string path) {
        using BinaryReader br = new BinaryReader(File.OpenRead(path));

        int magic = ReadBigEndianInt(br); // 2049
        int count = ReadBigEndianInt(br);

        int[] labels = new int[count];
        for (int i = 0; i < count; i++) {
            labels[i] = br.ReadByte();
        }

        return labels;
    }

    public static Data[] LoadAllTrainingData() {
        MNISTDatabase database = new MNISTDatabase("./MNIST/train-images.idx3-ubyte", "./MNIST/train-labels.idx1-ubyte");
        return database.ReadBatch(database.Size);
    }


    readonly BinaryReader br_images;
    readonly BinaryReader br_labels;

    public int Size;
    public int Index;
    public int Rows;
    public int Cols;

    public MNISTDatabase(string image_path, string label_path) {
        br_images = new BinaryReader(File.OpenRead(image_path));
        br_labels = new BinaryReader(File.OpenRead(label_path));

        int magic_i = ReadBigEndianInt(br_images); // 2051
        int magic_l = ReadBigEndianInt(br_labels); // 2049

        int size_l = ReadBigEndianInt(br_labels);;
        int size_i = ReadBigEndianInt(br_images);

        if (size_l != size_i) {
            throw new Exception("MNIST Database image and label count not matching! Files might be corrupted.");
        }

        Size = size_l;

        Rows = ReadBigEndianInt(br_images);
        Cols = ReadBigEndianInt(br_images);
    }

    public void CloseLoad() {
        br_images.Dispose();
        br_labels.Dispose();
    }

    public Data[] ReadBatch(int batchSize) {
        int loops = Math.Min(batchSize, Size - Index);
        if (loops <= 0) return null;

        Data[] r = new Data[loops];

        for (int i = 0; i < loops; i++) {
            Vector values = new Vector(Rows * Cols);
            for (int j = 0; j < values.Length; j++) {
                values[j] = br_images.ReadByte() / 255f;
            }

            r[i] = new Data(
                values,
                br_labels.ReadByte()
                );
        }

        Index += loops;

        if (Index >= Size) CloseLoad();

        return r;
    }
}