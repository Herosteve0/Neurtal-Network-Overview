using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using Neurtal_Network_Overview.Data;

namespace Neurtal_Network_Overview.System {
    /*
    InputNormalization: A change we can do to our input in order to make it more suitable for our network.

    OutputActivation: The function which we will use in order to properly activate the neurons of the last layer, in order to get the correct values.
                      Note that this function is specifically for the output, since if we were to put it in the hidden layers, it would break the strength
                      of each activation.

    LossCalculation
    */
    
    public enum ActivationFunctionsTypes {
        Sigmoid,
        ReLU
    }
    public abstract class ActivationFunctions {
        /*
         
        f(x) = 1 / (e^(-x) + 1), f: R -> (0, 1)
        f'(x) = e^(-x) / ( (e^(-x) + 1)^2 ) = f(x) * ( 1 - f(x) )
        
        This function confies all numbers into the range (0, 1),
        it returns a number close to 0 the closer the number is to negative infinity and close to 1 the closer the number is to positive infinity

        Sigmoid heavily punishes bad Neurons, while heavily rewarding good Neurons.

        */

        public static float Sigmoid(float value) {
            float e = (float)Math.Exp(-value);
            return 1 / (e + 1);
        }
        public static float SigmoidDerivative(float value) {
            float a = Sigmoid(value);
            return a * (1 - a);
        }

        /*
        
        f(x) = { x, x > 0       , f: R -> [0, + infinity)
               { 0, x <= 0

        This function stops any negative value from proceeding.

        ReLU keeps positive signals while supressing negative ones. In a more general sense, it only uses anything it can take advantage of and ignores anything it doesn't find worthy.
        This function is especially important for Transformers (LLM, GPT)
        
        */

        public static float ReLU(float value) {
            return value > 0 ? value : 0;
        }
        public static float ReLUDerivative(float value) {
            return value > 0 ? 1 : 0;
        }
    }

    public enum InputNormalizationFunctionsType {
        None,
        NormalizeMeadian
    }
    public abstract class InputNormalizationFunctions {


        /*
        
        For this project, we use the MNIST database, which gives us the gray scale values of images and the handler for that transforms that into floats from [0,1], with 0 being the value 0 and 255 being the value 1.
        None is pretty much "What if we used these numbers directly?" which isn't a bad approach, however it might take a bit more time for the network to adjust to using the range [0,1]

        */

        public static float[] None(float[] input) {
            return input;
        }

        /*
        
        NormalizeMedian essentially helps the network by slightly adjust the inputs for it. Instead of having a value [0, 1], we now instead have a value that relates to the pixel value compared to all other pixels.
        This our database is a solved problem, we can chuck the values directly (look at "mean" and "std" variables), however if you wanted to calculate the values yourself, you'd do:

        mean = sum      

        */

        public static float[] NormalizeMedian(float[] input) {
            const float mean = 0.1307f;
            const float std = 0.3081f;

            float[] r = new float[input.Length];
            for (int i = 0; i < input.Length; i++) {
                r[i] = (input[i] - mean) / std;
            }
            return r;
        }
    }

    public enum OutputFunctionsType {
        SoftMax
    }
    public abstract class OutputFunctions {
        /*

        SoftMax is a function that takes many values and returns the probability distribution of these values.
        The sum of this Vector will always be 1.

        */
        public static float[] SoftMax(float[] output) {
            int length = output.Length;
            float[] r = new float[length];

            float max = output[0];
            for (int i = 1; i < length; i++) {
                if (max < output[i]) max = output[i];
            }

            float sum = 0f;
            for (int i = 0; i < length; i++) {
                float e = (float)Math.Exp(output[i] - max);
                r[i] = e;
                sum += e;
            }

            for (int i = 0; i < length; i++) {
                r[i] /= sum;
            }

            return r;
        }
    }

    public enum LossFunctionsType {
        Mean,
        SoftMax
    }
    public abstract class LossFunctions {
        public static float Mean(float[] V, int label) {
            float a = V[label] - 1f;
            return a * a;
        }

        public static float SoftMax(float[] V, int label) {
            return -(float)Math.Log(V[label]);
        }
    }
    
    /*
    public class ScalarFunctions : INetworkFunctions {
        public void CalculateValue(CpuMatrix Weights, CpuVector Bias, CpuVector input, Func<float, float> ActivationFunc, CpuVector ValuesOut, CpuVector ActivationOut) {
            int Rows = Weights.Rows;
            int Columns = Weights.Columns;

            for (int row = 0; row < Rows; row++) {
                ValuesOut[row] = Bias[row];

                for (int col = 0; col < Columns; col++) {
                    ValuesOut[row] += Weights[row, col] * input[col];
                }

                ActivationOut[row] = ActivationFunc(ValuesOut[row]);
            }
        }

        public void Backward(CpuMatrix WeightsT, CpuVector Delta, CpuVector Values, Func<float, float> ActivationFuncDer, CpuVector DeltaOut) {
            int Rows = Delta.Length;
            int Columns = WeightsT.Columns;

            for (int row = 0; row < Rows; row++) {
                DeltaOut[row] = 0f;

                for (int col = 0; col < Columns; col++) {
                    DeltaOut[row] += WeightsT[row, col] * Delta[col];
                }

                DeltaOut[row] *= ActivationFuncDer(Values[row]);
            }
        }
        public void BackwardOutput(CpuVector Activation, CpuVector CorrectValues, CpuVector DeltaOut) {
            int Rows = Activation.Length;

            for (int row = 0; row < Rows; row++) {
                DeltaOut[row] = Activation[row] - CorrectValues[row];
            }
        }
        public void BackwardWeights(CpuVector Delta, CpuVector Activation, CpuMatrix WeightDelta) {
            int Rows = Delta.Length;
            int Columns = Activation.Length;

            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Columns; col++) {
                    WeightDelta[row, col] += Delta[row] * Activation[col];
                }
            }
        }
        public void BackwardBias(CpuVector Delta, CpuVector BiasDelta) {
            int Rows = Delta.Length;

            for (int row = 0; row < Rows; row++) {
                BiasDelta[row] += Delta[row];
            }
        }

        public void AdjustWeights(CpuMatrix WeightsDelta, float scale, CpuMatrix Weights, CpuMatrix WeightsT) {
            int Rows = WeightsDelta.Rows;
            int Columns = WeightsDelta.Columns;

            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Columns; col++) {
                    Weights[row, col] -= WeightsDelta[row, col] * scale;
                    WeightsT[col, row] -= WeightsDelta[row, col] * scale;
                }
            }
        }
        public void AdjustBias(CpuVector BiasDelta, float scale, CpuVector Bias) {
            int Rows = BiasDelta.Length;

            for (int row = 0; row < Rows; row++) {
                Bias[row] -= BiasDelta[row] * scale;
            }
        }

        public void Addiction(ref float[] Out, float[] A) {
            for (int i = 0; i < Out.Length; i++) {
                Out[i] += A[i];
            }
        }
    }
    */
    
    /*
    public class SimdFunctions : INetworkFunctions {
        private readonly int _simdWidth = Vector<float>.Count;

        public void CalculateValue(CpuMatrix Weights, CpuVector Bias, CpuVector input, Func<float, float> ActivationFunc, CpuVector ValuesOut, CpuVector ActivationOut) {
            for (int row = 0; row < Weights.Rows; row++) {
                float sum = Bias[row];
                int offset = row * Weights.Columns;

                int col = 0;
                for (; col <= Weights.Columns - _simdWidth; col += _simdWidth) {
                    var v_weights = new Vector<float>(Weights.Data, offset + col);
                    var v_x = new Vector<float>(input.Data, col);
                    sum += Vector.Dot(v_weights, v_x);
                }

                for (; col < Weights.Columns; col++) {
                    sum += Weights.Data[offset + col] * input.Data[col];
                }

                ValuesOut[row] = sum;
                ActivationOut[row] = ActivationFunc(sum);
            }
        }

        public void Backward(CpuMatrix WeightsT, CpuVector Delta, CpuVector Values, Func<float, float> ActivationFuncDer, CpuVector DeltaOut) {
            int Rows = WeightsT.Rows;
            int Columns = WeightsT.Columns;

            for (int row = 0; row < Rows; row++) {
                float sum = 0f;
                int offset = row * Columns;

                int col = 0;
                for (; col <= Columns - _simdWidth; col += _simdWidth) {
                    var v_weights = new Vector<float>(WeightsT.Data, offset + col);
                    var v_delta = new Vector<float>(Delta.Data, col);
                    sum += Vector.Dot(v_weights, v_delta);
                }

                for (; col < Columns; col++) {
                    sum += WeightsT.Data[offset + col] * Delta.Data[col];
                }

                DeltaOut.Data[row] = sum * ActivationFuncDer(Values.Data[row]);
            }
        }
        public void BackwardOutput(CpuVector Activation, CpuVector CorrectValues, CpuVector DeltaOut) {
            int Rows = Activation.Length;

            int i = 0;
            for (; i <= Rows - _simdWidth; i += _simdWidth) {
                var v_a = new Vector<float>(Activation.Data, i);
                var v_b = new Vector<float>(CorrectValues.Data, i);
                (v_a - v_b).CopyTo(DeltaOut.Data, i);
            }
            for (; i < Rows; i++) {
                DeltaOut.Data[i] = Activation[i] - CorrectValues[i];
            }
        }
        public void BackwardWeights(CpuVector Delta, CpuVector Activation, CpuMatrix WeightDelta) {
            int Rows = Delta.Length;
            int Columns = Activation.Length;

            for (int row = 0; row < Rows; row++) {
                int offset = row * Columns;

                int col = 0;
                for (; col <= Columns - _simdWidth; col += _simdWidth) {
                    var v = new Vector<float>(Activation.Data, col);
                    var v_weight = new Vector<float>(WeightDelta.Data, offset + col);

                    v = Delta.Data[row] * v;
                    (v + v_weight).CopyTo(WeightDelta.Data, offset + col);
                }

                for (; col < Columns; col++) {
                    WeightDelta.Data[offset + col] += Delta.Data[row] * Activation.Data[col];
                }
            }
        }
        public void BackwardBias(CpuVector Delta, CpuVector BiasDelta) {
            int Columns = Delta.Length;

            int col = 0;
            for (; col <= Columns - _simdWidth; col += _simdWidth) {
                var v = new Vector<float>(Delta.Data, col);
                var v_bias = new Vector<float>(BiasDelta.Data, col);
                (v + v_bias).CopyTo(BiasDelta.Data, col);
            }

            for (; col < Columns; col++) {
                BiasDelta.Data[col] += Delta.Data[col];
            }
        }

        public void AdjustWeights(CpuMatrix WeightsDelta, float scale, CpuMatrix Weights, CpuMatrix WeightsT) {
            if (Weights.Rows != WeightsDelta.Rows) throw new Exception("Weights and WeightsDelta don't have matching Rows!");
            if (Weights.Columns != WeightsDelta.Columns) throw new Exception("Weights and WeightsDelta don't have matching Columns!");

            int length = Weights.Rows * Weights.Columns;

            int i = 0;
            for (; i <= length - _simdWidth; i += _simdWidth) {
                var v_delta = new Vector<float>(WeightsDelta.Data, i);
                v_delta *= scale;
                var v = new Vector<float>(Weights.Data, i);
                (v - v_delta).CopyTo(Weights.Data, i);
            }
            for (; i < length; i++) {
                Weights.Data[i] -= WeightsDelta.Data[i] * scale;
            }

            for (int row = 0; row < WeightsT.Rows; row++) {
                for (int col = 0; col < WeightsT.Columns; col++) {
                    WeightsT[row, col] -= WeightsDelta[col, row] * scale;
                }
            }
        }
        public void AdjustBias(CpuVector BiasDelta, float scale, CpuVector Bias) {
            if (Bias.Length != BiasDelta.Length) throw new Exception("Bias and BiasDelta don't have matching Lengths!");

            int length = Bias.Length;

            int i = 0;
            for (; i <= length - _simdWidth; i += _simdWidth) {
                var v_delta = new Vector<float>(BiasDelta.Data, i);
                v_delta *= scale;
                var v = new Vector<float>(Bias.Data, i);
                (v - v_delta).CopyTo(Bias.Data, i);
            }
            for (; i < length; i++) {
                Bias.Data[i] -= BiasDelta.Data[i] * scale;
            }
        }

        public void Addiction(ref float[] Out, float[] A) {
            int i = 0;
            for (; i <= Out.Length - _simdWidth; i++) {
                var v = new Vector<float>(A, i);
                var v_out = new Vector<float>(Out, i);
                (v + v_out).CopyTo(Out, i);
            }

            for (; i < Out.Length; i++) {
                Out[i] += A[i];
            }
        }
    }
    
    public class GpuFunctions(Accelerator Accelerator) : INetworkFunctions {
        
        static void CalculateValueKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> weights,
            ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> input,
            ArrayView1D<float, Stride1D.Dense> valuesOut, ArrayView1D<float, Stride1D.Dense> activationOut,
            int inputSize) {

            int row = index;
            float sum = bias[row];
            int offset = row * inputSize;

            for (int col = 0; col < inputSize; col++) {
                sum += weights[offset + col] * input[col];
            }

            valuesOut[row] = sum;
            activationOut[row] = sum > 0 ? sum : 0
        }
        public void CalculateValue(GpuTensor Weights, GpuTensor Bias, GpuTensor input,
            Func<float, float> ActivationFunc, GpuTensor ValuesOut, GpuTensor ActivationOut) {

            var kernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, 
                int
            >(CalculateValueKernel);

            kernel(
                Weights.Data.IntExtent / input.TotalSize,
                Weights.Data.BaseView,
                Bias.Data.BaseView,
                input.Data.BaseView,
                ValuesOut.Data.BaseView,
                ActivationOut.Data.BaseView,
                input.TotalSize);
            
            Accelerator.Synchronize();
        }

        public void Backward(CpuMatrix WeightsT, CpuVector Delta, CpuVector Values, Func<float, float> ActivationFuncDer, CpuVector DeltaOut) {
        }
        public void BackwardOutput(CpuVector Activation, CpuVector CorrectValues, CpuVector DeltaOut) {
        }
        public void BackwardBias(CpuVector Delta, CpuVector BiasDelta) {
        }

        public void AdjustWeights(CpuMatrix WeightsDelta, float scale, CpuMatrix Weights, CpuMatrix WeightsT) {
        }
        public void AdjustBias(CpuVector BiasDelta, float scale, CpuVector Bias) {
        }

        public void Addiction(ref float[] Out, float[] A) {
        }
    }
    */
}