using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeuralNetworkSystem {
    /*
    
    These variables here are the functions the whole program needs in order to work.

    InputNormalization: A change we can do to our input in order to make it more suitable for our network.

    OutputActivation: The function which we will use in order to properly activate the neurons of the last layer, in order to get the correct values.
                      Note that this function is specifically for the output, since if we were to put it in the hidden layers, it would break the strength
                      of each activation.

    LossCalculation
    */
    public delegate float[] InputNormalization(float[] input);
    public delegate float[] OutputActivation(float[] input);
    public delegate float LossCalculation(float[] V, int label);

    public struct LayerFunctions {
        public INetworkFunctions NetworkFunctions;
        public InputNormalization normalization;
        public OutputActivation output_activation;

        public Func<float, float> activation_function;
        public Func<float, float> activation_function_derivative;

        public LayerFunctions(
            INetworkFunctions network_functions,
            InputNormalization normalization,
            OutputActivation output_activation,

            Func<float, float> activation_function,
            Func<float, float> activation_function_derivative
            ) {
            this.NetworkFunctions = network_functions;
            this.normalization = normalization;
            this.output_activation = output_activation;

            this.activation_function = activation_function;
            this.activation_function_derivative = activation_function_derivative;
        }
    }

    public class FunctionManager {
        public static Func<float, float> GetActivationFunction(ActivationFunctionsTypes type) {
            switch (type) {
                case ActivationFunctionsTypes.Sigmoid: return ActivationFunctions.Sigmoid;
                case ActivationFunctionsTypes.ReLU: return ActivationFunctions.ReLU;
            }
            return null;
        }
        public static Func<float, float> GetActivationDerivativeFunction(ActivationFunctionsTypes type) {
            switch (type) {
                case ActivationFunctionsTypes.Sigmoid: return ActivationFunctions.SigmoidDerivative;
                case ActivationFunctionsTypes.ReLU: return ActivationFunctions.ReLUDerivative;
            }
            return null;
        }
        public static InputNormalization GetInputNormalizationFunction(InputNormalizationFunctionsType type) {
            switch (type) {
                case InputNormalizationFunctionsType.None: return InputNormalizationFunctions.None;
                case InputNormalizationFunctionsType.NormalizeMeadian: return InputNormalizationFunctions.NormalizeMedian;
            }
            return null;
        }
        public static OutputActivation GetOutputFunction(OutputFunctionsType type) {
            switch (type) {
                case OutputFunctionsType.SoftMax: return OutputFunctions.SoftMax;
            }
            return null;
        }
        public static LossCalculation GetLossFunction(LossFunctionsType type) {
            switch (type) {
                case LossFunctionsType.Mean: return LossFunctions.Mean;
                case LossFunctionsType.SoftMax: return LossFunctions.SoftMax;
            }
            return null;
        }

        public static INetworkFunctions GetNetworkFunctions() {
            if (Avx.IsSupported) {
                if (ProgramManager.useAVX512) return new AVX512Functions();
                return new AVXFunctions();
            } else if (Sse.IsSupported) return new SSEFunctions();
            return new ScalarFunctions();
        }

        public static LayerFunctions GetFunctions(
                InputNormalizationFunctionsType normalization,
                OutputFunctionsType output_activation,

                ActivationFunctionsTypes activation_function
            ) {
            return new LayerFunctions(
                GetNetworkFunctions(),
                GetInputNormalizationFunction(normalization),
                GetOutputFunction(output_activation),

                GetActivationFunction(activation_function),
                GetActivationDerivativeFunction(activation_function)
                );
        }
    }

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
            float mean = 0.1307f;
            float std = 0.3081f;

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

    public enum NetworkFunctionsTypes {
        Scalar,
        SSE,
        AVX,
        AVX512
    }

    public abstract class INetworkFunctions {
        // Value = Weights * Input + Bias
        // Activation = ActivationFunc(Value)
        public virtual void CalculateValue(Matrix Weights, Vector Bias, Vector input, Func<float, float> ActivationFunc, Vector ValuesOut, Vector ActivationOut) { }

        // DeltaOut = (NextLayer-WeightsT * NextLayer-Delta) ⊙ ActivationFuncDer(Values)
        public virtual void Backward(Matrix WeightsT, Vector Delta, Vector Values, Func<float, float> ActivationFuncDer, Vector DeltaOut) { }

        // DeltaOut = Activation - CorrectValues
        public virtual void BackwardOutput(Vector Activation, Vector CorrectValues, Vector DeltaOut) { }

        // WeightsDelta += Delta * PreviousLayer-Activation
        public virtual void BackwardWeights(Vector Delta, Vector Activation, Matrix WeightDelta) { }

        // BiasDelta += Delta
        public virtual void BackwardBias(Vector Delta, Vector BiasDelta) { }

        // Weights -= WeightsDelta * scale
        // WeightsT -= WeightsDelta * scale (Scalar)
        public virtual void AdjustWeights(Matrix WeightsDelta, float scale, Matrix Weights, Matrix WeightsT) { }
        
        // Bias -= BiasDelta * scale
        public virtual void AdjustBias(Vector BiasDelta, float scale, Vector Bias) { }

        // Out = A + B
        public virtual void Addiction(ref float[] Out, float[] A) { }
    }

    public class ScalarFunctions : INetworkFunctions {
        public override void CalculateValue(Matrix Weights, Vector Bias, Vector input, Func<float, float> ActivationFunc, Vector ValuesOut, Vector ActivationOut) {
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

        public override void Backward(Matrix WeightsT, Vector Delta, Vector Values, Func<float, float> ActivationFuncDer, Vector DeltaOut) {
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
        public override void BackwardOutput(Vector Activation, Vector CorrectValues, Vector DeltaOut) {
            int Rows = Activation.Length;

            for (int row = 0; row < Rows; row++) {
                DeltaOut[row] = Activation[row] - CorrectValues[row];
            }
        }
        public override void BackwardWeights(Vector Delta, Vector Activation, Matrix WeightDelta) {
            int Rows = Delta.Length;
            int Columns = Activation.Length;

            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Columns; col++) {
                    WeightDelta[row, col] += Delta[row] * Activation[col];
                }
            }
        }
        public override void BackwardBias(Vector Delta, Vector BiasDelta) {
            int Rows = Delta.Length;

            for (int row = 0; row < Rows; row++) {
                BiasDelta[row] += Delta[row];
            }
        }

        public override void AdjustWeights(Matrix WeightsDelta, float scale, Matrix Weights, Matrix WeightsT) {
            int Rows = WeightsDelta.Rows;
            int Columns = WeightsDelta.Columns;

            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Columns; col++) {
                    Weights[row, col] -= WeightsDelta[row, col] * scale;
                    WeightsT[col, row] -= WeightsDelta[row, col] * scale;
                }
            }
        }
        public override void AdjustBias(Vector BiasDelta, float scale, Vector Bias) {
            int Rows = BiasDelta.Length;

            for (int row = 0; row < Rows; row++) {
                Bias[row] -= BiasDelta[row] * scale;
            }
        }

        public override void Addiction(ref float[] Out, float[] A) {
            for (int i = 0; i < Out.Length; i++) {
                Out[i] += A[i];
            }
        }
    }

    public class SSEFunctions : INetworkFunctions {
        int simd_width = Vector128<float>.Count;

        public override void CalculateValue(Matrix Weights, Vector Bias, Vector input, Func<float, float> ActivationFunc, Vector ValuesOut, Vector ActivationOut) {
            for (int row = 0; row < Weights.Rows; row++) {
                float sum = Bias[row];
                int offset = row * Weights.Columns;

                int col = 0;
                for (; col <= Weights.Columns - simd_width; col += simd_width) {
                    var v_weights = Vector128.Create<float>(Weights.Data, offset + col);
                    var v_x = Vector128.Create<float>(input.Data, col);
                    sum += Vector128.Dot(v_weights, v_x);
                }

                for (; col < Weights.Columns; col++) {
                    sum += Weights.Data[offset + col] * input.Data[col];
                }

                ValuesOut[row] = sum;
                ActivationOut[row] = ActivationFunc(sum);
            }
        }

        public override void Backward(Matrix WeightsT, Vector Delta, Vector Values, Func<float, float> ActivationFuncDer, Vector DeltaOut) {
            int Rows = WeightsT.Rows;
            int Columns = WeightsT.Columns;

            for (int row = 0; row < Rows; row++) {
                float sum = 0f;
                int offset = row * Columns;

                int col = 0;
                for (; col <= Columns - simd_width; col += simd_width) {
                    var v_weights = Vector128.Create<float>(WeightsT.Data, offset + col);
                    var v_delta = Vector128.Create<float>(Delta.Data, col);
                    sum += Vector128.Dot(v_weights, v_delta);
                }

                for (; col < Columns; col++) {
                    sum += WeightsT.Data[offset + col] * Delta.Data[col];
                }

                DeltaOut.Data[row] = sum * ActivationFuncDer(Values.Data[row]);
            }
        }
        public override void BackwardOutput(Vector Activation, Vector CorrectValues, Vector DeltaOut) {
            int Rows = Activation.Length;

            int i = 0;
            for (; i <= Rows - simd_width; i += simd_width) {
                var v_a = Vector128.Create<float>(Activation.Data, i);
                var v_b = Vector128.Create<float>(CorrectValues.Data, i);
                (v_a - v_b).CopyTo(DeltaOut.Data, i);
            }
            for (; i < Rows; i++) {
                DeltaOut.Data[i] = Activation[i] - CorrectValues[i];
            }
        }
        public override void BackwardWeights(Vector Delta, Vector Activation, Matrix WeightDelta) {
            int Rows = Delta.Length;
            int Columns = Activation.Length;

            for (int row = 0; row < Rows; row++) {
                int offset = row * Columns;

                int col = 0;
                for (; col <= Columns - simd_width; col += simd_width) {
                    var v = Vector128.Create<float>(Activation.Data, col);
                    var v_weight = Vector128.Create<float>(WeightDelta.Data, offset + col);

                    v = Delta.Data[row] * v;
                    (v + v_weight).CopyTo(WeightDelta.Data, offset + col);
                }

                for (; col < Columns; col++) {
                    WeightDelta.Data[offset + col] += Delta.Data[row] * Activation.Data[col];
                }
            }
        }
        public override void BackwardBias(Vector Delta, Vector BiasDelta) {
            int Columns = Delta.Length;

            int col = 0;
            for (; col <= Columns - simd_width; col += simd_width) {
                var v = Vector128.Create<float>(Delta.Data, col);
                var v_bias = Vector128.Create<float>(BiasDelta.Data, col);
                (v + v_bias).CopyTo(BiasDelta.Data, col);
            }

            for (; col < Columns; col++) {
                BiasDelta.Data[col] += Delta.Data[col];
            }
        }

        public override void AdjustWeights(Matrix WeightsDelta, float scale, Matrix Weights, Matrix WeightsT) {
            if (Weights.Rows != WeightsDelta.Rows) throw new Exception("Weights and WeightsDelta don't have matching Rows!");
            if (Weights.Columns != WeightsDelta.Columns) throw new Exception("Weights and WeightsDelta don't have matching Columns!");

            int length = Weights.Rows * Weights.Columns;

            int i = 0;
            for (; i <= length - simd_width; i += simd_width) {
                var v_delta = Vector128.Create<float>(WeightsDelta.Data, i);
                v_delta *= scale;
                var v = Vector128.Create<float>(Weights.Data, i);
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
        public override void AdjustBias(Vector BiasDelta, float scale, Vector Bias) {
            if (Bias.Length != BiasDelta.Length) throw new Exception("Bias and BiasDelta don't have matching Lengths!");

            int length = Bias.Length;

            int i = 0;
            for (; i <= length - simd_width; i += simd_width) {
                var v_delta = Vector128.Create<float>(BiasDelta.Data, i);
                v_delta *= scale;
                var v = Vector128.Create<float>(Bias.Data, i);
                (v - v_delta).CopyTo(Bias.Data, i);
            }
            for (; i < length; i++) {
                Bias.Data[i] -= BiasDelta.Data[i] * scale;
            }
        }

        public override void Addiction(ref float[] Out, float[] A) {
            int i = 0;
            for (; i <= Out.Length - simd_width; i++) {
                var v = Vector128.Create<float>(A, i);
                var v_out = Vector128.Create<float>(Out, i);
                (v + v_out).CopyTo(Out, i);
            }

            for (; i < Out.Length; i++) {
                Out[i] += A[i];
            }
        }
    }

    public class AVXFunctions : INetworkFunctions {
        int simd_width = Vector256<float>.Count;

        public override void CalculateValue(Matrix Weights, Vector Bias, Vector input, Func<float, float> ActivationFunc, Vector ValuesOut, Vector ActivationOut) {
            for (int row = 0; row < Weights.Rows; row++) {
                float sum = Bias[row];
                int offset = row * Weights.Columns;

                int col = 0;
                for (; col <= Weights.Columns - simd_width; col += simd_width) {
                    var v_weights = Vector256.Create<float>(Weights.Data, offset + col);
                    var v_x = Vector256.Create<float>(input.Data, col);
                    sum += Vector256.Dot(v_weights, v_x);
                }

                for (; col < Weights.Columns; col++) {
                    sum += Weights.Data[offset + col] * input.Data[col];
                }

                ValuesOut[row] = sum;
                ActivationOut[row] = ActivationFunc(sum);
            }
        }

        public override void Backward(Matrix WeightsT, Vector Delta, Vector Values, Func<float, float> ActivationFuncDer, Vector DeltaOut) {
            int Rows = WeightsT.Rows;
            int Columns = WeightsT.Columns;

            for (int row = 0; row < Rows; row++) {
                float sum = 0f;
                int offset = row * Columns;

                int col = 0;
                for (; col <= Columns - simd_width; col += simd_width) {
                    var v_weights = Vector256.Create<float>(WeightsT.Data, offset + col);
                    var v_delta = Vector256.Create<float>(Delta.Data, col);
                    sum += Vector256.Dot(v_weights, v_delta);
                }

                for (; col < Columns; col++) {
                    sum += WeightsT.Data[offset + col] * Delta.Data[col];
                }

                DeltaOut.Data[row] = sum * ActivationFuncDer(Values.Data[row]);
            }
        }
        public override void BackwardOutput(Vector Activation, Vector CorrectValues, Vector DeltaOut) {
            int Rows = Activation.Length;

            int i = 0;
            for (; i <= Rows - simd_width; i += simd_width) {
                var v_a = Vector256.Create<float>(Activation.Data, i);
                var v_b = Vector256.Create<float>(CorrectValues.Data, i);
                (v_a - v_b).CopyTo(DeltaOut.Data, i);
            }
            for (; i < Rows; i++) {
                DeltaOut.Data[i] = Activation[i] - CorrectValues[i];
            }
        }
        public override void BackwardWeights(Vector Delta, Vector Activation, Matrix WeightDelta) {
            int Rows = Delta.Length;
            int Columns = Activation.Length;

            for (int row = 0; row < Rows; row++) {
                int offset = row * Columns;

                int col = 0;
                for (; col <= Columns - simd_width; col += simd_width) {
                    var v = Vector256.Create<float>(Activation.Data, col);
                    var v_weight = Vector256.Create<float>(WeightDelta.Data, offset + col);

                    v = Delta.Data[row] * v;
                    (v + v_weight).CopyTo(WeightDelta.Data, offset + col);
                }

                for (; col < Columns; col++) {
                    WeightDelta.Data[offset + col] += Delta.Data[row] * Activation.Data[col];
                }
            }
        }
        public override void BackwardBias(Vector Delta, Vector BiasDelta) {
            int Columns = Delta.Length;

            int col = 0;
            for (; col <= Columns - simd_width; col += simd_width) {
                var v = Vector256.Create<float>(Delta.Data, col);
                var v_bias = Vector256.Create<float>(BiasDelta.Data, col);
                (v + v_bias).CopyTo(BiasDelta.Data, col);
            }

            for (; col < Columns; col++) {
                BiasDelta.Data[col] += Delta.Data[col];
            }
        }

        public override void AdjustWeights(Matrix WeightsDelta, float scale, Matrix Weights, Matrix WeightsT) {
            if (Weights.Rows != WeightsDelta.Rows) throw new Exception("Weights and WeightsDelta don't have matching Rows!");
            if (Weights.Columns != WeightsDelta.Columns) throw new Exception("Weights and WeightsDelta don't have matching Columns!");

            int length = Weights.Rows * Weights.Columns;

            int i = 0;
            for (; i <= length - simd_width; i += simd_width) {
                var v_delta = Vector256.Create<float>(WeightsDelta.Data, i);
                v_delta *= scale;
                var v = Vector256.Create<float>(Weights.Data, i);
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
        public override void AdjustBias(Vector BiasDelta, float scale, Vector Bias) {
            if (Bias.Length != BiasDelta.Length) throw new Exception("Bias and BiasDelta don't have matching Lengths!");

            int length = Bias.Length;

            int i = 0;
            for (; i <= length - simd_width; i += simd_width) {
                var v_delta = Vector256.Create<float>(BiasDelta.Data, i);
                v_delta *= scale;
                var v = Vector256.Create<float>(Bias.Data, i);
                (v - v_delta).CopyTo(Bias.Data, i);
            }
            for (; i < length; i++) {
                Bias.Data[i] -= BiasDelta.Data[i] * scale;
            }
        }

        public override void Addiction(ref float[] Out, float[] A) {
            int i = 0;
            for (; i <= Out.Length - simd_width; i++) {
                var v = Vector256.Create<float>(A, i);
                var v_out = Vector256.Create<float>(Out, i);
                (v + v_out).CopyTo(Out, i);
            }

            for (; i < Out.Length; i++) {
                Out[i] += A[i];
            }
        }
    }

    public class AVX512Functions : INetworkFunctions{
        int simd_width = Vector512<float>.Count;

        public override void CalculateValue(Matrix Weights, Vector Bias, Vector input, Func<float, float> ActivationFunc, Vector ValuesOut, Vector ActivationOut) {
            for (int row = 0; row < Weights.Rows; row++) {
                float sum = Bias[row];
                int offset = row * Weights.Columns;

                int col = 0;
                for (; col <= Weights.Columns - simd_width; col += simd_width) {
                    var v_weights = Vector512.Create<float>(Weights.Data, offset + col);
                    var v_x = Vector512.Create<float>(input.Data, col);
                    sum += Vector512.Dot(v_weights, v_x);
                }

                for (; col < Weights.Columns; col++) {
                    sum += Weights.Data[offset + col] * input.Data[col];
                }

                ValuesOut[row] = sum;
                ActivationOut[row] = ActivationFunc(sum);
            }
        }

        public override void Backward(Matrix WeightsT, Vector Delta, Vector Values, Func<float, float> ActivationFuncDer, Vector DeltaOut) {
            int Rows = WeightsT.Rows;
            int Columns = WeightsT.Columns;

            for (int row = 0; row < Rows; row++) {
                float sum = 0f;
                int offset = row * Columns;

                int col = 0;
                for (; col <= Columns - simd_width; col += simd_width) {
                    var v_weights = Vector512.Create<float>(WeightsT.Data, offset + col);
                    var v_delta = Vector512.Create<float>(Delta.Data, col);
                    sum += Vector512.Dot(v_weights, v_delta);
                }

                for (; col < Columns; col++) {
                    sum += WeightsT.Data[offset + col] * Delta.Data[col];
                }

                DeltaOut.Data[row] = sum * ActivationFuncDer(Values.Data[row]);
            }
        }
        public override void BackwardOutput(Vector Activation, Vector CorrectValues, Vector DeltaOut) {
            int Rows = Activation.Length;

            int i = 0;
            for (; i <= Rows - simd_width; i += simd_width) {
                var v_a = Vector512.Create<float>(Activation.Data, i);
                var v_b = Vector512.Create<float>(CorrectValues.Data, i);
                (v_a - v_b).CopyTo(DeltaOut.Data, i);
            }
            for (; i < Rows; i++) {
                DeltaOut.Data[i] = Activation[i] - CorrectValues[i];
            }
        }
        public override void BackwardWeights(Vector Delta, Vector Activation, Matrix WeightDelta) {
            int Rows = Delta.Length;
            int Columns = Activation.Length;

            for (int row = 0; row < Rows; row++) {
                int offset = row * Columns;

                int col = 0;
                for (; col <= Columns - simd_width; col += simd_width) {
                    var v = Vector512.Create<float>(Activation.Data, col);
                    var v_weight = Vector512.Create<float>(WeightDelta.Data, offset + col);

                    v = Delta.Data[row] * v;
                    (v + v_weight).CopyTo(WeightDelta.Data, offset + col);
                }

                for (; col < Columns; col++) {
                    WeightDelta.Data[offset + col] += Delta.Data[row] * Activation.Data[col];
                }
            }
        }
        public override void BackwardBias(Vector Delta, Vector BiasDelta) {
            int Columns = Delta.Length;

            int col = 0;
            for (; col <= Columns - simd_width; col += simd_width) {
                var v = Vector512.Create<float>(Delta.Data, col);
                var v_bias = Vector512.Create<float>(BiasDelta.Data, col);
                (v + v_bias).CopyTo(BiasDelta.Data, col);
            }

            for (; col < Columns; col++) {
                BiasDelta.Data[col] += Delta.Data[col];
            }
        }

        public override void AdjustWeights(Matrix WeightsDelta, float scale, Matrix Weights, Matrix WeightsT) {
            if (Weights.Rows != WeightsDelta.Rows) throw new Exception("Weights and WeightsDelta don't have matching Rows!");
            if (Weights.Columns != WeightsDelta.Columns) throw new Exception("Weights and WeightsDelta don't have matching Columns!");

            int length = Weights.Rows * Weights.Columns;

            int i = 0;
            for (; i <= length - simd_width; i += simd_width) {
                var v_delta = Vector512.Create<float>(WeightsDelta.Data, i);
                v_delta *= scale;
                var v = Vector512.Create<float>(Weights.Data, i);
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
        public override void AdjustBias(Vector BiasDelta, float scale, Vector Bias) {
            if (Bias.Length != BiasDelta.Length) throw new Exception("Bias and BiasDelta don't have matching Lengths!");

            int length = Bias.Length;

            int i = 0;
            for (; i <= length - simd_width; i += simd_width) {
                var v_delta = Vector512.Create<float>(BiasDelta.Data, i);
                v_delta *= scale;
                var v = Vector512.Create<float>(Bias.Data, i);
                (v - v_delta).CopyTo(Bias.Data, i);
            }
            for (; i < length; i++) {
                Bias.Data[i] -= BiasDelta.Data[i] * scale;
            }
        }

        public override void Addiction(ref float[] Out, float[] A) {
            int i = 0;
            for (; i <= Out.Length - simd_width; i++) {
                var v = Vector512.Create<float>(A, i);
                var v_out = Vector512.Create<float>(Out, i);
                (v + v_out).CopyTo(Out, i);
            }

            for (; i < Out.Length; i++) {
                Out[i] += A[i];
            }
        }
    }
}
