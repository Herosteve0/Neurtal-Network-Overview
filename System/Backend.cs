using System.Numerics;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using Neurtal_Network_Overview.Data;

namespace Neurtal_Network_Overview.System {
	
	public interface IBackend {
		/**
		 * Creates an appropriate tensor.
		 */
		ITensor CreateTensor(params int[] shape);
		/**
		 * Creates an appropriate tensor with the given data.
		 * Data is copied.
		 */
		ITensor CreateTensor(float[] data, params int[] shape);
		
		/**
		 * Creates an appropriate tensor filled with Zeros.
		 */
		ITensor Zero(params int[] shape);
		/**
		 * Creates an appropriate randomized tensor.
		 */
		ITensor Random(Random random, float min, float max, params int[] shape);

		/**
		 * Sets the tensor values to a given array of floats.
		 */
		void Set(ITensor a, float[] data);
		/**
		 * Convers the tensor into an array of floats.
		 */
		float[] Read(ITensor a);
		
		/**
		 * Returns a Transposed Matrix.
		 */
		void MatrixTranspose(ITensor a, ITensor output);

		/**
		 * Returns the index of the max element.
		 */
		int MaxVectorIndex(ITensor a);
		
		/**
		 * Returns the loss of a given Vector.
		 */
		float Loss(DataGuess<float> a);
		
		/**
		 * The calculations for Activations of the first layer.
		 */
		void Input(ITensor input, ITensor activationOut);

		/**
		 * The calculation for the Values of a Layer.
		 */
		void Linear(ITensor input, ITensor weights, ITensor bias, ITensor valuesOut);
		
		/**
		 * The calculations for the Activations of a Layer.
		 */
		void Activation(ITensor values, ITensor activationOut);
		/**
		 * The calculations for the Output of the last layer.
		 */
		void Output(ITensor values, ITensor activationOut);

		/**
		 * The calculations for the Delta value that will be applied to the Weights and Biases.
		 */
		void Backward(ITensor values, ITensor delta, ITensor weightsT, ITensor deltaOut);
		/**
		 * The calculations for the initial Delta value at the output Layer.
		 */
		void BackwardOutput(ITensor activations, ITensor expectedValues, ITensor deltaOut);
		/**
		 * The calculations for the change that will be applied to the Weights and Biases.
		 */
		void BackPropagation(ITensor delta, ITensor activations, ITensor weightDelta, ITensor biasDelta);
		
		/**
		 * Adjusts the Weights and Biases in proportion with the calculated Delta.
		 */
		void Adjust(ITensor weightDelta, ITensor biasDelta, float scale, ITensor weights, ITensor weightsT, ITensor bias);
	}
	
	// CPU

	public abstract class CpuBackend : IBackend {
		public ITensor CreateTensor(params int[] shape) {
			return new CpuTensor(shape);
		}
		public ITensor CreateTensor(float[] data, params int[] shape) {
			return new CpuTensor(data, shape);
		}
		public ITensor Zero(params int[] shape) {
			return CpuTensor.Zero(shape);
		}
		public ITensor Random(Random random, float min, float max, params int[] shape) {
			return CpuTensor.Random(random, min, max, shape);
		}

		public void Set(ITensor a, float[] data) {
			((CpuTensor)a).Data = data;
		}
		public float[] Read(ITensor a) {
			return ((CpuTensor)a).Data.ToArray();
		}

		public void MatrixTranspose(ITensor a, ITensor output) {
			CpuMatrixTranpose((CpuTensor)a, (CpuTensor)output);
		}
		public int MaxVectorIndex(ITensor a) {
			return CpuMaxVectorIndex((CpuTensor)a);
		}

		public float Loss(DataGuess<float> guess) {
			return -(float)Math.Log(((CpuTensor)guess.Output)[guess.Data.Label]);
		}

		public void Input(ITensor input, ITensor activationOut) {
			CpuInput((CpuTensor)input, (CpuTensor)activationOut);
		}

		public void Linear(ITensor input, ITensor weights, ITensor bias, ITensor valuesOut) {
			CpuLinear((CpuTensor)input, (CpuTensor)weights, (CpuTensor)bias, (CpuTensor)valuesOut);
		}
		public void Activation(ITensor values, ITensor activationOut) {
			CpuActivation((CpuTensor)values, (CpuTensor)activationOut);
		}
		public void Output(ITensor values, ITensor activationOut) {
			CpuOutput((CpuTensor)values, (CpuTensor)activationOut);
		}

		public void Backward(ITensor values, ITensor delta, ITensor weightsT, ITensor deltaOut) {
			CpuBackward((CpuTensor)values, (CpuTensor)delta, (CpuTensor)weightsT, (CpuTensor)deltaOut);
		}

		public void BackwardOutput(ITensor activations, ITensor expectedValues, ITensor deltaOut) {
			CpuBackwardOutput((CpuTensor)activations, (CpuTensor)expectedValues, (CpuTensor)deltaOut);
		}

		public void BackPropagation(ITensor delta, ITensor activations, ITensor weightDelta, ITensor biasDelta) {
			CpuBackPropagation((CpuTensor)delta, (CpuTensor)activations, (CpuTensor)weightDelta, (CpuTensor)biasDelta);
		}

		public void Adjust(ITensor weightDelta, ITensor biasDelta, float scale, ITensor weights, ITensor weightsT, ITensor bias) {
			CpuAdjust((CpuTensor)weightDelta, (CpuTensor)biasDelta, scale, (CpuTensor)weights, (CpuTensor)weightsT, (CpuTensor)bias);
		}

		protected abstract int  CpuMaxVectorIndex(CpuTensor a);
		protected abstract void CpuMatrixTranpose(CpuTensor a, CpuTensor output);
		
		protected abstract void CpuInput(CpuTensor input, CpuTensor activationOut);
		
		protected abstract void CpuLinear(CpuTensor input, CpuTensor weights, CpuTensor bias, CpuTensor valuesOut);
		protected abstract void CpuActivation(CpuTensor values, CpuTensor activationOut);
		protected abstract void CpuOutput(CpuTensor values, CpuTensor activationOut);
		
		protected abstract void CpuBackward(CpuTensor values, CpuTensor delta, CpuTensor weightsT, CpuTensor deltaOut);
		protected abstract void CpuBackwardOutput(CpuTensor activations, CpuTensor expectedValues, CpuTensor deltaOut);
		protected abstract void CpuBackPropagation(CpuTensor delta, CpuTensor activations, CpuTensor weightDelta, CpuTensor biasDelta);
		
		protected abstract void CpuAdjust(CpuTensor weightDelta, CpuTensor biasDelta, float scale, CpuTensor weights, CpuTensor weightsT, CpuTensor bias);
	}

	/**
	 * Mathematically correct backend.
	 */
	public class ScalarBackend : CpuBackend {
		public ScalarBackend() {
			Console.WriteLine("Using Scalar Backend.");
		}
		
		private static void Addition(CpuTensor a, CpuTensor b, CpuTensor output) {
			for (int i = 0; i < output.Length; i++) {
				output.Data[i] = a.Data[i] + b.Data[i];
			}
		}
		private static void Subtraction(CpuTensor a, CpuTensor b, CpuTensor output) {
			for (int i = 0; i < output.Length; i++) {
				output.Data[i] = a.Data[i] - b.Data[i];
			}
		}
		private static void Scale(CpuTensor a, float scalar, CpuTensor output) {
			for (int i = 0; i < output.Length; i++) {
				output.Data[i] = a.Data[i] * scalar;
			}
		}
		private static void MatrixVectorMultiplication(CpuTensor a, CpuTensor b, CpuTensor output) {
			for (int row = 0; row < a.Shape[0]; row++) {
				float sum = 0;
				for (int col = 0; col < a.Shape[1]; col++) {
					 sum += a[row, col] * b[col];
				}
				output[row] = sum;
			}
		}
		private static void VectorVectorMultiplication(CpuTensor a, CpuTensor b, CpuTensor output) {
			for (int row = 0; row < a.Shape[0]; row++) {
				for (int col = 0; col < b.Shape[0]; col++) {
					output[row, col] = a[row] * b[col];
				}
			}
		}
		private static void ElementMultiplication(CpuTensor a, CpuTensor b, CpuTensor output) {
			for (int i = 0; i < a.Length; i++) {
				output.Data[i] = a.Data[i] * b.Data[i];
			}
		}
		private static void VectorNormalize(CpuTensor a, CpuTensor output) {
            const float mean = 0.1307f;
            const float std = 0.3081f;

            for (int i = 0; i < a.Shape[0]; i++) {
                output[i] = (a[i] - mean) / std;
            }
		}
		private static void VectorReLu(CpuTensor a, CpuTensor output) {
			for (int i = 0; i < a.Shape[0]; i++) {
				output[i] = a[i] > 0 ? a[i] : 0;
			}
		}
		private static void VectorReLuDerivative(CpuTensor a, CpuTensor output) {
			for (int i = 0; i < a.Shape[0]; i++) {
				output[i] = a[i] > 0 ? 1 : 0;
			}
		}
		private static void VectorSoftMax(CpuTensor a, CpuTensor output) {
			float max = a[0];
			for (int i = 1; i < a.Shape[0]; i++) {
				if (max < a[i]) max = a[i];
			}

			float sum = 0f;
			for (int i = 0; i < a.Shape[0]; i++) {
				float e = (float)Math.Exp(a[i] - max);
				output[i] = e;
				sum += e;
			}

			for (int i = 0; i < a.Shape[0]; i++) {
				output[i] /= sum;
			}
		}

		protected override void CpuMatrixTranpose(CpuTensor a, CpuTensor output) {
			for (int row = 0; row < a.Shape[0]; row++) {
				for (int col = 0; col < a.Shape[1]; col++) {
					output[col, row] = a[row, col];
				}
			}
		}

		protected override int CpuMaxVectorIndex(CpuTensor a) {
			int max = 0;
			for (int i = 1; i < a.Shape[0]; i++) {
				if (a[i] > a[max]) max = i;
			}
			return max;
		}
		
		protected override void CpuInput(CpuTensor input, CpuTensor activationOut) {
			/*
			 * InputNormalizing(Input)
			 */
			VectorNormalize(input, activationOut);
		}
		
		protected override void CpuLinear(CpuTensor input, CpuTensor weights, CpuTensor bias, CpuTensor valuesOut) {
			CpuTensor temp = new CpuTensor(weights.Shape[0]);

			/*
			 * Weights * Input
			 */
			MatrixVectorMultiplication(weights, input, temp);
			
			/*
			 * (Weights * Input) + Bias
			 */
			Addition(temp, bias, valuesOut);
		}
		protected override void CpuActivation(CpuTensor values, CpuTensor activationOut) {
			/*
			 * ActivationFunc(Value)
			 */
			VectorReLu(values, activationOut);
		}
		protected override void CpuOutput(CpuTensor values, CpuTensor activationOut) {
			/*
			 * OutputFunc(Value)
			 */
			VectorSoftMax(values, activationOut);
		}

		protected override void CpuBackward(CpuTensor values, CpuTensor delta, CpuTensor weightsT, CpuTensor deltaOut) {
			CpuTensor temp = new CpuTensor(weightsT.Shape[0]);
			
			/*
			 * Weights^T * Delta
			 */
			MatrixVectorMultiplication(weightsT, delta, temp);

			CpuTensor tempDer = new CpuTensor(weightsT.Shape[0]);
			
			/*
			 * 
			 */
			VectorReLuDerivative(values, tempDer);
			
			/*
			 * (Weights^T * Delta) ● ActivationFunc'(Value)
			 */
			ElementMultiplication(temp, tempDer, deltaOut);

		}
		protected override void CpuBackwardOutput(CpuTensor activations, CpuTensor expectedValues, CpuTensor deltaOut) {
			
			/*
			 * Activation - CorrectValues
			 */
			
			Subtraction(activations, expectedValues, deltaOut);
		}
		protected override void CpuBackPropagation(CpuTensor delta, CpuTensor activations, CpuTensor weightDelta, CpuTensor biasDelta) {
			CpuTensor temp = new CpuTensor(delta.Shape[0], activations.Shape[0]);
			
			/*
			 * Delta * Activations
			 */
			VectorVectorMultiplication(delta, activations, temp);
			
			/*
			 * WeightsDelta += Delta * Activations
			 */
			Addition(weightDelta, temp, weightDelta);
			
			/*
			 * BiasDelta += Delta
			 */
			Addition(biasDelta, delta, biasDelta);
		}

		protected override void CpuAdjust(CpuTensor weightDelta, CpuTensor biasDelta, float scale, CpuTensor weights, CpuTensor weightsT, CpuTensor bias) {
			CpuTensor temp = new CpuTensor(weights.Shape[0], weights.Shape[1]);
			/*
			 * WeightDelta * scale
			 */
			Scale(weightDelta, scale, temp);
			/*
			 * Weights -= WeightsDelta * scale
			 */
			Subtraction(weights, temp, weights);
			
			CpuMatrixTranpose(weights, weightsT);

			temp = new CpuTensor(bias.Shape[0]);
			/*
			 * BiasDelta * scale
			 */
			Scale(biasDelta, scale, temp);
			/*
			 * Bias -= BiasDelta * scale
			 */
			Subtraction(bias, temp, bias);
		}
	}

	/**
	 * Cpu Simd powered backend.
	 */
	public class SimdBackend : CpuBackend {
		private readonly int _simdWidth = Vector<float>.Count;

		public SimdBackend() {
			Console.WriteLine($"Using Simd Backend with width {_simdWidth}.");
		}

		protected override int CpuMaxVectorIndex(CpuTensor a) {
			ReadOnlySpan<float> span = a.Data;
			
			int[] laneIndices = new int[_simdWidth];
			for (int k = 0; k < _simdWidth; k++) laneIndices[k] = k;
			Vector<int> vIndex = new (laneIndices);
			Vector<int> vSteps = new (_simdWidth);
			
			Vector<int> vMaxIndex = Vector<int>.Zero;
			Vector<float> vMax = new (float.NegativeInfinity);

			int i = 0;
			for (; i <= a.Length - _simdWidth; i += _simdWidth) {
				Vector<float> vValues = new (span.Slice(i, _simdWidth));

				Vector<int> vMask = Vector.GreaterThan(vValues, vMax);
				
				vMax = Vector.ConditionalSelect(vMask, vValues, vMax);
				vMaxIndex = Vector.ConditionalSelect(vMask, vIndex, vMaxIndex);
				
				vIndex += vSteps;
			}
			

			int maxIndex = vMaxIndex[0];
			float max = vMax[0];
			for (int j = 1; j < _simdWidth; j++) { 
				if (vMax[j] > max) {
					max = vMax[j];
					maxIndex = vMaxIndex[j];
				}
			}

			for (; i < a.Length; i++) {
				if (span[i] > max) {
					max = span[i];
					maxIndex = i;
				}
			}

			return maxIndex;
		}

		protected override void CpuMatrixTranpose(CpuTensor a, CpuTensor output) {
			for (int row = 0; row < a.Shape[0]; row++) {
				for (int col = 0; col < a.Shape[1]; col++) {
					output[col, row] = a[row, col];
				}
			}
		}

		protected override void CpuInput(CpuTensor input, CpuTensor activationOut) {
			const float mean = 0.1307f;
			const float std = 0.3081f;

			Vector<float> vMean = new (mean);
			Vector<float> vStd = new (std);
			
			ReadOnlySpan<float> inputSpan = input.Data;
			
			Span<float> activationOutSpan = activationOut.Data;

			int i = 0;
			for (; i <= input.Length - _simdWidth; i += _simdWidth) {
				Vector<float> vInput = new (inputSpan.Slice(i, _simdWidth));
				((vInput - vMean) / vStd).CopyTo(activationOutSpan.Slice(i, _simdWidth));
			}

			for (; i < input.Length; i++) {
				activationOutSpan[i] = (input[i] - mean) / std;
			}
		}

		protected override void CpuLinear(CpuTensor input, CpuTensor weights, CpuTensor bias, CpuTensor valuesOut) {
			int rows = weights.Shape[0];
			int columns = weights.Shape[1];

			ReadOnlySpan<float> weightsSpan = weights.Data;
			ReadOnlySpan<float> inputSpan = input.Data;
			ReadOnlySpan<float> biasSpan = bias.Data;
			
			Span<float> valuesOutSpan = valuesOut.Data;
			
			for (int row = 0; row < rows; row++) {
				int offset = row * columns;

				Vector<float> vSum = Vector<float>.Zero;
				
				int col = 0;
				for (; col <= columns - _simdWidth; col += _simdWidth) {
					Vector<float> vWeights = new (weightsSpan.Slice(offset + col, _simdWidth));
					Vector<float> vInput = new (inputSpan.Slice(col, _simdWidth));
					vSum += vWeights * vInput;
				}
				
				float sum = biasSpan[row] + Vector.Sum(vSum);

				for (; col < columns; col++) {
					sum += weightsSpan[offset + col] * inputSpan[col];
				}

				valuesOutSpan[row] = sum;
			}
		}

		protected override void CpuActivation(CpuTensor values, CpuTensor activationOut) {
			ReadOnlySpan<float> valuesSpan = values.Data;
			
			Span<float> activationOutSpan = activationOut.Data;
			
			Vector<float> vZero = Vector<float>.Zero;

			int i = 0;
			for (; i <= values.Length - _simdWidth; i += _simdWidth) {
				Vector<float> vValues = new (valuesSpan.Slice(i, _simdWidth));
				Vector.Max(vValues, vZero).CopyTo(activationOutSpan.Slice(i, _simdWidth));
			}

			for (; i < values.Length; i++) {
				activationOutSpan[i] = valuesSpan[i] > 0 ?  valuesSpan[i] : 0;
			}
		}

		protected override void CpuOutput(CpuTensor values, CpuTensor activationOut) {
			ReadOnlySpan<float> valuesSpan = values.Data;
			
			Span<float> activationOutSpan = activationOut.Data;

			int i = 0;
			
			// Obtaining Max
			
			Vector<float> vMax = Vector<float>.Zero;
			for (; i <= values.Length - _simdWidth; i += _simdWidth) {
				Vector<float> vValues = new (valuesSpan.Slice(i, _simdWidth));
				vMax = Vector.Max(vMax, vValues);
			}

			float max = 0f;
			for (int j = 0; j < _simdWidth; j++) {
				max = Math.Max(max, vMax[j]);
			}

			for (; i < values.Length; i ++) {
				max = Math.Max(max, valuesSpan[i]);
			}
			
			// Creating each e^(...) value and the sum of them all
			
			vMax =  new Vector<float>(max);
			Vector<float> vSum = Vector<float>.Zero;
			
			for (i = 0; i <= values.Length - _simdWidth; i += _simdWidth) {
				Vector<float> vValues = new (valuesSpan.Slice(i, _simdWidth));
				Vector<float> vE = Vector.Exp(vValues - vMax);
				
				vE.CopyTo(activationOutSpan.Slice(i, _simdWidth));
				vSum += vE;
			}

			float sum = Vector.Sum(vSum);
			
			for (; i < values.Length; i++) {
				float e = (float)Math.Exp(valuesSpan[i] - max);
				activationOutSpan[i] = e;
				sum += e;
			}

			// Divide the final output with the sum
			
			for (i = 0; i < activationOut.Length - _simdWidth; i += _simdWidth) {
				Vector<float> vActivation = new (activationOutSpan.Slice(i, _simdWidth));
				(vActivation / sum).CopyTo(activationOutSpan.Slice(i, _simdWidth));
			}

			for (; i < activationOut.Length; i++) {
				activationOutSpan[i] /= sum;
			}
		}

		protected override void CpuBackward(CpuTensor values, CpuTensor delta, CpuTensor weightsT, CpuTensor deltaOut) {
			int rows = weightsT.Shape[0];
			int columns = weightsT.Shape[1];
			
			ReadOnlySpan<float> valuesSpan = values.Data;
			ReadOnlySpan<float> deltaSpan = delta.Data;
			ReadOnlySpan<float> weightsTSpan = weightsT.Data;
			
			Span<float> deltaOutSpan = deltaOut.Data;

			for (int row = 0; row < rows; row++) {
				int offset = row * columns;

				Vector<float> vSum = Vector<float>.Zero;
				
				int col = 0;
				for (; col <= columns - _simdWidth; col += _simdWidth) {
					Vector<float> vWeightsT = new(weightsTSpan.Slice(offset + col, _simdWidth));
					Vector<float> vDelta = new(deltaSpan.Slice(col, _simdWidth));
					vSum += vWeightsT * vDelta;
				}
				
				float sum = Vector.Sum(vSum);

				for (; col < columns; col++) {
					sum += weightsTSpan[offset + col] * deltaSpan[col];
				}

				deltaOutSpan[row] = valuesSpan[row] > 0 ? sum : 0;
			}
		}

		protected override void CpuBackwardOutput(CpuTensor activations, CpuTensor expectedValues, CpuTensor deltaOut) {
			int rows = activations.Shape[0];
			
			ReadOnlySpan<float> activationsSpan = activations.Data;
			ReadOnlySpan<float> expectedValuesSpan = expectedValues.Data;
			
			Span<float> deltaOutSpan = deltaOut.Data;

			int row = 0;
			for (; row <= rows - _simdWidth; row += _simdWidth) {
				Vector<float> vActivation = new (activationsSpan.Slice(row, _simdWidth));
				Vector<float> vExpectedValues = new (expectedValuesSpan.Slice(row, _simdWidth));
				
				(vActivation - vExpectedValues).CopyTo(deltaOutSpan.Slice(row, _simdWidth));
			}

			for (; row < activationsSpan.Length; row++) {
				deltaOutSpan[row] = activationsSpan[row] - expectedValuesSpan[row];
			}
		}

		protected override void CpuBackPropagation(CpuTensor delta, CpuTensor activations, CpuTensor weightDelta, CpuTensor biasDelta) {
			int rows = delta.Shape[0];
			int columns = activations.Shape[0];
			
			ReadOnlySpan<float> deltaSpan = delta.Data;
			ReadOnlySpan<float> activationsSpan = activations.Data;
			
			Span<float> weightDeltaSpan = weightDelta.Data;
			Span<float> biasDeltaSpan = biasDelta.Data;

			int row = 0;
			for (; row < rows; row++) {
				int offset = row * columns;

				int col = 0;
				for (; col <= columns - _simdWidth; col += _simdWidth) {
					Vector<float> vActivation = new (activationsSpan.Slice(col, _simdWidth));
					Vector<float> vWeightsDelta = new (weightDeltaSpan.Slice(offset + col, _simdWidth));
					
					(vActivation * deltaSpan[row] + vWeightsDelta).CopyTo(weightDeltaSpan.Slice(offset + col, _simdWidth));
				}

				for (; col < columns; col++) {
					weightDeltaSpan[offset + col] += deltaSpan[row] * activationsSpan[col];
				}
			}

			row = 0;
			for (; row < rows - _simdWidth; row += _simdWidth) {
				Vector<float> vDelta = new (deltaSpan.Slice(row, _simdWidth));
				Vector<float> vBiasDelta = new (biasDeltaSpan.Slice(row, _simdWidth));
				
				(vBiasDelta + vDelta).CopyTo(biasDeltaSpan.Slice(row, _simdWidth));
			}

			for (; row < rows; row++) {
				biasDeltaSpan[row] += deltaSpan[row];
			}
		}

		protected override void CpuAdjust(CpuTensor weightDelta, CpuTensor biasDelta, float scale, CpuTensor weights, CpuTensor weightsT, CpuTensor bias) {
			ReadOnlySpan<float> weightDeltaSpan = weightDelta.Data;
			ReadOnlySpan<float> biasDeltaSpan = biasDelta.Data;
			
			Span<float> weightsSpan = weights.Data;
			Span<float> biasSpan = bias.Data;
			
			int length = weights.Length;
			int i = 0;
			for (; i <= length - _simdWidth; i += _simdWidth) {
				Vector<float> vWeights = new (weightsSpan.Slice(i, _simdWidth));
				Vector<float> vWeightDelta = new (weightDeltaSpan.Slice(i, _simdWidth));
				(vWeights - vWeightDelta * scale).CopyTo(weightsSpan.Slice(i, _simdWidth));
			}

			for (; i < length; i++) {
				weightsSpan[i] -= weightDeltaSpan[i] * scale;
			}
			
			CpuMatrixTranpose(weights, weightsT);

			length = bias.Length;
			i = 0;
			for (; i <= length - _simdWidth; i += _simdWidth) {
				Vector<float> vBias = new (biasSpan.Slice(i, _simdWidth));
				Vector<float> vBiasDelta = new (biasDeltaSpan.Slice(i, _simdWidth));
				(vBias - vBiasDelta * scale).CopyTo(biasSpan.Slice(i, _simdWidth));
			}

			for (; i < length; i++) {
				biasDelta[i] -= biasDeltaSpan[i] * scale;
			}
		}
	}
	
	/**
	 * Gpu powered backend.
	 */
	public class GpuBackend : IBackend {

		private readonly Accelerator _accelerator;

		private readonly Action<
			Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int> _transposeKernel;
		private readonly Action<
			Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>> _maxVectorIndexKernel;

		private readonly Action<
			Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _inputKernel;

		private readonly Action<
			Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
			ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int> _linearKernel;
		private readonly Action<
			Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _activationKernel;
		private readonly Action<
			Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int> _outputKernel;

		private readonly Action<
			Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
			ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int> _backwardKernel;
		private readonly Action<
			Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
			ArrayView1D<float, Stride1D.Dense>> _backwardOutputKernel;
		private readonly Action<
			Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
			ArrayView1D<float, Stride1D.Dense>, int> _weightGradientKernel;
		private readonly Action<
			Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _biasGradientKernel;

		private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>
			_adjustKernel;
		
		public GpuBackend(Accelerator accelerator) {
			_accelerator = accelerator;
			
			Console.WriteLine($"Using Gpu Backend, on device: {accelerator.Name}");

			_transposeKernel = accelerator
				.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
					ArrayView1D<float, Stride1D.Dense>, int, int>(TransposeKernel);

			_maxVectorIndexKernel =
				accelerator
					.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
						ArrayView1D<int, Stride1D.Dense>>(MaxVectorIndex);

			_inputKernel = accelerator
				.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
					ArrayView1D<float, Stride1D.Dense>>(InputKernel);

			_linearKernel =
				accelerator
					.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
						ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
						ArrayView1D<float, Stride1D.Dense>, int>(LinearKernel);
			_activationKernel =
				accelerator
					.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
						ArrayView1D<float, Stride1D.Dense>>(ActivationKernel);
			_outputKernel =
				accelerator
					.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
						ArrayView1D<float, Stride1D.Dense>, int>(OutputKernel);

			_backwardKernel =
				accelerator
					.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
						ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
						ArrayView1D<float, Stride1D.Dense>, int>(BackwardKernel);
			_backwardOutputKernel =
				accelerator
					.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
						ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(BackwardOutputKernel);
			_weightGradientKernel =
				accelerator
					.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
						ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(
						WeightGradientKernel);
			_biasGradientKernel =
				accelerator
					.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
						ArrayView1D<float, Stride1D.Dense>>(BiasGradientKernel);
			
			_adjustKernel = 
				accelerator
					.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
						ArrayView1D<float, Stride1D.Dense>, float>(AdjustKernel);
		}

		public ITensor CreateTensor(params int[] shape) {
			return new GpuTensor(_accelerator, shape);
		}
		public ITensor CreateTensor(float[] data, params int[] shape) {
			return new GpuTensor(_accelerator, data, shape);
		}
		public ITensor Zero(params int[] shape) {
			return GpuTensor.Zero(_accelerator, shape);
		}
		public ITensor Random(Random random, float min, float max, params int[] shape) {
			return GpuTensor.Random(_accelerator, random, min, max, shape);
		}

		public void Set(ITensor a, float[] data) {
			((GpuTensor)a).Buffer.CopyFromCPU(data);
		}
		public float[] Read(ITensor a) {
			float[] r = new float[a.Length];
			_accelerator.Synchronize();
			((GpuTensor)a).Buffer.CopyToCPU(r);
			return r;
		}

		public void MatrixTranspose(ITensor a, ITensor output) {
			GpuTensor gInput = (GpuTensor)a;
			GpuTensor gOutput = (GpuTensor)output;
			
			int outputSize = gInput.Length;
			
			int rows = gInput.Shape[0];
			int cols = gInput.Shape[1];

			_transposeKernel(
				outputSize,
				gInput.Buffer.View,
				gOutput.Buffer.View,
				rows,
				cols
				);
		}

		public int MaxVectorIndex(ITensor a) {
			GpuTensor gA = (GpuTensor)a;

			using MemoryBuffer1D<int, Stride1D.Dense> result = _accelerator.Allocate1D<int>(1);

			_maxVectorIndexKernel(1, gA.Buffer.View, result.View);
			
			int[] cpuResult = new int[1];
			result.CopyToCPU(cpuResult);
			
			return cpuResult[0];
		}

		public float Loss(DataGuess<float> guess) {
			float[] output = Read(guess.Output);
			return -(float)Math.Log(output[guess.Data.Label]);
		}

		public void Input(ITensor input, ITensor activationOut) {
			GpuTensor gInput = (GpuTensor)input;
			GpuTensor gActivationOut = (GpuTensor)activationOut;
			
			int outputSize = activationOut.Shape[0];
			
			_inputKernel(
				outputSize,
				gInput.Buffer.View,
				gActivationOut.Buffer.View);
		}

		public void Linear(ITensor input, ITensor weights, ITensor bias, ITensor valuesOut) {
			GpuTensor gInput = (GpuTensor)input;
			GpuTensor gWeights = (GpuTensor)weights;
			GpuTensor gBias = (GpuTensor)bias;
			GpuTensor gValuesOut = (GpuTensor)valuesOut;

			int outputSize = valuesOut.Shape[0];
			int inputSize = input.Shape[0];

			_linearKernel(
				outputSize,
				gInput.Buffer.View,
				gWeights.Buffer.View,
				gBias.Buffer.View,
				gValuesOut.Buffer.View,
				inputSize
				);
		}
		public void Activation(ITensor values, ITensor activationOut) {
			GpuTensor gValues = (GpuTensor)values;
			GpuTensor gActivationOut = (GpuTensor)activationOut;

			int outputSize = activationOut.Shape[0];
			
			_activationKernel(
				outputSize,
				gValues.Buffer.View,
				gActivationOut.Buffer.View
				);
		}
		public void Output(ITensor values, ITensor activationOut) {
			GpuTensor gValues = (GpuTensor)values;
			GpuTensor gActivationOut = (GpuTensor)activationOut;

			int outputSize = activationOut.Shape[0];
			
			_outputKernel(
				outputSize,
				gValues.Buffer.View,
				gActivationOut.Buffer.View,
				outputSize
			);
		}

		public void Backward(ITensor values, ITensor delta, ITensor weightsT, ITensor deltaOut) {
			GpuTensor gValues = (GpuTensor)values;
			GpuTensor gDelta = (GpuTensor)delta;
			GpuTensor gWeightsT = (GpuTensor)weightsT;
			GpuTensor gDeltaOut = (GpuTensor)deltaOut;

			int outputSize = deltaOut.Shape[0];
			int deltaSize = delta.Shape[0];
			
			_backwardKernel(
				outputSize,
				gValues.Buffer.View,
				gDelta.Buffer.View,
				gWeightsT.Buffer.View,
				gDeltaOut.Buffer.View,
				deltaSize);
		}
		public void BackwardOutput(ITensor activations, ITensor expectedValues, ITensor deltaOut) {
			GpuTensor gActivations = (GpuTensor)activations;
			GpuTensor gExpectedValues = (GpuTensor)expectedValues;
			GpuTensor gDeltaOut = (GpuTensor)deltaOut;

			int outputSize = deltaOut.Shape[0];
			
			_backwardOutputKernel(
				outputSize,
				gActivations.Buffer.View,
				gExpectedValues.Buffer.View,
				gDeltaOut.Buffer.View);
		}
		public void BackPropagation(ITensor delta, ITensor activations, ITensor weightDelta, ITensor biasDelta) {
			GpuTensor gDelta = (GpuTensor)delta;
			GpuTensor gActivations = (GpuTensor)activations;
			GpuTensor gWeightsDelta = (GpuTensor)weightDelta;
			GpuTensor gBiasDelta = (GpuTensor)biasDelta;

			int columns = weightDelta.Shape[1];
			
			int outputSize = weightDelta.Length;
			_weightGradientKernel(
				outputSize,
				gDelta.Buffer.View,
				gActivations.Buffer.View,
				gWeightsDelta.Buffer.View,
				columns);

			outputSize = biasDelta.Length;
			_biasGradientKernel(
				outputSize,
				gDelta.Buffer.View,
				gBiasDelta.Buffer.View);
		}

		public void Adjust(ITensor weightDelta, ITensor biasDelta, float scale, ITensor weights, ITensor weightsT, ITensor bias) {
			GpuTensor gWeightDelta = (GpuTensor)weightDelta;
			GpuTensor gBiasDelta = (GpuTensor)biasDelta;
			
			GpuTensor gWeights = (GpuTensor)weights;
			GpuTensor gWeightsT = (GpuTensor)weightsT;
			GpuTensor gBias = (GpuTensor)bias;
			
			int outputSize = gWeights.Length;
			_adjustKernel(
				outputSize,
				gWeights.Buffer.View,
				gWeightDelta.Buffer.View,
				scale);

			int rows = gWeights.Shape[0];
			int cols = gWeights.Shape[1];
			
			_transposeKernel(
				outputSize,
				gWeights.Buffer.View,
				gWeightsT.Buffer.View,
				rows,
				cols
				);
			
			outputSize = gBias.Length;
			_adjustKernel(
				outputSize,
				gBias.Buffer.View,
				gBiasDelta.Buffer.View,
				scale);
		}
		
		private static void TransposeKernel(
			Index1D i,
			ArrayView1D<float, Stride1D.Dense> input,
			ArrayView1D<float, Stride1D.Dense> output,
			int rows,
			int cols
		) {
			int row = i / cols;
			int col = i % cols;
			
			output[col * rows + row] = input[row * cols + col];
		}

		// Temporary
		private static void MaxVectorIndex(
			Index1D index,
			ArrayView1D<float, Stride1D.Dense> input,
			ArrayView1D<int, Stride1D.Dense> output
		) {
			if (index != 0)
				return;

			int maxIndex = 0;
			float max = input[0];

			for (int i = 1; i < input.Length; i++) {
				if (input[i] > max) {
					max = input[i];
					maxIndex = i;
				}
			}

			output[0] = maxIndex;
		}
		
		private static void InputKernel(
			Index1D i,
			ArrayView1D<float, Stride1D.Dense> input,
			ArrayView1D<float, Stride1D.Dense> activationOut
		) {
			const float mean = 0.1307f;
			const float std = 0.3081f;
			
			activationOut[i] = (input[i] - mean) / std;
		}

		private static void LinearKernel(
			Index1D row,
			ArrayView1D<float, Stride1D.Dense> input,
			ArrayView1D<float, Stride1D.Dense> weights,
			ArrayView1D<float, Stride1D.Dense> bias,
			ArrayView1D<float, Stride1D.Dense> valuesOut,
			int columns
		) {
			float sum = bias[row];

			int offset = row * columns;

			for (int col = 0; col < columns; col++) {
				sum += weights[offset+ col] * input[col];
			}

			valuesOut[row] = sum;
		}
		private static void ActivationKernel(
			Index1D i,
			ArrayView1D<float, Stride1D.Dense> values,
			ArrayView1D<float, Stride1D.Dense> activationOut
		) {
			activationOut[i] = values[i] > 0 ? values[i] : 0;
		}
		private static void OutputKernel(
			Index1D index,
			ArrayView1D<float, Stride1D.Dense> values,
			ArrayView1D<float, Stride1D.Dense> activationOut,
			int length
		) {
			if (index != 0)
				return;

			float max = values[0];

			for (int i = 1; i < values.Length; i++)
			{
				if (values[i] > max)
					max = values[i];
			}

			float sum = 0f;

			for (int i = 0; i < values.Length; i++)
			{
				float e = XMath.Exp(values[i] - max);
				activationOut[i] = e;
				sum += e;
			}

			for (int i = 0; i < values.Length; i++)
			{
				activationOut[i] /= sum;
			}
		}
		
		private static void BackwardKernel(
			Index1D row,
			ArrayView1D<float, Stride1D.Dense> values,
			ArrayView1D<float, Stride1D.Dense> delta,
			ArrayView1D<float, Stride1D.Dense> weightsT,
			ArrayView1D<float, Stride1D.Dense> deltaOut,
			int deltaSize
		) {
			float sum = 0;
			
			int offset = row * deltaSize;

			for (int col = 0; col < deltaSize; col++) {
				sum += weightsT[offset + col] * delta[col];
			}
			
			deltaOut[row] = values[row] > 0 ? sum : 0;
		}
		private static void BackwardOutputKernel(
			Index1D i,
			ArrayView1D<float, Stride1D.Dense> activations,
			ArrayView1D<float, Stride1D.Dense> expectedValues,
			ArrayView1D<float, Stride1D.Dense> deltaOut
		) {
			deltaOut[i] = activations[i] - expectedValues[i];
		}
		
		private static void WeightGradientKernel(
			Index1D i,
			ArrayView1D<float, Stride1D.Dense> delta,
			ArrayView1D<float, Stride1D.Dense> activations,
			ArrayView1D<float, Stride1D.Dense> weightDelta,
			int columns
		) {
			int row = i / columns;
			int col = i % columns;
			
			weightDelta[i] += delta[row] * activations[col];
		}
		private static void BiasGradientKernel(
			Index1D i,
			ArrayView1D<float, Stride1D.Dense> delta,
			ArrayView1D<float, Stride1D.Dense> biasDelta
		) {
			biasDelta[i] += delta[i];
		}
		
		private static void AdjustKernel(
			Index1D i,
			ArrayView1D<float, Stride1D.Dense> parameter,
			ArrayView1D<float, Stride1D.Dense> gradient,
			float scale
		) {
			parameter[i] -= gradient[i] * scale;
		}
	}
}