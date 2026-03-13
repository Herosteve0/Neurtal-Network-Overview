using System.Text;

namespace NeuralNetworkSystem {
    public class Vector {
        public Vector(int length) {
            Length = length;
            Data = new float[length];
            for (int i = 0; i < length; i++) Data[i] = 0;
        }
        public Vector(float[] values) {
            Length = values.Length;
            Data = new float[Length];
            for (int i = 0; i < Length; i++) Data[i] = values[i];
        }

        public readonly int Length;
        public float[] Data;

        public float this[int index] {
            get => Data[index];
            set => Data[index] = value;
        }

        public static Vector SingleValue(int length, int index, float value = 1f) {
            Vector R = new Vector(length);
            for (int i = 0; i < length; i++) R[i] = (i == index) ? value : 0;
            return R;
        }

        public static Vector Random(int length, float min = 0f, float max = 1f) {
            Vector R = new Vector(length);
            
            float scale = max - min;
            for (int i = 0; i < length; i++) R[i] = min + ProgramManager.random.NextSingle() * scale;
            
            return R;
        }

        public override string ToString() {
            StringBuilder r = new StringBuilder().AppendLine();

            r.Append("(");
            for (int i = 0; i < Length; i++) {
                r.Append(Data[i].ToString());
                if (i < Length - 1) r.Append(", ");
            }
            r.Append(")");
            return r.ToString();
        }

        public int MaxIndex() {
            int r = 0;
            for (int i = 1; i < Data.Length; i++) {
                if (Data[r] < Data[i]) r = i;
            }
            return r;
        }
    }

    public class Matrix {
        public Matrix(int rows, int columns) {
            Rows = rows;
            Columns = columns;
            Data = new float[rows * columns];
        }

        public int Rows { get; }
        public int Columns { get; }

        public float[] Data;

        public virtual float this[int row, int column] {
            get => Data[row * Columns + column];
            set => Data[row * Columns + column] = value;
        }

        public static Matrix Random(int rows, int cols, float min = 0f, float max = 1f) {
            Matrix R = new Matrix(rows, cols);

            float scale = max - min;
            
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    R[i, j] = min + ProgramManager.random.NextSingle() * scale;
                }
            }
            return R;
        }

        public override string ToString() {
            StringBuilder r = new StringBuilder().AppendLine();

            int[] max = new int[Columns];
            for (int j = 0; j < Columns; j++) {
                max[j] = this[0, j].ToString().Length;
                for (int i = 1; i < Rows; i++) {
                    max[j] = Math.Max(this[i, j].ToString().Length, max[j]);
                }
            }

            for (int i = 0; i < Rows; i++) {
                for (int j = 0; j < Columns; j++) {
                    r.Append(this[i, j].ToString().PadRight(max[j]));
                    if (j < Columns - 1) r.Append("|");
                }
                if (i < Rows - 1) r.AppendLine();
            }

            return r.ToString();
        }

        public void Transpose(Matrix Out) {
            if (Rows != Out.Columns) throw new Exception("Tried to output transpose result to Matrix with unequal length (Rows - Columns)!");
            if (Columns != Out.Rows) throw new Exception("Tried to output transpose result to Matrix with unequal length (Columns - Rows)!");

            for (int i = 0; i < Rows; i++) {
                for (int j = 0; j < Columns; j++) {
                    Out[j, i] = this[i, j];
                }
            }
        }
    }
}