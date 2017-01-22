using System;
using System.Linq;

namespace MathML
{
    public static class Classifier
    {
        public static double Sigmoid(double v)
            => 1.0 / (1 + Math.Exp(-v));

        public static double DotProduct(double[] xs, double[] ys)
            => xs.Zip(ys, (x, y) => x * y).Sum();

        public static double Hypothesis(double[] x, double[] theta)
            => Sigmoid(DotProduct(x, theta));

        public static double Probability(double[] x, double[] theta)
            => Hypothesis(x, theta);

        public static double Cost(double[] x, double y, double[] theta)
            => -y * Math.Log(Hypothesis(x, theta)) - (1 - y) * Math.Log(1 - Hypothesis(x, theta));

        public static double Cost(double[][] xs, double[] ys, double[] theta)
            => xs.Zip(ys, (x, y) => Cost(x, y, theta)).Sum() / xs.Length;

        public static double[] ScalarProduct(double[] xs, double y)
            => xs.Select(x => x * y).ToArray();

        public static double[] Gradient(double[] x, double y, double[] theta)
            => ScalarProduct(x, (Hypothesis(x, theta) - y));

        public static double[] VectorAdd(double[][] xs1)
            => xs1.Aggregate(new double[xs1[0].Length], (accum, xs) => accum.Zip(xs, (a, x) => a + x).ToArray());

        public static double[] Gradient(double[][] xs, double[] ys, double[] theta)
            => VectorAdd(xs.Zip(ys, (x, y) => Gradient(x, y, theta)).ToArray());

        public static double[] VectorAdd(double[] xs, double[] ys)
            => xs.Zip(ys, (x, y) => x + y).ToArray();

        /// <summary>
        ///     
        /// </summary>
        /// <param name="xs"></param>
        /// <param name="ys"></param>
        /// <param name="theta"></param>
        /// <param name="alpha"></param>
        /// <param name="delta"></param>
        /// <param name="limit"></param>
        /// <returns></returns>
        public static double[] Train(double[][] xs, double[] ys, double[] theta, double alpha, double delta, ref int limit)
        {
            double prevCost = Classifier.Cost(xs, ys, theta);
            while (--limit > 0)
            {
                theta = VectorAdd(theta, ScalarProduct(Gradient(xs, ys, theta), -alpha));
                var cost = Classifier.Cost(xs, ys, theta);
                // Console.WriteLine($"cost: {cost}");
                if (Math.Abs(cost - prevCost) < delta)
                {
                    break;
                }
                prevCost = cost;
            }
            return theta;
        }
    }
}
