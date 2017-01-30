//   Copyright 2017 Matt Howlett
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

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

        public static double Cost(double[][] xs, double[] ys, double[] theta, double lambda)
            => Cost(xs, ys, theta) 
                + theta.Skip(1).Sum(t => t*t) * lambda / (2 * xs.Length);

        public static double[] ScalarProduct(double[] xs, double y)
            => xs.Select(x => x * y).ToArray();

        public static double[] Gradient(double[] x, double y, double[] theta)
            => ScalarProduct(x, (Hypothesis(x, theta) - y));

        public static double[] VectorAdd(double[][] xs1)
            => xs1.Aggregate(
                 new double[xs1[0].Length], 
                 (accum, xs) => accum.Zip(xs, (a, x) => a + x).ToArray()
               );

        public static double[] Gradient(double[][] xs, double[] ys, double[] theta)
            => ScalarProduct(
                 VectorAdd(xs.Zip(ys, (x, y) => Gradient(x, y, theta)).ToArray()), 
                 1.0 / xs.Length
               );

        public static double[] Gradient(double[][] xs, double[] ys, double[] theta, double lambda)
            => ScalarProduct(
                 VectorAdd(
                   xs.Zip(
                     ys, 
                     (fvs, y) => VectorAdd(
                                   Gradient(fvs, y, theta), 
                                   ScalarProduct(new[] {0.0}.Concat(theta.Skip(1)).ToArray(), lambda)
                                 )
                   ).ToArray()
                 ), 
                 1.0 / xs.Length
               );

        public static double[] VectorAdd(double[] xs, double[] ys)
            => xs.Zip(ys, (x, y) => x + y).ToArray();

        public static double[] Train(double[][] xs, double[] ys, double[] theta, double minAlpha, double maxAlpha, ref int iterations)
        {
            var alpha = maxAlpha;
            double prevCost = Classifier.Cost(xs, ys, theta);
            while (iterations-- > 0)
            {
                theta = VectorAdd(theta, ScalarProduct(Gradient(xs, ys, theta), -alpha));
                var cost = Classifier.Cost(xs, ys, theta);
                if (cost > prevCost)
                {
                    alpha /= 2;
                    if (alpha < minAlpha)
                    {
                        break;
                    }
                }
                prevCost = cost;
            }
            return theta;
        }

        public static double[] Train(double[][] xs, double[] ys, double[] theta, double lambda, double minAlpha, double maxAlpha, ref int iterations)
        {
            var alpha = maxAlpha;
            double prevCost = Classifier.Cost(xs, ys, theta, lambda);
            while (iterations-- > 0)
            {
                theta = VectorAdd(theta, ScalarProduct(Gradient(xs, ys, theta, lambda), -alpha));
                var cost = Classifier.Cost(xs, ys, theta, lambda);
                if (cost > prevCost)
                {
                    alpha /= 2;
                    if (alpha < minAlpha)
                    {
                        break;
                    }
                }
                prevCost = cost;
            }
            return theta;
        }

    }
}
