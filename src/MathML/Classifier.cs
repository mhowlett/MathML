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

        unsafe public static void FastGradient(double[,] xs, double[] ys, double[] theta, double[] thetaGradientOut)
        {
            fixed (double* theta_fixed = theta)
            fixed (double* ys_fixed = ys)
            fixed (double* xs_fixed = xs)
            fixed (double* result_fixed = thetaGradientOut)
            {
                double* ysPtr = ys_fixed;
                double* xsPtr = xs_fixed;
                double* resultPtr = result_fixed;
                double* thetaPtr = theta_fixed;

                for (int j=0; j<theta.Length; ++j)
                {
                    *(resultPtr++) = 0.0;
                }

                for (int i=0; i<ys.Length; ++i)
                {
                    double* xsPtr_save = xsPtr;
                    double h = 0;
                    thetaPtr = theta_fixed;
                    for (int j=0; j<theta.Length; ++j)
                    {
                        h += *(thetaPtr++) * *(xsPtr++);
                    }
                    double delta = Sigmoid(h) - *(ysPtr++);

                    resultPtr = result_fixed;
                    for (int j=0; j<theta.Length; ++j)
                    {
                        *(resultPtr++) += delta * *(xsPtr_save++);
                    }
                }

                resultPtr = result_fixed;
                double trainingSetSize = ys.Length;
                for (int j=0; j<theta.Length; ++j)
                {
                    *(resultPtr++) /= trainingSetSize;
                }
            }
        }

        unsafe public static void FastGradient(double[,] xs, double[] ys, double[] theta, double lambda, double weightClass0, double[] thetaGradientOut)
        {
            int thetaLen = theta.Length;

            fixed (double* theta_fixed = theta)
            fixed (double* ys_fixed = ys)
            fixed (double* xs_fixed = xs)
            fixed (double* thetaGradient_fixed = thetaGradientOut)
            {
                double* ysPtr = ys_fixed;
                double* xsPtr = xs_fixed;
                double* resultPtr = thetaGradient_fixed;
                double* thetaPtr = theta_fixed;

                for (int j=0; j<thetaLen; ++j)
                {
                    *(resultPtr++) = 0.0;
                }
                
                double weightTotal = 0.0;
                for (int i=0; i<ys.Length; ++i)
                {
                    double weight = ys[i] > 0.5 ? 1.0 : weightClass0;
                    weightTotal += weight;
                    
                    double* xsPtr_save = xsPtr;
                    double h = 0;
                    thetaPtr = theta_fixed;
                    for (int j=0; j<thetaLen; ++j)
                    {
                        h += *(thetaPtr++) * *(xsPtr++);
                    }
                    double delta = Sigmoid(h) - *(ysPtr++);

                    resultPtr = thetaGradient_fixed;
                    for (int j=0; j<thetaLen; ++j)
                    {
                        *(resultPtr++) += weight * delta * *(xsPtr_save++);
                    }
                }

                double trainingSetSize = ys.Length;

                resultPtr = thetaGradient_fixed;
                *(resultPtr++) /= weightTotal;
                thetaPtr = theta_fixed + 1;
                for (int j=1; j<thetaLen; ++j)
                {
                    *resultPtr += lambda * *(thetaPtr++);
                    *(resultPtr++) /= weightTotal;
                }
            }
        }

        public static double[] Gradient(double[][] xs, double[] ys, double[] theta, double lambda)
            => ScalarProduct(
                 VectorAdd(
                   xs.Zip(
                     ys, 
                     (fvs, y) => VectorAdd(
                                   Gradient(fvs, y, theta), 
                                   ScalarProduct(new[] {0.0}.Concat(theta.Skip(1)).ToArray(), lambda/xs.Length)
                                 )
                   ).ToArray()
                 ), 
                 1.0 / xs.Length
               );

        public static double[] VectorAdd(double[] xs, double[] ys)
            => xs.Zip(ys, (x, y) => x + y).ToArray();

        unsafe public static void VectorAddFast(double[] xs, double[] ys, double[] resultOut)
        {
            fixed (double* xs_fixed = xs)
            fixed (double* ys_fixed = ys)
            fixed (double* result_fixed = resultOut)
            {
                double* xsPtr = xs_fixed;
                double* ysPtr = ys_fixed;
                double* resultPtr = result_fixed;

                for (int i=0; i<xs.Length; ++i)
                {
                    *(resultPtr++) = *(xsPtr++) + *(ysPtr++);
                }
            }
        }

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
            double prevCost = Cost(xs, ys, theta, lambda);
            while (iterations-- > 0)
            {
                theta = VectorAdd(theta, ScalarProduct(Gradient(xs, ys, theta, lambda), -alpha));
                var cost = Cost(xs, ys, theta, lambda);
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

        unsafe public static double[] FastTrain(
            double[,] xs, 
            double[] ys, 
            double[] theta, 
            double lambda, 
            double weightClass0,
            double alpha, 
            int iterations,
            int halfLives)
        {
            double[] thetaGradientOut = new double[theta.Length];
            int thetaLen = theta.Length;

            double alphaHalfLife = iterations / halfLives;
            double alphaDecay = Math.Pow(0.5, 1.0/alphaHalfLife) - 1;

            fixed (double* thetaGradient_fixed = thetaGradientOut)
            fixed (double* theta_fixed = theta)
            {
                while (iterations-- > 0)
                {
                    FastGradient(xs, ys, theta, lambda, weightClass0, thetaGradientOut);

                    double* thetaPtr = theta_fixed;
                    double* thetaGradientPtr = thetaGradient_fixed;

                    for (int i=0; i<thetaLen; ++i)
                    {
                        *thetaPtr = *thetaPtr - alpha * *(thetaGradientPtr++);
                        thetaPtr += 1;
                    }

                    alpha *= (1+alphaDecay);
                }
            }

            return theta;
        }
    }
}
