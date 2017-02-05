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

using System.Linq;
using Xunit;


namespace MathML.Test
{
    public class Tests
    {
        [Fact]
        public void Sigmoid()
        {
            Assert.InRange(Classifier.Sigmoid(0), 0.49, 0.51);
            Assert.InRange(Classifier.Sigmoid(2), 0.8, 1.0);
            Assert.InRange(Classifier.Sigmoid(-2), 0, 0.2);
        }

        [Fact]
        public void DotProduct()
        {
            Assert.Equal(Classifier.DotProduct(new double[] { 1, 2 }, new double[] { 2, 3 }), 8.0);
        }

        [Fact]
        public void ScalarProduct()
        {
            Assert.Equal(Classifier.ScalarProduct(new double[] { 3, 4 }, 5), new double[] { 15, 20 });
        }

        [Fact]
        public void VectorAdd()
        {
            Assert.Equal(
                Classifier.VectorAdd(new double[] { 1, 1 }, new double[] { 3, 5 }), 
                new double[] { 4, 6 }
            );

            var xs = new double[][]
            {
                new double[] { 1, 2 },
                new double[] { 1, 3 },
                new double[] { 1, 4 },
                new double[] { 1, 5 }
            };

            Assert.Equal(Classifier.VectorAdd(xs), new double[] { 4, 14 });
        }

        [Fact]
        public void Hypothesis()
        {
            var p = Classifier.Probability(
                x: new double[] { 1, 1 }, 
                theta: new double[] { -1, -1 }
            );
            Assert.InRange(p, 0, 0.5);

            p = Classifier.Probability(
                x: new double[] { 1, 1 }, 
                theta: new double[] { 1, 1 }
            );
            Assert.InRange(p, 0.5, 1.0);
        }

        [Fact]
        public void Probability()
        {
            var x = new double[] { 1, 1 };
            var theta = new double[] { -1, -1 };
            var p1 = Classifier.Hypothesis(x, theta);
            var p2 = Classifier.Probability(x, theta);
            Assert.Equal(p1, p2);
        }

        [Fact]
        public void FastGradient()
        {
            var xs = new double[][]
            {
                new double[] { 1, 2 },
                new double[] { 1, 3 },
                new double[] { 1, 4 },
                new double[] { 1, 5 }
            };
            var xs2 = new double[4, 2] 
            {
                { 1, 2 },
                { 1, 3 },
                { 1, 4 },
                { 1, 5}
            };

            var ys = new double[] { 0, 0, 0, 1 };
            var theta = new double[] { 1, 1 };
            var thetaGradient = new double[2];

            var grad1 = Classifier.Gradient(xs, ys, theta);
            Classifier.FastGradient(xs2, ys, theta, thetaGradient);

            Assert.InRange(grad1[0], thetaGradient[0]-0.001, thetaGradient[0]+0.001);
            Assert.InRange(grad1[1], thetaGradient[1]-0.001, thetaGradient[1]+0.001);
        }

        [Fact]
        public void FastGradientLambda()
        {
            var xs = new double[][]
            {
                new double[] { 1, 2, 3 },
                new double[] { 1, 3, 4 },
                new double[] { 1, 4, 5 },
                new double[] { 1, 5, 6 }
            };
            var xs2 = new double[4, 3] 
            {
                { 1, 2, 3 },
                { 1, 3, 4 },
                { 1, 4, 5 },
                { 1, 5, 6 }
            };

            var ys = new double[] { 0, 0, 0, 1 };
            var theta = new double[] { 1, 1, 1 };
            var thetaGradient = new double[3];

            var grad1 = Classifier.Gradient(xs, ys, theta, 0.1);
            Classifier.FastGradient(xs2, ys, theta, 0.1, 1.0, thetaGradient);

            Assert.InRange(grad1[0], thetaGradient[0]-0.001, thetaGradient[0]+0.001);
            Assert.InRange(grad1[1], thetaGradient[1]-0.001, thetaGradient[1]+0.001);
            Assert.InRange(grad1[2], thetaGradient[2]-0.001, thetaGradient[2]+0.001);
        }

        [Fact]
        public void FastTrain()
        {
            var xs1 = new double[][]
            {
                new double[] { 1, 2 },
                new double[] { 1, 3 },
                new double[] { 1, 4 },
                new double[] { 1, 5 }
            };

            var xs2 = new double[4,2]
            {
                { 1, 2 },
                { 1, 3 },
                { 1, 4 },
                { 1, 5 }
            };
            var ys = new double[] { 0, 0, 0, 1 };
            var theta = new double[] { 1, 1 };

            theta = Classifier.FastTrain(
                xs2, ys,
                theta,
                lambda: 0.0,
                weightClass0: 1.0,
                alpha: 0.5,
                iterations: 100000,
                halfLives: 3
            );

            var ps = xs1.Select(x => Classifier.Probability(x, theta)).ToList();

            Assert.InRange(ps[0], 0, 0.1);
            Assert.InRange(ps[1], 0, 0.1);
            Assert.InRange(ps[2], 0, 0.1);
            Assert.InRange(ps[3], 0.9, 1.0);
        }

        [Fact]
        public void Train()
        {
            var xs = new double[][]
            {
                new double[] { 1, 2 },
                new double[] { 1, 3 },
                new double[] { 1, 4 },
                new double[] { 1, 5 }
            };
            var ys = new double[] { 0, 0, 0, 1 };
            var theta = new double[] { 1, 1 };

            var limit = 100000;
            theta = Classifier.Train(
                xs, ys,
                theta,
                0.001,
                0.5,
                ref limit
            );

            var ps = xs.Select(x => Classifier.Probability(x, theta)).ToList();

            Assert.InRange(ps[0], 0, 0.1);
            Assert.InRange(ps[1], 0, 0.1);
            Assert.InRange(ps[2], 0, 0.1);
            Assert.InRange(ps[3], 0.9, 1.0);
        }

        [Fact]
        public void TrainLambda()
        {
            var xs = new double[][]
            {
                new double[] { 1, 2 },
                new double[] { 1, 3 },
                new double[] { 1, 4 },
                new double[] { 1, 5 }
            };
            var ys = new double[] { 0, 0, 0, 1 };
            var theta = new double[] { 1, 1 };
            var lambda = 0.0001;

            var limit = 100000;
            theta = Classifier.Train(
                xs, ys,
                theta,
                lambda,
                0.001,
                0.5,
                ref limit
            );

            var ps = xs.Select(x => Classifier.Probability(x, theta)).ToList();

            Assert.InRange(ps[0], 0, 0.1);
            Assert.InRange(ps[1], 0, 0.1);
            Assert.InRange(ps[2], 0, 0.1);
            Assert.InRange(ps[3], 0.9, 1.0);
        }

    }
}
