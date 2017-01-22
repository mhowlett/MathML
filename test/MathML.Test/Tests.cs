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
                0.02,
                0.0001,
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
