using System.Numerics;

namespace Visualizer;

public class Input
{
    public int N { get; }
    public int M { get; }
    public Vector2[] Points { get; }
    public int Seed { get; }
    private readonly int _totalWeight;

    public Input(int n, int m, Vector2[] points, int seed)
    {
        N = n;
        M = m;
        Points = points;
        Seed = seed;

        for (int x = 0; x < n; x++)
        {
            for (int y = 0; y < n; y++)
            {
                _totalWeight += CalculateWeight(x, y);
            }
        }
    }

    public static async Task<Input> ReadFromFileAsync(TextReader reader, int seed)
    {
        var nmString = await reader.ReadLineAsync() ?? throw new InvalidOperationException();
        var nm = nmString.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
        var n = nm[0];
        var m = nm[1];
        var points = new Vector2[m];

        for (int i = 0; i < points.Length; i++)
        {
            var xyString = await reader.ReadLineAsync() ?? throw new InvalidOperationException();
            var xy = xyString.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();

            points[i] = new Vector2(xy[0], xy[1]);
        }

        return new Input(n, m, points, seed);
    }

    public double CalculatePointScore(Vector2 v) => 
        1e6 * CalculateWeight((int)v.X, (int)v.Y) / _totalWeight * N * N / M;

    private int CalculateWeight(int x, int y)
    {
        var c = (N - 1) / 2;
        var dx = x - c;
        var dy = y - c;
        return dx * dx + dy * dy + 1;
    }
}