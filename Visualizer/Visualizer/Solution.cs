using System.Numerics;

namespace Visualizer;

public record Solution(Vector2[][] Rectangles, int Score)
{
    public static async Task<Solution?> ReadFromFileAsync(TextReader reader, Input input)
    {
        var kString = await reader.ReadLineAsync();

        if (kString is null || !int.TryParse(kString, out var k))
        {
            return null;
        }

        var rectangles = new Vector2[k][];

        for (int i = 0; i < rectangles.Length; i++)
        {
            var rectangle = new Vector2[4];
            var pointsString = await reader.ReadLineAsync() ?? throw new InvalidOperationException();
            var points = pointsString.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();

            for (int j = 0; j < rectangle.Length; j++)
            {
                var x = points[2 * j];
                var y = points[2 * j + 1];
                rectangle[j] = new Vector2(x, y);
            }

            rectangles[i] = rectangle;
        }

        return new Solution(rectangles, CalculateScore(input, rectangles));
    }

    private static int CalculateScore(Input input, Vector2[][] rectangles)
    {
        var score = 0.0;

        foreach (var p in input.Points)
        {
            score += input.CalculatePointScore(p);
        }

        foreach (var rectangle in rectangles)
        {
            score += input.CalculatePointScore(rectangle[0]);
        }

        return (int)Math.Round(score);
    }
}