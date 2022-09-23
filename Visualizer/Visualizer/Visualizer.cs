using SkiaSharp;
using System.Drawing;
using System.Numerics;

namespace Visualizer;

public class Visualizer
{
    private const int CanvasHeight = 1000;
    private const int CanvasWidth = 2000;
    private const int CanvasHeightWithPadding = CanvasHeight + 2 * CanvasPadding;
    private const int CanvasWidthWithPadding = CanvasWidth + 2 * CanvasPadding;
    private const int CanvasPadding = 50;
    private const int RectRadius = 7;
    private const int GridRadius = 3;
    private const int StrokeWidth = 2;
    private const int GraphWidthThick = 4;
    private const int GraphWidthRegular = 2;
    private const int GraphWidthThin = 1;
    private const int FontSize = 70;
    private static readonly Vector2 OriginalPoint = new Vector2(CanvasPadding, CanvasPadding);
    private static readonly SKColor White = new(0xFF, 0xFF, 0xFF);
    private static readonly SKColor Black = new(0x20, 0x20, 0x20);
    private static readonly SKColor DarkGray = new(0x40, 0x40, 0x40);
    private static readonly SKColor Gray = new(0xB0, 0xB0, 0xB0);
    private static readonly SKColor LightGray = new(0xF0, 0xF0, 0xF0);
    private static readonly SKColor Blue = new(0x3D, 0x79, 0xF7);
    private static readonly SKColor Red = new(0xF2, 0x49, 0x49);
    private readonly Input _input;
    private readonly float _unitLength;

    public Visualizer(Input input)
    {
        _input = input;
        _unitLength = (float)CanvasHeight / (input.N - 1);
    }

    public SKBitmap Visualize(IReadOnlyList<Solution> solutions, int index)
    {
        var imageInfo = new SKImageInfo(CanvasWidthWithPadding, CanvasHeightWithPadding, SKColorType.Rgba8888, SKAlphaType.Opaque);
        var bitmap = new SKBitmap(imageInfo);
        using var canvas = new SKCanvas(bitmap);
        var solution = solutions[index];

        canvas.Clear(White);
        DrawGrid(canvas);
        DrawRectangles(canvas, solution);
        DrawScore(canvas, solution);
        DrawGraph(canvas, solutions, index);

        return bitmap;
    }

    private void DrawGrid(SKCanvas canvas)
    {
        using var paint = new SKPaint
        {
            Style = SKPaintStyle.Fill,
            Color = Gray,
            IsAntialias = true
        };

        for (int x = 0; x < _input.N; x++)
        {
            for (int y = 0; y < _input.N; y++)
            {
                var v = ToWindowCoordinate(new Vector2(x, y));
                canvas.DrawCircle(v, GridRadius, paint);
            }
        }
    }

    private void DrawRectangles(SKCanvas canvas, Solution solution)
    {
        using var linePaint = new SKPaint
        {
            Style = SKPaintStyle.Stroke,
            StrokeWidth = StrokeWidth,
            Color = DarkGray,
            IsAntialias = true
        };

        foreach (var rectangle in solution.Rectangles)
        {
            for (int i = 0; i < 4; i++)
            {
                var v0 = ToWindowCoordinate(rectangle[i]);
                var v1 = ToWindowCoordinate(rectangle[(i + 1) % rectangle.Length]);
                canvas.DrawLine(v0, v1, linePaint);
            }
        }

        using var pointPaint = new SKPaint
        {
            Style = SKPaintStyle.Fill,
            Color = DarkGray,
            IsAntialias = true
        };

        foreach (var point in _input.Points)
        {
            var v0 = ToWindowCoordinate(point);
            canvas.DrawCircle(v0, RectRadius, pointPaint);
        }

        var scoreMax = _input.CalculatePointScore(new Vector2(0, (float)(_input.N - 1) / 2));

        foreach (var rectangle in solution.Rectangles)
        {
            var v0 = ToWindowCoordinate(rectangle[0]);
            var score = _input.CalculatePointScore(rectangle[0]);
            pointPaint.Color = Interpolate(Blue, Red, score / scoreMax);
            canvas.DrawCircle(v0, RectRadius, pointPaint);
        }
    }

    private void DrawScore(SKCanvas canvas, Solution solution)
    {
        using var paint = new SKPaint
        {
            Color = Black,
            TextSize = FontSize,
            IsAntialias = true
        };

        var point = new SKPoint(1100, CanvasPadding + 200);
        canvas.DrawText($"Score: {solution.Score}", point, paint);
    }

    private void DrawGraph(SKCanvas canvas, IReadOnlyList<Solution> solutions, int index)
    {
        const int left = 1100;
        const int right = CanvasWidthWithPadding - CanvasPadding;
        const int top = 250 + CanvasPadding;
        const int bottom = 850 + CanvasPadding;
        const float maxScore = 3e6f;

        float GetX(float x) => (right - left) * x + left;
        float GetY(float y) => (top - bottom) * y + bottom;

        using var paint = new SKPaint
        {
            Style = SKPaintStyle.Stroke,
            StrokeWidth = GraphWidthRegular,
            Color = Gray,
            IsAntialias = true
        };


        for (int i = 1; i <= 2; i++)
        {
            canvas.DrawLine(GetX(0), GetY(i / 3.0f), GetX(1), GetY(i / 3.0f), paint);
        }

        paint.Color = LightGray;
        paint.StrokeWidth = GraphWidthThin;

        for (int i = 0; i < 30; i++)
        {
            if (i % 10 == 0)
            {
                continue;
            }

            canvas.DrawLine(GetX(0), GetY(i / 30.0f), GetX(1), GetY(i / 30.0f), paint);
        }

        paint.Color = Blue;
        paint.StrokeWidth = GraphWidthThick;
        paint.StrokeJoin = SKStrokeJoin.Round;
        var path = new SKPath();
        path.MoveTo(GetX(0), GetY(solutions[0].Score / maxScore));

        for (int i = 1; i < solutions.Count; i++)
        {
            var x = GetX((float)i / (solutions.Count - 1));
            var y = GetY(solutions[i].Score / maxScore);
            path.LineTo(x, y);
        }

        canvas.DrawPath(path, paint);

        paint.Color = Black;
        canvas.DrawRect(left, top, right - left, bottom - top, paint);

        paint.Color = Red;
        var currentX = GetX((float)index / (solutions.Count - 1));
        canvas.DrawLine(currentX, GetY(0), currentX, GetY(1), paint);
    }

    private SKPoint ToWindowCoordinate(Vector2 v)
    {
        var vec = v * _unitLength + OriginalPoint;
        return new SKPoint(vec.X, CanvasHeightWithPadding - vec.Y);
    }

    private SKColor Interpolate(SKColor c0, SKColor c1, double x)
    {
        x = Math.Max(Math.Min(x, 1), 0);
        var r = (byte)Math.Round(c0.Red * (1 - x) + c1.Red * x);
        var g = (byte)Math.Round(c0.Green * (1 - x) + c1.Green * x);
        var b = (byte)Math.Round(c0.Blue * (1 - x) + c1.Blue * x);
        return new SKColor(r, g, b);
    }
}