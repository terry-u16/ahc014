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
    private const int GraphWidth = 1;
    private const int FontSize = 70;
    private static readonly Vector2 OriginalPoint = new Vector2(CanvasPadding, CanvasPadding);
    private static readonly SKColor White = new(0xFF, 0xFF, 0xFF);
    private static readonly SKColor Black = new(0x20, 0x20, 0x20);
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
            Color = Black,
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
            Color = Black,
            IsAntialias = true
        };

        foreach (var point in _input.Points)
        {
            var v0 = ToWindowCoordinate(point);
            canvas.DrawCircle(v0, RectRadius, pointPaint);
        }

        foreach (var rectangle in solution.Rectangles)
        {
            var v0 = ToWindowCoordinate(rectangle[0]);
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

        float GetX(float x) => (right - left) * x + left;
        float GetY(float y) => (top - bottom) * y + bottom;

        using var paint = new SKPaint
        {
            Style = SKPaintStyle.Stroke,
            StrokeWidth = StrokeWidth,
            Color = Gray,
            IsAntialias = true
        };

        for (int i = 1; i <= 2; i++)
        {
            canvas.DrawLine(GetX(0), GetY(i / 3.0f), GetX(1), GetY(i / 3.0f), paint);
        }

        paint.Color = LightGray;

        for (int i = 0; i < 30; i++)
        {
            if (i % 10 == 0)
            {
                continue;
            }

            canvas.DrawLine(GetX(0), GetY(i / 30.0f), GetX(1), GetY(i / 30.0f), paint);
        }

        paint.Color = Blue;

        for (int i = 0; i + 1 < solutions.Count; i++)
        {
            var x0 = GetX((float)i / (solutions.Count - 1));
            var x1 = GetX((float)(i + 1) / (solutions.Count - 1));
            var y0 = GetY((float)solutions[i].Score / 3_000_000);
            var y1 = GetY((float)solutions[i + 1].Score / 3_000_000);
            canvas.DrawLine(x0, y0, x1, y1, paint);
        }

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
}