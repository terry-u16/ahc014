using Cysharp.Diagnostics;
using Visualizer;


var app = ConsoleApp.Create(args);
app.AddRootCommand(async (
    [Option("i")] string inputPath,
    [Option("s")] string solutionPath,
    [Option("o")] string outputPath,
    ConsoleAppContext context) =>
{
    var tempDirectory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
    Directory.CreateDirectory(tempDirectory);

    try
    {
        if (File.Exists(outputPath))
        {
            File.Delete(outputPath);
        }

        var input = await ReadInputAsync(inputPath);
        using var solutionReader = new StreamReader(solutionPath);
        var solutions = new List<Solution>();

        while (await Solution.ReadFromFileAsync(solutionReader, input) is Solution solution)
        {
            solutions.Add(solution);
        }

        var visualizer = new Visualizer.Visualizer(input);

        await Parallel.ForEachAsync(Enumerable.Range(0, solutions.Count), context.CancellationToken, async (i, ct) =>
        {
            using var bitmap = visualizer.Visualize(solutions, i);
            var data = bitmap.Encode(SkiaSharp.SKEncodedImageFormat.Png, 100);
            var path = Path.Combine(tempDirectory, $"{i:000000}.png");
            await File.WriteAllBytesAsync(path, data.ToArray(), ct);
        });

        await ConvertToMovie(outputPath, tempDirectory, context.CancellationToken);
    }
    finally
    {
        Directory.Delete(tempDirectory, true);
    }
});

app.Run();


static async Task<Input> ReadInputAsync(string inputPath)
{
    using var reader = new StreamReader(inputPath);
    var input = await Input.ReadFromFileAsync(reader);
    return input;
}

async Task ConvertToMovie(string s, string imageDirectory, CancellationToken ct = default)
{
    var imageTemplate = Path.Join(imageDirectory, "%06d.png");
    var ffmpegCommand = $"ffmpeg -r 30 -i {imageTemplate} -vcodec libx264 -pix_fmt yuv420p -r 30 {s}";

    var (_, stdOut, _) = ProcessX.GetDualAsyncEnumerable(ffmpegCommand);
    await foreach (var item in stdOut.WithCancellation(ct))
    {
        Console.WriteLine(item);
    }
}