// See https://aka.ms/new-console-template for more information
using Cysharp.Diagnostics;

var lockObject = new object();
var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, _) =>
{
    cts.Cancel();
};

await Run("cargo build --release --bin ahc014-a", Enumerable.Empty<string>(), cts.Token);
File.Move(@"..\target\release\ahc014-a.exe", @".\ahc014-a.exe");

var resultFilePath = $@".\data\parameter_sample\result_{DateTimeOffset.Now.ToString("yyyyMMdd_HHmmss")}.csv";
await using var resultWriter = new StreamWriter(resultFilePath, false);
await resultWriter.WriteLineAsync("seed,score,temp0,temp1");

var tempDirectory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
Directory.CreateDirectory(tempDirectory);


try
{
    var option = new ParallelOptions { CancellationToken = cts.Token, MaxDegreeOfParallelism = 16 };

    while (true)
    {
        await Parallel.ForEachAsync(Enumerable.Range(0, 10000), option, async (seed, ct) =>
        {
            var seedPath = $@".\data\in\{seed:0000}.txt";
            var input = await File.ReadAllLinesAsync(seedPath, ct);

            var random = Random.Shared;
            var tempHigh = 5.0 * Math.Pow(10.0, random.NextDouble());
            var tempLow = 1.0 * Math.Pow(10.0, random.NextDouble());

            var outputPath = Path.Join(tempDirectory, $"{Guid.NewGuid()}.txt");
            using (var writer = new StreamWriter(outputPath))
            {
                foreach (var line in await Run($@".\ahc014-a.exe {tempHigh} {tempLow}", input, ct))
                {
                    await writer.WriteLineAsync(line);
                }
            }

            var scoreText = (await Run($@".\vis.exe {seedPath} {outputPath}", Enumerable.Empty<string>(), ct)).First();
            var score = int.Parse(scoreText.Replace("Score = ", ""));

            var resultText = $"{seed},{score},{tempHigh},{tempLow}";

            lock (lockObject)
            {
                resultWriter.WriteLine(resultText);
            }

            Console.WriteLine(resultText);
        });
    }
}
finally
{
    Directory.Delete(tempDirectory, true);
}

static async Task<List<string>> Run(string command, IEnumerable<string> input, CancellationToken ct = default)
{
    // first argument is Process, if you want to know ProcessID, use StandardInput, use it.
    var (process, stdOut, _) = ProcessX.GetDualAsyncEnumerable(command);

    using (var stdIn = process.StandardInput)
    {
        foreach (var line in input)
        {
            stdIn.WriteLine(line);
        }
    }

    var outList = new List<string>();

    await Task.Run(async () =>
    {
        await foreach (var item in stdOut.WithCancellation(ct))
        {
            outList.Add(item);
        }
    }, ct);

    return outList;
}