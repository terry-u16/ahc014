param(
    [Parameter(mandatory)]
    [int]
    $seed
)

$in = ".\data\in\{0:0000}.txt" -f $seed
$out = "out.txt"
$env:MOVIE = "true"
Get-Content $in | cargo run --bin ahc014-a --release > $out
Remove-Item env:MOVIE
dotnet run --project .\Visualizer\Visualizer\ -- -i $in -s $out -o out.mp4
Invoke-Item out.mp4