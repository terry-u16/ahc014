param(
    [Parameter(mandatory)]
    [int]
    $seed
)

$in = ".\data\in\{0:0000}.txt" -f $seed
Get-Content $in | cargo run --bin ahc014-a --release > out.txt
.\vis.exe $in out.txt