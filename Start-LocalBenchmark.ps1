Write-Host "[Compile]"
cargo build --release --bin ahc014-a
Move-Item ../target/release/ahc014-a.exe . -Force
Write-Host "[Run]"
$env:DURATION_MUL = "0.6"
dotnet marathon run-local