Write-Host "[Compile]"
cargo build --release --bin ahc014-a
Move-Item ../target/release/ahc014-a.exe .
Write-Host "[Run]"
dotnet marathon run-local