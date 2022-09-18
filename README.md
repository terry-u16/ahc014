# AHC014

## 初期設定

### テストケースアップロード

```sh
gsutil -m cp -r ./data/in/ gs://terry-u16-marathon-storage/ahc014
```

### ジャッジアップロード

`runner_config.json` のコンパイルオプションを編集する。

```json
"CompileOption": {
    "ExeName": "vis",
    "Files": [
        {
            "Source": "tools/src/bin/vis.rs",
            "Destination": "src/bin/main.rs"
        },
        {
            "Source": "tools/src/lib.rs",
            "Destination": "src/lib.rs"
        },
        {
            "Source": "tools/Cargo.toml",
            "Destination": "Cargo.toml"
        }
    ]
}
```
