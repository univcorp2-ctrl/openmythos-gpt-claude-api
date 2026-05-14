# OpenMythos GPT / Claude API Bridge

OpenMythos を **Custom GPT Actions** や **Claude tool use** から呼び出すための FastAPI サーバーです。

重要: OpenMythos は研究用の PyTorch 実装です。このリポジトリは、ChatGPT や Claude のモデル本体を置き換えるものではなく、外部 API として OpenMythos を起動し、GPT / Claude から呼び出せるようにするためのブリッジです。

## できること

- `POST /generate` で OpenMythos にプロンプトを渡す
- Custom GPT Actions 用 OpenAPI schema を同梱
- Claude tool use 用 JSON schema を同梱
- Bearer token 認証に対応
- Docker 起動に対応
- 後から学習済み重み `.pt` / `.pth` / `.bin` を読み込み可能

## 注意

デフォルトでは **tiny の未学習モデル** を起動します。動作確認はできますが、まともな文章生成は期待できません。
実用するには、OpenMythos で学習した重みを `OPENMYTHOS_WEIGHTS_PATH` に指定してください。

## 1. ローカル起動

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
uvicorn openmythos_api.main:app --host 0.0.0.0 --port 8000
```

動作確認:

```bash
curl http://localhost:8000/health
```

生成テスト:

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Hello, OpenMythos",
    "max_new_tokens": 32,
    "n_loops": 4,
    "temperature": 0.8,
    "top_k": 40
  }'
```

## 2. APIキーを付ける

公開URLで使う場合は必ず API キーを設定してください。

```bash
export OPENMYTHOS_API_KEY='change-me-long-random-token'
uvicorn openmythos_api.main:app --host 0.0.0.0 --port 8000
```

呼び出し側:

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Authorization: Bearer change-me-long-random-token' \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"test"}'
```

## 3. Docker起動

```bash
docker build -t openmythos-api .
docker run --rm -p 8000:8000 \
  -e OPENMYTHOS_API_KEY='change-me' \
  openmythos-api
```

GPU を使う場合は、NVIDIA Container Toolkit をセットアップしたうえで以下のように起動します。

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e OPENMYTHOS_DEVICE=cuda \
  -e OPENMYTHOS_API_KEY='change-me' \
  openmythos-api
```

## 4. 学習済み重みを使う

```bash
export OPENMYTHOS_WEIGHTS_PATH=/models/openmythos.pt
export OPENMYTHOS_DEVICE=cuda
uvicorn openmythos_api.main:app --host 0.0.0.0 --port 8000
```

Docker の場合:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/models:/models \
  -e OPENMYTHOS_WEIGHTS_PATH=/models/openmythos.pt \
  -e OPENMYTHOS_DEVICE=cuda \
  -e OPENMYTHOS_API_KEY='change-me' \
  openmythos-api
```

## 5. 主な環境変数

| 変数 | デフォルト | 説明 |
|---|---:|---|
| `OPENMYTHOS_API_KEY` | 空 | 設定した場合、`Authorization: Bearer ...` が必須 |
| `OPENMYTHOS_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `OPENMYTHOS_DTYPE` | `auto` | `auto`, `float32`, `float16`, `bfloat16` |
| `OPENMYTHOS_MODEL_VARIANT` | `tiny` | `tiny`, `1b`, `3b`, `10b`, `50b`, `100b`, `500b`, `1t` |
| `OPENMYTHOS_ATTN_TYPE` | `gqa` | tiny モデルの attention。`gqa` または `mla` |
| `OPENMYTHOS_TOKENIZER` | `openai/gpt-oss-20b` | Hugging Face tokenizer ID またはローカルパス |
| `OPENMYTHOS_WEIGHTS_PATH` | 空 | `state_dict` のファイルパス |
| `OPENMYTHOS_MAX_SEQ_LEN` | `512` | tiny モデルの最大入力長 |
| `OPENMYTHOS_MAX_LOOP_ITERS` | `4` | tiny モデルの最大ループ回数 |
| `OPENMYTHOS_MAX_NEW_TOKENS_LIMIT` | `256` | APIで許可する最大生成トークン数 |

## 6. Custom GPT Actions で使う

1. このAPIを Render / Railway / Fly.io / Cloud Run / EC2 などにデプロイ
2. HTTPS の公開URLを用意
3. `actions/openapi.yaml` の `servers.url` を自分のURLに変更
4. Custom GPT の Actions に schema を貼る
5. 認証を使う場合は Bearer token を設定

`actions/openapi.yaml`:

```yaml
servers:
  - url: https://your-domain.example.com
```

## 7. Claude tool use で使う

`schemas/claude_tool.json` に Claude API 用の tool schema を入れています。

Claude が tool call を返したら、あなたのアプリ側でこの API の `/generate` を呼び出して、結果を Claude に返してください。

## 8. エンドポイント

### `GET /health`

APIの状態を返します。モデルはロードしません。

### `POST /generate`

Request:

```json
{
  "prompt": "Hello",
  "max_new_tokens": 64,
  "n_loops": 4,
  "temperature": 0.8,
  "top_k": 40,
  "return_full_text": false
}
```

Response:

```json
{
  "text": "...",
  "full_text": "...",
  "input_tokens": 3,
  "output_tokens": 64,
  "model_variant": "tiny",
  "device": "cpu",
  "warning": "Default tiny model is randomly initialized unless OPENMYTHOS_WEIGHTS_PATH is set."
}
```

## 9. 開発

```bash
pip install -r requirements-dev.txt
pytest
```

## 10. 本番運用メモ

- APIキーを必ず設定する
- GPUインスタンス推奨
- 初回起動時に Hugging Face tokenizer のダウンロードが走る場合があります
- `OPENMYTHOS_MODEL_VARIANT=1b` 以上は大きなGPUメモリが必要です
- 未学習のモデルは意味のある返答をしません
