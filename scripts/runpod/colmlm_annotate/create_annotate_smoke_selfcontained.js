/**
 * Self-contained Co-LMLM annotate smoke on a single A100.
 * Embeds only the annotate script; downloads model + one raw shard from Drive via gdown.
 *
 * Usage: node create_annotate_smoke_selfcontained.js [gpuType] [podName]
 */
const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");

const gpuType = process.argv[2] ?? "NVIDIA A100-SXM4-80GB";
const podName = process.argv[3] ?? "colmlm-annotate-smoke-a100";
const here = __dirname;

const mcp = JSON.parse(
  fs.readFileSync(path.join(os.homedir(), ".cursor", "mcp.json"), "utf8")
);
const apiKey = mcp.mcpServers.runpod.env.RUNPOD_API_KEY;

const pySrc = fs.readFileSync(path.join(here, "annotate_modernbert.py"));
const pyB64 = pySrc.toString("base64");

// Model: Drive folder 1SCV0Nrpn9wq11e-SCBljOZ3rdZ_iW9Hl / final/
const MODEL_FILES = {
  "config.json": "1D9bK95ufdfrqSZAMAU6tGxZu7LO_whY_",
  "model.safetensors": "1GBu_cEG66wU3DHDb7zORq-d9de6-ncCu",
  "tokenizer.json": "1TMBoQb4Pt6EQ0T80MYZ5GAvgF468to9_",
  "tokenizer_config.json": "1uxivqqMdh7SJAFL7uJmK6YFGZgAXeXXD",
};

// Dataset: Drive folder 1y1JR1CMXm0rx5PugsXhjLAJf3qH5C6uu / shards/train-00000.jsonl.gz
const SHARD_ID = "1YML6bp4oltgbzDXRnWURly1H1EUnMwg3";
const SHARD_NAME = "train-00000.jsonl.gz";

function gdownLine(id, dest) {
  // Newer gdown dropped --fuzzy; pass the file id directly.
  return `python3 -m gdown "${id}" -O "${dest}"`;
}

const modelDownloads = Object.entries(MODEL_FILES)
  .map(([name, id]) => gdownLine(id, name))
  .join("\n");

const inner = [
  "set -Eeuo pipefail",
  "export PYTHONUNBUFFERED=1",
  "ROOT=/workspace/colmlm_annotate",
  "mkdir -p \"$ROOT\"/{model/final,input/shards,output}",
  "cd \"$ROOT\"",
  `echo '${pyB64}' | base64 -d > \"$ROOT/annotate_modernbert.py\"`,
  "echo '[deps] pip install transformers gdown zstandard'",
  "python3 -m pip install -q -U 'transformers>=4.48' zstandard gdown",
  "echo '[model] downloading ModernBERT tagger final/ from Drive'",
  "cd \"$ROOT/model/final\"",
  modelDownloads,
  "ls -lah \"$ROOT/model/final\"",
  "test -f \"$ROOT/model/final/config.json\"",
  "test -f \"$ROOT/model/final/model.safetensors\"",
  "echo '[data] downloading one FineWeb raw shard from Drive (~90 MB)'",
  "cd \"$ROOT/input/shards\"",
  gdownLine(SHARD_ID, SHARD_NAME),
  "ls -lah \"$ROOT/input/shards\"",
  `test -f \"$ROOT/input/shards/${SHARD_NAME}\"`,
  "cd \"$ROOT\"",
  "nvidia-smi -L || true",
  "echo '[annotate] smoke: 200 docs from train-00000'",
  "python3 \"$ROOT/annotate_modernbert.py\" \\",
  "  --model-dir \"$ROOT/model/final\" \\",
  "  --input-dir \"$ROOT/input\" \\",
  "  --output-dir \"$ROOT/output\" \\",
  "  --id-field doc_id \\",
  "  --batch 32 \\",
  "  --max-files 1 \\",
  "  --max-docs-per-file 200 \\",
  "  --worker-index 0 \\",
  "  --num-workers 1 \\",
  "  --verify \\",
  "  2>&1 | tee \"$ROOT/smoke.log\"",
  "echo SMOKE_EXIT=$? | tee -a \"$ROOT/smoke.log\"",
  "echo SMOKE_DONE | tee -a \"$ROOT/smoke.log\"",
  "ls -lah \"$ROOT/output\" | tee -a \"$ROOT/smoke.log\"",
  "sleep 1800",
].join("\n");

const body = JSON.stringify({
  name: podName,
  imageName: "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  cloudType: "COMMUNITY",
  gpuTypeIds: [gpuType],
  gpuCount: 1,
  containerDiskInGb: 40,
  volumeInGb: 40,
  volumeMountPath: "/workspace",
  ports: ["22/tcp", "8888/http"],
  dockerStartCmd: ["bash", "-lc", inner],
});

console.log("payloadBytes", Buffer.byteLength(body));

const req = https.request(
  {
    hostname: "rest.runpod.io",
    path: "/v1/pods",
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(body),
    },
  },
  (res) => {
    let d = "";
    res.on("data", (c) => (d += c));
    res.on("end", () => {
      console.log("status", res.statusCode);
      try {
        const j = JSON.parse(d);
        if (j.id) {
          console.log("podId", j.id);
          console.log("costPerHr", j.costPerHr);
          console.log("gpuType", gpuType);
          console.log("shard", SHARD_NAME);
        } else {
          console.log(d.slice(0, 2000));
        }
      } catch {
        console.log(d.slice(0, 2000));
      }
    });
  }
);
req.on("error", (e) => console.error(e));
req.write(body);
req.end();
