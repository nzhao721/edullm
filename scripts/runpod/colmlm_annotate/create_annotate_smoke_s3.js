/**
 * Co-LMLM annotate smoke: pull staged bundle from S3, run 200-doc smoke on 1xA100.
 *
 * Usage: node create_annotate_smoke_s3.js <aws-session.env> [gpuType] [podName]
 */
const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");

const awsPath = process.argv[2];
const gpuType = process.argv[3] ?? "NVIDIA A100-SXM4-80GB";
const podName = process.argv[4] ?? "colmlm-annotate-smoke-a100";
const smokePrefix =
  process.argv[5] ?? "s3://edullm-checkpoints/runpod/colmlm-annotate-smoke";

if (!awsPath || !fs.existsSync(awsPath)) {
  console.error("usage: node create_annotate_smoke_s3.js <aws-session.env> [gpu] [name]");
  process.exit(2);
}

const mcp = JSON.parse(
  fs.readFileSync(path.join(os.homedir(), ".cursor", "mcp.json"), "utf8")
);
const apiKey = mcp.mcpServers.runpod.env.RUNPOD_API_KEY;

const t = fs.readFileSync(awsPath, "utf8");
const pick = (k) => {
  const m = t.match(new RegExp(`export ${k}='([^']*)'`));
  return m ? m[1] : "";
};

const ak = pick("AWS_ACCESS_KEY_ID");
const sk = pick("AWS_SECRET_ACCESS_KEY");
const tok = pick("AWS_SESSION_TOKEN");
const region = pick("AWS_DEFAULT_REGION") || pick("AWS_REGION") || "us-east-1";
if (!ak || !sk || !tok) {
  console.error("aws-session.env missing keys");
  process.exit(2);
}

const here = __dirname;
const pyB64 = fs.readFileSync(path.join(here, "annotate_modernbert.py")).toString("base64");

const inner = [
  "set -Eeuo pipefail",
  "export PYTHONUNBUFFERED=1",
  `export AWS_ACCESS_KEY_ID='${ak}'`,
  `export AWS_SECRET_ACCESS_KEY='${sk}'`,
  `export AWS_SESSION_TOKEN='${tok}'`,
  `export AWS_DEFAULT_REGION='${region}'`,
  `export AWS_REGION='${region}'`,
  `SMOKE_S3='${smokePrefix}'`,
  "ROOT=/workspace/colmlm_annotate",
  "mkdir -p \"$ROOT\"/{model/final,input/shards,output}",
  "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl unzip >/dev/null",
  "if ! command -v aws >/dev/null 2>&1; then",
  "  curl -fsSL 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o /tmp/awscliv2.zip",
  "  unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install -i /usr/local/aws-cli -b /usr/local/bin >/dev/null",
  "fi",
  `echo '${pyB64}' | base64 -d > \"$ROOT/annotate_modernbert.py\"`,
  "echo '[s3] syncing smoke bundle'",
  "aws s3 sync \"$SMOKE_S3/model/final\" \"$ROOT/model/final\"",
  "aws s3 sync \"$SMOKE_S3/input\" \"$ROOT/input\"",
  "test -f \"$ROOT/model/final/model.safetensors\"",
  "test -f \"$ROOT/input/shards/train-00000.jsonl.gz\"",
  "ls -lah \"$ROOT/model/final\" \"$ROOT/input/shards\"",
  "echo '[deps] transformers zstandard'",
  "python3 -m pip install -q -U 'transformers>=4.48' zstandard",
  "nvidia-smi -L || true",
  "echo '[annotate] smoke: 200 docs'",
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
  "aws s3 sync \"$ROOT/output\" \"$SMOKE_S3/output/\" || true",
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
          console.log("smokeS3", smokePrefix);
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
