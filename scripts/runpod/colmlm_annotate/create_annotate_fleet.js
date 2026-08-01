/**
 * Launch 19 single-A100 RunPod jobs for Co-LMLM ModernBERT annotate.
 * Each worker downloads only its shard + shared model from S3.
 *
 * Usage:
 *   node create_annotate_fleet.js <aws-session.env> [gpuType] [s3Prefix]
 */
const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");

const awsPath = process.argv[2];
const gpuType = process.argv[3] ?? "NVIDIA A100-SXM4-80GB";
const s3Prefix =
  process.argv[4] ?? "s3://edullm-checkpoints/runpod/colmlm-annotate";
const numWorkers = Number(process.argv[5] ?? "19");
const startIndex = Number(process.argv[6] ?? "0");
const endIndex = Number(process.argv[7] ?? String(numWorkers)); // exclusive

if (!awsPath || !fs.existsSync(awsPath)) {
  console.error(
    "usage: node create_annotate_fleet.js <aws-session.env> [gpu] [s3Prefix] [numWorkers] [start] [end]"
  );
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

function shardName(i) {
  return `train-${String(i).padStart(5, "0")}.jsonl.gz`;
}

function buildInner(workerIndex) {
  const shard = shardName(workerIndex);
  return [
    "set -Eeuo pipefail",
    "export PYTHONUNBUFFERED=1",
    `export AWS_ACCESS_KEY_ID='${ak}'`,
    `export AWS_SECRET_ACCESS_KEY='${sk}'`,
    `export AWS_SESSION_TOKEN='${tok}'`,
    `export AWS_DEFAULT_REGION='${region}'`,
    `export AWS_REGION='${region}'`,
    `S3_PREFIX='${s3Prefix}'`,
    `WORKER_INDEX='${workerIndex}'`,
    `NUM_WORKERS='${numWorkers}'`,
    `SHARD_NAME='${shard}'`,
    "ROOT=/workspace/colmlm_annotate",
    "mkdir -p \"$ROOT\"/{model/final,input/shards,output}",
    "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl unzip >/dev/null",
    "if ! command -v aws >/dev/null 2>&1; then",
    "  curl -fsSL 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o /tmp/awscliv2.zip",
    "  unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install -i /usr/local/aws-cli -b /usr/local/bin >/dev/null",
    "fi",
    `echo '${pyB64}' | base64 -d > \"$ROOT/annotate_modernbert.py\"`,
    "echo \"[s3] worker=${WORKER_INDEX}/${NUM_WORKERS} shard=${SHARD_NAME}\"",
    "aws s3 sync \"$S3_PREFIX/model/final\" \"$ROOT/model/final\"",
    // Only this worker's shard — avoid pulling all 19 onto every pod.
    "aws s3 cp \"$S3_PREFIX/input/shards/${SHARD_NAME}\" \"$ROOT/input/shards/${SHARD_NAME}\"",
    "test -f \"$ROOT/model/final/model.safetensors\"",
    "test -f \"$ROOT/input/shards/${SHARD_NAME}\"",
    "python3 -m pip install -q -U 'transformers>=4.48' zstandard",
    "nvidia-smi -L || true",
    "echo \"[annotate] starting worker ${WORKER_INDEX}\"",
    // Local input has a single file, so use num-workers=1 (not fleet size).
    "python3 \"$ROOT/annotate_modernbert.py\" \\",
    "  --model-dir \"$ROOT/model/final\" \\",
    "  --input-dir \"$ROOT/input\" \\",
    "  --output-dir \"$ROOT/output\" \\",
    "  --id-field doc_id \\",
    "  --batch 32 \\",
    "  --worker-index 0 \\",
    "  --num-workers 1 \\",
    "  --verify \\",
    "  2>&1 | tee \"$ROOT/annotate.log\"",
    "echo ANNOTATE_EXIT=$? | tee -a \"$ROOT/annotate.log\"",
    "echo ANNOTATE_DONE worker=$WORKER_INDEX | tee -a \"$ROOT/annotate.log\"",
    "aws s3 sync \"$ROOT/output\" \"$S3_PREFIX/output/worker-${WORKER_INDEX}/\"",
    "aws s3 cp \"$ROOT/annotate.log\" \"$S3_PREFIX/logs/worker-${WORKER_INDEX}.log\" || true",
    "sleep 300",
  ].join("\n");
}

function createPod(workerIndex) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      name: `colmlm-annotate-w${String(workerIndex).padStart(2, "0")}`,
      imageName: "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
      cloudType: "COMMUNITY",
      gpuTypeIds: [gpuType],
      gpuCount: 1,
      containerDiskInGb: 40,
      volumeInGb: 40,
      volumeMountPath: "/workspace",
      ports: ["22/tcp", "8888/http"],
      dockerStartCmd: ["bash", "-lc", buildInner(workerIndex)],
    });

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
          try {
            const j = JSON.parse(d);
            if (j.id) {
              resolve({
                workerIndex,
                podId: j.id,
                costPerHr: j.costPerHr,
                status: res.statusCode,
              });
            } else {
              resolve({
                workerIndex,
                error: d.slice(0, 500),
                status: res.statusCode,
              });
            }
          } catch (e) {
            resolve({ workerIndex, error: String(e), raw: d.slice(0, 300) });
          }
        });
      }
    );
    req.on("error", (e) => reject(e));
    req.write(body);
    req.end();
  });
}

(async () => {
  console.log(
    JSON.stringify({
      s3Prefix,
      gpuType,
      numWorkers,
      startIndex,
      endIndex,
    })
  );
  const results = [];
  // Create sequentially to avoid API rate limits; small delay between creates.
  for (let i = startIndex; i < endIndex; i++) {
    const r = await createPod(i);
    results.push(r);
    console.log(JSON.stringify(r));
    await new Promise((r) => setTimeout(r, 1500));
  }
  const ok = results.filter((r) => r.podId);
  const bad = results.filter((r) => !r.podId);
  console.log(
    JSON.stringify({
      created: ok.length,
      failed: bad.length,
      podIds: ok.map((r) => r.podId),
      estimatedCostPerHr: ok.length * (ok[0]?.costPerHr || 1.39),
    })
  );
  if (bad.length) process.exitCode = 1;
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
