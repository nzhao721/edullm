// Patch dockerStartCmd on an existing MixLaw pod with freshly minted AWS env,
// then the caller should restart-pod so the new start command runs.
const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");

const podId = process.argv[2];
const awsPath = process.argv[3];
const scriptPath = process.argv[4];
const deviceBatchSize = process.argv[5] ?? "16";
const gpuCount = process.argv[6] ?? "8";
const codeS3Uri =
  process.argv[7] ?? "s3://edullm-checkpoints/runpod/mixlaw-local-code.tgz";

if (!podId || !awsPath || !scriptPath) {
  console.error(
    "usage: relaunch_mixlaw_pod_cmd.js <podId> <aws.env> <run.sh> [batch] [nproc] [codeS3]"
  );
  process.exit(2);
}

const mcp = JSON.parse(
  fs.readFileSync(`${process.env.USERPROFILE}/.cursor/mcp.json`, "utf8")
);
const apiKey = mcp.mcpServers.runpod.env.RUNPOD_API_KEY;

let t = fs.readFileSync(awsPath, "utf8");
const pick = (k) => {
  const m = t.match(new RegExp(`export ${k}='([^']*)'`));
  return m ? m[1] : "";
};

function readWandbKey() {
  if (process.env.WANDB_API_KEY) {
    return process.env.WANDB_API_KEY.trim();
  }
  const candidates = [
    path.join(os.homedir(), ".wandb_api_key"),
    path.join("C:", "Users", process.env.USERNAME || "", ".wandb_api_key"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) {
      return fs.readFileSync(p, "utf8").trim();
    }
  }
  return "";
}

function readHfToken() {
  if (process.env.HF_TOKEN) {
    return process.env.HF_TOKEN.trim();
  }
  if (process.env.HUGGING_FACE_HUB_TOKEN) {
    return process.env.HUGGING_FACE_HUB_TOKEN.trim();
  }
  const candidates = [
    path.join(os.homedir(), ".cache", "huggingface", "token"),
    path.join(os.homedir(), ".huggingface", "token"),
    path.join(os.homedir(), ".hf_token"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) {
      return fs.readFileSync(p, "utf8").trim();
    }
  }
  return "";
}

const wandbKey = readWandbKey();
if (!wandbKey) {
  console.error("missing WANDB_API_KEY");
  process.exit(2);
}
const hfToken = readHfToken();

const smoke = fs.readFileSync(scriptPath, "utf8");
const b64 = Buffer.from(smoke, "utf8").toString("base64");
const esc = (s) => s.replace(/'/g, "'\\''");

const inner = [
  "set -Eeuo pipefail",
  `export AWS_ACCESS_KEY_ID='${pick("AWS_ACCESS_KEY_ID")}'`,
  `export AWS_SECRET_ACCESS_KEY='${pick("AWS_SECRET_ACCESS_KEY")}'`,
  `export AWS_SESSION_TOKEN='${pick("AWS_SESSION_TOKEN")}'`,
  `export AWS_DEFAULT_REGION='${pick("AWS_DEFAULT_REGION") || "us-east-1"}'`,
  `export AWS_REGION='${pick("AWS_REGION") || pick("AWS_DEFAULT_REGION") || "us-east-1"}'`,
  `export DEVICE_BATCH_SIZE='${deviceBatchSize}'`,
  `export NPROC='${gpuCount}'`,
  `export TASK_LOSS_NPROC='${gpuCount}'`,
  `export RECOVERY_MODE='${process.env.RECOVERY_MODE || "resume"}'`,
  process.env.RESUME_LOAD_PATH
    ? `export RESUME_LOAD_PATH='${esc(process.env.RESUME_LOAD_PATH)}'`
    : null,
  `export REUSE_LOCAL_CODE='${process.env.REUSE_LOCAL_CODE || "0"}'`,
  process.env.CODE_WANDB_ARTIFACT
    ? `export CODE_WANDB_ARTIFACT='${esc(process.env.CODE_WANDB_ARTIFACT)}'`
    : null,
  `export CODE_S3_URI='${codeS3Uri}'`,
  "cat > /workspace/aws-session.env <<'AWSEOF'",
  `export AWS_ACCESS_KEY_ID='${pick("AWS_ACCESS_KEY_ID")}'`,
  `export AWS_SECRET_ACCESS_KEY='${pick("AWS_SECRET_ACCESS_KEY")}'`,
  `export AWS_SESSION_TOKEN='${pick("AWS_SESSION_TOKEN")}'`,
  `export AWS_DEFAULT_REGION='${pick("AWS_DEFAULT_REGION") || "us-east-1"}'`,
  `export AWS_REGION='${pick("AWS_REGION") || pick("AWS_DEFAULT_REGION") || "us-east-1"}'`,
  "AWSEOF",
  "cat > /workspace/wandb-session.env <<'WBEOF'",
  `export WANDB_API_KEY='${esc(wandbKey)}'`,
  "export WANDB_START_METHOD=thread",
  "export WANDB_MODE=online",
  "export WANDB_PROJECT=mixlaw",
  "export WANDB_GROUP=370m-validation",
  "WBEOF",
];
if (hfToken) {
  inner.push(
    "cat > /workspace/hf-session.env <<'HFEOF'",
    `export HF_TOKEN='${esc(hfToken)}'`,
    `export HUGGING_FACE_HUB_TOKEN='${esc(hfToken)}'`,
    "HFEOF"
  );
}
if (process.env.DEPLOY_LOCAL_CODE === "1") {
  const repoRoot = path.resolve(__dirname, "..", "..");
  const deployFiles = [
    "experiments/skill-dag/mixlaw/train_mixlaw_validation_370m.py",
    "experiments/skill-dag/mixlaw/mixlaw_wandb.py",
    "experiments/skill-dag/mixlaw/mixlaw_runtime.py",
    "experiments/skill-dag/mixlaw/launch_validation_370m.sh",
  ];
  for (const rel of deployFiles) {
    const source = path.join(repoRoot, ...rel.split("/"));
    const encoded = fs.readFileSync(source).toString("base64");
    const target = `/workspace/edullm/${rel}`;
    inner.push(
      `mkdir -p '${path.posix.dirname(target)}'`,
      `echo '${encoded}' | base64 -d > '${target}'`
    );
  }
}
inner.push(
  "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl unzip >/dev/null",
  "if ! command -v aws >/dev/null 2>&1; then",
  "  curl -fsSL 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o /tmp/awscliv2.zip",
  "  unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install -i /usr/local/aws-cli -b /usr/local/bin >/dev/null",
  "fi",
  `echo '${b64}' | base64 -d > /workspace/run_job.sh`,
  "chmod +x /workspace/run_job.sh",
  "bash /workspace/run_job.sh"
);
const body = JSON.stringify({
  dockerStartCmd: ["bash", "-lc", inner.filter(Boolean).join("\n")],
});

const req = https.request(
  {
    hostname: "rest.runpod.io",
    path: `/v1/pods/${podId}`,
    method: "PATCH",
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
      if (res.statusCode >= 200 && res.statusCode < 300) {
        console.log("patched_dockerStartCmd", podId);
      } else {
        console.log(d.slice(0, 800));
        process.exit(1);
      }
    });
  }
);
req.on("error", (e) => {
  console.error(e);
  process.exit(1);
});
req.write(body);
req.end();
