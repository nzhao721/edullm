const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");

const awsPath = process.argv[2];
const scriptPath = process.argv[3];
const gpuType = process.argv[4] ?? "NVIDIA L40S";
const podName = process.argv[5] ?? "edullm-runpod";
const gpuCount = Number(process.argv[6] ?? "1");
const volumeGb = Number(process.argv[7] ?? "50");
const deviceBatchSize = process.argv[8] ?? "8";
const codeS3Uri =
  process.argv[9] ?? "s3://edullm-checkpoints/runpod/mixlaw-local-code.tgz";

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

const wandbKey = readWandbKey();
if (!wandbKey) {
  console.error("missing WANDB_API_KEY (env or ~/.wandb_api_key)");
  process.exit(2);
}

const smoke = fs.readFileSync(scriptPath, "utf8");
const b64 = Buffer.from(smoke, "utf8").toString("base64");

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
  `export CODE_S3_URI='${codeS3Uri}'`,
  "cat > /workspace/aws-session.env <<'AWSEOF'",
  `export AWS_ACCESS_KEY_ID='${pick("AWS_ACCESS_KEY_ID")}'`,
  `export AWS_SECRET_ACCESS_KEY='${pick("AWS_SECRET_ACCESS_KEY")}'`,
  `export AWS_SESSION_TOKEN='${pick("AWS_SESSION_TOKEN")}'`,
  `export AWS_DEFAULT_REGION='${pick("AWS_DEFAULT_REGION") || "us-east-1"}'`,
  `export AWS_REGION='${pick("AWS_REGION") || pick("AWS_DEFAULT_REGION") || "us-east-1"}'`,
  "AWSEOF",
  "cat > /workspace/wandb-session.env <<'WBEOF'",
  `export WANDB_API_KEY='${wandbKey.replace(/'/g, "'\\''")}'`,
  "export WANDB_START_METHOD=thread",
  "export WANDB_MODE=online",
  "export WANDB_PROJECT=mixlaw",
  "export WANDB_GROUP=370m-validation",
  "WBEOF",
  // awscli comes from pip later; need a bootstrap aws for code fetch first.
  "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl unzip >/dev/null",
  "if ! command -v aws >/dev/null 2>&1; then",
  "  curl -fsSL 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o /tmp/awscliv2.zip",
  "  unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install -i /usr/local/aws-cli -b /usr/local/bin >/dev/null",
  "fi",
  `echo '${b64}' | base64 -d > /workspace/run_job.sh`,
  "chmod +x /workspace/run_job.sh",
  "bash /workspace/run_job.sh",
].join("\n");

const body = JSON.stringify({
  name: podName,
  imageName: "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  cloudType: "COMMUNITY",
  gpuTypeIds: [gpuType],
  gpuCount,
  containerDiskInGb: 40,
  volumeInGb: volumeGb,
  volumeMountPath: "/workspace",
  ports: ["22/tcp", "8888/http"],
  dockerStartCmd: ["bash", "-lc", inner],
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
      console.log("status", res.statusCode);
      try {
        const j = JSON.parse(d);
        if (j.id) {
          console.log("podId", j.id);
          console.log("costPerHr", j.costPerHr);
          console.log("gpuCount", gpuCount);
          console.log("codeS3Uri", codeS3Uri);
        } else {
          console.log(d.slice(0, 1200));
        }
      } catch (e) {
        console.log(d.slice(0, 1200));
      }
    });
  }
);
req.on("error", (e) => console.error(e));
req.write(body);
req.end();
