/**
 * Create a single-A100 RunPod for Co-LMLM ModernBERT annotate smoke.
 * Pod idles with SSH until the laptop uploads model+data and runs run_smoke.sh.
 *
 * Usage: node create_annotate_smoke_pod.js [gpuType] [podName]
 */
const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");

const gpuType = process.argv[2] ?? "NVIDIA A100-SXM4-80GB";
const podName = process.argv[3] ?? "colmlm-annotate-smoke-a100";

const mcp = JSON.parse(
  fs.readFileSync(path.join(os.homedir(), ".cursor", "mcp.json"), "utf8")
);
const apiKey = mcp.mcpServers.runpod.env.RUNPOD_API_KEY;

const pubKeyPath = path.join(os.homedir(), ".ssh", "runpod_ed25519.pub");
if (!fs.existsSync(pubKeyPath)) {
  console.error("missing SSH pubkey:", pubKeyPath);
  process.exit(2);
}
const publicKey = fs.readFileSync(pubKeyPath, "utf8").trim();

// Stay alive for SSH upload + smoke; terminate the pod when done.
const inner = [
  "set -Eeuo pipefail",
  "mkdir -p /workspace/colmlm_annotate",
  "echo '[pod] waiting for smoke payload under /workspace/colmlm_annotate'",
  "echo '[pod] touch /workspace/colmlm_annotate/START_SMOKE after upload to run'",
  "while [[ ! -f /workspace/colmlm_annotate/START_SMOKE ]]; do sleep 5; done",
  "bash /workspace/colmlm_annotate/run_smoke.sh 2>&1 | tee /workspace/colmlm_annotate/smoke.log",
  "echo SMOKE_EXIT=$? | tee -a /workspace/colmlm_annotate/smoke.log",
  "sleep 3600",
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
  env: {
    PUBLIC_KEY: publicKey,
  },
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
          console.log("gpuType", gpuType);
        } else {
          console.log(d.slice(0, 1500));
        }
      } catch {
        console.log(d.slice(0, 1500));
      }
    });
  }
);
req.on("error", (e) => console.error(e));
req.write(body);
req.end();
