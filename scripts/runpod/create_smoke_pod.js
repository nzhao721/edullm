const fs = require("fs");
const https = require("https");

const awsPath = process.argv[2];
const scriptPath = process.argv[3];
const gpuType = process.argv[4] ?? "NVIDIA L40S";
const podName = process.argv[5] ?? "curriculum-370m-smoke";
const deviceBatchSize = process.argv[6] ?? "8";
const mcp = JSON.parse(
  fs.readFileSync(`${process.env.USERPROFILE}/.cursor/mcp.json`, "utf8")
);
const apiKey = mcp.mcpServers.runpod.env.RUNPOD_API_KEY;

let t = fs.readFileSync(awsPath, "utf8");
const pick = (k) => {
  const m = t.match(new RegExp(`export ${k}='([^']*)'`));
  return m ? m[1] : "";
};

const smoke = fs.readFileSync(scriptPath, "utf8");
const b64 = Buffer.from(smoke, "utf8").toString("base64");

const inner = [
  "set -Eeuo pipefail",
  `export AWS_ACCESS_KEY_ID='${pick("AWS_ACCESS_KEY_ID")}'`,
  `export AWS_SECRET_ACCESS_KEY='${pick("AWS_SECRET_ACCESS_KEY")}'`,
  `export AWS_SESSION_TOKEN='${pick("AWS_SESSION_TOKEN")}'`,
  `export AWS_DEFAULT_REGION='${pick("AWS_DEFAULT_REGION")}'`,
  `export DEVICE_BATCH_SIZE='${deviceBatchSize}'`,
  "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git >/dev/null",
  `echo '${b64}' | base64 -d > /workspace/run_smoke.sh`,
  "chmod +x /workspace/run_smoke.sh",
  "bash /workspace/run_smoke.sh",
].join("\n");

const body = JSON.stringify({
  name: podName,
  imageName: "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  cloudType: "COMMUNITY",
  gpuTypeIds: [gpuType],
  gpuCount: 1,
  containerDiskInGb: 40,
  volumeInGb: 50,
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
      const j = JSON.parse(d);
      if (j.id) {
        console.log("podId", j.id);
        console.log("costPerHr", j.costPerHr);
      } else {
        console.log(d.slice(0, 800));
      }
    });
  }
);
req.on("error", (e) => console.error(e));
req.write(body);
req.end();
