"use strict";

// Create an SSH-ready RunPod with no cloud credentials or workload secrets.
const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");

const gpuType = process.argv[2] ?? "NVIDIA L40S";
const podName = process.argv[3] ?? "smollm2-colmlm";
const gpuCount = Number(process.argv[4] ?? "4");
const volumeGb = Number(process.argv[5] ?? "80");
const publicKeyPath = process.argv[6];
const cloudType = process.argv[7] ?? "COMMUNITY";

function readApiKey() {
  const mcpPath = path.join(process.env.USERPROFILE, ".cursor", "mcp.json");
  const mcp = JSON.parse(fs.readFileSync(mcpPath, "utf8"));
  const value = mcp?.mcpServers?.runpod?.env?.RUNPOD_API_KEY;
  if (!value) throw new Error(`RUNPOD_API_KEY missing from ${mcpPath}`);
  return value;
}

function readPublicKey() {
  const candidates = [
    publicKeyPath,
    path.join(os.homedir(), ".ssh", "id_ed25519.pub"),
    path.join(os.homedir(), ".ssh", "id_rsa.pub"),
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return fs.readFileSync(candidate, "utf8").trim();
  }
  throw new Error(`no SSH public key found under ${path.join(os.homedir(), ".ssh")}`);
}

const apiKey = readApiKey();

function request(method, requestPath, body) {
  return new Promise((resolve, reject) => {
    const encoded = body === undefined ? undefined : JSON.stringify(body);
    const req = https.request(
      {
        hostname: "rest.runpod.io",
        path: requestPath,
        method,
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          ...(encoded ? { "Content-Length": Buffer.byteLength(encoded) } : {}),
        },
      },
      (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          let parsed;
          try {
            parsed = data ? JSON.parse(data) : {};
          } catch {
            return reject(new Error(`${method} ${requestPath}: HTTP ${res.statusCode}: ${data}`));
          }
          if (res.statusCode < 200 || res.statusCode >= 300) {
            return reject(
              new Error(`${method} ${requestPath}: HTTP ${res.statusCode}: ${JSON.stringify(parsed)}`)
            );
          }
          resolve(parsed);
        });
      }
    );
    req.on("error", reject);
    req.setTimeout(30000, () => req.destroy(new Error(`${method} ${requestPath}: timeout`)));
    if (encoded) req.write(encoded);
    req.end();
  });
}

function sshEndpoint(pod) {
  const mappedPort = pod?.portMappings?.["22"] ?? pod?.portMappings?.[22];
  if (pod?.publicIp && mappedPort) {
    return { host: pod.publicIp, port: Number(mappedPort) };
  }
  const ports = pod?.runtime?.ports ?? pod?.ports ?? [];
  const ssh = ports.find(
    (item) => Number(item.private ?? item.privatePort ?? item.containerPort) === 22
  );
  if (!ssh) return null;
  return {
    host: ssh.ip ?? ssh.publicIp ?? pod.publicIp,
    port: Number(ssh.public ?? ssh.publicPort ?? ssh.hostPort),
  };
}

async function main() {
  if (process.argv[2] === "--delete") {
    const podId = process.argv[3];
    if (!podId) throw new Error("usage: create_idle_pod.js --delete POD_ID");
    await request("DELETE", `/v1/pods/${podId}`);
    process.stdout.write(JSON.stringify({ deleted: podId }) + "\n");
    return;
  }
  if (process.argv[2] === "--inspect") {
    const podId = process.argv[3];
    if (!podId) throw new Error("usage: create_idle_pod.js --inspect POD_ID");
    const pod = await request("GET", `/v1/pods/${podId}`);
    process.stdout.write(
      JSON.stringify({
        keys: Object.keys(pod),
        id: pod.id,
        status: pod.status ?? pod.desiredStatus,
        runtime: pod.runtime,
        ports: pod.ports,
        portMappings: pod.portMappings,
        publicIp: pod.publicIp,
        machineKeys: pod.machine ? Object.keys(pod.machine) : [],
      }) + "\n"
    );
    return;
  }
  const createBody = {
    name: podName,
    imageName: "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    cloudType,
    gpuTypeIds: [gpuType],
    gpuCount,
    containerDiskInGb: 40,
    volumeInGb: volumeGb,
    volumeMountPath: "/workspace",
    ports: ["22/tcp"],
    env: { PUBLIC_KEY: readPublicKey() },
  };
  let created;
  for (let attempt = 1; attempt <= 20; attempt += 1) {
    try {
      created = await request("POST", "/v1/pods", createBody);
      break;
    } catch (error) {
      if (!String(error).includes("no instances currently available") || attempt === 20) {
        throw error;
      }
      console.error(`capacity unavailable for ${gpuType} x${gpuCount}; retry ${attempt}/20`);
      await new Promise((resolve) => setTimeout(resolve, 15000));
    }
  }
  const podId = created.id;
  if (!podId) throw new Error(`create response has no pod id: ${JSON.stringify(created)}`);

  let ready = false;
  try {
    const deadline = Date.now() + 15 * 60 * 1000;
    while (Date.now() < deadline) {
      const pod = await request("GET", `/v1/pods/${podId}`);
      const ssh = sshEndpoint(pod);
      if (ssh?.host && ssh?.port) {
        ready = true;
        process.stdout.write(
          JSON.stringify({
            podId,
            name: podName,
            gpuType,
            gpuCount,
            cloudType,
            costPerHr: pod.cost ?? pod.costPerHr ?? created.costPerHr ?? null,
            sshHost: ssh.host,
            sshPort: ssh.port,
          }) + "\n"
        );
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }
    throw new Error(`pod ${podId} did not expose SSH within 15 minutes`);
  } finally {
    if (!ready) {
      try {
        await request("DELETE", `/v1/pods/${podId}`);
      } catch (cleanupError) {
        console.error(`failed to delete orphan pod ${podId}: ${cleanupError}`);
      }
    }
  }
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exit(1);
});
