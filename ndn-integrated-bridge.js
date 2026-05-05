import { WebSocketServer } from "ws";
import fetch from "node-fetch";
import http from "http";
import { openUplinks } from "@ndn/cli-common";
import {
  makeInMemoryDataStore,
  RepoProducer,
  PrefixRegShorter,
} from "@ndn/repo";
import { Data, Name, Interest, digestSigning } from "@ndn/packet";
import { toUtf8 } from "@ndn/util";
import crypto from "crypto";

/**
 * NDN INTEGRATED BRIDGE SERVER
 * Menghubungkan Browser Client (Shaka Player) dengan Python RL Inference Server.
 * Versi ini dioptimalkan untuk menyinkronkan metrik browser dengan model AI.
 */

// --- KONFIGURASI ---
const WS_PORT = 5151;
const PYTHON_RL_URL = "http://localhost:8000/predict";
const NDN_PREFIX = "/ndn/video/stream";

// HTTP Agent untuk efisiensi koneksi dan mencegah socket hang-up
const httpAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 50,
  maxFreeSockets: 10,
  timeout: 15000,
  freeSocketTimeout: 4000,
});

let store;

/**
 * Simulasi Ingesti Data (Producer Side)
 * Menjamin ketersediaan segmen video di jaringan NDN sesuai permintaan AI.
 */
async function ensureDataExists(quality, seq) {
  const targetName = new Name(NDN_PREFIX).append(
    quality,
    "seg",
    seq.toString(),
  );

  const existing = await store.find(new Interest(targetName));
  if (existing) return existing;

  console.log(
    `🛠️  [Auto-Ingest] Menyiapkan segmen ${seq} untuk kualitas: ${quality}`,
  );

  const mockContent = toUtf8(
    JSON.stringify({
      timestamp: Date.now(),
      sequence: seq,
      quality: quality,
      data: "RAW_VIDEO_CHUNK_BASE64_PLACEHOLDER",
    }),
  );

  const packet = new Data(targetName, Data.FreshnessPeriod(20000), mockContent);
  await digestSigning.sign(packet);
  await store.insert(packet);

  return packet;
}

/**
 * Fungsi konsultasi ke Python RL Server
 * Memastikan payload yang dikirim Sesuai dengan skema NDNMetrics (Python).
 */
async function consultPythonRL(payload, retryCount = 1) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);

  try {
    const response = await fetch(PYTHON_RL_URL, {
      method: "POST",
      agent: httpAgent,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorDetail = await response.text();
      throw new Error(`Python Server Error ${response.status}: ${errorDetail}`);
    }

    return await response.json();
  } catch (err) {
    clearTimeout(timeoutId);
    if (
      retryCount > 0 &&
      (err.code === "ECONNRESET" || err.name === "AbortError")
    ) {
      console.warn(
        `⚠️  Mencoba ulang konsultasi RL (Sisa upaya: ${retryCount})...`,
      );
      return consultPythonRL(payload, retryCount - 1);
    }
    throw err;
  }
}

async function startBridge() {
  try {
    // 1. Inisialisasi NDN Uplink
    await openUplinks();
    console.log("✅ Terhubung ke NFD Forwarder.");

    // 2. Inisialisasi Repository Data
    store = await makeInMemoryDataStore();
    RepoProducer.create(store, { reg: PrefixRegShorter(1) });
    console.log("📦 NDN Repository Aktif.");

    // 3. Jalankan WebSocket Server untuk Browser
    const wss = new WebSocketServer({ port: WS_PORT });
    console.log(`🚀 Bridge Server aktif di ws://localhost:${WS_PORT}`);

    wss.on("connection", (ws) => {
      const clientId = crypto.randomBytes(2).toString("hex").toUpperCase();
      console.log(`🔌 [Client-${clientId}] Browser terhubung.`);

      ws.on("message", async (message) => {
        try {
          const msg = JSON.parse(message);

          if (msg.type === "REQUEST_RL") {
            const startTime = Date.now();

            /**
             * ADAPTASI INPUT (Sinkronisasi Model AI)
             * Memetakan 'last_quality' dari Shaka Player ke 'last_index' untuk Python PPO.
             */
            const payloadToPython = {
              client_id: clientId,
              buffer: parseFloat(msg.metrics?.buffer ?? 0),
              throughput: parseFloat(msg.metrics?.throughput ?? 0),
              // Konversi last_quality (client) -> last_index (AI)
              last_index: Math.min(
                2,
                Math.max(0, parseInt(msg.metrics?.last_quality ?? 1)),
              ),
              rtt: parseFloat(msg.metrics?.rtt ?? 0), // Default 0 jika tidak dikirim browser
            };

            // Sanitasi Akhir: Mencegah pengiriman nilai NaN ke Python
            if (isNaN(payloadToPython.buffer)) payloadToPython.buffer = 0;
            if (isNaN(payloadToPython.throughput))
              payloadToPython.throughput = 0.5;

            // Eksekusi prediksi melalui Python
            const result = await consultPythonRL(payloadToPython);

            const qualities = ["low", "mid", "high"];
            const qualityStr = qualities[result.bitrate_index] || "mid";

            // Pastikan segmen video yang diputuskan AI tersedia di jaringan (Repo)
            const dataPacket = await ensureDataExists(
              qualityStr,
              msg.next_seg || 0,
            );

            const processTime = Date.now() - startTime;
            console.log(
              `🧠 [Client-${clientId}] Keputusan AI: ${qualityStr.toUpperCase()} | ` +
                `RTT: ${payloadToPython.rtt}ms | Status: ${result.status} | Proc: ${processTime}ms`,
            );

            // Mengirim balik jawaban komplit ke Shaka Player Client
            ws.send(
              JSON.stringify({
                type: "DATA_RESPONSE",
                quality: result.bitrate_index, // Shaka Client membaca 'quality'
                bitrate_value: result.bitrate_value,
                volatility: result.volatility,
                is_cached: result.is_cached,
                name: dataPacket.name.toString(),
                payload: dataPacket.content.toString(),
                status: result.status,
                mean_tp: result.mean_throughput,
              }),
            );
          }
        } catch (err) {
          console.error(`❌ [Client-${clientId}] Bridge Error: ${err.message}`);
          ws.send(
            JSON.stringify({
              type: "ERROR",
              message: "Sinkronisasi Model RL Gagal",
              reason: err.message,
            }),
          );
        }
      });

      ws.on("close", () => console.log(`📴 [Client-${clientId}] Terputus.`));
    });
  } catch (err) {
    console.error("❌ Fatal Error saat memulai Bridge:", err);
    process.exit(1);
  }
}

startBridge();
