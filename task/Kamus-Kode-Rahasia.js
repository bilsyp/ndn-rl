// Fungsi Pembulatan untuk Client & Bridge
function getQuantizedName(state) {
  // 1. Buffer Bucket
  const b = state.buffer < 5 ? "panic" : state.buffer < 15 ? "safe" : "full";

  // 2. Throughput Bucket (Dibulatkan ke Mbps terdekat agar lebih fleksibel)
  const w =
    state.throughput < 2 ? "slow" : state.throughput < 6 ? "mid" : "fast";

  // 3. CWND Bucket
  const c = state.cwnd < 10 ? "congested" : state.cwnd < 40 ? "stable" : "wide";

  // 4. Segment Grouping (Agar tidak terlalu spesifik per satu segmen)
  const s = Math.floor(state.segment / 10);

  // Hasil Akhir: Nama NDN yang akan dicari di Mading (Router)
  return `/ndn/memo/s${s}/${b}/${w}/${c}`;
}
