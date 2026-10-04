package org.inklab.bench

import android.app.Activity
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import android.view.WindowManager
import android.widget.TextView
import java.io.File

/**
 * InkBench: the netime protocol on Android. For every workload x K x backend:
 * 20 warm-up submissions, then 3 x 300 back-to-back submissions, then 3 x ~15 s
 * paced at the game cadence (one submission every K/30 s), recording busy time.
 * Rows go to <external files>/bench_<model>_<time>.csv in sweep.sh's format plus
 * device and thermal columns. "InkBench: DONE <path>" in logcat marks the end.
 * With --ez check true, every accelerated LiteRT run is followed (after its timing)
 * by one submission of the same model on litert-cpu from the same inputs, and
 * "CHECK" lines log how far the backend's outputs are from that fp32 reference.
 * With --ez dump true as well, the inputs, the reference and the backend's outputs
 * are written as raw little-endian float32 to <external files>/dump/<condition>/.
 *
 *   adb shell am start -n org.inklab.bench/.MainActivity \
 *       --es backends gles,litert-cpu,litert-gpu,nnapi,qnn-htp --es workloads ink,heat_s32,heat_s8,grayscott_s8
 */
class MainActivity : Activity() {
    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        status = TextView(this).apply { textSize = 14f; setPadding(32, 96, 32, 32) }
        setContentView(status)
        val backends = (intent.getStringExtra("backends") ?: "gles,litert-cpu,litert-gpu,nnapi,qnn-htp").split(",")
        val workloads = (intent.getStringExtra("workloads") ?: "ink,heat_s32,heat_s8,grayscott_s8").split(",")
        val ks = (intent.getStringExtra("ks") ?: "1,2,4,8").split(",").map { it.toInt() }
        val check = intent.getBooleanExtra("check", false)
        val dump = intent.getBooleanExtra("dump", false)
        Thread({ runAll(backends, workloads, ks, check, dump) }, "bench").start()
    }

    private fun show(msg: String) {
        Log.i(TAG, msg)
        runOnUiThread { status.text = msg }
    }

    private fun runAll(backends: List<String>, workloads: List<String>, ks: List<Int>, check: Boolean, dump: Boolean) {
        val out = File(getExternalFilesDir(null), "bench_${Build.MODEL.replace(' ', '_')}_${System.currentTimeMillis() / 1000}.csv")
        out.writeText("device,soc,workload,K,units,mode,rep,ms_per_submission,thermal,battery_c\n")
        val soc = if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE
        show("InkBench on ${Build.MODEL} ($soc), Android ${Build.VERSION.RELEASE}")
        for (wl in workloads) for (k in ks) for (backend in backends) {
            val tag = "$wl K=$k $backend"
            val runner: Pair<() -> Unit, AutoCloseable> = try {
                if (backend == "gles") {
                    val (name, sweeps) = splitWorkload(wl)
                    val g = GlesStep(name, sweeps); Pair({ g.submit(k) }, g)
                } else {
                    Log.i(TAG, "PLACEMENT-BEGIN $tag")
                    val m = LiteRtStep(this, assetFor(wl, k), backend)
                    Log.i(TAG, "PLACEMENT-END $tag")
                    Pair({ m.submit() }, m)
                }
            } catch (t: Throwable) {
                show("SKIP $tag: ${t.javaClass.simpleName}: ${t.message}")
                continue
            }
            try {
                repeat(20) { runner.first() }
                for (rep in 1..3) {
                    val t0 = SystemClock.elapsedRealtimeNanos()
                    repeat(300) { runner.first() }
                    val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6 / 300
                    row(out, soc, wl, k, backend, "b2b", rep, ms)
                }
                val calls = 450 / k
                val periodNs = (k / 30.0 * 1e9).toLong()
                for (rep in 1..3) {
                    var busy = 0L
                    val t0 = SystemClock.elapsedRealtimeNanos()
                    for (i in 0 until calls) {
                        val a = SystemClock.elapsedRealtimeNanos()
                        runner.first()
                        busy += SystemClock.elapsedRealtimeNanos() - a
                        val wait = t0 + (i + 1) * periodNs - SystemClock.elapsedRealtimeNanos()
                        if (wait > 0) Thread.sleep(wait / 1_000_000, (wait % 1_000_000).toInt())
                    }
                    row(out, soc, wl, k, backend, "paced", rep, busy / 1e6 / calls)
                }
                val m = runner.second as? LiteRtStep
                if (check && m != null && backend != "litert-cpu") try { checkAgainstCpu(tag, m, assetFor(wl, k), dump) }
                    catch (t: Throwable) { show("CHECK $tag failed: ${t.javaClass.simpleName}: ${t.message}") }
                show("done $tag")
            } catch (t: Throwable) {
                show("FAIL $tag: ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                runner.second.close()
            }
        }
        show("DONE ${out.absolutePath}")
    }

    /** Compares [m]'s last outputs with one fp32 XNNPACK submission of [asset] (same seeded inputs). */
    private fun checkAgainstCpu(tag: String, m: LiteRtStep, asset: String, dump: Boolean) {
        val got = m.outputFloats()
        val want = LiteRtStep(this, asset, "litert-cpu").use { it.submit(); it.outputFloats() }
        if (dump) {
            val dir = File(getExternalFilesDir(null), "dump/" + tag.replace(' ', '_').replace("=", ""))
            dir.mkdirs()
            File(dir, "meta.txt").writeText(m.describe())
            fun save(name: String, a: FloatArray) = File(dir, name).writeBytes(
                java.nio.ByteBuffer.allocate(a.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    .also { it.asFloatBuffer().put(a) }.array())
            m.inputFloats().forEachIndexed { i, a -> save("in$i.f32", a) }
            want.forEachIndexed { i, a -> a?.let { save("want$i.f32", it) } }
            got.forEachIndexed { i, a -> a?.let { save("got$i.f32", it) } }
        }
        for ((i, pair) in want.zip(got).withIndex()) {
            val (w, g) = pair
            if (w == null || g == null || w.size != g.size) { show("CHECK $tag out$i skipped"); continue }
            var lo = Float.MAX_VALUE; var hi = -Float.MAX_VALUE
            var maxAbs = 0.0; var sq = 0.0; var bad = 0
            for (j in w.indices) {
                lo = minOf(lo, w[j]); hi = maxOf(hi, w[j])
                if (!g[j].isFinite()) { bad++; continue }
                val d = Math.abs(g[j].toDouble() - w[j]); maxAbs = maxOf(maxAbs, d); sq += d * d
            }
            val range = (hi - lo).toDouble().takeIf { it > 0 } ?: 1.0
            val rms = Math.sqrt(sq / w.size)
            show("CHECK $tag out$i n=${w.size} range=${f(range)} maxabs=${f(maxAbs)} rms=${f(rms)} " +
                "max_of_range=${f(maxAbs / range)} rms_of_range=${f(rms / range)} nonfinite=$bad")
        }
    }

    private fun f(x: Double) = "%.3g".format(java.util.Locale.ROOT, x)

    private fun row(out: File, soc: String, wl: String, k: Int, units: String, mode: String, rep: Int, ms: Double) {
        val thermal = if (Build.VERSION.SDK_INT >= 29)
            (getSystemService(POWER_SERVICE) as PowerManager).currentThermalStatus else -1
        val batt = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            ?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1)?.div(10.0) ?: -1.0
        out.appendText("${Build.MODEL},$soc,$wl,$k,$units,$mode,$rep,${"%.4f".format(java.util.Locale.ROOT, ms)},$thermal,$batt\n")
    }

    /** "heat_s32" and "heat_s32_ec" -> ("heat", 32); "ink", "ink_i32_ec_fs" -> ("ink", 26). */
    private fun splitWorkload(wl: String): Pair<String, Int> {
        val parts = wl.split("_")
        return if (parts[0] == "ink") Pair("ink", 26)
            else Pair(parts[0], parts.first { it.matches(Regex("s\\d+")) }.substring(1).toInt())
    }

    /** Variants (see tflite_graph.py), combinable, in any order in the workload name:
     *  _i32 builds the gather index in int32, _ec pads by edge slices instead of MIRROR_PAD,
     *  _fs is the fp16-safe vorticity, _g1 gathers one channel at a time.
     *  padtest is MIRROR_PAD alone, gathertest the index chain and GATHER alone (K is ignored). */
    private fun assetFor(wl: String, k: Int): String {
        if (wl == "padtest" || wl == "gathertest") return "${wl}_128x256.tflite"
        val (name, sweeps) = splitWorkload(wl)
        val parts = wl.split("_")
        val v = listOf("i32", "ec", "fs", "g1").filter { it in parts }.joinToString("") { "_$it" }
        return if (name == "ink") "ink_128x256_j26${v}_u$k.tflite" else "${name}_128x256_s${sweeps}${v}_u$k.tflite"
    }

    companion object { const val TAG = "InkBench" }
}
