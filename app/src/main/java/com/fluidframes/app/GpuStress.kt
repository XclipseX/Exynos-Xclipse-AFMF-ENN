package com.fluidframes.app

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import com.fluidframes.vulkan.ClBridge
import com.fluidframes.vulkan.NativePipeline

/**
 * Stress A/B: chạy liên tục chuỗi preprocess -> flowpack -> warp (giống đường NN thật, flow NPU giả)
 * bằng Vulkan, OpenCL (chép khung) và OpenCL (dma_buf, không chép), xen kẽ từng đoạn để chia đều ảnh hưởng nhiệt.
 * Mỗi đoạn chạy hết sức (khung nối khung) và ghi ms, fps, điện, nhiệt, xung GPU.
 */
object GpuStress {
    private const val OUT_W = 2340
    private const val OUT_H = 1080
    private const val CHUNK = 20

    private class Backend(val name: String, val run: (Int) -> DoubleArray?, val close: () -> Unit)

    private class Seg(val backend: String, val round: Int) {
        val wall = ArrayList<Double>()
        val gpu = ArrayList<Double>()
        var secs = 0.0
        val watts = ArrayList<Double>()
        val mhz = ArrayList<Int>()
        val load = ArrayList<Int>()
        var tempStart = 0.0
        var tempEnd = 0.0
        var headroomEnd = Float.NaN
        var error: String? = null
        val fps get() = if (secs > 0) wall.size / secs else 0.0
    }

    private fun pct(v: List<Double>, p: Double): Double {
        if (v.isEmpty()) return Double.NaN
        val s = v.sorted()
        return s[((s.size - 1) * p).toInt().coerceIn(0, s.size - 1)]
    }

    private fun battery(ctx: Context): Triple<Double, Double, Boolean> {   // (W, °C, đang sạc)
        val b = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val t = (b?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0
        val mv = b?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
        val plugged = (b?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val ua = ctx.getSystemService(BatteryManager::class.java).getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val ma = if (kotlin.math.abs(ua) > 20000) ua / 1000.0 else ua.toDouble()   // máy báo µA hoặc mA
        return Triple(kotlin.math.abs(ma) * mv / 1_000_000.0, t, plugged)
    }

    /** Chạy toàn bộ bài; [progress] nhận chuỗi trạng thái. Trả báo cáo dạng chữ. */
    fun run(ctx: Context, procLong: Int = 640, segSec: Int = 15, rounds: Int = 4, progress: (String) -> Unit): String {
        val (pw, ph) = NativePipeline.procSize(procLong, OUT_W, OUT_H)
        val notes = StringBuilder()
        val backends = ArrayList<Backend>()

        val vk = NativePipeline.stressOpen(ctx, OUT_W, OUT_H, pw, ph)
        if (vk != 0L) backends += Backend("Vulkan", { NativePipeline.stressRun(vk, it) }, { NativePipeline.stressClose(vk) })
        else notes.append("Vulkan: không mở được: ${NativePipeline.stressError()}\n")
        for ((mode, name) in listOf(0 to "OpenCL chép", 1 to "OpenCL dma_buf")) {
            val h = try { ClBridge.stressOpen(ctx, OUT_W, OUT_H, pw, ph, mode) } catch (t: Throwable) { 0L }
            val info = try { ClBridge.stressInfo() } catch (t: Throwable) { t.toString() }
            if (h != 0L) {
                backends += Backend(name, { ClBridge.stressRun(h, it) }, { ClBridge.stressClose(h) })
                notes.append("$name: $info\n")
            } else notes.append("$name: không mở được: $info\n")
        }
        if (backends.isEmpty()) return notes.toString().trim()

        val sys = SysLoad()
        val pm = ctx.getSystemService(PowerManager::class.java)
        var charging = false
        val segs = ArrayList<Seg>()
        try {
            // Làm nóng: 8 giây backend đầu, không tính.
            progress("Làm nóng GPU…")
            val tw = System.nanoTime()
            while ((System.nanoTime() - tw) / 1e9 < 8) backends[0].run(CHUNK) ?: break

            val total = rounds * backends.size
            for (r in 0 until rounds) {
                val n = backends.size
                for (k in 0 until n) {
                    val be = backends[(k + r) % n]   // xoay thứ tự mỗi vòng
                    val seg = Seg(be.name, r + 1)
                    progress("Vòng ${r + 1}/$rounds · ${be.name} (${segs.size + 1}/$total)…")
                    seg.tempStart = battery(ctx).second
                    val t0 = System.nanoTime()
                    var lastSample = t0
                    while (true) {
                        val ts = System.nanoTime()
                        val out = be.run(CHUNK)
                        if (out == null) { seg.error = "lỗi khi chạy"; break }
                        seg.secs += (System.nanoTime() - ts) / 1e9
                        for (i in 0 until CHUNK) { seg.wall += out[i]; seg.gpu += out[CHUNK + i] }
                        val now = System.nanoTime()
                        if (now - lastSample > 1_000_000_000L) {
                            lastSample = now
                            val (w, _, pl) = battery(ctx)
                            if (pl) charging = true
                            seg.watts += w
                            sys.gpuMhz()?.let { seg.mhz += it }
                            sys.gpuPct()?.let { seg.load += it }
                        }
                        if ((now - t0) / 1e9 >= segSec) break
                    }
                    seg.tempEnd = battery(ctx).second
                    seg.headroomEnd = try { pm.getThermalHeadroom(0) } catch (_: Throwable) { Float.NaN }
                    segs += seg
                }
            }
        } finally {
            backends.forEach { try { it.close() } catch (_: Throwable) {} }
        }
        return report(segs, notes.toString(), pw, ph, segSec, charging)
    }

    private fun report(segs: List<Seg>, notes: String, pw: Int, ph: Int, segSec: Int, charging: Boolean): String {
        val sb = StringBuilder()
        sb.append("Stress A/B: khung ${OUT_W}x$OUT_H, proc ${pw}x$ph, preprocess→flowpack→warp, chạy hết sức, ${segSec}s mỗi đoạn\n")
        if (charging) sb.append("⚠ Đang cắm sạc: số W không đúng, rút sạc rồi chạy lại để đo điện\n")
        sb.append(notes).append('\n')
        sb.append("TỔNG HỢP (fps = khung/giây khi chạy hết sức; J/khung = W ÷ fps, gồm cả màn hình)\n")
        for (name in segs.map { it.backend }.distinct()) {
            val ss = segs.filter { it.backend == name && it.error == null && it.wall.isNotEmpty() }
            if (ss.isEmpty()) { sb.append("$name: lỗi\n"); continue }
            val wall = ss.flatMap { it.wall }; val gpu = ss.flatMap { it.gpu }.filter { it >= 0 }
            val fps = wall.size / ss.sumOf { it.secs }
            val w = ss.flatMap { it.watts }.average()
            val mhz = ss.flatMap { it.mhz }.let { if (it.isEmpty()) "n/a" else "%.0f".format(it.average()) }
            sb.append("%s: %.0f fps · khung p50 %.2f / p99 %.2f ms · GPU p50 %.2f / p99 %.2f ms · %.2f W · %.1f mJ/khung · GPU %s MHz\n".format(
                name, fps, pct(wall, 0.5), pct(wall, 0.99), pct(gpu, 0.5), pct(gpu, 0.99), w, if (fps > 0) w / fps * 1000 else Double.NaN, mhz))
        }
        sb.append("\nTỪNG ĐOẠN (theo thời gian)\n")
        for (s in segs) {
            if (s.error != null) { sb.append("V${s.round} ${s.backend}: ${s.error}\n"); continue }
            sb.append("V%d %s: %.0f fps · p50 %.2f p99 %.2f ms · GPU p50 %.2f ms · %.2f W · pin %.1f→%.1f°C · headroom %.2f · GPU %s MHz %s%%\n".format(
                s.round, s.backend, s.fps, pct(s.wall, 0.5), pct(s.wall, 0.99), pct(s.gpu.filter { it >= 0 }, 0.5),
                s.watts.average(), s.tempStart, s.tempEnd, s.headroomEnd,
                if (s.mhz.isEmpty()) "n/a" else "%.0f".format(s.mhz.average()),
                if (s.load.isEmpty()) "n/a" else "%.0f".format(s.load.average())))
        }
        return sb.toString().trim()
    }
}
