package com.fluidframes.app

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import com.fluidframes.telemetry.ApTemp
import com.fluidframes.vulkan.ClBridge
import com.fluidframes.vulkan.NativePipeline
import java.util.concurrent.locks.LockSupport

/**
 * Stress A/B toàn diện (~2 phút): chuỗi preprocess -> flowpack -> warp (giống đường NN thật, flow NPU giả)
 * trên mọi backend: Vulkan thường, Vulkan ưu tiên HIGH, Vulkan REALTIME (nếu được cấp), OpenCL dma_buf, OpenCL chép.
 * Mỗi backend chạy 2 chế độ: hết sức (đo trần) và 120 khung/giây cố định (đo điện/xung/trễ ở tải thật).
 * Đo nền khi rảnh để tính điện THÊM. Xen kẽ thứ tự mỗi vòng để chia đều ảnh hưởng nhiệt.
 */
object GpuStress {
    private const val OUT_W = 2340
    private const val OUT_H = 1080
    private const val PACE_FPS = 120
    private const val SEG_SEC = 6.0
    private const val ROUNDS = 2

    private class Backend(val name: String, val run: (Int) -> DoubleArray?, val close: () -> Unit)

    private class Seg(val backend: String, val paced: Boolean, val round: Int) {
        val wall = ArrayList<Double>(); val gpu = ArrayList<Double>()
        var secs = 0.0; var first = Double.NaN; var missed = 0
        val watts = ArrayList<Double>(); val mhz = ArrayList<Int>(); val load = ArrayList<Int>(); val cpu = ArrayList<Double>()
        var apStart = Float.NaN; var apEnd = Float.NaN; var batStart = 0.0; var batEnd = 0.0; var headroom = Float.NaN
        var error: String? = null
        val fps get() = if (secs > 0) wall.size / secs else 0.0
    }

    private fun pct(v: List<Double>, p: Double): Double {
        if (v.isEmpty()) return Double.NaN
        val s = v.sorted(); return s[((s.size - 1) * p).toInt().coerceIn(0, s.size - 1)]
    }
    private fun avgI(v: List<Int>) = if (v.isEmpty()) "n/a" else "%.0f".format(v.average())
    private fun f1(v: Float) = if (v.isNaN()) "n/a" else "%.1f".format(v)

    private fun battery(ctx: Context): Triple<Double, Double, Boolean> {   // (W, °C pin, đang sạc)
        val b = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val t = (b?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0
        val mv = b?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
        val plugged = (b?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val ua = ctx.getSystemService(BatteryManager::class.java).getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val ma = if (kotlin.math.abs(ua) > 20000) ua / 1000.0 else ua.toDouble()   // máy báo µA hoặc mA
        return Triple(kotlin.math.abs(ma) * mv / 1_000_000.0, t, plugged)
    }

    fun run(ctx: Context, procLong: Int = 640, progress: (String) -> Unit): String {
        val (pw, ph) = NativePipeline.procSize(procLong, OUT_W, OUT_H)
        val notes = StringBuilder()
        val backends = ArrayList<Backend>()

        for ((prio, name) in listOf(0 to "Vulkan", 1 to "Vulkan HIGH", 2 to "Vulkan REALTIME")) {
            val h = NativePipeline.stressOpen(ctx, OUT_W, OUT_H, pw, ph, prio)
            if (h == 0L) { notes.append("$name: không mở được: ${NativePipeline.stressError()}\n"); continue }
            val info = NativePipeline.stressPrio(h)
            notes.append("$name: ưu tiên $info\n")
            // Không được cấp ưu tiên thì bản này trùng "Vulkan" thường: bỏ để tiết kiệm thời gian.
            if (prio > 0 && !info.contains("được cấp")) { NativePipeline.stressClose(h); continue }
            backends += Backend(name, { NativePipeline.stressRun(h, it) }, { NativePipeline.stressClose(h) })
        }
        for ((mode, name) in listOf(1 to "OpenCL dma_buf", 0 to "OpenCL chép")) {
            val h = try { ClBridge.stressOpen(ctx, OUT_W, OUT_H, pw, ph, mode) } catch (t: Throwable) { 0L }
            val info = try { ClBridge.stressInfo() } catch (t: Throwable) { t.toString() }
            if (h != 0L) backends += Backend(name, { ClBridge.stressRun(h, it) }, { ClBridge.stressClose(h) })
            else notes.append("$name: không mở được: $info\n")
        }
        if (backends.isEmpty()) return notes.toString().trim()

        val sys = SysLoad()
        val pm = ctx.getSystemService(PowerManager::class.java)
        var charging = false
        val segs = ArrayList<Seg>()
        var idleW = Double.NaN; var idleAp = Float.NaN
        val othersStart = ApTemp.readOthers()
        try {
            // Nền: 3 giây không chạy GPU -> điện nền (màn hình, CPU app).
            progress("Đo điện nền (rảnh)…")
            val iw = ArrayList<Double>(); val ti = System.nanoTime()
            while ((System.nanoTime() - ti) / 1e9 < 3) { iw += battery(ctx).first; Thread.sleep(250) }
            idleW = iw.average(); idleAp = ApTemp.read()

            progress("Làm nóng GPU…")
            val tw = System.nanoTime()
            while ((System.nanoTime() - tw) / 1e9 < 3) backends[0].run(20) ?: break

            val plan = ArrayList<Pair<Backend, Boolean>>()
            for (r in 0 until ROUNDS) {
                val order = backends.indices.map { backends[(it + r) % backends.size] }.let { if (r % 2 == 1) it.reversed() else it }
                for (be in order) { plan += be to false; plan += be to true }
            }
            plan.forEachIndexed { idx, (be, paced) ->
                val seg = Seg(be.name, paced, idx / (backends.size * 2) + 1)
                progress("${idx + 1}/${plan.size} · ${be.name} · ${if (paced) "$PACE_FPS fps cố định" else "hết sức"}…")
                seg.apStart = ApTemp.read(); seg.batStart = battery(ctx).second
                sys.cpuPct()
                val period = 1_000_000_000L / PACE_FPS
                val t0 = System.nanoTime()
                var next = t0
                var lastSample = t0
                while (true) {
                    val n = if (paced) 1 else 20
                    val ts = System.nanoTime()
                    val out = be.run(n)
                    if (out == null) { seg.error = "lỗi khi chạy"; break }
                    seg.secs += (System.nanoTime() - ts) / 1e9
                    for (i in 0 until n) {
                        if (seg.first.isNaN()) seg.first = out[i]
                        seg.wall += out[i]; seg.gpu += out[n + i]
                        if (paced && out[i] > 1000.0 / PACE_FPS) seg.missed++
                    }
                    val now = System.nanoTime()
                    if (now - lastSample > 500_000_000L) {
                        lastSample = now
                        val (w, _, pl) = battery(ctx); if (pl) charging = true
                        seg.watts += w
                        sys.gpuMhz()?.let { seg.mhz += it }
                        sys.gpuPct()?.let { seg.load += it }
                        sys.cpuPct()?.let { seg.cpu += it }
                    }
                    if ((now - t0) / 1e9 >= SEG_SEC) break
                    if (paced) {
                        next += period
                        val wait = next - System.nanoTime()
                        if (wait > 0) LockSupport.parkNanos(wait) else next = System.nanoTime()
                    }
                }
                // Chế độ cố định: thời gian thực của đoạn (gồm cả lúc chờ) để fps = khung thật đã chạy / giây.
                if (paced) seg.secs = (System.nanoTime() - t0) / 1e9
                seg.apEnd = ApTemp.read(); seg.batEnd = battery(ctx).second
                seg.headroom = try { pm.getThermalHeadroom(0) } catch (_: Throwable) { Float.NaN }
                segs += seg
            }
        } finally {
            backends.forEach { try { it.close() } catch (_: Throwable) {} }
        }
        return report(segs, notes.toString(), pw, ph, charging, idleW, idleAp, othersStart)
    }

    private fun report(segs: List<Seg>, notes: String, pw: Int, ph: Int, charging: Boolean,
                       idleW: Double, idleAp: Float, others: Map<String, Float>): String {
        val sb = StringBuilder()
        sb.append("Stress A/B toàn diện: khung ${OUT_W}x$OUT_H, proc ${pw}x$ph, preprocess→flowpack→warp, ")
        sb.append("${SEG_SEC.toInt()}s mỗi đoạn × $ROUNDS vòng, 2 chế độ (hết sức / $PACE_FPS fps cố định)\n")
        if (charging) sb.append("⚠ Đang cắm sạc: số W không đúng\n")
        sb.append(notes)
        sb.append("Nền khi rảnh: %.2f W · AP %s°C · thermistor khác: %s\n\n".format(idleW, f1(idleAp),
            others.entries.joinToString { "${it.key} ${f1(it.value)}" }))

        for (paced in listOf(false, true)) {
            sb.append(if (paced) "== $PACE_FPS FPS CỐ ĐỊNH (tải giống chạy thật) ==\n" else "== HẾT SỨC (trần hiệu năng) ==\n")
            for (name in segs.map { it.backend }.distinct()) {
                val ss = segs.filter { it.backend == name && it.paced == paced && it.error == null && it.wall.isNotEmpty() }
                if (ss.isEmpty()) { sb.append("$name: lỗi\n"); continue }
                val wall = ss.flatMap { it.wall }; val gpu = ss.flatMap { it.gpu }.filter { it >= 0 }
                val fps = wall.size / ss.sumOf { it.secs }
                val w = ss.flatMap { it.watts }.average()
                val extra = w - idleW
                sb.append("%s: %.0f fps · khung p50 %.2f p95 %.2f p99 %.2f max %.2f ms · GPU p50 %.2f p99 %.2f ms\n".format(
                    name, fps, pct(wall, .5), pct(wall, .95), pct(wall, .99), wall.maxOrNull() ?: Double.NaN, pct(gpu, .5), pct(gpu, .99)))
                sb.append("   %.2f W (thêm %.2f W) · %.1f mJ/khung thêm · GPU %s MHz %s%% · CPU %s%%".format(
                    w, extra, if (fps > 0) extra / fps * 1000 else Double.NaN,
                    avgI(ss.flatMap { it.mhz }), avgI(ss.flatMap { it.load }),
                    ss.flatMap { it.cpu }.let { if (it.isEmpty()) "n/a" else "%.0f".format(it.average()) }))
                if (paced) sb.append(" · trễ hạn %.1f%%".format(100.0 * ss.sumOf { it.missed } / wall.size))
                sb.append(" · khung đầu đoạn %.2f ms\n".format(ss.map { it.first }.average()))
            }
            sb.append('\n')
        }
        sb.append("TỪNG ĐOẠN (theo thời gian)\n")
        for (s in segs) {
            val tag = "V${s.round} ${s.backend} ${if (s.paced) "@$PACE_FPS" else "max"}"
            if (s.error != null) { sb.append("$tag: ${s.error}\n"); continue }
            sb.append("%s: %.0f fps · p50 %.2f p99 %.2f ms · %.2f W · AP %s→%s°C · pin %.1f→%.1f°C · headroom %.2f · GPU %s MHz\n".format(
                tag, s.fps, pct(s.wall, .5), pct(s.wall, .99), s.watts.average(), f1(s.apStart), f1(s.apEnd),
                s.batStart, s.batEnd, s.headroom, avgI(s.mhz)))
        }
        return sb.toString().trim()
    }
}
