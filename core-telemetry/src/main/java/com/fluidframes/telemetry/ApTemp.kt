package com.fluidframes.telemetry

import java.io.FileInputStream

/**
 * Nhiệt độ SoC (AP) trên máy Samsung không cần root: SELinux chặn /sys/class/thermal nhưng cho đọc
 * các thermistor của Samsung ở /sys/class/sec/sec-*-thermistor/temperature (đơn vị thường là 0.1 °C).
 */
object ApTemp {
    private val apPaths = listOf(
        "/sys/class/sec/sec-ap-thermistor/temperature",
        "/sys/devices/platform/sec-thermistor/temperature",
        "/sys/devices/virtual/sec/sec-thermistor/temperature",
    )

    /** Các thermistor phụ (để báo cáo, không nhầm với AP). */
    val others = linkedMapOf(
        "PA" to "/sys/class/sec/sec-pa-thermistor/temperature",
        "WiFi" to "/sys/class/sec/sec-wf-thermistor/temperature",
        "Pin" to "/sys/class/sec/sec-bat-thermistor/temperature",
        "Sạc" to "/sys/class/sec/sec-chg-thermistor/temperature",
    )

    fun readPath(p: String): Float = try {
        FileInputStream(p).use { f ->
            val b = ByteArray(64)
            val n = f.read(b)
            if (n <= 0) Float.NaN else {
                var t = String(b, 0, n).trim().toFloat()
                if (t > 1000f) t /= 1000f else if (t > 200f) t /= 10f
                if (t > 20f && t < 125f) t else Float.NaN   // bỏ cảm biến ảo (20.0 cố định, giá trị vô lý)
            }
        }
    } catch (_: Throwable) { Float.NaN }

    /** Nhiệt AP (°C), NaN nếu không đọc được. */
    fun read(): Float {
        for (p in apPaths) { val t = readPath(p); if (!t.isNaN()) return t }
        return Float.NaN
    }

    fun readOthers(): Map<String, Float> = others.mapValues { readPath(it.value) }
}
