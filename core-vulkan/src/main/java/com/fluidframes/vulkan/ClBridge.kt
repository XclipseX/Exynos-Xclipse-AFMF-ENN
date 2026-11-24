package com.fluidframes.vulkan

import android.content.Context

/** Thí nghiệm OpenCL so với Vulkan: probe khả năng driver + bench kernel warp. */
object ClBridge {
    init {
        System.loadLibrary("fluidframes_native")
    }

    @JvmStatic external fun probe(): String
    @JvmStatic external fun benchWarp(src: String, outW: Int, outH: Int, procW: Int, procH: Int, iters: Int): String

    /** Stress chuỗi khung CL (preprocess + flowpack + warp). mode 0 = copy (upload mỗi khung), 1 = dmabuf zero-copy.
     *  Trả handle (0 = lỗi, xem stressInfo()). */
    @JvmStatic external fun stressOpen(src: String, outW: Int, outH: Int, procW: Int, procH: Int, mode: Int): Long
    /** Thông tin cấu hình / lỗi gần nhất của stress. */
    @JvmStatic external fun stressInfo(): String?
    /** Chạy `frames` khung: [0, frames) = wall ms, [frames, 2*frames) = GPU ms (tổng START..END các lệnh). null = lỗi. */
    @JvmStatic external fun stressRun(h: Long, frames: Int): DoubleArray?
    @JvmStatic external fun stressClose(h: Long)

    /** Đọc assets/cl/frame_kernels.cl rồi bench kernel "warp". Không ném lỗi. */
    fun benchWarp(context: Context, outW: Int, outH: Int, procW: Int, procH: Int, iters: Int): String {
        val src = try {
            context.assets.open("cl/frame_kernels.cl").bufferedReader().use { it.readText() }
        } catch (t: Throwable) {
            return "OpenCL warp: không đọc được assets/cl/frame_kernels.cl: ${t.message}"
        }
        return try {
            benchWarp(src, outW, outH, procW, procH, iters)
        } catch (t: Throwable) {
            "OpenCL warp: lỗi JNI: ${t.message}"
        }
    }

    /** Đọc assets/cl/frame_kernels.cl rồi stressOpen. Trả 0 khi lỗi (chi tiết ở stressInfo()). Không ném lỗi. */
    fun stressOpen(context: Context, outW: Int, outH: Int, procW: Int, procH: Int, mode: Int): Long {
        val src = try {
            context.assets.open("cl/frame_kernels.cl").bufferedReader().use { it.readText() }
        } catch (t: Throwable) {
            return 0L
        }
        return try {
            stressOpen(src, outW, outH, procW, procH, mode)
        } catch (t: Throwable) {
            0L
        }
    }
}
