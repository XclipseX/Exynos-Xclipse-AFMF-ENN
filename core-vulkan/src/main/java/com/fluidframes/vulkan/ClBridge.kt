package com.fluidframes.vulkan

import android.content.Context

/** Thí nghiệm OpenCL so với Vulkan: probe khả năng driver + bench kernel warp. */
object ClBridge {
    init {
        System.loadLibrary("fluidframes_native")
    }

    @JvmStatic external fun probe(): String
    @JvmStatic external fun benchWarp(src: String, outW: Int, outH: Int, procW: Int, procH: Int, iters: Int): String

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
}
