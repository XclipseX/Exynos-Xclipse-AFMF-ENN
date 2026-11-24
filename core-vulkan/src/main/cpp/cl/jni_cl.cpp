// JNI cho thí nghiệm OpenCL (ClBridge.kt). Không bao giờ ném ra ngoài / crash.
#include <jni.h>

#include <exception>
#include <mutex>
#include <string>
#include <vector>

#include "cl_stress.h"

namespace ffcl {
std::string clProbeReport();
std::string clBenchWarp(const std::string& clSource, int outW, int outH, int procW, int procH, int iters);
}  // namespace ffcl

namespace {
jstring ToJ(JNIEnv* env, const std::string& s) { return env->NewStringUTF(s.c_str()); }

std::mutex g_stressMu;
std::string g_stressInfo;  // thông tin/lỗi stress gần nhất
void SetStressInfo(const std::string& s) {
    std::lock_guard<std::mutex> lk(g_stressMu);
    g_stressInfo = s;
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_fluidframes_vulkan_ClBridge_probe(JNIEnv* env, jclass) {
    try {
        return ToJ(env, ffcl::clProbeReport());
    } catch (const std::exception& ex) {
        return ToJ(env, std::string("OpenCL probe: ngoại lệ ") + ex.what());
    } catch (...) {
        return ToJ(env, "OpenCL probe: ngoại lệ không rõ");
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_fluidframes_vulkan_ClBridge_benchWarp(JNIEnv* env, jclass, jstring src, jint outW, jint outH,
                                               jint procW, jint procH, jint iters) {
    try {
        if (!src) return ToJ(env, "OpenCL warp: source null");
        const char* c = env->GetStringUTFChars(src, nullptr);
        if (!c) return ToJ(env, "OpenCL warp: không đọc được source");
        std::string s(c);
        env->ReleaseStringUTFChars(src, c);
        return ToJ(env, ffcl::clBenchWarp(s, outW, outH, procW, procH, iters));
    } catch (const std::exception& ex) {
        return ToJ(env, std::string("OpenCL warp: ngoại lệ ") + ex.what());
    } catch (...) {
        return ToJ(env, "OpenCL warp: ngoại lệ không rõ");
    }
}

// ---------------------------------------------------------------------------------------------
// Stress chuỗi khung CL (cl_stress.h). Lỗi/thông tin đọc bằng stressInfo().
// ---------------------------------------------------------------------------------------------
extern "C" JNIEXPORT jlong JNICALL
Java_com_fluidframes_vulkan_ClBridge_stressOpen(JNIEnv* env, jclass, jstring src, jint outW, jint outH,
                                                jint procW, jint procH, jint mode) {
    try {
        if (!src) { SetStressInfo("CL stress: source null"); return 0; }
        const char* c = env->GetStringUTFChars(src, nullptr);
        if (!c) { SetStressInfo("CL stress: không đọc được source"); return 0; }
        std::string s(c);
        env->ReleaseStringUTFChars(src, c);
        std::string info;
        void* h = ffcl::clStressOpen(s, outW, outH, procW, procH, mode, &info);
        SetStressInfo(info);
        return static_cast<jlong>(reinterpret_cast<intptr_t>(h));
    } catch (const std::exception& ex) {
        SetStressInfo(std::string("CL stress: ngoại lệ ") + ex.what());
    } catch (...) {
        SetStressInfo("CL stress: ngoại lệ không rõ");
    }
    return 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_fluidframes_vulkan_ClBridge_stressInfo(JNIEnv* env, jclass) {
    try {
        std::string s;
        {
            std::lock_guard<std::mutex> lk(g_stressMu);
            s = g_stressInfo;
        }
        return ToJ(env, s);
    } catch (...) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_fluidframes_vulkan_ClBridge_stressRun(JNIEnv* env, jclass, jlong h, jint frames) {
    try {
        if (h == 0 || frames <= 0 || frames > 1000000) {
            SetStressInfo("CL stress: handle 0 hoặc frames ngoài [1, 1e6]");
            return nullptr;
        }
        std::vector<double> out(static_cast<size_t>(frames) * 2);
        std::string info;
        if (!ffcl::clStressRun(reinterpret_cast<void*>(static_cast<intptr_t>(h)), frames, out.data(), &info)) {
            SetStressInfo(info);
            return nullptr;
        }
        SetStressInfo(info);
        jdoubleArray arr = env->NewDoubleArray(static_cast<jsize>(out.size()));
        if (!arr) return nullptr;
        env->SetDoubleArrayRegion(arr, 0, static_cast<jsize>(out.size()), out.data());
        return arr;
    } catch (const std::exception& ex) {
        SetStressInfo(std::string("CL stress: ngoại lệ ") + ex.what());
    } catch (...) {
        SetStressInfo("CL stress: ngoại lệ không rõ");
    }
    return nullptr;
}

extern "C" JNIEXPORT void JNICALL
Java_com_fluidframes_vulkan_ClBridge_stressClose(JNIEnv*, jclass, jlong h) {
    try {
        if (h != 0) ffcl::clStressClose(reinterpret_cast<void*>(static_cast<intptr_t>(h)));
    } catch (...) {
    }
}
