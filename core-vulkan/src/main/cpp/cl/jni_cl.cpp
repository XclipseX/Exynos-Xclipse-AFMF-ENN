// JNI cho thí nghiệm OpenCL (ClBridge.kt). Không bao giờ ném ra ngoài / crash.
#include <jni.h>

#include <exception>
#include <string>

namespace ffcl {
std::string clProbeReport();
std::string clBenchWarp(const std::string& clSource, int outW, int outH, int procW, int procH, int iters);
}  // namespace ffcl

namespace {
jstring ToJ(JNIEnv* env, const std::string& s) { return env->NewStringUTF(s.c_str()); }
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
