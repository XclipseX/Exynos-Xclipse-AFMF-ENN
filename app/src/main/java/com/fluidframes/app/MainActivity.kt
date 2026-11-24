package com.fluidframes.app

import android.Manifest
import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionConfig
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import com.fluidframes.app.telemetry.TelemetryRecorder
import com.fluidframes.app.ui.AppEntry
import com.fluidframes.app.ui.AppRoot
import com.fluidframes.app.ui.TelemetryScreen
import com.fluidframes.app.ui.FluidTheme
import com.fluidframes.app.ui.HomeScreen
import com.fluidframes.app.ui.NpuScreen
import com.fluidframes.app.ui.SettingsScreen
import com.fluidframes.enn.NpuDiagnostics
import com.fluidframes.overlay.FluidSettings
import com.fluidframes.overlay.SettingsStore

/**
 * Luồng xin quyền: Trợ năng (bắt buộc) -> thông báo (Android 13+) -> ghi màn hình (chọn 1 app).
 * Giao diện Compose theo tinh thần One UI (docs/DESIGN.md).
 */
class MainActivity : ComponentActivity() {

    private lateinit var store: SettingsStore
    private var a11yOn by mutableStateOf(false)
    private var notifOn by mutableStateOf(true)
    private var settings by mutableStateOf(FluidSettings())
    private var message by mutableStateOf("")
    private var report by mutableStateOf<NpuDiagnostics.Report?>(null)
    private var diagBusy by mutableStateOf(false)
    private var gpuBench by mutableStateOf<String?>(null)
    private var gpuBusy by mutableStateOf(false)
    private val apps: List<AppEntry> by lazy {
        val i = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
        packageManager.queryIntentActivities(i, 0)
            .map { AppEntry(it.loadLabel(packageManager).toString(), it.activityInfo.packageName) }
            .filter { it.pkg != packageName }.distinctBy { it.pkg }.sortedBy { it.label.lowercase() }
    }

    private val notificationLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            notifOn = granted || Build.VERSION.SDK_INT < 33
            // Từ chối thông báo không chặn FGS; đi tiếp bước ghi màn hình nếu người dùng đang bấm Bắt đầu.
            if (pendingStart) { pendingStart = false; requestProjection() }
        }
    private var pendingStart = false

    private val projectionLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            val data = result.data
            if (result.resultCode == Activity.RESULT_OK && data != null) {
                ContextCompat.startForegroundService(this, ProjectionService.startIntent(this, result.resultCode, data))
                message = ""
            } else {
                message = "Bạn đã từ chối quyền ghi màn hình. Bấm Bắt đầu để thử lại."
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        store = SettingsStore(this)
        settings = store.load()
        refreshState()

        setContent {
            FluidTheme {
                val running by ProjectionService.running.collectAsState()
                val live by ProjectionService.live.collectAsState()
                AppRoot(
                    home = {
                        HomeScreen(
                            running = running, a11yOn = a11yOn, notifOn = notifOn, live = live, message = message,
                            onStart = ::onStartClicked,
                            onStop = { startService(ProjectionService.stopIntent(this)); message = "" },
                            onOpenA11y = ::openAccessibility,
                            onRequestNotif = { notificationLauncher.launch(Manifest.permission.POST_NOTIFICATIONS) },
                        )
                    },
                    settings = {
                        SettingsScreen(settings = settings, running = running) {
                            settings = it
                            store.save(it)
                        }
                    },
                    npu = {
                        NpuScreen(report = report, busy = diagBusy, onRun = ::runDiagnostics,
                            onCopy = ::copyReport, onShare = ::shareReport,
                            gpuBench = gpuBench, gpuBusy = gpuBusy, onGpuBench = ::runGpuBench, onCopyGpu = ::copyGpuBench, onGpuStress = ::runGpuStress)
                    },
                    telemetry = {
                        val ts by TelemetryRecorder.state.collectAsState()
                        TelemetryScreen(
                            state = ts, apps = apps, pipelineRunning = running,
                            onPick = { TelemetryRecorder.start(this, it.pkg, it.label) },
                            onStop = { Thread { TelemetryRecorder.stopNow() }.start() },
                            onShare = { f -> startActivity(Intent.createChooser(
                                Intent(Intent.ACTION_SEND).setType(f.mime).putExtra(Intent.EXTRA_STREAM, f.uri)
                                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION), "Chia sẻ ${f.name}")) },
                            onOpen = { f -> try { startActivity(Intent(Intent.ACTION_VIEW).setDataAndType(f.uri, f.mime)
                                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)) } catch (_: Throwable) {
                                Toast.makeText(this, "Không có app mở file này — dùng Chia sẻ hoặc mở trong Download", Toast.LENGTH_LONG).show() } },
                            onReset = { TelemetryRecorder.reset() },
                        )
                    },
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        refreshState()
    }

    private fun refreshState() {
        a11yOn = OverlayAccessibilityService.instance != null
        notifOn = Build.VERSION.SDK_INT < 33 ||
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
        if (a11yOn && message.startsWith("Hãy bật")) message = ""
    }

    private fun openAccessibility() {
        message = "Trong Trợ năng, chọn “Xclipse ENN-SS overlay” rồi bật. Cài lại APK sẽ tự tắt dịch vụ này."
        startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
    }

    private fun onStartClicked() {
        if (OverlayAccessibilityService.instance == null) {
            message = "Hãy bật dịch vụ Trợ năng “Xclipse ENN-SS overlay”, rồi quay lại bấm Bắt đầu."
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
            return
        }
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            pendingStart = true
            notificationLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            return
        }
        requestProjection()
    }

    private fun requestProjection() {
        val mpm = getSystemService(MediaProjectionManager::class.java)
        val intent = if (Build.VERSION.SDK_INT >= 34) {
            // Android 14+: chọn "Một ứng dụng" (game) để overlay của mình không nằm trong khung hình.
            mpm.createScreenCaptureIntent(MediaProjectionConfig.createConfigForUserChoice())
        } else {
            mpm.createScreenCaptureIntent()
        }
        projectionLauncher.launch(intent)
    }

    private fun runDiagnostics() {
        if (diagBusy) return
        diagBusy = true
        Thread {
            val r = try {
                NpuDiagnostics.run(this)
            } catch (t: Throwable) {
                NpuDiagnostics.Report(
                    listOf(NpuDiagnostics.Row("Lỗi khi chẩn đoán", NpuDiagnostics.Level.FAIL, "${t.javaClass.simpleName}: ${t.message}")),
                    null, false,
                )
            }
            runOnUiThread { report = r; diagBusy = false }
        }.start()
    }

    /** Đo warp OpenCL vs Vulkan ở 2340x1080 (proc 180p và 320p), kèm thông tin driver OpenCL. */
    private fun runGpuBench() {
        if (gpuBusy) return
        gpuBusy = true; gpuBench = null
        Thread {
            val sb = StringBuilder()
            sb.append(try { com.fluidframes.vulkan.ClBridge.probe() } catch (t: Throwable) { "OpenCL probe: ${t.javaClass.simpleName}: ${t.message}" }).append("\n\n")
            // Làm nóng GPU trước (xung GPU tăng dần), rồi mỗi mức chạy Vulkan -> OpenCL -> Vulkan để thấy độ lệch do xung.
            val (wpw, wph) = com.fluidframes.vulkan.NativePipeline.procSize(640, 2340, 1080)
            com.fluidframes.vulkan.NativePipeline.benchWarp(this, 2340, 1080, wpw, wph, 100)
            sb.append("Khung ra 2340x1080, 100 lần mỗi bài, tv = trung vị\n\n")
            for (ls in intArrayOf(192, 640, 1280)) {
                val (pw, ph) = com.fluidframes.vulkan.NativePipeline.procSize(ls, 2340, 1080)
                sb.append("— Mức ${ls}: —\n")
                sb.append(com.fluidframes.vulkan.NativePipeline.benchWarp(this, 2340, 1080, pw, ph, 100)).append('\n')
                sb.append(try { com.fluidframes.vulkan.ClBridge.benchWarp(this, 2340, 1080, pw, ph, 100) } catch (t: Throwable) { "OpenCL warp: ${t.javaClass.simpleName}: ${t.message}" }).append('\n')
                sb.append(com.fluidframes.vulkan.NativePipeline.benchWarp(this, 2340, 1080, pw, ph, 100)).append("\n\n")
            }
            val text = sb.toString().trim()
            runOnUiThread { gpuBench = text; gpuBusy = false }
        }.start()
    }

    /** Stress A/B Vulkan vs OpenCL (~4 phút), giữ màn hình sáng trong lúc chạy. */
    private fun runGpuStress() {
        if (gpuBusy) return
        gpuBusy = true; gpuBench = "Đang chuẩn bị…"
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        Thread {
            val text = try {
                GpuStress.run(this) { s -> runOnUiThread { gpuBench = s } }
            } catch (t: Throwable) { "Stress lỗi: ${t.javaClass.simpleName}: ${t.message}" }
            runOnUiThread {
                gpuBench = text; gpuBusy = false
                window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }.start()
    }

    private fun copyGpuBench() {
        val text = gpuBench ?: return
        getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("Xclipse ENN-SS GPU", text))
        Toast.makeText(this, "Đã sao chép kết quả đo GPU", Toast.LENGTH_SHORT).show()
    }

    private fun copyReport() {
        val text = listOfNotNull(report?.toText(), gpuBench).joinToString("\n\n").ifBlank { return }
        getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("Xclipse ENN-SS NPU", text))
        Toast.makeText(this, "Đã sao chép báo cáo", Toast.LENGTH_SHORT).show()
    }

    private fun shareReport() {
        val text = report?.toText() ?: return
        startActivity(Intent.createChooser(
            Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, text), "Chia sẻ báo cáo NPU"))
    }
}
