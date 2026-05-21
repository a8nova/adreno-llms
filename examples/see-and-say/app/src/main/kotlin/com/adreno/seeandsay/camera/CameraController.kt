package com.adreno.seeandsay.camera

import android.content.Context
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.File
import java.util.concurrent.Executors

/**
 * CameraX wrapper: binds Preview + ImageCapture to a [LifecycleOwner] and
 * writes captured JPEGs to a caller-provided file in the app's private dir.
 */
class CameraController(private val context: Context) {

    private var imageCapture: ImageCapture? = null
    private val captureExecutor = Executors.newSingleThreadExecutor()

    fun bind(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            // Cap resolution — SmolVLM resizes internally; large captures only slow us down.
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(Size(1024, 1024), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER),
                )
                .build()

            val preview = Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .build()
                .apply { setSurfaceProvider(previewView.surfaceProvider) }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setResolutionSelector(resolutionSelector)
                .build()

            provider.unbindAll()
            provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
        }, ContextCompat.getMainExecutor(context))
    }

    fun capture(target: File, onResult: (Result<File>) -> Unit) {
        val ic = imageCapture ?: run {
            onResult(Result.failure(IllegalStateException("CameraX not bound yet")))
            return
        }
        target.parentFile?.mkdirs()
        val opts = ImageCapture.OutputFileOptions.Builder(target).build()
        ic.takePicture(opts, captureExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(out: ImageCapture.OutputFileResults) {
                onResult(Result.success(target))
            }
            override fun onError(e: ImageCaptureException) {
                Log.e(TAG, "capture failed", e)
                onResult(Result.failure(e))
            }
        })
    }

    fun shutdown() {
        captureExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraController"
    }
}
