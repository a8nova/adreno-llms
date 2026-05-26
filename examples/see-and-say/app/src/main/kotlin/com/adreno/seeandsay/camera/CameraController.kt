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
    private var boundLifecycle: LifecycleOwner? = null
    private var boundPreview: PreviewView? = null
    // Default to the rear camera — the see-and-say use case is "point at the
    // world and ask about it". Toggled by flip() from the UI.
    var lensFacing: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        private set

    fun bind(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        boundLifecycle = lifecycleOwner
        boundPreview = previewView
        rebind()
    }

    /**
     * Flip between rear and front cameras. Cheap — just rebinds CameraX with
     * the other selector. PreviewView stays mounted so there's no surface
     * teardown flicker. Returns the new facing for the UI to reflect.
     */
    fun flip(): CameraSelector {
        lensFacing = if (lensFacing == CameraSelector.DEFAULT_BACK_CAMERA)
            CameraSelector.DEFAULT_FRONT_CAMERA
        else
            CameraSelector.DEFAULT_BACK_CAMERA
        rebind()
        return lensFacing
    }

    private fun rebind() {
        val lo = boundLifecycle ?: return
        val pv = boundPreview ?: return
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
                .apply { setSurfaceProvider(pv.surfaceProvider) }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setResolutionSelector(resolutionSelector)
                .build()

            provider.unbindAll()
            provider.bindToLifecycle(lo, lensFacing, preview, imageCapture)
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
