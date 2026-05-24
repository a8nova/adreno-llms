package com.adreno.seeandsay.runner

import android.content.Context
import android.content.res.AssetManager
import java.io.File
import java.io.FileOutputStream

/**
 * Copies on-device-model assets from the APK's `assets/` tree into the app's
 * `filesDir`. The native binaries treat `filesDir/{smolvlm,mmstts}/` as their
 * working directory and resolve `weights/`, `kernels/`, `assets/` relative to
 * that, so the on-disk shape must mirror the APK layout exactly.
 */
class AssetExtractor(private val context: Context) {

    data class Progress(val bytesDone: Long, val bytesTotal: Long, val currentFile: String)

    private val targetDir: File = context.filesDir
    private val flagFile: File = File(targetDir, FLAG_FILE)

    fun alreadyExtracted(): Boolean = flagFile.exists()

    fun extract(onProgress: (Progress) -> Unit) {
        if (alreadyExtracted()) return
        val am = context.assets

        val files = mutableListOf<String>()
        for (root in ROOTS) walk(am, root, files)

        var total = 0L
        for (rel in files) {
            total += sizeOf(am, rel)
        }

        var done = 0L
        val buf = ByteArray(BUFFER_BYTES)
        for (rel in files) {
            val dest = File(targetDir, rel)
            dest.parentFile?.mkdirs()
            am.open(rel).use { input ->
                FileOutputStream(dest).use { output ->
                    while (true) {
                        val n = input.read(buf)
                        if (n <= 0) break
                        output.write(buf, 0, n)
                        done += n
                        onProgress(Progress(done, total, rel))
                    }
                }
            }
        }

        flagFile.writeText(System.currentTimeMillis().toString())
    }

    private fun walk(am: AssetManager, path: String, out: MutableList<String>) {
        val children = am.list(path).orEmpty()
        if (children.isEmpty()) {
            // Leaf — treat as a file. (An empty directory would also land here,
            // but our prepare script never produces those.)
            out += path
            return
        }
        for (child in children) {
            walk(am, "$path/$child", out)
        }
    }

    private fun sizeOf(am: AssetManager, rel: String): Long {
        // openFd works only on uncompressed entries; AGP build.gradle sets
        // noCompress for .bin and .cl so the big files take this fast path.
        return try {
            am.openFd(rel).use { it.declaredLength }
        } catch (_: Throwable) {
            am.open(rel).use { stream ->
                var n = 0L
                val skipBuf = ByteArray(64 * 1024)
                while (true) {
                    val r = stream.read(skipBuf)
                    if (r <= 0) break
                    n += r
                }
                n
            }
        }
    }

    companion object {
        // v2: switched mms-tts from flat layout (weights/*.bin) to nested per-
        // language layout (weights/eng/* + weights/amh/*) to support runtime
        // language switching via --lang <code>. Any device that previously ran
        // a v1 build needs a one-time re-extraction so the new tree shows up.
        // v3: added kernels/slice_along_t.cl for the vocoder-chunked streaming
        // path. Devices on v2 must re-extract or the binary fails to open the
        // new kernel at runtime.
        // v6: smolvlm bin now ships both 32x32 (key
        // model.vision_model.embeddings.position_embedding.weight) and 24x24
        // (key …weight_384) pos-embed variants for the Fast/Quality vision
        // resolution toggle. Devices on v5 don't have the _384 key in their
        // extracted weights/model.fp16.bin and crash with
        // "weight key not found: …position_embedding.weight_384" on first
        // vision pass at IMAGE_SIZE=384.
        // v7: added weights/amh/* + assets/amh/* for Amharic TTS. Devices on
        // v6 only have eng/ extracted; switching language to Amharic in
        // Settings fails with "Tokenizer::load failed to open
        // weights/amh/tokenizer_vocab.bin".
        // BUMP THIS whenever any file is added/changed under app/src/main/
        // assets/{mmstts,smolvlm} so existing installs pick up the new tree.
        private const val FLAG_FILE = ".extracted_v7"
        private const val BUFFER_BYTES = 256 * 1024
        // Top-level asset directories we extract — limits the scan so we don't
        // walk into AGP's internal asset paths.
        private val ROOTS = listOf("smolvlm", "mmstts")
    }
}
