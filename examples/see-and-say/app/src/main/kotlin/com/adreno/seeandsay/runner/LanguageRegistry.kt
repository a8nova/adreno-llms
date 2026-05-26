package com.adreno.seeandsay.runner

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream

/**
 * On-disk registry + downloader for MMS-TTS language packs.
 *
 * Three sources of "language is available":
 *   1. Bundled — eng + amh ship inside the APK assets, extracted on first
 *      launch by AssetExtractor. Always usable.
 *   2. Installed — language pack was downloaded from HF and extracted under
 *      <filesDir>/mmstts/{weights,assets}/<code>/. Persisted across launches.
 *   3. Available — listed in languages.json (fetched from HF on first open of
 *      the Language Picker) but not yet installed. Tap "Download" to get one.
 *
 * Status snapshot lives in [entries] as a StateFlow. UI subscribes; downloader
 * mutates as bytes arrive. Single-flight: one download at a time, queued by
 * call order.
 */
class LanguageRegistry(private val context: Context) {

    data class LangEntry(
        val code: String,
        val name: String,
        val nativeName: String,
        val script: String,
        val sizeBytes: Long,
        val status: Status,
    ) {
        sealed interface Status {
            data object Installed : Status
            data object Available : Status
            data class Downloading(val bytesDone: Long, val bytesTotal: Long) : Status {
                val percent: Float get() = if (bytesTotal > 0) bytesDone.toFloat() / bytesTotal else 0f
            }
            data class Failed(val message: String) : Status
        }
    }

    private val _entries = MutableStateFlow<List<LangEntry>>(emptyList())
    val entries: StateFlow<List<LangEntry>> = _entries.asStateFlow()

    private val mmsttsDir: File = File(context.filesDir, "mmstts")
    private val cacheDir: File = File(context.filesDir, "lang_cache").also { it.mkdirs() }
    private val cachedRegistry: File = File(cacheDir, "languages.json")

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    @Volatile private var activeDownload: Job? = null

    init {
        // Initial state from on-disk: bundled + installed, no remote fetch yet.
        scope.launch { rescanInstalled() }
    }

    /**
     * Snapshot what's currently on disk plus the bundled defaults. Called on
     * init and after any install/uninstall to keep UI in sync without a
     * network roundtrip.
     */
    private suspend fun rescanInstalled() = withContext(Dispatchers.IO) {
        // Start from cached registry if we have one (gives all 1140 entries
        // with display names). Otherwise just the bundled defaults.
        val available: MutableList<LangEntry> = mutableListOf()
        if (cachedRegistry.exists()) {
            try {
                val arr = JSONArray(cachedRegistry.readText())
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    available += LangEntry(
                        code = o.getString("code"),
                        name = o.optString("name", o.getString("code")),
                        nativeName = o.optString("native_name", o.optString("name", o.getString("code"))),
                        script = o.optString("script", ""),
                        sizeBytes = o.optLong("size_bytes", 75 * 1024 * 1024L),
                        status = LangEntry.Status.Available,
                    )
                }
            } catch (t: Throwable) {
                Log.w(TAG, "cached languages.json parse failed", t)
            }
        }

        val byCode = available.associateBy { it.code }.toMutableMap()

        // Seed with suggested defaults so there's something to show even
        // without a network fetch on first launch.
        for ((code, name, native) in SUGGESTED_DEFAULTS) {
            if (code !in byCode) {
                byCode[code] = LangEntry(
                    code = code, name = name, nativeName = native,
                    script = "", sizeBytes = 75L * 1024 * 1024,
                    status = LangEntry.Status.Available,
                )
            }
        }

        // Mark any on-disk lang as Installed.
        for ((code, e) in byCode) {
            if (isOnDisk(code)) {
                byCode[code] = e.copy(status = LangEntry.Status.Installed)
            }
        }
        _entries.value = byCode.values.sortedBy { it.name.lowercase() }
    }

    private fun isOnDisk(code: String): Boolean {
        // The "this language works" signal is the tokenizer_vocab.bin file —
        // the C++ binary fails immediately at start if it's missing.
        return File(mmsttsDir, "weights/$code/tokenizer_vocab.bin").exists()
    }

    /**
     * Best-effort fetch of languages.json from HF. Cached locally; safe to
     * call repeatedly. UI shows the previous cache while this runs.
     */
    suspend fun refreshRegistry(): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val url = URL("https://huggingface.co/$HF_REPO/resolve/main/mms-tts/languages.json")
            (url.openConnection() as HttpURLConnection).run {
                connectTimeout = 15_000
                readTimeout = 30_000
                requestMethod = "GET"
                inputStream.use { ins ->
                    val text = ins.bufferedReader().readText()
                    // Validate before overwriting cache.
                    JSONArray(text)
                    cachedRegistry.writeText(text)
                }
                disconnect()
            }
            rescanInstalled()
            Result.success(Unit)
        } catch (t: Throwable) {
            Log.w(TAG, "refreshRegistry failed", t)
            Result.failure(t)
        }
    }

    /**
     * Download + extract a language pack. Idempotent — silently no-ops if the
     * language is already installed or bundled. Cancels any in-flight download
     * before starting (single-flight).
     */
    fun download(code: String) {
        activeDownload?.cancel()
        activeDownload = scope.launch {
            try {
                doDownload(code)
            } catch (t: Throwable) {
                Log.w(TAG, "download($code) failed", t)
                updateOne(code) { it.copy(status = LangEntry.Status.Failed(t.message ?: t.javaClass.simpleName)) }
            }
        }
    }

    fun cancel(code: String) {
        activeDownload?.cancel()
        activeDownload = null
        updateOne(code) { e ->
            if (e.status is LangEntry.Status.Downloading)
                e.copy(status = LangEntry.Status.Available)
            else e
        }
    }

    private suspend fun doDownload(code: String) = withContext(Dispatchers.IO) {
        // Skip if already on-disk to avoid clobbering during an accidental
        // double-tap.
        val current = _entries.value.firstOrNull { it.code == code }
        if (current?.status == LangEntry.Status.Installed) {
            return@withContext
        }
        val totalGuess = current?.sizeBytes ?: (75L * 1024 * 1024)
        updateOne(code) { it.copy(status = LangEntry.Status.Downloading(0, totalGuess)) }

        val url = URL("https://huggingface.co/$HF_REPO/resolve/main/mms-tts/mms-tts-$code.zip")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 60_000
            requestMethod = "GET"
            instanceFollowRedirects = true
        }
        try {
            conn.connect()
            val rc = conn.responseCode
            if (rc !in 200..299) {
                throw IOException("HTTP $rc for $url")
            }
            val total = conn.contentLengthLong.let { if (it > 0) it else totalGuess }

            // Counting stream so the zip extractor's read calls also update
            // our byte counter (HF serves the zip as a single Stored stream,
            // so reading through ZipInputStream pulls bytes off the socket
            // 1:1 with what we report).
            var bytesRead = 0L
            val counting = object : java.io.FilterInputStream(conn.inputStream) {
                override fun read(b: ByteArray, off: Int, len: Int): Int {
                    val n = super.read(b, off, len)
                    if (n > 0) {
                        bytesRead += n
                        // Throttle the update flood — only fire when % advances by ≥1.
                        if (bytesRead == total ||
                            (bytesRead * 100 / total) != ((bytesRead - n) * 100 / total)) {
                            updateOne(code) { it.copy(status = LangEntry.Status.Downloading(bytesRead, total)) }
                        }
                    }
                    return n
                }
            }

            extractZipTo(counting, mmsttsDir, code)
            // Final progress bump in case the last %-bucket didn't trigger.
            updateOne(code) { it.copy(status = LangEntry.Status.Downloading(total, total)) }
            updateOne(code) { it.copy(status = LangEntry.Status.Installed) }
        } finally {
            try { conn.disconnect() } catch (_: Throwable) {}
        }
    }

    /**
     * Stream a zip into <root>/{weights,assets}/<code>/<rel>. Atomic write per
     * file: write to .tmp, fsync, rename. If an entry has the wrong code
     * (server returned the wrong pack) the whole extraction is rolled back to
     * avoid mixing two languages.
     */
    private fun extractZipTo(input: java.io.InputStream, root: File, expectedCode: String) {
        val written = mutableListOf<File>()
        try {
            ZipInputStream(input).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    if (entry.isDirectory) continue
                    val name = entry.name
                    // Zip layout we wrote: weights/<code>/* and assets/<code>/*.
                    // Reject anything else for safety (zip-slip + wrong-pack).
                    if ("../" in name || name.startsWith("/")) {
                        throw IOException("rejected entry path: $name")
                    }
                    val expectedPrefixA = "weights/$expectedCode/"
                    val expectedPrefixB = "assets/$expectedCode/"
                    if (!name.startsWith(expectedPrefixA) && !name.startsWith(expectedPrefixB)) {
                        throw IOException("zip entry $name doesn't match expected code $expectedCode")
                    }
                    val dst = File(root, name)
                    dst.parentFile?.mkdirs()
                    val tmp = File(dst.parentFile, dst.name + ".tmp")
                    tmp.outputStream().use { out -> zip.copyTo(out, bufferSize = 256 * 1024) }
                    if (dst.exists()) dst.delete()
                    if (!tmp.renameTo(dst)) {
                        throw IOException("rename failed: $tmp → $dst")
                    }
                    written += dst
                }
            }
        } catch (t: Throwable) {
            // Roll back: delete partial files so the language doesn't half-install.
            for (f in written) f.delete()
            throw t
        }
    }

    private fun updateOne(code: String, transform: (LangEntry) -> LangEntry) {
        _entries.update { list ->
            list.map { if (it.code == code) transform(it) else it }
        }
    }

    fun shutdown() { scope.cancel() }

    fun uninstall(code: String) {
        File(mmsttsDir, "weights/$code").deleteRecursively()
        File(mmsttsDir, "assets/$code").deleteRecursively()
        updateOne(code) { it.copy(status = LangEntry.Status.Available) }
    }

    fun wipeAll() {
        val installed = _entries.value.filter { it.status == LangEntry.Status.Installed }
        for (e in installed) {
            File(mmsttsDir, "weights/${e.code}").deleteRecursively()
            File(mmsttsDir, "assets/${e.code}").deleteRecursively()
        }
        _entries.update { list ->
            list.map {
                if (it.status == LangEntry.Status.Installed)
                    it.copy(status = LangEntry.Status.Available)
                else it
            }
        }
    }

    /** Snapshot of installed codes only — fed into the Settings radio list. */
    fun installedCodes(): List<LangEntry> = _entries.value.filter {
        it.status == LangEntry.Status.Installed
    }

    companion object {
        private const val TAG = "LanguageRegistry"
        private const val HF_REPO = "a8nova/adreno-llms-weights"

        // Suggested defaults shown when the registry hasn't been fetched yet.
        private val SUGGESTED_DEFAULTS = listOf(
            Triple("eng", "English",  "English"),
            Triple("amh", "Amharic",  "አማርኛ"),
            Triple("ara", "Arabic",   "العربية"),
        )
    }
}

/** StateFlow.update — copy of Compose's pattern, inlined to avoid pulling
 *  another dep into this runner module. */
private inline fun <T> MutableStateFlow<T>.update(transform: (T) -> T) {
    while (true) {
        val prev = value
        val next = transform(prev)
        if (compareAndSet(prev, next)) return
    }
}
