package com.adreno.seeandsay.runner

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.SendChannel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.consumeAsFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.File
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.io.PrintWriter
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread
import com.adreno.seeandsay.DeviceInfo
import com.adreno.seeandsay.SamplerSettings

/**
 * Long-lived `libsmolvlm.so --interactive` process. Replaces the one-shot
 * SmolVLMRunner: the binary stays resident across queries, so subsequent
 * Ask-and-Speak rounds skip the ~5–15 s cold-load (weights mmap, OpenCL
 * kernel compile, vision-tower prewarm).
 *
 * Protocol (verified against src/models/smolvlm-256m-instruct/src/main.cpp):
 *   - On startup the binary prints `ready. commands:` to stderr after weights
 *     load + decode-kernel prewarm.
 *   - Commands sent over stdin:
 *       `/image <abs path>`  load image + reset conversation
 *       `/reset`             drop image + KV cache
 *       `/quit`              clean exit (also: blank line, EOF)
 *       `<text>`             prompt for current turn
 *   - Tokens stream to stdout as raw UTF-8 (no separator).
 *   - End-of-turn is signaled by a `✓ turn N` line on stderr.
 *
 * Followup queries against the same image reuse KV cache — we only resend
 * `/image` when the path changes.
 */
class SmolVLMSession(context: Context) {

    private val nativeLibDir: String = context.applicationInfo.nativeLibraryDir
    private val cwd: File = File(context.filesDir, "smolvlm")

    @Volatile private var process: Process? = null
    @Volatile private var writer: PrintWriter? = null

    // Recreated on EACH start() so stop() + start() (Configurations dialog
    // sampler/image-size changes) actually waits for the NEW process's
    // "ready." marker instead of returning instantly because the old
    // CompletableDeferred was already completed on the first launch.
    private var readyDeferred = CompletableDeferred<Unit>()
    private val mutex = Mutex()

    @Volatile private var stdoutTarget: SendChannel<String>? = null
    @Volatile private var stderrTarget: SendChannel<String>? = null

    /**
     * Callback fired once per process start when the OpenCL device-info block
     * has finished printing to stderr. Lets MainViewModel expose hardware
     * details + compatibility analysis without each consumer re-running
     * clGetDeviceInfo.
     */
    @Volatile var onDeviceInfo: ((DeviceInfo) -> Unit)? = null

    sealed interface Output {
        data class Token(val text: String) : Output
        data class Final(val text: String) : Output
        data class Failed(val message: String) : Output
        /** Per-turn benchmark, parsed from the `✓ turn …` stderr summary. */
        data class Stats(
            val prefillTokens: Int,
            val prefillSec: Double,
            val prefillTps: Double,
            val decodeTokens: Int,
            val decodeSec: Double,
            val decodeTps: Double,
            val ttfsSec: Double,
            val ctxPos: Int,
            val rssMb: Int?,
            val peakRssMb: Int?,
        ) : Output
    }

    suspend fun start(settings: SamplerSettings = SamplerSettings()) = withContext(Dispatchers.IO) {
        if (process?.isAlive == true) return@withContext
        require(cwd.isDirectory) { "smolvlm cwd missing: $cwd" }
        // Reset the ready signal for this fresh launch — see field comment.
        readyDeferred = CompletableDeferred()

        val bin = "$nativeLibDir/libsmolvlm.so"
        // Sampler knobs (Configurations dialog in the UI) are passed at process
        // start. Changing any of them requires a stop()+start() — see
        // MainViewModel.updateSamplerSettings.
        val args = mutableListOf(
            bin, "--interactive",
            "--temperature", settings.temperature.toString(),
            "--top-k", settings.topK.toString(),
            "--top-p", settings.topP.toString(),
            "--max-tokens", settings.maxNewTokens.toString(),
            "--image-size", settings.imageSize.toString(),
        )
        val pb = ProcessBuilder(args).directory(cwd)
        pb.environment()["LD_LIBRARY_PATH"] = "$nativeLibDir:/vendor/lib64:/system/lib64"
        pb.environment()["NNOPT_DEBUG_LAYERS"] = "0"
        // Dump the exact token ids + decoded chat-template each turn (including
        // follow-ups) so we can diff against HF's reference rendering when an
        // answer looks off. Surfaces as `stderr: ▶ prompt total_pos=N turn=K …`
        // in Logcat under the SmolVLMSession tag.
        pb.environment()["NNOPT_DUMP_PROMPT_IDS"] = "1"
        // Diagnostic: have the binary print [kernel_cache] HIT/MISS/SAVE lines so we
        // can see in logcat whether the on-device binary cache is actually being
        // used across app restarts. Cheap (one line per kernel build, ~20 lines
        // first launch then zero on subsequent launches). Drop this when the cache
        // is confirmed stable in production.
        pb.environment()["NNOPT_KCACHE_LOG"] = "1"
        // Insert a brief GPU-idle window between each vision-transformer layer
        // so SurfaceFlinger can composite a frame during vision_pipe. Without
        // this, the entire screen goes white for the ~10 s vision pass because
        // Adreno 620's single CU starves the compositor. Trade-off: ~2-3 %
        // slowdown on vision_pipe (~240 ms across 12 layers).
        pb.environment()["NNOPT_GPU_YIELD"] = "1"
        // Run a synthetic 512x512 vision pass during binary startup so CLBlast
        // GEMM tuning + weight uploads happen BEFORE the first user query.
        // Eats ~10-12s of app launch but every subsequent vision pass starts
        // hot — eliminates the thermally-variable 10-14s cold first-query
        // TTFT. start() already waits up to 60 s for `ready.`; vision prewarm
        // (~12 s) fits comfortably under that.
        pb.environment()["NNOPT_PREWARM_VISION"] = "1"
        // No NNOPT_QCOM_PRIORITY override — falls through to the C++ default
        // (HIGH). We tried LOW (~2x slower) and NORMAL (some slowdown); both
        // were too costly. Going back to HIGH for raw speed and revisiting the
        // white-flash issue via a different mechanism (e.g. timesliced clFlush,
        // smaller workgroups, or accepting it as a device-level artifact).

        val p = pb.start()
        process = p
        writer = PrintWriter(BufferedWriter(OutputStreamWriter(p.outputStream)), /* autoFlush = */ true)

        thread(name = "smolvlm-stdout", isDaemon = true) {
            try {
                val reader = InputStreamReader(p.inputStream, Charsets.UTF_8)
                val buf = CharArray(256)
                while (true) {
                    val n = reader.read(buf)
                    if (n < 0) break
                    if (n > 0) stdoutTarget?.trySend(String(buf, 0, n))
                }
            } catch (_: Throwable) { /* expected on shutdown */ }
        }

        thread(name = "smolvlm-stderr", isDaemon = true) {
            // Buffer for the "── OpenCL device …" block. We append lines
            // between the opener and the closing "─────" line, then parse
            // once at end so a partial block never reaches onDeviceInfo.
            val deviceBlock = mutableListOf<String>()
            var inDeviceBlock = false
            try {
                BufferedReader(InputStreamReader(p.errorStream, Charsets.UTF_8)).useLines { lines ->
                    for (line in lines) {
                        Log.v(TAG, "stderr: $line")
                        stderrTarget?.trySend(line)
                        if (!readyDeferred.isCompleted && line.startsWith("ready.")) {
                            readyDeferred.complete(Unit)
                        }
                        // Device-info capture (one-shot per process).
                        if (line.startsWith("── OpenCL device")) {
                            inDeviceBlock = true
                            deviceBlock.clear()
                        } else if (inDeviceBlock) {
                            if (line.startsWith("──")) {
                                inDeviceBlock = false
                                parseDeviceInfo(deviceBlock)?.let { info ->
                                    try { onDeviceInfo?.invoke(info) }
                                    catch (t: Throwable) {
                                        Log.w(TAG, "onDeviceInfo callback threw", t)
                                    }
                                }
                            } else {
                                deviceBlock.add(line)
                            }
                        }
                    }
                }
            } catch (_: Throwable) { /* expected on shutdown */ }
        }

        withTimeout(60_000) { readyDeferred.await() }
        Log.i(TAG, "smolvlm session ready")
    }

    /**
     * Fire-and-forget: ship `/reset` + `/image <jpegPath>` to the binary so the
     * vision tower (~10 s) can begin running while the user is still composing
     * their prompt. The binary processes lines serially, so the eventual
     * prompt this image will be asked with will queue BEHIND the in-flight
     * vision pass. By the time the prompt arrives the binary's image_loaded
     * state is set and the LLM prefill can start immediately.
     *
     * Returns once the bytes are written to stdin (does NOT wait for the
     * vision tower to finish). Caller should hold the call_path (the jpeg
     * absolute path) and pair it with subsequent `ask(prompt)` calls so the
     * follow-up router treats turn-1 as "image already loaded" → just send
     * the prompt, no second `/image`.
     */
    suspend fun preloadImage(jpegPath: String) = withContext(Dispatchers.IO) {
        if (process?.isAlive != true) return@withContext
        mutex.withLock {
            writer?.let { w ->
                Log.d(TAG, "preloadImage: /reset + /image $jpegPath")
                w.println("/reset")
                w.println("/image $jpegPath")
                w.flush()
            }
        }
    }

    /**
     * Start a fresh conversation about [jpegPath] by issuing `/image <path>`
     * (which resets the binary's KV cache + image embeddings) and then
     * sending the prompt. Use this whenever the image being asked about is
     * different from the previous turn.
     */
    fun askWithImage(jpegPath: String, prompt: String): Flow<Output> =
        runQuery(loadImage = jpegPath, prompt = prompt)

    /**
     * Continue the current conversation with a follow-up prompt. The image
     * loaded by the previous askWithImage() call stays in KV cache; the model
     * answers as if mid-conversation about the same image.
     */
    fun ask(prompt: String): Flow<Output> =
        runQuery(loadImage = null, prompt = prompt)

    /**
     * Start a fresh text-only conversation. Sends `/reset` (clearing any image
     * + KV cache from a prior turn) then the prompt. The binary's
     * build_vlm_prompt(image_present=false, …) path handles the rest — no
     * image tokens, no vision tower pass.
     */
    fun askTextOnly(prompt: String): Flow<Output> =
        runQuery(loadImage = null, prompt = prompt, resetFirst = true)

    private fun runQuery(loadImage: String?, prompt: String, resetFirst: Boolean = false): Flow<Output> = channelFlow {
        if (process?.isAlive != true) {
            send(Output.Failed("smolvlm session not running"))
            return@channelFlow
        }

        mutex.withLock {
            val stdoutCh = Channel<String>(Channel.BUFFERED)
            val stderrCh = Channel<String>(Channel.BUFFERED)
            stdoutTarget = stdoutCh
            stderrTarget = stderrCh

            val accum = StringBuilder()
            val turnDoneRegex = Regex("""✓\s*turn\s+\d+""")
            val statsRegex = Regex(
                """prefill\s+(\d+)\s*tok\s+([0-9.]+)s\s*\(([0-9.]+)\s*tok/s\)\s+decode\s+(\d+)\s*tok\s+([0-9.]+)s\s*\(([0-9.]+)\s*tok/s\)\s+ctx=(\d+)""",
            )
            val errorRegex = Regex("""ERROR|FAIL""", RegexOption.IGNORE_CASE)
            val done = CompletableDeferred<Unit>()
            var firstTokenMs: Long = -1
            val promptSendMs = System.currentTimeMillis()

            try {
                writer?.let { w ->
                    if (loadImage != null) {
                        // `/image` in the binary doesn't actually wipe the KV
                        // cache (despite a stale comment claiming it does). We
                        // need an explicit `/reset` first, otherwise the prior
                        // conversation's tokens leak into this new turn and
                        // the model's follow-up answers stop making sense.
                        Log.d(TAG, "askWithImage: /reset + /image $loadImage")
                        w.println("/reset")
                        w.println("/image $loadImage")
                    } else if (resetFirst) {
                        Log.d(TAG, "askTextOnly: /reset (no image, fresh conversation)")
                        w.println("/reset")
                    } else {
                        Log.d(TAG, "ask (follow-up, no /image, KV cache reused)")
                    }
                    w.println(prompt)
                    w.flush()
                }

                coroutineScope {
                    val stdoutJob = launch {
                        for (chunk in stdoutCh) {
                            if (firstTokenMs < 0 && chunk.isNotEmpty()) {
                                firstTokenMs = System.currentTimeMillis()
                            }
                            accum.append(chunk)
                            send(Output.Token(chunk))
                        }
                    }
                    val stderrJob = launch {
                        for (line in stderrCh) {
                            if (turnDoneRegex.containsMatchIn(line)) {
                                send(Output.Final(accum.toString().trimEnd('\n', ' ')))
                                statsRegex.find(line)?.let { m ->
                                    val ttfs = if (firstTokenMs > 0)
                                        (firstTokenMs - promptSendMs) / 1000.0
                                    else Double.NaN
                                    send(
                                        Output.Stats(
                                            prefillTokens = m.groupValues[1].toInt(),
                                            prefillSec = m.groupValues[2].toDouble(),
                                            prefillTps = m.groupValues[3].toDouble(),
                                            decodeTokens = m.groupValues[4].toInt(),
                                            decodeSec = m.groupValues[5].toDouble(),
                                            decodeTps = m.groupValues[6].toDouble(),
                                            ttfsSec = ttfs,
                                            ctxPos = m.groupValues[7].toInt(),
                                            rssMb = readProcKb(pidOf(process), "VmRSS")?.let { it / 1024 },
                                            peakRssMb = readProcKb(pidOf(process), "VmHWM")?.let { it / 1024 },
                                        ),
                                    )
                                }
                                done.complete(Unit)
                                break
                            } else if (errorRegex.containsMatchIn(line)) {
                                send(Output.Failed(line))
                            }
                        }
                    }
                    try {
                        withTimeout(120_000) { done.await() }
                    } catch (_: TimeoutCancellationException) {
                        send(Output.Failed("smolvlm timed out (no ✓ turn after 120 s)"))
                    } finally {
                        stdoutJob.cancel()
                        stderrJob.cancel()
                    }
                }
            } finally {
                stdoutTarget = null
                stderrTarget = null
                stdoutCh.close()
                stderrCh.close()
            }
        }
    }.flowOn(Dispatchers.IO)

    fun isAlive(): Boolean = process?.isAlive == true

    /** PID of the running subprocess, or null if not running. Used by the
     *  ViewModel's live-RSS poller to read /proc/<pid>/status. */
    fun pid(): Long? = pidOf(process)

    /** Read VmRSS from /proc/<pid>/status, in KB. Cheap (~1 ms file read). */
    fun rssKb(): Int? = readProcKb(pid(), "VmRSS")

    fun stop() {
        try { writer?.println("/quit"); writer?.flush() } catch (_: Throwable) {}
        try { writer?.close() } catch (_: Throwable) {}
        process?.let { p ->
            if (p.isAlive) {
                p.destroy()
                try {
                    if (!p.waitFor(1, TimeUnit.SECONDS)) p.destroyForcibly()
                } catch (_: InterruptedException) {
                    p.destroyForcibly()
                }
            }
        }
        process = null
        writer = null
    }

    /** Reads a kB value from /proc/<pid>/status, e.g. "VmRSS" (current) or "VmHWM" (peak). */
    private fun readProcKb(pid: Long?, field: String): Int? {
        if (pid == null) {
            Log.w(TAG, "readProcKb($field): pid is null")
            return null
        }
        val prefix = "$field:"
        return try {
            val f = File("/proc/$pid/status")
            if (!f.canRead()) {
                Log.w(TAG, "readProcKb($field): /proc/$pid/status not readable (exists=${f.exists()})")
                return null
            }
            val line = f.readLines().firstOrNull { it.startsWith(prefix) }
            if (line == null) {
                Log.w(TAG, "readProcKb($field): line not found")
                return null
            }
            val kb = line.removePrefix(prefix).trim().split(Regex("\\s+")).firstOrNull()?.toIntOrNull()
            Log.d(TAG, "readProcKb($field, pid=$pid) = $kb kB")
            kb
        } catch (t: Throwable) {
            Log.w(TAG, "readProcKb($field) failed", t)
            null
        }
    }

    /**
     * Subprocess pid via reflection — `java.lang.Process.pid()` is API 26+ on
     * paper but is invisible to Kotlin against some Android SDK stubs. Reflection
     * is robust and the call is rare (once per query).
     */
    private fun pidOf(p: Process?): Long? {
        if (p == null) return null
        // Try Java 9+ Process.pid() via reflection first.
        return try {
            (p.javaClass.getMethod("pid").invoke(p) as? Long).also {
                Log.d(TAG, "pidOf via reflection = $it")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "pid() via reflection failed; falling back to toString parse", t)
            // Fallback: parse from Process.toString() which on Android typically
            // looks like "Process[pid=1234, ...]" or similar.
            try {
                val s = p.toString()
                val m = Regex("""pid=(\d+)""").find(s)
                m?.groupValues?.get(1)?.toLongOrNull().also {
                    Log.d(TAG, "pidOf via toString = $it")
                }
            } catch (_: Throwable) { null }
        }
    }

    companion object {
        private const val TAG = "SmolVLMSession"

        /**
         * Parses the body of the "── OpenCL device" block printed by
         * opencl_context.cpp at process start. Returns null if a required
         * field is missing (defensive — keeps onDeviceInfo from firing with
         * partial data on a future opencl_context refactor).
         */
        internal fun parseDeviceInfo(lines: List<String>): DeviceInfo? {
            val kv = mutableMapOf<String, String>()
            // Lines look like: "  platform        QUALCOMM Snapdragon(TM)"
            val rx = Regex("""^\s+(\S+)\s+(.+)$""")
            for (line in lines) {
                rx.matchEntire(line)?.let { m ->
                    kv[m.groupValues[1]] = m.groupValues[2].trim()
                }
            }
            fun yn(k: String) = kv[k]?.equals("yes", ignoreCase = true) == true
            fun intOf(k: String): Int? {
                val raw = kv[k] ?: return null
                // Strip trailing unit, e.g. "1708 MB" → "1708", "32 KB" → "32".
                val numTok = raw.trim().split(Regex("\\s+")).firstOrNull() ?: return null
                return numTok.toIntOrNull()
            }
            val platform = kv["platform"] ?: return null
            val device = kv["device"] ?: return null
            val openclVersion = kv["version"] ?: ""
            val driver = kv["driver"] ?: ""
            val cu = intOf("compute_units") ?: return null
            val clockMhz = intOf("max_clock_MHz") ?: 0
            val maxWg = intOf("max_workgroup") ?: return null
            val gmem = intOf("global_mem") ?: return null
            val lmem = intOf("local_mem") ?: return null
            return DeviceInfo(
                platform = platform,
                device = device,
                openclVersion = openclVersion,
                driver = driver,
                computeUnits = cu,
                maxClockMhz = clockMhz,
                maxWorkgroup = maxWg,
                globalMemMb = gmem,
                localMemKb = lmem,
                hasFp16 = yn("cl_khr_fp16"),
                hasQcomPerfHint = yn("qcom_perf_hint"),
                hasQcomRecordableQueues = yn("qcom_recordable_queues"),
                hasQcomDotProduct8 = yn("qcom_dot_product8"),
            )
        }
    }
}
