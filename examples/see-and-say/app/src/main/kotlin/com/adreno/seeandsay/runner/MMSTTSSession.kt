package com.adreno.seeandsay.runner

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.SendChannel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.DataInputStream
import java.io.File
import java.io.IOException
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.io.PrintWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

/**
 * Long-lived `libmmstts.so --interactive` process. Replaces the one-shot
 * MMSTTSRunner: the binary stays resident across queries, so subsequent
 * speak() calls skip the ~3–5 s cold-load (weights mmap, vocoder kernel
 * compile, NNOPT_PREWARM forward).
 *
 * Protocol (verified against src/models/mms-tts/src/main.cpp:368–429):
 *   - On startup, prints `ready. type text, blank line or Ctrl-D to quit.` to stderr.
 *   - Each utterance: send `<text>\n` on stdin. Binary writes `output.wav` in
 *     cwd, then prints a perf line and `TTS_WAV_READY output.wav` to stderr.
 *   - Blank line or EOF exits the REPL.
 */
class MMSTTSSession(context: Context) {

    private val nativeLibDir: String = context.applicationInfo.nativeLibraryDir
    private val cwd: File = File(context.filesDir, "mmstts")
    private val wavOut: File = File(cwd, "output.wav")

    @Volatile private var process: Process? = null
    @Volatile private var writer: PrintWriter? = null
    @Volatile private var pcmIn: DataInputStream? = null

    // Recreated on EACH start() so that stop() + start() (e.g. on TTS-settings
    // change via the Configurations dialog) properly waits for the NEW process
    // to print "ready." instead of returning instantly because the old
    // CompletableDeferred was already completed on the first launch.
    private var readyDeferred = CompletableDeferred<Unit>()
    private val mutex = Mutex()
    @Volatile private var stderrTarget: SendChannel<String>? = null

    data class Result(
        /** Full PCM int16 sample buffer. Always populated when binary runs in
         *  NNOPT_PCM_STREAM mode (the default we set). Prefer this over [wav]
         *  for low-latency AudioTrack playback (no MediaPlayer.prepare()). */
        val pcm: ShortArray?,
        val sampleRate: Int,
        /** Persisted WAV file. Always non-null: even in PCM-stream mode we
         *  emit a WAV alongside so legacy callers (Speak tab perf panel,
         *  retry-from-disk paths) still work. */
        val wav: File,
        val audioSec: Double,
        val rtf: Double,
        val fwdSec: Double,
        val ttfsSec: Double,
        val rssMb: Int?,
        val peakRssMb: Int?,
    ) {
        // ShortArray screws up data class equals; we don't compare Results so
        // override to satisfy the compiler without doing element-wise compare.
        override fun equals(other: Any?): Boolean = this === other
        override fun hashCode(): Int = System.identityHashCode(this)
    }

    suspend fun start(
        langCode: String = "",
        tts: com.adreno.seeandsay.TtsSettings = com.adreno.seeandsay.TtsSettings(),
    ) = withContext(Dispatchers.IO) {
        if (process?.isAlive == true) return@withContext
        require(cwd.isDirectory) { "mmstts cwd missing: $cwd" }
        // Reset the ready signal for this fresh launch — see field comment.
        readyDeferred = CompletableDeferred()

        val bin = "$nativeLibDir/libmmstts.so"
        val args = buildList {
            add("--interactive")
            if (langCode.isNotBlank()) { add("--lang"); add(langCode) }
            // VITS knobs surfaced by the Configurations dialog. The binary
            // parses these in main.cpp and applies them per-utterance: noise
            // scales multiply the rng-filled noise buffers before forward,
            // length-scale is forwarded via NNOPT_TTS_LENGTH_SCALE inside the
            // binary (read by backbone.cpp::tts_forward_graph).
            add("--noise-scale");   add(tts.noiseScale.toString())
            add("--noise-scale-w"); add(tts.noiseScaleW.toString())
            add("--length-scale");  add(tts.lengthScale.toString())
        }
        val pb = ProcessBuilder(listOf(bin) + args).directory(cwd)
        pb.environment()["LD_LIBRARY_PATH"] = "$nativeLibDir:/vendor/lib64:/system/lib64"
        pb.environment()["NNOPT_DEBUG_LAYERS"] = "0"
        // Prewarm: run a full synthetic forward at binary startup (~7 s wall)
        // so the FIRST real user query is ~1-1.5 s faster (weight uploads,
        // CLBlast partial shape-tuning, conv_pre weight-image pack all primed).
        // Now that the kernel binary cache (opencl_context.cpp::
        // nnopt_build_program_cached) puts JIT on disk, prewarm's main
        // remaining benefit is the weight upload + per-shape tuning — JIT
        // itself no longer pays at runtime either way. Re-enabled because
        // users who run many queries per session save ~1.5 s × N at the cost
        // of one ~7 s launch wait; for a multi-utterance demo that's a win.
        pb.environment()["NNOPT_PREWARM"] = "1"
        // Diagnostic: same as smolvlm — print [kernel_cache] HIT/MISS/SAVE lines
        // so logcat shows whether the binary cache is doing its job across app
        // restarts. Drop this once we've confirmed the cache is stable.
        pb.environment()["NNOPT_KCACHE_LOG"] = "1"
        // Phase C: stream raw int16 PCM samples on stdout instead of writing
        // an intermediate WAV file. Saves the ~50 ms write_wav on the binary
        // side AND the ~150-300 ms MediaPlayer.prepare() on the Kotlin side
        // (we feed AudioTrack directly). Per-sentence TTFS win of 200-500 ms,
        // largest on short sentences where playback prep is a bigger fraction.
        pb.environment()["NNOPT_PCM_STREAM"] = "1"

        val p = pb.start()
        process = p
        writer = PrintWriter(BufferedWriter(OutputStreamWriter(p.outputStream)), /* autoFlush = */ true)
        // Wrap stdout in DataInputStream so speak() can read exact byte counts
        // on demand. We do NOT spawn a drainer thread here — stdout is
        // request/response under mutex.withLock in speak(), so nothing else
        // contends for the pipe.
        pcmIn = DataInputStream(p.inputStream)

        thread(name = "mmstts-stderr", isDaemon = true) {
            try {
                BufferedReader(InputStreamReader(p.errorStream, Charsets.UTF_8)).useLines { lines ->
                    for (line in lines) {
                        Log.v(TAG, "stderr: $line")
                        stderrTarget?.trySend(line)
                        if (!readyDeferred.isCompleted && line.startsWith("ready.")) {
                            readyDeferred.complete(Unit)
                        }
                    }
                }
            } catch (_: Throwable) {}
        }

        withTimeout(60_000) { readyDeferred.await() }
        Log.i(TAG, "mmstts session ready")
    }

    /**
     * Synthesizes [text] and returns the resulting WAV. If [persistTo] is set,
     * the WAV is copied to that path so subsequent speak() calls won't
     * overwrite it — needed for sentence-level streaming where multiple WAVs
     * are queued for playback while later sentences are still synthesizing.
     */
    suspend fun speak(text: String, persistTo: File? = null): Result {
        if (process?.isAlive != true) throw IllegalStateException("mmstts session not running")
        if (text.isBlank()) throw IllegalArgumentException("text is empty (REPL would exit)")

        return mutex.withLock {
            // No WAV file expected from binary in PCM-stream mode. Keep the
            // stale file removed so a regression to WAV mode is obvious.
            if (wavOut.exists()) wavOut.delete()

            val stderrCh = Channel<String>(Channel.BUFFERED)
            stderrTarget = stderrCh

            val perfRegex = Regex("""done\s+audio=([0-9.]+)\s*s\s+rtf=([0-9.]+)\s+fwd=([0-9.]+)""")
            val pcmBeginRegex = Regex("""TTS_PCM_BEGIN\s+(\d+)\s+(\d+)""")
            val pcmEndRegex = Regex("""TTS_PCM_END""")
            val wavReadyRegex = Regex("""TTS_WAV_READY""")
            var audioSec = Double.NaN
            var rtf = Double.NaN
            var fwdSec = Double.NaN
            var pcmSamples: ShortArray? = null
            var pcmSampleRate = 16000
            val begin = CompletableDeferred<Pair<Int, Int>>()  // (numSamples, sampleRate)
            val done = CompletableDeferred<Unit>()

            val sendMs = System.currentTimeMillis()
            try {
                val safeText = text.replace(Regex("[\r\n]+"), " ").trim()
                writer?.println(safeText)
                writer?.flush()

                coroutineScope {
                    // Stderr line parser: looks for perf line, BEGIN marker,
                    // and either PCM_END or legacy WAV_READY (fallback).
                    val reader = launch {
                        for (line in stderrCh) {
                            perfRegex.find(line)?.let { m ->
                                audioSec = m.groupValues[1].toDouble()
                                rtf = m.groupValues[2].toDouble()
                                fwdSec = m.groupValues[3].toDouble()
                            }
                            pcmBeginRegex.find(line)?.let { m ->
                                val n = m.groupValues[1].toInt()
                                val sr = m.groupValues[2].toInt()
                                if (!begin.isCompleted) begin.complete(n to sr)
                            }
                            if (pcmEndRegex.containsMatchIn(line) ||
                                wavReadyRegex.containsMatchIn(line)) {
                                done.complete(Unit)
                                break
                            }
                        }
                    }

                    // PCM reader: blocks on stdout until we've slurped the
                    // exact byte count the binary advertised via BEGIN. Runs
                    // on Dispatchers.IO since DataInputStream.readFully blocks.
                    val pcmReader = launch(Dispatchers.IO) {
                        val (numSamples, sampleRate) = try {
                            withTimeout(45_000) { begin.await() }
                        } catch (_: TimeoutCancellationException) {
                            throw IOException("mmstts timed out before TTS_PCM_BEGIN")
                        }
                        pcmSampleRate = sampleRate
                        val nbytes = numSamples * 2
                        val raw = ByteArray(nbytes)
                        pcmIn?.readFully(raw)
                            ?: throw IOException("mmstts pcmIn is null")
                        // int16 LE → ShortArray
                        val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
                        val shorts = ShortArray(numSamples)
                        val sb = bb.asShortBuffer()
                        sb.get(shorts)
                        pcmSamples = shorts
                    }

                    try {
                        withTimeout(45_000) { done.await() }
                        pcmReader.join()
                    } catch (_: TimeoutCancellationException) {
                        throw IOException("mmstts timed out (no TTS_PCM_END after 45 s)")
                    } finally {
                        reader.cancel()
                        pcmReader.cancel()
                    }
                }
            } finally {
                stderrTarget = null
                stderrCh.close()
            }

            val samples = pcmSamples
                ?: throw IOException("mmstts produced no PCM samples")
            val ttfs = (System.currentTimeMillis() - sendMs) / 1000.0
            val pid = pidOf(process)

            // Persist a WAV alongside the PCM for the Speak tab perf panel
            // and any caller that asked for persistTo. Best-effort — if WAV
            // write fails, the PCM is still good for AudioTrack playback.
            val targetWav = persistTo ?: wavOut
            targetWav.parentFile?.mkdirs()
            try {
                writeWav(targetWav, samples, pcmSampleRate)
            } catch (t: Throwable) {
                Log.w(TAG, "writeWav fallback failed (PCM is still usable)", t)
            }

            Result(
                pcm = samples,
                sampleRate = pcmSampleRate,
                wav = targetWav,
                audioSec = audioSec,
                rtf = rtf,
                fwdSec = fwdSec,
                ttfsSec = ttfs,
                rssMb = readProcKb(pid, "VmRSS")?.let { it / 1024 },
                peakRssMb = readProcKb(pid, "VmHWM")?.let { it / 1024 },
            )
        }
    }

    /** Minimal RIFF/WAV serializer (16-bit mono PCM). Used as a backstop so
     *  legacy callers that expect a File still work. */
    private fun writeWav(out: File, samples: ShortArray, sampleRate: Int) {
        val nbytes = samples.size * 2
        val bb = ByteBuffer.allocate(44 + nbytes).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray()); bb.putInt(36 + nbytes)
        bb.put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray()); bb.putInt(16)
        bb.putShort(1)              // PCM
        bb.putShort(1)              // mono
        bb.putInt(sampleRate)
        bb.putInt(sampleRate * 2)   // byte rate
        bb.putShort(2)              // block align
        bb.putShort(16)             // bits per sample
        bb.put("data".toByteArray()); bb.putInt(nbytes)
        for (s in samples) bb.putShort(s)
        out.writeBytes(bb.array())
    }

    private fun readProcKb(pid: Long?, field: String): Int? {
        if (pid == null) {
            Log.w(TAG, "readProcKb($field): pid is null")
            return null
        }
        val prefix = "$field:"
        return try {
            val f = File("/proc/$pid/status")
            if (!f.canRead()) {
                Log.w(TAG, "readProcKb($field): /proc/$pid/status not readable")
                return null
            }
            val line = f.readLines().firstOrNull { it.startsWith(prefix) } ?: return null
            val kb = line.removePrefix(prefix).trim().split(Regex("\\s+")).firstOrNull()?.toIntOrNull()
            Log.d(TAG, "readProcKb($field, pid=$pid) = $kb kB")
            kb
        } catch (t: Throwable) {
            Log.w(TAG, "readProcKb($field) failed", t)
            null
        }
    }

    private fun pidOf(p: Process?): Long? {
        if (p == null) return null
        return try {
            (p.javaClass.getMethod("pid").invoke(p) as? Long).also {
                Log.d(TAG, "pidOf via reflection = $it")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "pid() via reflection failed", t)
            try {
                Regex("""pid=(\d+)""").find(p.toString())?.groupValues?.get(1)?.toLongOrNull()
            } catch (_: Throwable) { null }
        }
    }

    fun isAlive(): Boolean = process?.isAlive == true

    /** PID of the running subprocess, null if not running. */
    fun pid(): Long? = pidOf(process)

    /** VmRSS (resident set size) of the subprocess in KB. Returns null if
     *  the process isn't running or /proc isn't readable. */
    fun rssKb(): Int? = readProcKb(pid(), "VmRSS")

    fun stop() {
        // A blank line cleanly exits the REPL.
        try { writer?.println(""); writer?.flush() } catch (_: Throwable) {}
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

    companion object {
        private const val TAG = "MMSTTSSession"
    }
}
