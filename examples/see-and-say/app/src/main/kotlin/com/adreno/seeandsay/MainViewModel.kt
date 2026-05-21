package com.adreno.seeandsay

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import coil.imageLoader
import coil.request.ImageRequest
import com.adreno.seeandsay.audio.WavPlayer
import com.adreno.seeandsay.runner.AssetExtractor
import com.adreno.seeandsay.runner.MMSTTSSession
import com.adreno.seeandsay.runner.SmolVLMSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

sealed interface UiPhase {
    data object Initial : UiPhase
    data class Extracting(val bytesDone: Long, val bytesTotal: Long, val file: String) : UiPhase
    data class WarmingUp(val step: String) : UiPhase
    data object Ready : UiPhase
    data class Error(val message: String) : UiPhase
}

enum class AskPhase {
    Thinking,      // Prompt sent; SmolVLM is prefilling — no tokens yet.
    Streaming,     // SmolVLM is decoding; tokens are appearing.
    Synthesizing,  // SmolVLM done; MMS-TTS is generating WAV.
    Speaking,      // WAV is playing through the speaker.
    Done,          // Playback complete (or final reached and no TTS).
    Failed,        // Something blew up; see [error].
}

sealed interface CameraFlow {
    data object Preview : CameraFlow
    data class Captured(val jpeg: File) : CameraFlow
    data class Asking(
        val jpeg: File,
        val phase: AskPhase,
        val streamed: String,
        val error: String? = null,
    ) : CameraFlow
}

sealed interface SpeakFlow {
    data object Idle : SpeakFlow
    data object Synthesizing : SpeakFlow
    data class Done(
        val audioSec: Double,
        val rtf: Double,
        val fwdSec: Double,
        val ttfsSec: Double,
        val rssMb: Int?,
        val peakRssMb: Int?,
        val sentenceCount: Int,
        val e2eSec: Double,
    ) : SpeakFlow
    data class Failed(val message: String) : SpeakFlow
}

// Languages supported by MMS-TTS at runtime. Each maps to a `--lang <code>`
// flag on the binary, which reads from `weights/<code>/`. Switching language
// restarts the MMS-TTS session — the binary is single-language at launch.
enum class TtsLanguage(val code: String, val label: String, val available: Boolean) {
    English("eng", "English", true),
    Amharic("amh", "Amharic (አማርኛ)", true),
}

/** Per-query performance snapshot for the debug panel at the bottom of the camera screen. */
data class AskMetrics(
    val prefillTokens: Int? = null,
    val prefillSec: Double? = null,
    val prefillTps: Double? = null,
    val decodeTokens: Int? = null,
    val decodeSec: Double? = null,
    val decodeTps: Double? = null,
    val smolvlmTtfsSec: Double? = null,
    val ctxPos: Int? = null,
    val smolvlmRssMb: Int? = null,
    val smolvlmPeakRssMb: Int? = null,
    val ttsAudioSec: Double? = null,
    val ttsRtf: Double? = null,
    val ttsFwdSec: Double? = null,
    val ttsTtfsSec: Double? = null,
    val mmsttsRssMb: Int? = null,
    val mmsttsPeakRssMb: Int? = null,
    val e2eSec: Double? = null,
)

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val context = app.applicationContext
    private val extractor = AssetExtractor(context)
    private val smolvlm = SmolVLMSession(context)
    private val mmstts = MMSTTSSession(context)
    private val wavPlayer = WavPlayer()

    private val _phase = MutableStateFlow<UiPhase>(UiPhase.Initial)
    val phase: StateFlow<UiPhase> = _phase.asStateFlow()

    private val _camera = MutableStateFlow<CameraFlow>(CameraFlow.Preview)
    val camera: StateFlow<CameraFlow> = _camera.asStateFlow()

    private val _speak = MutableStateFlow<SpeakFlow>(SpeakFlow.Idle)
    val speak: StateFlow<SpeakFlow> = _speak.asStateFlow()

    private val _language = MutableStateFlow(TtsLanguage.English)
    val language: StateFlow<TtsLanguage> = _language.asStateFlow()

    private val _lastAnswer = MutableStateFlow<String?>(null)
    val lastAnswer: StateFlow<String?> = _lastAnswer.asStateFlow()

    private val _metrics = MutableStateFlow(AskMetrics())
    val metrics: StateFlow<AskMetrics> = _metrics.asStateFlow()

    fun setLanguage(lang: TtsLanguage) {
        if (_language.value == lang) return
        _language.value = lang
        // The MMS-TTS binary loads a single language at launch via --lang <code>.
        // Switching means: stop the current session, start a fresh one. Done off
        // the main thread because start() blocks on the binary's `ready.` marker.
        viewModelScope.launch(Dispatchers.IO) {
            try {
                mmstts.stop()
                mmstts.start(lang.code)
            } catch (t: Throwable) {
                Log.w("MainViewModel", "language switch to ${lang.code} failed", t)
            }
        }
    }

    private var askJob: Job? = null
    private var speakJob: Job? = null

    init { kickoffExtraction() }

    private fun kickoffExtraction() {
        viewModelScope.launch {
            try {
                if (!extractor.alreadyExtracted()) {
                    _phase.value = UiPhase.Extracting(0L, 0L, "")
                    withContext(Dispatchers.IO) {
                        extractor.extract { p ->
                            _phase.value = UiPhase.Extracting(p.bytesDone, p.bytesTotal, p.currentFile)
                        }
                    }
                }
                startSessions()
            } catch (t: Throwable) {
                _phase.value = UiPhase.Error(t.message ?: t.javaClass.simpleName)
            }
        }
    }

    /**
     * Start both binaries in long-lived interactive (REPL) mode. After this
     * point the processes stay resident; every Ask & Speak is a stdin write
     * + stdout/stderr read — no fork, no weight reload, no kernel rebuild.
     *
     * For SmolVLM we additionally run a 1-token vision warmup against the
     * bundled fixture so the vision-tower kernels are compiled before the
     * first user query.
     */
    private suspend fun startSessions() {
        _phase.value = UiPhase.WarmingUp("Loading SmolVLM vision-language model…")
        try {
            withContext(Dispatchers.IO) { smolvlm.start() }
        } catch (t: Throwable) {
            _phase.value = UiPhase.Error("SmolVLM start failed: ${t.message}")
            return
        }

        // Vision-tower warmup removed. The kernel binary cache
        // (opencl_context.cpp::nnopt_build_program_cached) now persists
        // compiled kernels to disk and reloads via clCreateProgramWithBinary
        // on subsequent launches — confirmed by the [kernel_cache] HIT lines
        // in logcat. The warmup's only remaining cost was running the FULL
        // vision tower on a fixture image (~10 s) which produced no benefit
        // for real queries (those run vision on the actual user image either
        // way, and image preload — see onCaptured() — already overlaps it
        // with user think time). Net win: ~10 s off app launch.

        _phase.value = UiPhase.WarmingUp("Loading MMS-TTS speech model…")
        try {
            withContext(Dispatchers.IO) { mmstts.start(_language.value.code) }
        } catch (t: Throwable) {
            _phase.value = UiPhase.Error("MMS-TTS start failed: ${t.message}")
            return
        }

        _phase.value = UiPhase.Ready
    }

    /** Lightweight liveness signal for the persistent status pill in the UI. */
    fun modelsAlive(): Pair<Boolean, Boolean> = smolvlm.isAlive() to mmstts.isAlive()

    // ---- Camera flow --------------------------------------------------

    // Path of the image whose vision-tower pass has already been kicked off
    // via SmolVLMSession.preloadImage(). When askAndSpeak() runs with this
    // jpeg, we skip resending /image (the binary already has it loaded) and
    // just send the prompt — shaves the ~10 s vision pass off user-visible
    // TTFT whenever the user spent any time composing the prompt.
    @Volatile private var preloadedJpegPath: String? = null

    fun onCaptured(jpeg: File) {
        // Kick the vision-tower preload IMMEDIATELY — every ms we wait here is
        // a ms added to user-visible TTFT. We fire BEFORE flipping the UI
        // state and BEFORE Coil decode so the Adreno gets the vision pipe in
        // flight as early as possible. White-flash mitigation is the
        // NNOPT_GPU_YIELD per-layer clFinish in vision_pipe (set in
        // SmolVLMSession.start), not a UI-side delay. The View's
        // LaunchedEffect(jpeg) calls kickPreloadFor again as a backstop, but
        // the second call is a no-op (de-duped by preloadedJpegPath).
        kickPreloadFor(jpeg)
        // Pre-decode JPEG into Coil's cache + flip UI state in parallel.
        viewModelScope.launch {
            try {
                val req = ImageRequest.Builder(context).data(jpeg).build()
                context.imageLoader.execute(req)
            } catch (_: Throwable) { /* best-effort */ }
            _camera.value = CameraFlow.Captured(jpeg)
        }
    }

    /**
     * Kick off the vision-tower preload for [jpeg] in the background. Idempotent
     * (no-op if already preloading the same jpeg) so the View can call this
     * freely on every recomposition of CapturedView.
     */
    fun kickPreloadFor(jpeg: File) {
        val path = jpeg.absolutePath
        if (preloadedJpegPath == path) return  // already preloading or done
        preloadedJpegPath = path
        viewModelScope.launch(Dispatchers.IO) {
            try {
                smolvlm.preloadImage(path)
            } catch (t: Throwable) {
                Log.w("MainViewModel", "preloadImage failed (will fall back at askAndSpeak)", t)
            }
        }
    }

    fun onRetake() {
        askJob?.cancel()
        preloadedJpegPath = null
        _camera.value = CameraFlow.Preview
    }

    fun askAndSpeak(prompt: String) {
        val state = _camera.value
        val jpeg = (state as? CameraFlow.Captured)?.jpeg
            ?: (state as? CameraFlow.Asking)?.jpeg
            ?: return
        // Decide fresh-conversation vs follow-up vs preloaded-image-first-turn:
        //   Captured + preloadedJpegPath == jpeg.absolutePath
        //     → vision tower already kicked off in onCaptured; just send prompt
        //   Captured (no preload match)
        //     → fresh /image + reset (today's behaviour)
        //   Asking.Done
        //     → continuing same image, no /image, keep KV
        // We treat the preloaded-first-turn case as `isFollowUp = true` because
        // both paths send only the prompt line; the binary's interactive loop
        // distinguishes by its internal total_pos counter (==0 → build_vlm_prompt
        // with image_present=true, >0 → build_vlm_followup_user_turn).
        val preloadHit = state is CameraFlow.Captured &&
                         preloadedJpegPath == jpeg.absolutePath
        val isFollowUp = (state is CameraFlow.Asking && state.phase == AskPhase.Done) ||
                         preloadHit
        wavPlayer.stop()
        askJob?.cancel()
        _lastAnswer.value = null
        _camera.value = CameraFlow.Asking(jpeg, phase = AskPhase.Thinking, streamed = "")

        val askStart = System.currentTimeMillis()
        _metrics.value = AskMetrics()

        askJob = viewModelScope.launch(Dispatchers.Default) {
            val sb = StringBuilder()
            var finalText: String? = null
            var lastEmitMs = 0L
            val streamThrottleMs = 500L
            try {
                val flow = if (isFollowUp) smolvlm.ask(prompt)
                           else smolvlm.askWithImage(jpeg.absolutePath, prompt)
                flow.collect { out ->
                    when (out) {
                        is SmolVLMSession.Output.Token -> {
                            sb.append(out.text)
                            val now = System.currentTimeMillis()
                            // Emit on throttle window OR on a natural pause point
                            // (sentence end / whitespace burst) — keeps the UI
                            // alive without losing visual cadence.
                            val natural = out.text.contains('.') || out.text.contains('\n')
                            if (natural || now - lastEmitMs >= streamThrottleMs) {
                                _camera.value = CameraFlow.Asking(jpeg, AskPhase.Streaming, sb.toString())
                                lastEmitMs = now
                            }
                        }
                        is SmolVLMSession.Output.Final -> {
                            finalText = out.text
                            _camera.value = CameraFlow.Asking(jpeg, AskPhase.Synthesizing, out.text)
                        }
                        is SmolVLMSession.Output.Stats -> {
                            _metrics.value = _metrics.value.copy(
                                prefillTokens = out.prefillTokens,
                                prefillSec = out.prefillSec,
                                prefillTps = out.prefillTps,
                                decodeTokens = out.decodeTokens,
                                decodeSec = out.decodeSec,
                                decodeTps = out.decodeTps,
                                smolvlmTtfsSec = out.ttfsSec,
                                ctxPos = out.ctxPos,
                                smolvlmRssMb = out.rssMb,
                                smolvlmPeakRssMb = out.peakRssMb,
                            )
                        }
                        is SmolVLMSession.Output.Failed -> {
                            _camera.value = CameraFlow.Asking(jpeg, AskPhase.Failed, sb.toString(), error = out.message)
                        }
                    }
                }
            } catch (t: Throwable) {
                _camera.value = CameraFlow.Asking(jpeg, AskPhase.Failed, sb.toString(), error = t.message)
                return@launch
            }

            val finalShown = finalText ?: sb.toString().trim()
            _lastAnswer.value = finalShown
            _metrics.value = _metrics.value.copy(
                e2eSec = (System.currentTimeMillis() - askStart) / 1000.0,
            )
            // TTS is disabled in the Camera tab — text answer only. Use the
            // Speak tab if you want the answer read aloud (paste the text or
            // use the dedicated Speak-the-last-answer affordance below).
            _camera.value = CameraFlow.Asking(jpeg, AskPhase.Done, finalShown)
        }
    }

    /**
     * Sentence-level streaming TTS: split [text] on `.?!`, synthesize each
     * sentence sequentially, but ENQUEUE for playback the moment each WAV is
     * ready. The user hears sentence 1 while sentence 2 is still being
     * synthesized — perceived latency drops by ~5–10 s on paragraph answers.
     *
     * Each sentence's WAV is persisted to a unique path so the next
     * synthesis's overwrite of output.wav doesn't corrupt queued playback.
     */
    private suspend fun speakStreamed(jpeg: File, text: String, askStart: Long) {
        val sentences = splitSentences(text)
        val cwd = File(context.filesDir, "mmstts")
        val audioDir = File(cwd, "audio_queue").also { it.mkdirs() }

        // Clean any stale fragments from a prior ask so the directory doesn't grow.
        audioDir.listFiles()?.forEach { it.delete() }

        _camera.value = CameraFlow.Asking(jpeg, AskPhase.Synthesizing, text)
        wavPlayer.stop()  // ensure queue is fresh

        var firstAudio = true
        var lastResult: com.adreno.seeandsay.runner.MMSTTSSession.Result? = null
        for ((index, sentence) in sentences.withIndex()) {
            val target = File(audioDir, "utt_${index.toString().padStart(3, '0')}.wav")
            val res = mmstts.speak(sentence, persistTo = target)
            lastResult = res

            // Update metrics with the latest sentence's numbers (cumulative
            // numbers would be more accurate but live-overwrite is OK for v1).
            _metrics.value = _metrics.value.copy(
                ttsAudioSec = res.audioSec,
                ttsRtf = res.rtf,
                ttsFwdSec = res.fwdSec,
                ttsTtfsSec = res.ttfsSec,
                mmsttsRssMb = res.rssMb,
                mmsttsPeakRssMb = res.peakRssMb,
            )

            // Enqueue this sentence. PCM path (AudioTrack) when available —
            // skips MediaPlayer.prepare()'s ~150-300 ms per sentence and
            // starts audio in ~10-20 ms. The first sentence flips us into
            // "Speaking" — subsequent ones queue behind it.
            val pcm = res.pcm
            val onDone: () -> Unit = {
                if (index == sentences.lastIndex) {
                    val cur = _camera.value
                    if (cur is CameraFlow.Asking) _camera.value = cur.copy(phase = AskPhase.Done)
                }
            }
            if (pcm != null) {
                wavPlayer.enqueuePcm(pcm, res.sampleRate, onComplete = onDone)
            } else {
                wavPlayer.enqueue(res.wav, onComplete = onDone)
            }
            if (firstAudio) {
                _camera.value = CameraFlow.Asking(jpeg, AskPhase.Speaking, text)
                firstAudio = false
            }
        }
        _metrics.value = _metrics.value.copy(
            e2eSec = (System.currentTimeMillis() - askStart) / 1000.0,
        )
    }

    private fun splitSentences(text: String): List<String> {
        // Split only on sentence-ending punctuation. Sub-sentence (comma)
        // splitting was tried and reverted: at RTF ≈ 1.45 the next chunk's
        // synthesis (~1.5s of wall per ~1s of audio) always finishes AFTER
        // the current chunk's playback, producing audible mid-clause silence.
        // Sentence boundaries are natural prosodic pauses, so the same gap
        // doesn't sound broken — it sounds like a breath.
        val parts = text.split(Regex("""(?<=[.!?])\s+"""))
            .map { it.trim() }
            .filter { it.isNotEmpty() }
        if (parts.size <= 1) return parts
        // Merge tiny opener fragments (< 6 chars: "Hi.", "Hello!", "Wow.") into
        // the next sentence — a 1-word opener plays in ~0.5s and the gap to
        // sentence 2 dominates the listening experience. 6 chars also keeps
        // the per-sentence text_encoder overhead worthwhile.
        val merged = mutableListOf<StringBuilder>()
        merged.add(StringBuilder())
        for (p in parts) {
            if (merged.last().isEmpty()) merged.last().append(p)
            else if (merged.last().length < 6) merged.last().append(' ').append(p)
            else merged.add(StringBuilder(p))
        }
        return merged.map { it.toString() }
    }

    /** Replay TTS on the last SmolVLM answer without re-running the VLM. */
    fun replayLastAnswer() {
        val text = _lastAnswer.value ?: return
        val state = _camera.value as? CameraFlow.Asking ?: return
        wavPlayer.stop()
        askJob?.cancel()
        askJob = viewModelScope.launch(Dispatchers.Default) {
            _camera.value = state.copy(phase = AskPhase.Synthesizing)
            try {
                speakStreamed(state.jpeg, text, System.currentTimeMillis())
            } catch (t: Throwable) {
                _camera.value = state.copy(phase = AskPhase.Failed, error = "TTS failed: ${t.message}")
            }
        }
    }

    // ---- Speak tab flow -----------------------------------------------

    fun speak(text: String) {
        if (text.isBlank()) return
        speakJob?.cancel()
        wavPlayer.stop()
        _speak.value = SpeakFlow.Synthesizing
        speakJob = viewModelScope.launch(Dispatchers.Default) {
            val startMs = System.currentTimeMillis()
            try {
                val sentences = splitSentences(text)
                val cwd = File(context.filesDir, "mmstts")
                val audioDir = File(cwd, "speak_queue").also { it.mkdirs() }
                audioDir.listFiles()?.forEach { it.delete() }

                // Pre-buffer policy: ZERO. Sentence 1 plays the instant it's
                // synthesized; subsequent sentences synthesize in parallel and
                // queue behind it. Previously this was 2, which doubled TTFS
                // for any 2+ sentence response — at the current RTF ≈ 1.45 the
                // sentence-2 synthesis usually completes before sentence-1
                // playback finishes (a 1.45-second synthesis covers 1 second
                // of speech, so anything ≥ 1.5s long buys us enough headroom).
                // If sentence 2 runs over, we eat a small audible gap instead
                // of a guaranteed multi-second wall before any sound plays.
                val prebufferTarget = 0
                var firstWav: com.adreno.seeandsay.runner.MMSTTSSession.Result? = null
                var lastRes: com.adreno.seeandsay.runner.MMSTTSSession.Result? = null
                // Aggregate metrics across all sentences for the live perf panel.
                var totalAudioSec = 0.0
                var totalFwdSec = 0.0
                var maxPeakMb: Int? = null
                // Hold Results in order so we can drain them through the player
                // after the pre-buffer window closes. PCM payloads live in
                // memory (~80 KB per second of audio) so this is cheap.
                val buffered = mutableListOf<com.adreno.seeandsay.runner.MMSTTSSession.Result>()

                fun playOne(r: com.adreno.seeandsay.runner.MMSTTSSession.Result, idx: Int) {
                    val cb = onSentenceComplete(
                        idx, sentences.lastIndex, totalAudioSec, totalFwdSec,
                        lastRes, maxPeakMb, sentences.size, startMs,
                    )
                    val pcm = r.pcm
                    if (pcm != null) wavPlayer.enqueuePcm(pcm, r.sampleRate, cb)
                    else            wavPlayer.enqueue(r.wav, cb)
                }

                for ((index, sentence) in sentences.withIndex()) {
                    val target = File(audioDir, "speak_${index.toString().padStart(3, '0')}.wav")
                    val res = mmstts.speak(sentence, persistTo = target)
                    if (firstWav == null) firstWav = res
                    lastRes = res
                    totalAudioSec += res.audioSec
                    totalFwdSec += res.fwdSec
                    maxPeakMb = maxOf(maxPeakMb ?: 0, res.peakRssMb ?: 0).takeIf { it > 0 }
                    buffered.add(res)

                    if (index < prebufferTarget) {
                        // Still pre-buffering. Don't enqueue yet.
                    } else if (index == prebufferTarget) {
                        // First post-buffer sentence — drain everything we have.
                        for ((i, r) in buffered.withIndex()) playOne(r, i)
                    } else {
                        playOne(res, index)
                    }
                }

                // Edge case: sentences.size < prebufferTarget+1 meant we never
                // hit the drain branch. Flush whatever we have now.
                if (sentences.size <= prebufferTarget) {
                    for ((i, r) in buffered.withIndex()) playOne(r, i)
                }
            } catch (t: Throwable) {
                _speak.value = SpeakFlow.Failed(t.message ?: t.javaClass.simpleName)
            }
        }
    }

    private fun onSentenceComplete(
        index: Int,
        lastIndex: Int,
        totalAudioSec: Double,
        totalFwdSec: Double,
        lastRes: com.adreno.seeandsay.runner.MMSTTSSession.Result?,
        maxPeakMb: Int?,
        sentenceCount: Int,
        startMs: Long,
    ): () -> Unit = {
        if (index == lastIndex) {
            val r = lastRes
            _speak.value = if (r != null) {
                val totalRtf = if (totalAudioSec > 0) totalFwdSec / totalAudioSec else r.rtf
                SpeakFlow.Done(
                    audioSec = totalAudioSec,
                    rtf = totalRtf,
                    fwdSec = totalFwdSec,
                    ttfsSec = r.ttfsSec,
                    rssMb = r.rssMb,
                    peakRssMb = maxPeakMb ?: r.peakRssMb,
                    sentenceCount = sentenceCount,
                    e2eSec = (System.currentTimeMillis() - startMs) / 1000.0,
                )
            } else SpeakFlow.Idle
        }
    }

    fun stopSpeak() {
        speakJob?.cancel()
        wavPlayer.stop()
        _speak.value = SpeakFlow.Idle
    }

    /**
     * Trim model-generated text down to something MMS-TTS can synthesize without
     * pegging the GPU for half a minute. Prefers a sentence boundary inside the
     * cap range; falls back to a hard cut + ellipsis.
     */
    private fun capForTts(text: String, maxChars: Int = 240): String {
        val t = text.trim()
        if (t.length <= maxChars) return t
        // Find the last sentence-ending punctuation within the cap window.
        val window = t.take(maxChars)
        val lastBoundary = listOf('.', '!', '?', ';', '\n')
            .map { window.lastIndexOf(it) }
            .max()
        return if (lastBoundary >= maxChars / 2) {
            t.substring(0, lastBoundary + 1)
        } else {
            t.substring(0, maxChars).trimEnd() + "…"
        }
    }

    override fun onCleared() {
        askJob?.cancel()
        speakJob?.cancel()
        wavPlayer.stop()
        smolvlm.stop()
        mmstts.stop()
        super.onCleared()
    }
}
