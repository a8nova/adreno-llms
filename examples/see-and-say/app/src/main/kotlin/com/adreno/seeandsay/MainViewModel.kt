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
    /**
     * Synthesizing one sentence of a multi-sentence speak() job.
     * Reported as 1-based for display; e.g. (3, 8) means "sentence 3 of 8 is
     * currently being synthesized." Lets the UI show progress instead of
     * leaving the user staring at a frozen "Synthesizing…" for 20+ seconds.
     */
    data class Synthesizing(val sentenceIndex: Int = 0, val sentenceTotal: Int = 0) : SpeakFlow
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

/**
 * Languages supported at runtime — dynamic. Bundled defaults (eng + amh)
 * are extracted from the APK assets on first launch; any other code shows
 * up here AFTER the user downloads its language pack via LanguagePickerScreen.
 *
 * `code` maps to `--lang <code>` on the binary, which reads from
 * `<filesDir>/mmstts/weights/<code>/`. `label` is the display string in the
 * Settings radio list. `available` is true once the language's
 * `tokenizer_vocab.bin` exists on disk; false means greyed-out (shouldn't
 * happen for the dynamic list — those entries are filtered out).
 */
data class TtsLanguage(val code: String, val label: String, val available: Boolean)

/**
 * SmolVLM sampler knobs surfaced to the user via the Configurations dialog.
 * These are passed as CLI flags to the binary at process start — changing any
 * of them restarts the SmolVLM session (kills the process and re-launches
 * with new args). Defaults match the binary's own greedy defaults so that the
 * dialog "matches reality" before the user touches it.
 */
data class SamplerSettings(
    val maxNewTokens: Int = 128,  // bumped from 64 — 64 truncates SmolVLM mid-sentence on detailed prompts
    val topK: Int = 1,            // greedy default; > 1 enables stochastic sampling
    val topP: Float = 1.0f,
    val temperature: Float = 0.0f,
    // Fast (384, 36 image tokens, ~44% less vision compute) vs Quality (512,
    // 64 image tokens, matches upstream training). Passed as --image-size to
    // the binary. The bin ships both pos-embed variants so switching at
    // runtime needs no re-conversion.
    val imageSize: Int = 384,
)

/**
 * SmolVLM's chat template (`<|im_start|>User:<image>...<end_of_utterance>\nAssistant:`)
 * has no native system role — idefics3 wasn't trained with one. We approximate
 * it by prepending the system text to the user prompt in Kotlin: the model
 * sees `<system_prompt>\n\n<user_question>` as one user turn. Effect is
 * looser than Gemma's structured system slot, but for SmolVLM-256M this nudges
 * verbosity / format reliably enough to be useful.
 */
const val DEFAULT_SYSTEM_PROMPT: String =
    "You are a careful visual describer. For every image, list what you see in detail: " +
    "the main subjects, their colors and shapes, any text or labels, the setting or background, " +
    "and any notable details. Be specific and concrete, and answer the user's question completely."

/**
 * MMS-TTS / VITS knobs surfaced via the Configurations dialog. Passed as
 * --noise-scale / --noise-scale-w / --length-scale to the mms-tts binary at
 * process start; changing them restarts the TTS session.
 *
 * Defaults match HuggingFace VitsModel.generate(): noise_scale=0.667 (prior
 * z variance), noise_scale_w=0.8 (duration-predictor latent variance),
 * length_scale=1.0 (speech-rate multiplier — < 1.0 faster, > 1.0 slower).
 */
data class TtsSettings(
    val noiseScale: Float = 0.667f,
    val noiseScaleW: Float = 0.8f,
    val lengthScale: Float = 1.0f,
    /**
     * Playback pacing for multi-sentence text.
     *  true  — synthesize ALL sentences first, then play continuously (no
     *          mid-utterance silence gaps). Costs an upfront wait roughly
     *          equal to total_audio_seconds × RTF.
     *  false — start playing sentence 1 the moment it's ready, while later
     *          sentences are still synthesizing. With RTF > 1 (common on
     *          mobile) you hear noticeable silence between sentences.
     */
    val prebufferAll: Boolean = true,
)

/**
 * Snapshot of the device's OpenCL device-info block (printed by
 * opencl_context.cpp at process start, captured from SmolVLM's stderr stream).
 * Used to render the "Device & compatibility" section in Settings and decide
 * whether the current hardware can run our kernels.
 */
data class DeviceInfo(
    val platform: String,
    val device: String,
    val openclVersion: String,
    val driver: String,
    val computeUnits: Int,
    val maxClockMhz: Int,
    val maxWorkgroup: Int,
    val globalMemMb: Int,
    val localMemKb: Int,
    val hasFp16: Boolean,
    val hasQcomPerfHint: Boolean,
    val hasQcomRecordableQueues: Boolean,
    val hasQcomDotProduct8: Boolean,
) {
    /** True iff the device meets every hard requirement to run our fp16 build. */
    val isCompatible: Boolean
        get() = hasFp16 &&
                maxWorkgroup >= 64 &&
                globalMemMb >= 1024 &&
                localMemKb >= 12

    /** True iff the device is an Adreno (which is what our kernels are tuned for). */
    val isAdreno: Boolean get() = device.contains("Adreno", ignoreCase = true)
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

    private val _language = MutableStateFlow(TtsLanguage("eng", "English", true))
    val language: StateFlow<TtsLanguage> = _language.asStateFlow()

    // Language registry + downloader. Public so the Compose screen can wire
    // directly to its StateFlow; ViewModel doesn't need to mediate the UI.
    val languageRegistry = com.adreno.seeandsay.runner.LanguageRegistry(context)

    /**
     * Installed-or-bundled languages, projected into the form the Settings
     * radio list consumes. Derived from LanguageRegistry — adding a language
     * via download automatically expands this list.
     */
    val installedLanguages: StateFlow<List<TtsLanguage>> = run {
        val flow = kotlinx.coroutines.flow.MutableStateFlow<List<TtsLanguage>>(
            listOf(TtsLanguage("eng", "English", true), TtsLanguage("amh", "Amharic (አማርኛ)", true)),
        )
        viewModelScope.launch {
            languageRegistry.entries.collect { entries ->
                flow.value = entries
                    .filter { e ->
                        e.status is com.adreno.seeandsay.runner.LanguageRegistry.LangEntry.Status.Bundled ||
                        e.status is com.adreno.seeandsay.runner.LanguageRegistry.LangEntry.Status.Installed
                    }
                    .map { e ->
                        val pretty = buildString {
                            append(e.name)
                            if (e.nativeName.isNotBlank() && e.nativeName != e.name) {
                                append(" (").append(e.nativeName).append(")")
                            }
                        }
                        TtsLanguage(e.code, pretty, true)
                    }
                    .sortedBy { it.label.lowercase() }
            }
        }
        flow
    }

    private val _lastAnswer = MutableStateFlow<String?>(null)
    val lastAnswer: StateFlow<String?> = _lastAnswer.asStateFlow()

    private val _metrics = MutableStateFlow(AskMetrics())
    val metrics: StateFlow<AskMetrics> = _metrics.asStateFlow()

    private val _samplerSettings = MutableStateFlow(SamplerSettings())
    val samplerSettings: StateFlow<SamplerSettings> = _samplerSettings.asStateFlow()

    private val _ttsSettings = MutableStateFlow(TtsSettings())
    val ttsSettings: StateFlow<TtsSettings> = _ttsSettings.asStateFlow()

    // Populated once by SmolVLMSession when its stderr emits the OpenCL device
    // block at process start. SmolVLM runs first so we always have this by the
    // time Settings is opened.
    private val _deviceInfo = MutableStateFlow<DeviceInfo?>(null)
    val deviceInfo: StateFlow<DeviceInfo?> = _deviceInfo.asStateFlow()

    /**
     * Live VmRSS readings for both subprocesses, polled every ~1.2 s. Drives
     * the top-bar status pill so the user can watch memory move as inference
     * runs. Reads /proc/<pid>/status — a ~1 ms file read, free vs. inference.
     */
    data class LiveRss(val smolvlmMb: Int?, val mmsttsMb: Int?)

    private val _liveRss = MutableStateFlow(LiveRss(null, null))
    val liveRss: StateFlow<LiveRss> = _liveRss.asStateFlow()

    init {
        smolvlm.onDeviceInfo = { info -> _deviceInfo.value = info }
        // Live-RSS poller. Lifecycle-bound via viewModelScope; cancels with
        // the ViewModel. Quiet when either subprocess is down (returns null).
        viewModelScope.launch(Dispatchers.IO) {
            while (true) {
                val sv = smolvlm.rssKb()?.let { it / 1024 }
                val mt = mmstts.rssKb()?.let { it / 1024 }
                _liveRss.value = LiveRss(sv, mt)
                kotlinx.coroutines.delay(1200)
            }
        }
    }

    // Plain text prepended to every user prompt sent to SmolVLM. Per-query
    // (no binary restart on change) — see askAndSpeak() for the prepend site.
    private val _systemPrompt = MutableStateFlow(DEFAULT_SYSTEM_PROMPT)
    val systemPrompt: StateFlow<String> = _systemPrompt.asStateFlow()

    fun updateSystemPrompt(text: String) { _systemPrompt.value = text }

    /**
     * Apply new sampler settings. If they differ from the current values, this
     * kicks off a SmolVLM process restart in the background (the binary takes
     * --temperature / --top-k / --top-p / --max-tokens at launch only).
     * During the restart the UI flips to UiPhase.WarmingUp so the loader
     * screen reappears with the chip+fire animation. Preloaded image state is
     * cleared because the new process has no KV cache.
     */
    fun updateSamplerSettings(new: SamplerSettings) {
        if (_samplerSettings.value == new) return
        _samplerSettings.value = new
        viewModelScope.launch(Dispatchers.IO) {
            _phase.value = UiPhase.WarmingUp("Applying new sampler settings…")
            preloadedJpegPath = null
            try {
                smolvlm.stop()
                smolvlm.start(new)
                _phase.value = UiPhase.Ready
            } catch (t: Throwable) {
                _phase.value = UiPhase.Error("SmolVLM restart failed: ${t.message}")
            }
        }
    }

    /**
     * Apply new MMS-TTS / VITS knobs. Restarts the mms-tts session in the
     * background (length_scale + noise scales are passed at process start);
     * the SmolVLM session is untouched.
     */
    fun updateTtsSettings(new: TtsSettings) {
        if (_ttsSettings.value == new) return
        _ttsSettings.value = new
        viewModelScope.launch(Dispatchers.IO) {
            _phase.value = UiPhase.WarmingUp("Applying new TTS settings…")
            try {
                mmstts.stop()
                mmstts.start(_language.value.code, new)
                _phase.value = UiPhase.Ready
            } catch (t: Throwable) {
                _phase.value = UiPhase.Error("MMS-TTS restart failed: ${t.message}")
            }
        }
    }

    /** Convenience overload — accepts a raw code string from the Language Picker. */
    fun setLanguage(code: String) {
        val installed = installedLanguages.value
        val lang = installed.firstOrNull { it.code == code }
            ?: TtsLanguage(code, code, true)
        setLanguage(lang)
    }

    fun setLanguage(lang: TtsLanguage) {
        if (_language.value == lang) return
        val previous = _language.value
        _language.value = lang
        // The MMS-TTS binary loads a single language at launch via --lang <code>.
        // Switching means: stop the current session, start a fresh one. Done off
        // the main thread because start() blocks on the binary's `ready.` marker.
        // If start fails (missing weights for this language, OpenCL OOM, …) we
        // surface the error to the UI AND revert _language so the user can
        // pick a working option — otherwise the next Speak would fire
        // `mmstts session not running` with no explanation.
        viewModelScope.launch(Dispatchers.IO) {
            _phase.value = UiPhase.WarmingUp("Loading ${lang.label}…")
            try {
                mmstts.stop()
                mmstts.start(lang.code, _ttsSettings.value)
                _phase.value = UiPhase.Ready
            } catch (t: Throwable) {
                Log.w("MainViewModel", "language switch to ${lang.code} failed", t)
                _language.value = previous
                // Best-effort: bring the previous language's session back so
                // the rest of the app keeps working.
                try { mmstts.stop(); mmstts.start(previous.code, _ttsSettings.value) }
                catch (_: Throwable) { /* nothing more we can do here */ }
                _phase.value = UiPhase.Error(
                    "Failed to load ${lang.label} — weights/${lang.code}/ probably missing. " +
                    "Reverted to ${previous.label}.",
                )
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
            withContext(Dispatchers.IO) { smolvlm.start(_samplerSettings.value) }
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
            withContext(Dispatchers.IO) { mmstts.start(_language.value.code, _ttsSettings.value) }
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
            // Prepend the system prompt ONLY to the first turn (build_vlm_prompt
            // path). Follow-ups already have the system context in the KV cache
            // from turn 1 — re-adding it on every follow-up wastes context.
            val sys = _systemPrompt.value.trim()
            val effectivePrompt = if (!isFollowUp && sys.isNotEmpty())
                "$sys\n\n$prompt"
            else
                prompt
            try {
                val flow = if (isFollowUp) smolvlm.ask(effectivePrompt)
                           else smolvlm.askWithImage(jpeg.absolutePath, effectivePrompt)
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
            // Camera mode = vision-only. Auto-TTS was previously firing here,
            // but the user wants the camera tab to stop at "text answer
            // visible" — TTS lives in the dedicated Text-to-speech screen.
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
        _speak.value = SpeakFlow.Synthesizing(0, 0)
        speakJob = viewModelScope.launch(Dispatchers.Default) {
            val startMs = System.currentTimeMillis()
            try {
                val sentences = splitSentences(text)
                val cwd = File(context.filesDir, "mmstts")
                val audioDir = File(cwd, "speak_queue").also { it.mkdirs() }
                audioDir.listFiles()?.forEach { it.delete() }

                // Pre-buffer policy: depends on tts.prebufferAll.
                //   true  → buffer ALL sentences first, then play continuously
                //           (no mid-utterance silence — recommended whenever
                //           RTF > 1, which is the case on every Adreno 6xx).
                //   false → drain sentence 1 the instant it's ready; later
                //           sentences may not finish synthesizing in time and
                //           you hear a gap before each one.
                // Hard-coded "0" was the old default and is the source of the
                // "huge silence gaps mid-output" complaint.
                val prebufferTarget = if (_ttsSettings.value.prebufferAll)
                    sentences.size  // never satisfies index < target → all sentences buffer
                else
                    0
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
                    // Surface per-sentence progress so the user knows the job
                    // is alive during the multi-second synthesis windows.
                    _speak.value = SpeakFlow.Synthesizing(index + 1, sentences.size)
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
        languageRegistry.shutdown()
        super.onCleared()
    }
}
