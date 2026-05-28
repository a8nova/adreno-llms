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

/**
 * Unified chat: one screen, text + optional image attachment. Each turn is a
 * [ChatTurn] in [chatHistory]. Mid-conversation image attachment triggers a
 * silent /reset on the binary and pushes a SYSTEM divider into history so the
 * UI can render "— new conversation —".
 */
sealed interface ChatFlow {
    data object Idle : ChatFlow
    /** SmolVLM is prefilling (image pass + image-token expansion). No tokens yet. */
    data object Thinking : ChatFlow
    /** SmolVLM is decoding into the current assistant turn (history.last). */
    data object Streaming : ChatFlow
    /** Generation done; optional TTS in progress for the last assistant turn. */
    data object Speaking : ChatFlow
    data class Failed(val message: String) : ChatFlow
}

enum class ChatRole { USER, ASSISTANT, SYSTEM }

data class ChatTurn(
    val role: ChatRole,
    val text: String,
    /** Set on USER turns that attached an image. Path to local JPEG. */
    val imagePath: String? = null,
)

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

    /**
     * One-time CC-BY-NC 4.0 license acceptance for Facebook's MMS-TTS weights
     * (both bundled and downloaded language packs). SmolVLM is Apache 2.0 —
     * no acceptance needed. Stored in SharedPreferences, sticky across app
     * updates. Increment LICENSE_KEY's version suffix to force re-acceptance
     * if the license text materially changes.
     */
    private val prefs = context.getSharedPreferences("license_prefs", android.content.Context.MODE_PRIVATE)
    private val _licenseAccepted = MutableStateFlow(prefs.getBoolean(LICENSE_KEY, false))
    val licenseAccepted: StateFlow<Boolean> = _licenseAccepted.asStateFlow()

    fun acceptLicense() {
        prefs.edit().putBoolean(LICENSE_KEY, true).apply()
        _licenseAccepted.value = true
    }

    companion object {
        private const val LICENSE_KEY = "mms_tts_cc_by_nc_4_0_accepted_v1"
    }

    /**
     * Exponential moving average of the TTS real-time factor (synthesis_time /
     * audio_duration). Updated after every `mmstts.speak()` call. Used to
     * compute the adaptive prebuffer target: `ceil(measuredRtf)` sentences
     * must be synthesized before playback starts so the playback runway
     * always stays ahead of synthesis. Default 6.0 (conservative for Adreno
     * 6xx) until the first measurement arrives.
     */
    @Volatile private var measuredRtf: Double = 6.0
    private val rtfAlpha = 0.3  // EMA smoothing factor — recent calls weigh more

    private fun updateRtf(result: MMSTTSSession.Result) {
        if (result.rtf > 0 && result.rtf.isFinite()) {
            measuredRtf = rtfAlpha * result.rtf + (1 - rtfAlpha) * measuredRtf
        }
    }

    /** Measured RTF exposed for the Configurations dialog info line. */
    val currentRtf: Double get() = measuredRtf

    /**
     * Adaptive prebuffer: synthesize this many sentences before starting
     * playback. At RTF r, each chunk plays in d seconds but needs r*d to
     * synthesize. Prebuffering ceil(r) chunks creates enough runway that
     * subsequent chunks finish before the buffer drains. Capped to
     * [sentenceCount] (when the text is short, prebuffer everything).
     */
    private fun adaptivePrebufferTarget(sentenceCount: Int): Int =
        kotlin.math.ceil(measuredRtf).toInt().coerceIn(1, sentenceCount)

    private val _phase = MutableStateFlow<UiPhase>(UiPhase.Initial)
    val phase: StateFlow<UiPhase> = _phase.asStateFlow()

    private val _camera = MutableStateFlow<CameraFlow>(CameraFlow.Preview)
    val camera: StateFlow<CameraFlow> = _camera.asStateFlow()

    // ---- Unified chat state -------------------------------------------
    private val _chat = MutableStateFlow<ChatFlow>(ChatFlow.Idle)
    val chat: StateFlow<ChatFlow> = _chat.asStateFlow()

    private val _chatHistory = MutableStateFlow<List<ChatTurn>>(emptyList())
    val chatHistory: StateFlow<List<ChatTurn>> = _chatHistory.asStateFlow()

    /** Auto-TTS the assistant response in chat. Off by default — text chat is iterative. */
    private val _chatTtsEnabled = MutableStateFlow(false)
    val chatTtsEnabled: StateFlow<Boolean> = _chatTtsEnabled.asStateFlow()

    /** Pending image attachment composed via the "+" button. Cleared on send or cancel. */
    private val _pendingAttachment = MutableStateFlow<File?>(null)
    val pendingAttachment: StateFlow<File?> = _pendingAttachment.asStateFlow()

    private var chatJob: Job? = null

    private val _speak = MutableStateFlow<SpeakFlow>(SpeakFlow.Idle)
    val speak: StateFlow<SpeakFlow> = _speak.asStateFlow()

    private val _ttsLoading = MutableStateFlow<String?>(null)
    val ttsLoading: StateFlow<String?> = _ttsLoading.asStateFlow()

    private val _language = MutableStateFlow<TtsLanguage?>(null)
    val language: StateFlow<TtsLanguage?> = _language.asStateFlow()

    // Language registry + downloader. Public so the Compose screen can wire
    // directly to its StateFlow; ViewModel doesn't need to mediate the UI.
    val languageRegistry = com.adreno.seeandsay.runner.LanguageRegistry(context)

    /**
     * Installed-or-bundled languages, projected into the form the Settings
     * radio list consumes. Derived from LanguageRegistry — adding a language
     * via download automatically expands this list.
     */
    val installedLanguages: StateFlow<List<TtsLanguage>> = run {
        val flow = kotlinx.coroutines.flow.MutableStateFlow<List<TtsLanguage>>(emptyList())
        viewModelScope.launch {
            languageRegistry.entries.collect { entries ->
                flow.value = entries
                    .filter { e ->
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

                // Auto-select first installed language if none is selected.
                val current = _language.value
                if (current == null && flow.value.isNotEmpty()) {
                    val first = flow.value.first()
                    _language.value = first
                    // Start TTS for the auto-selected language.
                    ttsStartJob?.cancel()
                    ttsStartJob = launch(Dispatchers.IO) {
                        _ttsLoading.value = "Loading ${first.label}…"
                        try { mmstts.start(first.code, _ttsSettings.value, prewarm = false) }
                        catch (t: Throwable) { Log.w("MainViewModel", "auto-start TTS failed", t) }
                        _ttsLoading.value = null
                    }
                }
                // Clear selection if current language was uninstalled.
                if (current != null && flow.value.none { it.code == current.code }) {
                    mmstts.stop()
                    _language.value = flow.value.firstOrNull()
                }
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
        val lang = _language.value ?: return
        ttsStartJob?.cancel()
        ttsStartJob = viewModelScope.launch(Dispatchers.IO) {
            _ttsLoading.value = "Applying TTS settings…"
            try {
                mmstts.stop()
                mmstts.start(lang.code, new, prewarm = false)
            } catch (t: Throwable) {
                Log.w("MainViewModel", "TTS settings restart failed: ${t.message}", t)
            }
            _ttsLoading.value = null
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
        ttsStartJob?.cancel()
        ttsStartJob = viewModelScope.launch(Dispatchers.IO) {
            _ttsLoading.value = "Switching to ${lang.label}…"
            try {
                mmstts.stop()
                mmstts.start(lang.code, _ttsSettings.value, prewarm = false)
            } catch (t: Throwable) {
                Log.w("MainViewModel", "language switch to ${lang.code} failed", t)
                _language.value = previous
                if (previous != null) {
                    try { mmstts.stop(); mmstts.start(previous.code, _ttsSettings.value, prewarm = false) }
                    catch (_: Throwable) { /* nothing more we can do here */ }
                }
                _ttsLoading.value = "Failed to load ${lang.label}. Reverted to ${previous?.label ?: "none"}."
                kotlinx.coroutines.delay(3000)
            }
            _ttsLoading.value = null
        }
    }

    fun wipeAllLanguages() {
        ttsStartJob?.cancel()
        mmstts.stop()
        _language.value = null
        languageRegistry.wipeAll()
    }

    private var askJob: Job? = null
    private var speakJob: Job? = null
    private var ttsStartJob: Job? = null

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

        _phase.value = UiPhase.Ready

        // Start MMS-TTS in the background if a language is already installed.
        val lang = _language.value
        if (lang != null) {
            ttsStartJob = viewModelScope.launch(Dispatchers.IO) {
                _ttsLoading.value = "Loading TTS…"
                try {
                    mmstts.start(lang.code, _ttsSettings.value, prewarm = false)
                } catch (t: Throwable) {
                    Log.w("MainViewModel", "background MMS-TTS start failed: ${t.message}", t)
                }
                _ttsLoading.value = null
            }
        }
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
        ensureTtsReady()
        val sentences = splitSentences(text)
        val cwd = File(context.filesDir, "mmstts")
        val audioDir = File(cwd, "audio_queue").also { it.mkdirs() }

        audioDir.listFiles()?.forEach { it.delete() }

        _camera.value = CameraFlow.Asking(jpeg, AskPhase.Synthesizing, text)
        wavPlayer.stop()

        val prebufferTarget = adaptivePrebufferTarget(sentences.size)
        val buffered = mutableListOf<com.adreno.seeandsay.runner.MMSTTSSession.Result>()

        fun enqueueOne(res: com.adreno.seeandsay.runner.MMSTTSSession.Result, index: Int) {
            val onDone: () -> Unit = {
                if (index == sentences.lastIndex) {
                    val cur = _camera.value
                    if (cur is CameraFlow.Asking) _camera.value = cur.copy(phase = AskPhase.Done)
                }
            }
            val pcm = res.pcm
            if (pcm != null) wavPlayer.enqueuePcm(pcm, res.sampleRate, onComplete = onDone)
            else             wavPlayer.enqueue(res.wav, onComplete = onDone)
        }

        var playbackStarted = false
        for ((index, sentence) in sentences.withIndex()) {
            val target = File(audioDir, "utt_${index.toString().padStart(3, '0')}.wav")
            val res = mmstts.speak(sentence, persistTo = target)
            updateRtf(res)
            buffered.add(res)

            _metrics.value = _metrics.value.copy(
                ttsAudioSec = res.audioSec,
                ttsRtf = res.rtf,
                ttsFwdSec = res.fwdSec,
                ttsTtfsSec = res.ttfsSec,
                mmsttsRssMb = res.rssMb,
                mmsttsPeakRssMb = res.peakRssMb,
            )

            if (index < prebufferTarget) {
                // Still building the playback runway.
            } else if (!playbackStarted) {
                // Runway built — drain all buffered sentences then stream.
                for ((i, r) in buffered.withIndex()) enqueueOne(r, i)
                _camera.value = CameraFlow.Asking(jpeg, AskPhase.Speaking, text)
                playbackStarted = true
            } else {
                enqueueOne(res, index)
            }
        }

        // Short text where sentences.size <= prebufferTarget: flush now.
        if (!playbackStarted) {
            for ((i, r) in buffered.withIndex()) enqueueOne(r, i)
            _camera.value = CameraFlow.Asking(jpeg, AskPhase.Speaking, text)
        }

        _metrics.value = _metrics.value.copy(
            e2eSec = (System.currentTimeMillis() - askStart) / 1000.0,
        )
    }

    private fun splitSentences(text: String): List<String> {
        // Split on sentence-ending punctuation: ASCII (.!?) plus Ethiopic
        // full stop ። (U+1362) and question mark ፧ (U+1367). Without the
        // Ethiopic marks, Amharic text arrives as one giant sentence and
        // streaming provides no latency benefit.
        val parts = text.split(Regex("""(?<=[.!?።፧])\s+"""))
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

    // ---- Unified chat flow --------------------------------------------

    fun setPendingAttachment(jpeg: File) { _pendingAttachment.value = jpeg }
    fun clearPendingAttachment() { _pendingAttachment.value = null }
    fun toggleChatTts() { _chatTtsEnabled.value = !_chatTtsEnabled.value }

    /**
     * Start a fresh chat: clear history + binary KV cache. Cheap — sends a
     * `/reset` to the SmolVLM session and zeroes the in-memory turn list.
     */
    fun newChat() {
        chatJob?.cancel()
        wavPlayer.stop()
        _chatHistory.value = emptyList()
        _pendingAttachment.value = null
        _chat.value = ChatFlow.Idle
        // Wipe last-turn perf so the top-bar metrics strip disappears with
        // the conversation. Without this, "decode 14.3 tok/s" from the
        // previous chat would linger until the first new reply lands.
        _metrics.value = AskMetrics()
        viewModelScope.launch(Dispatchers.IO) {
            // Issue /reset through a no-op query path. Simplest: rely on the
            // next askChat() call to do the /reset itself (askWithImage and
            // askTextOnly both send /reset).
        }
    }

    /**
     * Send a chat message. Routes to:
     *   - askWithImage if [imagePath] is set (silent reset if mid-conversation)
     *   - askTextOnly  if this is turn 1 and no image
     *   - ask          if this is a follow-up text-only turn
     *
     * The assistant's response streams into a placeholder ASSISTANT turn at
     * the end of [_chatHistory]. On completion, optionally pipes the final
     * text through MMS-TTS if [_chatTtsEnabled] is true.
     */
    fun askChat(prompt: String, imagePath: String? = null) {
        if (prompt.isBlank() && imagePath == null) return
        chatJob?.cancel()
        wavPlayer.stop()

        val history = _chatHistory.value
        val hasPriorTurns = history.any { it.role != ChatRole.SYSTEM }

        // Mid-conversation image attachment → silent /reset + system divider.
        val withDivider = if (imagePath != null && hasPriorTurns) {
            _chatHistory.value + ChatTurn(ChatRole.SYSTEM, "— new conversation —")
        } else {
            _chatHistory.value
        }

        val userTurn = ChatTurn(ChatRole.USER, prompt.trim(), imagePath = imagePath)
        val placeholder = ChatTurn(ChatRole.ASSISTANT, "")
        _chatHistory.value = withDivider + userTurn + placeholder
        _pendingAttachment.value = null
        _chat.value = ChatFlow.Thinking

        // After a divider, we're starting fresh — no prior turns from the
        // binary's perspective either (askWithImage sends /reset internally).
        val isFollowUp = imagePath == null && hasPriorTurns

        val askStart = System.currentTimeMillis()
        _metrics.value = AskMetrics()

        chatJob = viewModelScope.launch(Dispatchers.Default) {
            val sb = StringBuilder()
            var finalText: String? = null
            var lastEmitMs = 0L
            val streamThrottleMs = 200L

            // No system-prompt prepend in chat. SmolVLM-256M is small enough
            // that prepending a verbose system instruction to a terse user
            // message ("what do u see") makes the model echo the instructions
            // back as if they were the answer. Camera mode dodges this by
            // preloading the image before the user types (so the first user
            // turn is treated as a follow-up and the system prompt is skipped);
            // chat has no such window. Send the user's message verbatim and
            // let SmolVLM's image-grounded prefill handle description.
            try {
                val flow = when {
                    imagePath != null -> smolvlm.askWithImage(imagePath, prompt)
                    isFollowUp        -> smolvlm.ask(prompt)
                    else              -> smolvlm.askTextOnly(prompt)
                }
                flow.collect { out ->
                    when (out) {
                        is SmolVLMSession.Output.Token -> {
                            sb.append(out.text)
                            val now = System.currentTimeMillis()
                            val natural = out.text.contains('.') || out.text.contains('\n')
                            if (natural || now - lastEmitMs >= streamThrottleMs) {
                                updateLastAssistantTurn(sb.toString())
                                _chat.value = ChatFlow.Streaming
                                lastEmitMs = now
                            }
                        }
                        is SmolVLMSession.Output.Final -> {
                            finalText = out.text
                            updateLastAssistantTurn(out.text)
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
                            _chat.value = ChatFlow.Failed(out.message)
                            return@collect
                        }
                    }
                }
            } catch (t: Throwable) {
                _chat.value = ChatFlow.Failed(t.message ?: t.javaClass.simpleName)
                return@launch
            }

            val finalShown = finalText ?: sb.toString().trim()
            updateLastAssistantTurn(finalShown)
            _lastAnswer.value = finalShown
            _metrics.value = _metrics.value.copy(
                e2eSec = (System.currentTimeMillis() - askStart) / 1000.0,
            )

            // Optional TTS — only if a language is installed AND user toggled it on.
            if (_chatTtsEnabled.value && _language.value != null && finalShown.isNotBlank()) {
                _chat.value = ChatFlow.Speaking
                try {
                    speakStreamedNoUi(finalShown)
                } catch (t: Throwable) {
                    Log.w("MainViewModel", "chat TTS failed", t)
                }
            }
            _chat.value = ChatFlow.Idle
        }
    }

    private fun updateLastAssistantTurn(text: String) {
        val list = _chatHistory.value.toMutableList()
        val lastIdx = list.indexOfLast { it.role == ChatRole.ASSISTANT }
        if (lastIdx >= 0) {
            list[lastIdx] = list[lastIdx].copy(text = text)
            _chatHistory.value = list
        }
    }

    /**
     * Synthesize+play [text] through MMS-TTS without touching SpeakFlow state.
     * Used by chat auto-TTS so the Speak-tab UI doesn't show in-progress state
     * while the user is on the chat screen.
     */
    private suspend fun speakStreamedNoUi(text: String) {
        ensureTtsReady()
        val sentences = splitSentences(text)
        val cwd = File(context.filesDir, "mmstts")
        val audioDir = File(cwd, "chat_audio_queue").also { it.mkdirs() }
        audioDir.listFiles()?.forEach { it.delete() }
        wavPlayer.stop()

        val prebufferTarget = adaptivePrebufferTarget(sentences.size)
        val buffered = mutableListOf<com.adreno.seeandsay.runner.MMSTTSSession.Result>()
        var playbackStarted = false

        fun enqueueOne(r: com.adreno.seeandsay.runner.MMSTTSSession.Result) {
            val pcm = r.pcm
            if (pcm != null) wavPlayer.enqueuePcm(pcm, r.sampleRate)
            else             wavPlayer.enqueue(r.wav)
        }

        for ((index, sentence) in sentences.withIndex()) {
            val target = File(audioDir, "chat_${index.toString().padStart(3, '0')}.wav")
            val res = mmstts.speak(sentence, persistTo = target)
            updateRtf(res)
            buffered.add(res)

            if (index < prebufferTarget) {
                // building runway
            } else if (!playbackStarted) {
                for (r in buffered) enqueueOne(r)
                playbackStarted = true
            } else {
                enqueueOne(res)
            }
        }
        if (!playbackStarted) {
            for (r in buffered) enqueueOne(r)
        }
    }

    // ---- Speak tab flow -----------------------------------------------

    private suspend fun ensureTtsReady() {
        val lang = _language.value
            ?: throw IllegalStateException("No TTS language installed. Download one from Languages.")
        if (mmstts.isAlive()) return
        _ttsLoading.value = "Starting TTS…"
        withContext(Dispatchers.IO) {
            mmstts.start(lang.code, _ttsSettings.value)
        }
        _ttsLoading.value = null
    }

    fun speak(text: String) {
        if (text.isBlank()) return
        speakJob?.cancel()
        wavPlayer.stop()
        _speak.value = SpeakFlow.Synthesizing(0, 0)
        speakJob = viewModelScope.launch(Dispatchers.Default) {
            val startMs = System.currentTimeMillis()
            try {
                ensureTtsReady()
                val sentences = splitSentences(text)
                val cwd = File(context.filesDir, "mmstts")
                val audioDir = File(cwd, "speak_queue").also { it.mkdirs() }
                audioDir.listFiles()?.forEach { it.delete() }

                // Adaptive prebuffer: synthesize ceil(RTF) sentences before
                // starting playback so the playback runway stays ahead of
                // synthesis. Short texts (fewer sentences than ceil(RTF))
                // naturally prebuffer everything. Long texts start playing
                // after the runway is built — up to 55% faster TTFS vs
                // the old prebuffer-all approach.
                val prebufferTarget = adaptivePrebufferTarget(sentences.size)
                var lastRes: com.adreno.seeandsay.runner.MMSTTSSession.Result? = null
                var totalAudioSec = 0.0
                var totalFwdSec = 0.0
                var maxPeakMb: Int? = null
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
                    _speak.value = SpeakFlow.Synthesizing(index + 1, sentences.size)
                    val target = File(audioDir, "speak_${index.toString().padStart(3, '0')}.wav")
                    val res = mmstts.speak(sentence, persistTo = target)
                    updateRtf(res)
                    lastRes = res
                    totalAudioSec += res.audioSec
                    totalFwdSec += res.fwdSec
                    maxPeakMb = maxOf(maxPeakMb ?: 0, res.peakRssMb ?: 0).takeIf { it > 0 }
                    buffered.add(res)

                    if (index < prebufferTarget) {
                        // Still pre-buffering.
                    } else if (index == prebufferTarget) {
                        for ((i, r) in buffered.withIndex()) playOne(r, i)
                    } else {
                        playOne(res, index)
                    }
                }

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
        // Includes Ethiopic full stop ። (U+1362) and question mark ፧ (U+1367).
        val window = t.take(maxChars)
        val lastBoundary = listOf('.', '!', '?', ';', '\n', '።', '፧')
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
        chatJob?.cancel()
        wavPlayer.stop()
        smolvlm.stop()
        mmstts.stop()
        languageRegistry.shutdown()
        super.onCleared()
    }
}
