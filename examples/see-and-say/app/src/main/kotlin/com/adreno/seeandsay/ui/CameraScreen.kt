package com.adreno.seeandsay.ui

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.FlipCameraAndroid
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Lens
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil.compose.AsyncImage
import com.adreno.seeandsay.AskPhase
import com.adreno.seeandsay.CameraFlow
import com.adreno.seeandsay.MainViewModel
import com.adreno.seeandsay.R
import com.adreno.seeandsay.camera.CameraController
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.File

@Composable
fun CameraScreen(viewModel: MainViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED,
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
    ) { granted -> hasPermission = granted }

    if (!hasPermission) {
        Column(
            modifier = Modifier.fillMaxSize().padding(32.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = context.getString(R.string.camera_permission_rationale),
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onBackground,
            )
            Spacer(Modifier.height(24.dp))
            Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                Text(context.getString(R.string.grant_camera_permission))
            }
        }
        return
    }

    val flow by viewModel.camera.collectAsStateWithLifecycle()
    val sampler by viewModel.samplerSettings.collectAsStateWithLifecycle()
    val tts by viewModel.ttsSettings.collectAsStateWithLifecycle()
    val systemPrompt by viewModel.systemPrompt.collectAsStateWithLifecycle()
    var showConfigs by remember { mutableStateOf(false) }

    // LivePreview stays always-mounted so its CameraX binding doesn't churn
    // on every state change. CapturedView is rendered conditionally — when in
    // Preview state we don't render it at all, so its (previously alpha=0)
    // surface no longer intercepts touch events on the LivePreview buttons
    // underneath. That was the "can't take a second picture" bug.
    Box(modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        LivePreview(viewModel, lifecycleOwner)

        if (flow !is CameraFlow.Preview) {
            when (val state = flow) {
                is CameraFlow.Captured -> CapturedView(viewModel, jpeg = state.jpeg, phase = null, streamed = "", error = null)
                is CameraFlow.Asking -> CapturedView(viewModel, jpeg = state.jpeg, phase = state.phase, streamed = state.streamed, error = state.error)
                else -> {}
            }
        }

        // Top-right Configurations launcher. Sits above both LivePreview and
        // CapturedView so the user can tune sampler settings whether or not an
        // image is loaded. Tapping opens the dialog; OK fires off a SmolVLM
        // restart (the binary takes these as CLI flags at launch only).
        FilledIconButton(
            onClick = { showConfigs = true },
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 12.dp, end = 12.dp)
                .size(44.dp),
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                contentColor = MaterialTheme.colorScheme.onSurface,
            ),
        ) {
            Icon(Icons.Filled.Tune, contentDescription = "Configurations", modifier = Modifier.size(22.dp))
        }

        if (showConfigs) {
            ConfigurationsDialog(
                current = sampler,
                currentTts = tts,
                currentSystemPrompt = systemPrompt,
                defaultSystemPrompt = com.adreno.seeandsay.DEFAULT_SYSTEM_PROMPT,
                currentRtf = viewModel.currentRtf,
                onDismiss = { showConfigs = false },
                onConfirm = { newSampler, newTts, newSys ->
                    viewModel.updateSamplerSettings(newSampler)
                    viewModel.updateTtsSettings(newTts)
                    viewModel.updateSystemPrompt(newSys)
                    showConfigs = false
                },
            )
        }
    }
}

@Composable
private fun LivePreview(viewModel: MainViewModel, lifecycleOwner: androidx.lifecycle.LifecycleOwner) {
    val context = LocalContext.current
    val controller = remember { CameraController(context) }
    var streaming by remember { mutableStateOf(false) }
    // Tracked separately from controller.lensFacing so a re-composition
    // updates the icon's contentDescription. Initial value matches
    // CameraController's default (back).
    var facingFront by remember { mutableStateOf(false) }

    val pickImageLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
    ) { uri: Uri? ->
        if (uri != null) {
            val target = File(context.filesDir, "captures/picked.jpg")
            target.parentFile?.mkdirs()
            context.contentResolver.openInputStream(uri)?.use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
            if (target.exists() && target.length() > 0) viewModel.onCaptured(target)
        }
    }

    // Form-style layout: bounded camera preview at top, then shutter, then
    // upload-from-gallery. Fixed pixel sizes everywhere — aspectRatio
    // combined with heightIn produces conflicting constraints that pushed
    // the whole cluster to the bottom of large screens. No verticalScroll
    // either; if a future device can't fit ~600dp of content we add it back
    // explicitly.
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        // ── Fixed-height camera preview ────────────────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(260.dp)
                .clip(RoundedCornerShape(16.dp))
                .background(androidx.compose.ui.graphics.Color.Black),
        ) {
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { ctx ->
                    val pv = PreviewView(ctx).apply {
                        setBackgroundColor(android.graphics.Color.BLACK)
                        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                        previewStreamState.observe(lifecycleOwner) { s ->
                            streaming = s == PreviewView.StreamState.STREAMING
                        }
                    }
                    controller.bind(lifecycleOwner, pv)
                    pv
                },
            )
            // Opaque scrim until the first camera frame lands — kills the
            // white flash on fold/unfold + cold bind.
            if (!streaming) {
                Box(modifier = Modifier.fillMaxSize().background(androidx.compose.ui.graphics.Color.Black))
            }
            // Flip-camera pill (overlaid top-left of the preview box).
            Row(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(8.dp)
                    .clip(RoundedCornerShape(20.dp))
                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.85f))
                    .clickable {
                        controller.flip()
                        facingFront = !facingFront
                    }
                    .padding(horizontal = 10.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    Icons.Filled.FlipCameraAndroid,
                    contentDescription = "Flip camera",
                    tint = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.size(18.dp),
                )
                Spacer(Modifier.size(4.dp))
                Text(
                    text = if (facingFront) "Front" else "Back",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurface,
                )
            }
        }

        Spacer(Modifier.height(16.dp))

        // ── Primary action: big shutter button ─────────────────────────────
        FilledIconButton(
            onClick = {
                val target = File(context.filesDir, "captures/last.jpg")
                controller.capture(target) { result ->
                    result.onSuccess { viewModel.onCaptured(it) }
                }
            },
            modifier = Modifier.size(72.dp),
            shape = CircleShape,
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = MaterialTheme.colorScheme.primary,
                contentColor = MaterialTheme.colorScheme.onPrimary,
            ),
        ) {
            Icon(Icons.Filled.Lens, contentDescription = "Capture photo", modifier = Modifier.size(40.dp))
        }
        Text(
            text = "Tap to snap a photo",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.55f),
            modifier = Modifier.padding(top = 4.dp),
        )

        Spacer(Modifier.height(20.dp))

        // ── Divider with "or" ──────────────────────────────────────────────
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            androidx.compose.material3.HorizontalDivider(
                modifier = Modifier.weight(1f),
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.2f),
            )
            Text(
                text = "  or  ",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.55f),
            )
            androidx.compose.material3.HorizontalDivider(
                modifier = Modifier.weight(1f),
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.2f),
            )
        }

        Spacer(Modifier.height(12.dp))

        // ── Alt action: upload from gallery (full-width, prominent) ────────
        Button(
            onClick = { pickImageLauncher.launch("image/*") },
            modifier = Modifier.fillMaxWidth().height(54.dp),
        ) {
            Icon(Icons.Filled.Image, contentDescription = null, modifier = Modifier.size(22.dp))
            Spacer(Modifier.size(8.dp))
            Text("Upload from gallery", style = MaterialTheme.typography.titleSmall)
        }

        Spacer(Modifier.height(8.dp))

        // ── Tertiary: sample image (small, low-key) ────────────────────────
        OutlinedButton(
            onClick = {
                val sample = File(context.filesDir, "smolvlm/fixtures/warmup.jpg")
                if (sample.exists()) viewModel.onCaptured(sample)
            },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Icon(Icons.Filled.Star, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(Modifier.size(8.dp))
            Text("Use sample image", style = MaterialTheme.typography.bodySmall)
        }
    }
}

/**
 * Captured-state UI. Layout order, top to bottom:
 *   1. ✏︎ YOUR QUESTION   (labeled input field — clearly editable)
 *   2. [Ask] button
 *   3. 📷 your image      (small thumbnail with retake corner button)
 *   4. 💬 ANSWER          (labeled, clearly read-only, distinct background)
 *   5. perf panel         (dim, at bottom)
 *
 * The two sections — "your question" and "answer" — have *different* visual
 * affordances (border vs. filled card, edit cursor vs. selectable text) so
 * it's never ambiguous which one accepts typing.
 */
@Composable
private fun CapturedView(viewModel: MainViewModel, jpeg: File, phase: AskPhase?, streamed: String, error: String?) {
    val ctx = LocalContext.current
    var prompt by remember(jpeg) { mutableStateOf(ctx.getString(R.string.camera_default_prompt)) }
    val keyboard = LocalSoftwareKeyboardController.current
    val working = phase != null && phase != AskPhase.Done && phase != AskPhase.Failed
    val finished = phase == AskPhase.Done || phase == AskPhase.Failed
    val hasAnswer = streamed.isNotEmpty()
    val isFollowUp = finished // a question after one has already been answered

    // Defer the vision-tower preload until AFTER Compose has painted the
    // captured-view transition at least once. withFrameNanos suspends until
    // the next Choreographer frame callback (~16 ms), giving the compositor a
    // vsync slot to crossfade LivePreview → CapturedView before the single
    // Adreno 620 compute unit gets saturated by the vision encoder. Without
    // this gate, the GPU burst from preloadImage() starves SurfaceFlinger and
    // the transition shows a brief white flash. Idempotent on the VM side
    // (kickPreloadFor de-dupes by jpeg path), so safe across recompositions.
    LaunchedEffect(jpeg) {
        withFrameNanos { /* one-shot: wait for first frame */ }
        viewModel.kickPreloadFor(jpeg)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            // Force an opaque black backdrop. Without this, Crossfade allocates
            // an offscreen render buffer for CapturedView that on Adreno's Skia
            // path appears to default-clear to white-with-alpha-0 — when the
            // Crossfade composites our partially-rendered tree at alpha < 1.0,
            // any un-filled (transparent) pixels show as WHITE over LivePreview
            // for the duration of the 280 ms fade. An explicit opaque background
            // fills every pixel of the captured-view buffer with black first.
            .background(MaterialTheme.colorScheme.background)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.Top,
    ) {
        // ── 1. INPUT: "Your question" ───────────────────────────────────────
        Text(
            text = if (isFollowUp) "✏︎  Ask a follow-up about this image" else "✏︎  Your question",
            color = MaterialTheme.colorScheme.primary,
            style = MaterialTheme.typography.labelMedium,
        )
        Spacer(Modifier.height(4.dp))

        // Preset prompt chips. Tapping a chip rewrites the prompt text box; the user
        // can still edit before tapping "Ask & Speak". All five presets are
        // designed to elicit SHORT responses so the end-to-end demo
        // (vision → LLM decode → TTS) finishes faster — decode and TTS
        // budgets scale with response length, vision tower is fixed.
        val presets = remember {
            listOf(
                R.string.preset_describe_label to R.string.preset_describe,
                R.string.preset_caption_label  to R.string.preset_caption,
                R.string.preset_colors_label   to R.string.preset_colors,
                R.string.preset_read_label     to R.string.preset_read,
                R.string.preset_guess_label    to R.string.preset_guess,
            )
        }
        LazyRow(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            items(presets) { (labelRes, textRes) ->
                AssistChip(
                    onClick = { prompt = ctx.getString(textRes) },
                    label = { Text(ctx.getString(labelRes), style = MaterialTheme.typography.labelSmall) },
                    enabled = !working,
                )
            }
        }
        Spacer(Modifier.height(4.dp))

        OutlinedTextField(
            value = prompt,
            onValueChange = { prompt = it },
            modifier = Modifier.fillMaxWidth(),
            placeholder = { Text("e.g. What is in this picture?", style = MaterialTheme.typography.bodySmall) },
            textStyle = MaterialTheme.typography.bodyMedium,
            singleLine = false,
            maxLines = 3,
            enabled = !working,
        )
        Spacer(Modifier.height(8.dp))
        Button(
            onClick = { keyboard?.hide(); viewModel.askAndSpeak(prompt) },
            modifier = Modifier.fillMaxWidth(),
            enabled = !working,
        ) {
            Text(when {
                working -> "Working…"
                isFollowUp -> "Ask follow-up"
                else -> "Ask"
            })
        }

        Spacer(Modifier.height(14.dp))

        // ── 2. IMAGE: small thumbnail of what we're asking about ────────────
        Text(
            text = "📷  About this image",
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.65f),
            style = MaterialTheme.typography.labelMedium,
        )
        Spacer(Modifier.height(4.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 80.dp, max = 140.dp)
                .clip(RoundedCornerShape(10.dp)),
        ) {
            AsyncImage(model = jpeg, contentDescription = null, modifier = Modifier.fillMaxSize())
            if (!working) {
                FilledIconButton(
                    onClick = { viewModel.onRetake() },
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(6.dp)
                        .size(32.dp),
                    colors = IconButtonDefaults.filledIconButtonColors(
                        containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                        contentColor = MaterialTheme.colorScheme.onSurface,
                    ),
                ) {
                    Icon(Icons.Filled.Refresh, contentDescription = "Retake", modifier = Modifier.size(18.dp))
                }
            }
        }

        // ── 3. PHASE BANNER (only when a query is in flight or just finished) ─
        if (phase != null) {
            Spacer(Modifier.height(10.dp))
            PhaseBanner(phase = phase, error = error)
        }

        // ── 4. ANSWER: clearly labeled, clearly read-only ───────────────────
        if (hasAnswer) {
            Spacer(Modifier.height(10.dp))
            Text(
                text = "💬  Answer",
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.labelMedium,
            )
            Spacer(Modifier.height(4.dp))
            // Read-only card. Background uses a distinct surfaceVariant so it
            // visually contrasts the OutlinedTextField above.
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.8f))
                    .padding(horizontal = 14.dp, vertical = 14.dp),
            ) {
                if (finished) {
                    Text(
                        text = streamed,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontFamily = FontFamily.SansSerif,
                        fontWeight = FontWeight.Medium,
                        style = MaterialTheme.typography.titleMedium,
                    )
                } else {
                    StreamingAnswerView(text = streamed)
                }
            }
        }

        // ── 5. Perf panel — at the bottom, only when idle ───────────────────
        if (!working) {
            val metrics by viewModel.metrics.collectAsStateWithLifecycle()
            Spacer(Modifier.height(14.dp))
            MetricsPanel(metrics)
        }
    }
}

/**
 * Native TextView fast-path for the streaming answer. Avoids Compose's text
 * remeasure cost per token by routing text mutation through a TextView's
 * setText() which only invalidates the TextView itself, not the whole tree.
 */
@Composable
private fun StreamingAnswerView(text: String) {
    val onSurface = MaterialTheme.colorScheme.onSurface
    AndroidView(
        modifier = Modifier.fillMaxWidth(),
        factory = { ctx ->
            android.widget.TextView(ctx).apply {
                setTextColor(android.graphics.Color.argb(
                    (onSurface.alpha * 255).toInt().coerceIn(0, 255),
                    (onSurface.red * 255).toInt().coerceIn(0, 255),
                    (onSurface.green * 255).toInt().coerceIn(0, 255),
                    (onSurface.blue * 255).toInt().coerceIn(0, 255),
                ))
                textSize = 16f
                setLineSpacing(2f, 1.15f)
            }
        },
        update = { tv -> tv.text = text },
    )
}

@Composable
private fun PhaseBanner(phase: AskPhase, error: String?) {
    // CPU-driven text ticker driven from Dispatchers.Default so it keeps ticking
    // even when the main thread is starved (cover-display throttling, Compose
    // recompose pressure during heavy GPU compute, etc.).
    // Run the ticker through a TextView too: when Compose can't pump frames
    // (cover-display GPU contention with the model), the banner's seconds
    // counter visibly freezes. A native TextView updates independently of the
    // Compose recomposition loop.
    PhaseBannerNative(phase, error)
}

@Composable
private fun PhaseBannerNative(phase: AskPhase, error: String?) {
    val primaryColor = MaterialTheme.colorScheme.primary
    val onSurfaceDim = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
    val errorColor = MaterialTheme.colorScheme.error

    val (label, busy, color) = when (phase) {
        AskPhase.Thinking -> Triple("thinking — preparing model & analyzing image", true, primaryColor)
        AskPhase.Streaming -> Triple("writing answer", true, primaryColor)
        AskPhase.Synthesizing -> Triple("generating speech", true, primaryColor)
        AskPhase.Speaking -> Triple("speaking", true, primaryColor)
        AskPhase.Done -> Triple("✓  done", false, onSurfaceDim)
        AskPhase.Failed -> Triple("⚠︎  " + (error ?: "something went wrong"), false, errorColor)
    }
    val icon = when (phase) {
        AskPhase.Thinking -> "🤔"
        AskPhase.Streaming -> "✍︎"
        AskPhase.Synthesizing -> "🎙"
        AskPhase.Speaking -> "🔊"
        else -> ""
    }
    val androidColor = android.graphics.Color.argb(
        (color.alpha * 255).toInt().coerceIn(0, 255),
        (color.red * 255).toInt().coerceIn(0, 255),
        (color.green * 255).toInt().coerceIn(0, 255),
        (color.blue * 255).toInt().coerceIn(0, 255),
    )

    // Remember one ticker per phase. When phase changes, the previous one is
    // stopped via DisposableEffect's onDispose. This prevents the bug where
    // every parent recomposition spawned a fresh ticker thread.
    val ticker = androidx.compose.runtime.remember(phase) { TickerHandle() }
    androidx.compose.runtime.DisposableEffect(phase) {
        onDispose { ticker.stop() }
    }

    AndroidView(
        modifier = Modifier.fillMaxWidth(),
        factory = { ctx ->
            android.widget.TextView(ctx).apply {
                textSize = 12f
                setTextColor(androidColor)
            }
        },
        update = { tv ->
            tv.setTextColor(androidColor)
            if (busy) {
                // Idempotent: attach() is a no-op if this ticker is already
                // pumping into the same view.
                ticker.attach(tv, label = label, icon = icon)
            } else {
                ticker.stop()
                tv.text = label
            }
        },
    )
}

/**
 * Pumps a seconds-counter text into a TextView on a daemon thread, fully
 * independent of Compose recomposition. Once attached, keeps running for the
 * life of the phase regardless of how often the parent recomposes.
 */
private class TickerHandle {
    @Volatile private var stopped = false
    @Volatile private var started = false
    private var thread: Thread? = null

    fun stop() {
        stopped = true
        thread?.interrupt()
    }

    fun attach(tv: android.widget.TextView, label: String, icon: String) {
        if (started) return
        started = true
        val start = System.currentTimeMillis()
        thread = Thread({
            while (!stopped) {
                val elapsed = System.currentTimeMillis() - start
                val dots = ".".repeat(((elapsed / 250) % 4).toInt() + 1)
                val secs = "%.1fs".format(elapsed / 1000.0)
                val text = "$icon  $label$dots  · $secs"
                tv.post { if (!stopped) tv.text = text }
                try { Thread.sleep(250) } catch (_: InterruptedException) { return@Thread }
            }
        }, "phase-ticker").apply { isDaemon = true; start() }
    }
}

@Composable
private fun PhaseBannerOldDoNotUse(phase: AskPhase, error: String?) {
    val tickMs by produceState(initialValue = 0L, phase) {
        if (phase == AskPhase.Done || phase == AskPhase.Failed) return@produceState
        withContext(Dispatchers.Default) {
            val start = System.currentTimeMillis()
            while (isActive) {
                value = System.currentTimeMillis() - start
                delay(200)
            }
        }
    }

    val (label, busy, color) = when (phase) {
        AskPhase.Thinking -> Triple("thinking — preparing model & analyzing image", true, MaterialTheme.colorScheme.primary)
        AskPhase.Streaming -> Triple("writing answer", true, MaterialTheme.colorScheme.primary)
        AskPhase.Synthesizing -> Triple("generating speech", true, MaterialTheme.colorScheme.primary)
        AskPhase.Speaking -> Triple("speaking", true, MaterialTheme.colorScheme.primary)
        AskPhase.Done -> Triple("✓  done", false, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
        AskPhase.Failed -> Triple("⚠︎  " + (error ?: "something went wrong"), false, MaterialTheme.colorScheme.error)
    }
    val icon = when (phase) {
        AskPhase.Thinking -> "🤔"
        AskPhase.Streaming -> "✍︎"
        AskPhase.Synthesizing -> "🎙"
        AskPhase.Speaking -> "🔊"
        else -> ""
    }
    val text = if (busy) {
        val dots = ".".repeat(((tickMs / 200) % 4).toInt() + 1)
        val secs = "%.1fs".format(tickMs / 1000.0)
        "$icon  $label$dots  · $secs"
    } else {
        label
    }
    Text(text = text, color = color, style = MaterialTheme.typography.labelMedium)
}
