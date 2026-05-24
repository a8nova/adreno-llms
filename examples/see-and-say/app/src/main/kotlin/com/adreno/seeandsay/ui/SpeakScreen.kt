package com.adreno.seeandsay.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.Button
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.adreno.seeandsay.MainViewModel
import com.adreno.seeandsay.SpeakFlow
import com.adreno.seeandsay.TtsLanguage
// Note: TtsLanguage is now a data class, not an enum. References to
// TtsLanguage.Amharic / .English have been replaced with code-string checks.
import kotlinx.coroutines.delay

@Composable
fun SpeakScreen(viewModel: MainViewModel) {
    val language by viewModel.language.collectAsStateWithLifecycle()
    val examples = examplesFor(language)
    var text by rememberSaveable(language.code) { mutableStateOf(examples.first()) }
    val state by viewModel.speak.collectAsStateWithLifecycle()
    val sampler by viewModel.samplerSettings.collectAsStateWithLifecycle()
    val tts by viewModel.ttsSettings.collectAsStateWithLifecycle()
    val systemPrompt by viewModel.systemPrompt.collectAsStateWithLifecycle()
    var showConfigs by remember { mutableStateOf(false) }
    val device by viewModel.deviceInfo.collectAsStateWithLifecycle()

    // Input character budget. MMS-TTS synthesis cost scales with sentence
    // length (text_encoder + flow_inverse are both O(T_frames), and T_frames
    // ≈ 2× chars). On low-memory devices (Adreno 619 with 1708 MB GPU mem on
    // the Tab A9 we tested) a 250-char Amharic input triggers Android's LMK
    // mid-synthesis and the app process is killed. 600 chars is enough for
    // 3-4 short sentences without OOM on the marginal devices.
    val maxChars = if ((device?.globalMemMb ?: 4096) < 2048) 350 else 800

    Box(modifier = Modifier.fillMaxSize()) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.Top,
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(
                text = "✏︎  Text to speak  ·  ${language.label}",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.primary,
            )
            Text(
                text = "${text.length}/$maxChars",
                style = MaterialTheme.typography.labelSmall,
                color = if (text.length > maxChars)
                    MaterialTheme.colorScheme.error
                else
                    MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            )
        }
        Spacer(Modifier.height(6.dp))
        OutlinedTextField(
            value = text,
            onValueChange = { new ->
                // Hard-cap input to maxChars. Trying to type past it is a
                // silent no-op rather than an alert toast — less disruptive,
                // and the counter turning red is the visible signal.
                if (new.length <= maxChars) text = new
            },
            modifier = Modifier.fillMaxWidth().height(110.dp),
            textStyle = MaterialTheme.typography.bodyMedium,
            placeholder = {
                Text(
                    text = if (language.code == "amh") "በዚህ ይተይቡ…" else "Type text here…",
                    style = MaterialTheme.typography.bodySmall,
                )
            },
        )
        // On low-memory devices: warn explicitly so user knows why the cap is tight.
        if (maxChars < 500) {
            Spacer(Modifier.height(4.dp))
            Text(
                text = "⚠  Device GPU memory is < 2 GB; long inputs can crash the app " +
                       "mid-synthesis. Cap is $maxChars chars.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.tertiary.copy(alpha = 0.85f),
            )
        }

        Spacer(Modifier.height(10.dp))

        // ── Example chips — language-aware, tap to fill the text field ──────
        Text(
            text = "📋  Try an example",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.75f),
        )
        Spacer(Modifier.height(6.dp))
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            examples.forEach { example ->
                ExampleChip(text = example, onClick = { text = example })
            }
        }

        Spacer(Modifier.height(20.dp))

        // ONE big button. While the synth is running it morphs into Stop;
        // otherwise it's the primary Speak button. No second affordance —
        // the tap target stays in the same spot, just the action changes.
        val busy = state is SpeakFlow.Synthesizing
        Button(
            onClick = { if (busy) viewModel.stopSpeak() else viewModel.speak(text) },
            enabled = busy || text.isNotBlank(),
            modifier = Modifier
                .fillMaxWidth()
                .height(64.dp),
        ) {
            Text(
                text = if (busy) "⏹  Stop" else "🔊  Speak",
                style = MaterialTheme.typography.titleMedium,
            )
        }

        Spacer(Modifier.height(14.dp))

        when (val s = state) {
            is SpeakFlow.Synthesizing -> {
                val tickMs by produceState(0L, state) {
                    val t0 = System.currentTimeMillis()
                    while (true) { value = System.currentTimeMillis() - t0; delay(200) }
                }
                val dots = ".".repeat(((tickMs / 200) % 4).toInt() + 1)
                val progress = if (s.sentenceTotal > 0)
                    "  · sentence ${s.sentenceIndex}/${s.sentenceTotal}"
                else
                    ""
                Text(
                    text = "🎙  generating speech$dots  · %.1fs%s".format(tickMs / 1000.0, progress),
                    color = MaterialTheme.colorScheme.primary,
                    style = MaterialTheme.typography.labelMedium,
                )
            }
            is SpeakFlow.Done -> {
                Text(
                    text = "✓  done",
                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.75f),
                    style = MaterialTheme.typography.bodySmall,
                )
                Spacer(Modifier.height(10.dp))
                SpeakMetricsPanel(s)
            }
            is SpeakFlow.Failed -> Text(
                text = "⚠︎  ${s.message}",
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
            )
            SpeakFlow.Idle -> Text(" ", style = MaterialTheme.typography.bodySmall)
        }
    }
        // Top-right Configurations launcher — same affordance as the Camera tab.
        // Restarting either the SmolVLM or MMS-TTS session via this dialog
        // bounces the corresponding process; the loader screen reappears
        // during the restart.
        FilledIconButton(
            onClick = { showConfigs = true },
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 6.dp, end = 6.dp)
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

/**
 * Per-utterance MMS-TTS perf block. Same monospace style as the Camera tab's
 * MetricsPanel; lets you eyeball synth speed (RTF) and peak memory off the
 * device without adb.
 */
@Composable
private fun SpeakMetricsPanel(s: SpeakFlow.Done) {
    val mono = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace)
    val dim = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f)
    val accent = MaterialTheme.colorScheme.primary.copy(alpha = 0.9f)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.65f))
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Text("PERF — last utterance", color = accent, style = mono)
        metricRow("sentences", s.sentenceCount.toString(), mono, dim)
        metricRow("ttfs (first)", "%.2f s".format(s.ttfsSec), mono, dim)
        metricRow("fwd (total)", "%.2f s".format(s.fwdSec), mono, dim)
        metricRow("audio (total)", "%.2f s".format(s.audioSec), mono, dim)
        metricRow("rtf", "%.2f".format(s.rtf), mono, dim)
        metricRow("rss", s.rssMb?.let { "%d MB".format(it) } ?: "—", mono, dim)
        metricRow("peak rss", s.peakRssMb?.let { "%d MB".format(it) } ?: "—", mono, dim)
        Spacer(Modifier.height(2.dp))
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("end-to-end", color = accent, style = mono)
            Text("%.2f s".format(s.e2eSec), color = MaterialTheme.colorScheme.onSurface, style = mono)
        }
    }
}

@Composable
private fun metricRow(
    label: String,
    value: String,
    mono: androidx.compose.ui.text.TextStyle,
    dim: androidx.compose.ui.graphics.Color,
) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text("  $label", color = dim, style = mono)
        Text(value, color = MaterialTheme.colorScheme.onSurface, style = mono)
    }
}

@Composable
private fun ExampleChip(text: String, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(MaterialTheme.colorScheme.surface)
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 8.dp),
    ) {
        Text(
            text = text,
            color = MaterialTheme.colorScheme.onSurface,
            style = MaterialTheme.typography.bodyMedium,
            maxLines = 2,
        )
    }
}

/**
 * Curated test prompts per language — short enough to synthesize fast on the
 * device, long enough to show off the voice. Pick from a variety of registers
 * (greeting, descriptive, demo-narrative) so the user can pressure-test.
 */
private fun examplesFor(lang: TtsLanguage): List<String> = when (lang.code) {
    "eng" -> listOf(
        "Hello! My name is Adreno. I'm an on-device speech model running entirely on this phone.",
        "The quick brown fox jumps over the lazy dog. Speech synthesis is harder than it looks.",
        "This entire app, including the language model and the text-to-speech, runs without any internet connection.",
    )
    "amh" -> listOf(
        "ሰላም! ስሜ አድሬኖ ነው። ይህን ስልክ ላይ የሚሰራ የንግግር ሞዴል ነኝ።",
        "የአየር ሁኔታው ዛሬ በጣም ጥሩ ነው። ሙቀቱ ሃያ አምስት ዲግሪ ሴንቲግሬድ ነው።",
        "ይህ ሙሉ መተግበሪያ፣ የቋንቋ ሞዴል እና የጽሁፍ-ወደ-ንግግር ጨምሮ፣ ያለ ኢንተርኔት ግንኙነት ይሰራል።",
    )
    // For all the downloaded languages we have no curated examples; show a
    // generic placeholder so the chip strip isn't empty.
    else -> listOf(
        "Hello.",
        "This is a test.",
        "The quick brown fox jumps over the lazy dog.",
    )
}
