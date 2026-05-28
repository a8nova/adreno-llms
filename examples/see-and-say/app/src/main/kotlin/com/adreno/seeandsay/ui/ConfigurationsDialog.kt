package com.adreno.seeandsay.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.foundation.layout.heightIn
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.adreno.seeandsay.SamplerSettings
import com.adreno.seeandsay.TtsSettings
import kotlin.math.roundToInt

/**
 * Modeled after the Google AI Edge Gallery "Configurations" dialog: a column
 * of slider + text-field pairs for the main sampler knobs, plus a GPU/CPU
 * accelerator pill and an OK/Cancel footer.
 *
 * The dialog owns its own draft state and only commits on OK — Cancel
 * discards. This matches the user's expectation that scrubbing a slider
 * shouldn't fire a SmolVLM process restart on every micro-movement.
 */
@Composable
fun ConfigurationsDialog(
    current: SamplerSettings,
    currentTts: TtsSettings,
    currentSystemPrompt: String,
    defaultSystemPrompt: String,
    currentRtf: Double,
    onDismiss: () -> Unit,
    onConfirm: (SamplerSettings, TtsSettings, String) -> Unit,
) {
    var maxTokens by remember { mutableStateOf(current.maxNewTokens.toFloat()) }
    var topK by remember { mutableStateOf(current.topK.toFloat()) }
    var topP by remember { mutableStateOf(current.topP) }
    var temperature by remember { mutableStateOf(current.temperature) }
    var noiseScale by remember { mutableStateOf(currentTts.noiseScale) }
    var noiseScaleW by remember { mutableStateOf(currentTts.noiseScaleW) }
    var lengthScale by remember { mutableStateOf(currentTts.lengthScale) }
    var systemPrompt by remember { mutableStateOf(currentSystemPrompt) }
    var imageSize by remember { mutableStateOf(current.imageSize) }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false),
    ) {
        Surface(
            shape = RoundedCornerShape(20.dp),
            color = MaterialTheme.colorScheme.surface,
            tonalElevation = 4.dp,
            modifier = Modifier
                .widthIn(min = 280.dp, max = 480.dp)
                // Cap height so the OK/Cancel footer is always visible:
                // content above scrolls inside its weight(1f) box, footer
                // stays pinned. Without a max, the dialog grows tall enough
                // to push the footer off-screen and the user has to scroll
                // the WHOLE screen to reach it.
                .fillMaxHeight(0.88f)
                .padding(16.dp),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Text(
                    text = "Configurations",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                )

                // Scrollable middle region — fills available vertical space.
                // Title above and OK/Cancel below stay pinned.
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                ) {

                Text(
                    text = "System prompt",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.primary,
                )
                OutlinedTextField(
                    value = systemPrompt,
                    onValueChange = { systemPrompt = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 96.dp),
                    label = { Text("Prepended to your prompt") },
                    placeholder = { Text("(empty — no system prompt)") },
                    maxLines = 6,
                    singleLine = false,
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End,
                ) {
                    TextButton(onClick = { systemPrompt = defaultSystemPrompt }) {
                        Text("Reset to default")
                    }
                }
                Text(
                    text = "SmolVLM doesn't have a native system slot — this text is prepended " +
                           "to every fresh-image query. Follow-up turns reuse it via KV cache.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f),
                )

                HorizontalDivider(color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.12f))

                Text(
                    text = "SmolVLM (image → text)",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.primary,
                )
                Text(
                    text = "Vision resolution",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                ImageSizeToggle(
                    selected = imageSize,
                    onSelect = { imageSize = it },
                )
                Text(
                    text = "Fast: 384px (36 image tokens, ~44% less vision compute). " +
                           "Quality: 512px (64 image tokens, matches upstream training). " +
                           "Both variants are baked into the weights bundle.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                )

                SliderRow(
                    label = "Max tokens (16-512)",
                    value = maxTokens,
                    // 512 keeps decode under ~60 s at Adreno 620's ~10 tok/s.
                    // Binary KV cache (2048 tokens, llama_sdpa_attention.cpp:43)
                    // could handle more, but UX gets painful past this.
                    range = 16f..512f,
                    steps = (512 - 16) - 1,
                    valueText = maxTokens.roundToInt().toString(),
                    onValueChange = { maxTokens = it },
                )
                SliderRow(
                    label = "TopK (1-100)",
                    value = topK,
                    range = 1f..100f,
                    steps = 99 - 1,
                    valueText = topK.roundToInt().toString(),
                    onValueChange = { topK = it },
                )
                SliderRow(
                    label = "TopP (0.00-1.00)",
                    value = topP,
                    range = 0f..1f,
                    steps = 0,
                    valueText = "%.2f".format(topP),
                    onValueChange = { topP = it },
                )
                SliderRow(
                    label = "Temperature (0.00-2.00)",
                    value = temperature,
                    range = 0f..2f,
                    steps = 0,
                    valueText = "%.2f".format(temperature),
                    onValueChange = { temperature = it },
                )

                HorizontalDivider(color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.12f))

                Text(
                    text = "MMS-TTS (text → speech)",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.primary,
                )
                SliderRow(
                    label = "Length scale (0.5 fast – 2.0 slow)",
                    value = lengthScale,
                    range = 0.5f..2.0f,
                    steps = 0,
                    valueText = "%.2f".format(lengthScale),
                    onValueChange = { lengthScale = it },
                )
                SliderRow(
                    label = "Noise scale (prior z, 0.0-1.5)",
                    value = noiseScale,
                    range = 0f..1.5f,
                    steps = 0,
                    valueText = "%.2f".format(noiseScale),
                    onValueChange = { noiseScale = it },
                )
                Text(
                    text = "Playback pacing",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                val prebufK = kotlin.math.ceil(currentRtf).toInt()
                Text(
                    text = "Adaptive prebuffer: RTF %.1f  →  prebuffer %d sentence%s before playback starts.".format(
                        currentRtf, prebufK, if (prebufK != 1) "s" else "",
                    ),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                )

                SliderRow(
                    label = "Noise scale w (duration, 0.0-1.5)",
                    value = noiseScaleW,
                    range = 0f..1.5f,
                    steps = 0,
                    valueText = "%.2f".format(noiseScaleW),
                    onValueChange = { noiseScaleW = it },
                )

                HorizontalDivider(color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.12f))

                Column {
                    Text(
                        text = "Accelerator",
                        style = MaterialTheme.typography.titleSmall,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Spacer(Modifier.height(6.dp))
                    AcceleratorPill()
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "Hand-written OpenCL on the Adreno GPU. " +
                               "CPU path isn't supported by this port.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    )
                }

                Text(
                    text = "Heads-up: changing SmolVLM knobs restarts the VLM session; " +
                           "changing MMS-TTS knobs restarts the TTS session. ~5–15 s warmup either way.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f),
                )
                }  // ── end scrollable middle region ─────────────────────────

                // Pinned footer — always visible, no scrolling needed to reach OK.
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End,
                ) {
                    TextButton(onClick = onDismiss) { Text("Cancel") }
                    Spacer(Modifier.width(8.dp))
                    TextButton(onClick = {
                        val s = SamplerSettings(
                            maxNewTokens = maxTokens.roundToInt().coerceIn(16, 512),
                            topK = topK.roundToInt().coerceIn(1, 100),
                            topP = topP.coerceIn(0f, 1f),
                            temperature = temperature.coerceIn(0f, 2f),
                            imageSize = if (imageSize == 512) 512 else 384,
                        )
                        val t = TtsSettings(
                            noiseScale = noiseScale.coerceIn(0f, 1.5f),
                            noiseScaleW = noiseScaleW.coerceIn(0f, 1.5f),
                            lengthScale = lengthScale.coerceIn(0.5f, 2.0f),
                        )
                        onConfirm(s, t, systemPrompt)
                    }) { Text("OK") }
                }
            }
        }
    }
}

@Composable
private fun SliderRow(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    steps: Int,
    valueText: String,
    onValueChange: (Float) -> Unit,
) {
    Column {
        Text(
            text = label,
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Spacer(Modifier.height(4.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Slider(
                value = value,
                onValueChange = onValueChange,
                valueRange = range,
                steps = steps,
                modifier = Modifier.weight(1f),
            )
            Spacer(Modifier.width(12.dp))
            Box(
                modifier = Modifier
                    .width(64.dp)
                    .height(36.dp),
                contentAlignment = Alignment.Center,
            ) {
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    tonalElevation = 0.dp,
                    modifier = Modifier.fillMaxWidth().height(36.dp),
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Text(
                            text = valueText,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface,
                            fontWeight = FontWeight.Medium,
                            textAlign = TextAlign.Center,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ImageSizeToggle(selected: Int, onSelect: (Int) -> Unit) {
    val options = listOf(
        Triple(384, "Fast", "384px · 36 tokens"),
        Triple(512, "Quality", "512px · 64 tokens"),
    )
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        options.forEach { (value, label, sub) ->
            val isSelected = value == selected
            Surface(
                shape = RoundedCornerShape(20.dp),
                color = if (isSelected) MaterialTheme.colorScheme.primary
                        else MaterialTheme.colorScheme.surfaceVariant,
                tonalElevation = 0.dp,
                modifier = Modifier
                    .weight(1f)
                    .clickable { onSelect(value) },
            ) {
                Column(
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        text = label,
                        style = MaterialTheme.typography.labelLarge,
                        color = if (isSelected) MaterialTheme.colorScheme.onPrimary
                                else MaterialTheme.colorScheme.onSurface,
                        fontWeight = FontWeight.Medium,
                    )
                    Text(
                        text = sub,
                        style = MaterialTheme.typography.bodySmall,
                        color = if (isSelected) MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.85f)
                                else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    )
                }
            }
        }
    }
}

@Composable
private fun AcceleratorPill() {
    Surface(
        shape = RoundedCornerShape(20.dp),
        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.15f),
        modifier = Modifier.padding(top = 2.dp),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "✓  GPU",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Medium,
            )
        }
    }
}
