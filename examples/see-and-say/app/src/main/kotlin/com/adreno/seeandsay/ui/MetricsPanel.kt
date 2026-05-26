package com.adreno.seeandsay.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.adreno.seeandsay.AskMetrics

/**
 * Compact per-query metrics block, parked at the bottom of the camera screen.
 * Lets us read perf numbers off the device without adb during ad-hoc testing
 * (peak memory, decode tok/s, time-to-first-sample, RTF, etc.).
 *
 * Designed to be removable later — all data flows from [AskMetrics] in the
 * ViewModel. Delete this composable + its caller and the rest of the app is
 * unaffected.
 */
@Composable
fun MetricsPanel(metrics: AskMetrics, modifier: Modifier = Modifier) {
    val hasAny = listOfNotNull(
        metrics.prefillTps, metrics.decodeTps, metrics.smolvlmRssMb,
        metrics.ttsAudioSec, metrics.ttsRtf, metrics.mmsttsRssMb, metrics.e2eSec,
    ).isNotEmpty()
    if (!hasAny) return

    val mono = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace)
    val dim = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f)
    val accent = MaterialTheme.colorScheme.primary.copy(alpha = 0.9f)

    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.65f))
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Text("PERF — last query", color = accent, style = mono)
        Spacer(Modifier.height(2.dp))

        Section(title = "SmolVLM", color = accent, mono = mono, dim = dim) {
            metric("ttfs",     metrics.smolvlmTtfsSec?.let { "%.2f s".format(it) }, mono, dim)
            metric("prefill",  metrics.prefillSec?.let { sec ->
                val tok = metrics.prefillTokens ?: 0
                val tps = metrics.prefillTps ?: 0.0
                "%d tok · %.2f s · %.1f tok/s".format(tok, sec, tps)
            }, mono, dim)
            metric("decode",   metrics.decodeSec?.let { sec ->
                val tok = metrics.decodeTokens ?: 0
                val tps = metrics.decodeTps ?: 0.0
                "%d tok · %.2f s · %.1f tok/s".format(tok, sec, tps)
            }, mono, dim)
            metric("ctx",      metrics.ctxPos?.toString(), mono, dim)
            metric("rss",      metrics.smolvlmRssMb?.let { "%d MB".format(it) }, mono, dim)
            metric("peak rss", metrics.smolvlmPeakRssMb?.let { "%d MB".format(it) }, mono, dim)
        }

        Spacer(Modifier.height(4.dp))

        Section(title = "MMS-TTS", color = accent, mono = mono, dim = dim) {
            metric("ttfs",      metrics.ttsTtfsSec?.let { "%.2f s".format(it) }, mono, dim)
            metric("fwd",       metrics.ttsFwdSec?.let { "%.2f s".format(it) }, mono, dim)
            metric("audio",     metrics.ttsAudioSec?.let { "%.2f s".format(it) }, mono, dim)
            metric("rtf",       metrics.ttsRtf?.let { "%.2f".format(it) }, mono, dim)
            metric("rss",       metrics.mmsttsRssMb?.let { "%d MB".format(it) }, mono, dim)
            metric("peak rss",  metrics.mmsttsPeakRssMb?.let { "%d MB".format(it) }, mono, dim)
        }

        // Combined peak across both subprocesses — the number you actually care
        // about for "will this device OOM?".
        val combinedPeak = listOfNotNull(metrics.smolvlmPeakRssMb, metrics.mmsttsPeakRssMb).sum()
        if (combinedPeak > 0) {
            Spacer(Modifier.height(4.dp))
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("combined peak", color = accent, style = mono)
                Text("%d MB".format(combinedPeak), color = MaterialTheme.colorScheme.onSurface, style = mono)
            }
        }

        if (metrics.e2eSec != null) {
            Spacer(Modifier.height(4.dp))
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("end-to-end", color = accent, style = mono)
                Text("%.2f s".format(metrics.e2eSec), color = MaterialTheme.colorScheme.onSurface, style = mono)
            }
        }
    }
}

@Composable
private fun Section(
    title: String,
    color: androidx.compose.ui.graphics.Color,
    mono: androidx.compose.ui.text.TextStyle,
    dim: androidx.compose.ui.graphics.Color,
    content: @Composable () -> Unit,
) {
    Text(title, color = color, style = mono)
    content()
}

@Composable
private fun metric(
    label: String,
    value: String?,
    mono: androidx.compose.ui.text.TextStyle,
    dim: androidx.compose.ui.graphics.Color,
) {
    // Always render the row so the panel layout is stable and the user can
    // tell at a glance whether the metric is "not measured yet" (—) vs the
    // row simply not existing.
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text("  $label", color = dim, style = mono)
        Text(value ?: "—", color = MaterialTheme.colorScheme.onSurface, style = mono)
    }
}
