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
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

/**
 * One-time license gate shown on first launch (and again only if [LICENSE_KEY]
 * version bumps). Covers Facebook's MMS-TTS weights, which ship under
 * **CC-BY-NC 4.0** (non-commercial only). SmolVLM weights are Apache 2.0 and
 * don't need this — but we surface the project's own Apache 2.0 code license
 * here for transparency.
 *
 * The dialog blocks Home until the user checks "I agree" and taps Continue.
 * Acceptance is sticky via SharedPreferences (see MainViewModel.acceptLicense).
 */
@Composable
fun LicenseAgreementScreen(onAccept: () -> Unit) {
    var checked by remember { mutableStateOf(false) }
    var expanded by remember { mutableStateOf(false) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(horizontal = 20.dp, vertical = 16.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
        ) {
            Text(
                "Before you begin",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.onBackground,
                fontWeight = FontWeight.Bold,
            )
            Spacer(Modifier.height(6.dp))
            Text(
                "This app runs two on-device models. Each has its own license — " +
                "please review before continuing.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.75f),
            )

            Spacer(Modifier.height(20.dp))

            // ── SmolVLM-256M: Apache 2.0 ──────────────────────────────────
            LicenseCard(
                title = "SmolVLM-256M-Instruct",
                license = "Apache 2.0",
                summary = "Permissive open-source license. Commercial use allowed; " +
                          "attribution required.",
                tone = MaterialTheme.colorScheme.primary,
            )

            Spacer(Modifier.height(12.dp))

            // ── MMS-TTS: CC-BY-NC 4.0 ─────────────────────────────────────
            LicenseCard(
                title = "MMS-TTS (Facebook)",
                license = "CC-BY-NC 4.0",
                summary = "Non-commercial use only. Attribution required. " +
                          "Applies to every language pack — bundled and downloaded.",
                tone = MaterialTheme.colorScheme.tertiary,
                bullets = listOf(
                    "You may use, share, and adapt the audio outputs for personal, research, or educational purposes.",
                    "Commercial use of MMS-TTS audio (selling, advertising, monetized products) is NOT permitted under this license.",
                    "You must credit Meta AI / Facebook when you share generated audio.",
                    "Per-language packs are downloaded on demand from a HuggingFace mirror, but the license terms are identical to the upstream facebook/mms-tts-* repos.",
                ),
            )

            Spacer(Modifier.height(12.dp))

            // ── Expandable full-text section ──────────────────────────────
            Surface(
                shape = RoundedCornerShape(12.dp),
                color = MaterialTheme.colorScheme.surface,
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { expanded = !expanded },
            ) {
                Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            "Read full CC-BY-NC 4.0 terms",
                            style = MaterialTheme.typography.titleSmall,
                            color = MaterialTheme.colorScheme.onSurface,
                            modifier = Modifier.weight(1f),
                        )
                        Icon(
                            if (expanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                        )
                    }
                    if (expanded) {
                        Spacer(Modifier.height(10.dp))
                        Text(
                            FULL_CC_BY_NC_SUMMARY,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f),
                        )
                        Spacer(Modifier.height(8.dp))
                        Text(
                            "Full license: creativecommons.org/licenses/by-nc/4.0/legalcode",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                }
            }

            Spacer(Modifier.height(20.dp))

            // ── Agree checkbox ────────────────────────────────────────────
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { checked = !checked }
                    .padding(vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Checkbox(
                    checked = checked,
                    onCheckedChange = { checked = it },
                )
                Spacer(Modifier.size(4.dp))
                Text(
                    "I have read and agree to the terms above, including the non-commercial " +
                    "restriction on MMS-TTS audio.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onBackground,
                )
            }

            Spacer(Modifier.height(16.dp))

            Button(
                onClick = onAccept,
                enabled = checked,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Continue", style = MaterialTheme.typography.titleSmall)
            }

            Spacer(Modifier.height(40.dp))
        }
    }
}

@Composable
private fun LicenseCard(
    title: String,
    license: String,
    summary: String,
    tone: androidx.compose.ui.graphics.Color,
    bullets: List<String> = emptyList(),
) {
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 1.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    title,
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f),
                )
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = tone.copy(alpha = 0.15f),
                ) {
                    Text(
                        license,
                        style = MaterialTheme.typography.labelMedium,
                        color = tone,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    )
                }
            }
            Spacer(Modifier.height(6.dp))
            Text(
                summary,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f),
            )
            if (bullets.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
                for (b in bullets) {
                    Row(
                        modifier = Modifier.padding(vertical = 2.dp),
                        verticalAlignment = Alignment.Top,
                    ) {
                        Text(
                            "•",
                            style = MaterialTheme.typography.bodySmall,
                            color = tone,
                            modifier = Modifier.padding(end = 6.dp),
                        )
                        Text(
                            b,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.75f),
                        )
                    }
                }
            }
        }
    }
}

private val FULL_CC_BY_NC_SUMMARY = """
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

You are free to:
  • Share — copy and redistribute the material in any medium or format
  • Adapt — remix, transform, and build upon the material

Under the following terms:
  • Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
  • NonCommercial — You may not use the material for commercial purposes.
  • No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

This is a human-readable summary of (and not a substitute for) the full license. The full legal text is at the URL below.
""".trimIndent()
