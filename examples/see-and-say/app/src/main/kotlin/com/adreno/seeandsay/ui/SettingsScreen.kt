package com.adreno.seeandsay.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.adreno.seeandsay.DeviceInfo
import com.adreno.seeandsay.MainViewModel

@Composable
fun SettingsScreen(viewModel: MainViewModel) {
    val current by viewModel.language.collectAsStateWithLifecycle()
    val device by viewModel.deviceInfo.collectAsStateWithLifecycle()
    val licenseAccepted by viewModel.licenseAccepted.collectAsStateWithLifecycle()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.Top,
    ) {
        Text("TTS language", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.height(6.dp))
        // Read-only display. The active-language picker lives on the
        // Text-to-speech screen now (next to the input field) so the user
        // changes it where they actually use it.
        Text(
            text = "Active: ${current?.label ?: "none"}",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Text(
            text = "Change from the Text-to-speech screen, or download more from Languages.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )

        Spacer(Modifier.height(16.dp))
        HorizontalDivider(color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.15f))
        Spacer(Modifier.height(16.dp))

        Text("How the chain runs", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.height(6.dp))
        Text(
            text = "Each ask spawns SmolVLM, waits for it to finish, then spawns MMS-TTS. " +
                "Both load weights from disk and rebuild OpenCL kernels on cold launch — " +
                "expect 5–15 s for the first run, faster after that as the GPU kernel cache warms.",
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.75f),
            style = MaterialTheme.typography.bodySmall,
        )

        Spacer(Modifier.height(12.dp))
        Text("On-device only", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.height(6.dp))
        Text(
            text = "Both models run via hand-written OpenCL on the Adreno GPU. " +
                "No network is used. Captured images and audio never leave the device.",
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.75f),
            style = MaterialTheme.typography.bodySmall,
        )

        Spacer(Modifier.height(16.dp))
        HorizontalDivider(color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.15f))
        Spacer(Modifier.height(16.dp))

        Text(
            "Device & compatibility",
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.primary,
        )
        Spacer(Modifier.height(6.dp))
        DeviceCompatibilitySection(device)

        Spacer(Modifier.height(16.dp))
        HorizontalDivider(color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.15f))
        Spacer(Modifier.height(16.dp))

        Text(
            "Models & licenses",
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.primary,
        )
        Spacer(Modifier.height(8.dp))

        LicenseEntry(
            name = "SmolVLM-256M-Instruct",
            holder = "HuggingFaceTB",
            license = "Apache-2.0",
            url = "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct",
        )
        Spacer(Modifier.height(10.dp))
        LicenseEntry(
            name = "MMS-TTS",
            holder = "Meta / FAIR",
            license = "CC-BY-NC 4.0  (non-commercial)",
            url = "https://huggingface.co/facebook/mms-tts",
            accepted = licenseAccepted,
        )
        Spacer(Modifier.height(10.dp))
        LicenseEntry(
            name = "uroman romanization tables",
            holder = "isi-nlp",
            license = "MIT",
            url = "https://github.com/isi-nlp/uroman",
        )
        Spacer(Modifier.height(10.dp))
        LicenseEntry(
            name = "See & Say + adreno-llms runtime",
            holder = "a8nova",
            license = "Apache-2.0",
            url = "https://github.com/a8nova/adreno-llms",
        )

        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun DeviceCompatibilitySection(info: DeviceInfo?) {
    if (info == null) {
        Text(
            text = "Device info will appear here once SmolVLM finishes booting.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )
        return
    }
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        YourDeviceCard(info)
        ModelCompatibilityCard(info)
        if (info.isCompatible) {
            LowLevelDetailsCard(info)
        }
    }
}

// ── Card 1: Your device ────────────────────────────────────────────────────
@Composable
private fun YourDeviceCard(info: DeviceInfo) {
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 2.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = "Your device",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.65f),
            )
            Spacer(Modifier.height(4.dp))

            // Phone model — pulled from android.os.Build. Most devices report
            // manufacturer as lowercase ("motorola", "samsung"); title-case it
            // so "Motorola razr (2020)" reads naturally.
            Text(
                text = formatPhoneModel(),
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.SemiBold,
            )

            // GPU name — Adreno 620 (number parsed from CL_DEVICE_VERSION;
            // CL_DEVICE_NAME on Adreno 620 only reports "QUALCOMM Adreno(TM)"
            // without the trailing model number, so we fish it out of the
            // version string).
            Text(
                text = formatGpuName(info),
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f),
            )
            Text(
                text = info.platform,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
            )
            Spacer(Modifier.height(12.dp))
            // 2x2 spec grid — most-actionable numbers at a glance.
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                SpecCell("GPU memory",  "${info.globalMemMb} MB", modifier = Modifier.weight(1f))
                SpecCell("Local mem",   "${info.localMemKb} KB",  modifier = Modifier.weight(1f))
            }
            Spacer(Modifier.height(8.dp))
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                SpecCell("Compute units", "${info.computeUnits}",     modifier = Modifier.weight(1f))
                SpecCell("Max workgroup", "${info.maxWorkgroup}",     modifier = Modifier.weight(1f))
            }
            if (info.maxClockMhz > 1) {
                Spacer(Modifier.height(8.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    SpecCell("Max clock",  "${info.maxClockMhz} MHz", modifier = Modifier.weight(1f))
                    SpecCell("OpenCL",     info.openclVersion.substringBefore(" Adreno").take(14), modifier = Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun SpecCell(label: String, value: String, modifier: Modifier = Modifier) {
    Surface(
        shape = RoundedCornerShape(10.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f),
        modifier = modifier.padding(horizontal = 2.dp),
    ) {
        Column(modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp)) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
            )
            Text(
                text = value,
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.SemiBold,
            )
        }
    }
}

// ── Card 2: Will the models run on this device? ────────────────────────────
@Composable
private fun ModelCompatibilityCard(info: DeviceInfo) {
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 2.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = "Will it run?",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.65f),
            )
            Spacer(Modifier.height(8.dp))

            // Hard blockers first — if any of these fail, NOTHING runs.
            if (!info.isCompatible) {
                Text(
                    text = "✗ Device does not meet minimum requirements",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.error,
                    fontWeight = FontWeight.SemiBold,
                )
                Spacer(Modifier.height(8.dp))
                if (!info.hasFp16)              FailLine("Missing cl_khr_fp16 — all weights are fp16, binary refuses to start")
                if (info.maxWorkgroup < 64)     FailLine("Max workgroup ${info.maxWorkgroup} < 64 — GEMV / RMSNorm kernels need WG=64")
                if (info.globalMemMb < 1024)    FailLine("Only ${info.globalMemMb} MB GPU memory — need ≥ 1024 MB for weights + activations")
                if (info.localMemKb < 12)       FailLine("Only ${info.localMemKb} KB local memory — need ≥ 12 KB for cooperative reductions")
                return@Column
            }

            // Two on-device models. SmolVLM has two image-resolution modes
            // (toggle in Configurations); MMS-TTS has multiple language packs.
            val quality512Ok = info.globalMemMb >= 1700
            ModelStatus(
                name        = "SmolVLM-256M-Instruct (image → text)",
                detail      = "~500 MB weights · Fast mode runs; Quality mode " +
                              (if (quality512Ok) "runs" else "tight, may OOM on this device"),
                status      = if (quality512Ok) ModelRunStatus.OK else ModelRunStatus.MAY_OOM,
                statusLabel = if (quality512Ok) "Runs" else "Use Fast mode",
            )
            Spacer(Modifier.height(8.dp))
            val amhOk = info.globalMemMb >= 1500
            ModelStatus(
                name        = "MMS-TTS (text → speech)",
                detail      = "~70 MB per language · downloaded on demand from HuggingFace" +
                              (if (!amhOk) " · keep inputs short" else ""),
                status      = if (amhOk) ModelRunStatus.OK else ModelRunStatus.MAY_OOM,
                statusLabel = if (amhOk) "Runs" else "Short inputs",
            )

            if (!info.isAdreno) {
                Spacer(Modifier.height(10.dp))
                Text(
                    text = "Kernels are tuned for Adreno wave-size 64 + Qualcomm extensions. " +
                           "On Mali / PowerVR everything runs but expect 2–5× slowdown.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.tertiary,
                )
            }
        }
    }
}

private enum class ModelRunStatus { OK, MAY_OOM, FAIL }

@Composable
private fun ModelStatus(
    name: String,
    detail: String,
    status: ModelRunStatus,
    statusLabel: String,
) {
    val (color, glyph) = when (status) {
        ModelRunStatus.OK      -> MaterialTheme.colorScheme.primary  to "✓"
        ModelRunStatus.MAY_OOM -> MaterialTheme.colorScheme.tertiary to "⚠"
        ModelRunStatus.FAIL    -> MaterialTheme.colorScheme.error    to "✗"
    }
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = glyph,
            style = MaterialTheme.typography.titleMedium,
            color = color,
            modifier = Modifier.padding(end = 8.dp),
        )
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = name,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.Medium,
            )
            Text(
                text = detail,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f),
            )
        }
        Surface(
            shape = RoundedCornerShape(8.dp),
            color = color.copy(alpha = 0.15f),
        ) {
            Text(
                text = statusLabel,
                style = MaterialTheme.typography.labelSmall,
                color = color,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
            )
        }
    }
}

@Composable
private fun FailLine(text: String) {
    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
        Text(
            text = "  ✗ ",
            color = MaterialTheme.colorScheme.error,
            style = MaterialTheme.typography.bodySmall,
        )
        Text(
            text = text,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f),
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

// ── Card 3: Low-level details (collapsible-flat dump, for the curious) ─────
@Composable
private fun LowLevelDetailsCard(info: DeviceInfo) {
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 1.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = "Low-level details",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.65f),
            )
            Spacer(Modifier.height(8.dp))
            FieldRow("OpenCL version",  info.openclVersion)
            FieldRow("Driver",          info.driver)
            FieldRow("Compute units",   "${info.computeUnits}")
            if (info.maxClockMhz > 1) FieldRow("Max clock",   "${info.maxClockMhz} MHz")
            FieldRow("Max workgroup",   "${info.maxWorkgroup}")
            FieldRow("Global memory",   "${info.globalMemMb} MB")
            FieldRow("Local memory",    "${info.localMemKb} KB")

            Spacer(Modifier.height(10.dp))
            Text(
                text = "Extensions",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
            )
            Spacer(Modifier.height(4.dp))
            ExtensionRow("cl_khr_fp16",               info.hasFp16,               required = true)
            ExtensionRow("cl_qcom_perf_hint",         info.hasQcomPerfHint,       required = false)
            ExtensionRow("cl_qcom_recordable_queues", info.hasQcomRecordableQueues, required = false)
            ExtensionRow("cl_qcom_dot_product8",      info.hasQcomDotProduct8,    required = false)
        }
    }
}

@Composable
private fun FieldRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground,
            fontWeight = FontWeight.Medium,
            textAlign = androidx.compose.ui.text.style.TextAlign.End,
            modifier = Modifier.weight(1f).padding(start = 12.dp),
        )
    }
}

@Composable
private fun ExtensionRow(name: String, present: Boolean, required: Boolean) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 1.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = if (present) "✓ " else (if (required) "✗ " else "○ "),
            style = MaterialTheme.typography.bodySmall,
            color = when {
                present  -> MaterialTheme.colorScheme.primary
                required -> MaterialTheme.colorScheme.error
                else     -> MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f)
            },
        )
        Text(
            text = name + (if (!required && !present) "  (optional)" else ""),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.85f),
        )
    }
}

@Composable
private fun ReqRow(label: String, ok: Boolean, rationale: String) {
    Column(modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = if (ok) "✓ " else "✗ ",
                style = MaterialTheme.typography.bodySmall,
                color = if (ok) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
            )
            Text(
                text = label,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onBackground,
                fontWeight = FontWeight.Medium,
            )
        }
        Text(
            text = rationale,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.55f),
            modifier = Modifier.padding(start = 16.dp),
        )
    }
}

@Composable
private fun LicenseEntry(
    name: String,
    holder: String,
    license: String,
    url: String,
    accepted: Boolean = false,
) {
    val uriHandler = LocalUriHandler.current
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { uriHandler.openUri(url) }
            .padding(vertical = 4.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = name,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onBackground,
                modifier = Modifier.weight(1f),
            )
            if (accepted) {
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.primary.copy(alpha = 0.15f),
                ) {
                    Text(
                        text = "✓ Accepted",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                    )
                }
            }
        }
        Spacer(Modifier.height(2.dp))
        Text(
            text = "$holder  ·  $license",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
        )
        Text(
            text = "Learn more and see license  ↗",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.primary,
        )
    }
}

/**
 * "motorola razr (2020)" → "Motorola razr (2020)". Build.MANUFACTURER comes
 * in lowercase on most OEMs; Build.MODEL may or may not redundantly include
 * the manufacturer string (Motorola, Samsung both do; Pixel doesn't). We
 * dedupe by checking if MODEL already starts with MANUFACTURER.
 */
private fun formatPhoneModel(): String {
    val mfg = android.os.Build.MANUFACTURER.orEmpty()
    val model = android.os.Build.MODEL.orEmpty()
    val mfgTitle = mfg.replaceFirstChar { it.uppercase() }
    return when {
        model.isEmpty() -> mfgTitle
        model.startsWith(mfg, ignoreCase = true) -> model.replaceFirstChar { it.uppercase() }
        else -> "$mfgTitle $model"
    }
}

/**
 * Extracts a clean GPU label like "Adreno 620" from the OpenCL device info.
 * CL_DEVICE_NAME on Adreno 620 ICDs is just "QUALCOMM Adreno(TM)" (no number);
 * the actual GPU number lives in CL_DEVICE_VERSION ("OpenCL 2.0 Adreno(TM) 620").
 * We grab the model number from the version string when device-name lacks it.
 */
private fun formatGpuName(info: DeviceInfo): String {
    val cleanedDeviceName = info.device
        .replace(Regex("""\s*\(TM\)|\s*\(R\)|QUALCOMM\s*""", RegexOption.IGNORE_CASE), "")
        .trim()
    if (Regex("""\d""").containsMatchIn(cleanedDeviceName)) return cleanedDeviceName
    val match = Regex("""Adreno(?:\(TM\))?\s+(\d+)""", RegexOption.IGNORE_CASE).find(info.openclVersion)
    val number = match?.groupValues?.get(1)
    return if (number != null) "$cleanedDeviceName $number" else cleanedDeviceName
}

