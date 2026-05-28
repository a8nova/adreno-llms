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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.Memory
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.adreno.seeandsay.MainViewModel

/**
 * App home / mode picker. Two big cards: image chat vs. text-to-speech.
 * Settings is a small cog in the top-right. Device-status pill at the
 * bottom so the user sees what hardware is running everything.
 */
@Composable
fun HomeScreen(
    viewModel: MainViewModel,
    onChat: () -> Unit,
    onTts: () -> Unit,
    onLanguages: () -> Unit,
    onSettings: () -> Unit,
) {
    val device by viewModel.deviceInfo.collectAsStateWithLifecycle()
    val live by viewModel.liveRss.collectAsStateWithLifecycle()

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(20.dp),
    ) {
        // Top-right settings cog.
        FilledIconButton(
            onClick = onSettings,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .size(44.dp),
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                contentColor = MaterialTheme.colorScheme.onSurface,
            ),
        ) {
            Icon(Icons.Filled.Settings, contentDescription = "Settings", modifier = Modifier.size(22.dp))
        }

        Column(
            modifier = Modifier
                .align(Alignment.Center)
                .fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = "See & Say",
                style = MaterialTheme.typography.displaySmall,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = "On-device VLM + TTS",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            )
            Spacer(Modifier.height(36.dp))

            ModeCard(
                icon = Icons.AutoMirrored.Filled.Chat,
                title = "Chat",
                subtitle = "Send a message — text only, or attach a photo with +. " +
                           "SmolVLM runs the chat; toggle the speaker for spoken replies.",
                accent = MaterialTheme.colorScheme.primary,
                onClick = onChat,
            )
            Spacer(Modifier.height(16.dp))
            ModeCard(
                icon = Icons.Filled.GraphicEq,
                title = "Text to speech",
                subtitle = "Type text in any installed language; hear it spoken " +
                           "on-device via MMS-TTS (VITS).",
                accent = MaterialTheme.colorScheme.tertiary,
                onClick = onTts,
            )
            Spacer(Modifier.height(16.dp))
            ModeCard(
                icon = Icons.Filled.Language,
                title = "Languages",
                subtitle = "Download any of ~1140 MMS-TTS languages on demand. " +
                           "~70 MB per language pack.",
                accent = MaterialTheme.colorScheme.secondary,
                onClick = onLanguages,
            )
        }

        // Bottom: device-status pill.
        device?.let { d ->
            Surface(
                shape = RoundedCornerShape(20.dp),
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.75f),
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 12.dp),
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        Icons.Outlined.Memory,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(16.dp),
                    )
                    Spacer(Modifier.size(6.dp))
                    // Combined RSS of both inference subprocesses, vs the
                    // GPU's global memory total. RSS is process-side (CPU)
                    // memory which includes mmap'd weights — apples-to-
                    // oranges with GPU memory strictly, but it's the most
                    // useful single "what's the app using right now" signal.
                    val usedMb = ((live.smolvlmMb ?: 0) + (live.mmsttsMb ?: 0)).takeIf { it > 0 }
                    Text(
                        text = "Running on " +
                               d.device.replace(Regex("""\s*\(TM\)|\s*\(R\)|QUALCOMM\s*""", RegexOption.IGNORE_CASE), "").trim() +
                               " · " +
                               (if (usedMb != null) "$usedMb / ${d.globalMemMb} MB" else "${d.globalMemMb} MB"),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f),
                    )
                }
            }
        }
    }
}

@Composable
private fun ModeCard(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    subtitle: String,
    accent: Color,
    onClick: () -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(18.dp),
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 2.dp,
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 18.dp, vertical = 18.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Big accent-color circle with the icon.
            Surface(
                shape = RoundedCornerShape(50),
                color = accent.copy(alpha = 0.15f),
                modifier = Modifier.size(56.dp),
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(
                        icon,
                        contentDescription = null,
                        tint = accent,
                        modifier = Modifier.size(28.dp),
                    )
                }
            }
            Spacer(Modifier.size(14.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.SemiBold,
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.65f),
                )
            }
        }
    }
}
