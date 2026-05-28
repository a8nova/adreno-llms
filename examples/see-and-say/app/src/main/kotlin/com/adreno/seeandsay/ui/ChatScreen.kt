package com.adreno.seeandsay.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Translate
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material.icons.automirrored.filled.VolumeOff
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil.compose.AsyncImage
import com.adreno.seeandsay.ChatFlow
import com.adreno.seeandsay.ChatRole
import com.adreno.seeandsay.ChatTurn
import com.adreno.seeandsay.MainViewModel
import java.io.File

/**
 * Unified chat screen. One vertical list of turns; bottom input bar with
 * a `+` button that opens a "Take photo" / "Choose from gallery" sheet.
 * Sending without an attachment is a text-only chat (LM-only); sending with
 * an attachment is an image-grounded turn (silent /reset if mid-conversation).
 *
 * The auto-TTS toggle in the input bar plays the assistant response through
 * MMS-TTS when on. Default off — text chat is iterative.
 */
@Composable
fun ChatScreen(
    viewModel: MainViewModel,
    onOpenCameraCapture: () -> Unit,
) {
    val context = LocalContext.current
    val history by viewModel.chatHistory.collectAsStateWithLifecycle()
    val chatState by viewModel.chat.collectAsStateWithLifecycle()
    val pendingAttachment by viewModel.pendingAttachment.collectAsStateWithLifecycle()
    val ttsEnabled by viewModel.chatTtsEnabled.collectAsStateWithLifecycle()
    val language by viewModel.language.collectAsStateWithLifecycle()
    val sampler by viewModel.samplerSettings.collectAsStateWithLifecycle()
    val tts by viewModel.ttsSettings.collectAsStateWithLifecycle()
    val systemPrompt by viewModel.systemPrompt.collectAsStateWithLifecycle()
    val metrics by viewModel.metrics.collectAsStateWithLifecycle()

    var prompt by remember { mutableStateOf("") }
    var showAttachSheet by remember { mutableStateOf(false) }
    var showConfigs by remember { mutableStateOf(false) }
    val listState = rememberLazyListState()

    // Auto-scroll to the bottom whenever history grows OR the streaming turn
    // gets longer (text-length change inside the last assistant bubble).
    val lastText = history.lastOrNull()?.text ?: ""
    LaunchedEffect(history.size, lastText) {
        if (history.isNotEmpty()) listState.animateScrollToItem(history.size - 1)
    }

    val gallery = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
    ) { uri: Uri? ->
        if (uri != null) {
            val target = File(context.filesDir, "captures/chat_picked.jpg")
            target.parentFile?.mkdirs()
            context.contentResolver.openInputStream(uri)?.use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
            if (target.exists() && target.length() > 0) viewModel.setPendingAttachment(target)
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        Column(modifier = Modifier.fillMaxSize().imePadding()) {
            // ── Dynamic perf strip ───────────────────────────────────────
            // Single-line summary of last-turn perf. Updates live as tokens
            // stream and as TTS finishes. TTS section only renders when chat
            // TTS is enabled — otherwise the "RTF ·  MB" entries would always
            // dash out and waste horizontal space.
            ChatMetricsLine(metrics = metrics, showTts = ttsEnabled)

            // ── Top action row: new chat + settings ──────────────────────
            // Model identity lives in the global ModelStatusPill in MainActivity's
            // top bar (always visible on every non-Home screen).
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 6.dp),
                horizontalArrangement = Arrangement.End,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (history.isNotEmpty()) {
                    TextButton(onClick = { viewModel.newChat() }) {
                        Icon(Icons.Filled.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(Modifier.size(4.dp))
                        Text("New chat", maxLines = 1, softWrap = false)
                    }
                    Spacer(Modifier.size(6.dp))
                }
                FilledIconButton(
                    onClick = { showConfigs = true },
                    modifier = Modifier.size(40.dp),
                    colors = IconButtonDefaults.filledIconButtonColors(
                        containerColor = MaterialTheme.colorScheme.surface,
                        contentColor = MaterialTheme.colorScheme.onSurface,
                    ),
                ) { Icon(Icons.Filled.Tune, contentDescription = "Configurations") }
            }

            // ── Voice / language mismatch banner ──────────────────────────
            // SmolVLM-256M only outputs English. If the user has a non-English
            // MMS-TTS voice selected AND chat TTS is enabled, the English reply
            // gets fed to the wrong-language model and produces garbage audio.
            // Warn + offer one-tap switch to the bundled English pack.
            if (ttsEnabled && language != null && language!!.code != "eng") {
                LanguageMismatchBanner(
                    currentLang = language!!.label,
                    onSwitchToEnglish = { viewModel.setLanguage("eng") },
                )
            }

            // ── Conversation history ──────────────────────────────────────
            if (history.isEmpty()) {
                Box(modifier = Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
                    EmptyState()
                }
            } else {
                LazyColumn(
                    state = listState,
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    // The empty placeholder assistant bubble at history.last
                    // already represents "thinking" via its text — no separate
                    // ThinkingIndicator bubble below it.
                    items(history) { turn -> TurnBubble(turn) }
                }
            }

            // ── Input bar ─────────────────────────────────────────────────
            // Attachment thumbnail (shown above input when an image is pending).
            pendingAttachment?.let { jpeg ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Box {
                        AsyncImage(
                            model = jpeg,
                            contentDescription = "Attached image",
                            modifier = Modifier
                                .size(64.dp)
                                .clip(RoundedCornerShape(8.dp)),
                        )
                        FilledIconButton(
                            onClick = { viewModel.clearPendingAttachment() },
                            modifier = Modifier
                                .size(22.dp)
                                .align(Alignment.TopEnd),
                            colors = IconButtonDefaults.filledIconButtonColors(
                                containerColor = MaterialTheme.colorScheme.errorContainer,
                                contentColor = MaterialTheme.colorScheme.onErrorContainer,
                            ),
                        ) {
                            Icon(Icons.Filled.Close, contentDescription = "Remove image", modifier = Modifier.size(14.dp))
                        }
                    }
                    Spacer(Modifier.size(10.dp))
                    Text(
                        text = "Attached • ${jpeg.length() / 1024} KB",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                    )
                }
            }

            val busy = chatState is ChatFlow.Thinking ||
                       chatState is ChatFlow.Streaming ||
                       chatState is ChatFlow.Speaking
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 8.dp),
                verticalAlignment = Alignment.Bottom,
            ) {
                FilledIconButton(
                    onClick = { showAttachSheet = true },
                    enabled = !busy,
                    modifier = Modifier.size(44.dp),
                    colors = IconButtonDefaults.filledIconButtonColors(
                        containerColor = MaterialTheme.colorScheme.surface,
                        contentColor = MaterialTheme.colorScheme.onSurface,
                    ),
                ) { Icon(Icons.Filled.Add, contentDescription = "Attach") }

                Spacer(Modifier.size(6.dp))

                OutlinedTextField(
                    value = prompt,
                    onValueChange = { prompt = it },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Message") },
                    enabled = !busy,
                    maxLines = 4,
                    shape = RoundedCornerShape(20.dp),
                )

                Spacer(Modifier.size(6.dp))

                // Speaker toggle (auto-TTS on/off). Disabled if no language installed.
                val ttsAvailable = language != null
                FilledIconButton(
                    onClick = { viewModel.toggleChatTts() },
                    enabled = ttsAvailable,
                    modifier = Modifier.size(44.dp),
                    colors = IconButtonDefaults.filledIconButtonColors(
                        containerColor = if (ttsEnabled) MaterialTheme.colorScheme.tertiaryContainer
                                         else MaterialTheme.colorScheme.surface,
                        contentColor = if (ttsEnabled) MaterialTheme.colorScheme.onTertiaryContainer
                                       else MaterialTheme.colorScheme.onSurface,
                    ),
                ) {
                    Icon(
                        if (ttsEnabled) Icons.AutoMirrored.Filled.VolumeUp else Icons.AutoMirrored.Filled.VolumeOff,
                        contentDescription = if (ttsEnabled) "Auto-TTS on" else "Auto-TTS off",
                    )
                }

                Spacer(Modifier.size(6.dp))

                val canSend = !busy && (prompt.isNotBlank() || pendingAttachment != null)
                FilledIconButton(
                    onClick = {
                        val text = prompt.trim()
                        val img = pendingAttachment?.absolutePath
                        prompt = ""
                        viewModel.askChat(text, img)
                    },
                    enabled = canSend,
                    modifier = Modifier.size(44.dp),
                    colors = IconButtonDefaults.filledIconButtonColors(
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor = MaterialTheme.colorScheme.onPrimary,
                    ),
                ) { Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "Send") }
            }
        }

        // ── Failure banner ──────────────────────────────────────────────────
        (chatState as? ChatFlow.Failed)?.let { failed ->
            Surface(
                color = MaterialTheme.colorScheme.errorContainer,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 10.dp, start = 12.dp, end = 12.dp),
                shape = RoundedCornerShape(10.dp),
            ) {
                Text(
                    text = "Error: ${failed.message}",
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                )
            }
        }
    }

    if (showAttachSheet) {
        AttachmentSheet(
            onTakePhoto = { showAttachSheet = false; onOpenCameraCapture() },
            onChooseGallery = { showAttachSheet = false; gallery.launch("image/*") },
            onDismiss = { showAttachSheet = false },
        )
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


@Composable
private fun TurnBubble(turn: ChatTurn) {
    when (turn.role) {
        ChatRole.SYSTEM -> {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                Text(
                    text = turn.text,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
                    modifier = Modifier.padding(vertical = 8.dp),
                )
            }
        }
        ChatRole.USER -> {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                Column(
                    horizontalAlignment = Alignment.End,
                    modifier = Modifier.widthIn(max = 320.dp),
                ) {
                    turn.imagePath?.let { p ->
                        AsyncImage(
                            model = File(p),
                            contentDescription = "Sent image",
                            modifier = Modifier
                                .heightIn(max = 200.dp)
                                .clip(RoundedCornerShape(12.dp)),
                        )
                        if (turn.text.isNotBlank()) Spacer(Modifier.size(4.dp))
                    }
                    if (turn.text.isNotBlank()) {
                        Surface(
                            color = MaterialTheme.colorScheme.primary,
                            shape = RoundedCornerShape(14.dp),
                        ) {
                            Text(
                                text = turn.text,
                                color = MaterialTheme.colorScheme.onPrimary,
                                style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                            )
                        }
                    }
                }
            }
        }
        ChatRole.ASSISTANT -> {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Start) {
                Surface(
                    color = MaterialTheme.colorScheme.surface,
                    shape = RoundedCornerShape(14.dp),
                    tonalElevation = 1.dp,
                    modifier = Modifier.widthIn(max = 320.dp),
                ) {
                    val empty = turn.text.isEmpty()
                    Text(
                        text = if (empty) "Thinking…" else turn.text,
                        color = if (empty) MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f)
                                else MaterialTheme.colorScheme.onSurface,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun EmptyState() {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = "Start chatting",
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.onBackground,
            fontWeight = FontWeight.SemiBold,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = "Type a message, or tap + to attach a photo.\n" +
                   "Everything runs on-device.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            textAlign = androidx.compose.ui.text.style.TextAlign.Center,
        )
    }
}

/**
 * Single-line live perf summary parked at the top of the chat. Updates as the
 * view-model's AskMetrics flow ticks — decode tok/s lands when the turn ends,
 * TTS fields land when audio finishes. Renders nothing until there's at least
 * one number; then horizontally scrolls if the line overflows.
 *
 * TTS fields are omitted entirely when [showTts] is false to keep the strip
 * compact for text-only chats.
 */
@Composable
private fun ChatMetricsLine(
    metrics: com.adreno.seeandsay.AskMetrics,
    showTts: Boolean,
) {
    val hasAny = listOfNotNull(
        metrics.decodeTps, metrics.smolvlmTtfsSec,
        if (showTts) metrics.ttsRtf else null,
        if (showTts) metrics.ttsAudioSec else null,
    ).isNotEmpty()
    if (!hasAny) return

    val mono = MaterialTheme.typography.labelSmall.copy(fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace)
    val dim = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.55f)
    val accent = MaterialTheme.colorScheme.primary

    Surface(
        shape = RoundedCornerShape(10.dp),
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
    ) {
        Row(
            modifier = Modifier
                .padding(horizontal = 10.dp, vertical = 6.dp)
                .horizontalScroll(rememberScrollState()),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // SmolVLM perf — rss/peak intentionally omitted; the global
            // ModelStatusPill in MainActivity's top bar already shows live RSS.
            // TTFT = Time To First Token (LLM term). The TTS section below
            // uses TTFS = Time To First Sample (audio-output term).
            Text("⚡", color = accent, style = mono)
            Spacer(Modifier.size(6.dp))
            metricInline("decode", metrics.decodeTps?.let { "%.1f tok/s".format(it) }, accent, dim, mono)
            metricInline("ttft",   metrics.smolvlmTtfsSec?.let { "%.1fs".format(it) }, accent, dim, mono)

            // TTS perf — only when chat-TTS is enabled.
            if (showTts) {
                Spacer(Modifier.size(8.dp))
                Text("·", color = dim, style = mono)
                Spacer(Modifier.size(8.dp))
                Text("🔊", style = mono)
                Spacer(Modifier.size(6.dp))
                metricInline("rtf",   metrics.ttsRtf?.let { "%.2f".format(it) }, accent, dim, mono)
                metricInline("ttfs",  metrics.ttsTtfsSec?.let { "%.1fs".format(it) }, accent, dim, mono)
                metricInline("audio", metrics.ttsAudioSec?.let { "%.1fs".format(it) }, accent, dim, mono)
            }

            // End-to-end at the tail — the "did this feel slow?" single number.
            metrics.e2eSec?.let {
                Spacer(Modifier.size(8.dp))
                Text("·", color = dim, style = mono)
                Spacer(Modifier.size(8.dp))
                metricInline("e2e", "%.1fs".format(it), accent, dim, mono)
            }
        }
    }
}

@Composable
private fun metricInline(
    label: String,
    value: String?,
    accent: androidx.compose.ui.graphics.Color,
    dim: androidx.compose.ui.graphics.Color,
    mono: androidx.compose.ui.text.TextStyle,
) {
    if (value == null) return
    Text(label, color = dim, style = mono)
    Spacer(Modifier.size(2.dp))
    Text(value, color = accent, style = mono)
    Spacer(Modifier.size(8.dp))
}

@Composable
private fun LanguageMismatchBanner(
    currentLang: String,
    onSwitchToEnglish: () -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.tertiaryContainer,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 6.dp),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                Icons.Filled.WarningAmber,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onTertiaryContainer,
                modifier = Modifier.size(20.dp),
            )
            Spacer(Modifier.size(10.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    "Voice mismatch",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onTertiaryContainer,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    "Chat replies are in English, but your voice is set to $currentLang. " +
                    "Audio won't sound right.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onTertiaryContainer.copy(alpha = 0.85f),
                )
            }
            Spacer(Modifier.size(8.dp))
            TextButton(onClick = onSwitchToEnglish) {
                Icon(Icons.Filled.Translate, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.size(4.dp))
                Text("Use English", maxLines = 1, softWrap = false)
            }
        }
    }
}

@Composable
private fun AttachmentSheet(
    onTakePhoto: () -> Unit,
    onChooseGallery: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {},
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
        title = { Text("Attach an image") },
        text = {
            Column {
                AttachmentOption(
                    icon = Icons.Filled.PhotoCamera,
                    label = "Take photo",
                    onClick = onTakePhoto,
                )
                Spacer(Modifier.height(8.dp))
                AttachmentOption(
                    icon = Icons.Filled.Image,
                    label = "Choose from gallery",
                    onClick = onChooseGallery,
                )
            }
        },
    )
}

@Composable
private fun AttachmentOption(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    onClick: () -> Unit,
) {
    Surface(
        color = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(12.dp),
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.size(12.dp))
            Text(label, style = MaterialTheme.typography.bodyLarge)
        }
    }
}
