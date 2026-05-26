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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
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
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.adreno.seeandsay.MainViewModel
import com.adreno.seeandsay.runner.LanguageRegistry
import kotlinx.coroutines.launch

/**
 * Browse-and-download UI for MMS-TTS language packs. Auto-fetches the
 * registry from HF on first open; the list survives across opens via the
 * disk cache.
 *
 * `onUseLanguage` is invoked after the user activates (taps "Use this") an
 * installed/bundled language — typically used by the host to navigate back
 * to the Speak screen so the user can immediately try the new voice.
 */
@Composable
fun LanguagePickerScreen(
    viewModel: MainViewModel,
    onUseLanguage: () -> Unit = {},
) {
    val registry = viewModel.languageRegistry
    val entries by registry.entries.collectAsStateWithLifecycle()
    val active by viewModel.language.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    var query by remember { mutableStateOf("") }
    var refreshing by remember { mutableStateOf(false) }
    var refreshError by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        // Auto-refresh on first open if the cache is empty — gets the user to
        // a populated list without an explicit tap. Failures fall back to
        // the bundled-only view; user can tap Refresh to retry.
        if (entries.size <= 2) {
            refreshing = true
            val r = registry.refreshRegistry()
            refreshError = r.exceptionOrNull()?.message
            refreshing = false
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 12.dp),
    ) {
        // Search row + refresh.
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp, bottom = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                modifier = Modifier.weight(1f),
                leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null, modifier = Modifier.size(18.dp)) },
                placeholder = { Text("Search ${entries.size} languages…", style = MaterialTheme.typography.bodySmall) },
                singleLine = true,
                textStyle = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.size(8.dp))
            FilledIconButton(
                onClick = {
                    scope.launch {
                        refreshing = true
                        val r = registry.refreshRegistry()
                        refreshError = r.exceptionOrNull()?.message
                        refreshing = false
                    }
                },
                modifier = Modifier.size(44.dp),
                enabled = !refreshing,
                colors = IconButtonDefaults.filledIconButtonColors(
                    containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                    contentColor = MaterialTheme.colorScheme.onSurface,
                ),
            ) {
                Icon(Icons.Filled.Refresh, contentDescription = "Refresh list", modifier = Modifier.size(20.dp))
            }
        }
        if (refreshError != null && entries.size <= 2) {
            Text(
                text = "Couldn't fetch language list: $refreshError",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(vertical = 4.dp),
            )
        }

        // Summary counts — gives the user a sense of scale.
        val installed = entries.count { it.status is LanguageRegistry.LangEntry.Status.Installed }
        var showWipeConfirm by remember { mutableStateOf(false) }
        Row(
            modifier = Modifier.fillMaxWidth().padding(start = 4.dp, bottom = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "$installed installed · ${entries.size - installed} available",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.55f),
            )
            if (installed > 0) {
                Spacer(Modifier.weight(1f))
                TextButton(onClick = { showWipeConfirm = true }) {
                    Text("Wipe all", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.error)
                }
            }
        }
        if (showWipeConfirm) {
            androidx.compose.material3.AlertDialog(
                onDismissRequest = { showWipeConfirm = false },
                title = { Text("Wipe all languages?") },
                text = { Text("This deletes all downloaded language packs. You'll need to re-download at least one to use TTS.") },
                confirmButton = {
                    TextButton(onClick = { showWipeConfirm = false; viewModel.wipeAllLanguages() }) {
                        Text("Wipe", color = MaterialTheme.colorScheme.error)
                    }
                },
                dismissButton = { TextButton(onClick = { showWipeConfirm = false }) { Text("Cancel") } },
            )
        }
        Text(
            text = "~70 MB per language, downloaded from HuggingFace",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.4f),
            modifier = Modifier.padding(start = 4.dp, bottom = 6.dp),
        )

        val filtered = remember(entries, query) {
            if (query.isBlank()) entries
            else entries.filter { e ->
                val q = query.lowercase()
                q in e.code.lowercase() ||
                q in e.name.lowercase() ||
                q in e.nativeName.lowercase()
            }
        }

        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(6.dp),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(bottom = 24.dp),
        ) {
            items(filtered, key = { it.code }) { e ->
                LangRow(
                    entry = e,
                    isActive = e.code == active?.code,
                    onDownload = { registry.download(e.code) },
                    onCancel = { registry.cancel(e.code) },
                    onRetry = { registry.download(e.code) },
                    onUse = {
                        viewModel.setLanguage(e.code)
                        onUseLanguage()
                    },
                )
            }
            if (filtered.isEmpty()) {
                item {
                    Text(
                        text = if (query.isBlank())
                            "No languages yet. Tap refresh to fetch the list from HuggingFace."
                        else
                            "No matches for \"$query\"",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                        modifier = Modifier.padding(16.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun LangRow(
    entry: LanguageRegistry.LangEntry,
    isActive: Boolean,
    onDownload: () -> Unit,
    onCancel: () -> Unit,
    onRetry: () -> Unit,
    onUse: () -> Unit,
) {
    val installedOrBundled = entry.status is LanguageRegistry.LangEntry.Status.Installed
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = if (isActive)
            MaterialTheme.colorScheme.primary.copy(alpha = 0.10f)
        else
            MaterialTheme.colorScheme.surface,
        tonalElevation = 1.dp,
        modifier = Modifier
            .fillMaxWidth()
            .then(if (installedOrBundled && !isActive) Modifier.clickable(onClick = onUse) else Modifier),
    ) {
        Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    // Native name first (most distinctive for the user), then
                    // English name + size on the next line.
                    Text(
                        text = entry.nativeName.ifBlank { entry.name },
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                        fontWeight = FontWeight.Medium,
                    )
                    val subtitle = buildString {
                        if (entry.name != entry.nativeName && entry.name.isNotBlank()) {
                            append(entry.name)
                        }
                        if (entry.script.isNotBlank()) {
                            if (isNotEmpty()) append(" · ")
                            append(entry.script)
                        }
                        if (isNotEmpty()) append(" · ")
                        append(entry.code)
                        if (entry.sizeBytes > 0) {
                            append(" · ${entry.sizeBytes / 1024 / 1024} MB")
                        }
                    }
                    Text(
                        text = subtitle,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    )
                }
                Spacer(Modifier.size(8.dp))
                StatusAction(entry, isActive, onDownload, onCancel, onRetry, onUse)
            }
            // Downloading: progress bar below the row.
            (entry.status as? LanguageRegistry.LangEntry.Status.Downloading)?.let { d ->
                Spacer(Modifier.size(8.dp))
                LinearProgressIndicator(
                    progress = { d.percent },
                    modifier = Modifier.fillMaxWidth().height(4.dp),
                    color = MaterialTheme.colorScheme.primary,
                    trackColor = MaterialTheme.colorScheme.surfaceVariant,
                )
                Text(
                    text = "${d.bytesDone / 1024 / 1024} / ${d.bytesTotal / 1024 / 1024} MB",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f),
                )
            }
        }
    }
}

@Composable
private fun StatusAction(
    entry: LanguageRegistry.LangEntry,
    isActive: Boolean,
    onDownload: () -> Unit,
    onCancel: () -> Unit,
    onRetry: () -> Unit,
    onUse: () -> Unit,
) {
    when (val s = entry.status) {
        is LanguageRegistry.LangEntry.Status.Installed -> {
            if (isActive) {
                InstalledPill("In use")
            } else {
                TextButton(onClick = onUse, contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 12.dp, vertical = 6.dp)) {
                    Text("Use", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Medium)
                }
            }
        }
        is LanguageRegistry.LangEntry.Status.Available -> {
            FilledIconButton(
                onClick = onDownload,
                modifier = Modifier.size(40.dp),
                colors = IconButtonDefaults.filledIconButtonColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = MaterialTheme.colorScheme.onPrimary,
                ),
            ) {
                Icon(Icons.Filled.CloudDownload, contentDescription = "Download", modifier = Modifier.size(20.dp))
            }
        }
        is LanguageRegistry.LangEntry.Status.Downloading -> {
            FilledIconButton(
                onClick = onCancel,
                modifier = Modifier.size(40.dp),
                colors = IconButtonDefaults.filledIconButtonColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                    contentColor = MaterialTheme.colorScheme.error,
                ),
            ) {
                Icon(Icons.Filled.Cancel, contentDescription = "Cancel", modifier = Modifier.size(20.dp))
            }
        }
        is LanguageRegistry.LangEntry.Status.Failed -> {
            Column(horizontalAlignment = Alignment.End) {
                TextButton(onClick = onRetry, contentPadding = androidx.compose.foundation.layout.PaddingValues(8.dp)) {
                    Text("Retry", style = MaterialTheme.typography.labelMedium)
                }
                Text(
                    text = s.message.take(40),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        }
    }
}

@Composable
private fun InstalledPill(label: String) {
    Surface(
        shape = RoundedCornerShape(10.dp),
        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.15f),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                Icons.Filled.CheckCircle,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(14.dp),
            )
            Spacer(Modifier.size(4.dp))
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Medium,
            )
        }
    }
}
