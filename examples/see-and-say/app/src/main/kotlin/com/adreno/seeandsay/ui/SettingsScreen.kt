package com.adreno.seeandsay.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.selection.selectable
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.adreno.seeandsay.MainViewModel
import com.adreno.seeandsay.TtsLanguage

@Composable
fun SettingsScreen(viewModel: MainViewModel) {
    val current by viewModel.language.collectAsStateWithLifecycle()

    Column(
        modifier = Modifier.fillMaxSize().padding(horizontal = 14.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.Top,
    ) {
        Text("TTS language", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.height(8.dp))

        TtsLanguage.values().forEach { lang ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .selectable(
                        selected = current == lang,
                        enabled = lang.available,
                        role = Role.RadioButton,
                        onClick = { viewModel.setLanguage(lang) },
                    )
                    .padding(vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                RadioButton(
                    selected = current == lang,
                    onClick = { if (lang.available) viewModel.setLanguage(lang) },
                    enabled = lang.available,
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    text = lang.label,
                    color = if (lang.available) MaterialTheme.colorScheme.onBackground
                            else MaterialTheme.colorScheme.onBackground.copy(alpha = 0.4f),
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }

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
    }
}

