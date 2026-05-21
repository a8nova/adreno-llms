package com.adreno.seeandsay

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.ui.Alignment
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TabRow
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import com.adreno.seeandsay.ui.CameraScreen
import com.adreno.seeandsay.ui.LoaderScreen
import com.adreno.seeandsay.ui.SettingsScreen
import com.adreno.seeandsay.ui.SpeakScreen
import com.adreno.seeandsay.ui.theme.SeeAndSayTheme

class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()

    /** Re-apply the black window background after every config change. */
    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        super.onConfigurationChanged(newConfig)
        window.setBackgroundDrawable(android.graphics.drawable.ColorDrawable(android.graphics.Color.BLACK))
        window.decorView.invalidate()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Pin a black drawable at the WindowManager level so any transient state
        // (fold/unfold transitions, Activity restarts the OS might still trigger,
        // task-snapshot reuse) renders against black rather than the system
        // default. windowBackground in XML is one path; this is the belt.
        window.setBackgroundDrawable(android.graphics.drawable.ColorDrawable(android.graphics.Color.BLACK))
        // Listen for display additions/removals — on Razr, unfolding adds the
        // inner display. We use this as a signal to force a black redraw the
        // moment a new display appears, masking any system-side transition.
        val displayManager = getSystemService(android.hardware.display.DisplayManager::class.java)
        displayManager?.registerDisplayListener(object : android.hardware.display.DisplayManager.DisplayListener {
            override fun onDisplayAdded(displayId: Int) {
                runOnUiThread { window.decorView.invalidate() }
            }
            override fun onDisplayRemoved(displayId: Int) {
                runOnUiThread { window.decorView.invalidate() }
            }
            override fun onDisplayChanged(displayId: Int) {
                runOnUiThread { window.decorView.invalidate() }
            }
        }, android.os.Handler(mainLooper))
        enableEdgeToEdge()
        setContent {
            SeeAndSayTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    val phase by viewModel.phase.collectAsStateWithLifecycle()
                    when (phase) {
                        UiPhase.Ready -> MainTabs(viewModel)
                        else -> LoaderScreen(phase)
                    }
                }
            }
        }
    }
}

@Composable
private fun MainTabs(viewModel: MainViewModel) {
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("📷  See", "🔊  Speak", "⚙  Settings")

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(top = 16.dp),
    ) {
        // Persistent model-status pill — always shows what's loaded, so the
        // user never has to wonder "is SmolVLM ready? is MMS-TTS ready?".
        ModelStatusPill(viewModel)

        // Color the selected tab + indicator with the same primary orange used
        // by the "Ask" / "Ask follow-up" buttons so the navigation reads as
        // part of the same design language.
        TabRow(
            selectedTabIndex = selectedTab,
            containerColor = MaterialTheme.colorScheme.background,
            contentColor = MaterialTheme.colorScheme.primary,
            indicator = { tabPositions ->
                if (selectedTab < tabPositions.size) {
                    androidx.compose.material3.TabRowDefaults.SecondaryIndicator(
                        modifier = Modifier.tabIndicatorOffset(tabPositions[selectedTab]),
                        color = MaterialTheme.colorScheme.primary,
                        height = 3.dp,
                    )
                }
            },
        ) {
            tabs.forEachIndexed { i, title ->
                Tab(
                    selected = selectedTab == i,
                    onClick = { selectedTab = i },
                    selectedContentColor = MaterialTheme.colorScheme.primary,
                    unselectedContentColor = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.55f),
                    text = {
                        Text(
                            text = title,
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = if (selectedTab == i)
                                androidx.compose.ui.text.font.FontWeight.SemiBold
                            else androidx.compose.ui.text.font.FontWeight.Normal,
                        )
                    },
                )
            }
        }
        Box(
            modifier = Modifier.weight(1f).fillMaxSize(),
        ) {
            when (selectedTab) {
                0 -> CameraScreen(viewModel)
                1 -> SpeakScreen(viewModel)
                else -> SettingsScreen(viewModel)
            }
        }
    }
}

@Composable
private fun ModelStatusPill(viewModel: MainViewModel) {
    val (visionAlive, voiceAlive) = remember { viewModel.modelsAlive() }
    val visionColor = if (visionAlive) androidx.compose.ui.graphics.Color(0xFF22C55E) else MaterialTheme.colorScheme.error
    val voiceColor = if (voiceAlive) androidx.compose.ui.graphics.Color(0xFF22C55E) else MaterialTheme.colorScheme.error
    val dim = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.55f)
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text("●", color = visionColor, style = MaterialTheme.typography.labelSmall)
        Spacer(Modifier.width(4.dp))
        Text("vision", color = dim, style = MaterialTheme.typography.labelSmall)
        Spacer(Modifier.width(12.dp))
        Text("●", color = voiceColor, style = MaterialTheme.typography.labelSmall)
        Spacer(Modifier.width(4.dp))
        Text("voice", color = dim, style = MaterialTheme.typography.labelSmall)
    }
}
