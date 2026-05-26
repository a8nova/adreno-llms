package com.adreno.seeandsay

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.activity.OnBackPressedCallback
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.ui.Alignment
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.adreno.seeandsay.ui.CameraScreen
import com.adreno.seeandsay.ui.HomeScreen
import com.adreno.seeandsay.ui.LanguagePickerScreen
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
                        UiPhase.Ready -> AppNavigator(viewModel, onExit = { finish() })
                        else -> LoaderScreen(phase)
                    }
                }
            }
        }
    }
}

/** Single screen the app currently displays. */
private enum class Route { HOME, CHAT, TTS, LANGUAGES, SETTINGS }

@Composable
private fun AppNavigator(viewModel: MainViewModel, onExit: () -> Unit) {
    var route by remember { mutableStateOf(Route.HOME) }

    // Wire the system back gesture: pop to Home if we're on a sub-screen,
    // exit the activity from Home (matches every other Android app).
    val activity = LocalContext.current as? ComponentActivity
    DisposableEffect(route, activity) {
        val cb = object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (route == Route.HOME) onExit() else route = Route.HOME
            }
        }
        activity?.onBackPressedDispatcher?.addCallback(cb)
        onDispose { cb.remove() }
    }

    // Scaffold with EXPLICIT zero-inset to short-circuit Samsung Android 16's
    // edge-to-edge reporting (the diagnostic borders confirmed Scaffold's
    // contentWindowInsets.top was being reported at ~50% of the tablet's
    // screen, double-counted with .statusBarsPadding() on the topBar Row).
    // We apply statusBarsPadding ONCE on the outer Scaffold modifier so the
    // top bar lands right under the status bar — no more, no less.
    androidx.compose.material3.Scaffold(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .statusBarsPadding()
            .navigationBarsPadding(),
        contentWindowInsets = androidx.compose.foundation.layout.WindowInsets(0, 0, 0, 0),
        topBar = {
            if (route != Route.HOME) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(MaterialTheme.colorScheme.background)
                        .padding(horizontal = 8.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    FilledIconButton(
                        onClick = { route = Route.HOME },
                        modifier = Modifier.size(40.dp),
                        colors = IconButtonDefaults.filledIconButtonColors(
                            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                            contentColor = MaterialTheme.colorScheme.onSurface,
                        ),
                    ) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back to home",
                            modifier = Modifier.size(20.dp),
                        )
                    }
                    Spacer(Modifier.width(10.dp))
                    Text(
                        text = when (route) {
                            Route.CHAT -> "Chat about images"
                            Route.TTS -> "Text to speech"
                            Route.LANGUAGES -> "Languages"
                            Route.SETTINGS -> "Settings"
                            Route.HOME -> ""
                        },
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onBackground,
                        modifier = Modifier.weight(1f),
                    )
                    ModelStatusPill(viewModel)
                }
            }
        },
        containerColor = MaterialTheme.colorScheme.background,
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            when (route) {
                Route.HOME -> HomeScreen(
                    viewModel = viewModel,
                    onChat = { route = Route.CHAT },
                    onTts = { route = Route.TTS },
                    onLanguages = { route = Route.LANGUAGES },
                    onSettings = { route = Route.SETTINGS },
                )
                Route.CHAT -> CameraScreen(viewModel)
                Route.TTS -> SpeakScreen(
                    viewModel = viewModel,
                    onManageLanguages = { route = Route.LANGUAGES },
                )
                Route.LANGUAGES -> LanguagePickerScreen(
                    viewModel = viewModel,
                    onUseLanguage = { route = Route.TTS },
                )
                Route.SETTINGS -> SettingsScreen(viewModel)
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
    val live by viewModel.liveRss.collectAsStateWithLifecycle()
    // NO fillMaxWidth — this pill is placed in a Row alongside a weight(1f)
    // Text title. fillMaxWidth on the pill collides with the title's weight,
    // and on Android 16 Samsung the resulting measurement explosion makes
    // the outer topBar Row report ~50% of screen height, pushing the entire
    // body content into the bottom half. wrapContentWidth keeps the pill at
    // its natural size.
    Row(
        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text("●", color = visionColor, style = MaterialTheme.typography.labelSmall)
        Spacer(Modifier.width(4.dp))
        Text(
            text = "vision" + (live.smolvlmMb?.let { " ${it} MB" } ?: ""),
            color = dim,
            style = MaterialTheme.typography.labelSmall,
        )
        Spacer(Modifier.width(12.dp))
        Text("●", color = voiceColor, style = MaterialTheme.typography.labelSmall)
        Spacer(Modifier.width(4.dp))
        Text(
            text = "voice" + (live.mmsttsMb?.let { " ${it} MB" } ?: ""),
            color = dim,
            style = MaterialTheme.typography.labelSmall,
        )
    }
}
