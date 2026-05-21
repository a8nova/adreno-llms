package com.adreno.seeandsay.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val AdrenoOrange = Color(0xFFFF6B1A)
private val AdrenoOrangeMuted = Color(0xFFBF4F12)
private val NearBlack = Color(0xFF0B0B0B)
private val Charcoal = Color(0xFF1A1A1A)
private val SoftWhite = Color(0xFFEDEDED)

private val DarkColors = darkColorScheme(
    primary = AdrenoOrange,
    onPrimary = NearBlack,
    primaryContainer = AdrenoOrangeMuted,
    onPrimaryContainer = SoftWhite,
    background = NearBlack,
    onBackground = SoftWhite,
    surface = Charcoal,
    onSurface = SoftWhite,
    surfaceVariant = Charcoal,
    onSurfaceVariant = SoftWhite,
)

private val LightColors = lightColorScheme(
    primary = AdrenoOrange,
    onPrimary = SoftWhite,
    background = SoftWhite,
    onBackground = NearBlack,
    surface = SoftWhite,
    onSurface = NearBlack,
)

@Composable
fun SeeAndSayTheme(
    darkTheme: Boolean = isSystemInDarkTheme() || true,
    content: @Composable () -> Unit,
) {
    val colors = if (darkTheme) DarkColors else LightColors
    MaterialTheme(colorScheme = colors, content = content)
}
