package com.adreno.seeandsay.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.MediaPlayer
import android.util.Log
import java.io.File
import kotlin.concurrent.thread

/**
 * Queue-aware WAV player. `enqueue()` either starts playback immediately if
 * the player is idle, or appends to the queue and plays it back when the
 * current clip finishes. This lets the caller stream sentence-level audio:
 * sentence 1 starts playing as soon as it's synthesized while sentences 2+
 * are still being generated in the background.
 *
 * AudioAttributes are USAGE_MEDIA + CONTENT_TYPE_SPEECH so playback routes
 * through the standard media stream (the volume rocker controls it).
 */
class WavPlayer {

    // Internal item type — either a WAV file (legacy) or raw PCM samples (PCM
    // stream mode). Same queue/onComplete plumbing handles both transparently.
    private sealed class Item(val onComplete: () -> Unit) {
        class WavFile(val file: File, onComplete: () -> Unit) : Item(onComplete)
        class Pcm(val samples: ShortArray, val sampleRate: Int, onComplete: () -> Unit) : Item(onComplete)
    }

    private val lock = Any()
    private val queue: ArrayDeque<Item> = ArrayDeque()
    private var current: MediaPlayer? = null
    private var currentTrack: AudioTrack? = null
    private var currentTrackThread: Thread? = null
    @Volatile private var stopped = false

    private val mediaAttrs: AudioAttributes = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_MEDIA)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()

    /** Plays [wav] right now, dropping any queued clips. */
    fun play(wav: File, onComplete: () -> Unit = {}) {
        stop()
        synchronized(lock) {
            stopped = false
            startInternalLocked(wav, onComplete)
        }
    }

    /** Appends [wav] to the queue; starts playback if nothing else is playing. */
    fun enqueue(wav: File, onComplete: () -> Unit = {}) {
        enqueueItem(Item.WavFile(wav, onComplete))
    }

    /** Appends a raw PCM int16 buffer (mono) for low-latency AudioTrack
     *  playback. Skips MediaPlayer.prepare() — playback starts in ~10-20 ms
     *  vs ~150-300 ms for a WAV File. Use this on the hot speak path. */
    fun enqueuePcm(samples: ShortArray, sampleRate: Int, onComplete: () -> Unit = {}) {
        enqueueItem(Item.Pcm(samples, sampleRate, onComplete))
    }

    private fun enqueueItem(item: Item) {
        synchronized(lock) {
            stopped = false
            if (current == null && currentTrack == null) {
                startItemLocked(item)
            } else {
                Log.d(TAG, "enqueue($item) — queued behind ${queue.size + 1} clip(s)")
                queue.add(item)
            }
        }
    }

    private fun startItemLocked(item: Item) {
        when (item) {
            is Item.WavFile -> startInternalLocked(item.file, item.onComplete)
            is Item.Pcm     -> startPcmLocked(item.samples, item.sampleRate, item.onComplete)
        }
    }

    private fun startPcmLocked(samples: ShortArray, sampleRate: Int, onComplete: () -> Unit) {
        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(samples.size * 2)
        val track = AudioTrack(
            AudioManager.STREAM_MUSIC,
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf,
            AudioTrack.MODE_STREAM,
        )
        track.setVolume(1.0f)
        currentTrack = track
        Log.d(TAG, "play PCM samples=${samples.size} sr=$sampleRate (~${samples.size * 1000L / sampleRate} ms)")
        track.play()
        // Spawn one writer thread per clip — write blocks the thread but lets
        // the AudioTrack hardware ramp immediately. On Adreno-class devices
        // first-sample latency is ~10-20 ms. When write returns, we wait for
        // the on-device buffer to drain and then fire onComplete + advance.
        val t = thread(name = "wavplayer-pcm", isDaemon = true) {
            try {
                var off = 0
                while (off < samples.size) {
                    val wrote = track.write(samples, off, samples.size - off)
                    if (wrote <= 0) break
                    off += wrote
                }
                // Wait for the device to actually play out the queued samples,
                // not just for the buffer to be filled. Sleep slightly longer
                // than the clip duration so the last samples come out audibly
                // before we tear the track down.
                val durMs = samples.size * 1000L / sampleRate
                Thread.sleep(durMs + 50)
            } catch (t: Throwable) {
                Log.w(TAG, "pcm writer thread failed", t)
            } finally {
                try { track.stop() } catch (_: Throwable) {}
                try { track.release() } catch (_: Throwable) {}
                Log.d(TAG, "pcm playback complete (n=${samples.size})")
                onComplete()
                synchronized(lock) {
                    if (currentTrack === track) {
                        currentTrack = null
                        currentTrackThread = null
                    }
                }
                advanceQueue()
            }
        }
        currentTrackThread = t
    }

    private fun startInternalLocked(wav: File, onComplete: () -> Unit) {
        Log.d(TAG, "play(${wav.absolutePath}, size=${wav.length()})")
        val mp = MediaPlayer().apply {
            setAudioAttributes(mediaAttrs)
            setVolume(1.0f, 1.0f)
            setDataSource(wav.absolutePath)
            setOnCompletionListener {
                Log.d(TAG, "playback complete: ${wav.name}")
                onComplete()
                releaseSafely(this)
                advanceQueue()
            }
            setOnErrorListener { _, what, extra ->
                Log.e(TAG, "MediaPlayer error what=$what extra=$extra clip=${wav.name}")
                releaseSafely(this)
                onComplete()
                advanceQueue()
                true
            }
            prepare()
            start()
        }
        current = mp
    }

    private fun advanceQueue() {
        synchronized(lock) {
            if (stopped) {
                queue.clear()
                return
            }
            val next = queue.removeFirstOrNull() ?: return
            startItemLocked(next)
        }
    }

    fun stop() {
        synchronized(lock) {
            stopped = true
            queue.clear()
            current?.let { releaseSafely(it) }
            current = null
            currentTrack?.let { t ->
                try { t.stop() } catch (_: Throwable) {}
                try { t.release() } catch (_: Throwable) {}
            }
            currentTrack = null
            currentTrackThread = null
        }
    }

    private fun releaseSafely(mp: MediaPlayer) {
        try { if (mp.isPlaying) mp.stop() } catch (_: IllegalStateException) {}
        try { mp.release() } catch (_: Throwable) {}
        if (current === mp) current = null
    }

    companion object {
        private const val TAG = "WavPlayer"
    }
}
