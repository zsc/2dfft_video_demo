#!/usr/bin/env python3
"""
Video 2D FFT Processor
Processes a video by converting to grayscale and applying 2D FFT to each frame.
Generates an HTML visualization with synchronized playback.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import base64


def process_frame_fft(frame):
    """
    Apply 2D FFT to a grayscale frame.
    Returns the log magnitude spectrum with proper normalization.
    """
    # Apply 2D FFT
    fft = np.fft.fft2(frame)
    # Shift zero-frequency component to center
    fft_shifted = np.fft.fftshift(fft)
    # Calculate magnitude spectrum
    magnitude = np.abs(fft_shifted)
    # Apply log transform (add 1 to avoid log(0))
    log_magnitude = np.log1p(magnitude)
    # Normalize to 0-255 range for visualization
    normalized = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return normalized.astype(np.uint8)


def process_video(input_path, output_path):
    """
    Process video: convert to grayscale and apply 2D FFT to each frame.
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    # Create video writer with H.264 codec for browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply 2D FFT
        fft_frame = process_frame_fft(gray)

        # Convert back to BGR for video output
        fft_bgr = cv2.cvtColor(fft_frame, cv2.COLOR_GRAY2BGR)

        # Write frame
        out.write(fft_bgr)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()

    print(f"Processing complete! Output saved to: {output_path}")
    return width, height, fps, total_frames


def generate_html(input_video, output_video, html_path):
    """
    Generate HTML file with synchronized video playback.
    """
    # Convert video paths to relative paths for HTML
    input_name = Path(input_video).name
    output_name = Path(output_video).name

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2D FFT Video Comparison</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #fff;
            font-family: Arial, sans-serif;
        }
        .container {
            max-width: 1800px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
        }
        .video-container {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .video-wrapper {
            flex: 1;
            min-width: 400px;
            max-width: 800px;
        }
        .video-wrapper h2 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 18px;
        }
        video {
            width: 100%;
            height: auto;
            background-color: #000;
            display: block;
        }
        .controls {
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .control-group:last-child {
            margin-bottom: 0;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #45a049;
        }
        button:active {
            background-color: #3d8b40;
        }
        .speed-control {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .speed-btn {
            padding: 8px 15px;
            font-size: 14px;
            background-color: #2196F3;
        }
        .speed-btn:hover {
            background-color: #0b7dda;
        }
        .speed-btn.active {
            background-color: #0b7dda;
            font-weight: bold;
        }
        input[type="range"] {
            flex: 1;
            min-width: 200px;
            height: 6px;
            background: #444;
            border-radius: 3px;
            outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 16px;
            height: 16px;
            background: #4CAF50;
            cursor: pointer;
            border-radius: 50%;
        }
        input[type="range"]::-moz-range-thumb {
            width: 16px;
            height: 16px;
            background: #4CAF50;
            cursor: pointer;
            border-radius: 50%;
            border: none;
        }
        .time-display {
            font-family: monospace;
            font-size: 16px;
            min-width: 150px;
        }
        label {
            font-weight: bold;
            min-width: 80px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>2D FFT Video Analysis</h1>

        <div class="controls">
            <div class="control-group">
                <button id="playPauseBtn" onclick="togglePlayPause()">▶ Play</button>
                <button onclick="stepBackward()">⏮ -1 Frame</button>
                <button onclick="stepForward()">⏭ +1 Frame</button>
                <div class="time-display">
                    <span id="currentTime">0:00.00</span> / <span id="duration">0:00.00</span>
                </div>
            </div>

            <div class="control-group">
                <label>Progress:</label>
                <input type="range" id="seekBar" value="0" min="0" max="100" step="0.1" oninput="seek()">
            </div>

            <div class="control-group">
                <label>Speed:</label>
                <div class="speed-control">
                    <button class="speed-btn" onclick="setSpeed(0.25)">0.25x</button>
                    <button class="speed-btn" onclick="setSpeed(0.5)">0.5x</button>
                    <button class="speed-btn active" onclick="setSpeed(1)">1x</button>
                    <button class="speed-btn" onclick="setSpeed(1.5)">1.5x</button>
                    <button class="speed-btn" onclick="setSpeed(2)">2x</button>
                </div>
            </div>
        </div>

        <div class="video-container">
            <div class="video-wrapper">
                <h2>Original Video (Grayscale)</h2>
                <video id="video1" src="INPUT_VIDEO"></video>
            </div>
            <div class="video-wrapper">
                <h2>2D FFT (Log Magnitude Spectrum)</h2>
                <video id="video2" src="OUTPUT_VIDEO"></video>
            </div>
        </div>
    </div>

    <script>
        const video1 = document.getElementById('video1');
        const video2 = document.getElementById('video2');
        const playPauseBtn = document.getElementById('playPauseBtn');
        const seekBar = document.getElementById('seekBar');
        const currentTimeDisplay = document.getElementById('currentTime');
        const durationDisplay = document.getElementById('duration');

        let isSeeking = false;

        // Synchronize videos
        function syncVideos(source) {
            if (!isSeeking) {
                const target = source === video1 ? video2 : video1;
                if (Math.abs(source.currentTime - target.currentTime) > 0.05) {
                    target.currentTime = source.currentTime;
                }
            }
        }

        video1.addEventListener('timeupdate', () => {
            syncVideos(video1);
            updateProgress();
        });

        video2.addEventListener('timeupdate', () => {
            syncVideos(video2);
        });

        // Play/Pause
        function togglePlayPause() {
            if (video1.paused) {
                video1.play();
                video2.play();
                playPauseBtn.textContent = '⏸ Pause';
            } else {
                video1.pause();
                video2.pause();
                playPauseBtn.textContent = '▶ Play';
            }
        }

        // Seek
        function seek() {
            isSeeking = true;
            const time = (seekBar.value / 100) * video1.duration;
            video1.currentTime = time;
            video2.currentTime = time;
            setTimeout(() => { isSeeking = false; }, 100);
        }

        // Update progress bar and time display
        function updateProgress() {
            if (!isSeeking && video1.duration) {
                seekBar.value = (video1.currentTime / video1.duration) * 100;
                currentTimeDisplay.textContent = formatTime(video1.currentTime);
            }
        }

        // Set duration when metadata is loaded
        video1.addEventListener('loadedmetadata', () => {
            durationDisplay.textContent = formatTime(video1.duration);
        });

        // Format time as M:SS.ss
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = (seconds % 60).toFixed(2);
            return `${mins}:${secs.padStart(5, '0')}`;
        }

        // Speed control
        function setSpeed(speed) {
            video1.playbackRate = speed;
            video2.playbackRate = speed;

            // Update active button
            document.querySelectorAll('.speed-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
        }

        // Frame stepping
        function stepForward() {
            const wasPlaying = !video1.paused;
            video1.pause();
            video2.pause();
            playPauseBtn.textContent = '▶ Play';

            // Step forward by 1/fps seconds (approximate)
            const fps = 30; // Approximate, adjust if needed
            video1.currentTime = Math.min(video1.currentTime + 1/fps, video1.duration);
            video2.currentTime = video1.currentTime;
        }

        function stepBackward() {
            const wasPlaying = !video1.paused;
            video1.pause();
            video2.pause();
            playPauseBtn.textContent = '▶ Play';

            // Step backward by 1/fps seconds (approximate)
            const fps = 30; // Approximate, adjust if needed
            video1.currentTime = Math.max(video1.currentTime - 1/fps, 0);
            video2.currentTime = video1.currentTime;
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case ' ':
                    e.preventDefault();
                    togglePlayPause();
                    break;
                case 'ArrowRight':
                    stepForward();
                    break;
                case 'ArrowLeft':
                    stepBackward();
                    break;
            }
        });

        // Sync play/pause state
        video1.addEventListener('play', () => { video2.play(); });
        video1.addEventListener('pause', () => { video2.pause(); });
        video2.addEventListener('play', () => { video1.play(); });
        video2.addEventListener('pause', () => { video1.pause(); });
    </script>
</body>
</html>"""

    # Replace placeholders
    html_content = html_content.replace('INPUT_VIDEO', input_name)
    html_content = html_content.replace('OUTPUT_VIDEO', output_name)

    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML visualization generated: {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Process video with 2D FFT and generate HTML visualization'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output', '-o',
        default='output_fft.mp4',
        help='Path to output video file (default: output_fft.mp4)'
    )
    parser.add_argument(
        '--html',
        default='visualization.html',
        help='Path to output HTML file (default: visualization.html)'
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Process video
    try:
        output_path = str(Path(args.output).resolve())
        process_video(str(input_path), output_path)

        # Generate HTML
        html_path = str(Path(args.html).resolve())
        generate_html(str(input_path), output_path, html_path)

        print(f"\n✓ Complete!")
        print(f"  - Processed video: {output_path}")
        print(f"  - HTML viewer: {html_path}")
        print(f"\nOpen {args.html} in your browser to view the synchronized videos.")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
