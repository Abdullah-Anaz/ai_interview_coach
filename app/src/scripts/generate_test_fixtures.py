import shutil
import subprocess
from pathlib import Path
from typing import List


def execute_ffmpeg_command(args: List[str]) -> None:
    """
    Executes a shell-agnostic FFmpeg command to process media files.

    Args:
        args (List[str]): Sequence of FFmpeg arguments.

    Raises:
        RuntimeError: If the FFmpeg process returns a non-zero exit code or is not found.
    """
    try:
        command: List[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError("FFmpeg executable not found. Ensure FFmpeg is installed and in PATH.") from e
    except subprocess.CalledProcessError as e:
        error_message: str = e.stderr.decode().strip()
        raise RuntimeError(f"FFmpeg command failed: {error_message}") from e


def generate_silent_video(source_path: Path, target_path: Path) -> None:
    """
    Generates a video container completely stripped of audio streams.

    Args:
        source_path (Path): Path to the source MP4 file.
        target_path (Path): Path to the output silent MP4 file.
    """
    execute_ffmpeg_command(["-i", str(source_path), "-c:v", "copy", "-an", str(target_path)])


def generate_audio_only(source_path: Path, target_path: Path) -> None:
    """
    Generates an audio-only WAV file stripped of all video streams.

    Args:
        source_path (Path): Path to the source MP4 file.
        target_path (Path): Path to the output WAV file.
    """
    execute_ffmpeg_command(["-i", str(source_path), "-vn", "-c:a", "pcm_s16le", str(target_path)])


def generate_low_fps_video(source_path: Path, target_path: Path) -> None:
    """
    Generates a video container forcefully downsampled to 10 frames per second.

    Args:
        source_path (Path): Path to the source MP4 file.
        target_path (Path): Path to the output low-FPS MP4 file.
    """
    execute_ffmpeg_command(["-i", str(source_path), "-r", "10", "-c:a", "copy", str(target_path)])


def generate_high_sample_rate_audio(source_path: Path, target_path: Path) -> None:
    """
    Generates an audio-only WAV file upsampled to 48,000 Hz.

    Args:
        source_path (Path): Path to the source MP4 file.
        target_path (Path): Path to the output high sample rate WAV file.
    """
    execute_ffmpeg_command(
        ["-i", str(source_path), "-vn", "-ar", "48000", "-c:a", "pcm_s16le", str(target_path)]
    )


def generate_fractional_second_video(source_path: Path, target_path: Path) -> None:
    """
    Generates a media container forcefully truncated to 0.5 seconds in duration.

    Args:
        source_path (Path): Path to the source MP4 file.
        target_path (Path): Path to the output truncated MP4 file.
    """
    execute_ffmpeg_command(["-i", str(source_path), "-t", "0.5", "-c", "copy", str(target_path)])


def generate_invalid_format_file(target_path: Path) -> None:
    """
    Generates a plaintext file masquerading as a media file to trigger format errors.

    Args:
        target_path (Path): Path to the output invalid format file.
    """
    target_path.write_text("This is not a valid media container.")


def generate_empty_file(target_path: Path) -> None:
    """
    Generates a zero-byte file to trigger empty file stream errors.

    Args:
        target_path (Path): Path to the output empty file.
    """
    target_path.touch()


def generate_all_fixtures(source_file_name: str = "test.mp4") -> None:
    """
    Orchestrates the generation of all media fixtures required by the test suite.

    Args:
        source_file_name (str): Name of the source file located in src/tests/data.

    Raises:
        FileNotFoundError: If the source video file does not exist.
    """
    data_dir: Path = Path("app/src/tests/data")
    source_path: Path = data_dir / source_file_name

    if not source_path.is_file():
        raise FileNotFoundError(f"Source file {source_path} not found. Please provide a valid video.")

    standard_target: Path = data_dir / "standard_interview.mp4"
    shutil.copy2(source_path, standard_target)

    generate_silent_video(source_path, data_dir / "silent_video.mp4")
    generate_audio_only(source_path, data_dir / "audio_only.wav")
    generate_low_fps_video(source_path, data_dir / "low_fps_video.mp4")
    generate_high_sample_rate_audio(source_path, data_dir / "high_sample_rate.wav")
    generate_fractional_second_video(source_path, data_dir / "fractional_second_video.mp4")
    generate_invalid_format_file(data_dir / "invalid_format.txt")
    generate_empty_file(data_dir / "empty_file.mp4")


if __name__ == "__main__":
    generate_all_fixtures()