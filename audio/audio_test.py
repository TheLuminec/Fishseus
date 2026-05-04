import subprocess
import sys
import time


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2          # 16-bit PCM
CHUNK_FRAMES = 1024
PRINT_INTERVAL = 0.5


def find_capture_device():
    """
    Try to discover a capture device name from `arecord -L`.
    Returns a device string suitable for arecord -D, or None if not found.
    """
    try:
        result = subprocess.run(
            ["arecord", "-L"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        print(f"Failed to query ALSA devices: {exc}")
        return None

    lines = [line.rstrip() for line in result.stdout.splitlines()]

    preferred_prefixes = [
        "default",
        "sysdefault",
        "plughw:",
        "hw:",
    ]

    candidates = []
    for line in lines:
        if not line or line.startswith(" "):
            continue
        candidates.append(line)

    for prefix in preferred_prefixes:
        for candidate in candidates:
            if candidate == prefix or candidate.startswith(prefix):
                return candidate

    return candidates[0] if candidates else None


def mean_absolute_amplitude(data: bytes) -> float:
    """
    Compute average absolute amplitude for 16-bit little-endian mono PCM.
    Returns a value roughly in the range 0..32768.
    """
    if len(data) < 2:
        return 0.0

    total = 0
    sample_count = 0

    for i in range(0, len(data) - 1, 2):
        sample = int.from_bytes(data[i:i + 2], byteorder="little", signed=True)
        total += abs(sample)
        sample_count += 1

    if sample_count == 0:
        return 0.0

    return total / sample_count


def main() -> int:
    device = find_capture_device()
    if device is None:
        print("No ALSA capture device found.")
        print("Run `arecord -l` and `arecord -L` to verify your INMP441 is configured.")
        return 1

    print(f"Using ALSA capture device: {device}")

    cmd = [
        "arecord",
        "-D", device,
        "-q",
        "-t", "raw",
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", str(CHANNELS),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except FileNotFoundError:
        print("`arecord` not found. Install ALSA utilities: sudo apt install alsa-utils")
        return 1
    except Exception as exc:
        print(f"Failed to start arecord: {exc}")
        return 1

    bytes_per_chunk = CHUNK_FRAMES * CHANNELS * SAMPLE_WIDTH
    last_print = time.monotonic()

    amplitude_sum = 0.0
    chunk_count = 0

    print("Streaming mic data... Press Ctrl+C to stop.")

    try:
        while True:
            if proc.stdout is None:
                print("No stdout from arecord process.")
                return 1

            data = proc.stdout.read(bytes_per_chunk)
            if not data:
                err = ""
                if proc.stderr is not None:
                    err = proc.stderr.read().decode(errors="ignore")
                print("No audio data received from arecord.")
                if err.strip():
                    print(err.strip())
                return 1

            avg_amplitude = mean_absolute_amplitude(data)

            amplitude_sum += avg_amplitude
            chunk_count += 1

            now = time.monotonic()
            if now - last_print >= PRINT_INTERVAL:
                mean_amp = amplitude_sum / max(chunk_count, 1)
                normalized = mean_amp / 32768.0
                print(
                    f"Average amplitude: {mean_amp:.2f} "
                    f"(normalized: {normalized:.4f})"
                )
                amplitude_sum = 0.0
                chunk_count = 0
                last_print = now

    except KeyboardInterrupt:
        print("\nStopping audio test...")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
