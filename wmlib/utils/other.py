from pathlib import Path
import subprocess


def snapshot_src(src, target, exclude_from):
    """Save the runnable source alongside each experiment."""
    source, target = Path(src).resolve(), Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-a", f"--exclude-from={exclude_from}"]
    # Custom run directories can also be located inside the source tree.
    try:
        run_directory = target.relative_to(source).parts[0]
    except ValueError:
        pass
    else:
        command.append(f"--exclude=/{run_directory}/")
    subprocess.run(command + [str(source) + "/", str(target) + "/"], check=True)
