# Cross-platform execution

The application supports Windows, macOS, and Linux on a desktop Python
environment with a supported webcam driver. Use a Python version supported by
the installed MediaPipe and PyQt5 wheels (the development environment uses
Python 3.12).

Install from a clone or extracted source directory:

```bash
python -m pip install --upgrade pip
python -m pip install .
python scripts/download_pose_model.py
ai-trainer-live
```

Alternatively, run directly from the source directory:

```bash
python -m pip install -r requirements.txt
python scripts/download_pose_model.py
python scripts/run_live_pose.py
```

The **Select dataset folder** button in the live-pose window accepts the root
directory of an extracted AI Hub dataset. The selected path is saved outside
the installation directory, so it continues to work after upgrades and when
the application directory is read-only. The same setting becomes the default
input for `python main.py` or the installed `ai-trainer` command when building
a reference.

For unattended or containerized runs, set the path without opening the UI:

```bash
AI_TRAINER_DATASET_PATH=/path/to/aihub-dataset ai-trainer-live
```

On Windows, set the environment variable with `set` or PowerShell `$env:`;
on macOS/Linux use `export`. Camera permission must be granted to the terminal
or Python application in the operating-system privacy settings. On Linux, the
current user must also be allowed to read the camera device, normally
`/dev/video0`.

No desktop application can operate without a supported camera driver, a
working OpenCV/MediaPipe wheel for the CPU architecture, and the operating
system's camera permission. The application now reports these conditions in a
platform-neutral way rather than relying on Windows-only paths or messages.
