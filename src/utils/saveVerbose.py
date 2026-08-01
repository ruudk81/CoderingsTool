"""
saveVerbose.py - Capture and save all verbose console output during pipeline execution.

Context manager that captures stdout while still printing to the console, then
writes it to exports/verbose_logs/ under the canonical export name
({dataset}_{var}_{sample}_log_step{N}.txt, see utils.exportNaming). A rerun of
the same step on the same dataset overwrites the previous log.

Usage:
    from utils.saveVerbose import VerboseCapture

    with VerboseCapture(
        filename="dataset.sav",
        var_name="Q1",
        sample_size=500,
        step=5,
    ):
        ...
"""

import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.exportNaming import export_filename


def build_log_filename(
    filename: str,
    var_name: str,
    sample_size: Optional[int],
    step: int,
) -> str:
    """The canonical name of a verbose log: the step number as a doctype."""
    return export_filename(filename, var_name, sample_size, f"log_step{step}", "txt")


class TeeOutput:
    """Capture stdout while still printing to console."""

    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.captured = io.StringIO()

    def write(self, text):
        self.original_stdout.write(text)
        self.captured.write(text)

    def flush(self):
        self.original_stdout.flush()

    def getvalue(self) -> str:
        return self.captured.getvalue()


class VerboseCapture:
    """
    Context manager to capture all verbose console output during pipeline execution.

    Captures stdout while still printing to console, then saves to the canonical
    log file for this step (overwriting any previous run of it).

    Args:
        filename: Data filename (e.g., "M000000 Associatiemonitor Merk X.sav")
        var_name: Variable identifier (e.g., "Qd1_combined")
        sample_size: Sample size (int or None for full dataset)
        step: Pipeline step (0-9)
        output_dir: Output directory (defaults to exports/verbose_logs/)
    """

    def __init__(
        self,
        filename: str,
        var_name: str,
        sample_size: Optional[int],
        step: int,
        output_dir: Optional[Path] = None,
    ):
        self.filename = filename
        self.var_name = var_name
        self.sample_size = sample_size
        self.step = step

        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            self.output_dir = project_root / "exports" / "verbose_logs"
        else:
            self.output_dir = Path(output_dir)

        self._tee: Optional[TeeOutput] = None
        self._original_stdout = None
        self._start_time: Optional[datetime] = None

    def __enter__(self) -> 'VerboseCapture':
        """Start capturing stdout."""
        self._start_time = datetime.now()
        self._original_stdout = sys.stdout
        self._tee = TeeOutput(self._original_stdout)
        sys.stdout = self._tee
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore stdout and save captured output."""
        sys.stdout = self._original_stdout

        if self._tee is not None:
            self._save_output(self._tee.getvalue(), exc_type is not None)

    def _build_output_filename(self) -> str:
        return build_log_filename(
            self.filename, self.var_name, self.sample_size, self.step)

    def _build_header(self) -> str:
        """Build the file header with metadata."""
        lines = [
            "=" * 70,
            "PIPELINE VERBOSE OUTPUT LOG",
            "=" * 70,
            f"Dataset: {self.filename}",
            f"Variable: {self.var_name}",
            f"Sample size: {self.sample_size if self.sample_size is not None else 'full'}",
            f"Step: {self.step}",
            f"Start time: {self._start_time.strftime('%Y-%m-%d %H:%M:%S') if self._start_time else 'unknown'}",
            f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
        ]
        return "\n".join(lines)

    def _save_output(self, captured_output: str, had_error: bool = False) -> None:
        """Save the captured output to file."""
        try:
            # Create output directory if needed
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Build output path
            output_filename = self._build_output_filename()
            output_path = self.output_dir / output_filename

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self._build_header())
                f.write(captured_output)

                if had_error:
                    f.write("\n\n" + "=" * 70 + "\n")
                    f.write("NOTE: Pipeline execution ended with an error\n")
                    f.write("=" * 70 + "\n")

            print(f"\n✓ Verbose output saved to: {output_path}")

        except Exception as e:
            print(f"Warning: Failed to save verbose output: {e}")

    def get_output_path(self) -> Path:
        """Get the path where the output will be/was saved."""
        return self.output_dir / self._build_output_filename()

    @staticmethod
    def find_latest_log(
        filename: str,
        var_name: str,
        sample_size: Optional[int],
        step: int,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """The log file for this step, or None.

        There is at most one: the name is deterministic and a rerun overwrites
        it. No glob, no mtime sorting.
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "exports" / "verbose_logs"

        path = Path(output_dir) / build_log_filename(
            filename, var_name, sample_size, step)
        return path if path.exists() else None

    @staticmethod
    def load_log_content(log_path: Path) -> Optional[str]:
        """
        Load the content of a verbose log file.

        Args:
            log_path: Path to the log file

        Returns:
            Log content as string, or None if failed to load
        """
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None
