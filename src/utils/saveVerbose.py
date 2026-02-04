"""
saveVerbose.py - Capture and save all verbose console output during pipeline execution.

This utility provides a context manager that captures all stdout output while still
printing to the console, then saves it to a timestamped file for debugging and audit trails.

Usage (standalone pipeline):
    from utils.saveVerbose import VerboseCapture

    with VerboseCapture(
        filename="dataset.sav",
        variable_key="Q1",
        sample_size=500,
        run_until_step=5
    ):
        # All pipeline execution code here
        ...

Usage (Streamlit with append mode):
    with VerboseCapture(
        filename=st.session_state.filename,
        variable_key=variable_key,
        sample_size=sample_size,
        run_until_step=current_step,
        append_mode=True
    ):
        result = pipeline.step_N_function(...)
"""

import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


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

    Captures stdout while still printing to console, then saves to a timestamped file.

    Args:
        filename: Data filename (e.g., "M000000 Associatiemonitor Merk X.sav")
        variable_key: Variable identifier (e.g., "Qd1_combined")
        sample_size: Sample size (int or None for full dataset)
        run_until_step: Pipeline step to run until (0-9)
        output_dir: Output directory (defaults to exports/verbose_logs/)
        append_mode: If True, append to existing file instead of creating new one
        session_id: Optional session ID for Streamlit (used to group logs)
    """

    def __init__(
        self,
        filename: str,
        variable_key: str,
        sample_size: Optional[int],
        run_until_step: int,
        output_dir: Optional[Path] = None,
        append_mode: bool = False,
        session_id: Optional[str] = None
    ):
        self.filename = filename
        self.variable_key = variable_key
        self.sample_size = sample_size
        self.run_until_step = run_until_step
        self.append_mode = append_mode
        self.session_id = session_id

        # Determine output directory
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
        # Restore original stdout
        sys.stdout = self._original_stdout

        # Save captured output
        if self._tee is not None:
            self._save_output(self._tee.getvalue(), exc_type is not None)

    def _build_output_filename(self) -> str:
        """Build the output filename from parameters."""
        # Extract base name from filename
        base_name = Path(self.filename).stem
        # Clean up base name (remove spaces, special chars)
        base_name_clean = base_name.replace(" ", "_")[:50]

        # Use full variable key (cache key) - no truncation for exact cache matching
        var_key_clean = self.variable_key.replace(" ", "_")

        # Timestamp
        timestamp = self._start_time.strftime("%Y%m%d_%H%M%S") if self._start_time else datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"{base_name_clean}_{var_key_clean}_step{self.run_until_step}_{timestamp}.txt"

    def _build_header(self) -> str:
        """Build the file header with metadata."""
        lines = [
            "=" * 70,
            "PIPELINE VERBOSE OUTPUT LOG",
            "=" * 70,
            f"Dataset: {self.filename}",
            f"Variable: {self.variable_key}",
            f"Sample size: {self.sample_size if self.sample_size else 'full'}",
            f"Run until step: {self.run_until_step}",
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

            # Determine write mode
            mode = 'a' if self.append_mode and output_path.exists() else 'w'

            with open(output_path, mode, encoding='utf-8') as f:
                if mode == 'w':
                    # Write header for new files
                    f.write(self._build_header())

                if self.append_mode and mode == 'a':
                    # Add separator for appended content
                    f.write("\n" + "-" * 70 + "\n")
                    f.write(f"[Appended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
                    f.write("-" * 70 + "\n\n")

                # Write captured output
                f.write(captured_output)

                # Add error note if applicable
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
        variable_key: str,
        step: int,
        output_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Find the most recent verbose log file matching the given parameters.

        Args:
            filename: Data filename (e.g., "M000000 Associatiemonitor Merk X.sav")
            variable_key: Cache key (e.g., "Qd1_combined_2000")
            step: Pipeline step number (0-9)
            output_dir: Output directory (defaults to exports/verbose_logs/)

        Returns:
            Path to the most recent matching log file, or None if not found
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "exports" / "verbose_logs"

        if not output_dir.exists():
            return None

        # Build pattern to match: {base_name}_{cache_key}_step{N}_*.txt
        base_name = Path(filename).stem
        base_name_clean = base_name.replace(" ", "_")[:50]
        var_key_clean = variable_key.replace(" ", "_")

        pattern = f"{base_name_clean}_{var_key_clean}_step{step}_*.txt"

        # Find matching files
        matching_files = list(output_dir.glob(pattern))

        if not matching_files:
            return None

        # Sort by filename (timestamp is at the end, so alphabetical = chronological)
        matching_files.sort(reverse=True)

        return matching_files[0]

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
