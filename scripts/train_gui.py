import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


PERCENT_PATTERN = re.compile(r"(\d{1,3})%\|")
ITER_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")
EVAL_ITER_PATTERN = re.compile(r"\[ITER\s+(\d+)\]")
MAX_REASONABLE_ITERATIONS = 10_000_000


class TrainGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Triangle Splatting Training")
        self.repo_root = Path(__file__).resolve().parents[1]
        self.process: subprocess.Popen | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.iteration_cap_warning_shown = False

        self.scene_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.images_folder = tk.StringVar(value="images")
        self.iterations = tk.StringVar(value="30000")
        self.eval_enabled = tk.BooleanVar(value=True)
        self.indoor_enabled = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="Idle")

        self._build_ui()
        self.root.after(100, self._poll_output)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(7, weight=1)

        ttk.Label(frame, text="Scene folder (-s):").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.scene_path).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Browse", command=self._browse_scene).grid(row=0, column=2)

        ttk.Label(frame, text="Model output folder (-m):").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.model_path).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Browse", command=self._browse_model).grid(row=1, column=2)

        ttk.Label(frame, text="Images folder (-i):").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.images_folder).grid(row=2, column=1, sticky="ew", padx=6)

        ttk.Label(frame, text="Iterations:").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.iterations).grid(row=3, column=1, sticky="ew", padx=6)

        options = ttk.Frame(frame)
        options.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 4))
        ttk.Checkbutton(options, text="--eval", variable=self.eval_enabled).grid(row=0, column=0, padx=(0, 12))
        ttk.Checkbutton(options, text="--indoor", variable=self.indoor_enabled).grid(row=0, column=1)

        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 8))
        self.start_button = ttk.Button(buttons, text="Start Training", command=self._start_training)
        self.start_button.grid(row=0, column=0, padx=(0, 8))
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop_training, state="disabled")
        self.stop_button.grid(row=0, column=1)

        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate", maximum=100)
        self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(4, 2))

        ttk.Label(frame, textvariable=self.status_text).grid(row=7, column=0, columnspan=3, sticky="w", pady=(2, 4))

        self.log_text = tk.Text(frame, height=16, wrap="word")
        self.log_text.grid(row=8, column=0, columnspan=3, sticky="nsew")
        self.log_text.configure(state="disabled")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=8, column=3, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _browse_scene(self) -> None:
        selected = filedialog.askdirectory(title="Select scene folder")
        if selected:
            self.scene_path.set(selected)

    def _browse_model(self) -> None:
        selected = filedialog.askdirectory(title="Select model output folder")
        if selected:
            self.model_path.set(selected)

    def _validate_inputs(self) -> tuple[bool, int]:
        if not self.scene_path.get().strip():
            messagebox.showerror("Missing input", "Please select a scene folder.")
            return False, 0
        if not self.model_path.get().strip():
            messagebox.showerror("Missing input", "Please select a model output folder.")
            return False, 0
        try:
            iterations = int(self.iterations.get())
            if iterations <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid iterations", "Iterations must be a positive integer.")
            return False, 0
        return True, iterations

    def _build_command(self) -> list[str]:
        command = [
            sys.executable,
            "train.py",
            "-s",
            self.scene_path.get().strip(),
            "-m",
            self.model_path.get().strip(),
            "-i",
            self.images_folder.get().strip() or "images",
            "--iterations",
            self.iterations.get().strip(),
        ]
        if self.eval_enabled.get():
            command.append("--eval")
        if self.indoor_enabled.get():
            command.append("--indoor")
        return command

    def _start_training(self) -> None:
        if self.process is not None:
            return
        valid, iterations = self._validate_inputs()
        if not valid:
            return

        self.progress.configure(maximum=iterations, value=0)
        self.iteration_cap_warning_shown = False
        self.status_text.set("Starting training...")
        self._append_log("Starting training process...\n")
        command = self._build_command()
        self._append_log("Command: " + " ".join(command) + "\n")

        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.process = None
            messagebox.showerror("Failed to start", str(exc))
            self.status_text.set("Failed to start.")
            return

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=self._stream_output, daemon=True).start()

    def _stream_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.output_queue.put(line)
        return_code = self.process.wait()
        self.output_queue.put(f"__PROCESS_DONE__:{return_code}")

    def _poll_output(self) -> None:
        while True:
            try:
                message = self.output_queue.get_nowait()
            except queue.Empty:
                break

            if message.startswith("__PROCESS_DONE__:"):
                return_code = int(message.split(":", 1)[1])
                self._finish_process(return_code)
                continue

            self._append_log(message)
            self._update_progress(message)

        self.root.after(100, self._poll_output)

    def _update_progress(self, line: str) -> None:
        percent_match = PERCENT_PATTERN.search(line)
        if percent_match:
            percent = min(max(int(percent_match.group(1)), 0), 100)
            value = (percent / 100.0) * float(self.progress.cget("maximum"))
            self.progress.configure(value=value)
            self.status_text.set(f"Training... {percent}%")

        iter_match = ITER_PATTERN.search(line)
        if iter_match:
            current, total = int(iter_match.group(1)), int(iter_match.group(2))
            if total > 0 and total <= MAX_REASONABLE_ITERATIONS:
                if int(self.progress.cget("maximum")) != total:
                    self.progress.configure(maximum=total)
                self.progress.configure(value=min(current, total))
                self.status_text.set(f"Training... {current}/{total}")
            elif total > MAX_REASONABLE_ITERATIONS and not self.iteration_cap_warning_shown:
                self.iteration_cap_warning_shown = True
                self._append_log(
                    f"\nWarning: reported iteration total ({total}) exceeds supported progress "
                    f"range ({MAX_REASONABLE_ITERATIONS}); using percentage updates only.\n"
                )

        eval_iter_match = EVAL_ITER_PATTERN.search(line)
        if eval_iter_match:
            current_eval = int(eval_iter_match.group(1))
            self.progress.configure(value=min(current_eval, int(self.progress.cget("maximum"))))
            self.status_text.set(f"Evaluating at iteration {current_eval}")

    def _finish_process(self, return_code: int) -> None:
        self.process = None
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if return_code == 0:
            self.progress.configure(value=self.progress.cget("maximum"))
            self.status_text.set("Training finished successfully.")
            self._append_log("\nTraining finished successfully.\n")
        else:
            self.status_text.set(f"Training failed (exit code {return_code}).")
            self._append_log(f"\nTraining stopped with exit code {return_code}.\n")

    def _stop_training(self) -> None:
        if self.process is None:
            return
        self.status_text.set("Stopping training...")
        self._append_log("\nStopping training process...\n")
        self._terminate_process()

    def _terminate_process(self) -> None:
        if self.process is None:
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._append_log("Process did not stop gracefully, forcing kill...\n")
            self.process.kill()
            self.process.wait(timeout=5)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        if self.process is not None:
            if messagebox.askyesno("Exit", "Training is still running. Stop it and exit?"):
                self._terminate_process()
            else:
                return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    root.geometry("920x620")
    TrainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
