#!/usr/bin/env python3
"""
Milestone 2 + 3: App Audio Capture, Live Transcription, and Llama 3.2 Agent (Ollama).
Single window: transcription panel, Agent Answer button, AI response panel.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime

from audio_capture import (
    AppAudioCapture,
    get_running_audio_apps,
    find_blackhole_input_device,
    SAMPLE_RATE,
)
from transcription import Transcriber
from agent import get_agent_response


# ---- 90s black & green theme ----
BG_DARK = "#0a0a0a"
BG_PANEL = "#0d100d"
GREEN_BRIGHT = "#00ff88"
GREEN_DIM = "#00aa55"
GREEN_BORDER = "#006633"
TEXT_PRIMARY = "#a8ffa8"
TEXT_DIM = "#507050"
RED_ERR = "#cc4444"
FONT_UI = ("Courier New", 11)
FONT_MONO = ("Courier New", 10)


class LevelMeter(tk.Frame):
    """90s-style horizontal audio level meter (green on black)."""

    def __init__(self, parent, width=200, height=22, **kwargs):
        super().__init__(parent, bg=BG_DARK, **kwargs)
        self._width = width
        self._height = height
        self._level = 0.0
        self._canvas = tk.Canvas(
            self,
            width=width,
            height=height,
            bg=BG_PANEL,
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.create_rectangle(0, 0, width, height, outline=GREEN_BORDER, width=1)

    def set_level(self, value: float) -> None:
        self._level = max(0.0, min(1.0, value))

    def _draw(self) -> None:
        self._canvas.delete("level")
        w, h = self._width, self._height
        fill_w = int(w * self._level)
        if fill_w > 0:
            self._canvas.create_rectangle(0, 0, fill_w, h, fill=GREEN_BRIGHT, outline="", tags="level")
        self._canvas.create_rectangle(fill_w, 0, w, h, fill=BG_PANEL, outline="", tags="level")

    def redraw(self) -> None:
        self._draw()


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GodWhisper — Live Transcription + Agent (Llama 3.2)")
        self.root.minsize(500, 560)
        self.root.configure(bg=BG_DARK)
        self._apply_theme()

        self._capture: AppAudioCapture | None = None
        self._transcriber: Transcriber | None = None
        self._transcription_queue: queue.Queue | None = None
        self._level = 0.0
        self._status = "Stopped"
        self._playback_var = tk.BooleanVar(value=False)
        self._mute_var = tk.BooleanVar(value=False)
        self._language_var = tk.StringVar(value="en-US")
        self._transcription_last_sent_len = 0
        self._agent_busy = False

        self._build_ui()
        self._refresh_app_list()
        self._schedule_level_update()

    def _apply_theme(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=BG_DARK, foreground=TEXT_PRIMARY, font=FONT_UI)
        style.configure("TFrame", background=BG_DARK)
        style.configure("TLabelframe", background=BG_DARK, foreground=GREEN_BRIGHT)
        style.configure("TLabelframe.Label", background=BG_DARK, foreground=GREEN_BRIGHT, font=(FONT_UI[0], FONT_UI[1], "bold"))
        style.configure("TLabel", background=BG_DARK, foreground=TEXT_PRIMARY)
        style.configure("TButton", background=BG_PANEL, foreground=GREEN_BRIGHT, padding=(10, 6))
        style.map("TButton", background=[("active", GREEN_BORDER), ("pressed", GREEN_DIM)])
        style.configure("TCheckbutton", background=BG_DARK, foreground=TEXT_PRIMARY)
        style.map("TCheckbutton", background=[("active", BG_DARK)])
        style.configure("TCombobox", fieldbackground=BG_PANEL, background=BG_PANEL, foreground=TEXT_PRIMARY)

    def _panel(self, parent: tk.Misc, title: str):
        """Returns (outer_frame, inner_frame). Pack outer; add content to inner. 90s green-bordered box."""
        outer = tk.Frame(parent, bg=GREEN_BORDER, padx=1, pady=1)
        inner = tk.Frame(outer, bg=BG_PANEL, padx=10, pady=8)
        inner.pack(fill=tk.BOTH, expand=True)
        tk.Label(inner, text=f" {title} ", font=FONT_UI, fg=GREEN_BRIGHT, bg=BG_PANEL).pack(anchor=tk.W)
        return outer, inner

    def _build_ui(self) -> None:
        title_bar = tk.Frame(self.root, bg=BG_DARK)
        title_bar.pack(fill=tk.X, pady=(8, 0))
        tk.Label(title_bar, text=" GODWHISPER ", font=(FONT_UI[0], 14, "bold"), fg=GREEN_BRIGHT, bg=BG_DARK).pack()
        tk.Label(title_bar, text=" live transcription · agent ", font=FONT_UI, fg=TEXT_DIM, bg=BG_DARK).pack()
        main = tk.Frame(self.root, bg=BG_DARK, padx=14, pady=12)
        main.pack(fill=tk.BOTH, expand=True)

        # ---- Control strip ----
        top_outer, top_inner = self._panel(main, "CAPTURE & PLAYBACK")
        top_outer.pack(fill=tk.X, pady=(0, 10))

        row0 = tk.Frame(top_inner, bg=BG_PANEL)
        row0.pack(fill=tk.X, pady=(0, 6))
        tk.Label(row0, text="App:", font=FONT_UI, fg=TEXT_DIM, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 6))
        self._app_var = tk.StringVar()
        self._app_combo = ttk.Combobox(row0, textvariable=self._app_var, state="readonly", width=28, font=FONT_UI)
        self._app_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self._btn_refresh = ttk.Button(row0, text="Refresh", command=self._refresh_app_list)
        self._btn_refresh.pack(side=tk.LEFT)

        row1 = tk.Frame(top_inner, bg=BG_PANEL)
        row1.pack(fill=tk.X, pady=(0, 6))
        tk.Label(row1, text="Status:", font=FONT_UI, fg=TEXT_DIM, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 6))
        self._status_label = tk.Label(row1, text="Stopped", font=FONT_UI, fg=TEXT_DIM, bg=BG_PANEL)
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        row2 = tk.Frame(top_inner, bg=BG_PANEL)
        row2.pack(fill=tk.X, pady=(0, 6))
        tk.Label(row2, text="Level:", font=FONT_UI, fg=TEXT_DIM, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 6))
        self._level_meter = LevelMeter(row2, width=240, height=22)
        self._level_meter.pack(side=tk.LEFT, fill=tk.X, expand=True)

        row3 = tk.Frame(top_inner, bg=BG_PANEL)
        row3.pack(fill=tk.X, pady=(0, 6))
        self._playback_cb = ttk.Checkbutton(row3, text="Playback", variable=self._playback_var, command=self._on_playback_toggle)
        self._playback_cb.pack(side=tk.LEFT, padx=(0, 16))
        self._mute_cb = ttk.Checkbutton(row3, text="Mute", variable=self._mute_var, command=self._on_mute_toggle)
        self._mute_cb.pack(side=tk.LEFT)

        row4 = tk.Frame(top_inner, bg=BG_PANEL)
        row4.pack(fill=tk.X, pady=(0, 4))
        tk.Label(row4, text="Language:", font=FONT_UI, fg=TEXT_DIM, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 6))
        self._lang_combo = ttk.Combobox(row4, textvariable=self._language_var, state="readonly", width=14, font=FONT_UI, values=["en-US (English)", "it-IT (Italian)"])
        self._lang_combo.pack(side=tk.LEFT, padx=(0, 8))
        self._language_var.set("en-US (English)")
        self._lang_combo.bind("<<ComboboxSelected>>", self._on_language_change)

        row5 = tk.Frame(top_inner, bg=BG_PANEL)
        row5.pack(fill=tk.X, pady=(8, 0))
        self._start_btn = ttk.Button(row5, text="  START  ", command=self._on_start)
        self._start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._stop_btn = ttk.Button(row5, text="  STOP  ", command=self._on_stop, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT)

        # ---- Live transcription ----
        trans_outer, trans_inner = self._panel(main, "LIVE TRANSCRIPTION")
        trans_outer.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        btn_row = tk.Frame(trans_inner, bg=BG_PANEL)
        btn_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(btn_row, text="Clear", command=self._clear_transcription).pack(side=tk.LEFT)
        self._transcription_text = scrolledtext.ScrolledText(
            trans_inner,
            wrap=tk.WORD,
            height=9,
            font=FONT_MONO,
            bg=BG_DARK,
            fg=TEXT_PRIMARY,
            insertbackground=GREEN_BRIGHT,
            relief=tk.FLAT,
            padx=8,
            pady=8,
            selectbackground=GREEN_BORDER,
            selectforeground=TEXT_PRIMARY,
        )
        self._transcription_text.pack(fill=tk.BOTH, expand=True)

        # ---- Agent Answer (prominent) ----
        agent_frame = tk.Frame(main, bg=BG_DARK)
        agent_frame.pack(fill=tk.X, pady=(8, 8))
        self._agent_btn = tk.Button(
            agent_frame,
            text="  AGENT ANSWER  ",
            font=(FONT_UI[0], 12, "bold"),
            fg=BG_DARK,
            bg=GREEN_BRIGHT,
            activeforeground=BG_DARK,
            activebackground=TEXT_PRIMARY,
            relief=tk.RAISED,
            bd=2,
            cursor="hand2",
            command=self._on_agent_answer,
        )
        self._agent_btn.pack()

        # ---- AI response ----
        ai_outer, ai_inner = self._panel(main, "AI AGENT RESPONSE")
        ai_outer.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        ai_btn_row = tk.Frame(ai_inner, bg=BG_PANEL)
        ai_btn_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(ai_btn_row, text="Clear", command=self._clear_ai_responses).pack(side=tk.LEFT)
        self._ai_response_text = scrolledtext.ScrolledText(
            ai_inner,
            wrap=tk.WORD,
            height=7,
            font=FONT_MONO,
            bg=BG_DARK,
            fg=TEXT_PRIMARY,
            insertbackground=GREEN_BRIGHT,
            relief=tk.FLAT,
            padx=8,
            pady=8,
            selectbackground=GREEN_BORDER,
            selectforeground=TEXT_PRIMARY,
        )
        self._ai_response_text.pack(fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _refresh_app_list(self) -> None:
        apps = get_running_audio_apps()
        self._app_combo["values"] = apps if apps else ["(No apps — run Zoom, Chrome, etc.)"]
        if apps and not self._app_var.get():
            self._app_var.set(apps[0])

    def _on_status(self, msg: str) -> None:
        self._status = msg
        color = GREEN_BRIGHT if "Capturing" in msg and "Failed" not in msg else TEXT_DIM
        if "Failed" in msg:
            color = RED_ERR
        self.root.after(0, lambda m=msg, c=color: self._status_label.config(text=m, fg=c))

    def _on_level(self, rms: float) -> None:
        self._level = rms

    def _on_transcription(self, text: str) -> None:
        def append():
            txt = self._transcription_text
            txt.insert(tk.END, text + " ")
            txt.see(tk.END)
        self.root.after(0, append)

    def _on_language_change(self, event=None) -> None:
        sel = self._language_var.get()
        code = "en-US" if "English" in sel else "it-IT"
        if self._transcriber:
            self._transcriber.set_language_code(code)

    def _clear_transcription(self) -> None:
        self._transcription_text.delete("1.0", tk.END)
        self._transcription_last_sent_len = 0

    def _clear_ai_responses(self) -> None:
        self._ai_response_text.delete("1.0", tk.END)

    def _on_agent_answer(self) -> None:
        if self._agent_busy:
            return
        full = self._transcription_text.get("1.0", tk.END)
        new_text = full[self._transcription_last_sent_len :].strip()
        if not new_text:
            return
        self._transcription_last_sent_len = len(full)
        self._agent_busy = True
        self._agent_btn.config(state=tk.DISABLED, bg=TEXT_DIM)
        self._append_ai_response("\n[Generating...]\n")

        def run_agent():
            try:
                response = get_agent_response(new_text)
            except Exception as e:
                response = f"[Agent error: {e}]"
            self.root.after(0, lambda: self._agent_done(response))

        threading.Thread(target=run_agent, daemon=True).start()

    def _agent_done(self, response: str) -> None:
        self._agent_busy = False
        self._agent_btn.config(state=tk.NORMAL, bg=GREEN_BRIGHT)
        txt = self._ai_response_text
        content = txt.get("1.0", tk.END)
        if "[Generating...]" in content:
            lines = content.split("\n")
            new_lines = [l for l in lines if "[Generating...]" not in l]
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, "\n".join(new_lines))
            if new_lines and not new_lines[-1].endswith("\n"):
                txt.insert(tk.END, "\n")
        ts = datetime.now().strftime("%H:%M:%S")
        self._append_ai_response(f"[{ts}]\n{response}\n\n")
        self._ai_response_text.see(tk.END)

    def _append_ai_response(self, text: str) -> None:
        self._ai_response_text.insert(tk.END, text)
        self._ai_response_text.see(tk.END)

    def _schedule_level_update(self) -> None:
        def update():
            self._level_meter.set_level(self._level * 8)
            self._level_meter.redraw()
            self.root.after(50, update)
        self.root.after(50, update)

    def _start_capture_and_transcription(self) -> None:
        if find_blackhole_input_device() is None:
            self._on_status("Failed to capture: BlackHole not found. Install BlackHole and set app output to it.")
            return
        self._transcription_queue = queue.Queue(maxsize=600)  # enough for 5 s chunks
        self._capture = AppAudioCapture(
            on_level=self._on_level,
            on_status=self._on_status,
            transcription_queue=self._transcription_queue,
        )
        if not self._capture.start_capture():
            self._capture = None
            return
        code = "en-US" if "English" in self._language_var.get() else "it-IT"
        self._transcriber = Transcriber(
            self._transcription_queue,
            on_transcription=self._on_transcription,
            language_code=code,
        )
        self._transcriber.start()
        self._on_status("Capturing audio — 1.2 s chunks, fast (click Stop to end)")
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)

    def _on_start(self) -> None:
        self._start_capture_and_transcription()

    def _on_stop(self) -> None:
        if self._transcriber:
            self._transcriber.stop()
        if self._capture:
            self._capture.stop_capture()
        self._capture = None
        self._transcriber = None
        self._transcription_queue = None
        self._on_status("Stopped")
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)

    def _on_playback_toggle(self) -> None:
        if self._capture:
            self._capture.set_playback(self._playback_var.get())

    def _on_mute_toggle(self) -> None:
        if self._capture:
            self._capture.set_muted(self._mute_var.get())

    def _on_close(self) -> None:
        if self._transcriber:
            self._transcriber.stop()
        if self._capture:
            self._capture.stop_capture()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = App()
    app.run()


if __name__ == "__main__":
    main()
