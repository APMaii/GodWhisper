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
from agent import get_agent_response, DEFAULT_PERSONA


# ---- Theme definitions (select in Settings) ----
THEMES = {
    "light_blue": {
        "BG_ROOT": "#cce6ff",        # slightly softer main bg
        "BG_CARD": "#e6f2ff",        # lighter card bg
        "BG_INPUT": "#f2f9ff",       # lighter input bg
        "BORDER": "#3399ff",         # same vibrant border
        "TEXT": "#0d1a26",           # darker for readability
        "TEXT_MUTED": "#4d6680",     # muted but readable
        "ACCENT": "#007acc",         # slightly brighter accent
        "ACCENT_HOVER": "#005fa3",   # hover deeper but compatible
        "HIGHLIGHT_BG": "#fff8b3",   # soft highlight, not harsh
        "RED_ERR": "#d32f2f",        # error slightly darker, matches palette
    },
    "90s_green": {
        "BG_ROOT": "#0f1f0f",        # softened black-green root
        "BG_CARD": "#162416",        # dark card for contrast
        "BG_INPUT": "#1a2a1a",       # slightly brighter input
        "BORDER": "#00cc66",         # border compatible with accent
        "TEXT": "#00ff99",           # neon but readable
        "TEXT_MUTED": "#00cc77",     # muted neon green
        "ACCENT": "#00ff99",         # matches text
        "ACCENT_HOVER": "#00b366",   # hover slightly darker
        "HIGHLIGHT_BG": "#004422",   # soft dark highlight
        "RED_ERR": "#ff5555",        # bright but not clashing
    },
    "black_red": {
        "BG_ROOT": "#1e0c0c",        # slightly lighter root
        "BG_CARD": "#2b1414",        # deep card background
        "BG_INPUT": "#3e1f1f",       # input softer
        "BORDER": "#cc0000",         # matches accent
        "TEXT": "#ffdede",           # softer text, less harsh
        "TEXT_MUTED": "#e6bfbf",     # muted and readable
        "ACCENT": "#e53935",         # accent vivid but not too bright
        "ACCENT_HOVER": "#b71c1c",   # hover darker for contrast
        "HIGHLIGHT_BG": "#4f1f1f",   # matches palette, softer
        "RED_ERR": "#ff5555",        # noticeable error
    },
    "white_modern": {
        "BG_ROOT": "#f9f9f9",        # very soft root
        "BG_CARD": "#ffffff",        # card white stays
        "BG_INPUT": "#ffffff",       # input white
        "BORDER": "#d6d6d6",         # soft border
        "TEXT": "#212121",           # main text strong contrast
        "TEXT_MUTED": "#757575",     # muted grey
        "ACCENT": "#1976d2",         # main accent
        "ACCENT_HOVER": "#115293",   # hover accent darker
        "HIGHLIGHT_BG": "#e3f2fd",   # subtle highlight
        "RED_ERR": "#d32f2f",        # error compatible
    },
}


# Mutable current theme (updated in Settings)
THEME = dict(THEMES["light_blue"])
FONT_UI = ("Helvetica Neue", 11)
FONT_TRANSCRIPTION = ("Helvetica Neue", 12)
FONT_AGENT = ("Georgia", 11)


class LevelMeter(tk.Frame):
    """Horizontal level meter (colors from THEME)."""

    def __init__(self, parent, width=200, height=20, **kwargs):
        super().__init__(parent, bg=THEME["BG_ROOT"], **kwargs)
        self._width = width
        self._height = height
        self._level = 0.0
        self._canvas = tk.Canvas(
            self, width=width, height=height,
            bg=THEME["BG_INPUT"], highlightthickness=0, bd=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

    def set_level(self, value: float) -> None:
        self._level = max(0.0, min(1.0, value))

    def _draw(self) -> None:
        self._canvas.delete("level")
        w, h = self._width, self._height
        fill_w = int(w * self._level)
        if fill_w > 0:
            self._canvas.create_rectangle(0, 0, fill_w, h, fill=THEME["ACCENT"], outline="", tags="level")
        self._canvas.create_rectangle(fill_w, 0, w, h, fill=THEME["BG_INPUT"], outline="", tags="level")

    def redraw(self) -> None:
        self._draw()


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GodWhisper — Live Transcription + Agent (Llama 3.2)")
        self.root.minsize(520, 620)
        self._chunk_duration_sec = 1.2
        self._theme_name = "light_blue"
        self._agent_persona = DEFAULT_PERSONA
        self.root.configure(bg=THEME["BG_ROOT"])
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
        style.configure(".", background=THEME["BG_ROOT"], foreground=THEME["TEXT"], font=FONT_UI)
        style.configure("TFrame", background=THEME["BG_ROOT"])
        style.configure("TLabelframe", background=THEME["BG_CARD"], foreground=THEME["TEXT"])
        style.configure("TLabelframe.Label", background=THEME["BG_CARD"], foreground=THEME["TEXT"], font=(FONT_UI[0], FONT_UI[1], "bold"))
        style.configure("TLabel", background=THEME["BG_CARD"], foreground=THEME["TEXT"])
        style.configure("TButton", background=THEME["ACCENT"], foreground="#ffffff", padding=(12, 8))
        style.map("TButton", background=[("active", THEME["ACCENT_HOVER"]), ("pressed", THEME["ACCENT_HOVER"])])
        style.configure("TCheckbutton", background=THEME["BG_CARD"], foreground=THEME["TEXT"])
        style.map("TCheckbutton", background=[("active", THEME["BG_CARD"])])
        style.configure("TCombobox", fieldbackground=THEME["BG_INPUT"], background=THEME["BG_INPUT"], foreground=THEME["TEXT"])

    def _card(self, parent: tk.Misc, title: str):
        """Card panel: outer has border, inner has padding. Pack outer; add content to inner."""
        outer = tk.Frame(parent, bg=THEME["BORDER"], padx=1, pady=1)
        inner = tk.Frame(outer, bg=THEME["BG_CARD"], padx=14, pady=12)
        inner.pack(fill=tk.BOTH, expand=True)
        tk.Label(inner, text=title, font=(FONT_UI[0], FONT_UI[1], "bold"), fg=THEME["TEXT"], bg=THEME["BG_CARD"]).pack(anchor=tk.W)
        return outer, inner

    def _build_ui(self) -> None:
        # Title + Settings (top right)
        title_bar = tk.Frame(self.root, bg=THEME["BG_ROOT"])
        title_bar.pack(fill=tk.X, pady=(16, 8))
        tk.Label(title_bar, text="GodWhisper", font=(FONT_UI[0], 18, "bold"), fg=THEME["TEXT"], bg=THEME["BG_ROOT"]).pack(side=tk.LEFT)
        tk.Label(title_bar, text="Live transcription · Agent", font=FONT_UI, fg=THEME["TEXT_MUTED"], bg=THEME["BG_ROOT"]).pack(side=tk.LEFT, padx=(8, 0))
        self._settings_btn = ttk.Button(title_bar, text="Settings", command=self._open_settings)
        self._settings_btn.pack(side=tk.RIGHT, padx=(0, 20))

        main = tk.Frame(self.root, bg=THEME["BG_ROOT"], padx=20, pady=8)
        main.pack(fill=tk.BOTH, expand=True)

        # ---- Main strip: only Status, Level, Start, Stop ----
        self._app_var = tk.StringVar()
        strip = tk.Frame(main, bg=THEME["BORDER"], padx=1, pady=1)
        strip.pack(fill=tk.X, pady=(0, 8))
        inner = tk.Frame(strip, bg=THEME["BG_CARD"], padx=6, pady=4)
        inner.pack(fill=tk.X)
        F = ("Helvetica Neue", 9)
        row = tk.Frame(inner, bg=THEME["BG_CARD"])
        row.pack(fill=tk.X)
        tk.Label(row, text="Status", font=F, fg=THEME["TEXT_MUTED"], bg=THEME["BG_CARD"]).pack(side=tk.LEFT, padx=(0, 4))
        self._status_label = tk.Label(row, text="Stopped", font=F, fg=THEME["TEXT_MUTED"], bg=THEME["BG_CARD"])
        self._status_label.pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(row, text="Level", font=F, fg=THEME["TEXT_MUTED"], bg=THEME["BG_CARD"]).pack(side=tk.LEFT, padx=(0, 4))
        self._level_meter = LevelMeter(row, width=80, height=14)
        self._level_meter.pack(side=tk.LEFT, padx=(0, 10))
        self._start_btn = ttk.Button(row, text="Start", command=self._on_start)
        self._start_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._stop_btn = ttk.Button(row, text="Stop", command=self._on_stop, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT)

        # ---- Agent button (above the two columns) ----
        agent_frame = tk.Frame(main, bg=THEME["BG_ROOT"])
        agent_frame.pack(fill=tk.X, pady=(0, 12))
        self._agent_btn = tk.Button(
            agent_frame,
            text="Agent answer",
            font=(FONT_UI[0], 13, "bold"),
            fg="#ffffff",
            bg=THEME["ACCENT"],
            activeforeground="#ffffff",
            activebackground=THEME["ACCENT_HOVER"],
            relief=tk.FLAT,
            bd=0,
            padx=24,
            pady=10,
            cursor="hand2",
            command=self._on_agent_answer,
        )
        self._agent_btn.pack()

        # ---- Two columns: Live transcription | AI agent response ----
        two_cols = tk.Frame(main, bg=THEME["BG_ROOT"])
        two_cols.pack(fill=tk.BOTH, expand=True, pady=(0, 0))

        # Left column: Live transcription
        trans_outer, trans_inner = self._card(two_cols, "Live transcription")
        trans_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        btn_row = tk.Frame(trans_inner, bg=THEME["BG_CARD"])
        btn_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn_row, text="Clear", command=self._clear_transcription).pack(side=tk.LEFT)
        self._transcription_text = scrolledtext.ScrolledText(
            trans_inner,
            wrap=tk.WORD,
            height=12,
            font=FONT_TRANSCRIPTION,
            bg=THEME["BG_INPUT"],
            fg=THEME["TEXT"],
            insertbackground=THEME["TEXT"],
            relief=tk.FLAT,
            padx=10,
            pady=10,
            selectbackground=THEME["HIGHLIGHT_BG"],
            selectforeground=THEME["TEXT"],
        )
        self._transcription_text.pack(fill=tk.BOTH, expand=True)

        # Right column: AI agent response
        ai_outer, ai_inner = self._card(two_cols, "AI agent response")
        ai_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        ai_btn_row = tk.Frame(ai_inner, bg=THEME["BG_CARD"])
        ai_btn_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(ai_btn_row, text="Clear", command=self._clear_ai_responses).pack(side=tk.LEFT)
        self._ai_response_text = scrolledtext.ScrolledText(
            ai_inner,
            wrap=tk.WORD,
            height=12,
            font=FONT_AGENT,
            bg=THEME["BG_INPUT"],
            fg=THEME["TEXT"],
            insertbackground=THEME["TEXT"],
            relief=tk.FLAT,
            padx=10,
            pady=10,
            selectbackground=THEME["HIGHLIGHT_BG"],
            selectforeground=THEME["TEXT"],
        )
        self._ai_response_text.pack(fill=tk.BOTH, expand=True)

        self._title_bar = title_bar
        self._strip = strip
        self._strip_inner = inner
        self._strip_row = row
        self._agent_frame = agent_frame
        self._two_cols = two_cols
        self._main = main
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _open_settings(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("500x520")
        win.configure(bg=THEME["BG_ROOT"])
        f = tk.Frame(win, bg=THEME["BG_ROOT"], padx=16, pady=16)
        f.pack(fill=tk.BOTH, expand=True)

        # App
        tk.Label(f, text="App (route its output to BlackHole)", font=FONT_UI, fg=THEME["TEXT"], bg=THEME["BG_ROOT"]).pack(anchor=tk.W)
        app_row = tk.Frame(f, bg=THEME["BG_ROOT"])
        app_row.pack(fill=tk.X, pady=(2, 10))
        app_combo = ttk.Combobox(app_row, textvariable=self._app_var, state="readonly", width=28, font=FONT_UI)
        app_combo.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(app_row, text="Refresh", command=lambda: self._refresh_app_list(app_combo)).pack(side=tk.LEFT)
        self._refresh_app_list(app_combo)

        # Language
        tk.Label(f, text="Language", font=FONT_UI, fg=THEME["TEXT"], bg=THEME["BG_ROOT"]).pack(anchor=tk.W)
        lang_combo = ttk.Combobox(f, textvariable=self._language_var, state="readonly", width=22, font=FONT_UI, values=["en-US (English)", "it-IT (Italian)"])
        lang_combo.pack(anchor=tk.W, pady=(2, 10))
        lang_combo.bind("<<ComboboxSelected>>", self._on_language_change)

        # Playback & Mute
        opt_row = tk.Frame(f, bg=THEME["BG_ROOT"])
        opt_row.pack(fill=tk.X, pady=(0, 10))
        playback_cb = ttk.Checkbutton(opt_row, text="Playback", variable=self._playback_var, command=self._on_playback_toggle)
        playback_cb.pack(side=tk.LEFT, padx=(0, 16))
        mute_cb = ttk.Checkbutton(opt_row, text="Mute", variable=self._mute_var, command=self._on_mute_toggle)
        mute_cb.pack(side=tk.LEFT)

        tk.Label(f, text="Chunk duration (seconds)", font=FONT_UI, fg=THEME["TEXT"], bg=THEME["BG_ROOT"]).pack(anchor=tk.W)
        chunk_var = tk.StringVar(value=str(self._chunk_duration_sec))
        chunk_entry = tk.Entry(f, textvariable=chunk_var, width=10, font=FONT_UI)
        chunk_entry.pack(anchor=tk.W, pady=(2, 12))

        tk.Label(f, text="Theme", font=FONT_UI, fg=THEME["TEXT"], bg=THEME["BG_ROOT"]).pack(anchor=tk.W)
        theme_var = tk.StringVar(value=self._theme_name)
        theme_combo = ttk.Combobox(f, textvariable=theme_var, state="readonly", width=20, values=list(THEMES.keys()))
        theme_combo.pack(anchor=tk.W, pady=(2, 12))

        tk.Label(f, text="Agent instruction (persona)", font=FONT_UI, fg=THEME["TEXT"], bg=THEME["BG_ROOT"]).pack(anchor=tk.W)
        persona_text = scrolledtext.ScrolledText(f, wrap=tk.WORD, height=6, font=("Helvetica Neue", 10), bg=THEME["BG_INPUT"], fg=THEME["TEXT"])
        persona_text.pack(fill=tk.BOTH, expand=True, pady=(2, 12))
        persona_text.insert("1.0", self._agent_persona)

        def save():
            try:
                sec = float(chunk_var.get().strip())
                sec = max(0.5, min(10.0, sec))
                self._chunk_duration_sec = sec
            except ValueError:
                pass
            name = theme_var.get().strip()
            if name in THEMES:
                self._theme_name = name
                THEME.update(THEMES[name])
                self._apply_theme()
                self._apply_theme_to_widgets()
            self._agent_persona = persona_text.get("1.0", tk.END).strip() or DEFAULT_PERSONA
            win.destroy()

        ttk.Button(f, text="Save", command=save).pack(anchor=tk.W, pady=(0, 8))

    def _apply_theme_to_widgets(self) -> None:
        self.root.configure(bg=THEME["BG_ROOT"])
        self._title_bar.configure(bg=THEME["BG_ROOT"])
        for i, w in enumerate(self._title_bar.winfo_children()):
            if isinstance(w, tk.Label):
                w.configure(bg=THEME["BG_ROOT"], fg=THEME["TEXT"] if i == 0 else THEME["TEXT_MUTED"])
        self._main.configure(bg=THEME["BG_ROOT"])
        self._strip.configure(bg=THEME["BORDER"])
        self._strip_inner.configure(bg=THEME["BG_CARD"])
        self._strip_row.configure(bg=THEME["BG_CARD"])
        for w in self._strip_row.winfo_children():
            if isinstance(w, tk.Label):
                w.configure(bg=THEME["BG_CARD"], fg=THEME["TEXT_MUTED"])
        self._status_label.configure(bg=THEME["BG_CARD"], fg=THEME["TEXT_MUTED"])
        self._level_meter.configure(bg=THEME["BG_ROOT"])
        self._level_meter._canvas.configure(bg=THEME["BG_INPUT"])
        self._agent_frame.configure(bg=THEME["BG_ROOT"])
        self._agent_btn.configure(bg=THEME["ACCENT"], activebackground=THEME["ACCENT_HOVER"])
        self._two_cols.configure(bg=THEME["BG_ROOT"])
        self._transcription_text.configure(bg=THEME["BG_INPUT"], fg=THEME["TEXT"], insertbackground=THEME["TEXT"], selectbackground=THEME["HIGHLIGHT_BG"], selectforeground=THEME["TEXT"])
        self._ai_response_text.configure(bg=THEME["BG_INPUT"], fg=THEME["TEXT"], insertbackground=THEME["TEXT"], selectbackground=THEME["HIGHLIGHT_BG"], selectforeground=THEME["TEXT"])
        for card_outer in self._two_cols.winfo_children():
            card_outer.configure(bg=THEME["BORDER"])
            for inner in card_outer.winfo_children():
                inner.configure(bg=THEME["BG_CARD"])
                for child in inner.winfo_children():
                    if isinstance(child, tk.Label):
                        child.configure(bg=THEME["BG_CARD"], fg=THEME["TEXT"])
                    elif isinstance(child, tk.Frame):
                        child.configure(bg=THEME["BG_CARD"])

    def _refresh_app_list(self, combo: ttk.Combobox | None = None) -> None:
        apps = get_running_audio_apps()
        target = combo if combo is not None else getattr(self, "_app_combo", None)
        if target is not None:
            target["values"] = apps if apps else ["(No apps — run Zoom, Chrome, etc.)"]
        if apps and not self._app_var.get():
            self._app_var.set(apps[0])

    def _on_status(self, msg: str) -> None:
        self._status = msg
        color = THEME["ACCENT"] if "Capturing" in msg and "Failed" not in msg else THEME["TEXT_MUTED"]
        if "Failed" in msg:
            color = THEME["RED_ERR"]
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
        # Separator in transcription: mark what was sent to agent
        self._transcription_text.insert(tk.END, "\n——— sent to agent ———\n")
        self._transcription_text.see(tk.END)
        self._transcription_last_sent_len = len(self._transcription_text.get("1.0", tk.END))
        self._agent_busy = True
        self._agent_btn.config(state=tk.DISABLED, bg="#9e9e9e")
        self._append_ai_response("\n[Generating...]\n")

        def run_agent():
            try:
                response = get_agent_response(new_text, persona=self._agent_persona)
            except Exception as e:
                response = f"[Agent error: {e}]"
            self.root.after(0, lambda: self._agent_done(response))

        threading.Thread(target=run_agent, daemon=True).start()

    def _agent_done(self, response: str) -> None:
        self._agent_busy = False
        self._agent_btn.config(state=tk.NORMAL, bg=THEME["ACCENT"])
        txt = self._ai_response_text
        content = txt.get("1.0", tk.END)
        if "[Generating...]" in content:
            lines = content.split("\n")
            new_lines = [l for l in lines if "[Generating...]" not in l]
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, "\n".join(new_lines))
            if new_lines and not new_lines[-1].endswith("\n"):
                txt.insert(tk.END, "\n")
        # Separator between agent answers
        if txt.get("1.0", tk.END).strip():
            self._append_ai_response("\n—————————————\n")
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
            chunk_duration_sec=self._chunk_duration_sec,
        )
        self._transcriber.start()
        self._on_status(f"Capturing audio — {self._chunk_duration_sec} s chunks (click Stop to end)")
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
