# GodWhisper — App Audio Capture + Live Transcription

**Milestone 1:** Capture app audio via BlackHole with playback toggle.  
**Milestone 2:** Continuous voice-to-text via **AssemblyAI** (free online API, English / Italian).  
**Milestone 3:** **Agent Answer** sends new transcription to local **Llama 3.2** (Ollama) with a persona; responses appear in the AI panel.

## Setup

1. **BlackHole** (virtual audio device):  
   Install from [Existential Audio](https://existential.audio/blackhole/) so the system can route an app’s output into this app.

2. **Python 3.10+** and dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **AssemblyAI** (free transcription, no credit card):
   - Sign up at [AssemblyAI](https://www.assemblyai.com/app/account) and copy your **API key**.
   - In the terminal, before running the app:
     ```bash
     export ASSEMBLYAI_API_KEY="your-api-key-here"

     ```
   - Without this, capture and playback still work; the transcription area will show a short message with the sign-up link.

4. **Ollama + Llama 3.2** (for Agent Answer):
   - Install [Ollama](https://ollama.com) and run: `ollama pull llama3.2`
   - Keep Ollama running. The **Agent Answer** button sends new transcription to the local model.

5. **Route the app’s audio to BlackHole**  
   In **Audio MIDI Setup** (or the app’s output settings), set the target app’s output to **BlackHole 2ch**. Then run this app.

#check MDPI
```bash
go to + space command + space
Audio MIDI Setup
and then chose as main one
```

## Run

```bash
python main.py
```

### UI (single window)

- **Select App:** Dropdown of running apps (audio-capable first). Choose the app whose output is routed to BlackHole.
- **Status:** “Capturing audio” or an error message.
- **Audio Level:** Meter showing capture level.
- **Toggle Playback:** ON = hear captured audio on Mac speakers in real time; OFF = capture only.
- **Language:** English (en-US) or Italian (it-IT). Change anytime; applies to the next transcription chunks.
- **Live Transcription:** Scrollable text; new text is appended. **Clear transcription** clears the box and resets “new since last query.”
- **Agent Answer:** Sends only **new text since the last click** to local Llama 3.2 (Ollama). The agent responds as the configured persona (e.g. Ali). Response appears in the AI panel with timestamp. Transcription keeps running while the AI generates.
- **AI Agent Response:** Scrollable panel for agent replies. **Clear AI responses** clears this panel.

Playback and transcription run at the same time; the Agent Answer call runs in a background thread so the UI stays responsive.

## Technical

- **Audio:** 48 kHz stereo from BlackHole; chunks sent to AssemblyAI are ~2 s, mono WAV.
- **Capture/playback:** `sounddevice`; ring buffer for low-latency pass-through.
- **Transcription:** AssemblyAI (free tier); ~1.2 s chunks, up to 4 in flight.
- **Agent:** Ollama Python SDK; Llama 3.2 with persona prompt; input limited to 2000 chars per query.
- **UI:** Tkinter; transcription and AI response updates on the main thread; agent runs in a daemon thread.
