# LiveKit Emotional Voice AI Agent

A simple voice AI agent with emotion detection that works with the LiveKit playground.

## Setup

1. **Clone and install dependencies:**

   ```console
   cd agent-starter-python
   uv sync
   ```

2. **Copy environment file:**

   ```console
   cp .env.example .env.local
   ```

3. **Add your API keys to `.env.local`:**

   ```
   LIVEKIT_URL=your_livekit_url
   LIVEKIT_API_KEY=your_api_key
   LIVEKIT_API_SECRET=your_api_secret
   GROQ_API_KEY=your_groq_key
   DEEPGRAM_API_KEY=your_deepgram_key
   ELEVENLABS_API_KEY=your_elevenlabs_key
   ```

4. **Download required models:**
   ```console
   uv run python src/agent.py download-files
   ```

## Run the Agent

Start the agent for use with LiveKit playground:

```console
uv run python src/agent.py dev
```

## Use in LiveKit Playground

1. Start your agent with the `dev` command above
2. Go to your LiveKit project dashboard
3. Open the "Agents" section
4. Connect to your running agent
5. Join the playground room and start talking!

The agent will:

- Listen to your speech (STT via Deepgram)
- Detect emotions in your words
- Respond with appropriate emotional tone (LLM via Groq)
- Speak back with emotional voice synthesis (TTS via ElevenLabs)

That's it! Your emotional AI agent is ready to use.
