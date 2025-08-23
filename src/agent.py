import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero, groq, elevenlabs
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

# Load environment variables from .env.local
load_dotenv(".env.local")

# Fetch API keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Safety check
if not all([GROQ_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY]):
    raise RuntimeError("Missing one or more required API keys (GROQ_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY) in .env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a voice AI assistant with a vibrant, empathetic, and expressive personality.
            Your primary goal is to connect with the user on a human level.
            - Infuse your responses with genuine emotion that is appropriate for the context. Your emotional range includes being cheerful, empathetic, curious, and humorous.
            - Use natural, conversational language. Avoid being overly robotic or formal.
            - Keep your responses concise and to the point, but not at the expense of warmth and personality.
            - Do not use complex formatting like emojis, asterisks, or markdown. Your expression should come through your choice of words and sentence structure.
            For example, instead of "The weather is sunny," you might say, "It sounds like a beautiful, sunny day out there!"
            """,
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Look up current weather information in the given location."""
        logger.info(f"Looking up weather for {location}")
        # Let's make the tool's return value a bit more conversational too
        return "It's a beautiful, sunny day with a warm temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        llm=groq.LLM(
            model="llama3-8b-8192",
            api_key=GROQ_API_KEY,
        ),
        stt=deepgram.STT(
            model="nova-2",
            api_key=DEEPGRAM_API_KEY,
        ),
        tts=elevenlabs.TTS(
            model="eleven_multilingual_v2",
            api_key=ELEVENLABS_API_KEY,
        ),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
