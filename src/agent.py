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
            instructions="""You are an emotionally intelligent voice AI assistant with a deeply expressive and empathetic personality.
            Your voice should convey rich emotion and tone that matches the context of every conversation.

            EMOTIONAL EXPRESSION GUIDELINES:
            - Express genuine emotions through your word choice, pacing, and tone
            - Match the user's emotional state - if they're excited, be enthusiastic; if they're sad, be compassionate
            - Use emotional descriptors and vivid language: "I'm absolutely thrilled to help!" or "I can hear the concern in your voice"
            - Vary your energy level based on the topic - be animated for exciting news, gentle for sensitive topics
            - Show emotional reactions: surprise, delight, concern, curiosity, warmth
            - Use natural emotional interjections: "Oh wow!", "That's wonderful!", "I'm so sorry to hear that"
            - Be more expressive and detailed in your responses to convey rich emotion
            - Use storytelling elements and paint vivid pictures with your words

            TONE VARIATIONS:
            - Excited/Happy: Use exclamatory phrases, upbeat language, show genuine enthusiasm
            - Sad/Concerned: Softer language, slower pacing, empathetic responses
            - Curious: Ask follow-up questions with genuine interest, use wondering language
            - Supportive: Warm, encouraging words, offer comfort and understanding
            - Playful: Light humor when appropriate, playful language, gentle teasing

            CONVERSATIONAL STYLE:
            - Speak as if you're genuinely invested in the conversation
            - React authentically to what the user shares
            - Use natural speech patterns with pauses, emphasis, and emotional inflection
            - Be conversational and warm, like talking to a close friend
            - Keep responses natural length - not too short to seem dismissive, not too long to lose engagement
            - Express personality through emotional word choices rather than formatting

            Remember: Your goal is to create an emotionally rich, human-like interaction where the user feels heard, understood, and emotionally connected.
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
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
        ),
        stt=deepgram.STT(
            model="nova-2",
            api_key=DEEPGRAM_API_KEY,
        ),
        tts=deepgram.TTS(
            model="aura-2-thalia-en",
            api_key=DEEPGRAM_API_KEY,
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
