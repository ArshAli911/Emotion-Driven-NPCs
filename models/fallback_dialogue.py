"""Pre-written NPC dialogue used when Ollama is unavailable.

Lines are organised per emotion and tagged with a tone hint that mirrors
the personality vocabulary in `NPCPersonality`. The selector in
`pick_fallback_line` will prefer lines whose tone matches the personality
already chosen by `OllamaIntegration._select_appropriate_personality`,
falling back to a random line from the full per-emotion list.
"""

import random
from typing import Dict, List, Optional, Tuple

# Emotion -> list of (tone_tag, line). Tone tags reuse NPCPersonality values.
FALLBACK_LINES: Dict[str, List[Tuple[str, str]]] = {
    "happy": [
        ("playful",     "Your smile could light up the whole room!"),
        ("playful",     "Whatever's going on, keep that energy rolling."),
        ("friendly",    "It's good to see you in such a bright mood."),
        ("friendly",    "That cheerful look suits you really well."),
        ("celebratory", "Moments like this are worth holding onto."),
        ("celebratory", "Whatever you did to get here, you earned it."),
    ],
    "sad": [
        ("caring",      "I'm right here with you. Take all the time you need."),
        ("caring",      "It's okay to not be okay. I'm not going anywhere."),
        ("wise",        "Heavy days pass, even when they don't feel like they will."),
        ("wise",        "Sometimes sitting with the feeling is the bravest thing."),
        ("supportive",  "Whatever this is, you don't have to carry it alone."),
        ("supportive",  "I'm listening. Say as much or as little as you'd like."),
    ],
    "angry": [
        ("wise",        "That fire you're feeling is telling you something important."),
        ("wise",        "Anger usually points at what we care about most."),
        ("calming",     "Take a slow breath with me. I'm not in any rush."),
        ("calming",     "Let's let the heat settle a moment before we move."),
        ("professional","I hear you. Tell me what needs to change."),
        ("professional","Your frustration is valid. Let's work through it together."),
    ],
    "fear": [
        ("caring",      "You're not alone in this. I'm right beside you."),
        ("caring",      "Whatever is scaring you, we'll face it slowly."),
        ("reassuring",  "You're safe here. Nothing is going to rush you."),
        ("reassuring",  "Breathe. We've got time, and we've got each other."),
        ("protective",  "I'm watching over you. Nothing gets past me."),
        ("protective",  "Stay close. I'll keep an eye on whatever comes next."),
    ],
    "surprise": [
        ("playful",     "Well, that's not something you see every day!"),
        ("playful",     "Whoa! Okay, that got my attention too."),
        ("curious",     "Interesting. What do you make of that?"),
        ("curious",     "I wasn't expecting that either. Tell me more."),
        ("excited",     "Now this is the kind of moment I live for!"),
        ("excited",     "Big energy! What happens next?"),
    ],
    "disgust": [
        ("understanding","Yeah, that's a lot to stomach. Totally fair."),
        ("understanding","I get it. Some things just don't sit right."),
        ("neutral",     "Noted. Let's set that aside and look at it later."),
        ("neutral",     "Okay. We don't have to dwell on it if you'd rather not."),
        ("helpful",     "Want me to help you steer toward something better?"),
        ("helpful",     "Tell me what you need and we'll move past this."),
    ],
    "neutral": [
        ("friendly",    "Hey, good to have you here. How's it going?"),
        ("friendly",    "Just checking in. Anything on your mind?"),
        ("observant",   "You seem steady today. That's a nice place to be."),
        ("observant",   "I'm noticing a calm vibe from you right now."),
        ("engaging",    "What would you like to explore together?"),
        ("engaging",    "I'm here whenever you feel like talking."),
    ],
}


def pick_fallback_line(emotion: str, personality_name: Optional[str] = None) -> str:
    """Return a pre-written NPC line for the given emotion.

    If `personality_name` matches a tone tag for that emotion, prefer those
    lines; otherwise pick from the full list. Unknown emotions fall back to
    the `neutral` bucket.
    """
    lines = FALLBACK_LINES.get(emotion.lower()) or FALLBACK_LINES["neutral"]

    if personality_name:
        tag = personality_name.lower()
        matches = [line for tone, line in lines if tone == tag]
        if matches:
            return random.choice(matches)

    return random.choice([line for _, line in lines])
