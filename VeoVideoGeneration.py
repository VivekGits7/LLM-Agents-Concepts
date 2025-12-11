"""
VeoVideoGeneration - AI Video Generation using Google Veo API

This module generates a multi-part cinematic video sequence using Google's Veo 3.1
video generation model. It creates a 4-part promotional video for an "AI Evaluator"
product that demonstrates AI-powered exam paper evaluation.

Video Sequence:
    Part 1: AI robot emerges from a crate in a dramatic reveal scene
    Part 2: Robot demonstrates scanning and evaluating exam papers
    Part 3: Results visualization showing efficiency metrics and fairness
    Part 4: Global finale showing worldwide adoption of the AI system

Features:
    - Sequential video generation with scene continuity
    - Video extension using previous clips as reference
    - Automatic polling for operation completion
    - Individual MP4 file output for each video part

Requirements:
    - google-genai package
    - Valid Google API key with Veo access
    - Internet connection for API calls

Output Files:
    - Generated_Video_Part1.mp4: Base video (8 seconds, 720p, 16:9)
    - Generated_Video_Part2.mp4: First extension
    - Generated_Video_Part3.mp4: Second extension
    - Generated_Video_Part4.mp4: Final extension

Usage:
    Run the script directly to generate all 4 video parts:

    $ python VeoVideoGeneration.py

Note:
    Video generation may take several minutes per part due to API processing time.
    The script polls the API every 10 seconds until each operation completes.

Author: [Your Name]
Date: 2025
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# 🔑 API Key from .env
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
client = genai.Client(api_key=api_key)

# ============================================
# PROMPTS (Each for a sequential extension)
# ============================================
prompts = [
    # PART 1: Base intro
    """
    {
  "description": "In a dark chamber illuminated by the deep indigo hue #1B113D, a sleek metallic crate rests at the center. The surface bears a subtle glowing emblem: 'AI Evaluator'. A soft pulse of blue light ripples across the walls, hinting at energy within. Slowly, the crate opens in symmetrical precision, releasing a soft mist and radiant white-blue beams. From within, an elegant AI robot emerges — silver, chrome, and softly illuminated — stepping forward with deliberate calm. Around it, thousands of handwritten exam paper sheets float upward in graceful slow motion. The robot observes, eyes glowing with analytical intent, as the camera glides smoothly around the scene.",
  "style": "cinematic photorealistic sci-fi reveal",
  "camera": "slow 360° orbit with gentle zoom toward the robot as it awakens",
  "lighting": "dark base with cool highlights; deep indigo background (#1B113D) accented with white-blue rim light from the crate",
  "room": "minimal futuristic void infused with soft ambient mist and glowing indigo gradients",
  "elements": [
    "AI Evaluator crate with glowing emblem",
    "AI humanoid robot emerging gracefully",
    "floating handwritten exam sheets",
    "slow ambient mist rising",
    "subtle holographic sparks around crate"
  ],
  "motion": "crate opens; mist rises; robot steps out; exam papers lift upward",
  "ending": "robot looks forward as floating papers surround it in perfect balance against the indigo background",
  "text": "none",
  "keywords": [
    "AI Evaluator",
    "robot reveal",
    "crate opening",
    "deep indigo lighting",
    "futuristic education",
    "cinematic awakening"
  ]
}
    """,

    # PART 2: Feature demonstration
    """
    {
  "description": "Continuing from the previous scene, the camera zooms into the AI robot now fully activated. Its eyes glow bright cyan against the indigo #1B113D background. Dozens of exam paper sheets orbit around it. The robot extends its arms, releasing fine beams of scanning light that sweep across the papers at incredible speed. Each paper glows momentarily as data streams into holographic layers — handwriting converts into digital text, answers analyzed, and fair scores calculated instantly. Holographic dashboards form briefly, showing efficiency metrics and evaluation speed. The robot moves with elegance, precision, and total neutrality.",
  "style": "cinematic photorealistic sci-fi",
  "camera": "dynamic mid-range dolly with circular tracking around robot and papers",
  "lighting": "deep indigo ambient (#1B113D) with radiant cyan and silver glows from scanning beams",
  "room": "floating holographic space with thin light grids under the indigo haze",
  "elements": [
    "AI robot scanning floating papers",
    "rapid holographic data streams",
    "dynamic particle glows",
    "evaluation dashboards forming in light",
    "papers rotating in circular motion"
  ],
  "motion": "robot activates scanners; beams sweep across papers; holographic metrics form and fade",
  "ending": "camera pulls back revealing hundreds of evaluated papers glowing in perfect symmetry around the robot",
  "text": "none",
  "keywords": [
    "AI evaluation",
    "robot grading",
    "handwritten exam scanning",
    "neutral grading",
    "cinematic indigo tone",
    "fairness and precision"
  ]
}
    """,

    # PART 3: Result scene
    """
    {
  "description": "The evaluated papers dissolve into streams of glowing blue data that spiral upward, transforming into holographic charts showing time and cost reduction. The robot stands still at the center of the indigo #1B113D chamber, surrounded by floating analytics: time saved, resources reduced, and evaluation speed multiplied. Around the robot, silhouettes of human teachers appear as soft holograms — they watch as the AI handles thousands of papers in seconds. Subtle motion graphics visualize a drop in human workload and operational cost. The robot raises its hand, projecting a glowing sphere symbolizing equality and balance — every student, evaluated fairly.",
  "style": "cinematic futuristic tech-commercial",
  "camera": "slow zoom out with gentle rotational drift, emphasizing holographic metrics and symbolic motion",
  "lighting": "deep indigo base (#1B113D) with radiant blue and silver energy highlights",
  "room": "minimal digital space with floating data holograms and shimmering floor reflections",
  "elements": [
    "AI robot centered under light beam",
    "data streams converting to metrics",
    "holographic teachers observing",
    "floating fairness sphere",
    "soft mist and light flares"
  ],
  "motion": "data ascends; metrics animate; robot projects sphere of light; holograms appear and fade",
  "ending": "robot stands calm as the fairness sphere glows softly above its palm, symbolizing balanced evaluation for all",
  "text": "none",
  "keywords": [
    "AI efficiency",
    "cost reduction",
    "time saving",
    "education technology",
    "fair evaluation",
    "cinematic sci-fi tone"
  ]
}   
    """,

    # PART 4: Final brand outro
    """
    {
  "description": "The sphere of fairness bursts into radiant blue light, transforming the scene into a vast digital world map glowing in indigo #1B113D. Thousands of light connections spread across continents — representing universities, schools, and institutions powered by AI Evaluator. In each connection node, holographic visuals show students receiving results instantly and teachers reviewing insights effortlessly. The AI robot appears in the sky as a symbolic guardian of academic integrity, its reflection mirrored in the digital grid below. As the music softens, the environment fades into a calm glow — a message of global equality, efficiency, and trust in AI-driven evaluation.",
  "style": "cinematic photorealistic global finale",
  "camera": "wide aerial pullback revealing entire global digital network",
  "lighting": "indigo base (#1B113D) with radiant blue and silver world-light veins",
  "room": "expansive digital world map made of holographic grids and light threads",
  "elements": [
    "glowing fairness sphere expanding into world map",
    "network of light connections across continents",
    "floating holographic universities and exam systems",
    "AI robot reflected above digital earth",
    "soft indigo-to-blue fade"
  ],
  "motion": "sphere expands; network lights up; data connections flow globally; camera rises to wide aerial reveal",
  "ending": "the glowing world map stabilizes under calm blue light with the AI robot’s reflection shimmering above, symbolizing a future of unbiased evaluation",
  "text": "none",
  "keywords": [
    "AI Evaluator",
    "global network",
    "education equality",
    "time and cost efficiency",
    "AI grading system",
    "indigo cinematic aesthetic",
    "fairness and trust"
  ]
}
    """
]

# ============================================
# FUNCTION: Wait for operation to finish
# ============================================
def wait_for_operation(op, label="video"):
    while not op.done:
        print(f"⏳ Waiting for {label} generation...")
        time.sleep(10)
        op = client.operations.get(op)
    print(f"✅ {label.capitalize()} generated successfully.")
    return op

# ============================================
# STEP 1: Generate first video
# ============================================
print("🎬 Generating Part 1 (base video)...")

op = client.models.generate_videos(
    model="veo-3.1-generate-preview",
    prompt=prompts[0],
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        duration_seconds=8,
        aspect_ratio="16:9",
        resolution="720p",
    ),
)
op = wait_for_operation(op, "base video")

# Save first clip
video_clip = op.response.generated_videos[0]
client.files.download(file=video_clip.video)
video_clip.video.save("Generated_Video_Part1.mp4")
print("💾 Saved: Generated_Video_Part1.mp4")

# ============================================
# STEP 2–4: Extend video 3 times
# ============================================
for i in range(1, 4):
    print(f"\n🎬 Extending Video — Part {i + 1}...")
    op = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        video=video_clip,  # extend from previous
        prompt=prompts[i],
        config=types.GenerateVideosConfig(
            number_of_videos=1,
            resolution="720p",
        ),
    )

    op = wait_for_operation(op, f"extension part {i + 1}")
    video_clip = op.response.generated_videos[0]

    filename = f"Generated_Video_Part{i + 1}.mp4"
    client.files.download(file=video_clip.video)
    video_clip.video.save(filename)
    print(f"💾 Saved: {filename}")

print("\n🎉 All 4 parts generated successfully and saved!")
