from __future__ import annotations

import re
import time
from pathlib import Path
from faster_whisper import WhisperModel

INPUT_DIR = Path("videos")
OUTPUT_DIR = Path("results/transcripts")
MODEL = "large-v3"
SENTENCES_PER_PARAGRAPH = 4


def to_paragraphs(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    paragraphs = [
        " ".join(sentences[i : i + SENTENCES_PER_PARAGRAPH])
        for i in range(0, len(sentences), SENTENCES_PER_PARAGRAPH)
    ]
    return "\n\n".join(paragraphs)


def transcribe_file(model, path: Path) -> float:
    segments_gen, info = model.transcribe(
        str(path),
        language="en",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    text = " ".join(seg.text.strip() for seg in segments_gen).strip()
    (OUTPUT_DIR / f"{path.stem}.txt").write_text(to_paragraphs(text) + "\n", encoding="utf-8")
    return getattr(info, "duration", 0.0) or 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob("*.mp4"))
    if not files:
        raise SystemExit(f"No .mp4 files in {INPUT_DIR}")

    model = WhisperModel(MODEL, device="cuda", compute_type="float16")

    total_audio = 0.0
    t0 = time.time()
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name} ...", flush=True)
        total_audio += transcribe_file(model, path)

    elapsed = time.time() - t0
    print(
        f"\nDone. {len(files)} files ({total_audio/60:.1f} min of audio) "
        f"in {elapsed/60:.1f} min. Output: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
