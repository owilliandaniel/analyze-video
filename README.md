# analyze-video

A [Claude Code](https://claude.com/claude-code) skill that transcribes any local video (a meeting recording, a screen
demo, a lecture) fully offline into **one file**, `transcricao.md`: the speech with timestamps plus short `[tela]` lines
describing what is on screen.

Say "analyze this video `<path>`" or type `/analyze-video <path>`.

## How it works

`scripts/prepare_video.py` does the heavy part, with no LLM:

1. **Speech:** embedded subtitles if the file has them; otherwise [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   (`large-v3-turbo`, GPU with CPU fallback, voice activity filter), with optional project terms in the initial prompt.
2. **Screen changes:** the video is scanned at 1 frame/s in 160×90 grey; blocks that keep changing (camera tiles, the
   active-speaker frame) are masked per 60 s window, and a frame is kept when the rest of the screen changes against the
   last kept frame, once it settles. A slow-changing screen still gets a frame every `--floor-s` seconds.
3. **3×3 contact sheets** with the timestamp on each tile, and **parts** of 10 minutes (speech + sheets + frames).

Claude reads the parts, writes one `[hh:mm:ss] …` line per relevant screen change, and `--finish` merges those lines into
`transcricao.md` by time and deletes the work folder (`%TEMP%\analyze-video\…`). Output goes to
`Downloads\<video name>_analise\transcricao.md`; the video itself is never moved or changed. The transcript header and
notes are in Brazilian Portuguese.

## Install

```
winget install Gyan.FFmpeg
pip install faster-whisper pillow
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"   # NVIDIA GPU only
git clone https://github.com/owilliandaniel/analyze-video
mklink /J "%USERPROFILE%\.claude\skills\analyze-video" "<clone folder>"
```

The Whisper model (~1.6 GB) downloads once on first use.

## Vocabulary

Optional `<project root>\.claude\video-context.md` with a `## Vocabulário` section (comma-separated terms) helps Whisper
with names and jargon. See [`examples/video-context.md`](examples/video-context.md).

## Script

```
python scripts/prepare_video.py VIDEO --out DIR [--vocab FILE ...] [--lang pt] [--model large-v3-turbo]
       [--device auto|cuda|cpu] [--from hh:mm:ss --to hh:mm:ss] [--chunk-min 10] [--floor-s 60]
       [--scene 0.08] [--whisper] [--force] [--work DIR]
python scripts/prepare_video.py VIDEO --out DIR --finish
```

Stages are cached in the work folder; running again skips what is done with the same parameters.
