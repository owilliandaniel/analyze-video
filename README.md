# analyze-video

A [Claude Code](https://claude.com/claude-code) skill that analyzes a local video (a Teams meeting recording, a screen
demo) fully offline: a timestamped transcript, frames of the shared screen, and a report that relates what was said and
shown to the project open in Claude Code.

Say "analyze this video `<path>`" or type `/analyze-video <path>`.

## How it works

`scripts/prepare_video.py` does the heavy part, with no LLM:

1. **Transcript:** embedded subtitles if the file has them; otherwise [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   (`large-v3-turbo`, GPU with CPU fallback, voice activity filter), with project terms in the initial prompt.
2. **Frames of the shared screen:** the video is scanned at 1 frame/s in 160×90 grey; blocks that keep changing (camera
   tiles, the active-speaker frame) are masked per 60 s window, and a frame is kept when the rest of the screen changes
   against the last kept frame, once it settles. A slow-changing screen still gets a frame every `--floor-s` seconds.
3. **3×3 contact sheets** with the timestamp on each tile, and **parts** of 10 minutes (transcript + sheets + frames).

Claude then reads the parts in order, keeps notes per part, and writes `RELATORIO.md`. Output goes to
`Downloads\<video name>_analise\`: `transcricao.md`, `transcricao.srt`, `RELATORIO.md`, `quadros\`. The working
folder is deleted at the end; the video itself is never moved or changed. Reports and transcripts are in Brazilian
Portuguese.

## Install

```
winget install Gyan.FFmpeg
pip install faster-whisper pillow
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"   # NVIDIA GPU only
git clone https://github.com/owilliandaniel/analyze-video
mklink /J "%USERPROFILE%\.claude\skills\analyze-video" "<clone folder>"
```

The Whisper model (~1.6 GB) downloads once on first use.

## Project context

Optional `<project root>\.claude\video-context.md` tells the skill the Whisper vocabulary (`## Vocabulário` section),
what to read to relate the video to the project, and how to shape the report. See
[`examples/video-context.md`](examples/video-context.md).

## Script options

```
python scripts/prepare_video.py VIDEO --out DIR [--vocab FILE ...] [--lang pt] [--model large-v3-turbo]
       [--device auto|cuda|cpu] [--from hh:mm:ss --to hh:mm:ss] [--chunk-min 10] [--floor-s 60]
       [--scene 0.08] [--whisper] [--force]
```

Stages are cached in `DIR\_work`; running again skips what is done with the same parameters.
