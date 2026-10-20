---
name: analyze-video
description: Transcribe any local video or audio file (meeting recording, screen demo, lecture; .mp4/.mkv/.mov/.webm/.mp3/.m4a) fully offline into a single transcricao.md with timestamped speech plus short [tela] lines describing what is on screen. Use when the user says "analyze this video", "transcribe this video", "analisar vídeo", "transcrever o vídeo", or types /analyze-video <path>.
---

# Analyze video

The result is **one file**: `transcricao.md`, with the speech (embedded subtitles or faster-whisper on the GPU) and short
`[tela]` lines saying what is on screen. Everything else is temporary and deleted at the end. Nothing leaves the machine
except the one-time Whisper model download (`large-v3-turbo`, ~1.6 GB, Hugging Face cache).

## 1. Inputs

- **Video:** the path in the user's message (quote it: paths have spaces and accents). If there is none, ask for it.
  Never move, rename or delete the video.
- **Output:** `%USERPROFILE%\Downloads\<video file name without extension>_analise\transcricao.md`, even when the video
  is elsewhere. That folder ends with only this file.
- **Vocabulary (optional):** `<project root>\.claude\video-context.md`, section `## Vocabulário`, improves technical terms.

## 2. Prepare (background)

```
python "<this skill>\scripts\prepare_video.py" "<video>" --out "<output folder>" [--vocab "<project root>\.claude\video-context.md"]
```

- Use `python` without `-I` (the packages are in the user site-packages). Run it with `run_in_background` and wait for
  the completion notice; do not poll. A 1h30 meeting takes ~5 min of transcription on the GPU plus a few minutes of frame
  scanning (more on the first run, model download).
- It writes `transcricao.md` with the speech only, and a work folder in `%TEMP%\analyze-video\<output folder name>\`
  (`MANIFEST.md`, `partes\`, `grades\`, `quadros\`, `prepare.log`). Re-running skips finished stages.
- Options: `--from hh:mm:ss --to hh:mm:ss`, `--lang`, `--device cpu`, `--whisper` (ignore embedded subtitles),
  `--scene 0.08` (lower = more frames), `--floor-s 60`, `--chunk-min 10`, `--force`.

## 3. Describe the screen

1. Read `MANIFEST.md` in the work folder.
2. For each `partes\NN_*.md`: read it, then the contact sheets it lists (timestamp in each tile's corner). Open a
   full-resolution frame only when on-screen text matters.
3. Write `telas\NN.md` in the work folder, one line per relevant screen change: `[hh:mm:ss] <what is on screen>` —
   the application and what it shows (document, sheet, dialog, tree, model view), with names, values and file names
   that can be read. Short and factual; no interpretation, no people's names. Skip camera-only frames and repeats.
   If the context gets compacted, continue from the `telas\` files already written.

## 4. Finish

```
python "<this skill>\scripts\prepare_video.py" "<video>" --out "<output folder>" --finish
```

It merges the `[tela]` lines into `transcricao.md` by time and deletes the work folder. Check that the output folder
has only `transcricao.md`.

## 5. Reply

- **A single video:** say only that the transcript is ready, with its path and the video length.
- **The video is part of a set of files the user is analyzing together** (or the user asks to relate it): after the
  transcript is ready, read it and relate it in chat to the other material and to the open project. No extra files.
- Record nothing in the project (logs, glossary, task boards, memory) without the user's ok.

## Requirements

`ffmpeg`/`ffprobe` (Windows: `winget install Gyan.FFmpeg`; the script also finds them in
`%LOCALAPPDATA%\Microsoft\WinGet\Links`), `pip install faster-whisper pillow`, and for an NVIDIA GPU
`pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"` (without them it runs on the CPU). Audio is decoded by ffmpeg
and passed as an array, so faster-whisper's own decoder (PyAV) is not used. If they are missing, tell the user the
commands above instead of running the script.
