---
name: analyze-video
description: Analyze a local video file (Teams meeting recording, screen demo; .mp4/.mkv/.mov/.webm) fully offline. Produces a timestamped transcript (embedded subtitles or faster-whisper on the GPU) plus frames of the shared screen, reads them in parts and writes a report cross-referenced with the open project. Use when the user says "analyze this video", "analisar vídeo", "transcrever a reunião", gives a video path to understand, or types /analyze-video <path>.
---

# Analyze video

`scripts/prepare_video.py` does the heavy part with no LLM (transcript, frame picking, 3x3 contact sheets, parts of
10 min). You then read the parts in order and write the report. Nothing leaves the machine except the one-time Whisper
model download (`large-v3-turbo`, ~1.6 GB, Hugging Face cache).

## 1. Inputs

- **Video:** the path in the user's message (quote it: paths have spaces and accents). If there is none, ask for it.
  Never move, rename or delete the video.
- **Output:** `%USERPROFILE%\Downloads\<video file name without extension>_analise\` (user decision), even when the video
  is elsewhere.
- **Project context:** `<project root>\.claude\video-context.md`, project root = git root of the working directory. It
  has the `## Vocabulário` section for Whisper, what to read to relate the video to the project, the report columns and
  rules. Missing file: relate only to what is already in context (CLAUDE.md, memory) and say so in the report.

## 2. Prepare (background)

```
python "<this skill>\scripts\prepare_video.py" "<video>" --out "<output>" --vocab "<project root>\.claude\video-context.md"
```

- Use `python` without `-I` (the packages are in the user site-packages). Run it with `run_in_background` and wait for
  the completion notice; do not poll. 1h30 takes about 5-10 min on the GPU (more on the first run, model download).
- While it runs, read the files the project context lists (backlog, where the project stopped).
- Useful options: `--from hh:mm:ss --to hh:mm:ss` (window), `--device cpu`, `--whisper` (ignore embedded subtitles),
  `--scene 0.08` (lower = more frames), `--floor-s 60`, `--chunk-min 10`, `--force`. Re-running skips finished stages.
- On failure, read `<output>\_work\prepare.log`.

## 3. Read the parts

1. Read `<output>\_work\MANIFEST.md`.
2. For each `_work\partes\NN_*.md`: read it, then the contact sheets it lists (the timestamp is in each tile's corner).
   Open a full-resolution frame only when on-screen text matters (a dialog, a tree, attribute values, a file name).
3. Before moving to the next part, write `_work\notas\NN.md`: short bullets `[hh:mm:ss] fact` — decisions, requests,
   problems shown on screen, element/command/file names, numbers, anything that touches a project item. Use roles, not
   people's names. If the context gets compacted, resume from these notes instead of re-reading parts.
4. Sheets without screen content (only cameras) need no attention; the transcript carries those parts.

## 4. Report

Write `<output>\RELATORIO.md` in Brazilian Portuguese:

```
# <assunto da reunião em poucas palavras> (<data da gravação>)

Vídeo `<caminho>` · <duração> · transcrição: <fonte do MANIFEST> · `transcricao.md`. Horários = posição no vídeo.

## Resumo
<5-10 linhas: do que se tratou, o que foi decidido, o que ficou pendente>

## Decisões e pendências
<uma tabela por área/disciplina do contexto do projeto; sem contexto, uma tabela só>
| Decisão ou pendência | Momento | <coluna "hoje" do contexto> | Tarefa |

## Perguntas em aberto
<o que precisa de resposta, e de quem (papel)>

## Ações candidatas (nada foi executado)
<código, configuração, registros; cada uma ligada a um momento do vídeo>

## Índice por assunto
| Momento | Assunto |
```

- Every claim gets a `hh:mm:ss`. Mark doubtful transcript passages as such instead of guessing.
- Frames cited in the report: copy them to `<output>\quadros\` and link them relatively (`quadros/hh-mm-ss.jpg`).

## 5. Finish

- After `RELATORIO.md` is written: delete `<output>\_work\` (keeps `transcricao.md`, `transcricao.srt`, `RELATORIO.md`,
  `quadros\`).
- Reply in chat with a short summary, the report path and the candidate actions. Record nothing in the project (logs,
  glossary, task boards, memory) without the user's ok.

## Requirements

`ffmpeg`/`ffprobe` (Windows: `winget install Gyan.FFmpeg`; the script also finds them in
`%LOCALAPPDATA%\Microsoft\WinGet\Links`), `pip install faster-whisper pillow`, and for an NVIDIA GPU
`pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"` (without them it runs on the CPU). Audio is decoded by ffmpeg
and passed as an array, so faster-whisper's own decoder (PyAV) is not used. If they are missing, tell the user the
commands above instead of running the script.
