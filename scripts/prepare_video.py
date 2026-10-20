#!/usr/bin/env python
"""Prepara um vídeo (ex.: gravação do Teams) para leitura pelo Claude, tudo local:
transcrição com horário (legenda embutida ou faster-whisper), quadros da tela compartilhada,
grades 3x3 com o horário e partes de N minutos.

    python prepare_video.py VIDEO --out PASTA [--vocab ARQ ...] [--from hh:mm:ss --to hh:mm:ss] ...

Etapas com cache em PASTA/_work: rodar de novo pula o que já está pronto com os mesmos parâmetros.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import site
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
SCAN_W, SCAN_H, BLOCK = 160, 90, 5   # quadro de varredura 160x90, grade de 32x18 blocos
CHANGE = 8          # diferença de cinza (média do bloco) que conta como mudança
NOISY_RATE = 0.25   # bloco que muda em mais de 25% dos segundos (janela de 60 s) = câmera/destaque
MIN_STABLE = 0.2    # menos de 20% de blocos estáveis = só câmeras, sem tela para guardar
MIN_GAP = 2         # segundos mínimos entre dois quadros guardados
TILE_W, TILE_H = 640, 360
SKILL_DIR = Path(__file__).resolve().parent.parent

LOG_PATH: Path | None = None
FFMPEG = FFPROBE = ""


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def hms(t: float) -> str:
    t = max(0, int(t))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def parse_time(s: str | None) -> float | None:
    if not s:
        return None
    parts = [float(p) for p in s.replace(",", ".").split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def find_tool(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    # winget põe o alias aqui; o PATH novo só vale em terminal reaberto
    cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / f"{name}.exe"
    if cand.exists():
        return str(cand)
    sys.exit(f"{name} não encontrado (instalar: winget install Gyan.FFmpeg)")


def cached(path: Path, key: dict, force: bool) -> dict | None:
    if force or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("key") == key else None


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------- vocabulário ----------

def read_vocab(files: list[Path]) -> str:
    terms: list[str] = []
    for f in files:
        if not f.exists():
            log(f"vocabulário não encontrado: {f}")
            continue
        lines = f.read_text(encoding="utf-8").splitlines()
        # num .md, só a seção "## Vocabulário" (ou "## Vocabulary")
        if f.suffix.lower() == ".md":
            sect, inside = [], False
            for ln in lines:
                if ln.startswith("## "):
                    inside = ln[3:].strip().lower().startswith("vocab")
                    continue
                if inside:
                    sect.append(ln)
            lines = sect
        for ln in lines:
            ln = ln.strip()
            if ln.startswith("#"):  # comentário só no início da linha ("150#" é termo)
                continue
            if ln.startswith("- "):
                ln = ln[2:]
            for t in ln.split(","):
                t = t.strip().strip("`")
                if t and t not in terms:
                    terms.append(t)
    prompt = "Reunião técnica em português. Termos: "
    out = []
    for t in terms:  # initial_prompt do Whisper aceita ~224 tokens
        if len(prompt) + len(", ".join(out + [t])) > 700:
            break
        out.append(t)
    return prompt + ", ".join(out) + "." if out else ""


# ---------- metadados e transcrição ----------

def probe(video: Path) -> dict:
    p = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video)],
                       capture_output=True, check=True)
    return json.loads(p.stdout.decode("utf-8", "replace"))


def parse_srt(text: str) -> list[dict]:
    segs = []
    pat = re.compile(r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)[^\n]*\n(.*?)(?:\n\s*\n|\Z)", re.S)
    for a, b, body in pat.findall(text):
        body = re.sub(r"<[^>]+>", "", body).strip().replace("\n", " ")
        if body:
            segs.append({"start": parse_time(a), "end": parse_time(b), "text": body})
    return segs


def add_cuda_dll_dirs() -> None:
    # as rodas nvidia-* do pip deixam as DLLs em site-packages/nvidia/*/bin, fora da busca do Windows
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for root in roots:
        for d in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def load_audio(video: Path, start: float, end: float) -> np.ndarray:
    # decodifica pelo ffmpeg e entrega um array: não depende do av.open do faster-whisper
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
           "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def run_whisper(audio: np.ndarray, args, prompt: str, offset: float) -> tuple[list[dict], dict]:
    add_cuda_dll_dirs()
    from faster_whisper import WhisperModel

    total = len(audio) / SAMPLE_RATE
    attempts = {"auto": [("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")],
                "cuda": [("cuda", "float16"), ("cuda", "int8_float16")],
                "cpu": [("cpu", "int8")]}[args.device]
    last = None
    for device, compute in attempts:
        try:
            log(f"Whisper {args.model} em {device}/{compute}")
            model = WhisperModel(args.model, device=device, compute_type=compute, cpu_threads=os.cpu_count() or 4)
            segs, info = model.transcribe(audio, language=args.lang or None, vad_filter=True, beam_size=5,
                                          condition_on_previous_text=False, initial_prompt=prompt or None)
            out, next_log = [], 300.0
            for s in segs:  # a falha de CUDA pode aparecer só na primeira decodificação
                text = s.text.strip()
                if text:
                    out.append({"start": round(s.start + offset, 2), "end": round(s.end + offset, 2), "text": text})
                if s.end >= next_log:
                    log(f"  transcrito {hms(s.end)} de {hms(total)}")
                    next_log += 300.0
            return out, {"source": f"Whisper {args.model} ({device}/{compute})", "language": info.language}
        except Exception as e:  # noqa: BLE001
            last = e
            log(f"  falhou em {device}/{compute}: {e}")
    sys.exit(f"Whisper falhou: {last}")


def transcribe(args, video: Path, info: dict, start: float, end: float, work: Path, prompt: str) -> dict:
    path = work / "transcript.json"
    subs = [s for s in info.get("streams", []) if s.get("codec_type") == "subtitle"]
    use_subs = bool(subs) and not args.whisper
    key = {"start": start, "end": end, "subs": use_subs, "model": args.model, "lang": args.lang, "prompt": prompt}
    data = cached(path, key, args.force)
    if data:
        log("transcrição em cache")
        return data
    t0 = time.time()
    if use_subs:
        log("legenda embutida encontrada: usando no lugar do Whisper")
        srt = work / "embedded.srt"
        subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", str(video), "-map", "0:s:0", str(srt)], check=True)
        segs = [s for s in parse_srt(srt.read_text(encoding="utf-8", errors="replace")) if start <= s["start"] < end]
        meta = {"source": "legenda embutida", "language": subs[0].get("tags", {}).get("language", "")}
    else:
        log(f"decodificando áudio {hms(start)}-{hms(end)}")
        audio = load_audio(video, start, end)
        segs, meta = run_whisper(audio, args, prompt, start)
    data = {"key": key, "meta": meta, "segments": segs, "seconds": round(time.time() - t0)}
    save_json(path, data)
    log(f"transcrição: {len(segs)} trechos em {data['seconds']} s")
    return data


def write_transcript(data: dict, out: Path, header: list[str]) -> None:
    segs = data["segments"]
    with open(out / "transcricao.srt", "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}\n\n")
    paras: list[tuple[float, list[str]]] = []
    last_end, para_start = -1e9, 0.0
    for s in segs:  # parágrafo novo em pausa > 1,5 s ou a cada ~45 s
        if not paras or s["start"] - last_end > 1.5 or s["start"] - para_start > 45:
            paras.append((s["start"], []))
            para_start = s["start"]
        paras[-1][1].append(s["text"])
        last_end = s["end"]
    head = header + [f"Fonte: {data['meta']['source']}. Transcrição automática: termos técnicos podem sair errados."]
    body = [f"[{hms(t)}] {' '.join(txt)}" for t, txt in paras] or ["(sem fala)"]
    (out / "transcricao.md").write_text("\n\n".join(head + body) + "\n", encoding="utf-8")


# ---------- quadros da tela ----------

def scan(video: Path, start: float, end: float) -> np.ndarray:
    tail = ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video), "-an",
            "-vf", f"fps=1,scale={SCAN_W}:{SCAN_H},format=gray", "-f", "rawvideo", "-"]
    size = SCAN_W * SCAN_H
    for extra in (["-hwaccel", "auto"], []):
        p = subprocess.run([FFMPEG, "-nostdin", "-v", "error", *extra, *tail], capture_output=True)
        n = len(p.stdout) // size
        if p.returncode == 0 and n > 0:
            return np.frombuffer(p.stdout[:n * size], np.uint8).reshape(n, SCAN_H, SCAN_W)
        log(f"  varredura {'com' if extra else 'sem'} hwaccel falhou: {p.stderr.decode('utf-8', 'replace')[-300:]}")
    sys.exit("não consegui ler os quadros do vídeo")


def pick_times(frames: np.ndarray, args) -> list[tuple[int, str]]:
    n = len(frames)
    gh, gw = SCAN_H // BLOCK, SCAN_W // BLOCK
    blocks = frames.reshape(n, gh, BLOCK, gw, BLOCK).astype(np.float32).mean(axis=(2, 4))
    step = np.abs(np.diff(blocks, axis=0)) > CHANGE if n > 1 else np.zeros((0, gh, gw), bool)
    # bloco que vive mudando (câmera, moldura de quem fala) vira ruído, numa janela de 60 s
    csum = np.concatenate([np.zeros((1, gh, gw)), np.cumsum(step, axis=0)])
    idx = np.arange(n)
    lo, hi = np.clip(idx - 30, 0, n - 1), np.clip(idx + 30, 0, n - 1)
    noisy = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)[:, None, None] > NOISY_RATE
    total = gh * gw

    def settle(i: int) -> int:  # espera a tela assentar (animação, rolagem) por até 5 s
        for k in range(i, min(i + 6, n - 1)):
            stable = ~noisy[k]
            if (step[k] & stable).sum() < 0.02 * max(stable.sum(), 1):
                return k
        return min(i + 5, n - 1)

    kept: list[tuple[int, str]] = []
    last_b, last_t, i = None, -10**9, 0
    while i < n:
        stable = ~noisy[i]
        ns = int(stable.sum())
        if ns < MIN_STABLE * total:
            i += 1
            continue
        frac = 1.0 if last_b is None else float(((np.abs(blocks[i] - last_b) > CHANGE) & stable).sum()) / ns
        floor_due = i - last_t >= args.floor_s and frac >= 0.01
        if frac >= args.scene or floor_due:
            j = settle(i)
            if j - last_t >= MIN_GAP:
                kept.append((j, "tela" if frac >= args.scene else "mínimo"))
                last_b, last_t = blocks[j], j
            i = j + 1
            continue
        i += 1
    return kept


def extract_frame(video: Path, t: float, path: Path) -> None:
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                    "-q:v", "3", str(path)], check=True)


def frames_stage(args, video: Path, start: float, end: float, work: Path) -> dict:
    path = work / "frames.json"
    key = {"start": start, "end": end, "scene": args.scene, "floor": args.floor_s}
    data = cached(path, key, args.force)
    if data and all(Path(f["file"]).exists() for f in data["frames"]):
        log(f"quadros em cache ({len(data['frames'])})")
        return data
    t0 = time.time()
    log("varrendo o vídeo (1 quadro/s, 160x90)")
    scan_frames = scan(video, start, end)
    picks = pick_times(scan_frames, args)
    log(f"{len(picks)} quadros escolhidos de {len(scan_frames)} s varridos")
    fdir = work / "quadros"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    frames = [{"t": float(start + j), "reason": why, "file": str(fdir / f"f_{hms(start + j).replace(':', '-')}.jpg")}
              for j, why in picks]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda f: extract_frame(video, f["t"], Path(f["file"])), frames))
    data = {"key": key, "frames": frames, "scanned": len(scan_frames), "seconds": round(time.time() - t0)}
    save_json(path, data)
    log(f"quadros: {len(frames)} em {data['seconds']} s")
    return data


# ---------- grades e partes ----------

def chunk_bounds(start: float, end: float, minutes: float) -> list[tuple[float, float]]:
    size = max(60.0, minutes * 60)
    bounds, t = [], start
    while t < end:
        bounds.append((t, min(t + size, end)))
        t += size
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < 120:  # sobra curta entra na parte anterior
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    return bounds


def make_sheets(frames: list[dict], bounds: list[tuple[float, float]], sdir: Path) -> list[list[dict]]:
    from PIL import Image, ImageDraw, ImageFont

    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    per_chunk: list[list[dict]] = []
    for ci, (cs, ce) in enumerate(bounds, 1):
        group = [f for f in frames if cs <= f["t"] < ce or (ci == len(bounds) and f["t"] == ce)]
        sheets = []
        for k in range(0, len(group), 9):
            tiles = group[k:k + 9]
            sheet = Image.new("RGB", (3 * TILE_W, 3 * TILE_H), (32, 32, 32))
            draw = ImageDraw.Draw(sheet)
            for m, f in enumerate(tiles):
                x0, y0 = (m % 3) * TILE_W, (m // 3) * TILE_H
                with Image.open(f["file"]) as im:
                    im = im.convert("RGB")
                    im.thumbnail((TILE_W - 4, TILE_H - 4))
                    sheet.paste(im, (x0 + (TILE_W - im.width) // 2, y0 + (TILE_H - im.height) // 2))
                label = hms(f["t"])
                draw.rectangle([x0, y0, x0 + 130, y0 + 34], fill=(0, 0, 0))
                draw.text((x0 + 8, y0 + 3), label, fill=(255, 220, 0), font=font)
            name = sdir / f"s_{ci:02d}_{k // 9 + 1}.jpg"
            sheet.save(name, quality=85)
            sheets.append({"file": str(name), "from": tiles[0]["t"], "to": tiles[-1]["t"]})
        per_chunk.append(sheets)
    return per_chunk


def write_parts(bounds, segs, frames, sheets, work: Path) -> list[Path]:
    pdir = work / "partes"
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)
    files = []
    for ci, ((cs, ce), shs) in enumerate(zip(bounds, sheets), 1):
        lines = [f"# Parte {ci:02d}: {hms(cs)} a {hms(ce)}", ""]
        lines.append("## Grades (3x3, horário no canto de cada quadro)")
        lines += [f"- `{s['file']}` ({hms(s['from'])} a {hms(s['to'])})" for s in shs] or ["- nenhuma (sem troca de tela)"]
        fr = [f for f in frames if cs <= f["t"] < ce]
        lines += ["", "## Quadros em resolução cheia"]
        lines += [f"- {hms(f['t'])} ({f['reason']}): `{f['file']}`" for f in fr] or ["- nenhum"]
        lines += ["", "## Transcrição", ""]
        lines += [f"[{hms(s['start'])}] {s['text']}" for s in segs if cs <= s["start"] < ce] or ["(sem fala)"]
        path = pdir / f"{ci:02d}_{hms(cs).replace(':', '-')}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files.append(path)
    return files


# ---------- principal ----------

def main() -> None:
    global LOG_PATH, FFMPEG, FFPROBE
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Prepara um vídeo para análise: transcrição, quadros, grades e partes.")
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="pasta de saída (ex.: Downloads/<nome>_analise)")
    ap.add_argument("--vocab", action="append", default=[], help="arquivo de termos (.txt, ou .md com '## Vocabulário')")
    ap.add_argument("--lang", default="pt")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--from", dest="t_from", help="início da janela (hh:mm:ss)")
    ap.add_argument("--to", dest="t_to", help="fim da janela (hh:mm:ss)")
    ap.add_argument("--chunk-min", type=float, default=10, help="minutos por parte")
    ap.add_argument("--floor-s", type=float, default=60, help="um quadro a cada N s se a tela mudou um pouco")
    ap.add_argument("--scene", type=float, default=0.08, help="fração da tela que precisa mudar (menor = mais quadros)")
    ap.add_argument("--whisper", action="store_true", help="transcrever mesmo com legenda embutida")
    ap.add_argument("--force", action="store_true", help="refazer todas as etapas")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        sys.exit(f"vídeo não encontrado: {video}")
    out = Path(args.out).expanduser().resolve()
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    LOG_PATH = work / "prepare.log"
    FFMPEG, FFPROBE = find_tool("ffmpeg"), find_tool("ffprobe")
    t_all = time.time()
    log(f"vídeo: {video}")

    info = probe(video)
    duration = float(info.get("format", {}).get("duration", 0) or 0)
    start = parse_time(args.t_from) or 0.0
    end = min(parse_time(args.t_to) or duration, duration) if duration else (parse_time(args.t_to) or 0)
    if end <= start:
        sys.exit(f"janela inválida: {hms(start)} a {hms(end)} (duração {hms(duration)})")
    created = info.get("format", {}).get("tags", {}).get("creation_time", "")
    vstream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    log(f"duração {hms(duration)}, janela {hms(start)}-{hms(end)}, {vstream.get('width')}x{vstream.get('height')}")

    prompt = read_vocab([SKILL_DIR / "vocab-default.txt"] + [Path(v) for v in args.vocab])
    trans = transcribe(args, video, info, start, end, work, prompt)
    header = [f"# Transcrição: {video.name}",
              f"Duração {hms(duration)}" + (f", janela {hms(start)} a {hms(end)}" if (start, end) != (0.0, duration) else "")
              + (f", gravado em {created[:10]}" if created else "") + ". Horários = posição no vídeo."]
    write_transcript(trans, out, header)

    fr = frames_stage(args, video, start, end, work)
    bounds = chunk_bounds(start, end, args.chunk_min)
    sheets = make_sheets(fr["frames"], bounds, work / "grades")
    parts = write_parts(bounds, trans["segments"], fr["frames"], sheets, work)

    manifest = [
        f"# Manifesto: {video.name}", "",
        f"- Vídeo: `{video}`",
        f"- Duração: {hms(duration)}; janela analisada: {hms(start)} a {hms(end)}",
        f"- Gravado em: {created or 'desconhecido'}; imagem {vstream.get('width')}x{vstream.get('height')}",
        f"- Transcrição: {trans['meta']['source']}, idioma {trans['meta'].get('language')}, "
        f"{len(trans['segments'])} trechos ({trans.get('seconds', 0)} s); `{out / 'transcricao.md'}`",
        f"- Quadros: {len(fr['frames'])} de {fr['scanned']} s varridos ({fr.get('seconds', 0)} s); "
        f"--scene {args.scene}, --floor-s {args.floor_s}",
        f"- Grades: {sum(len(s) for s in sheets)}; partes: {len(parts)} de {args.chunk_min:g} min",
        f"- Vocabulário no prompt: {prompt or '(nenhum)'}",
        f"- Tempo total desta execução: {round(time.time() - t_all)} s", "",
        "## Partes", ""]
    manifest += [f"{i}. `{p}` ({len(s)} grades)" for i, (p, s) in enumerate(zip(parts, sheets), 1)]
    (work / "MANIFEST.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    log(f"pronto: {work / 'MANIFEST.md'}")


if __name__ == "__main__":
    main()
