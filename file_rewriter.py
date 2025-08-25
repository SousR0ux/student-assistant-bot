# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Callable, Awaitable, TYPE_CHECKING, Any, TypeAlias
import io

# ---- optional runtime deps ----
try:
    import docx  # type: ignore
    from docx.enum.text import WD_COLOR_INDEX  # type: ignore
except Exception:
    docx = None  # type: ignore
    WD_COLOR_INDEX = None  # type: ignore

# ---- types for static checking (no runtime import errors) ----
if TYPE_CHECKING:
    from docx.document import Document as DocxDocument  # type: ignore
    from docx.text.run import Run  # type: ignore
else:
    DocxDocument: TypeAlias = Any
    Run: TypeAlias = Any


@dataclass
class HighlightedPart:
    original_text: str
    part_index: int
    run_indices: List[Tuple[int, int]]  # (paragraph_index, run_index)


@dataclass
class RewrittenPart:
    rewritten_text: str
    original_part: HighlightedPart


def process_docx_for_rewrite(file_bytes: bytes) -> tuple[DocxDocument, List[HighlightedPart]]:
    """Собираем подряд идущие выделенные жёлтым фрагменты в «части»."""
    if docx is None:
        raise RuntimeError("Библиотека python-docx не установлена. Выполните: pip install python-docx")

    doc = docx.Document(io.BytesIO(file_bytes))
    parts: List[HighlightedPart] = []

    cur_txt: str = ""
    cur_runs: List[Tuple[int, int]] = []
    part_id = 0
    YELLOW = getattr(WD_COLOR_INDEX, "YELLOW", 7)

    for p_i, p in enumerate(doc.paragraphs):
        for r_i, run in enumerate(p.runs):
            if getattr(run.font, "highlight_color", None) == YELLOW:
                cur_txt += run.text
                cur_runs.append((p_i, r_i))
            else:
                if cur_txt:
                    parts.append(HighlightedPart(cur_txt, part_id, cur_runs))
                    part_id += 1
                    cur_txt, cur_runs = "", []

    if cur_txt:
        parts.append(HighlightedPart(cur_txt, part_id, cur_runs))

    return doc, parts


def _chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _build_prompt(chunk: str, tone: str, target_uniqueness: str) -> str:
    uniq = f"Цель по уникальности (антиплагиат): {target_uniqueness}.\n" if target_uniqueness else ""
    return (
        "Ты — академический редактор. Перепиши фрагмент так, чтобы:\n"
        f"- стиль: {tone};\n"
        "- сохранялась исходная мысль и структура;\n"
        "- не добавлять вступления/выводы от себя;\n"
        f"{uniq}\n"
        "ТЕКСТ:\n\"\"\"\n" + chunk + "\n\"\"\"\n"
        "ВЫВОД: (только перефразированный текст)"
    )


async def rewrite_highlighted_parts_async(
    parts: List[HighlightedPart],
    rewrite_fn: Callable[[str], Awaitable[str]],
    tone: str,
    target_uniqueness: str
) -> List[RewrittenPart]:
    out: List[RewrittenPart] = []
    for part in parts:
        if not part.original_text.strip():
            continue
        acc: List[str] = []
        for chunk in _chunk_text(part.original_text):
            prompt = _build_prompt(chunk, tone, target_uniqueness)
            rewritten = await rewrite_fn(prompt)
            acc.append(rewritten.strip().strip("`"))
        out.append(RewrittenPart(" ".join(acc).strip(), part))
    return out


def _is_bold(run: Run) -> bool:
    """True, если на run явно включено жирное начертание."""
    b1 = getattr(run, "bold", None)
    b2 = getattr(getattr(run, "font", None), "bold", None)
    return bool(b1 is True or b2 is True)


def _pick_base_run(doc: DocxDocument, indices: List[Tuple[int, int]]) -> Run:
    """
    Выбираем run, чей стиль берём за «базовый».
    Предпочитаем первый НЕ жирный; если все жирные — берём самый первый.
    """
    p0, r0 = indices[0]
    first: Run = doc.paragraphs[p0].runs[r0]  # type: ignore[assignment]
    for p_i, r_i in indices:
        candidate: Run = doc.paragraphs[p_i].runs[r_i]  # type: ignore[assignment]
        if not _is_bold(candidate):
            return candidate
    return first


def _apply_style_from_to(src: Run, dst: Run) -> None:
    """Копируем ключевые атрибуты (в т.ч. выключаем подсветку)."""
    try:
        dst.bold = getattr(src, "bold", None)
        dst.italic = getattr(src, "italic", None)
        dst.underline = getattr(src, "underline", None)
        dst.style = getattr(src, "style", None)

        sfont = getattr(src, "font", None)
        dfont = getattr(dst, "font", None)
        if sfont and dfont:
            if getattr(sfont, "name", None):
                dfont.name = sfont.name
            if getattr(sfont, "size", None):
                dfont.size = sfont.size
            dfont.highlight_color = None

        if dst.bold is None and _is_bold(dst):
            dst.bold = False
    except Exception:
        pass


def build_final_docx(original_doc: DocxDocument, rewritten_parts: List[RewrittenPart]) -> bytes:
    """
    Вставляем переписанные тексты в первый run каждой выделенной группы,
    копируя «нормальный» стиль и снимая подсветку. Остальные run'ы очищаем.
    """
    parts_map = {p.original_part.part_index: p for p in rewritten_parts}

    for part in parts_map.values():
        run_idxs = part.original_part.run_indices
        if not run_idxs:
            continue

        base_run = _pick_base_run(original_doc, run_idxs)

        tgt_p, tgt_r = run_idxs[0]
        target_run: Run = original_doc.paragraphs[tgt_p].runs[tgt_r]  # type: ignore[assignment]

        for p_i, r_i in run_idxs:
            rr: Run = original_doc.paragraphs[p_i].runs[r_i]  # type: ignore[assignment]
            try:
                rr.font.highlight_color = None
            except Exception:
                pass

        _apply_style_from_to(base_run, target_run)
        target_run.text = part.rewritten_text

        for p_i, r_i in run_idxs[1:]:
            original_doc.paragraphs[p_i].runs[r_i].text = ""

    bio = io.BytesIO()
    original_doc.save(bio)
    bio.seek(0)
    return bio.getvalue()
