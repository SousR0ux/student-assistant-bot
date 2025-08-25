# -*- coding: utf-8 -*-
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Tuple, Callable, Awaitable, TYPE_CHECKING, Any

# ----- Безопасные типы для подсказок (не ломают Pylance) -----
if TYPE_CHECKING:
    from docx.document import Document as DocxDocument
    from docx.text.run import Run
else:
    DocxDocument = Any
    Run = Any

# ----- python-docx (может отсутствовать в окружении) -----
try:
    import docx  # type: ignore
    from docx.enum.text import WD_COLOR_INDEX  # type: ignore
except ImportError:  # библиотека не установлена
    docx = None            # type: ignore
    WD_COLOR_INDEX = None  # type: ignore


# ===================== DATA CLASSES =====================

@dataclass
class HighlightedPart:
    """
    Один логический выделенный фрагмент в документе.
    run_indices: список кортежей (paragraph_index, run_index) для всех run'ов,
                 входящих в этот фрагмент.
    """
    original_text: str
    part_index: int
    run_indices: List[Tuple[int, int]]


@dataclass
class RewrittenPart:
    """Результат рерайта конкретного HighlightedPart."""
    rewritten_text: str
    original_part: HighlightedPart


# ===================== ВНУТРЕННИЕ ХЕЛПЕРЫ =====================

def _require_docx() -> None:
    if docx is None:  # pragma: no cover
        raise RuntimeError(
            "Библиотека python-docx не установлена. Выполните: pip install python-docx"
        )


def _is_yellow_highlight(run: Run) -> bool:
    """
    Проверка, что run выделен жёлтой заливкой.
    Работает и с Enum, и с числом, и со строковым представлением.
    """
    try:
        h = run.font.highlight_color
        if h is None:
            return False

        # Если есть Enum WD_COLOR_INDEX
        if WD_COLOR_INDEX is not None:
            try:
                return h == WD_COLOR_INDEX.YELLOW  # type: ignore[attr-defined]
            except Exception:
                pass

        # Иногда приходит int/str/enum-like
        if hasattr(h, "value"):
            return int(h.value) == 7
        if isinstance(h, int):
            return h == 7
        return str(h).upper().endswith("YELLOW")
    except Exception:
        return False


def _chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _build_prompt(chunk: str, tone: str, target_uniqueness: str) -> str:
    uniq = f"Цель по уникальности (антиплагиат): {target_uniqueness}.\n" if target_uniqueness else ""
    return (
        "Ты — академический редактор. Перепиши фрагмент так, чтобы:\n"
        f"- стиль: {tone};\n"
        "- сохранялась исходная мысль и структура;\n"
        "- не добавлять вступления/выводы от себя и лишние пояснения;\n"
        f"{uniq}\n"
        "ТЕКСТ ДЛЯ РЕРАЙТА:\n"
        f"\"\"\"\n{chunk}\n\"\"\"\n"
        "ВЫВОД: только перефразированный текст, без комментариев."
    )


def _replace_text_preserving_runs(doc: DocxDocument,
                                  indices: List[Tuple[int, int]],
                                  new_text: str) -> None:
    """
    Раскладывает new_text по тем же run'ам, что были в исходнике,
    не меняя их форматирования. У всех run'ов снимается подсветка.
    """
    if not indices:
        return

    runs: List[Run] = [doc.paragraphs[p].runs[r] for p, r in indices]

    # Снять подсветку
    for rn in runs:
        try:
            rn.font.highlight_color = None
        except Exception:
            pass

    # Длины исходных run'ов — для пропорционального распределения
    orig_lens = [len(getattr(rn, "text", "") or "") for rn in runs]
    total = sum(orig_lens)

    # Если все были пустыми — кладём всё в первый run
    if total == 0:
        runs[0].text = new_text
        for rn in runs[1:]:
            rn.text = ""
        return

    tgt_len = len(new_text)
    # Пропорциональная раздача символов
    alloc: List[int] = []
    acc = 0
    for i, L in enumerate(orig_lens):
        if i == len(orig_lens) - 1:
            n = tgt_len - acc
        else:
            n = round(tgt_len * (L / total))
            n = max(0, min(n, tgt_len - acc))
        alloc.append(n)
        acc += n

    pos = 0
    for rn, n in zip(runs, alloc):
        rn.text = new_text[pos : pos + n]
        pos += n
    if pos < tgt_len:  # хвост в последний run
        runs[-1].text += new_text[pos:]


# ===================== ПУБЛИЧНЫЕ ФУНКЦИИ =====================

def process_docx_for_rewrite(file_bytes: bytes) -> Tuple[DocxDocument, List[HighlightedPart]]:
    """
    Читает DOCX, собирает непрерывные (по run'ам) фрагменты с жёлтой заливкой.
    Возвращает открытый документ и список частей для рерайта.
    """
    _require_docx()
    doc: DocxDocument = docx.Document(io.BytesIO(file_bytes))  # type: ignore

    parts: List[HighlightedPart] = []
    cur_text: str = ""
    cur_runs: List[Tuple[int, int]] = []
    counter = 0

    for p_idx, p in enumerate(doc.paragraphs):
        for r_idx, run in enumerate(p.runs):
            if _is_yellow_highlight(run):
                cur_text += run.text
                cur_runs.append((p_idx, r_idx))
            else:
                if cur_runs:
                    parts.append(
                        HighlightedPart(
                            original_text=cur_text,
                            part_index=counter,
                            run_indices=cur_runs[:],
                        )
                    )
                    counter += 1
                    cur_text = ""
                    cur_runs = []

        # Разрыв абзаца — тоже конец непрерывного фрагмента
        if cur_runs:
            parts.append(
                HighlightedPart(
                    original_text=cur_text,
                    part_index=counter,
                    run_indices=cur_runs[:],
                )
            )
            counter += 1
            cur_text = ""
            cur_runs = []

    # На всякий случай, если закончили на подсветке
    if cur_runs:
        parts.append(
            HighlightedPart(
                original_text=cur_text,
                part_index=counter,
                run_indices=cur_runs[:],
            )
        )

    return doc, parts


async def rewrite_highlighted_parts_async(
    parts: List[HighlightedPart],
    rewrite_fn: Callable[[str], Awaitable[str]],
    tone: str = "официальный",
    target_uniqueness: str = ""
) -> List[RewrittenPart]:
    """
    Асинхронно переписывает каждый выделенный фрагмент через ваш колбэк rewrite_fn.
    Колбэк принимает СФОРМИРОВАННЫЙ prompt и возвращает текст ответа ИИ.
    """
    results: List[RewrittenPart] = []

    for part in parts:
        src = (part.original_text or "").strip()
        if not src:
            continue

        chunks = _chunk_text(src)
        rewritten_acc: List[str] = []

        for chunk in chunks:
            prompt = _build_prompt(chunk, tone, target_uniqueness)
            ai_out = await rewrite_fn(prompt)
            rewritten_acc.append((ai_out or "").strip().strip("`"))

        final_text = " ".join(x for x in rewritten_acc if x).strip()
        results.append(RewrittenPart(rewritten_text=final_text, original_part=part))

    return results


def build_final_docx(original_doc: DocxDocument,
                     rewritten_parts: List[RewrittenPart]) -> bytes:
    """
    Вставляет переписанные тексты в исходный документ,
    сохраняя форматирование (абзацы, стили run'ов, отступы).
    Снимается только подсветка.
    """
    for part in rewritten_parts:
        _replace_text_preserving_runs(
            original_doc,
            part.original_part.run_indices,
            part.rewritten_text
        )

    bio = io.BytesIO()
    original_doc.save(bio)
    bio.seek(0)
    return bio.getvalue()
