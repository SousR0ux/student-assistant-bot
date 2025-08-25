# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import List, Tuple, Callable, Awaitable, TYPE_CHECKING, Any

# ----- Безопасные типы для подсказок (не ломают Pylance) -----
if TYPE_CHECKING:
    from docx.document import Document as DocxDocument
    from docx.text.run import Run
else:
    DocxDocument = Any
    Run = Any

# ----- python-docx -----
try:
    import docx  # type: ignore
    from docx.enum.text import WD_COLOR_INDEX  # type: ignore
except ImportError:
    docx = None            # type: ignore
    WD_COLOR_INDEX = None  # type: ignore


# ===================== DATA CLASSES =====================

@dataclass
class HighlightedPart:
    original_text: str
    part_index: int
    run_indices: List[Tuple[int, int]]  # (paragraph_idx, run_idx)

@dataclass
class RewrittenPart:
    rewritten_text: str
    original_part: HighlightedPart


# ===================== INTERNAL HELPERS =====================

def _require_docx() -> None:
    if docx is None:  # pragma: no cover
        raise RuntimeError("Библиотека python-docx не установлена. Выполните: pip install python-docx")

def _is_yellow_highlight(run: Run) -> bool:
    try:
        h = run.font.highlight_color
        if h is None:
            return False
        if WD_COLOR_INDEX is not None:
            try:
                return h == WD_COLOR_INDEX.YELLOW  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(h, "value"):
            return int(h.value) == 7
        if isinstance(h, int):
            return h == 7
        return str(h).upper().endswith("YELLOW")
    except Exception:
        return False

def _chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    return [t[i:i + max_chars] for i in range(0, len(t), max_chars)]

def _clean_output(s: str) -> str:
    """Фильтр на случай, если модель прислала разметку/префиксы."""
    s = (s or "").strip()
    # часто встречающееся «обрамление»
    if s.startswith("```") and s.endswith("```"):
        s = s[3:-3].strip()
    s = s.strip("`").strip()
    # вырезаем типичные маркёры
    for bad in ("**TL;DR**", "TL;DR", "**Ответ:**", "Ответ:", "Вывод:", "Переписанный текст:", "Рерайт:"):
        s = s.replace(bad, "")
    return s.strip()

def _build_analyze_prompt(chunk: str, tone: str) -> str:
    return (
        "Ты — опытный научный редактор.\n"
        "ШАГ 1 (внутренний анализ, для себя): извлеки ключевые тезисы и термины из фрагмента.\n"
        "Верни строго JSON без комментариев в формате:\n"
        "{\n"
        '  "key_points": ["тезис1", "тезис2", ...],\n'
        '  "terms": ["термин1", "термин2", ...]\n'
        "}\n\n"
        f"Стиль целевого изложения: {tone}.\n"
        "Текст:\n"
        f"\"\"\"\n{chunk}\n\"\"\""
    )

def _build_rewrite_prompt(chunk: str, analysis_json: str, tone: str, target_uniqueness: str) -> str:
    uniq = f"Цель по уникальности (антиплагиат): {target_uniqueness}.\n" if target_uniqueness else ""
    return (
        "Ты — академический редактор. Сначала молча используй анализ ниже для понимания смысла, "
        "но НИКОГДА его не цитируй и не упоминай. Затем перепиши исходный текст.\n\n"
        "АНАЛИЗ (JSON, только для внутреннего использования):\n"
        f"{analysis_json}\n\n"
        "ТРЕБОВАНИЯ К РЕРАЙТУ:\n"
        f"- стиль: {tone};\n"
        "- сохраняй исходные факты, числа, определения, цитаты в «…» не перефразируй;\n"
        "- не добавляй информации, которой нет в исходнике; не пиши вводных, выводов и пояснений;\n"
        "- сохраняй логику и последовательность; избегай штампов и воды;\n"
        "- длина результата ~90–110% от исходного фрагмента;\n"
        "- сохраняй перечисления и структуру как в исходнике (если был абзац — остаётся абзац);\n"
        f"{uniq}"
        "ОТВЕТ:\n"
        "Верни ТОЛЬКО перефразированный текст без маркировок, заголовков, списков или Markdown, "
        "без слов-предисловий (например, «Переписанный текст:»)."
        "\n\n"
        "ИСХОДНЫЙ ТЕКСТ:\n"
        f"\"\"\"\n{chunk}\n\"\"\""
    )

def _build_single_pass_prompt(chunk: str, tone: str, target_uniqueness: str) -> str:
    uniq = f"Цель по уникальности (антиплагиат): {target_uniqueness}.\n" if target_uniqueness else ""
    return (
        "Ты — академический редактор. Сначала молча проанализируй смысл, затем перепиши.\n"
        "Не добавляй ничего сверх исходника, не пиши вступлений/выводов и комментариев.\n"
        f"- стиль: {tone};\n"
        "- факты/числа/термины сохраняй; цитаты в «…» не перефразируй;\n"
        "- длина ~90–110% изначального;\n"
        "- сохрани порядок и структуру.\n"
        f"{uniq}"
        "ОТВЕТ: только переписанный текст без маркировок и Markdown.\n\n"
        "Текст для рерайта:\n"
        f"\"\"\"\n{chunk}\n\"\"\""
    )

def _replace_text_preserving_runs(doc: DocxDocument,
                                  indices: List[Tuple[int, int]],
                                  new_text: str) -> None:
    """Кладём новый текст в те же runs, не трогая формат; подсветку снимаем."""
    if not indices:
        return
    runs: List[Run] = [doc.paragraphs[p].runs[r] for p, r in indices]
    for rn in runs:
        try:
            rn.font.highlight_color = None
        except Exception:
            pass

    orig_lens = [len(getattr(rn, "text", "") or "") for rn in runs]
    total = sum(orig_lens)
    if total == 0:
        runs[0].text = new_text
        for rn in runs[1:]:
            rn.text = ""
        return

    tgt_len = len(new_text)
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
        rn.text = new_text[pos:pos+n]
        pos += n
    if pos < tgt_len:
        runs[-1].text += new_text[pos:]


# ===================== PUBLIC API =====================

def process_docx_for_rewrite(file_bytes: bytes) -> Tuple[DocxDocument, List[HighlightedPart]]:
    """Собирает непрерывные жёлтые фрагменты. Возвращает документ и список частей."""
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
                    parts.append(HighlightedPart(cur_text, counter, cur_runs[:]))
                    counter += 1
                    cur_text, cur_runs = "", []

        if cur_runs:  # разрыв абзаца = конец фрагмента
            parts.append(HighlightedPart(cur_text, counter, cur_runs[:]))
            counter += 1
            cur_text, cur_runs = "", []

    if cur_runs:  # на всякий
        parts.append(HighlightedPart(cur_text, counter, cur_runs[:]))

    return doc, parts


async def rewrite_highlighted_parts_async(
    parts: List[HighlightedPart],
    rewrite_fn: Callable[[str], Awaitable[str]],
    tone: str = "официальный",
    target_uniqueness: str = "",
    deep_analyze: bool = True,
) -> List[RewrittenPart]:
    """
    Переписывает каждый выделенный фрагмент.
    - rewrite_fn(prompt) -> str  — ваш вызов ИИ (например, call_gemini).
    - deep_analyze=True   — двухшаговый режим: анализ JSON -> рерайт.
    """
    results: List[RewrittenPart] = []

    for part in parts:
        src = (part.original_text or "").strip()
        if not src:
            continue

        chunks = _chunk_text(src)
        rewritten_acc: List[str] = []

        for chunk in chunks:
            if deep_analyze:
                # Шаг 1: анализ
                a_prompt = _build_analyze_prompt(chunk, tone)
                analysis = await rewrite_fn(a_prompt)
                analysis = _clean_output(analysis)

                # Это должен быть JSON — но на всякий подстрахуемся
                try:
                    json.loads(analysis)
                except Exception:
                    # если пришёл не JSON, склеим безопасную болванку
                    analysis = json.dumps({"key_points": [], "terms": []}, ensure_ascii=False)

                # Шаг 2: рерайт по анализу
                r_prompt = _build_rewrite_prompt(chunk, analysis, tone, target_uniqueness)
                out = await rewrite_fn(r_prompt)
            else:
                # Однопроходный рерайт
                r_prompt = _build_single_pass_prompt(chunk, tone, target_uniqueness)
                out = await rewrite_fn(r_prompt)

            rewritten_acc.append(_clean_output(out))

        final_text = " ".join(x for x in rewritten_acc if x).strip()
        results.append(RewrittenPart(rewritten_text=final_text, original_part=part))

    return results


def build_final_docx(original_doc: DocxDocument,
                     rewritten_parts: List[RewrittenPart]) -> bytes:
    """Вставляет новый текст в те же runs; форматирование сохраняется, подсветка снимается."""
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
