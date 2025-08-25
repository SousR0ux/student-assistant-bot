# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Callable, Awaitable, TYPE_CHECKING, Any
import io

# --- необязательная зависимость ---
try:
    import docx
    from docx.enum.text import WD_COLOR_INDEX
except Exception:
    docx = None
    WD_COLOR_INDEX = None  # type: ignore[assignment]

# Подсказываем типы анализатору, но не требуем пакет в рантайме
if TYPE_CHECKING:
    from docx.document import Document as DocxDocument
else:
    DocxDocument = Any  # безопасная заглушка для рантайма


# ======================= МОДЕЛИ =======================

@dataclass
class HighlightedPart:
    original_text: str
    part_index: int
    run_indices: List[Tuple[int, int]]  # (paragraph_index, run_index)

@dataclass
class RewrittenPart:
    rewritten_text: str
    original_part: HighlightedPart


# ==================== ОСНОВНЫЕ ФУНКЦИИ ====================

def process_docx_for_rewrite(file_bytes: bytes) -> Tuple[DocxDocument, List[HighlightedPart]]:
    """
    Ищет подряд идущие runs, подсвеченные жёлтым, и собирает их в блоки.
    Возвращает открытый Document и список найденных блоков.
    """
    if docx is None:
        raise RuntimeError("Библиотека python-docx не установлена. Выполните: pip install python-docx")

    document = docx.Document(io.BytesIO(file_bytes))
    highlighted_parts: List[HighlightedPart] = []

    current_text: str = ""
    current_runs: List[Tuple[int, int]] = []
    part_counter = 0

    for p_idx, paragraph in enumerate(document.paragraphs):
        for r_idx, run in enumerate(paragraph.runs):
            is_yellow = (run.font.highlight_color == WD_COLOR_INDEX.YELLOW)  # type: ignore[union-attr]
            if is_yellow:
                current_text += run.text
                current_runs.append((p_idx, r_idx))
            else:
                # Закрываем предыдущий блок, если он был
                if current_text:
                    highlighted_parts.append(
                        HighlightedPart(
                            original_text=current_text,
                            part_index=part_counter,
                            run_indices=current_runs
                        )
                    )
                    part_counter += 1
                    current_text = ""
                    current_runs = []

    # Хвост в конце документа
    if current_text:
        highlighted_parts.append(
            HighlightedPart(
                original_text=current_text,
                part_index=part_counter,
                run_indices=current_runs
            )
        )

    return document, highlighted_parts


def _chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _build_prompt(chunk: str, tone: str, target_uniqueness: str) -> str:
    uniq_line = f"Цель по уникальности (антиплагиат): {target_uniqueness}.\n" if target_uniqueness else ""
    return (
        "Ты — академический редактор. Перепиши фрагмент курсовой работы так, чтобы:\n"
        f"- стиль: {tone};\n"
        "- сохранялась исходная мысль и структура;\n"
        "- не добавлять вступления/выводы от себя;\n"
        f"{uniq_line}\n"
        "ТЕКСТ ДЛЯ РЕРАЙТА:\n"
        f"\"\"\"\n{chunk}\n\"\"\"\n"
        "ВЫВОД:\n(только перефразированный текст без пояснений)"
    )


async def rewrite_highlighted_parts_async(
    parts: List[HighlightedPart],
    rewrite_fn: Callable[[str], Awaitable[str]],
    tone: str,
    target_uniqueness: str
) -> List[RewrittenPart]:
    """
    Для каждого выделенного блока вызывает `rewrite_fn(prompt)` (например, обёртка над Gemini)
    и собирает результат. Текст режется на куски по 8k символов.
    """
    results: List[RewrittenPart] = []

    for part in parts:
        if not part.original_text.strip():
            continue

        chunks = _chunk_text(part.original_text)
        rewritten_total = []

        for chunk in chunks:
            prompt = _build_prompt(chunk, tone, target_uniqueness)
            out = await rewrite_fn(prompt)
            rewritten_total.append(out.strip().strip("`"))

        results.append(
            RewrittenPart(
                rewritten_text=(" ".join(rewritten_total)).strip(),
                original_part=part
            )
        )

    return results


def build_final_docx(original_doc: DocxDocument, rewritten_parts: List[RewrittenPart]) -> bytes:
    """
    Заменяет текст в первых run'ах каждого выделенного блока на перефразированный,
    остальные run'ы этого блока очищает. Подсветка снимается.
    """
    parts_map = {p.original_part.part_index: p for p in rewritten_parts}

    for part in parts_map.values():
        for i, (p_idx, r_idx) in enumerate(part.original_part.run_indices):
            run_to_edit = original_doc.paragraphs[p_idx].runs[r_idx]
            if i == 0:
                run_to_edit.text = part.rewritten_text
                # type: ignore[union-attr] — в рантайме это всегда объект docx
                run_to_edit.font.highlight_color = None  # снимаем подсветку
            else:
                run_to_edit.text = ""

    buf = io.BytesIO()
    original_doc.save(buf)
    buf.seek(0)
    return buf.getvalue()
