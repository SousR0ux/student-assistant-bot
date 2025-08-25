# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import List, Tuple, Callable, Awaitable, TYPE_CHECKING, Any, Dict

# --- Опциональный импорт python-docx (чтобы не падать на типизации в редакторе) ---
try:
    import docx  # type: ignore
    from docx.enum.text import WD_COLOR_INDEX  # type: ignore
except Exception:  # библиотека может быть не установлена в среде проверки типов
    docx = None            # type: ignore
    WD_COLOR_INDEX = None  # type: ignore

if TYPE_CHECKING:
    from docx.document import Document as DocxDocument  # type: ignore
    from docx.text.run import Run  # type: ignore
else:
    DocxDocument = Any  # безопасная подстановка, чтобы Pylance не ругался
    Run = Any


# =========================
#        МОДЕЛИ
# =========================

@dataclass
class HighlightedPart:
    """
    Один выделенный (подсвеченный) пользователем фрагмент документа.
    aggregated_text  — исходный текст фрагмента (в порядке прохождения Run-ов).
    run_indices      — список пар (p_idx, r_idx) того, в каких параграфах/раннах лежал текст.
    """
    aggregated_text: str
    part_index: int
    run_indices: List[Tuple[int, int]]


@dataclass
class RewrittenPart:
    """Результат рерайта для одного HighlightedPart."""
    rewritten_text: str
    original_part: HighlightedPart


# =========================
#     ВСПОМОГАТЕЛЬНОЕ
# =========================

def _is_yellow_highlight(run: Run) -> bool:
    """Проверка, что у ранна жёлтая подсветка. Защита от отсутствия атрибута."""
    try:
        return getattr(run.font, "highlight_color", None) == getattr(WD_COLOR_INDEX, "YELLOW", None)
    except Exception:
        return False


def _chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    """Ровняем по кускам, чтобы не превышать лимиты модели."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _build_prompt(chunk: str, tone: str, target_uniqueness: str, deep_analyze: bool = True) -> str:
    """Жёсткий промпт: анализ смысла + запрет на подмену категорий и фактов."""
    uniq_line = f"Цель по уникальности (антиплагиат): {target_uniqueness}.\n" if target_uniqueness else ""
    analyze = (
        "<self_reflection>\n"
        "1) Подумай, какие требования к идеальному академическому перефразированию важны.\n"
        "2) Внутренне оцени результат по 6 критериям: точность смысла, сохранение категорий/терминов, "
        "отсутствие новых фактов, структурное соответствие (1↔1 по предложениям), связность, естественность стиля.\n"
        "3) Если какой-либо критерий не выполнен, перепиши свой ответ до соответствия.\n"
        "</self_reflection>\n\n"
        if deep_analyze else ""
    )
    return (
        "Ты — опытный академический редактор. Твоя задача — осторожно перефразировать текст "
        "без искажения смысла и без добавления фактов.\n\n"
        "<rules>\n"
        "- Переписывай СТРОГО предложение-за-предложением: на каждое входное предложение — одно переписанное.\n"
        "- Не меняй категории и заголовки: «Актуальность», «Цель курсовой работы», «Объект исследования», "
        "«Предмет исследования», «Задачи» и т. п. НЕЛЬЗЯ заменять «цель» на «предмет» и наоборот.\n"
        "- Сохраняй имена, цифры, хронологию, термины. Ничего не выдумывай и не добавляй.\n"
        f"- Стиль: {tone}. Объём каждого предложения ~0.8–1.2 от исходного.\n"
        "- Не пиши никаких комментариев, пояснений, списков правил и т. д. Выводи только перефразированный текст.\n"
        f"- {uniq_line}"
        "</rules>\n\n"
        f"{analyze}"
        "ТЕКСТ ДЛЯ РЕРАЙТА:\n"
        f"\"\"\"\n{chunk}\n\"\"\"\n\n"
        "ВЫВОД: только перефразированный текст без меток и комментариев."
    )


def _clean_model_text(s: str) -> str:
    """Убираем маркдауны/обёртки, лишние пробелы."""
    s = s.strip()
    # удаляем возможные бэктики/код-блоки
    s = s.strip("`").strip()
    # нормализуем множественные пробелы
    s = re.sub(r"[ \t]+", " ", s)
    # убираем пробелы перед знаками препинания
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s.strip()


# =========================
#   ОСНОВНЫЕ ФУНКЦИИ
# =========================

def process_docx_for_rewrite(file_bytes: bytes) -> Tuple[DocxDocument, List[HighlightedPart]]:
    """
    Считывает .docx, находит непрерывные последовательности Run-ов с жёлтой подсветкой
    и возвращает документ + список выделенных фрагментов.
    """
    if docx is None:
        raise RuntimeError("Библиотека python-docx не установлена. Установите: pip install python-docx")

    doc: DocxDocument = docx.Document(io.BytesIO(file_bytes))  # type: ignore
    parts: List[HighlightedPart] = []

    current_runs: List[Tuple[int, int]] = []
    current_text: List[str] = []
    part_counter = 0

    for p_idx, p in enumerate(doc.paragraphs):
        for r_idx, run in enumerate(p.runs):
            if _is_yellow_highlight(run):
                current_runs.append((p_idx, r_idx))
                current_text.append(run.text or "")
            else:
                if current_runs:
                    parts.append(
                        HighlightedPart(
                            aggregated_text="".join(current_text),
                            part_index=part_counter,
                            run_indices=current_runs[:],
                        )
                    )
                    part_counter += 1
                    current_runs.clear()
                    current_text.clear()

    # Хвост
    if current_runs:
        parts.append(
            HighlightedPart(
                aggregated_text="".join(current_text),
                part_index=part_counter,
                run_indices=current_runs[:],
            )
        )

    return doc, parts


async def rewrite_highlighted_parts_async(
    parts: List[HighlightedPart],
    rewrite_fn: Callable[[str], Awaitable[str]],
    tone: str = "официальный",
    target_uniqueness: str = "",
    deep_analyze: bool = True,
) -> List[RewrittenPart]:
    """
    Делит исходные тексты по кускам (на случай больших фрагментов),
    вызывает переданную функция `rewrite_fn(prompt)`, склеивает результат.
    """
    out: List[RewrittenPart] = []

    for part in parts:
        src = part.aggregated_text.strip()
        if not src:
            continue

        chunks = _chunk_text(src)
        rewritten_total: List[str] = []

        for chunk in chunks:
            prompt = _build_prompt(chunk, tone, target_uniqueness, deep_analyze=deep_analyze)
            resp = await rewrite_fn(prompt)
            rewritten_total.append(_clean_model_text(resp))

        out.append(
            RewrittenPart(
                rewritten_text=" ".join(rewritten_total).strip(),
                original_part=part,
            )
        )

    return out


def _distribute_text_by_runs(text: str, runs: List[Run]) -> List[str]:
    """
    Возвращает список частей `text`, распределённых по раннам пропорционально
    их первоначальной длине. Это позволяет сохранить формат конкретных раннов
    (жирность/курсив/капс и т. п.), не делая весь блок одним стилем.
    """
    if not runs:
        return []

    # исходные длины
    lengths = [len(r.text or "") for r in runs]
    total_len = sum(lengths)

    # если все длины нулевые — просто сверху вниз
    if total_len == 0:
        parts = []
        left = text
        for i in range(len(runs) - 1):
            parts.append("")  # ничего не было — ничего не кладём
        parts.append(left)
        return parts

    target_total = len(text)
    assigned: List[int] = []
    taken = 0
    for i, L in enumerate(lengths):
        if i == len(lengths) - 1:
            size = target_total - taken
        else:
            # доля символов
            share = (L / total_len) * target_total
            size = max(0, int(round(share)))
            # не превышаем остаток
            size = min(size, target_total - taken)
        assigned.append(size)
        taken += size

    # Срезы текста
    res: List[str] = []
    pos = 0
    for size in assigned:
        res.append(text[pos:pos + size])
        pos += size
    return res


def build_final_docx(original_doc: DocxDocument, rewritten_parts: List[RewrittenPart]) -> bytes:
    """
    Вставляет переписанный текст на место подсвеченных участков,
    снимает подсветку, при этом:
    - количество раннов и их стили сохраняются;
    - жирность/курсив/капс/шрифт/интервалы/отступы/стили абзацев НЕ меняются.
    """
    # Быстрый доступ по номеру фрагмента
    parts_map: Dict[int, RewrittenPart] = {
        p.original_part.part_index: p for p in rewritten_parts
    }

    for part_idx in sorted(parts_map.keys()):
        part = parts_map[part_idx]
        indices = part.original_part.run_indices

        # Собираем реальные ранны в исходном порядке
        runs: List[Run] = []
        for p_idx, r_idx in indices:
            try:
                runs.append(original_doc.paragraphs[p_idx].runs[r_idx])  # type: ignore
            except Exception:
                # если вдруг индекс «уехал», пропускаем
                continue

        if not runs:
            continue

        # Распределяем новый текст по этим же раннам
        slices = _distribute_text_by_runs(part.rewritten_text, runs)

        for run, new_text in zip(runs, slices):
            # подчищаем подсветку и меняем только текст
            try:
                if getattr(run.font, "highlight_color", None) is not None:
                    run.font.highlight_color = None
            except Exception:
                pass
            run.text = new_text

    bio = io.BytesIO()
    original_doc.save(bio)  # type: ignore
    bio.seek(0)
    return bio.getvalue()
