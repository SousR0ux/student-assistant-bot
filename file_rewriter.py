# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import io
import re
from docx.shared import Pt

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    import docx
    from docx import Document as DocxDocument
except Exception:
    docx = None
    DocxDocument = None

@dataclass
class Section:
    name: str
    text: str
    pages: Optional[List[int]] = None

@dataclass
class ParsedDocument:
    filetype: str
    page_count: int
    full_text: str
    sections: Dict[str, Section]
    pages_text: Optional[List[str]] = None

def _detect_filetype(filename: str, file_bytes: bytes) -> str:
    low = (filename or "").lower()
    if low.endswith(".pdf"): return "pdf"
    if low.endswith(".docx"): return "docx"
    if low.endswith(".txt"): return "txt"
    if file_bytes[:5] == b"%PDF-": return "pdf"
    if file_bytes[:2] == b"PK": return "docx"
    return "txt"

_SECTION_PATTERNS = {
    "toc":        r"(?:содержан\w+|оглавлен\w+)",
    "conclusion": r"(?:заключен\w+|вывод\w+)",
    "references": r"(?:список\s+литератур\w+|литература|источники)",
}
_CHAPTER_HINT = r"(?:^|\n)\s*(?:введен\w+|глава|раздел)\s*\d*\s*[).:-]?"

def _find_indices_by_keywords(text: str) -> Dict[str, int]:
    low = text.lower()
    pos = {}
    for k, pat in _SECTION_PATTERNS.items():
        m = re.search(pat, low, flags=re.IGNORECASE)
        if m: pos[k] = m.start()
    return pos

def _split_to_sections(text: str) -> Dict[str, Section]:
    t = text
    idx = _find_indices_by_keywords(t)
    n = len(t)

    # Определяем границы основной части
    main_start_pos = 0
    main_end_pos = n

    # Ищем начало основной части (после оглавления или по слову "Введение")
    toc_end_pos = 0
    if "toc" in idx:
        toc_text = t[idx["toc"]:]
        # Ищем конец оглавления по первой главе/введению
        m_ch = re.search(_CHAPTER_HINT, toc_text, flags=re.IGNORECASE)
        if m_ch:
            main_start_pos = idx["toc"] + m_ch.start()
            toc_end_pos = main_start_pos
        else:
            toc_end_pos = n # Если не нашли, считаем, что оглавление до конца
    else:
        # Если нет оглавления, ищем "Введение"
        m_ch = re.search(_CHAPTER_HINT, t, flags=re.IGNORECASE)
        if m_ch:
            main_start_pos = m_ch.start()

    # Ищем конец основной части (перед заключением или списком литературы)
    conclusion_pos = idx.get("conclusion", n)
    references_pos = idx.get("references", n)
    main_end_pos = min(conclusion_pos, references_pos)

    # Избегаем пересечения
    if main_start_pos >= main_end_pos:
        main_start_pos = toc_end_pos

    # Режем текст на секции
    s_cover = Section("cover", t[:toc_end_pos] if "toc" in idx else t[:main_start_pos])
    s_toc = Section("toc", t[idx["toc"]:toc_end_pos] if "toc" in idx else "")
    s_main = Section("main", t[main_start_pos:main_end_pos])
    s_conclusion = Section("conclusion", t[idx["conclusion"]:references_pos] if "conclusion" in idx and "references" in idx and idx["conclusion"] < idx["references"] else t[idx.get("conclusion", n):])
    s_references = Section("references", t[idx.get("references", n):])
    
    return {
        "cover": s_cover, "toc": s_toc, "main": s_main,
        "conclusion": s_conclusion, "references": s_references, "other": Section("other", "")
    }

def _parse_pdf(file_bytes: bytes) -> ParsedDocument:
    if PdfReader is None: raise RuntimeError("PyPDF2 не установлен. Установите: pip install PyPDF2")
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = [p.extract_text() or "" for p in reader.pages]
    full_text = "\n".join(pages_text)
    sections = _split_to_sections(full_text)
    return ParsedDocument("pdf", len(pages_text), full_text, sections, pages_text)

def _parse_docx(file_bytes: bytes) -> ParsedDocument:
    if DocxDocument is None: raise RuntimeError("python-docx не установлен. Установите: pip install python-docx")
    doc = DocxDocument(io.BytesIO(file_bytes))
    full_text = "\n".join([p.text for p in doc.paragraphs])
    sections = _split_to_sections(full_text)
    return ParsedDocument("docx", 0, full_text, sections, None)

def _parse_txt(file_bytes: bytes) -> ParsedDocument:
    try:
        full_text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        full_text = file_bytes.decode("cp1251", errors="ignore")
    sections = _split_to_sections(full_text)
    return ParsedDocument("txt", 0, full_text, sections, None)

def parse_document(file_bytes: bytes, filename: str) -> ParsedDocument:
    ftype = _detect_filetype(filename, file_bytes)
    if ftype == "pdf": return _parse_pdf(file_bytes)
    if ftype == "docx": return _parse_docx(file_bytes)
    return _parse_txt(file_bytes)

def _select_main_text_by_pages(parsed: ParsedDocument, page_range: Optional[Tuple[int, int]]) -> str:
    main_text = parsed.sections["main"].text or ""
    if not page_range: return main_text
    start, end = page_range
    if start is None or end is None: return main_text

    if parsed.filetype == "pdf" and parsed.pages_text:
        start = max(1, int(start))
        end = max(start, int(end))
        pages = parsed.pages_text[start-1:end]
        return "\n".join(pages) if pages else main_text
    else:
        # Для DOCX/TXT аппроксимация не очень надежна, лучше рерайтить всё
        return main_text

def _chunk_text(text: str, max_chars: int = 8000, overlap: int = 300) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars: return [text]
    chunks, i = [], 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        chunks.append(text[i:j])
        if j >= len(text): break
        i = j - overlap
    return chunks

def _build_prompt(chunk: str, tone: str, target_uniqueness: Optional[str]) -> str:
    uniq_line = f"Цель по уникальности (антиплагиат): {target_uniqueness}.\n" if target_uniqueness else ""
    return (
        "Ты — академический редактор. Перепиши фрагмент курсовой работы так, чтобы:\n"
        f"- стиль: {tone};\n"
        "- сохранялась исходная мысль и структура абзацев, списков и формул;\n"
        "- речевые штампы и повторы убрать, логику связать, водные конструкции не добавлять;\n"
        "- не выдумывать факты, цифры и источники; цифры и обозначения не искажать;\n"
        "- не добавлять вступления/выводы от себя;\n"
        "- объём сопоставим с исходником (±10%).\n"
        f"{uniq_line}\n"
        "ТЕКСТ ДЛЯ РЕРАЙТА:\n"
        f"\"\"\"\n{chunk}\n\"\"\"\n"
        "ВЫВОД:\n(только перефразированный текст без пояснений)"
    )

async def rewrite_main_async(
    parsed: ParsedDocument, rewrite_fn, tone: str = "официальный",
    target_uniqueness: Optional[str] = None, page_range: Optional[Tuple[int, int]] = None,
) -> str:
    main_text = _select_main_text_by_pages(parsed, page_range)
    if not main_text.strip(): return ""

    chunks = _chunk_text(main_text, max_chars=8000, overlap=300)
    rewritten_chunks: List[str] = []
    for ch in chunks:
        prompt = _build_prompt(ch, tone=tone, target_uniqueness=target_uniqueness)
        out = await rewrite_fn(prompt)
        out = out.strip().strip("`").strip()
        rewritten_chunks.append(out)

    return "\n".join(rewritten_chunks).strip()

def build_docx(parsed: ParsedDocument, new_main_text: str) -> bytes:
    if DocxDocument is None: raise RuntimeError("python-docx не установлен. Установите: pip install python-docx")
    doc = DocxDocument()
    
    # Стилизация (базовая)
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(14)
    
    def add_text_block(text_block: str):
        if text_block and text_block.strip():
            for para in text_block.strip().split("\n"):
                doc.add_paragraph(para.strip())

    # Собираем документ
    add_text_block(parsed.sections.get("cover", Section("cover", "")).text)
    add_text_block(parsed.sections.get("toc", Section("toc", "")).text)
    add_text_block(new_main_text or parsed.sections.get("main", Section("main", "")).text)
    add_text_block(parsed.sections.get("conclusion", Section("conclusion", "")).text)
    add_text_block(parsed.sections.get("references", Section("references", "")).text)

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()
