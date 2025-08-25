# -*- coding: utf-8 -*-

from typing import Tuple
from telegram import Update

async def read_telegram_file(update: Update) -> Tuple[bytes, str]:
    """
    Читает первый документ/файл из апдейта.
    Возвращает (file_bytes, filename).
    Бросает исключение, если файла нет.
    """
    doc = None
    if update.message and update.message.document:
        doc = update.message.document
    elif update.edited_message and update.edited_message.document:
        doc = update.edited_message.document

    if not doc:
        raise RuntimeError("Файл не найден в сообщении.")

    file = await doc.get_file()
    fb = await file.download_as_bytearray()
    return bytes(fb), (doc.file_name or "file.bin")
