# -*- coding: utf-8 -*-

import os, io, csv, time, html
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, InputFile
from telegram.ext import (
    Application, CommandHandler, ContextTypes, ConversationHandler,
    MessageHandler, CallbackQueryHandler, filters, PicklePersistence
)
from telegram.helpers import escape_markdown
from telegram.error import BadRequest

# ===== .env / .evn =====
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(filename=".env", raise_error_if_not_found=False))
    if os.path.exists(".evn"):
        load_dotenv(".evn")
except Exception:
    pass

# ===== CONFIG =====
BOT_TOKEN        = os.getenv("BOT_TOKEN")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
ADMIN_CHAT_ID    = os.getenv("ADMIN_CHAT_ID")
FREE_LIMIT       = int(os.getenv("FREE_LIMIT", "5"))
RL_WINDOW_SEC    = int(os.getenv("RL_WINDOW_SEC", "10"))  # троттлинг окно
RL_MAX_HITS      = int(os.getenv("RL_MAX_HITS", "3"))     # макс. запросов в окне
CAPTCHA_ENABLED  = os.getenv("CAPTCHA_ENABLED", "1") == "1"

# Рефералы
REF_BONUS_DAYS         = int(os.getenv("REF_BONUS_DAYS", "1"))   # награда рефереру (в днях безлимита)
REF_WELCOME_ATTEMPTS   = int(os.getenv("REF_WELCOME_ATTEMPTS", "2"))  # бонус приглашённому на сегодня (к лимиту)

def _parse_admin(s: Optional[str]) -> Optional[int]:
    try:
        return int(s) if s else None
    except Exception:
        return None

ADMIN_USER_ID: Optional[int] = _parse_admin(ADMIN_CHAT_ID)

# ===== LOGS =====
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== STATES =====
(
    MAIN_MENU, REWRITER_TEXT_INPUT, LITERATURE_TOPIC_INPUT, GOST_MENU,
    ADMIN_MENU, ADMIN_RESET_WAIT_ID, ADMIN_ADDSUB_WAIT_ID, ADMIN_ADDSUB_WAIT_DAYS,
    ADMIN_STATUS_WAIT_ID, ADMIN_DELSUB_WAIT_ID,
    CABINET_MENU, CAPTCHA_WAIT,
    SETTINGS_MENU, SETTINGS_TONE_WAIT, SETTINGS_GOST_WAIT,
    ADMIN_SEARCH_WAIT_QUERY, ADMIN_TAGS_WAIT_ID, ADMIN_TAGS_WAIT_VALUE,
    ADMIN_BROADCAST_SEGMENT, ADMIN_BROADCAST_WAIT_TEXT,
    ADMIN_SETLIMIT_WAIT_ID, ADMIN_SETLIMIT_WAIT_VALUES,
    ADMIN_BLACKLIST_WAIT_ID, ADMIN_SHADOW_WAIT_ID,
    CABINET_REF_MENU
) = range(25)

# ===== HELPERS: dates/roles =====
def _today() -> str: return datetime.now().strftime("%Y-%m-%d")
def is_admin(uid: int) -> bool: return ADMIN_USER_ID is not None and uid == ADMIN_USER_ID

def has_active_subscription(context: ContextTypes.DEFAULT_TYPE) -> bool:
    exp = context.user_data.get("subscription_expires")
    if not exp: return False
    try:
        return datetime.strptime(exp, "%Y-%m-%d").date() >= datetime.now().date()
    except Exception:
        return False

# ===== LIMITS (с учётом дневного бонуса) =====
def _limit_bonus_today(context: ContextTypes.DEFAULT_TYPE) -> int:
    lb = context.user_data.get("limit_bonus")
    if not lb or lb.get("date") != _today():
        context.user_data["limit_bonus"] = {"date": _today(), "extra": 0}
        return 0
    return int(lb.get("extra", 0))

def add_daily_bonus(context: ContextTypes.DEFAULT_TYPE, extra: int) -> None:
    lb = context.user_data.get("limit_bonus")
    if not lb or lb.get("date") != _today():
        context.user_data["limit_bonus"] = {"date": _today(), "extra": max(0, int(extra))}
    else:
        lb["extra"] = max(0, int(lb.get("extra", 0))) + max(0, int(extra))
        context.user_data["limit_bonus"] = lb

def get_daily_limit(context: ContextTypes.DEFAULT_TYPE) -> int:
    return FREE_LIMIT + _limit_bonus_today(context)

def get_user_usage(feature: str, context: ContextTypes.DEFAULT_TYPE) -> int:
    u = context.user_data.setdefault("usage", {})
    d = u.get(feature, {"count": 0, "date": _today()})
    if d.get("date") != _today():
        d = {"count": 0, "date": _today()}
        u[feature] = d
    return int(d.get("count", 0))

def increment_usage(feature: str, context: ContextTypes.DEFAULT_TYPE) -> int:
    u = context.user_data.setdefault("usage", {})
    d = u.get(feature, {"count": 0, "date": _today()})
    if d.get("date") != _today():
        d = {"count": 1, "date": _today()}
    else:
        d["count"] = int(d.get("count", 0)) + 1
        d["date"] = _today()
    u[feature] = d
    return d["count"]

def remaining_attempts(feature: str, context: ContextTypes.DEFAULT_TYPE, uid: int) -> str:
    if is_admin(uid) or has_active_subscription(context):
        return "∞ (Безлимит)"
    limit = get_daily_limit(context)
    return str(max(0, limit - get_user_usage(feature, context)))

# ===== HISTORY + analytics =====
def _push_history(context: ContextTypes.DEFAULT_TYPE, feature: str, size: int) -> None:
    hist: List[Dict[str, Any]] = context.user_data.setdefault("history", [])
    hist.append({"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "feature": feature, "size": int(size)})
    if len(hist) > 100: del hist[:-100]

def _format_history(context: ContextTypes.DEFAULT_TYPE, limit: int = 10) -> str:
    hist: List[Dict[str, Any]] = context.user_data.get("history", [])
    if not hist: return "Пока нет записей об использовании."
    return "\n".join(f"• {i['ts']}: {i['feature']} (длина ввода: {i['size']})"
                     for i in list(reversed(hist))[:limit])

def _record_ai_stat(application: Application, ok: bool) -> None:
    stats = application.bot_data.setdefault("ai_stats", [])
    stats.append({"ts": int(time.time()), "ok": bool(ok)})
    if len(stats) > 200: del stats[:-200]

def _service_snapshot(application: Application, n: int = 50) -> Dict[str, Any]:
    stats = application.bot_data.get("ai_stats", [])
    subset = stats[-n:] if n>0 else stats[:]
    if not subset: return {"count": 0, "error_rate": 0.0}
    errs = sum(1 for x in subset if not x.get("ok"))
    return {"count": len(subset), "error_rate": round(errs/len(subset)*100, 1)}

# ===== Captcha / Anti-spam / Blacklist =====
def _touch_seen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    u = update.effective_user
    if u:
        context.user_data["last_username"] = (u.username or "")[:64]
        context.user_data["first_name"] = (u.first_name or "")[:128]
        context.user_data["last_name"] = (u.last_name or "")[:128]

def _is_blacklisted(app: Application, uid: int) -> bool:
    bl = app.bot_data.get("blacklist", set())
    return uid in bl

def _is_shadowbanned(app: Application, uid: int) -> bool:
    sb = app.bot_data.get("shadowban", set())
    return uid in sb

def _rate_limit_ok(context: ContextTypes.DEFAULT_TYPE) -> tuple[bool, int]:
    now = time.time()
    arr: List[float] = context.user_data.setdefault("rl_times", [])
    arr = [t for t in arr if now - t <= RL_WINDOW_SEC]
    ok = len(arr) < RL_MAX_HITS
    if ok:
        arr.append(now)
        context.user_data["rl_times"] = arr
        return True, 0
    wait = RL_WINDOW_SEC - int(now - min(arr))
    return False, max(wait, 1)

def _need_captcha(context: ContextTypes.DEFAULT_TYPE, uid: int) -> bool:
    if not CAPTCHA_ENABLED: return False
    if is_admin(uid): return False
    return not context.user_data.get("captcha_ok", False)

def _gen_captcha(context: ContextTypes.DEFAULT_TYPE) -> str:
    import random
    a, b = random.randint(2, 9), random.randint(2, 9)
    context.user_data["captcha_answer"] = str(a + b)
    return f"Проверка: сколько будет {a} + {b}? Отправьте ответ числом."

# ===== Gemini =====
async def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "Ошибка: API-ключ для нейросети не настроен."
    api_url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-2.5-flash-preview-05-20:generateContent"
               f"?key={GEMINI_API_KEY}")
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        from httpx import AsyncClient
        async with AsyncClient() as client:
            r = await client.post(api_url, json=payload, timeout=60.0)
            r.raise_for_status()
            j = r.json()
            cand = j.get("candidates", [])
            if cand and cand[0].get("content", {}).get("parts"):
                return cand[0]["content"]["parts"][0]["text"]
            return "Не удалось получить корректный ответ от AI. Попробуйте позже."
    except Exception as e:
        return f"Произошла ошибка при обращении к нейросети: {e}"

def _no_literature_found(txt: str) -> bool:
    s = (txt or "").strip()
    if not s: return True
    low = s.lower()
    if low.startswith(("ошибка", "не удалось")): return True
    if any(x in low for x in ["ничего не найдено", "источники не найд", "нет данных", "no sources"]): return True
    if s.count("\n") <= 1 and len(s) < 60: return True
    return False

# ===== Keyboards =====
def main_menu_kb(uid: int) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("✍️ AI-Рерайтер (Уникальность)", callback_data="rewriter")],
        [InlineKeyboardButton("📚 Генератор списка литературы", callback_data="literature")],
        [InlineKeyboardButton("📋 Консультант по ГОСТу", callback_data="gost")],
        [InlineKeyboardButton("👤 Личный кабинет", callback_data="cabinet")],
    ]
    if is_admin(uid):
        rows.append([InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")])
    return InlineKeyboardMarkup(rows)

def back_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад в меню", callback_data="back_to_main_menu")]])

def contact_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💬 Написать автору", url="https://t.me/V_L_A_D_IS_L_A_V")],
        [InlineKeyboardButton("⬅️ Назад в меню", callback_data="back_to_main_menu")]
    ])

def cabinet_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🗂 История (10)", callback_data="cab_history")],
        [InlineKeyboardButton("📥 Экспорт истории (CSV)", callback_data="cab_export")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="cab_settings")],
        [InlineKeyboardButton("🎁 Рефералы", callback_data="cab_ref")],
        [InlineKeyboardButton("🔥 Получить безлимит", url="https://t.me/V_L_A_D_IS_L_A_V")],
        [InlineKeyboardButton("🔄 Обновить", callback_data="cab_refresh")],
        [InlineKeyboardButton("⬅️ Назад в меню", callback_data="back_to_main_menu")],
    ])

def admin_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔁 Сбросить лимиты", callback_data="admin_reset"),
         InlineKeyboardButton("📊 Статус пользователя", callback_data="admin_status")],
        [InlineKeyboardButton("➕ Добавить подписку", callback_data="admin_addsub"),
         InlineKeyboardButton("❌ Отменить подписку", callback_data="admin_delsub")],
        [InlineKeyboardButton("🔎 Поиск пользователя", callback_data="admin_search"),
         InlineKeyboardButton("🏷 Теги", callback_data="admin_tags")],
        [InlineKeyboardButton("📣 Рассылка", callback_data="admin_broadcast"),
         InlineKeyboardButton("📤 Экспорт CSV", callback_data="admin_export")],
        [InlineKeyboardButton("🚫 Блокировка", callback_data="admin_blacklist"),
         InlineKeyboardButton("👻 Теневой бан", callback_data="admin_shadow")],
        [InlineKeyboardButton("🎚 Задать лимиты", callback_data="admin_setlimit")],
        [InlineKeyboardButton("⬅️ Назад в главное меню", callback_data="back_to_main_menu")],
    ])

def admin_cancel_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅️ Назад в админ-панель", callback_data="admin_panel")],
        [InlineKeyboardButton("🏠 В меню", callback_data="back_to_main_menu")],
    ])

def settings_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🗣 Тон рерайта", callback_data="set_tone")],
        [InlineKeyboardButton("📏 ГОСТ по умолчанию", callback_data="set_gost")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="cabinet")],
    ])

# ===== Utils =====
TG_MD2_LIMIT = 3800
def _chunk_md2(text: str, limit: int = TG_MD2_LIMIT) -> List[str]:
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            head, line = line[:limit], line[limit:]
            if buf: chunks.append(buf); buf = ""
            chunks.append(head)
        if len(buf) + len(line) > limit:
            if buf: chunks.append(buf)
            buf = line
        else:
            buf += line
    if buf: chunks.append(buf)
    return chunks or ["_пусто_"]

async def _md2_send_chunks(msg, text: str, markup=None):
    parts = _chunk_md2(text, TG_MD2_LIMIT)
    if len(parts) == 1:
        await msg.edit_text(parts[0], parse_mode="MarkdownV2",
                            disable_web_page_preview=True, reply_markup=markup)
    else:
        await msg.edit_text(parts[0], parse_mode="MarkdownV2", disable_web_page_preview=True)
        for p in parts[1:-1]:
            await msg.reply_text(p, parse_mode="MarkdownV2", disable_web_page_preview=True)
        await msg.reply_text(parts[-1], parse_mode="MarkdownV2",
                             disable_web_page_preview=True, reply_markup=markup)

def _progress_bar(used: int, total: int, width: int = 20) -> str:
    if total <= 0: return "░"*width + " ∞"
    used = max(0, min(total, used))
    fill = int(width * used / total)
    return "█"*fill + "░"*(width - fill) + f" {used}/{total}"

def _ensure_first_seen(context: ContextTypes.DEFAULT_TYPE) -> None:
    if "first_seen" not in context.user_data:
        context.user_data["first_seen"] = _today()

async def _safe_edit_text(message, text: str, **kwargs):
    try:
        await message.edit_text(text, **kwargs)
    except BadRequest as e:
        if "not modified" in str(e).lower():
            await message.reply_text(text, **kwargs)
        else:
            raise

async def _ensure_bot_username(context: ContextTypes.DEFAULT_TYPE) -> str:
    name = context.application.bot_data.get("bot_username")
    if name:
        return name
    me = await context.bot.get_me()
    context.application.bot_data["bot_username"] = me.username
    return me.username

# ===== REFERRALS helpers =====
def _parse_ref_payload(args: List[str]) -> Optional[int]:
    if not args:
        return None
    payload = " ".join(args).strip()
    if payload.startswith("ref_"):
        suf = payload[4:]
        if suf.isdigit():
            return int(suf)
    return None

def _award_or_queue_bonus_for_referrer(context: ContextTypes.DEFAULT_TYPE, ref_id: int) -> Optional[str]:
    ud_map = context.application.user_data
    today = datetime.now().date()
    if ref_id in ud_map:
        ud = ud_map[ref_id]
        cur = ud.get("subscription_expires")
        if cur:
            try:
                cur_d = datetime.strptime(cur, "%Y-%m-%d").date()
            except Exception:
                cur_d = today
        else:
            cur_d = today
        base = max(cur_d, today)
        new_exp = base + timedelta(days=REF_BONUS_DAYS)
        ud["subscription_expires"] = new_exp.strftime("%Y-%m-%d")
        return new_exp.strftime("%d.%m.%Y")
    pending = context.application.bot_data.setdefault("ref_pending_days", {})
    pending[ref_id] = int(pending.get(ref_id, 0)) + REF_BONUS_DAYS
    return None

def _queue_ref_for_referrer(context: ContextTypes.DEFAULT_TYPE, ref_id: int, new_user_id: int) -> None:
    ud_map = context.application.user_data
    if ref_id in ud_map:
        refs = ud_map[ref_id].setdefault("refs", [])
        if new_user_id not in refs:
            refs.append(new_user_id)
    else:
        refs_map = context.application.bot_data.setdefault("referrals", {})
        lst = refs_map.setdefault(ref_id, [])
        if new_user_id not in lst:
            lst.append(new_user_id)

async def _apply_pending_for_current_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    bd = context.application.bot_data

    pending_days = 0
    ref_pending = bd.get("ref_pending_days", {})
    if uid in ref_pending:
        pending_days = int(ref_pending.pop(uid, 0))

    if pending_days > 0:
        today = datetime.now().date()
        cur = context.user_data.get("subscription_expires")
        if cur:
            try:
                cur_d = datetime.strptime(cur, "%Y-%m-%d").date()
            except Exception:
                cur_d = today
        else:
            cur_d = today
        base = max(cur_d, today)
        new_exp = base + timedelta(days=pending_days)
        context.user_data["subscription_expires"] = new_exp.strftime("%Y-%m-%d")
        try:
            await context.bot.send_message(
                chat_id=uid,
                text=f"🎁 Вам начислено +{pending_days} д. безлимита по рефералам.\n"
                     f"Доступ активен до {new_exp.strftime('%d.%m.%Y')} включительно."
            )
        except Exception:
            pass

    refs_map = bd.get("referrals", {})
    if uid in refs_map:
        pending_refs = refs_map.pop(uid, [])
        ud_refs = context.user_data.setdefault("refs", [])
        for r in pending_refs:
            if r not in ud_refs:
                ud_refs.append(r)

# ===== /start =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _touch_seen(update, context)
    uid = update.effective_user.id

    ref_from = None
    if update.message and context.args:
        ref_from = _parse_ref_payload(context.args)

    _ensure_first_seen(context)
    await _apply_pending_for_current_user(update, context)

    if ref_from and ref_from != uid and not context.user_data.get("ref_by"):
        context.user_data["ref_by"] = ref_from
        context.user_data["referred_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if REF_WELCOME_ATTEMPTS > 0:
            add_daily_bonus(context, REF_WELCOME_ATTEMPTS)

        _queue_ref_for_referrer(context, ref_from, uid)
        new_exp_h = _award_or_queue_bonus_for_referrer(context, ref_from)

        try:
            if new_exp_h:
                await context.bot.send_message(
                    chat_id=ref_from,
                    text=(f"🎉 Ваш приглашённый пользователь ID {uid} активировал бота.\n"
                          f"Начислено +{REF_BONUS_DAYS} д. безлимита. Доступ до {new_exp_h} включительно.")
                )
        except Exception:
            pass
        try:
            if REF_WELCOME_ATTEMPTS > 0:
                await update.effective_message.reply_text(
                    f"🎁 Добро пожаловать! Вам начислено +{REF_WELCOME_ATTEMPTS} доп. попыток на сегодня."
                )
        except Exception:
            pass

    if _is_blacklisted(context.application, uid):
        await update.effective_message.reply_html("🚫 Доступ к боту ограничен.")
        return ConversationHandler.END
    if _is_shadowbanned(context.application, uid):
        await update.effective_message.reply_text("Сервис временно перегружен, попробуйте позже.")
        return ConversationHandler.END

    if _need_captcha(context, uid):
        q = _gen_captcha(context)
        await update.effective_message.reply_text(q)
        return CAPTCHA_WAIT

    text = (f"Здравствуйте, {update.effective_user.mention_html()}!\n\n"
            "Я «Студент-Ассистент». Выберите инструмент:")
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit_text(update.callback_query.message, text, parse_mode="HTML", reply_markup=main_menu_kb(uid))
    else:
        await update.effective_message.reply_html(text, reply_markup=main_menu_kb(uid))
    return MAIN_MENU

# ===== CAPTCHA =====
async def captcha_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    ans = (update.message.text or "").strip()
    ok = context.user_data.get("captcha_answer")
    if ans == ok:
        context.user_data["captcha_ok"] = True
        await update.message.reply_text("✅ Спасибо! Продолжаем.")
        return await start(update, context)
    await update.message.reply_text("❌ Неверно. Попробуйте ещё раз. " + _gen_captcha(context))
    return CAPTCHA_WAIT

# ===== HELP & SUPPORT =====
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    t = (
        "📖 Помощь\n\n"
        "• /start — главное меню\n"
        "• /status — ваш доступ и остаток попыток\n"
        "• /settings — личные настройки\n"
        "• /report <сообщение> — жалоба с логами\n"
        "• /service — статус ИИ\n\n"
        "Для безлимита и вопросов: @V_L_A_D_IS_L_A_V"
    )
    await update.message.reply_text(t)

async def service_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    snap = _service_snapshot(context.application, 50)
    await update.message.reply_html(
        f"🩺 <b>Статус</b>\nПоследних запросов: {snap['count']}\nОшибка ИИ: {snap['error_rate']}%\n"
        "Если ошибка высока — возможны задержки/сбои.\n"
        "Поддержка: <a href='https://t.me/V_L_A_D_IS_L_A_V'>@V_L_A_D_IS_L_A_V</a>"
    )

async def report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (update.message.text or "").partition(" ")[2].strip()
    if not ADMIN_USER_ID:
        await update.message.reply_text("Администратор не настроен.")
        return
    last_req = context.user_data.get("last_request", {})
    hist = _format_history(context, limit=5)
    uname = f"@{(update.effective_user.username or '').strip()}" if update.effective_user.username else "—"
    html_msg = (
        f"🛎 <b>Жалоба</b>\n"
        f"ID: <code>{update.effective_user.id}</code>\n"
        f"Юзернейм: {html.escape(uname)}\n"
        f"Текст: {html.escape(msg) or '—'}\n\n"
        f"<b>Last request:</b> {html.escape(str(last_req))}\n\n"
        f"<b>History(5):</b>\n{html.escape(hist)}"
    )
    try:
        await context.bot.send_message(chat_id=ADMIN_USER_ID, text=html_msg, parse_mode="HTML")
        await update.message.reply_text("Спасибо! Сообщение отправлено.")
    except Exception as e:
        logger.warning("report send fail: %s", e)
        await update.message.reply_text("Не удалось отправить отчёт. Напишите @V_L_A_D_IS_L_A_V")

# ===== ADMIN COMMANDS (text) =====
async def reset_limit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Только администратор.")
        return
    try: target_id = int(context.args[0])
    except Exception:
        await update.message.reply_text("Используйте: /reset <user_id>")
        return
    ud = context.application.user_data.get(target_id)
    if not ud:
        await update.message.reply_text("Пользователь не найден.")
        return
    today = _today()
    usage = ud.setdefault("usage", {})
    usage["rewriter"] = {"count": 0, "date": today}
    usage["literature"] = {"count": 0, "date": today}
    await update.message.reply_text("✅ Сброшено.")
    try: await context.bot.send_message(chat_id=target_id, text="🎉 Ваш дневной лимит сброшен администратором.")
    except Exception: pass

async def add_subscription(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Только администратор.")
        return
    try:
        target_id = int(context.args[0]); days = int(context.args[1]); assert days>0
    except Exception:
        await update.message.reply_text("Используйте: /addsub <user_id> <days>")
        return
    ud = context.application.user_data.get(target_id)
    if not ud: await update.message.reply_text("Пользователь не найден."); return
    exp = (datetime.now().date() + timedelta(days=days)).strftime("%Y-%m-%d")
    ud["subscription_expires"] = exp
    exp_h = datetime.strptime(exp,'%Y-%m-%d').strftime('%d.%m.%Y')
    await update.message.reply_text(f"✅ Подписка до {exp_h}")
    try: await context.bot.send_message(chat_id=target_id, text=f"🎉 Вам выдали безлимит на {days} дн. Доступ до {exp_h} включительно.")
    except Exception: pass

async def del_subscription(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Только администратор.")
        return
    try: target_id = int(context.args[0])
    except Exception: await update.message.reply_text("Используйте: /delsub <user_id>"); return
    ud = context.application.user_data.get(target_id)
    if not ud: await update.message.reply_text("Пользователь не найден."); return
    if "subscription_expires" in ud:
        ud.pop("subscription_expires", None); await update.message.reply_text("🛑 Подписка отключена.")
        try: await context.bot.send_message(chat_id=target_id, text="🛑 Ваша подписка отключена.")
        except Exception: pass
    else:
        await update.message.reply_text("У пользователя нет активной подписки.")

async def check_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    if has_active_subscription(context):
        exp = datetime.strptime(context.user_data.get("subscription_expires"), "%Y-%m-%d").strftime("%d.%m.%Y")
        await update.message.reply_html(f"<b>Статус:</b> ✅ Безлимит до {exp}")
    else:
        limit = get_daily_limit(context)
        r = remaining_attempts("rewriter", context, uid)
        l = remaining_attempts("literature", context, uid)
        await update.message.reply_html(
            "<b>Статус:</b> базовый доступ\n"
            f"• Рерайтер: {r} из {limit}\n"
            f"• Литература: {l} из {limit}"
        )

# ===== ADMIN PANEL (callbacks + flows) =====
async def admin_panel_open(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_admin(update.effective_user.id):
        if update.callback_query: await update.callback_query.answer("Только администратор.", show_alert=True)
        else: await update.message.reply_text("Только администратор.")
        return MAIN_MENU
    if update.callback_query:
        await update.callback_query.answer()
        try:
            await update.callback_query.message.edit_text("⚙️ <b>Админ-панель</b>", parse_mode="HTML", reply_markup=admin_menu_kb())
        except BadRequest as e:
            if "not modified" in str(e).lower():
                await update.callback_query.message.reply_html("⚙️ <b>Админ-панель</b>", reply_markup=admin_menu_kb())
            else:
                raise
    else:
        await update.message.reply_html("⚙️ <b>Админ-панель</b>", reply_markup=admin_menu_kb())
    return ADMIN_MENU

# reset
async def admin_reset_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await _safe_edit_text(update.callback_query.message,
                          "Введите <b>ID</b> для сброса дневных лимитов:", parse_mode="HTML",
                          reply_markup=admin_cancel_kb())
    return ADMIN_RESET_WAIT_ID

async def admin_reset_receive_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try: target_id = int((update.message.text or "").strip())
    except Exception:
        await update.message.reply_text("Неверный ID. Введите число.", reply_markup=admin_cancel_kb()); return ADMIN_RESET_WAIT_ID
    ud = context.application.user_data.get(target_id)
    if not ud:
        await update.message.reply_text("Пользователь не найден.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    today = _today()
    u = ud.setdefault("usage", {})
    u["rewriter"]={"count":0,"date":today}; u["literature"]={"count":0,"date":today}
    await update.message.reply_text("✅ Сброшено.", reply_markup=admin_cancel_kb())
    try:
        await context.bot.send_message(chat_id=target_id, text="🎉 Ваш дневной лимит был сброшен администратором.")
    except Exception:
        pass
    return ADMIN_MENU

# addsub
async def admin_addsub_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await _safe_edit_text(update.callback_query.message,
                          "Введите <b>ID</b> для выдачи подписки:", parse_mode="HTML",
                          reply_markup=admin_cancel_kb())
    return ADMIN_ADDSUB_WAIT_ID

async def admin_addsub_receive_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try: target_id = int((update.message.text or "").strip())
    except Exception:
        await update.message.reply_text("Неверный ID. Введите число.", reply_markup=admin_cancel_kb()); return ADMIN_ADDSUB_WAIT_ID
    context.user_data["admin_addsub_target"] = target_id
    await update.message.reply_html(f"ID <code>{target_id}</code> принят. Введите <b>кол-во дней</b>:", reply_markup=admin_cancel_kb())
    return ADMIN_ADDSUB_WAIT_DAYS

async def admin_addsub_receive_days(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try: days = int((update.message.text or "").strip()); assert days>0
    except Exception:
        await update.message.reply_text("Неверное число. Введите положительное целое.", reply_markup=admin_cancel_kb()); return ADMIN_ADDSUB_WAIT_DAYS
    target_id = context.user_data.get("admin_addsub_target")
    if target_id is None:
        await update.message.reply_text("Сессия сброшена. Повторите.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    ud = context.application.user_data.get(target_id)
    if not ud:
        await update.message.reply_text("Пользователь не найден.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    exp = (datetime.now().date()+timedelta(days=days)).strftime("%Y-%m-%d")
    ud["subscription_expires"]=exp
    await update.message.reply_text("✅ Подписка выдана.", reply_markup=admin_cancel_kb())
    try:
        exp_h = datetime.strptime(exp,'%Y-%m-%d').strftime('%d.%m.%Y')
        await context.bot.send_message(chat_id=target_id,
            text=f"🎉 Вам активировали безлимит на {days} дней.\nДоступ действует до {exp_h} включительно.")
    except Exception:
        pass
    context.user_data.pop("admin_addsub_target", None)
    return ADMIN_MENU

# delsub
async def admin_delsub_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await _safe_edit_text(update.callback_query.message,
                          "Введите <b>ID</b> для отмены подписки:", parse_mode="HTML",
                          reply_markup=admin_cancel_kb())
    return ADMIN_DELSUB_WAIT_ID

async def admin_delsub_receive_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try: target_id = int((update.message.text or "").strip())
    except Exception:
        await update.message.reply_html("Неверный ID.", reply_markup=admin_cancel_kb()); return ADMIN_DELSUB_WAIT_ID
    ud = context.application.user_data.get(target_id)
    if not ud:
        await update.message.reply_text("Пользователь не найден.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    ud.pop("subscription_expires", None)
    await update.message.reply_text("🛑 Подписка отключена.", reply_markup=admin_cancel_kb())
    try:
        await context.bot.send_message(chat_id=target_id, text="🛑 Ваша подписка отключена администратором.")
    except Exception:
        pass
    return ADMIN_MENU

# status
async def admin_status_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await _safe_edit_text(update.callback_query.message,
                          "Введите <b>ID</b> для просмотра статуса:",
                          parse_mode="HTML", reply_markup=admin_cancel_kb())
    return ADMIN_STATUS_WAIT_ID

async def admin_status_receive_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try: target_id = int((update.message.text or "").strip())
    except Exception:
        await update.message.reply_text("Неверный ID.", reply_markup=admin_cancel_kb()); return ADMIN_STATUS_WAIT_ID
    ud = context.application.user_data.get(target_id)
    if not ud:
        await update.message.reply_text("Пользователь не найден.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    exp = ud.get("subscription_expires")
    sub_line = f"Подписка: до {datetime.strptime(exp,'%Y-%m-%d').strftime('%d.%m.%Y')}" if exp else "Подписка: нет"
    usage = ud.get("usage", {})
    rew = usage.get("rewriter", {"count":0,"date":_today()})
    lit = usage.get("literature", {"count":0,"date":_today()})
    tags = ", ".join(ud.get("tags", [])) or "—"
    refs = ", ".join(str(x) for x in ud.get("refs", [])) or "—"
    await update.message.reply_html(
        f"👤 <b>{target_id}</b>\n{sub_line}\n"
        f"Рерайтер: {rew.get('count',0)} ({rew.get('date','-')})\n"
        f"Литература: {lit.get('count',0)} ({lit.get('date','-')})\n"
        f"Теги: {html.escape(tags)}\n"
        f"Рефералы: {html.escape(refs)}",
        reply_markup=admin_cancel_kb()
    )
    return ADMIN_MENU

# search
async def admin_search_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await _safe_edit_text(update.callback_query.message,
                          "Введите ID или часть юзернейма:", reply_markup=admin_cancel_kb())
    return ADMIN_SEARCH_WAIT_QUERY

async def admin_search_do(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = (update.message.text or "").strip().lower()
    res = []
    ud_map = context.application.user_data
    if q.isdigit():
        uid = int(q)
        if uid in ud_map:
            u = ud_map[uid]; res.append((uid, u.get("last_username",""), u.get("first_seen","?")))
    else:
        for uid, data in ud_map.items():
            if q in (data.get("last_username","").lower()):
                res.append((uid, data.get("last_username",""), data.get("first_seen","?")))
            if len(res) >= 20: break
    if not res:
        await update.message.reply_text("Ничего не найдено.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    lines = [f"• {uid} @{name or '—'} (с {fs})" for uid,name,fs in res]
    await update.message.reply_text("Найдено:\n" + "\n".join(lines), reply_markup=admin_cancel_kb())
    return ADMIN_MENU

# tags
async def admin_tags_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await _safe_edit_text(update.callback_query.message,
                          "Введите ID пользователя:", reply_markup=admin_cancel_kb())
    return ADMIN_TAGS_WAIT_ID

async def admin_tags_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try: target = int((update.message.text or "").strip())
    except Exception:
        await update.message.reply_text("Неверный ID.", reply_markup=admin_cancel_kb()); return ADMIN_TAGS_WAIT_ID
    context.user_data["tags_target"] = target
    await update.message.reply_text("Введите теги через запятую (для удаления перед тегом поставьте -):\n"
                                    "пример: vip, блогер, -test", reply_markup=admin_cancel_kb())
    return ADMIN_TAGS_WAIT_VALUE

async def admin_tags_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    target = context.user_data.get("tags_target"); txt = (update.message.text or "").strip()
    if target is None:
        await update.message.reply_text("Сессия сброшена.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    ud = context.application.user_data.get(target)
    if not ud:
        await update.message.reply_text("Пользователь не найден.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    tags = [t.strip() for t in txt.split(",") if t.strip()]
    cur = set(ud.get("tags", []))
    for t in tags:
        if t.startswith("-"): cur.discard(t[1:].strip())
        else: cur.add(t)
    ud["tags"] = sorted([t for t in cur if t])
    await update.message.reply_text(f"Готово. Теги: {', '.join(ud['tags']) or '—'}", reply_markup=admin_cancel_kb())
    context.user_data.pop("tags_target", None)
    return ADMIN_MENU

# broadcast
async def admin_broadcast_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Все", callback_data="bseg_all"),
         InlineKeyboardButton("Free", callback_data="bseg_free"),
         InlineKeyboardButton("Pro", callback_data="bseg_subs"),
         InlineKeyboardButton("Неактивные 7д", callback_data="bseg_inactive")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="admin_panel")]
    ])
    await _safe_edit_text(update.callback_query.message, "Выберите сегмент рассылки:", reply_markup=kb)
    return ADMIN_BROADCAST_SEGMENT

async def admin_broadcast_pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    seg = update.callback_query.data
    context.user_data["b_segment"] = seg
    await _safe_edit_text(update.callback_query.message, "Введите текст рассылки:", reply_markup=admin_cancel_kb())
    return ADMIN_BROADCAST_WAIT_TEXT

async def admin_broadcast_send(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    seg = context.user_data.get("b_segment"); txt = (update.message.text or "").strip()
    if not seg or not txt:
        await update.message.reply_text("Нет сегмента или текста.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    ud_map = context.application.user_data
    sent = 0
    cutoff = datetime.now() - timedelta(days=7)
    for uid, data in list(ud_map.items()):
        if seg == "bseg_free" and data.get("subscription_expires"):
            try:
                if datetime.strptime(data["subscription_expires"], "%Y-%m-%d").date() >= datetime.now().date():
                    continue
            except Exception:
                pass
        if seg == "bseg_subs" and not data.get("subscription_expires"):
            continue
        if seg == "bseg_inactive":
            ls = data.get("last_seen")
            if ls:
                try:
                    if datetime.strptime(ls, "%Y-%m-%d %H:%M:%S") >= cutoff:
                        continue
                except Exception:
                    pass
        try:
            await context.bot.send_message(chat_id=uid, text=txt)
            sent += 1
        except Exception:
            continue
    await update.message.reply_text(f"Готово. Отправлено: {sent}", reply_markup=admin_cancel_kb())
    context.user_data.pop("b_segment", None)
    return ADMIN_MENU

# export users CSV
async def admin_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id","first_seen","last_seen","username","subscription_expires","tags","refs"])
    for uid, d in context.application.user_data.items():
        w.writerow([uid, d.get("first_seen",""), d.get("last_seen",""),
                    d.get("last_username",""), d.get("subscription_expires",""),
                    "|".join(d.get("tags", [])),
                    "|".join(str(x) for x in d.get("refs", []))])
    byte = io.BytesIO(buf.getvalue().encode("utf-8"))
    byte.name = "users_export.csv"
    await update.callback_query.answer()
    await update.callback_query.message.reply_document(InputFile(byte), caption="Экспорт пользователей (CSV)")
    return ADMIN_MENU

# set limits
async def admin_setlimit_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await _safe_edit_text(update.callback_query.message, "Введите ID пользователя:", reply_markup=admin_cancel_kb())
    return ADMIN_SETLIMIT_WAIT_ID

async def admin_setlimit_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try: target = int((update.message.text or "").strip())
    except Exception:
        await update.message.reply_text("Неверный ID.", reply_markup=admin_cancel_kb()); return ADMIN_SETLIMIT_WAIT_ID
    context.user_data["limit_target"] = target
    await update.message.reply_text("Введите два числа через пробел — рерайтер и литература (сегодня):\nнапр. 2 1",
                                    reply_markup=admin_cancel_kb())
    return ADMIN_SETLIMIT_WAIT_VALUES

async def admin_setlimit_values(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    target = context.user_data.get("limit_target")
    if target is None:
        await update.message.reply_text("Сессия сброшена.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    parts = (update.message.text or "").split()
    try: c1, c2 = int(parts[0]), int(parts[1])
    except Exception:
        await update.message.reply_text("Нужно два числа, пример: 3 2", reply_markup=admin_cancel_kb()); return ADMIN_SETLIMIT_WAIT_VALUES
    ud = context.application.user_data.get(target)
    if not ud: await update.message.reply_text("Пользователь не найден.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    today = _today()
    u = ud.setdefault("usage", {})
    u["rewriter"]={"count":max(0,c1),"date":today}
    u["literature"]={"count":max(0,c2),"date":today}
    await update.message.reply_text("✅ Установлено.", reply_markup=admin_cancel_kb())
    context.user_data.pop("limit_target", None); return ADMIN_MENU

# blacklist / shadowban
async def admin_blacklist_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await _safe_edit_text(update.callback_query.message,
                          "Введите ID для добавления/удаления из чёрного списка (для удаления добавьте минус перед ID):",
                          reply_markup=admin_cancel_kb())
    return ADMIN_BLACKLIST_WAIT_ID

async def admin_blacklist_apply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    txt = (update.message.text or "").strip()
    bl = context.application.bot_data.setdefault("blacklist", set())
    try:
        if txt.startswith("-"):
            bl.discard(int(txt[1:]))
            await update.message.reply_text("Удалён из чёрного списка.", reply_markup=admin_cancel_kb())
        else:
            bl.add(int(txt))
            await update.message.reply_text("Добавлен в чёрный список.", reply_markup=admin_cancel_kb())
    except Exception:
        await update.message.reply_text("Неверный формат.", reply_markup=admin_cancel_kb())
    return ADMIN_MENU

async def admin_shadow_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await _safe_edit_text(update.callback_query.message,
                          "Введите ID для теневого бана/снятия (аналогично блэклисту, -ID — снять):",
                          reply_markup=admin_cancel_kb())
    return ADMIN_SHADOW_WAIT_ID

async def admin_shadow_apply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    txt = (update.message.text or "").strip()
    sb = context.application.bot_data.setdefault("shadowban", set())
    try:
        if txt.startswith("-"):
            sb.discard(int(txt[1:])); await update.message.reply_text("Теневой бан снят.", reply_markup=admin_cancel_kb())
        else:
            sb.add(int(txt)); await update.message.reply_text("Теневой бан установлен.", reply_markup=admin_cancel_kb())
    except Exception:
        await update.message.reply_text("Неверный формат.", reply_markup=admin_cancel_kb())
    return ADMIN_MENU

# ===== CABINET =====
def _next_reset_str() -> str:
    tomorrow = datetime.now().date() + timedelta(days=1)
    return f"{tomorrow.strftime('%d.%m.%Y')} 00:00"

async def cabinet_open(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    used_rew = get_user_usage("rewriter", context)
    used_lit = get_user_usage("literature", context)
    left_rew = remaining_attempts("rewriter", context, uid)
    left_lit = remaining_attempts("literature", context, uid)
    tone = context.user_data.get("tone", "официальный")
    gost = context.user_data.get("gost", "универсальный")

    if has_active_subscription(context):
        exp = datetime.strptime(context.user_data["subscription_expires"], "%Y-%m-%d").strftime("%d.%m.%Y")
        sub_text = f"✅ Безлимит до {exp}"
        total = 0
    else:
        sub_text = "базовый доступ"
        total = get_daily_limit(context)

    pr1 = _progress_bar(used_rew if total else 0, total) if total else "∞"
    pr2 = _progress_bar(used_lit if total else 0, total) if total else "∞"

    text = (
        "👤 <b>Личный кабинет</b>\n\n"
        f"<b>ID:</b> <code>{uid}</code>\n"
        f"<b>Доступ:</b> {sub_text}\n"
        f"<b>Сброс лимитов:</b> { _next_reset_str() }\n\n"
        f"✍️ Рерайтер: {pr1} (ост: {left_rew})\n"
        f"📚 Литература: {pr2} (ост: {left_lit})\n\n"
        f"🗣 Тон рерайта: <b>{html.escape(tone)}</b>\n"
        f"📏 ГОСТ: <b>{html.escape(gost)}</b>"
    )
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit_text(update.callback_query.message, text, parse_mode="HTML",
                              reply_markup=cabinet_kb(), disable_web_page_preview=True)
    else:
        await update.effective_message.reply_html(text, reply_markup=cabinet_kb(), disable_web_page_preview=True)
    return CABINET_MENU

async def cabinet_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await cabinet_open(update, context)

async def cabinet_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.callback_query: 
        await update.callback_query.answer(); msg = update.callback_query.message
    else: 
        msg = update.effective_message
    await _safe_edit_text(msg, "🗂 <b>Последние 10 использований</b>\n\n"+_format_history(context,10),
                          parse_mode="HTML", reply_markup=cabinet_kb())
    return CABINET_MENU

async def cabinet_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    hist: List[Dict[str, Any]] = context.user_data.get("history", [])
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(["ts","feature","size"])
    for h in hist: w.writerow([h.get("ts",""), h.get("feature",""), h.get("size","")])
    byte = io.BytesIO(buf.getvalue().encode("utf-8")); byte.name="history.csv"
    await update.callback_query.answer()
    await update.callback_query.message.reply_document(InputFile(byte), caption="История (CSV)")
    return CABINET_MENU

# SETTINGS
async def settings_open(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await _safe_edit_text(update.callback_query.message, "⚙️ Настройки", reply_markup=settings_kb())
    return SETTINGS_MENU

async def settings_pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    data = update.callback_query.data
    if data == "set_tone":
        await _safe_edit_text(update.callback_query.message,
                              "Введите желаемый тон рерайта (напр.: официальный / научный / нейтральный):",
                              reply_markup=admin_cancel_kb())
        return SETTINGS_TONE_WAIT
    if data == "set_gost":
        await _safe_edit_text(update.callback_query.message,
                              "Введите ГОСТ по умолчанию (напр.: РАНХиГС 2021 / универсальный):",
                              reply_markup=admin_cancel_kb())
        return SETTINGS_GOST_WAIT
    return SETTINGS_MENU

async def settings_tone_save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["tone"] = (update.message.text or "официальный")[:50]
    await update.message.reply_text("✅ Тон сохранён.", reply_markup=back_menu_kb())
    return await cabinet_open(update, context)

async def settings_gost_save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["gost"] = (update.message.text or "универсальный")[:50]
    await update.message.reply_text("✅ ГОСТ сохранён.", reply_markup=back_menu_kb())
    return await cabinet_open(update, context)

# ===== GOST section =====
async def gost_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; await q.answer()
    kb = [
        [InlineKeyboardButton("Как оформить сноску?", callback_data="gost_footnote")],
        [InlineKeyboardButton("Как оформить список литературы?", callback_data="gost_references")],
        [InlineKeyboardButton("Общие требования (шрифт, отступы)", callback_data="gost_general")],
        [InlineKeyboardButton("⬅️ Назад в главное меню", callback_data="back_to_main_menu")],
    ]
    await _safe_edit_text(q.message,
                          "📋 **Консультант по ГОСТу**\n\nВыберите интересующий вас вопрос:",
                          reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    return GOST_MENU

async def gost_show_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query
    await q.answer()
    info_type = q.data
    text = ""
    if info_type == "gost_footnote":
        text = (
            "**📄 Оформление сносок (краткая шпаргалка)**\n\n"
            "Сноски ставятся внизу страницы. Нумерация сквозная по всему документу.\n\n"
            "**Пример (книга):**\n"
            "`¹ Иванов И. И. Название книги. – М.: Издательство, 2023. – С. 45.`\n\n"
            "**Пример (статья):**\n"
            "`² Петров П. П. Название статьи // Название журнала. – 2022. – № 2. – С. 12-15.`\n\n"
            "**Пример (интернет-ресурс):**\n"
            "`³ Название статьи [Электронный ресурс]. – Режим доступа: http://example.com (дата обращения: 23.08.2025).`"
        )
    elif info_type == "gost_references":
        text = (
            "**📚 Оформление списка литературы**\n\n"
            "Список составляется в алфавитном порядке по фамилии автора. Сначала идут русскоязычные источники, затем иностранные.\n\n"
            "**Пример (книга):**\n"
            "`Иванов, И. И. Название книги / И. И. Иванов. – Москва : Издательство, 2023. – 250 с.`\n\n"
            "**Пример (статья):**\n"
            "`Петров, П. П. Название статьи / П. П. Петров // Название журнала. – 2022. – № 2. – С. 12–15.`"
        )
    elif info_type == "gost_general":
        text = (
            "**⚙️ Общие требования к оформлению**\n\n"
            "• **Шрифт:** Times New Roman, 14 пт.\n"
            "• **Межстрочный интервал:** Полуторный (1,5).\n"
            "• **Выравнивание:** По ширине.\n"
            "• **Отступ первой строки (красная строка):** 1,25 см.\n"
            "• **Поля:** левое – 3 см, правое – 1 см, верхнее и нижнее – 2 см.\n\n"
            "*Внимание: всегда сверяйтесь с методическими указаниями вашего вуза, так как требования могут незначительно отличаться!*"
        )
    await _safe_edit_text(q.message, text,
                          reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад к вопросам", callback_data="gost_back")]]),
                          parse_mode="Markdown")
    return GOST_MENU

# ===== TOOLS =====
SIGNATURE_MD_V2 = (
    "\\-\\-\\-\n"
    "ℹ️ *Инструмент предоставлен автором Владиславом\\.*\n"
    "➡️ *[Свяжитесь со мной](https://t.me/V_L_A_D_IS_L_A_V)*"
)

async def rewriter_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; uid = update.effective_user.id; await q.answer()
    if _is_blacklisted(context.application, uid): await q.message.edit_text("🚫 Доступ ограничен."); return MAIN_MENU
    if _is_shadowbanned(context.application, uid): await q.message.edit_text("Сервис перегружен."); return MAIN_MENU

    if not is_admin(uid) and not has_active_subscription(context):
        limit = get_daily_limit(context)
        if get_user_usage("rewriter", context) >= limit:
            await _safe_edit_text(q.message,
                "🚫 <b>Дневной лимит исчерпан</b>\n\n"
                "Хотите продолжить без ожидания? Напишите: <a href='https://t.me/V_L_A_D_IS_L_A_V'>@V_L_A_D_IS_L_A_V</a>\n"
                f"Ваш ID: <code>{uid}</code>", parse_mode="HTML", reply_markup=contact_kb())
            return MAIN_MENU
    left = remaining_attempts("rewriter", context, uid)
    await _safe_edit_text(q.message,
        "✍️ *AI-Рерайтер*\n\nПришлите текст (до 1000 символов).\n\n"
        f"Доступно сегодня: *{left}*", parse_mode="Markdown", reply_markup=back_menu_kb())
    return REWRITER_TEXT_INPUT

async def literature_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; uid = update.effective_user.id; await q.answer()
    if _is_blacklisted(context.application, uid): await q.message.edit_text("🚫 Доступ ограничен."); return MAIN_MENU
    if _is_shadowbanned(context.application, uid): await q.message.edit_text("Сервис перегружен."); return MAIN_MENU

    if not is_admin(uid) and not has_active_subscription(context):
        limit = get_daily_limit(context)
        if get_user_usage("literature", context) >= limit:
            await _safe_edit_text(q.message,
                "🚫 <b>Дневной лимит исчерпан</b>\n\n"
                "Хотите продолжить без ожидания? Напишите: <a href='https://t.me/V_L_A_D_IS_L_A_V'>@V_L_A_D_IS_L_A_V</a>\n"
                f"Ваш ID: <code>{uid}</code>", parse_mode="HTML", reply_markup=contact_kb())
            return MAIN_MENU
    left = remaining_attempts("literature", context, uid)
    await _safe_edit_text(q.message,
        "📚 *Генератор списка литературы*\n\nНапишите тему.\n\n"
        f"Доступно сегодня: *{left}*", parse_mode="Markdown", reply_markup=back_menu_kb())
    return LITERATURE_TOPIC_INPUT

async def rewriter_process_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _touch_seen(update, context)
    uid = update.effective_user.id
    ok_rl, wait = _rate_limit_ok(context)
    if not ok_rl:
        await update.message.reply_text(f"⏳ Слишком часто. Попробуйте через {wait} сек.")
        return REWRITER_TEXT_INPUT

    if not is_admin(uid) and not has_active_subscription(context):
        limit = get_daily_limit(context)
        if get_user_usage("rewriter", context) >= limit:
            await update.message.reply_html(
                "🚫 <b>Дневной лимит исчерпан</b>\n\n"
                "Хотите продолжить без ожидания? Напишите: <a href='https://t.me/V_L_A_D_IS_L_A_V'>@V_L_A_D_IS_L_A_V</a>\n"
                f"Ваш ID: <code>{uid}</code>", reply_markup=contact_kb())
            return REWRITER_TEXT_INPUT

    user_text = (update.message.text or "")[:2000]
    context.user_data["last_request"] = {"feature":"rewriter","len":len(user_text),"ts":datetime.now().isoformat()}

    processing = await update.message.reply_text("⏳ Обрабатываю…")
    tone = context.user_data.get("tone", "официальный")
    prompt = (
        f"Перепиши текст в {tone} официально-деловом стиле; сохрани структуру и списки; "
        "улучши связность; числа в финчасти частично/полностью словами; без вступлений. \n\n"
        f"Исходный текст:\n\"\"\"\n{user_text}\n\"\"\""
    )
    txt = await call_gemini(prompt)

    success = not (txt.startswith("Ошибка") or "Не удалось получить корректный ответ" in txt)
    _record_ai_stat(context.application, success)
    if success and (not is_admin(uid)) and not has_active_subscription(context):
        increment_usage("rewriter", context)
    if success:
        _push_history(context, "rewriter", len(user_text))

    left = remaining_attempts("rewriter", context, uid)
    escaped = escape_markdown(txt, version=2)
    footer = f"\n\n*Доступно сегодня:* {left if left.startswith('∞') else f'*{left}*'}"
    full = f"*Готово\\! Вот перефразированный вариант:*\n\n{escaped}{footer}\n\n{SIGNATURE_MD_V2}"
    await _md2_send_chunks(processing, full, markup=back_menu_kb())
    return REWRITER_TEXT_INPUT

async def literature_process_topic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _touch_seen(update, context)
    uid = update.effective_user.id
    ok_rl, wait = _rate_limit_ok(context)
    if not ok_rl:
        await update.message.reply_text(f"⏳ Слишком часто. Попробуйте через {wait} сек.")
        return LITERATURE_TOPIC_INPUT

    if not is_admin(uid) and not has_active_subscription(context):
        limit = get_daily_limit(context)
        if get_user_usage("literature", context) >= limit:
            await update.message.reply_html(
                "🚫 <b>Дневной лимит исчерпан</b>\n\n"
                "Хотите продолжить без ожидания? Напишите: <a href='https://t.me/V_L_A_D_IS_L_A_V'>@V_L_A_D_IS_L_A_V</a>\n"
                f"Ваш ID: <code>{uid}</code>", reply_markup=contact_kb())
            return LITERATURE_TOPIC_INPUT

    topic = (update.message.text or "")[:500]
    context.user_data["last_request"] = {"feature":"literature","len":len(topic),"ts":datetime.now().isoformat()}
    processing = await update.message.reply_text("📚 Подбираю источники…")

    prompt = ("Ты — AI-эксперт-библиограф. Составь нумерованный список из 5–7 актуальных источников "
              "(книги, статьи) на русском языке; для каждого укажи автора, название, год.\n\n"
              f"Тема: \"{topic}\"")
    txt = await call_gemini(prompt)
    success = not _no_literature_found(txt)
    _record_ai_stat(context.application, success)

    if not success:
        await processing.edit_text(
            "😕 <b>Подходящие источники не нашлись</b>\n\n"
            "Сузьте тему (ключевые слова, годы/тип источника) или напишите мне — помогу вручную:\n"
            "<a href='https://t.me/V_L_A_D_IS_L_A_V'>@V_L_A_D_IS_L_A_V</a>",
            parse_mode="HTML", reply_markup=contact_kb(), disable_web_page_preview=True
        )
        return LITERATURE_TOPIC_INPUT

    if (not is_admin(uid)) and not has_active_subscription(context):
        increment_usage("literature", context)
    _push_history(context, "literature", len(topic))

    left = remaining_attempts("literature", context, uid)
    escaped = escape_markdown(txt, version=2)
    footer = f"\n\n*Доступно сегодня:* {left if left.startswith('∞') else f'*{left}*'}"
    full = f"*Готово\\! Вот рекомендуемый список литературы:*\n\n{escaped}{footer}\n\n{SIGNATURE_MD_V2}"
    await _md2_send_chunks(processing, full, markup=back_menu_kb())
    return LITERATURE_TOPIC_INPUT

# ===== CABINET - REFERRALS =====
async def cabinet_ref(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    uid = update.effective_user.id
    bot_user = await _ensure_bot_username(context)
    link = f"https://t.me/{bot_user}?start=ref_{uid}"
    refs = context.user_data.get("refs", [])
    cnt = len(refs) if isinstance(refs, list) else 0
    text = (
        "🎁 <b>Реферальная программа</b>\n\n"
        f"Ваша ссылка:\n<code>{html.escape(link)}</code>\n\n"
        f"За каждого друга — <b>+{REF_BONUS_DAYS} д.</b> безлимита.\n"
        f"Приглашённый получит <b>+{REF_WELCOME_ATTEMPTS}</b> доп. попытки на сегодня.\n\n"
        f"<b>Уже приглашено:</b> {cnt}\n"
        f"{'ID: ' + ', '.join(str(x) for x in refs) if cnt else ''}"
    )
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔗 Открыть ссылку", url=link)],
        [InlineKeyboardButton("⬅️ Назад", callback_data="cabinet")]
    ])
    await _safe_edit_text(update.callback_query.message, text, parse_mode="HTML",
                          reply_markup=kb, disable_web_page_preview=True)
    return CABINET_REF_MENU

# ===== CANCEL & ERRORS =====
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.effective_message.reply_text("Действие отменено.", reply_markup=back_menu_kb())
    return ConversationHandler.END

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled exception", exc_info=context.error)
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text("Упс! Ошибка. Попробуйте /start")
    except Exception:
        pass

# ===== MAIN =====
def main() -> None:
    if not BOT_TOKEN: raise RuntimeError("BOT_TOKEN не задан в .env/.evn")
    if not GEMINI_API_KEY: logger.warning("GEMINI_API_KEY не задан — AI-функции не работают.")

    try:
        persistence = PicklePersistence(filepath="bot_persistence")
    except TypeError:
        try: persistence = PicklePersistence(filename="bot_persistence")
        except TypeError: persistence = PicklePersistence("bot_persistence")

    app = Application.builder().token(BOT_TOKEN).persistence(persistence).build()
    app.add_error_handler(error_handler)

    # text commands
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("service", service_cmd))
    app.add_handler(CommandHandler("report", report_cmd))
    app.add_handler(CommandHandler("status", check_status))
    app.add_handler(CommandHandler("reset", reset_limit))
    app.add_handler(CommandHandler("addsub", add_subscription))
    app.add_handler(CommandHandler("delsub", del_subscription))

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                CallbackQueryHandler(rewriter_start, pattern="^rewriter$"),
                CallbackQueryHandler(literature_start, pattern="^literature$"),
                CallbackQueryHandler(gost_menu, pattern="^gost$"),
                CallbackQueryHandler(cabinet_open, pattern="^cabinet$"),
                CallbackQueryHandler(admin_panel_open, pattern="^admin_panel$"),
                CallbackQueryHandler(start, pattern="^back_to_main_menu$"),
            ],
            REWRITER_TEXT_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, rewriter_process_text),
                CallbackQueryHandler(start, pattern="^back_to_main_menu$"),
            ],
            LITERATURE_TOPIC_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, literature_process_topic),
                CallbackQueryHandler(start, pattern="^back_to_main_menu$"),
            ],
            GOST_MENU: [
                CallbackQueryHandler(gost_show_info, pattern="^gost_(footnote|references|general)$"),
                CallbackQueryHandler(gost_menu, pattern="^gost_back$"),
                CallbackQueryHandler(start, pattern="^back_to_main_menu$"),
            ],
            CABINET_MENU: [
                CallbackQueryHandler(cabinet_history, pattern="^cab_history$"),
                CallbackQueryHandler(cabinet_export, pattern="^cab_export$"),
                CallbackQueryHandler(cabinet_refresh, pattern="^cab_refresh$"),
                CallbackQueryHandler(settings_open, pattern="^cab_settings$"),
                CallbackQueryHandler(cabinet_ref, pattern="^cab_ref$"),
                CallbackQueryHandler(start, pattern="^back_to_main_menu$"),
            ],
            CAPTCHA_WAIT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, captcha_check),
                CommandHandler("start", start),
            ],
            SETTINGS_MENU: [
                CallbackQueryHandler(settings_pick, pattern="^(set_tone|set_gost)$"),
                CallbackQueryHandler(cabinet_open, pattern="^cabinet$"),
                CallbackQueryHandler(start, pattern="^back_to_main_menu$"),
            ],
            SETTINGS_TONE_WAIT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, settings_tone_save),
            ],
            SETTINGS_GOST_WAIT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, settings_gost_save),
            ],
            ADMIN_MENU: [
                CallbackQueryHandler(admin_reset_start, pattern="^admin_reset$"),
                CallbackQueryHandler(admin_addsub_start, pattern="^admin_addsub$"),
                CallbackQueryHandler(admin_delsub_start, pattern="^admin_delsub$"),
                CallbackQueryHandler(admin_status_start, pattern="^admin_status$"),
                CallbackQueryHandler(admin_search_start, pattern="^admin_search$"),
                CallbackQueryHandler(admin_tags_start, pattern="^admin_tags$"),
                CallbackQueryHandler(admin_broadcast_start, pattern="^admin_broadcast$"),
                CallbackQueryHandler(admin_export_csv, pattern="^admin_export$"),
                CallbackQueryHandler(admin_blacklist_start, pattern="^admin_blacklist$"),
                CallbackQueryHandler(admin_shadow_start, pattern="^admin_shadow$"),
                CallbackQueryHandler(admin_setlimit_start, pattern="^admin_setlimit$"),
                CallbackQueryHandler(admin_panel_open, pattern="^admin_panel$"),
                CallbackQueryHandler(start, pattern="^back_to_main_menu$"),
            ],
            ADMIN_RESET_WAIT_ID: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_reset_receive_id),
            ],
            ADMIN_ADDSUB_WAIT_ID: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_addsub_receive_id),
            ],
            ADMIN_ADDSUB_WAIT_DAYS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_addsub_receive_days),
            ],
            ADMIN_STATUS_WAIT_ID: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_status_receive_id),
            ],
            ADMIN_DELSUB_WAIT_ID: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_delsub_receive_id),
            ],
            ADMIN_SEARCH_WAIT_QUERY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_search_do),
            ],
            ADMIN_TAGS_WAIT_ID: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_tags_id),
            ],
            ADMIN_TAGS_WAIT_VALUE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_tags_set),
            ],
            ADMIN_BROADCAST_SEGMENT: [
                CallbackQueryHandler(admin_broadcast_pick, pattern="^bseg_(all|free|subs|inactive)$"),
                CallbackQueryHandler(admin_panel_open, pattern="^admin_panel$"),
            ],
            ADMIN_BROADCAST_WAIT_TEXT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_broadcast_send),
            ],
            ADMIN_SETLIMIT_WAIT_ID: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_setlimit_id),
            ],
            ADMIN_SETLIMIT_WAIT_VALUES: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_setlimit_values),
            ],
            ADMIN_BLACKLIST_WAIT_ID: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_blacklist_apply),
            ],
            ADMIN_SHADOW_WAIT_ID: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_shadow_apply),
            ],
            CABINET_REF_MENU: [
                CallbackQueryHandler(cabinet_open, pattern="^cabinet$"),
                CallbackQueryHandler(start, pattern="^back_to_main_menu$"),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel), CommandHandler("start", start)],
        allow_reentry=True, persistent=True, name="main_conversation"
    )

    app.add_handler(conv)
    logger.info("Бот запущен.")
    app.run_polling()

if __name__ == "__main__":
    main()
