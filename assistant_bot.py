# -*- coding: utf-8 -*-

import os, io, csv, time, html, random
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, InputFile
from telegram.ext import (
    Application, CommandHandler, ContextTypes, ConversationHandler,
    MessageHandler, CallbackQueryHandler, filters, PicklePersistence
)

# ===== .env / .evn =====
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(filename=".env", raise_error_if_not_found=False))
    if os.path.exists(".evn"):  # поддержка опечатки
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

# Рефералки
REF_BONUS_DAYS       = int(os.getenv("REF_BONUS_DAYS", "1"))
REF_WELCOME_ATTEMPTS = int(os.getenv("REF_WELCOME_ATTEMPTS", "2"))

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
    ADMIN_METRICS_MENU
) = range(25)

# ===== HELPERS: dates/roles =====
def _today() -> str: return datetime.now().strftime("%Y-%m-%d")
def _now_hms() -> str: return datetime.now().strftime("%H:%M:%S")
def _ymd(dt: datetime) -> str: return dt.strftime("%Y-%m-%d")
def is_admin(uid: int) -> bool: return ADMIN_USER_ID is not None and uid == ADMIN_USER_ID

def has_active_subscription(context: ContextTypes.DEFAULT_TYPE) -> bool:
    exp = context.user_data.get("subscription_expires")
    if not exp: return False
    try:
        return datetime.strptime(exp, "%Y-%m-%d").date() >= datetime.now().date()
    except Exception:
        return False

# ===== LIMITS =====
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
    # --- аналитика по функциям ---
    app = context.application
    stat = app.bot_data.setdefault("feature_usage_today", {})
    day = _today()
    day_map = stat.setdefault(day, {"rewriter": 0, "literature": 0})
    day_map[feature] = day_map.get(feature, 0) + 1
    return d["count"]

def remaining_attempts(feature: str, context: ContextTypes.DEFAULT_TYPE, uid: int) -> str:
    if is_admin(uid) or has_active_subscription(context):
        return "∞ (Безлимит)"
    return str(max(0, FREE_LIMIT - get_user_usage(feature, context)))

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
    subset = stats[-n:] if n > 0 else stats[:]
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
    # аналитика активности
    _track_active(context.application, update.effective_user.id)
    _ensure_first_seen(context)

def _is_blacklisted(app: Application, uid: int) -> bool:
    bl = app.bot_data.get("blacklist", set())
    return uid in bl

def _is_shadowbanned(app: Application, uid: int) -> bool:
    sb = app.bot_data.get("shadowban", set())
    return uid in sb

def _rate_limit_ok(context: ContextTypes.DEFAULT_TYPE) -> Tuple[bool, int]:
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
    a, b = random.randint(2, 9), random.randint(2, 9)
    context.user_data["captcha_answer"] = str(a + b)
    return f"Проверка: сколько будет {a} + {b}? Отправьте ответ числом."

# ===== Analytics core =====
def _track_active(app: Application, uid: int) -> None:
    bd = app.bot_data
    day = _today()
    dau = bd.setdefault("dau", {})
    day_set = dau.setdefault(day, set())
    day_set.add(uid)
    allu = bd.setdefault("all_users", set())
    allu.add(uid)

def _count_dau(bd: dict, days: int) -> int:
    today = datetime.now().date()
    dau: Dict[str, set] = bd.get("dau", {})
    seen: set = set()
    for i in range(days):
        d = _ymd(datetime.combine(today - timedelta(days=i), datetime.min.time()))
        if d in dau:
            seen |= set(dau[d])
    return len(seen)

def _count_new_users(app: Application, days: int) -> int:
    cutoff = datetime.now().date() - timedelta(days=days-1)
    c = 0
    for _, ud in app.user_data.items():
        fs = ud.get("first_seen")
        try:
            d = datetime.strptime(fs, "%Y-%m-%d").date()
            if d >= cutoff:
                c += 1
        except Exception:
            continue
    return c

def _feature_usage_today(bd: dict) -> Dict[str, int]:
    fm = bd.get("feature_usage_today", {}).get(_today(), {})
    return {"rewriter": int(fm.get("rewriter", 0)), "literature": int(fm.get("literature", 0))}

# ===== Gemini =====
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

async def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "Ошибка: API-ключ для нейросети не настроен."
    api_url = (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    try:
        from httpx import AsyncClient, HTTPStatusError
        async with AsyncClient(timeout=60.0) as client:
            r = await client.post(api_url, json=payload)
            try:
                r.raise_for_status()
            except HTTPStatusError:
                try:
                    err = r.json().get("error", {})
                    msg = err.get("message") or str(err)
                except Exception:
                    msg = r.text
                return f"Произошла ошибка при обращении к нейросети: {r.status_code} {msg}"
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
        [InlineKeyboardButton("🎚 Задать лимиты", callback_data="admin_setlimit"),
         InlineKeyboardButton("📈 Метрики", callback_data="admin_metrics")],
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

def admin_metrics_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Обновить", callback_data="admin_metrics_refresh"),
         InlineKeyboardButton("📥 Экспорт (CSV)", callback_data="admin_metrics_export")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="admin_panel")],
    ])

# ===== Utils (Markdown, не V2) =====
TG_MD_LIMIT = 3800
def _chunk_md(text: str, limit: int = TG_MD_LIMIT) -> List[str]:
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

async def _md_send_chunks(msg, text: str, markup=None):
    parts = _chunk_md(text, TG_MD_LIMIT)
    if len(parts) == 1:
        await msg.edit_text(parts[0], parse_mode="Markdown",
                            disable_web_page_preview=True, reply_markup=markup)
    else:
        await msg.edit_text(parts[0], parse_mode="Markdown", disable_web_page_preview=True)
        for p in parts[1:-1]:
            await msg.reply_text(p, parse_mode="Markdown", disable_web_page_preview=True)
        await msg.reply_text(parts[-1], parse_mode="Markdown",
                             disable_web_page_preview=True, reply_markup=markup)

def _progress_bar(used: int, total: int, width: int = 20) -> str:
    if total <= 0: return "░"*width + " ∞"
    used = max(0, min(total, used))
    fill = int(width * used / total)
    return "█"*fill + "░"*(width - fill) + f" {used}/{total}"

def _ensure_first_seen(context: ContextTypes.DEFAULT_TYPE) -> None:
    if "first_seen" not in context.user_data:
        context.user_data["first_seen"] = _today()

# ===== /start (+ рефералки) =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        args = context.args if hasattr(context, "args") else []
        if args:
            ref_from = int(args[0])
            me = update.effective_user.id
            if me != ref_from and not context.user_data.get("is_ref"):
                ref_ud = context.application.user_data.get(ref_from)
                if isinstance(ref_ud, dict):
                    refs = ref_ud.get("referrals", set())
                    if not isinstance(refs, set):
                        try: refs = set(refs) if refs else set()
                        except Exception: refs = set()
                    if me not in refs:
                        refs.add(me)
                        ref_ud["referrals"] = refs
                        context.user_data["is_ref"] = True
                        if REF_BONUS_DAYS > 0:
                            start_date = datetime.now().date()
                            current_exp_str = ref_ud.get("subscription_expires")
                            if current_exp_str:
                                try:
                                    current_exp_date = datetime.strptime(current_exp_str, "%Y-%m-%d").date()
                                    if current_exp_date > start_date:
                                        start_date = current_exp_date
                                except (ValueError, TypeError):
                                    pass
                            until = start_date + timedelta(days=REF_BONUS_DAYS)
                            ref_ud["subscription_expires"] = until.strftime("%Y-%m-%d")
                            try:
                                await context.bot.send_message(
                                    chat_id=ref_from,
                                    text=f"🎁 Новый реферал! Вам продлён безлимит до {until.strftime('%d.%m.%Y')}."
                                )
                            except Exception: pass
                        if REF_WELCOME_ATTEMPTS > 0:
                            usage = context.user_data.setdefault("usage", {})
                            for feat in ("rewriter", "literature"):
                                d = usage.get(feat, {"count": 0, "date": _today()})
                                if d.get("date") != _today():
                                    d = {"count": 0, "date": _today()}
                                d["count"] = max(0, d.get("count", 0) - REF_WELCOME_ATTEMPTS)
                                usage[feat] = d
    except Exception:
        pass

    _touch_seen(update, context)
    uid = update.effective_user.id

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
        await update.callback_query.message.edit_text(text, parse_mode="HTML", reply_markup=main_menu_kb(uid))
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
        "• /service — статус ИИ\n"
        "• /metrics — общая статистика (админ)\n\n"
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

# ===== add_subscription =====
async def add_subscription(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """(Только для админа) Добавляет или продлевает подписку пользователю на N дней."""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Только администратор.")
        return
    try:
        target_id = int(context.args[0])
        days = int(context.args[1])
        assert days > 0
    except Exception:
        await update.message.reply_text("Используйте: /addsub <user_id> <days>")
        return

    ud = context.application.user_data.get(target_id)
    if not ud:
        await update.message.reply_text("Пользователь не найден.")
        return

    start_date = datetime.now().date()
    current_exp_str = ud.get("subscription_expires")
    if current_exp_str:
        try:
            current_exp_date = datetime.strptime(current_exp_str, "%Y-%m-%d").date()
            if current_exp_date > start_date:
                start_date = current_exp_date
        except (ValueError, TypeError):
            pass

    new_exp_date = start_date + timedelta(days=days)
    new_exp_str = new_exp_date.strftime("%Y-%m-%d")
    ud["subscription_expires"] = new_exp_str

    new_exp_human = new_exp_date.strftime('%d.%m.%Y')
    await update.message.reply_text(f"✅ Подписка для {target_id} продлена/установлена до {new_exp_human}")

    try:
        await context.bot.send_message(
            chat_id=target_id,
            text=f"🎉 Ваша подписка продлена! Безлимитный доступ теперь действует до {new_exp_human} включительно."
        )
    except Exception:
        pass

async def del_subscription(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Только администратор.")
        return
    try: target_id = int(context.args[0])
    except Exception: await update.message.reply_text("Используйте: /delsub <user_id>"); return
    ud = context.application.user_data.get(target_id)
    if not ud: await update.message.reply_text("Пользователь не найден."); return
    had = "subscription_expires" in ud
    ud.pop("subscription_expires", None)
    await update.message.reply_text("🛑 Подписка отключена." if had else "У пользователя нет активной подписки.")

    try:
        await context.bot.send_message(chat_id=target_id, text="🛑 Ваша подписка отключена.")
    except Exception:
        pass

async def check_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    if has_active_subscription(context):
        exp = datetime.strptime(context.user_data.get("subscription_expires"), "%Y-%m-%d").strftime("%d.%м.%Y")
        await update.message.reply_html(f"<b>Статус:</b> ✅ Безлимит до {exp}")
    else:
        r = remaining_attempts("rewriter", context, uid)
        l = remaining_attempts("literature", context, uid)
        await update.message.reply_html(
            "<b>Статус:</b> базовый доступ\n"
            f"• Рерайтер: {r} из {FREE_LIMIT}\n"
            f"• Литература: {l} из {FREE_LIMIT}"
        )

# ===== ADMIN PANEL (callbacks + flows) =====
async def admin_panel_open(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_admin(update.effective_user.id):
        if update.callback_query: await update.callback_query.answer("Только администратор.", show_alert=True)
        else: await update.message.reply_text("Только администратор.")
        return MAIN_MENU
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.edit_text("⚙️ <b>Админ-панель</b>", parse_mode="HTML", reply_markup=admin_menu_kb())
    else:
        await update.message.reply_html("⚙️ <b>Админ-панель</b>", reply_markup=admin_menu_kb())
    return ADMIN_MENU

# reset
async def admin_reset_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await update.callback_query.message.edit_text("Введите <b>ID</b> для сброса дневных лимитов:", parse_mode="HTML",
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
    await update.message.reply_text("✅ Сброшено.", reply_markup=admin_cancel_kb()); return ADMIN_MENU

# addsub
async def admin_addsub_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await update.callback_query.message.edit_text("Введите <b>ID</b> для выдачи подписки:", parse_mode="HTML",
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
    try: days = int((update.message.text or "").strip()); assert days > 0
    except Exception:
        await update.message.reply_text("Неверное число. Введите положительное целое.", reply_markup=admin_cancel_kb()); return ADMIN_ADDSUB_WAIT_DAYS
    target_id = context.user_data.get("admin_addsub_target")
    if target_id is None:
        await update.message.reply_text("Сессия сброшена. Повторите.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    ud = context.application.user_data.get(target_id)
    if not ud:
        await update.message.reply_text("Пользователь не найден.", reply_markup=admin_cancel_kb()); return ADMIN_MENU

    start_date = datetime.now().date()
    current_exp_str = ud.get("subscription_expires")
    if current_exp_str:
        try:
            current_exp_date = datetime.strptime(current_exp_str, "%Y-%m-%d").date()
            if current_exp_date > start_date:
                start_date = current_exp_date
        except (ValueError, TypeError):
            pass

    exp = (start_date + timedelta(days=days)).strftime("%Y-%m-%d")
    ud["subscription_expires"]=exp
    await update.message.reply_text("✅ Подписка выдана.", reply_markup=admin_cancel_kb())
    try:
        await context.bot.send_message(chat_id=target_id, text=f"🎉 Вам активировали безлимит на {days} дней (до {datetime.strptime(exp,'%Y-%m-%d').strftime('%d.%m.%Y')}).")
    except Exception:
        pass
    context.user_data.pop("admin_addsub_target", None)
    return ADMIN_MENU

# delsub
async def admin_delsub_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await update.callback_query.message.edit_text("Введите <b>ID</b> для отмены подписки:", parse_mode="HTML",
                                                 reply_markup=admin_cancel_kb())
    return ADMIN_DELSUB_WAIT_ID

async def admin_delsub_receive_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try: target_id = int((update.message.text or "").strip())
    except Exception:
        await update.message.reply_html("Неверный ID.", reply_markup=admin_cancel_kb()); return ADMIN_DELSUB_WAIT_ID
    ud = context.application.user_data.get(target_id)
    if not ud:
        await update.message.reply_text("Пользователь не найден.", reply_markup=admin_cancel_kb()); return ADMIN_MENU
    had = "subscription_expires" in ud
    ud.pop("subscription_expires", None)
    await update.message.reply_text("🛑 Подписка отключена." if had else "У пользователя не было активной подписки.", reply_markup=admin_cancel_kb()); return ADMIN_MENU

# status
async def admin_status_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await update.callback_query.message.edit_text("Введите <b>ID</b> для просмотра статуса:",
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
    sub_line = f"Подписка: до {datetime.strptime(exp,'%Y-%m-%d').strftime('%d.%м.%Y')}" if exp else "Подписка: нет"
    usage = ud.get("usage", {})
    rew = usage.get("rewriter", {"count": 0, "date": _today()})
    lit = usage.get("literature", {"count": 0, "date": _today()})
    tags = ", ".join(ud.get("tags", [])) or "—"
    refs = len(ud.get("referrals", set())) if isinstance(ud.get("referrals", set()), set) else 0
    await update.message.reply_html(
        f"👤 <b>{target_id}</b>\n{sub_line}\n"
        f"Рерайтер: {rew.get('count',0)} ({rew.get('date','-')})\n"
        f"Литература: {lit.get('count',0)} ({lit.get('date','-')})\n"
        f"Теги: {html.escape(tags)}\n"
        f"Рефералов: {refs}",
        reply_markup=admin_cancel_kb()
    )
    return ADMIN_MENU

# search
async def admin_search_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await update.callback_query.message.edit_text("Введите ID или часть юзернейма:", reply_markup=admin_cancel_kb())
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
    await update.callback_query.message.edit_text("Введите ID пользователя:", reply_markup=admin_cancel_kb())
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
        [InlineKeyboardButton("🔇 Все (тихо)", callback_data="bseg_all_silent"),
         InlineKeyboardButton("🔇 Free (тихо)", callback_data="bseg_free_silent"),
         InlineKeyboardButton("🔇 Pro (тихо)", callback_data="bseg_subs_silent")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="admin_panel")]
    ])
    await update.callback_query.message.edit_text("Выберите сегмент рассылки:", reply_markup=kb)
    return ADMIN_BROADCAST_SEGMENT

async def admin_broadcast_pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    data = update.callback_query.data
    is_silent = data.endswith("_silent")
    segment = data.replace("_silent", "")
    context.user_data["b_segment"] = segment
    context.user_data["b_silent"] = is_silent

    mode_text = " (тихая рассылка)" if is_silent else ""
    await update.callback_query.message.edit_text(f"Введите текст рассылки{mode_text}:", reply_markup=admin_cancel_kb())
    return ADMIN_BROADCAST_WAIT_TEXT

async def admin_broadcast_send(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    seg = context.user_data.get("b_segment"); txt = (update.message.text or "").strip()
    is_silent = context.user_data.get("b_silent", False)

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
            except Exception: pass
        if seg == "bseg_subs" and not data.get("subscription_expires"):
            continue
        if seg == "bseg_inactive":
            ls = data.get("last_seen")
            if ls:
                try:
                    if datetime.strptime(ls, "%Y-%m-%d %H:%M:%S") >= cutoff:
                        continue
                except Exception: pass
        try:
            await context.bot.send_message(chat_id=uid, text=txt, disable_notification=is_silent)
            sent += 1
            time.sleep(0.1)
        except Exception:
            continue
    await update.message.reply_text(f"Готово. Отправлено: {sent}", reply_markup=admin_cancel_kb())
    context.user_data.pop("b_segment", None)
    context.user_data.pop("b_silent", None)
    return ADMIN_MENU

# export users CSV
async def admin_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id","first_seen","last_seen","username","subscription_expires","tags","referrals_count"])
    for uid, d in context.application.user_data.items():
        w.writerow([uid, d.get("first_seen",""), d.get("last_seen",""),
                    d.get("last_username",""), d.get("subscription_expires",""),
                    "|".join(d.get("tags", [])),
                    len(d.get("referrals", set())) if isinstance(d.get("referrals", set()), set) else 0])
    byte = io.BytesIO(buf.getvalue().encode("utf-8"))
    byte.name = "users_export.csv"
    await update.callback_query.answer()
    await update.callback_query.message.reply_document(InputFile(byte), caption="Экспорт пользователей (CSV)")
    return ADMIN_MENU

# set limits
async def admin_setlimit_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await update.callback_query.message.edit_text("Введите ID пользователя:", reply_markup=admin_cancel_kb())
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
    u["rewriter"] = {"count": max(0, c1), "date": today}
    u["literature"] = {"count": max(0, c2), "date": today}
    await update.message.reply_text("✅ Установлено.", reply_markup=admin_cancel_kb())
    context.user_data.pop("limit_target", None); return ADMIN_MENU

# blacklist / shadowban
async def admin_blacklist_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await update.callback_query.message.edit_text("Введите ID для добавления/удаления из чёрного списка (для удаления добавьте минус перед ID):",
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
    await update.callback_query.message.edit_text("Введите ID для теневого бана/снятия (аналогично блэклисту, -ID — снять):",
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

# ===== METRICS =====
def _metrics_text(app: Application) -> str:
    bd = app.bot_data
    total_unique = len(bd.get("all_users", set()))
    dau_1 = _count_dau(bd, 1)
    dau_7 = _count_dau(bd, 7)
    dau_30 = _count_dau(bd, 30)
    new_1  = _count_new_users(app, 1)
    new_7  = _count_new_users(app, 7)
    new_30 = _count_new_users(app, 30)
    feats = _feature_usage_today(bd)
    ts = f"{_today()} {_now_hms()}"
    return (
        f"📈 <b>Метрики</b> (обновлено: {ts})\n\n"
        f"• Всего уникальных: <b>{total_unique}</b>\n"
        f"• Активны сегодня (DAU): <b>{dau_1}</b>\n"
        f"• Активны 7 дней: <b>{dau_7}</b>\n"
        f"• Активны 30 дней: <b>{dau_30}</b>\n\n"
        f"• Новых сегодня: <b>{new_1}</b>\n"
        f"• Новых за 7 дней: <b>{new_7}</b>\n"
        f"• Новых за 30 дней: <b>{new_30}</b>\n\n"
        f"• Использование сегодня:\n"
        f"  ├─ Рерайтер: <b>{feats.get('rewriter',0)}</b>\n"
        f"  └─ Литература: <b>{feats.get('literature',0)}</b>"
    )

async def admin_metrics_open(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_admin(update.effective_user.id):
        await update.callback_query.answer("Только администратор.", show_alert=True)
        return ADMIN_MENU
    await update.callback_query.answer()
    text = _metrics_text(context.application)
    await update.callback_query.message.edit_text(text, parse_mode="HTML", reply_markup=admin_metrics_kb(), disable_web_page_preview=True)
    return ADMIN_METRICS_MENU

async def admin_metrics_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_admin(update.effective_user.id):
        await update.callback_query.answer("Только администратор.", show_alert=True)
        return ADMIN_MENU
    await update.callback_query.answer()
    text = _metrics_text(context.application) + "\u2060"
    await update.callback_query.message.edit_text(text, parse_mode="HTML", reply_markup=admin_metrics_kb(), disable_web_page_preview=True)
    return ADMIN_METRICS_MENU

async def admin_metrics_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_admin(update.effective_user.id):
        await update.callback_query.answer("Только администратор.", show_alert=True)
        return ADMIN_MENU
    await update.callback_query.answer()
    bd = context.application.bot_data
    dau = bd.get("dau", {})
    feat = bd.get("feature_usage_today", {})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date","dau","rewriter_uses","literature_uses","new_users"])
    for i in range(30):
        d = datetime.now().date() - timedelta(days=i)
        key = d.strftime("%Y-%m-%d")
        dau_cnt = len(dau.get(key, set()))
        f = feat.get(key, {})
        rew = int(f.get("rewriter", 0))
        lit = int(f.get("literature", 0))
        new_cnt = 0
        for _, ud in context.application.user_data.items():
            if ud.get("first_seen") == key:
                new_cnt += 1
        w.writerow([key, dau_cnt, rew, lit, new_cnt])
    byte = io.BytesIO(buf.getvalue().encode("utf-8"))
    byte.name = "metrics_30d.csv"
    await update.callback_query.message.reply_document(InputFile(byte), caption="Метрики за 30 дней (CSV)")
    return ADMIN_METRICS_MENU

async def metrics_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Команда доступна только администратору.")
        return
    text = _metrics_text(context.application)
    await update.message.reply_html(text)

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
        total = FREE_LIMIT

    pr1 = _progress_bar(used_rew if total else 0, total) if total else "∞"
    pr2 = _progress_bar(used_lit if total else 0, total) if total else "∞"

    text = (
        "👤 <b>Личный кабинет</b>\n\n"
        f"<b>ID:</b> <code>{uid}</code>\n"
        f"<b>Доступ:</b> {sub_text}\n"
        f"<b>Сброс лимитов:</b> { _next_reset_str() }\n\n"
        f"✍️ Рерайтер: {pr1} (ост: {html.escape(left_rew)})\n"
        f"📚 Литература: {pr2} (ост: {html.escape(left_lit)})\n\n"
        f"🗣 Тон рерайта: <b>{html.escape(tone)}</b>\n"
        f"📏 ГОСТ: <b>{html.escape(gost)}</b>"
    )
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.edit_text(text, parse_mode="HTML",
                                                     reply_markup=cabinet_kb(), disable_web_page_preview=True)
    else:
        await update.effective_message.reply_html(text, reply_markup=cabinet_kb(), disable_web_page_preview=True)
    return CABINET_MENU

async def cabinet_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await cabinet_open(update, context)

async def cabinet_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.callback_query: await update.callback_query.answer(); msg = update.callback_query.message
    else: msg = update.effective_message
    await msg.edit_text("🗂 <b>Последние 10 использований</b>\n\n"+_format_history(context,10),
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
    await update.callback_query.message.edit_text("⚙️ Настройки", reply_markup=settings_kb())
    return SETTINGS_MENU

async def settings_pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    data = update.callback_query.data
    if data == "set_tone":
        await update.callback_query.message.edit_text("Введите желаемый тон рерайта (напр.: официальный / научный / нейтральный):",
                                                      reply_markup=admin_cancel_kb())
        return SETTINGS_TONE_WAIT
    if data == "set_gost":
        await update.callback_query.message.edit_text("Введите ГОСТ по умолчанию (напр.: РАНХиГС 2021 / универсальный):",
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
    await q.edit_message_text(
        "📋 **Консультант по ГОСТу**\n\nВыберите интересующий вас вопрос:",
        reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown"
    )
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
            "*Всегда сверяйтесь с методичками вашего вуза — требования могут отличаться.*"
        )
    await q.edit_message_text(
        text=text,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад к вопросам", callback_data="gost_back")]]),
        parse_mode="Markdown"
    )
    return GOST_MENU

# ===== TOOLS =====
SIGNATURE_MD = (
    "---\n"
    "ℹ️ *Инструмент предоставлен автором Владиславом.*\n"
    "➡️ *[Свяжитесь со мной](https://t.me/V_L_A_D_IS_L_A_V)*"
)

async def rewriter_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; uid = update.effective_user.id; await q.answer()
    if _is_blacklisted(context.application, uid): await q.message.edit_text("🚫 Доступ ограничен."); return MAIN_MENU
    if _is_shadowbanned(context.application, uid): await q.message.edit_text("Сервис перегружен."); return MAIN_MENU

    if not is_admin(uid) and not has_active_subscription(context):
        if get_user_usage("rewriter", context) >= FREE_LIMIT:
            await q.edit_message_text(
                ("🚫 <b>Дневной лимит исчерпан</b>\n\n"
                 "Хотите продолжить без ожидания? Напишите: <a href='https://t.me/V_L_A_D_IS_L_A_V'>@V_L_A_D_IS_L_A_V</a>\n"
                 f"Ваш ID: <code>{uid}</code>"), parse_mode="HTML", reply_markup=contact_kb())
            return MAIN_MENU
    left = remaining_attempts("rewriter", context, uid)
    await q.edit_message_text(
        ("✍️ *AI-Рерайтер*\n\nПришлите текст (до 1000 символов).\n\n"
         f"Доступно сегодня: *{left}*"),
        parse_mode="Markdown", reply_markup=back_menu_kb())
    return REWRITER_TEXT_INPUT

async def literature_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; uid = update.effective_user.id; await q.answer()
    if _is_blacklisted(context.application, uid): await q.message.edit_text("🚫 Доступ ограничен."); return MAIN_MENU
    if _is_shadowbanned(context.application, uid): await q.message.edit_text("Сервис перегружен."); return MAIN_MENU

    if not is_admin(uid) and not has_active_subscription(context):
        if get_user_usage("literature", context) >= FREE_LIMIT:
            await q.edit_message_text(
                ("🚫 <b>Дневной лимит исчерпан</b>\n\n"
                 "Хотите продолжить без ожидания? Напишите: <a href='https://t.me/V_L_A_D_IS_L_A_V'>@V_L_A_D_IS_L_A_V</a>\n"
                 f"Ваш ID: <code>{uid}</code>"), parse_mode="HTML", reply_markup=contact_kb())
            return MAIN_MENU
    left = remaining_attempts("literature", context, uid)
    await q.edit_message_text(("📚 *Генератор списка литературы*\n\nНапишите тему.\n\n"
                               f"Доступно сегодня: *{left}*"),
                              parse_mode="Markdown", reply_markup=back_menu_kb())
    return LITERATURE_TOPIC_INPUT

async def rewriter_process_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _touch_seen(update, context)
    uid = update.effective_user.id
    ok_rl, wait = _rate_limit_ok(context)
    if not ok_rl:
        await update.message.reply_text(f"⏳ Слишком часто. Попробуйте через {wait} сек.")
        return REWRITER_TEXT_INPUT

    if not is_admin(uid) and not has_active_subscription(context):
        if get_user_usage("rewriter", context) >= FREE_LIMIT:
            await update.message.reply_html(
                ("🚫 <b>Дневной лимит исчерпан</b>\n\n"
                 "Хотите продолжить без ожидания? Напишите: <a href='https://t.me/V_L_A_D_IS_L_A_V'>@V_L_A_D_IS_L_A_V</a>\n"
                 f"Ваш ID: <code>{uid}</code>"), reply_markup=contact_kb())
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
    footer = f"\n\n*Доступно сегодня:* {left}"
    full = f"*Готово! Вот перефразированный вариант:*\n\n{txt}{footer}\n\n{SIGNATURE_MD}"
    await _md_send_chunks(processing, full, markup=back_menu_kb())
    return REWRITER_TEXT_INPUT

async def literature_process_topic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _touch_seen(update, context)
    uid = update.effective_user.id
    ok_rl, wait = _rate_limit_ok(context)
    if not ok_rl:
        await update.message.reply_text(f"⏳ Слишком часто. Попробуйте через {wait} сек.")
        return LITERATURE_TOPIC_INPUT

    if not is_admin(uid) and not has_active_subscription(context):
        if get_user_usage("literature", context) >= FREE_LIMIT:
            await update.message.reply_html(
                ("🚫 <b>Дневной лимит исчерпан</б>\n\n"
                 "Хотите продолжить без ожидания? Напишите: <a href='https://t.me/V_L_A_D_IS_L_A_V'>@V_L_A_D_IS_L_A_V</a>\n"
                 f"Ваш ID: <code>{uid}</code>"), reply_markup=contact_kb())
            return LITERATURE_TOPIC_INPUT

    topic = (update.message.text or "")[:500]
    context.user_data["last_request"] = {"feature":"literature","len":len(topic),"ts":datetime.now().isoformat()}
    processing = await update.message.reply_text("📚 Подбираю источники…")

    # Новый промпт: нумерация + комментарий
    prompt = (
        "Ты — AI-эксперт-библиограф. Составь НУМЕРОВАННЫЙ список 5–7 актуальных источников на русском.\n"
        "Для каждого укажи точную библиографию и КОРОТКИЙ комментарий (1–2 предложения), зачем источник полезен.\n"
        "Строгий формат:\n"
        "1. **Автор(ы), И. О.** (Год). *Название*. Город: Издательство / Журнал, №, страницы.\n"
        "**Комментарий:** …\n"
        "2. ...\n\n"
        f"Тема: «{topic}»\n"
        "Не добавляй ссылки/URL и лишние прелюдии/итоги."
    )
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
    footer = f"\n\n*Доступно сегодня:* {left}"
    full = f"*Готово! Вот рекомендуемый список литературы:*\n\n{txt}{footer}\n\n{SIGNATURE_MD}"
    await _md_send_chunks(processing, full, markup=back_menu_kb())
    return LITERATURE_TOPIC_INPUT

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
    app.add_handler(CommandHandler("metrics", metrics_cmd))
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
            SETTINGS_TONE_WAIT: [ MessageHandler(filters.TEXT & ~filters.COMMAND, settings_tone_save) ],
            SETTINGS_GOST_WAIT: [ MessageHandler(filters.TEXT & ~filters.COMMAND, settings_gost_save) ],
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
                CallbackQueryHandler(admin_metrics_open, pattern="^admin_metrics$"),
                CallbackQueryHandler(admin_panel_open, pattern="^admin_panel$"),
                CallbackQueryHandler(start, pattern="^back_to_main_menu$"),
            ],
            ADMIN_RESET_WAIT_ID: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_reset_receive_id) ],
            ADMIN_ADDSUB_WAIT_ID: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_addsub_receive_id) ],
            ADMIN_ADDSUB_WAIT_DAYS: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_addsub_receive_days) ],
            ADMIN_STATUS_WAIT_ID: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_status_receive_id) ],
            ADMIN_DELSUB_WAIT_ID: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_delsub_receive_id) ],
            ADMIN_SEARCH_WAIT_QUERY: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_search_do) ],
            ADMIN_TAGS_WAIT_ID: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_tags_id) ],
            ADMIN_TAGS_WAIT_VALUE: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_tags_set) ],
            ADMIN_BROADCAST_SEGMENT: [
                CallbackQueryHandler(admin_broadcast_pick, pattern="^bseg_(all|free|subs|inactive)(_silent)?$"),
                CallbackQueryHandler(admin_panel_open, pattern="^admin_panel$"),
            ],
            ADMIN_BROADCAST_WAIT_TEXT: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_broadcast_send) ],
            ADMIN_SETLIMIT_WAIT_ID: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_setlimit_id) ],
            ADMIN_SETLIMIT_WAIT_VALUES: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_setlimit_values) ],
            ADMIN_BLACKLIST_WAIT_ID: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_blacklist_apply) ],
            ADMIN_SHADOW_WAIT_ID: [ MessageHandler(filters.TEXT & ~filters.COMMAND, admin_shadow_apply) ],
            ADMIN_METRICS_MENU: [
                CallbackQueryHandler(admin_metrics_refresh, pattern="^admin_metrics_refresh$"),
                CallbackQueryHandler(admin_metrics_export, pattern="^admin_metrics_export$"),
                CallbackQueryHandler(admin_panel_open, pattern="^admin_panel$"),
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
