"""
Catch the Stars · Roguelike
===========================
Run:
    Double-click run_catch_stars.bat
    or: python catch_the_stars.py

Gameplay:
    Start by choosing Easy / Normal / Hard
    Clear each wave by catching enough stars → pick 1 of 3 upgrades → harder waves
    Special waves: Frenzy (3/7/11…) · Boss (5/10/15…) · Shop every 3 clears
    Shop: buy / skip (Continue) / pay to Refresh offers
    HP hits 0 → run ends. Press R or both fists to return to difficulty select

Controls:
    Hand / mouse / arrows or A D   move paddle
    Menu / upgrade / shop: move hand L/C/R, thumbs-up to confirm
    Shop: swipe hand down → Refresh (left) or Continue (right), thumbs-up confirm
    Shop keys: 1/2/3 buy · R refresh · Space skip/continue
    Q E F G T                      use items (bomb/slow/heart/magnet/gold rush)
    Both fists to camera           restart (back to difficulty menu)
    H                              toggle hand control
    R                              refresh in shop · menu after game over
    ESC                            quit
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import pygame

# ---------------------------------------------------------------------------
# 手势识别（MediaPipe 1.x）
# ---------------------------------------------------------------------------

HAND_OK = False
cv2 = mp = BaseOptions = HandLandmarker = HandLandmarkerOptions = RunningMode = None

try:
    import cv2 as _cv2
    import mediapipe as _mp
    from mediapipe.tasks.python import BaseOptions as _BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker as _HandLandmarker,
        HandLandmarkerOptions as _HandLandmarkerOptions,
        RunningMode as _RunningMode,
    )

    cv2, mp = _cv2, _mp
    BaseOptions = _BaseOptions
    HandLandmarker = _HandLandmarker
    HandLandmarkerOptions = _HandLandmarkerOptions
    RunningMode = _RunningMode
    HAND_OK = True
except Exception as exc:  # noqa: BLE001
    print(f"[Hint] Hand tracking unavailable (mouse/keyboard only): {exc}")

MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"


class HandController:
    SENSITIVITY = 2.2
    ACTIVE_LEFT = 0.18
    ACTIVE_RIGHT = 0.82

    def __init__(self, camera_index: int = 0):
        self.enabled = HAND_OK
        self.cap = None
        self.landmarker = None
        self.t0 = time.time()
        self.last_x_norm = None
        self.last_fingers = 0
        self.last_thumbs_up = False
        self.last_dual_fist = False
        self.last_card_hover = None  # 1 / 2 / 3 from hand X zones
        self.last_y_norm = None
        self.last_shop_target = None  # 1/2/3 / "refresh" / "continue"
        self._y_hist: list[float] = []
        self.last_swipe_down = False
        self.last_preview = None
        self.status = "Hand: off" if not HAND_OK else "Hand: starting"
        if not HAND_OK:
            return
        if not MODEL_PATH.exists():
            self.enabled = False
            self.status = f"Missing model: {MODEL_PATH.name}"
            return
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            self.landmarker = HandLandmarker.create_from_options(options)
            self.status = "Hand: show your index finger"
        except Exception as exc:  # noqa: BLE001
            self.enabled = False
            self.status = f"Hand failed: {exc}"
            self.close()

    @classmethod
    def map_hand_x(cls, raw_x: float) -> float:
        span = cls.ACTIVE_RIGHT - cls.ACTIVE_LEFT
        t = max(0.0, min(1.0, (raw_x - cls.ACTIVE_LEFT) / span))
        return max(0.0, min(1.0, (t - 0.5) * cls.SENSITIVITY + 0.5))

    @staticmethod
    def count_fingers(lms) -> int:
        """Count extended fingers (index/middle/ring/pinky)."""
        pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
        n = 0
        for tip_i, pip_i in pairs:
            if lms[tip_i].y < lms[pip_i].y - 0.02:
                n += 1
        return n

    @staticmethod
    def is_thumbs_up(lms) -> bool:
        """Thumbs-up: thumb up, other fingers curled."""
        thumb_up = lms[4].y < lms[3].y - 0.02 and lms[4].y < lms[2].y
        # Also require thumb tip clearly above wrist
        thumb_above_wrist = lms[4].y < lms[0].y - 0.05
        others_curled = all(
            lms[tip].y > lms[pip].y - 0.01
            for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
        )
        return thumb_up and thumb_above_wrist and others_curled

    @staticmethod
    def is_fist(lms) -> bool:
        """Closed fist: four fingers curled, thumb tucked near palm."""
        fingers_curled = all(
            lms[tip].y > lms[pip].y - 0.005
            for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
        )
        # Thumb tip close to index MCP (landmark 5) = tucked in
        dx = lms[4].x - lms[5].x
        dy = lms[4].y - lms[5].y
        thumb_tucked = (dx * dx + dy * dy) < 0.045
        # Not a thumbs-up (thumb should not stick up)
        thumb_not_up = lms[4].y > lms[2].y - 0.02
        return fingers_curled and thumb_tucked and thumb_not_up

    @staticmethod
    def card_from_x(raw_x: float) -> int:
        """Map hand X to card 1 / 2 / 3 (left / center / right)."""
        if raw_x < 0.38:
            return 1
        if raw_x < 0.62:
            return 2
        return 3

    def shop_target_from_hand(self, raw_x: float, raw_y: float):
        """Upper area = offer cards; swipe/move down = Refresh / Continue."""
        # Swipe-down or hand low → bottom actions
        if self.last_swipe_down or raw_y > 0.58:
            if raw_x < 0.48:
                return "refresh"
            return "continue"
        return self.card_from_x(raw_x)

    def update(self):
        if not self.enabled or self.cap is None or self.landmarker is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            self.status = "Hand: no camera frame"
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - self.t0) * 1000)
        result = self.landmarker.detect_for_video(mp_image, ts)
        self.last_dual_fist = False
        self.last_swipe_down = False
        if result.hand_landmarks:
            hands = result.hand_landmarks
            lms = hands[0]
            tip = lms[8]
            wrist = lms[0]
            self.last_x_norm = self.map_hand_x(tip.x)
            self.last_y_norm = tip.y
            self.last_fingers = self.count_fingers(lms)
            self.last_thumbs_up = self.is_thumbs_up(lms)
            # Use wrist X for card hover (stable while showing thumbs-up)
            self.last_card_hover = self.card_from_x(wrist.x)

            # Detect downward swipe (MediaPipe y grows downward)
            self._y_hist.append(tip.y)
            if len(self._y_hist) > 8:
                self._y_hist.pop(0)
            if len(self._y_hist) >= 5:
                if self._y_hist[-1] - self._y_hist[0] > 0.12:
                    self.last_swipe_down = True

            self.last_shop_target = self.shop_target_from_hand(wrist.x, tip.y)

            fists = sum(1 for h in hands if self.is_fist(h))
            self.last_dual_fist = fists >= 2 and len(hands) >= 2

            h, w = frame.shape[:2]
            for hand_lms in hands:
                t = hand_lms[8]
                color = (0, 180, 255) if self.is_fist(hand_lms) else (0, 255, 255)
                cv2.circle(frame, (int(t.x * w), int(t.y * h)), 12, color, -1)

            if self.last_dual_fist:
                label = "DUAL FIST = RESTART"
                col = (0, 120, 255)
                self.status = "Hand: both fists → restart"
            elif self.last_thumbs_up:
                label = f"THUMBS → {self.last_shop_target}"
                col = (0, 255, 0)
                self.status = f"Hand: thumbs-up → {self.last_shop_target}"
            elif self.last_shop_target in ("continue", "refresh"):
                label = str(self.last_shop_target).upper()
                col = (80, 220, 255)
                self.status = f"Hand: highlight {self.last_shop_target}"
            else:
                label = f"CARD {self.last_card_hover}"
                col = (255, 220, 0)
                self.status = f"Hand: highlight card {self.last_card_hover}"
            cv2.putText(
                frame,
                label,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                col,
                2,
            )
        else:
            self.last_fingers = 0
            self.last_thumbs_up = False
            self.last_dual_fist = False
            self.last_card_hover = None
            self.last_shop_target = None
            self.last_y_norm = None
            self._y_hist.clear()
            self.status = "Hand: no hand detected"
        preview = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2RGB)
        self.last_preview = preview
        return self.last_x_norm

    def close(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------

pygame.init()
try:
    pygame.key.stop_text_input()
except Exception:
    pass

WIDTH, HEIGHT = 520, 700
FPS = 60

WHITE = (255, 255, 255)
YELLOW = (255, 220, 80)
GOLD = (255, 180, 40)
SKY = (28, 42, 78)
PADDLE_COLOR = (100, 200, 255)
RED = (220, 80, 80)
HEART_RED = (230, 45, 70)
HEART_EMPTY = (70, 55, 75)
PINK = (255, 120, 160)
PURPLE = (180, 120, 255)
ORANGE = (255, 140, 60)
HINT = (180, 200, 230)
GREEN = (80, 220, 120)
CARD_BG = (35, 50, 90)
CARD_BORDER = (120, 180, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Catch the Stars · Roguelike")
clock = pygame.time.Clock()

# Sprites (cut-out characters)
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
GOLD_STAR_IMG = None
PADDLE_IMG = None
PADDLE_IMG_BASE_W = 100
try:
    gold_path = ASSETS_DIR / "gold_star.png"
    paddle_path = ASSETS_DIR / "paddle.png"
    if gold_path.exists():
        GOLD_STAR_IMG = pygame.image.load(str(gold_path)).convert_alpha()
    if paddle_path.exists():
        PADDLE_IMG = pygame.image.load(str(paddle_path)).convert_alpha()
        PADDLE_IMG_BASE_W = PADDLE_IMG.get_width()
except Exception as exc:  # noqa: BLE001
    print(f"[Hint] Sprite load failed: {exc}")

try:
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(16)
    pygame.mixer.set_reserved(1)  # channel 0 = BGM
except Exception:
    pass

try:
    font = pygame.font.SysFont("microsoftyahei", 26)
    small_font = pygame.font.SysFont("microsoftyahei", 17)
    big_font = pygame.font.SysFont("microsoftyahei", 44)
except Exception:
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 20)
    big_font = pygame.font.Font(None, 50)


# ---------------------------------------------------------------------------
# Sound (procedural — juicy SFX + looping BGM)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050


def _mix_tone(freq: float, ms: int, vol: float = 0.35, slide_to: float | None = None, harm: float = 0.35):
    """Richer beep: fundamental + octave harmonic + optional pitch slide."""
    n = max(1, int(SAMPLE_RATE * ms / 1000))
    buf = bytearray()
    for i in range(n):
        t = i / SAMPLE_RATE
        # punchy attack, soft release
        env = min(1.0, i / 180.0) * min(1.0, (n - i) / 500.0)
        f = freq if slide_to is None else freq + (slide_to - freq) * (i / n)
        wave = math.sin(2 * math.pi * f * t)
        wave += harm * math.sin(2 * math.pi * f * 2 * t)
        wave += harm * 0.4 * math.sin(2 * math.pi * f * 3 * t)
        val = int(vol * 32767 * env * wave / (1 + harm))
        val = max(-32767, min(32767, val))
        b = val.to_bytes(2, "little", signed=True)
        buf += b + b
    try:
        return pygame.mixer.Sound(buffer=bytes(buf))
    except Exception:
        return None


def _chord(freqs: list[float], ms: int = 120, vol: float = 0.28):
    """Play several pitches stacked for that 'ding!' feel."""
    n = max(1, int(SAMPLE_RATE * ms / 1000))
    buf = bytearray()
    for i in range(n):
        t = i / SAMPLE_RATE
        env = min(1.0, i / 120.0) * min(1.0, (n - i) / 400.0)
        wave = 0.0
        for f in freqs:
            wave += math.sin(2 * math.pi * f * t)
            wave += 0.3 * math.sin(2 * math.pi * f * 2 * t)
        wave /= max(1, len(freqs))
        val = int(vol * 32767 * env * wave)
        val = max(-32767, min(32767, val))
        b = val.to_bytes(2, "little", signed=True)
        buf += b + b
    try:
        return pygame.mixer.Sound(buffer=bytes(buf))
    except Exception:
        return None


def _make_bgm_loop(seconds: float = 8.0):
    """Upbeat looping arpeggio BGM (chiptune-ish)."""
    n = int(SAMPLE_RATE * seconds)
    # Bright major climb — keeps energy without clashing with SFX
    pattern = [261.63, 329.63, 392.00, 523.25, 392.00, 659.25, 523.25, 329.63]
    buf = bytearray()
    note_hz = 5.5
    for i in range(n):
        t = i / SAMPLE_RATE
        idx = int(t * note_hz) % len(pattern)
        f = pattern[idx]
        phase = (t * note_hz) % 1.0
        pluck = math.exp(-phase * 4.2) * 0.13
        sparkle = math.exp(-phase * 8.0) * 0.04 * math.sin(2 * math.pi * f * 2 * t)
        pad = 0.03 * math.sin(2 * math.pi * (f * 0.5) * t)
        bass = 0.035 * math.sin(2 * math.pi * 82.4 * t) * (0.55 + 0.45 * math.sin(2 * math.pi * 1.1 * t))
        wave = pluck * math.sin(2 * math.pi * f * t) + sparkle + pad + bass
        val = int(32767 * wave)
        val = max(-32767, min(32767, val))
        b = val.to_bytes(2, "little", signed=True)
        buf += b + b
    try:
        return pygame.mixer.Sound(buffer=bytes(buf))
    except Exception:
        return None


class SFX:
    """Juicy score ladder — higher combo = brighter chords."""

    # pentatonic climb that never sounds wrong
    SCORE_SCALE = [523, 587, 659, 784, 880, 988, 1175, 1319, 1568, 1760]

    def __init__(self):
        self.ok = pygame.mixer.get_init() is not None
        self.hit = _mix_tone(150, 160, 0.5, slide_to=70, harm=0.5) if self.ok else None
        self.heal = _chord([523, 659, 784], 160, 0.32) if self.ok else None
        self.gold = _chord([880, 1108, 1319, 1760], 180, 0.38) if self.ok else None
        self.shield = _mix_tone(420, 90, 0.32, slide_to=640) if self.ok else None
        self.wave = _chord([392, 523, 659, 784], 220, 0.3) if self.ok else None
        self.click = _mix_tone(720, 45, 0.22) if self.ok else None
        self._score_cache: dict[tuple, object] = {}
        self.bgm = _make_bgm_loop(6.0) if self.ok else None
        self._bgm_channel = None

    def start_bgm(self):
        if not self.ok or self.bgm is None:
            return
        try:
            self._bgm_channel = pygame.mixer.Channel(0)
            self._bgm_channel.set_volume(0.28)
            self._bgm_channel.play(self.bgm, loops=-1)
        except Exception:
            try:
                self.bgm.set_volume(0.28)
                self.bgm.play(loops=-1)
            except Exception:
                pass

    def stop_bgm(self):
        try:
            if self._bgm_channel:
                self._bgm_channel.stop()
            elif self.bgm:
                self.bgm.stop()
        except Exception:
            pass

    def _play(self, sound, volume: float | None = None):
        if self.ok and sound is not None:
            try:
                if volume is not None:
                    sound.set_volume(max(0.0, min(1.0, volume)))
                sound.play()
            except Exception:
                pass

    def score(self, combo: int, kind: str = "normal"):
        """Stacked chord that climbs — the higher the combo, the more hype."""
        idx = max(0, min(combo, len(self.SCORE_SCALE) - 1))
        key = (idx, kind)
        if key not in self._score_cache:
            f = self.SCORE_SCALE[idx]
            # chord: root + fifth + octave (+ sparkle on high combo)
            freqs = [f, f * 1.5, f * 2.0]
            if idx >= 3:
                freqs.append(f * 2.5)
            if idx >= 6:
                freqs.append(f * 3.0)
            if kind == "gold":
                freqs = [x * 1.12 for x in freqs]
                ms = 140 + idx * 12
                vol = 0.34 + idx * 0.03
            else:
                ms = 100 + idx * 10
                vol = 0.30 + idx * 0.025
            self._score_cache[key] = _chord(freqs, ms, min(0.55, vol))
        self._play(self._score_cache[key])
        if combo >= 2:
            spark_key = ("spark", idx)
            if spark_key not in self._score_cache:
                hi = self.SCORE_SCALE[min(idx + 2, len(self.SCORE_SCALE) - 1)]
                self._score_cache[spark_key] = _mix_tone(
                    hi, 55 + idx * 6, 0.22, slide_to=hi * 1.45, harm=0.65
                )
            self._play(self._score_cache[spark_key])
        if combo >= 5:
            fanfare_key = ("fan", idx)
            if fanfare_key not in self._score_cache:
                f = self.SCORE_SCALE[idx]
                self._score_cache[fanfare_key] = _chord(
                    [f * 1.5, f * 2.0, f * 3.0], 90 + idx * 8, 0.26
                )
            self._play(self._score_cache[fanfare_key])

    def play_hit(self):
        self._play(self.hit)

    def play_death(self):
        self._play(_mix_tone(380, 480, 0.5, slide_to=50, harm=0.7))

    def play_heal(self):
        self._play(self.heal)

    def play_gold(self):
        self._play(self.gold)

    def play_shield(self):
        self._play(self.shield)

    def play_wave(self):
        self._play(self.wave)

    def play_click(self):
        self._play(self.click)


sfx = SFX()

# Easy / Normal / Hard presets
DIFFICULTIES = {
    "easy": {
        "label": "Easy",
        "blurb": "More HP, slower stars",
        "hp": 5,
        "speed": 3.4,
        "accel": 0.05,
        "max_speed": 14,
        "diff0": 0.0,
        "bomb_weight": 0.08,
        "speed_mul": 1.0,
    },
    "normal": {
        "label": "Normal",
        "blurb": "Balanced challenge",
        "hp": 3,
        "speed": 4.6,
        "accel": 0.08,
        "max_speed": 22,
        "diff0": 0.0,
        "bomb_weight": 0.14,
        "speed_mul": 1.0,
    },
    "hard": {
        "label": "Hard",
        "blurb": "Fast stars, more bombs",
        "hp": 2,
        "speed": 6.2,
        "accel": 0.11,
        "max_speed": 26,
        "diff0": 1.2,
        "bomb_weight": 0.22,
        "speed_mul": 1.05,
    },
}
DIFF_ORDER = ("easy", "normal", "hard")

held = {
    pygame.K_LEFT: False,
    pygame.K_RIGHT: False,
    pygame.K_a: False,
    pygame.K_d: False,
}

TRAJECTORY_NAMES = ("straight", "sine", "drift", "zigzag")

# 星星种类：normal 得分；gold 高分；heart 回血；bomb 接住扣血；boss 巨型
STAR_TYPES = ("normal", "gold", "heart", "bomb", "boss")

# Shop pool: mix of one-shot items + mild permanent buffs (balanced)
def _shop_buff_wide(s):
    s["paddle_bonus"] += 10
    _apply_paddle_size(s)


def _shop_buff_luck(s):
    s["luck"] = min(0.45, s["luck"] + 0.06)


def _shop_buff_slow(s):
    s["speed_mul"] = max(0.62, s["speed_mul"] - 0.06)


def _shop_buff_magnet(s):
    s["magnet"] += 55


def _shop_buff_shield(s):
    s["shield"] += 1


def _shop_buff_follow(s):
    s["hand_follow"] = min(0.32, s["hand_follow"] + 0.04)
    s["hand_max_step"] += 2


def _shop_buff_quota(s):
    s["quota_cut"] = min(6, s["quota_cut"] + 1)


SHOP_POOL = [
    # Consumables
    {
        "id": "nuke",
        "kind": "item",
        "name": "Star Bomb",
        "desc": "Clear falling stars (Q)",
        "price": 25,
    },
    {
        "id": "slow",
        "kind": "item",
        "name": "Slow-Mo",
        "desc": "Stars crawl 5s (E)",
        "price": 28,
    },
    {
        "id": "heart",
        "kind": "item",
        "name": "Heart Pack",
        "desc": "Heal 1 HP (F)",
        "price": 32,
    },
    {
        "id": "ghost",
        "kind": "item",
        "name": "Ghost Veil",
        "desc": "Ignore next miss (auto)",
        "price": 38,
    },
    {
        "id": "magnet_pulse",
        "kind": "item",
        "name": "Magnet Burst",
        "desc": "Strong pull 4s (G)",
        "price": 26,
    },
    {
        "id": "gold_rush",
        "kind": "item",
        "name": "Gold Rush",
        "desc": "1.5x score for 6s (T)",
        "price": 34,
    },
    # Mild permanent buffs (weaker than upgrade cards)
    {
        "id": "buff_wide",
        "kind": "buff",
        "name": "Grip Tape",
        "desc": "Paddle a bit wider",
        "price": 40,
        "fn": _shop_buff_wide,
    },
    {
        "id": "buff_luck",
        "kind": "buff",
        "name": "Lucky Charm",
        "desc": "Slightly luckier stars",
        "price": 42,
        "fn": _shop_buff_luck,
    },
    {
        "id": "buff_slow",
        "kind": "buff",
        "name": "Molasses",
        "desc": "Stars a bit slower",
        "price": 48,
        "fn": _shop_buff_slow,
    },
    {
        "id": "buff_magnet",
        "kind": "buff",
        "name": "Fridge Magnet",
        "desc": "Weak star pull",
        "price": 44,
        "fn": _shop_buff_magnet,
    },
    {
        "id": "buff_shield",
        "kind": "buff",
        "name": "Tin Shield",
        "desc": "+1 shield charge",
        "price": 52,
        "fn": _shop_buff_shield,
    },
    {
        "id": "buff_follow",
        "kind": "buff",
        "name": "Steady Aim",
        "desc": "Smoother hand aim",
        "price": 30,
        "fn": _shop_buff_follow,
    },
    {
        "id": "buff_quota",
        "kind": "buff",
        "name": "Short Cut",
        "desc": "Wave goal -1",
        "price": 36,
        "fn": _shop_buff_quota,
    },
]

RARE_BOSS_DROPS = ("shield", "luck", "mult", "magnet")


def classify_wave(wave: int) -> str:
    """Special wave schedule: boss every 5, frenzy on 3/7/11… (boss wins ties)."""
    if wave >= 5 and wave % 5 == 0:
        return "boss"
    if wave % 4 == 3:
        return "frenzy"
    return "normal"


def should_open_shop(cleared_wave: int) -> bool:
    return cleared_wave >= 3 and cleared_wave % 3 == 0


def wave_quota_for(state) -> int:
    kind = state.get("wave_kind", "normal")
    if kind == "boss":
        return 1
    if kind == "frenzy":
        return 9999  # cleared by timer
    return max(4, 5 + state["wave"] - state.get("quota_cut", 0))


def stars_needed_on_screen(state) -> int:
    kind = state.get("wave_kind", "normal")
    if kind == "boss":
        return 0  # boss spawned explicitly
    if kind == "frenzy":
        return 5
    wave = state["wave"]
    if wave >= 4:
        return 3
    if wave >= 2:
        return 2
    return 1


# ---------------------------------------------------------------------------
# 升级卡（本局有效，死后清空）
# ---------------------------------------------------------------------------


def _up_wide(s):
    s["paddle_bonus"] += 22
    _apply_paddle_size(s)


def _up_life(s):
    s["hp"] = min(s["hp_max"] + 1, s["hp"] + 1)
    s["hp_max"] += 1


def _up_slow(s):
    s["speed_mul"] = max(0.55, s["speed_mul"] - 0.12)


def _up_magnet(s):
    s["magnet"] += 140


def _up_mult(s):
    s["score_mult"] += 1


def _up_quota(s):
    s["quota_cut"] += 2


def _up_luck(s):
    s["luck"] += 0.12


def _up_shield(s):
    s["shield"] += 1


def _up_follow(s):
    s["hand_follow"] = min(0.35, s["hand_follow"] + 0.06)
    s["hand_max_step"] += 4


UPGRADES = [
    {"id": "wide", "name": "Wider Paddle", "desc": "Paddle stays wider", "fn": _up_wide},
    {"id": "life", "name": "Extra Life", "desc": "Max HP +1 and heal 1", "fn": _up_life},
    {"id": "slow", "name": "Time Warp", "desc": "Stars fall slower", "fn": _up_slow},
    {"id": "magnet", "name": "Star Magnet", "desc": "Pull stars to paddle", "fn": _up_magnet},
    {"id": "mult", "name": "Score Boost", "desc": "Score multiplier +1", "fn": _up_mult},
    {"id": "quota", "name": "Shortcut", "desc": "Wave goal -2 stars", "fn": _up_quota},
    {"id": "luck", "name": "Lucky Stars", "desc": "More gold & hearts", "fn": _up_luck},
    {"id": "shield", "name": "Shield", "desc": "Block 1 hit", "fn": _up_shield},
    {"id": "follow", "name": "Steady Hand", "desc": "Smoother hand control", "fn": _up_follow},
]


def roll_upgrade_choices(owned_ids: list[str], n: int = 3) -> list[dict]:
    pool = [u for u in UPGRADES]
    # 同 id 可重复拿（叠层），所以不排除 owned
    return random.sample(pool, k=min(n, len(pool)))


# ---------------------------------------------------------------------------
# 星星 / 轨迹
# ---------------------------------------------------------------------------


def make_star(difficulty: float = 0.0, luck: float = 0.0, bomb_weight: float = 0.14):
    size = 24
    margin = 40
    x = float(random.randint(margin, WIDTH - size - margin))
    y = float(-size - random.randint(0, 80))

    weights = [
        max(0.05, 0.42 - difficulty * 0.06),
        0.24 + difficulty * 0.035,
        0.16 + difficulty * 0.03,
        0.18 + difficulty * 0.05,
    ]
    kind = random.choices(TRAJECTORY_NAMES, weights=weights, k=1)[0]

    type_weights = {
        "normal": max(0.2, 0.58 - luck * 0.1 - bomb_weight * 0.3),
        "gold": 0.15 + luck * 0.2,
        "heart": 0.06 + luck * 0.12,
        "bomb": bomb_weight + difficulty * 0.03,
    }
    stype = random.choices(
        list(type_weights.keys()), weights=list(type_weights.values()), k=1
    )[0]

    colors = {
        "normal": YELLOW,
        "gold": GOLD,
        "heart": PINK,
        "bomb": (40, 40, 45),
    }

    return {
        "rect": pygame.Rect(int(x), int(y), 32 if stype == "gold" else size, 32 if stype == "gold" else size),
        "x": x,
        "y": y,
        "base_x": x,
        "kind": kind,
        "stype": stype,
        "color": colors[stype],
        "phase": random.uniform(0, math.tau),
        "amp": 35 + difficulty * 12 + random.uniform(0, 20),
        "freq": 0.04 + difficulty * 0.01 + random.uniform(0, 0.025),
        "drift": random.choice([-1, 1])
        * (1.5 + difficulty * 0.4 + random.uniform(0, 1.0)),
        "zig_dir": random.choice([-1, 1]),
        "zig_timer": 0,
        "zig_period": random.randint(10, 22),
        "speed_factor": 1.55 if stype == "bomb" else (0.9 if stype == "heart" else (0.95 if stype == "gold" else 1.15)),
    }


def make_boss_star(difficulty: float = 0.0):
    """Huge slow boss — catching it splits into bonus stars + rare drop."""
    size = 72
    x = float(WIDTH // 2 - size // 2)
    y = float(-size - 20)
    return {
        "rect": pygame.Rect(int(x), int(y), size, size),
        "x": x,
        "y": y,
        "base_x": x,
        "kind": "sine",
        "stype": "boss",
        "color": PURPLE,
        "phase": 0.0,
        "amp": 50 + difficulty * 8,
        "freq": 0.025,
        "drift": 0.0,
        "zig_dir": 1,
        "zig_timer": 0,
        "zig_period": 20,
        "speed_factor": 0.42,
        "is_boss": True,
    }


def make_split_star(x: float, y: float, stype: str = "gold"):
    size = 22 if stype != "gold" else 28
    colors = {"normal": YELLOW, "gold": GOLD, "heart": PINK}
    return {
        "rect": pygame.Rect(int(x), int(y), size, size),
        "x": float(x),
        "y": float(y),
        "base_x": float(x),
        "kind": random.choice(("sine", "drift", "zigzag")),
        "stype": stype,
        "color": colors.get(stype, YELLOW),
        "phase": random.uniform(0, math.tau),
        "amp": 28,
        "freq": 0.05,
        "drift": random.choice([-1, 1]) * random.uniform(1.2, 2.4),
        "zig_dir": random.choice([-1, 1]),
        "zig_timer": 0,
        "zig_period": random.randint(10, 18),
        "speed_factor": 0.85 if stype == "gold" else 1.0,
    }


def sync_star_rect(star):
    star["rect"].x = int(star["x"])
    star["rect"].y = int(star["y"])


def step_star(star, fall_speed: float, paddle=None, magnet: float = 0.0):
    star["y"] += fall_speed * star["speed_factor"]
    kind = star["kind"]

    if kind == "sine":
        star["phase"] += star["freq"]
        star["x"] = star["base_x"] + star["amp"] * math.sin(star["phase"])
    elif kind == "drift":
        star["x"] += star["drift"]
        if star["x"] < 0 or star["x"] > WIDTH - star["rect"].width:
            star["drift"] *= -1
            star["x"] = max(0, min(star["x"], WIDTH - star["rect"].width))
    elif kind == "zigzag":
        star["zig_timer"] += 1
        star["x"] += star["zig_dir"] * (2.2 + abs(star["drift"]) * 0.4)
        if star["zig_timer"] >= star["zig_period"]:
            star["zig_timer"] = 0
            star["zig_dir"] *= -1
            star["zig_period"] = random.randint(10, 26)
        if star["x"] < 0 or star["x"] > WIDTH - star["rect"].width:
            star["zig_dir"] *= -1
            star["x"] = max(0, min(star["x"], WIDTH - star["rect"].width))

    # 磁力：更强吸附（炸弹除外）；越近拉力越大，并略微往下吸
    if magnet > 0 and paddle is not None and star["stype"] not in ("bomb",):
        # Boss only lightly magnetized
        mag_scale = 0.35 if star.get("stype") == "boss" else 1.0
        sx = star["x"] + star["rect"].width / 2
        sy = star["y"] + star["rect"].height / 2
        dx = paddle.centerx - sx
        dy = paddle.centery - sy
        dist = math.hypot(dx, dy)
        range_r = magnet + 80
        if 1 < dist < range_r and star["y"] > HEIGHT * 0.12:
            strength = (0.28 + 0.35 * (1.0 - dist / range_r)) * mag_scale
            star["x"] += dx * strength
            # 纵向也拉一点，更容易接到
            star["y"] += max(0, dy) * strength * 0.55
            star["base_x"] = star["x"]
            # 被吸住时减弱横向飘移，避免吸不动
            if kind in ("drift", "zigzag"):
                star["drift"] *= 0.92
                star["zig_dir"] = 1 if dx >= 0 else -1

    star["x"] = max(0, min(star["x"], WIDTH - star["rect"].width))
    star["y"] = min(star["y"], HEIGHT + 40)
    sync_star_rect(star)


def _apply_paddle_size(state):
    base = 100 + state["paddle_bonus"]
    base = max(48, min(220, base))
    cx = state["paddle"].centerx
    state["paddle"].width = base
    state["paddle"].centerx = cx
    state["paddle"].x = max(0, min(state["paddle"].x, WIDTH - state["paddle"].width))


# Compat wrappers (prefer wave_quota_for / stars_needed_on_screen)
def wave_quota(wave: int, quota_cut: int) -> int:
    return max(4, 5 + wave - quota_cut)


def stars_on_screen(wave: int) -> int:
    if wave >= 4:
        return 3
    if wave >= 2:
        return 2
    return 1


# ---------------------------------------------------------------------------
# 局内状态
# ---------------------------------------------------------------------------


def reset_run(hand_enabled: bool = True, difficulty_key: str = "normal"):
    preset = DIFFICULTIES.get(difficulty_key, DIFFICULTIES["normal"])
    paddle = pygame.Rect(WIDTH // 2 - 50, HEIGHT - 55, 100, 18)
    state = {
        "mode": "playing",
        "difficulty_key": difficulty_key,
        "paddle": paddle,
        "stars": [],
        "star_speed": preset["speed"],
        "base_star_speed": preset["speed"],
        "max_star_speed": preset["max_speed"],
        "star_accel": preset["accel"],
        "paddle_speed": 8,
        "hand_follow": 0.18,
        "hand_max_step": 22,
        "hand_enabled": hand_enabled,
        "score": 0,
        "hp": preset["hp"],
        "hp_max": preset["hp"],
        "shield": 0,
        "wave": 1,
        "caught_in_wave": 0,
        "difficulty": preset["diff0"],
        "bomb_weight": preset["bomb_weight"],
        "speed_mul": preset["speed_mul"],
        "magnet": 0.0,
        "score_mult": 1,
        "quota_cut": 0,
        "luck": 0.0,
        "paddle_bonus": 0,
        "upgrades_taken": [],
        "choices": [],
        "choice_rects": [],
        "menu_rects": [],
        "flash": None,
        "best_wave": 1,
        "gesture_hold": 0,
        "gesture_hold_id": None,
        "gesture_hover": None,
        "combo": 0,
        "popups": [],
        "particles": [],
        "hit_flash": 0,
        "score_flash": 0,
        "paddle_punch": 0,
        "shake": 0,
        "heart_pops": [],
        "fist_hold": 0,
        "wave_kind": "normal",
        "frenzy_timer": 0,
        "items": {
            "nuke": 0,
            "slow": 0,
            "heart": 0,
            "ghost": 0,
            "magnet_pulse": 0,
            "gold_rush": 0,
        },
        "slowmo_timer": 0,
        "magnet_pulse_timer": 0,
        "gold_rush_timer": 0,
        "shop_rects": [],
        "shop_offers": [],
        "shop_refresh_count": 0,
        "pending_splits": [],
    }
    begin_wave(state)
    return state


def new_menu(hand_enabled: bool = True):
    return {
        "mode": "menu",
        "hand_enabled": hand_enabled,
        "menu_rects": [],
        "gesture_hold": 0,
        "gesture_hold_id": None,
        "gesture_hover": None,
        "difficulty_key": "normal",
        "popups": [],
        "particles": [],
        "hit_flash": 0,
        "score_flash": 0,
        "paddle_punch": 0,
        "shake": 0,
        "flash": None,
        "combo": 0,
        "heart_pops": [],
        "fist_hold": 0,
        "upgrades_taken": [],
        "paddle": pygame.Rect(0, 0, 1, 1),
        "stars": [],
        "magnet": 0,
        "score": 0,
        "wave": 1,
        "caught_in_wave": 0,
        "quota_cut": 0,
        "score_mult": 1,
        "hp": 0,
        "hp_max": 0,
        "shield": 0,
        "wave_kind": "normal",
        "frenzy_timer": 0,
        "items": {
            "nuke": 0,
            "slow": 0,
            "heart": 0,
            "ghost": 0,
            "magnet_pulse": 0,
            "gold_rush": 0,
        },
        "slowmo_timer": 0,
        "magnet_pulse_timer": 0,
        "gold_rush_timer": 0,
        "shop_rects": [],
        "shop_offers": [],
        "shop_refresh_count": 0,
        "pending_splits": [],
    }


def begin_wave(state):
    """Start current wave content (normal / boss / frenzy)."""
    kind = classify_wave(state["wave"])
    state["wave_kind"] = kind
    state["caught_in_wave"] = 0
    state["stars"] = []
    state["pending_splits"] = []
    state["frenzy_timer"] = 0
    state["boss_victory_timer"] = 0
    state["gesture_hold"] = 0
    state["gesture_hold_id"] = None
    state["gesture_hover"] = None
    if kind == "boss":
        state["stars"].append(make_boss_star(state["difficulty"]))
        state["flash"] = ("BOSS WAVE! Catch the giant!", 70)
        sfx.play_wave()
    elif kind == "frenzy":
        state["frenzy_timer"] = 8 * FPS
        state["flash"] = ("FRENZY! 2x score — misses free!", 70)
        _refill_stars(state)
        sfx.play_wave()
    else:
        _refill_stars(state)
        state["flash"] = (f"Wave {state['wave']}", 40)


def _refill_stars(state):
    if state.get("wave_kind") == "boss":
        return
    need = stars_needed_on_screen(state)
    # Frenzy: prefer score stars, almost no bombs
    bomb_w = 0.02 if state.get("wave_kind") == "frenzy" else state.get("bomb_weight", 0.14)
    luck = state["luck"] + (0.15 if state.get("wave_kind") == "frenzy" else 0.0)
    while len(state["stars"]) < need:
        state["stars"].append(
            make_star(state["difficulty"], luck, bomb_w)
        )


def enter_shop(state):
    state["mode"] = "shop"
    state["stars"].clear()
    state["shop_offers"] = roll_shop_offers(3)
    state["shop_refresh_count"] = 0
    state["flash"] = (f"Shop — Wave {state['wave']} clear!", 50)
    state["gesture_hold"] = 0
    state["gesture_hold_id"] = None
    state["gesture_hover"] = None
    state["combo"] = 0
    sfx.play_wave()


def enter_upgrade(state):
    state["mode"] = "upgrade"
    state["choices"] = roll_upgrade_choices(state["upgrades_taken"], 3)
    state["stars"].clear()
    state["flash"] = (f"Wave {state['wave']} cleared!", 50)
    state["gesture_hold"] = 0
    state["gesture_hold_id"] = None
    state["gesture_hover"] = None
    state["combo"] = 0
    sfx.play_wave()


def roll_shop_offers(n: int = 3) -> list[dict]:
    pool = list(SHOP_POOL)
    return random.sample(pool, k=min(n, len(pool)))


def shop_price(item: dict, wave: int) -> int:
    return item["price"] + max(0, (wave - 3) // 3) * 4


def shop_refresh_price(state) -> int:
    base = 12 + max(0, (state["wave"] - 3) // 3) * 4
    return base + state.get("shop_refresh_count", 0) * 8


def buy_shop_item(state, index: int):
    if state["mode"] != "shop":
        return
    offers = state.get("shop_offers") or []
    if index < 0 or index >= len(offers):
        return
    item = offers[index]
    if item.get("sold"):
        state["flash"] = ("Already bought!", 30)
        return
    price = shop_price(item, state["wave"])
    if state["score"] < price:
        state["flash"] = (f"Need {price} score!", 35)
        sfx.play_hit()
        return
    state["score"] -= price
    if item.get("kind") == "buff":
        item["fn"](state)
        state["upgrades_taken"].append(f"shop:{item['id']}")
        state["flash"] = (f"Buff: {item['name']}!", 45)
    else:
        state["items"][item["id"]] = state["items"].get(item["id"], 0) + 1
        state["flash"] = (f"Bought {item['name']}!", 40)
    item["sold"] = True
    add_popup(state, f"-{price}", (WIDTH // 2, HEIGHT // 2), ORANGE, size="med")
    sfx.play_gold()


def refresh_shop(state):
    if state["mode"] != "shop":
        return
    cost = shop_refresh_price(state)
    if state["score"] < cost:
        state["flash"] = (f"Refresh needs {cost}!", 35)
        sfx.play_hit()
        return
    state["score"] -= cost
    state["shop_refresh_count"] = state.get("shop_refresh_count", 0) + 1
    state["shop_offers"] = roll_shop_offers(3)
    state["flash"] = (f"Refreshed (−{cost})", 40)
    add_popup(state, f"-{cost}", (WIDTH // 2, HEIGHT // 2 + 30), HINT, size="sm")
    sfx.play_click()


def leave_shop(state):
    if state["mode"] != "shop":
        return
    state["wave"] += 1
    preset = DIFFICULTIES.get(state.get("difficulty_key", "normal"), DIFFICULTIES["normal"])
    state["difficulty"] = min(8.0, state["difficulty"] + 0.7)
    state["base_star_speed"] = min(
        state["max_star_speed"],
        preset["speed"] + state["wave"] * 0.55,
    )
    state["star_speed"] = state["base_star_speed"]
    state["star_accel"] = min(0.16, preset["accel"] + state["wave"] * 0.01)
    state["mode"] = "playing"
    state["shop_rects"] = []
    state["shop_offers"] = []
    state["gesture_hold"] = 0
    state["gesture_hold_id"] = None
    state["gesture_hover"] = None
    begin_wave(state)
    sfx.play_click()


def use_item(state, item_id: str):
    if state["mode"] != "playing":
        return
    if state["items"].get(item_id, 0) <= 0:
        return
    if item_id == "nuke":
        state["items"]["nuke"] -= 1
        cleared = 0
        kept = []
        for star in state["stars"]:
            if star["stype"] == "boss":
                kept.append(star)
            else:
                cleared += 1
                spawn_burst(state, star["rect"].center, YELLOW, 6)
        state["stars"] = kept
        state["flash"] = (f"BOOM cleared {cleared}!", 40)
        sfx.play_gold()
        state["shake"] = 10
    elif item_id == "slow":
        state["items"]["slow"] -= 1
        state["slowmo_timer"] = 5 * FPS
        state["flash"] = ("Slow-Mo!", 40)
        sfx.play_wave()
    elif item_id == "heart":
        if state["hp"] >= state["hp_max"]:
            state["flash"] = ("HP already full!", 30)
            return
        state["items"]["heart"] -= 1
        state["hp"] += 1
        state["flash"] = ("+1 HP!", 35)
        sfx.play_heal()
    elif item_id == "magnet_pulse":
        state["items"]["magnet_pulse"] -= 1
        state["magnet_pulse_timer"] = 4 * FPS
        state["flash"] = ("Magnet Burst!", 40)
        sfx.play_wave()
    elif item_id == "gold_rush":
        state["items"]["gold_rush"] -= 1
        state["gold_rush_timer"] = 6 * FPS
        state["flash"] = ("Gold Rush 1.5x!", 40)
        sfx.play_gold()


def grant_boss_drop(state):
    drop_id = random.choice(RARE_BOSS_DROPS)
    for u in UPGRADES:
        if u["id"] == drop_id:
            u["fn"](state)
            state["upgrades_taken"].append(f"boss:{drop_id}")
            state["flash"] = (f"Boss drop: {u['name']}!", 70)
            return


def add_popup(state, text, pos, color=YELLOW, size: str = "sm", life: int = 40):
    state["popups"].append(
        {
            "text": text,
            "x": float(pos[0]),
            "y": float(pos[1]),
            "life": life,
            "max_life": life,
            "color": color,
            "size": size,
        }
    )


COMBO_WORDS = (
    (2, "NICE!"),
    (3, "GREAT!"),
    (4, "AWESOME!"),
    (5, "INSANE!"),
    (6, "ON FIRE!"),
    (8, "GODLIKE!!!"),
    (10, "UNSTOPPABLE!!!"),
)


def spawn_burst(state, pos, color, count: int = 12):
    for _ in range(count):
        ang = random.uniform(0, math.tau)
        spd = random.uniform(2.2, 8.5)
        state.setdefault("particles", []).append(
            {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "vx": math.cos(ang) * spd,
                "vy": math.sin(ang) * spd - random.uniform(1.5, 4.0),
                "life": random.randint(16, 34),
                "color": color,
                "r": random.randint(2, 5),
            }
        )


def juice_catch(state, pos, gained: int, kind: str = "normal"):
    """Audio + visual punch when you catch a star — scales with combo."""
    combo = state["combo"]
    color = {"gold": GOLD, "heart": PINK}.get(kind, YELLOW)
    bang = "!" * min(5, 1 + combo // 2)
    size = "big" if combo >= 5 else ("med" if combo >= 2 else "sm")
    add_popup(
        state,
        f"+{gained}{bang}",
        pos,
        color,
        size=size,
        life=42 + combo * 3,
    )

    word = None
    for thresh, label in COMBO_WORDS:
        if combo >= thresh:
            word = label
    if word:
        word_color = (255, 255, 140) if combo >= 6 else WHITE
        add_popup(
            state,
            word,
            (pos[0], pos[1] - 30),
            word_color,
            size="big" if combo >= 8 else "med",
            life=48 + combo * 2,
        )
        if combo >= 3:
            add_popup(
                state,
                f"x{combo}",
                (pos[0] + 36, pos[1] - 8),
                color,
                size="med",
                life=40,
            )

    n = 10 + min(24, combo * 3)
    if kind == "gold":
        n += 12
    elif kind == "heart":
        n += 6
    spawn_burst(state, pos, color, n)
    if kind == "gold":
        spawn_burst(state, pos, WHITE, 8)

    state["score_flash"] = min(16, 7 + combo)
    state["paddle_punch"] = min(14, 6 + combo // 2)
    if combo >= 4:
        state["shake"] = max(state.get("shake", 0), min(10, 3 + combo // 2))

    if kind == "gold":
        sfx.play_gold()
    elif kind == "heart":
        sfx.play_heal()
    sfx.score(combo, "gold" if kind == "gold" else "normal")


# HUD heart row (top-left under score lines)
HEART_HUD_X = 52
HEART_HUD_Y = 58
HEART_HUD_GAP = 26


def heart_slot_pos(index: int) -> tuple[int, int]:
    return HEART_HUD_X + index * HEART_HUD_GAP, HEART_HUD_Y


def draw_heart_shape(surf, cx: int, cy: int, size: float, color, outline=None):
    """Simple red heart: two circles + triangle."""
    r = max(3, int(size * 0.38))
    left = (int(cx - r * 0.55), int(cy - r * 0.15))
    right = (int(cx + r * 0.55), int(cy - r * 0.15))
    tip = (int(cx), int(cy + r * 1.35))
    mid_top = (int(cx), int(cy - r * 0.05))
    pygame.draw.circle(surf, color, left, r)
    pygame.draw.circle(surf, color, right, r)
    pygame.draw.polygon(surf, color, [left, right, tip, mid_top])
    if outline is not None:
        pygame.draw.circle(surf, outline, left, r, 1)
        pygame.draw.circle(surf, outline, right, r, 1)
        pygame.draw.lines(surf, outline, True, [left, tip, right], 1)


def draw_hp_hearts(state, ox: int = 0, oy: int = 0):
    """Filled red hearts = current HP; empty slots = lost max hearts."""
    label = small_font.render("HP", True, WHITE)
    screen.blit(label, (14 + ox, HEART_HUD_Y - 10 + oy))
    for i in range(max(0, state.get("hp_max", 0))):
        cx, cy = heart_slot_pos(i)
        filled = i < state.get("hp", 0)
        if filled:
            draw_heart_shape(screen, cx + ox, cy + oy, 18, HEART_RED)
        else:
            draw_heart_shape(screen, cx + ox, cy + oy, 18, HEART_EMPTY, outline=(110, 80, 95))
    # Shield as small cyan diamonds next to hearts
    if state.get("shield", 0) > 0:
        sx = HEART_HUD_X + state["hp_max"] * HEART_HUD_GAP + 10 + ox
        sy = HEART_HUD_Y + oy
        for s in range(state["shield"]):
            x = sx + s * 16
            pygame.draw.polygon(
                screen,
                (100, 220, 255),
                [(x, sy - 8), (x + 7, sy), (x, sy + 8), (x - 7, sy)],
            )


def pop_lost_heart(state):
    """Animate the heart that was just removed."""
    lost_index = state["hp"]  # after HP already decremented
    cx, cy = heart_slot_pos(lost_index)
    state.setdefault("heart_pops", []).append(
        {
            "x": float(cx),
            "y": float(cy),
            "vy": -2.5,
            "life": 36,
            "size": 18.0,
        }
    )


def take_damage(state, amount: int = 1):
    if state["shield"] > 0:
        state["shield"] -= 1
        state["flash"] = ("Shield blocked!", 45)
        sfx.play_shield()
        state["combo"] = 0
        return
    state["hp"] -= amount
    pop_lost_heart(state)
    state["flash"] = ("Hit -1", 40)
    state["hit_flash"] = 18
    state["shake"] = 12
    state["combo"] = 0
    sfx.play_hit()
    if state["hp"] <= 0:
        state["hp"] = 0
        state["mode"] = "game_over"
        state["best_wave"] = max(state.get("best_wave", 1), state["wave"])
        sfx.play_death()


def try_fist_restart(state, hand: HandController):
    """Hold both fists → return to difficulty menu (restart)."""
    if not state.get("hand_enabled"):
        state["fist_hold"] = 0
        return state
    if state["mode"] not in ("game_over", "playing", "upgrade", "shop"):
        state["fist_hold"] = 0
        return state
    need = 14 if state["mode"] == "game_over" else 28
    if hand.last_dual_fist:
        state["fist_hold"] = state.get("fist_hold", 0) + 1
        if state["fist_hold"] >= need:
            sfx.play_click()
            return new_menu(hand_enabled=state["hand_enabled"])
    else:
        state["fist_hold"] = 0
    return state


def update_shop_gesture(state, hand: HandController):
    """Shop: highlight cards / refresh / continue; thumbs-up confirms."""
    if state["mode"] != "shop":
        return
    if not (state["hand_enabled"] and hand.enabled):
        return

    target = hand.last_shop_target
    # Keep highlight while confirming with thumbs-up
    if hand.last_thumbs_up:
        if state["gesture_hover"] is None:
            state["gesture_hover"] = target
    else:
        state["gesture_hover"] = target

    hover = state["gesture_hover"]
    if hand.last_thumbs_up and hover is not None:
        key = ("shop", hover)
        if state["gesture_hold_id"] == key:
            state["gesture_hold"] += 1
        else:
            state["gesture_hold_id"] = key
            state["gesture_hold"] = 1
        if state["gesture_hold"] >= 12:
            if hover in (1, 2, 3):
                buy_shop_item(state, hover - 1)
            elif hover == "refresh":
                refresh_shop(state)
            elif hover == "continue":
                leave_shop(state)
            state["gesture_hold"] = 0
            state["gesture_hold_id"] = None
    else:
        state["gesture_hold"] = 0
        state["gesture_hold_id"] = None


def on_wave_clear(state):
    state["stars"].clear()
    state["pending_splits"] = []
    state["frenzy_timer"] = 0
    if should_open_shop(state["wave"]):
        enter_shop(state)
    else:
        enter_upgrade(state)


def apply_choice(state, index: int):
    if state["mode"] != "upgrade":
        return
    if index < 0 or index >= len(state["choices"]):
        return
    card = state["choices"][index]
    card["fn"](state)
    state["upgrades_taken"].append(card["id"])
    state["wave"] += 1
    preset = DIFFICULTIES.get(state.get("difficulty_key", "normal"), DIFFICULTIES["normal"])
    state["difficulty"] = min(8.0, state["difficulty"] + 0.7)
    state["base_star_speed"] = min(
        state["max_star_speed"],
        preset["speed"] + state["wave"] * 0.55,
    )
    state["star_speed"] = state["base_star_speed"]
    state["star_accel"] = min(0.16, preset["accel"] + state["wave"] * 0.01)
    state["mode"] = "playing"
    state["choices"] = []
    state["choice_rects"] = []
    state["gesture_hold"] = 0
    state["gesture_hold_id"] = None
    state["gesture_hover"] = None
    begin_wave(state)
    sfx.play_click()


def start_difficulty(state, key: str):
    hand_on = state.get("hand_enabled", True)
    new_state = reset_run(hand_enabled=hand_on, difficulty_key=key)
    sfx.play_click()
    return new_state


def update_menu_gesture(state, hand: HandController):
    if state["mode"] != "menu":
        return state
    if not (state["hand_enabled"] and hand.enabled):
        return state
    hover = hand.last_card_hover
    if hand.last_thumbs_up:
        if state["gesture_hover"] not in (1, 2, 3):
            state["gesture_hover"] = hover
    else:
        state["gesture_hover"] = hover
    if hand.last_thumbs_up and state["gesture_hover"] in (1, 2, 3):
        key = ("menu", state["gesture_hover"])
        if state["gesture_hold_id"] == key:
            state["gesture_hold"] += 1
        else:
            state["gesture_hold_id"] = key
            state["gesture_hold"] = 1
        if state["gesture_hold"] >= 12:
            diff_key = DIFF_ORDER[state["gesture_hover"] - 1]
            return start_difficulty(state, diff_key)
    else:
        state["gesture_hold"] = 0
        state["gesture_hold_id"] = None
    return state


def update_gesture_choice(state, hand: HandController):
    """Upgrade screen: move hand L/C/R to highlight, thumbs-up to confirm."""
    if state["mode"] != "upgrade":
        return
    if not (state["hand_enabled"] and hand.enabled):
        return

    # While confirming with thumbs-up, freeze the highlighted card
    if hand.last_thumbs_up:
        if state["gesture_hover"] not in (1, 2, 3):
            state["gesture_hover"] = hand.last_card_hover
    else:
        state["gesture_hover"] = hand.last_card_hover

    if hand.last_thumbs_up and state["gesture_hover"] in (1, 2, 3):
        key = ("thumb", state["gesture_hover"])
        if state["gesture_hold_id"] == key:
            state["gesture_hold"] += 1
        else:
            state["gesture_hold_id"] = key
            state["gesture_hold"] = 1
        # ~0.2s confirm
        if state["gesture_hold"] >= 12:
            apply_choice(state, state["gesture_hover"] - 1)
    else:
        state["gesture_hold"] = 0
        state["gesture_hold_id"] = None


# ---------------------------------------------------------------------------
# 更新 / 绘制
# ---------------------------------------------------------------------------


def move_paddle(state, move_left, move_right, control_x=None, from_hand=False):
    paddle = state["paddle"]
    if control_x is not None:
        if from_hand:
            dx = float(control_x) - paddle.centerx
            step = dx * state["hand_follow"]
            step = max(-state["hand_max_step"], min(state["hand_max_step"], step))
            if abs(dx) < 2:
                step = 0
            paddle.centerx = int(paddle.centerx + step)
        else:
            paddle.centerx = int(control_x)
    else:
        if move_left:
            paddle.x -= state["paddle_speed"]
        if move_right:
            paddle.x += state["paddle_speed"]
    paddle.x = max(0, min(paddle.x, WIDTH - paddle.width))


def update_playing(state, move_left, move_right, control_x=None, from_hand=False):
    if state["mode"] != "playing":
        return

    move_paddle(state, move_left, move_right, control_x, from_hand)

    if state.get("slowmo_timer", 0) > 0:
        state["slowmo_timer"] -= 1
    if state.get("magnet_pulse_timer", 0) > 0:
        state["magnet_pulse_timer"] -= 1
    if state.get("gold_rush_timer", 0) > 0:
        state["gold_rush_timer"] -= 1

    # Frenzy ends by timer
    if state.get("wave_kind") == "frenzy":
        state["frenzy_timer"] = max(0, state.get("frenzy_timer", 0) - 1)
        if state["frenzy_timer"] <= 0:
            on_wave_clear(state)
            return

    state["star_speed"] = min(
        state["star_speed"] + state["star_accel"] * 0.55,
        state["base_star_speed"] + 4.5,
        state["max_star_speed"],
    )
    fall = state["star_speed"] * state["speed_mul"]
    if state.get("slowmo_timer", 0) > 0:
        fall *= 0.35
    if state.get("wave_kind") == "frenzy":
        fall *= 1.15  # a bit faster rain

    frenzy = state.get("wave_kind") == "frenzy"
    score_mul = float(state["score_mult"]) * (2.0 if frenzy else 1.0)
    if state.get("gold_rush_timer", 0) > 0:
        score_mul *= 1.5

    magnet = state["magnet"]
    if state.get("magnet_pulse_timer", 0) > 0:
        magnet = magnet + 220

    quota = wave_quota_for(state)
    paddle = state["paddle"]
    kept = []
    spawned = list(state.get("pending_splits") or [])
    state["pending_splits"] = []

    for star in state["stars"]:
        step_star(star, fall, paddle, magnet)

        if paddle.colliderect(star["rect"]):
            st = star["stype"]
            pos = star["rect"].center
            if st == "bomb":
                take_damage(state, 1)
                state["flash"] = ("Bomb!", 28)
                add_popup(state, "BOOM", pos, ORANGE, size="med", life=36)
                spawn_burst(state, pos, ORANGE, 18)
                spawn_burst(state, pos, RED, 8)
            elif st == "boss":
                gained = int(20 * score_mul)
                state["score"] += gained
                state["caught_in_wave"] = max(state["caught_in_wave"] + 1, quota)
                state["combo"] += 1
                state["boss_victory_timer"] = 100
                juice_catch(state, pos, gained, "gold")
                add_popup(state, "BOSS DOWN!", pos, PURPLE, size="big", life=55)
                spawn_burst(state, pos, PURPLE, 28)
                spawn_burst(state, pos, GOLD, 16)
                grant_boss_drop(state)
                # Split into bonus stars
                for i in range(4):
                    sx = pos[0] - 40 + i * 28
                    stype = "gold" if i % 2 == 0 else "normal"
                    spawned.append(make_split_star(sx, pos[1] - 10, stype))
                if random.random() < 0.35 + state["luck"]:
                    spawned.append(make_split_star(pos[0], pos[1] - 30, "heart"))
                sfx.play_wave()
            elif st == "heart":
                if state["hp"] < state["hp_max"]:
                    state["hp"] += 1
                gained = int(1 * score_mul)
                state["score"] += gained
                state["caught_in_wave"] += 1
                state["combo"] += 1
                state["flash"] = ("Heal!", 30)
                juice_catch(state, pos, gained, "heart")
            elif st == "gold":
                gained = int(3 * score_mul)
                state["score"] += gained
                state["caught_in_wave"] += 1
                state["combo"] += 1
                juice_catch(state, pos, gained, "gold")
            else:
                gained = int(1 * score_mul)
                state["score"] += gained
                state["caught_in_wave"] += 1
                state["combo"] += 1
                juice_catch(state, pos, gained, "normal")
            continue

        if star["rect"].top > HEIGHT:
            if star["stype"] == "bomb":
                pass
            elif frenzy:
                pass
            elif state["items"].get("ghost", 0) > 0:
                state["items"]["ghost"] -= 1
                state["flash"] = ("Ghost saved you!", 35)
                add_popup(state, "SAVED", star["rect"].center, PURPLE, size="med")
                sfx.play_shield()
            else:
                take_damage(state, 1)
            continue

        kept.append(star)

    state["stars"] = kept + spawned

    if state["mode"] == "game_over":
        return

    if state.get("wave_kind") != "frenzy" and state["caught_in_wave"] >= quota:
        if state.get("boss_victory_timer", 0) > 0:
            state["boss_victory_timer"] -= 1
        else:
            on_wave_clear(state)
            return

    _refill_stars(state)


def draw_bomb(star):
    """画一颗小炸弹：黑球 + 引线 + 火花。"""
    cx, cy = star["rect"].center
    r = star["rect"].width // 2
    # 球体
    pygame.draw.circle(screen, (35, 35, 40), (cx, cy), r)
    pygame.draw.circle(screen, (90, 90, 100), (cx, cy), r, 2)
    # 高光
    pygame.draw.circle(screen, (70, 70, 80), (cx - 4, cy - 4), max(2, r // 4))
    # 引线
    fuse_end = (cx + 6, cy - r - 6)
    pygame.draw.line(screen, (180, 140, 80), (cx + 2, cy - r + 2), fuse_end, 2)
    # 火花（闪烁感用帧时间）
    spark = (255, 200, 60) if (pygame.time.get_ticks() // 100) % 2 == 0 else (255, 80, 40)
    pygame.draw.circle(screen, spark, fuse_end, 3)


def draw_boss(star):
    cx, cy = star["rect"].center
    r = star["rect"].width // 2
    pulse = 1.0 + 0.06 * math.sin(pygame.time.get_ticks() / 120)
    rr = int(r * pulse)
    pygame.draw.circle(screen, (60, 30, 90), (cx, cy), rr + 6)
    pygame.draw.circle(screen, PURPLE, (cx, cy), rr)
    pygame.draw.circle(screen, (255, 210, 120), (cx, cy), rr, 3)
    # crown dots
    for ang in (-0.6, 0, 0.6):
        px = int(cx + math.sin(ang) * rr * 0.55)
        py = int(cy - rr * 0.75 + abs(ang) * 8)
        pygame.draw.circle(screen, GOLD, (px, py), 5)
    label = small_font.render("BOSS", True, WHITE)
    screen.blit(label, label.get_rect(center=(cx, cy)))


def draw_star(star):
    r = star["rect"]
    if star["stype"] == "bomb":
        draw_bomb(star)
        return
    if star["stype"] == "boss":
        draw_boss(star)
        return
    if star["stype"] == "gold" and GOLD_STAR_IMG is not None:
        # Scale character to star hitbox (a bit larger for readability)
        target = max(r.width, r.height) + 10
        img = GOLD_STAR_IMG
        scale = target / max(img.get_width(), img.get_height())
        w = max(8, int(img.get_width() * scale))
        h = max(8, int(img.get_height() * scale))
        sprite = pygame.transform.smoothscale(img, (w, h))
        screen.blit(sprite, sprite.get_rect(center=r.center))
        return
    pygame.draw.circle(screen, star["color"], r.center, r.width // 2)
    if star["stype"] == "heart":
        pygame.draw.circle(screen, WHITE, r.center, r.width // 2, 1)


def draw_paddle(paddle_rect, punch: float = 0.0):
    """Draw paddle character; width change scales the whole sprite."""
    scale = 1.0 + punch * 0.045
    if PADDLE_IMG is None:
        w = max(8, int(paddle_rect.width * scale))
        h = max(8, int(paddle_rect.height * scale))
        r = pygame.Rect(0, 0, w, h)
        r.center = paddle_rect.center
        pygame.draw.rect(screen, PADDLE_COLOR, r, border_radius=8)
        return
    # Sprite width tracks paddle hitbox width; height keeps aspect ratio
    w = max(28, int(paddle_rect.width * 1.15 * scale))
    h = max(24, int(w * PADDLE_IMG.get_height() / PADDLE_IMG.get_width()))
    sprite = pygame.transform.smoothscale(PADDLE_IMG, (w, h))
    dest = sprite.get_rect(midbottom=(paddle_rect.centerx, paddle_rect.bottom + 6))
    screen.blit(sprite, dest)


def draw_upgrade(state):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((10, 15, 30, 200))
    screen.blit(overlay, (0, 0))

    title = big_font.render("Pick an Upgrade", True, WHITE)
    screen.blit(title, title.get_rect(center=(WIDTH // 2, 70)))
    sub = small_font.render("Move hand L/C/R, then thumbs-up", True, HINT)
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, 112)))

    if state.get("gesture_hover") in (1, 2, 3):
        hold = state.get("gesture_hold", 0)
        confirming = isinstance(state.get("gesture_hold_id"), tuple)
        pct = min(1.0, hold / 12) if confirming else 0.0
        tip = small_font.render(
            f"Card {state['gesture_hover']} — thumbs-up to confirm",
            True,
            GREEN,
        )
        screen.blit(tip, tip.get_rect(center=(WIDTH // 2, 138)))
        bar = pygame.Rect(WIDTH // 2 - 80, 152, 160, 10)
        pygame.draw.rect(screen, (40, 50, 70), bar, border_radius=4)
        if pct > 0:
            fill = bar.copy()
            fill.width = int(160 * pct)
            pygame.draw.rect(screen, GREEN, fill, border_radius=4)

    state["choice_rects"] = []
    card_w, card_h = 140, 190
    gap = 18
    total_w = 3 * card_w + 2 * gap
    start_x = (WIDTH - total_w) // 2
    y = 180

    for i, card in enumerate(state["choices"]):
        x = start_x + i * (card_w + gap)
        rect = pygame.Rect(x, y, card_w, card_h)
        state["choice_rects"].append(rect)
        hover = state.get("gesture_hover") == i + 1
        border = GREEN if hover else CARD_BORDER
        pygame.draw.rect(screen, CARD_BG, rect, border_radius=12)
        pygame.draw.rect(screen, border, rect, 3 if hover else 2, border_radius=12)
        key = font.render(str(i + 1), True, GREEN)
        screen.blit(key, (x + 12, y + 10))
        name = small_font.render(card["name"], True, WHITE)
        screen.blit(name, name.get_rect(centerx=rect.centerx, y=y + 50))
        words = card["desc"].split()
        line1, line2 = "", ""
        for w in words:
            trial = (line1 + " " + w).strip()
            if len(trial) <= 14:
                line1 = trial
            else:
                line2 = (line2 + " " + w).strip()
        d1 = small_font.render(line1, True, HINT)
        screen.blit(d1, d1.get_rect(centerx=rect.centerx, y=y + 100))
        if line2:
            d2 = small_font.render(line2, True, HINT)
            screen.blit(d2, d2.get_rect(centerx=rect.centerx, y=y + 122))


def draw_shop(state):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((15, 25, 20, 210))
    screen.blit(overlay, (0, 0))

    title = big_font.render("Shop", True, GOLD)
    screen.blit(title, title.get_rect(center=(WIDTH // 2, 48)))
    sub = small_font.render(
        f"Score {state['score']}   Skip free · Refresh costs score",
        True,
        HINT,
    )
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, 88)))
    tip = small_font.render(
        "Swipe down → Refresh/Continue, thumbs-up to confirm",
        True,
        GREEN,
    )
    screen.blit(tip, tip.get_rect(center=(WIDTH // 2, 112)))

    items = state.get("items", {})
    inv = small_font.render(
        f"Bag Q{items.get('nuke', 0)} E{items.get('slow', 0)} F{items.get('heart', 0)} "
        f"Ghost{items.get('ghost', 0)} G{items.get('magnet_pulse', 0)} T{items.get('gold_rush', 0)}",
        True,
        WHITE,
    )
    screen.blit(inv, inv.get_rect(center=(WIDTH // 2, 136)))

    hover = state.get("gesture_hover")
    confirming = (
        isinstance(state.get("gesture_hold_id"), tuple)
        and state["gesture_hold_id"][0] == "shop"
        and state.get("gesture_hold", 0) > 0
    )
    if confirming:
        pct = min(1.0, state.get("gesture_hold", 0) / 12)
        bar = pygame.Rect(WIDTH // 2 - 80, 154, 160, 8)
        pygame.draw.rect(screen, (40, 50, 70), bar, border_radius=4)
        fill = bar.copy()
        fill.width = int(160 * pct)
        pygame.draw.rect(screen, GOLD, fill, border_radius=4)

    state["shop_rects"] = []
    offers = state.get("shop_offers") or []
    card_w, card_h = 140, 190
    gap = 16
    total_w = 3 * card_w + 2 * gap
    start_x = (WIDTH - total_w) // 2
    y = 170
    for i, item in enumerate(offers):
        x = start_x + i * (card_w + gap)
        rect = pygame.Rect(x, y, card_w, card_h)
        state["shop_rects"].append(rect)
        price = shop_price(item, state["wave"])
        sold = item.get("sold", False)
        afford = (not sold) and state["score"] >= price
        is_hover = hover == i + 1
        border = GREEN if is_hover else (CARD_BORDER if afford else (90, 70, 70))
        pygame.draw.rect(screen, CARD_BG, rect, border_radius=12)
        pygame.draw.rect(screen, border, rect, 3 if is_hover else 2, border_radius=12)
        screen.blit(font.render(str(i + 1), True, GREEN), (x + 12, y + 8))
        kind_c = PURPLE if item.get("kind") == "buff" else HINT
        screen.blit(
            small_font.render("BUFF" if item.get("kind") == "buff" else "ITEM", True, kind_c),
            (x + 40, y + 12),
        )
        name = small_font.render(item["name"], True, WHITE if afford else HINT)
        screen.blit(name, name.get_rect(centerx=rect.centerx, y=y + 42))
        words = item["desc"].split()
        line1 = " ".join(words[:3])
        line2 = " ".join(words[3:])
        d1 = small_font.render(line1, True, HINT)
        screen.blit(d1, d1.get_rect(centerx=rect.centerx, y=y + 78))
        if line2:
            d2 = small_font.render(line2, True, HINT)
            screen.blit(d2, d2.get_rect(centerx=rect.centerx, y=y + 98))
        if sold:
            pr = font.render("SOLD", True, RED)
        else:
            pr = font.render(f"{price} pts", True, GOLD if afford else RED)
        screen.blit(pr, pr.get_rect(centerx=rect.centerx, y=y + 140))

    refresh_cost = shop_refresh_price(state)
    ref = pygame.Rect(40, 400, 190, 48)
    cont = pygame.Rect(WIDTH - 230, 400, 190, 48)
    state["shop_rects"].append(ref)
    state["shop_rects"].append(cont)

    ref_hover = hover == "refresh"
    cont_hover = hover == "continue"
    pygame.draw.rect(screen, (70, 55, 40), ref, border_radius=10)
    pygame.draw.rect(
        screen, ORANGE if ref_hover else (180, 140, 80), ref, 3 if ref_hover else 2, border_radius=10
    )
    rt = small_font.render(f"Refresh  {refresh_cost} pts", True, WHITE)
    screen.blit(rt, rt.get_rect(center=ref.center))
    rkey = small_font.render("R / swipe L + thumb", True, HINT)
    screen.blit(rkey, rkey.get_rect(centerx=ref.centerx, y=ref.bottom + 4))

    pygame.draw.rect(screen, (40, 90, 60), cont, border_radius=10)
    pygame.draw.rect(
        screen, GREEN if cont_hover else (80, 180, 100), cont, 3 if cont_hover else 2, border_radius=10
    )
    ct = font.render("Continue", True, WHITE)
    screen.blit(ct, ct.get_rect(center=cont.center))
    ckey = small_font.render("Space / swipe R + thumb", True, HINT)
    screen.blit(ckey, ckey.get_rect(centerx=cont.centerx, y=cont.bottom + 4))


def draw_menu(state):
    screen.fill(SKY)
    title = big_font.render("Catch the Stars", True, YELLOW)
    sub = font.render("Roguelike", True, HINT)
    screen.blit(title, title.get_rect(center=(WIDTH // 2, 90)))
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, 140)))
    tip = small_font.render("Choose difficulty — 1/2/3, click, or thumbs-up", True, HINT)
    screen.blit(tip, tip.get_rect(center=(WIDTH // 2, 180)))

    state["menu_rects"] = []
    card_w, card_h = 140, 160
    gap = 18
    total = 3 * card_w + 2 * gap
    start_x = (WIDTH - total) // 2
    y = 240
    for i, key in enumerate(DIFF_ORDER):
        preset = DIFFICULTIES[key]
        x = start_x + i * (card_w + gap)
        rect = pygame.Rect(x, y, card_w, card_h)
        state["menu_rects"].append(rect)
        hover = state.get("gesture_hover") == i + 1
        border = GREEN if hover else CARD_BORDER
        pygame.draw.rect(screen, CARD_BG, rect, border_radius=12)
        pygame.draw.rect(screen, border, rect, 3 if hover else 2, border_radius=12)
        screen.blit(font.render(str(i + 1), True, GREEN), (x + 12, y + 10))
        name = font.render(preset["label"], True, WHITE)
        screen.blit(name, name.get_rect(centerx=rect.centerx, y=y + 50))
        words = preset["blurb"].split()
        line1 = " ".join(words[:2])
        line2 = " ".join(words[2:])
        d1 = small_font.render(line1, True, HINT)
        screen.blit(d1, d1.get_rect(centerx=rect.centerx, y=y + 95))
        if line2:
            d2 = small_font.render(line2, True, HINT)
            screen.blit(d2, d2.get_rect(centerx=rect.centerx, y=y + 118))

    if state.get("gesture_hover") in (1, 2, 3) and state.get("gesture_hold", 0) > 0:
        pct = min(1.0, state["gesture_hold"] / 12)
        bar = pygame.Rect(WIDTH // 2 - 80, 430, 160, 10)
        pygame.draw.rect(screen, (40, 50, 70), bar, border_radius=4)
        fill = bar.copy()
        fill.width = int(160 * pct)
        pygame.draw.rect(screen, GREEN, fill, border_radius=4)


def draw(state, hand: HandController):
    if state["mode"] == "menu":
        draw_menu(state)
        if hand.last_preview is not None and state["hand_enabled"]:
            surf = pygame.image.frombuffer(
                hand.last_preview.tobytes(), (160, 120), "RGB"
            )
            screen.blit(surf, (WIDTH - 172, 10))
            pygame.draw.rect(screen, WHITE, (WIDTH - 172, 10, 160, 120), 2)
        screen.blit(small_font.render(hand.status, True, HINT), (14, HEIGHT - 28))
        pygame.display.flip()
        return

    ox = oy = 0
    if state.get("shake", 0) > 0:
        ox = random.randint(-5, 5)
        oy = random.randint(-4, 4)

    screen.fill(SKY)

    if state.get("magnet", 0) > 0 and state["mode"] == "playing":
        aura_r = int(state["magnet"] + 80)
        aura = pygame.Surface((aura_r * 2, aura_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(aura, (100, 200, 255, 35), (aura_r, aura_r), aura_r)
        pygame.draw.circle(aura, (100, 200, 255, 80), (aura_r, aura_r), aura_r, 2)
        screen.blit(
            aura,
            (
                state["paddle"].centerx - aura_r + ox,
                state["paddle"].centery - aura_r + oy,
            ),
        )

    draw_paddle(state["paddle"].move(ox, oy), punch=float(state.get("paddle_punch", 0)))

    if state["mode"] == "playing":
        for star in state["stars"]:
            old = star["rect"].topleft
            star["rect"].x += ox
            star["rect"].y += oy
            draw_star(star)
            star["rect"].topleft = old

    for part in state.get("particles", []):
        alpha_life = max(0, part["life"])
        r = max(1, int(part["r"] * (0.55 + 0.45 * min(1.0, alpha_life / 20))))
        pygame.draw.circle(
            screen,
            part["color"],
            (int(part["x"] + ox), int(part["y"] + oy)),
            r,
        )
    quota = wave_quota_for(state)
    diff_label = DIFFICULTIES.get(state.get("difficulty_key", "normal"), {}).get(
        "label", "Normal"
    )
    kind = state.get("wave_kind", "normal")
    kind_tag = {"boss": "BOSS", "frenzy": "FRENZY", "normal": ""}.get(kind, "")
    if kind == "frenzy":
        progress = f"TIME {max(0, state.get('frenzy_timer', 0) // FPS)}s"
    elif kind == "boss":
        progress = "Catch BOSS!"
    else:
        progress = f"{state['caught_in_wave']}/{quota}"
    mul_txt = state["score_mult"] * (2 if kind == "frenzy" else 1)
    if state.get("gold_rush_timer", 0) > 0:
        mul_txt = round(mul_txt * 1.5, 1)
    lines = [
        f"Wave {state['wave']}  {progress}  [{diff_label}] {kind_tag}",
        f"Score {state['score']}  x{mul_txt}  Combo {state.get('combo', 0)}",
    ]
    for i, t in enumerate(lines):
        screen.blit(small_font.render(t, True, WHITE), (14 + ox, 10 + i * 22 + oy))

    draw_hp_hearts(state, ox, oy)

    # Flying hearts when HP drops
    for hp_pop in state.get("heart_pops", []):
        fade = min(1.0, hp_pop["life"] / 10.0)
        sz = hp_pop["size"] * (1.0 + (36 - hp_pop["life"]) * 0.03)
        # draw with temp surface for alpha
        tmp = pygame.Surface((40, 40), pygame.SRCALPHA)
        col = (*HEART_RED, int(220 * fade))
        draw_heart_shape(tmp, 20, 18, sz, col)
        screen.blit(
            tmp,
            (int(hp_pop["x"] - 20 + ox), int(hp_pop["y"] - 18 + oy)),
        )

    items = state.get("items", {})
    bag = (
        f"Q{items.get('nuke', 0)} E{items.get('slow', 0)} F{items.get('heart', 0)} "
        f"Gh{items.get('ghost', 0)} G{items.get('magnet_pulse', 0)} T{items.get('gold_rush', 0)}"
    )
    if state.get("slowmo_timer", 0) > 0:
        bag += f"  SLOW{state['slowmo_timer'] // FPS + 1}"
    if state.get("magnet_pulse_timer", 0) > 0:
        bag += "  MAG"
    if state.get("gold_rush_timer", 0) > 0:
        bag += "  RUSH"
    screen.blit(small_font.render(bag, True, HINT), (14, 86))

    legend = small_font.render(
        "Yellow=pts  Gold$$$  Pink=heal  Bomb=dodge  Purple=BOSS", True, HINT
    )
    screen.blit(legend, (14, 108))

    status_color = (
        GREEN
        if "thumbs" in hand.status or "highlight" in hand.status or "fist" in hand.status
        else HINT
    )
    screen.blit(small_font.render(hand.status, True, status_color), (14, 130))

    if state.get("upgrades_taken"):
        tags = ",".join(state["upgrades_taken"][-4:])
        screen.blit(small_font.render(f"Picked: {tags}", True, HINT), (14, 152))

    screen.blit(
        small_font.render("Q/E/F items | fists restart | R menu", True, HINT),
        (14, HEIGHT - 28),
    )

    if state.get("fist_hold", 0) > 0 and state["mode"] in (
        "game_over",
        "playing",
        "upgrade",
        "shop",
    ):
        need = 14 if state["mode"] == "game_over" else 28
        pct = min(1.0, state["fist_hold"] / need)
        bar = pygame.Rect(WIDTH // 2 - 90, HEIGHT - 48, 180, 10)
        pygame.draw.rect(screen, (40, 50, 70), bar, border_radius=4)
        fill = bar.copy()
        fill.width = int(180 * pct)
        pygame.draw.rect(screen, ORANGE, fill, border_radius=4)
        tip = small_font.render("Both fists → Restart", True, ORANGE)
        screen.blit(tip, tip.get_rect(centerx=WIDTH // 2, y=HEIGHT - 70))

    for p in state.get("popups", []):
        fnt = small_font
        if p.get("size") == "med":
            fnt = font
        elif p.get("size") == "big":
            fnt = big_font
        # pop-in scale via alpha-ish fade at end
        surf = fnt.render(p["text"], True, p["color"])
        fade = min(1.0, p["life"] / 8.0) if p.get("max_life") else 1.0
        if fade < 1.0:
            surf = surf.copy()
            surf.set_alpha(int(255 * fade))
        screen.blit(surf, (int(p["x"] - surf.get_width() / 2 + ox), int(p["y"] + oy)))

    if hand.last_preview is not None and state["hand_enabled"]:
        surf = pygame.image.frombuffer(hand.last_preview.tobytes(), (160, 120), "RGB")
        screen.blit(surf, (WIDTH - 172, 10))
        pygame.draw.rect(screen, WHITE, (WIDTH - 172, 10, 160, 120), 2)

    if state.get("flash"):
        txt, _ = state["flash"]
        fs = font.render(txt, True, GREEN)
        screen.blit(fs, fs.get_rect(center=(WIDTH // 2 + ox, HEIGHT // 2 - 80 + oy)))

    if state.get("score_flash", 0) > 0:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        a = min(110, state["score_flash"] * 9)
        overlay.fill((255, 230, 120, a))
        screen.blit(overlay, (0, 0))

    if state.get("hit_flash", 0) > 0:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((180, 30, 30, min(140, state["hit_flash"] * 8)))
        screen.blit(overlay, (0, 0))
    if state["mode"] == "upgrade":
        draw_upgrade(state)

    if state["mode"] == "shop":
        draw_shop(state)

    if state["mode"] == "game_over":
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((20, 10, 20, 190))
        screen.blit(overlay, (0, 0))
        over = big_font.render("Run Over", True, RED)
        tip = font.render(
            f"Reached wave {state['wave']}  Score {state['score']}", True, WHITE
        )
        tip2 = font.render("R or both fists to restart", True, HINT)
        screen.blit(over, over.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40)))
        screen.blit(tip, tip.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 10)))
        screen.blit(tip2, tip2.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50)))

    pygame.display.flip()


def tick_flash(state):
    if state.get("flash"):
        txt, frames = state["flash"]
        frames -= 1
        state["flash"] = (txt, frames) if frames > 0 else None
    if state.get("hit_flash", 0) > 0:
        state["hit_flash"] -= 1
    if state.get("score_flash", 0) > 0:
        state["score_flash"] -= 1
    if state.get("paddle_punch", 0) > 0:
        state["paddle_punch"] -= 1
    if state.get("shake", 0) > 0:
        state["shake"] -= 1
    alive = []
    for p in state.get("popups", []):
        p["y"] -= 1.6 if p.get("size") in ("med", "big") else 1.2
        p["life"] -= 1
        if p["life"] > 0:
            alive.append(p)
    if "popups" in state:
        state["popups"] = alive

    parts = []
    for part in state.get("particles", []):
        part["x"] += part["vx"]
        part["y"] += part["vy"]
        part["vy"] += 0.22
        part["vx"] *= 0.98
        part["life"] -= 1
        if part["life"] > 0:
            parts.append(part)
    if "particles" in state:
        state["particles"] = parts

    pops = []
    for hp_pop in state.get("heart_pops", []):
        hp_pop["y"] += hp_pop["vy"]
        hp_pop["vy"] -= 0.08
        hp_pop["life"] -= 1
        hp_pop["size"] += 0.35
        if hp_pop["life"] > 0:
            pops.append(hp_pop)
    if "heart_pops" in state:
        state["heart_pops"] = pops


def main():
    global state
    hand = HandController()
    state = new_menu(hand_enabled=True)
    sfx.start_bgm()
    print(hand.status)
    print("Catch the Stars: pick Easy / Normal / Hard, then climb waves")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key in held:
                        held[event.key] = True
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_h:
                        state["hand_enabled"] = not state["hand_enabled"]
                        if not state["hand_enabled"]:
                            hand.status = "Hand: off (press H)"
                    if event.key == pygame.K_r and state["mode"] == "game_over":
                        state = new_menu(hand_enabled=state["hand_enabled"])
                    elif event.key == pygame.K_r and state["mode"] == "shop":
                        refresh_shop(state)
                    if state["mode"] == "upgrade":
                        if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                            apply_choice(state, event.key - pygame.K_1)
                    if state["mode"] == "shop":
                        if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                            buy_shop_item(state, event.key - pygame.K_1)
                        if event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_0):
                            leave_shop(state)
                    if state["mode"] == "playing":
                        if event.key == pygame.K_q:
                            use_item(state, "nuke")
                        if event.key == pygame.K_e:
                            use_item(state, "slow")
                        if event.key == pygame.K_f:
                            use_item(state, "heart")
                        if event.key == pygame.K_g:
                            use_item(state, "magnet_pulse")
                        if event.key == pygame.K_t:
                            use_item(state, "gold_rush")
                    if state["mode"] == "menu":
                        if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                            state = start_difficulty(
                                state, DIFF_ORDER[event.key - pygame.K_1]
                            )
                if event.type == pygame.KEYUP:
                    if event.key in held:
                        held[event.key] = False
                if event.type == pygame.WINDOWFOCUSLOST:
                    for k in held:
                        held[k] = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if state["mode"] == "upgrade":
                        for i, rect in enumerate(state["choice_rects"]):
                            if rect.collidepoint(event.pos):
                                apply_choice(state, i)
                                break
                    elif state["mode"] == "shop":
                        rects = state.get("shop_rects", [])
                        for i, rect in enumerate(rects):
                            if not rect.collidepoint(event.pos):
                                continue
                            if i < 3:
                                buy_shop_item(state, i)
                            elif i == 3:
                                refresh_shop(state)
                            else:
                                leave_shop(state)
                            break
                    elif state["mode"] == "menu":
                        for i, rect in enumerate(state.get("menu_rects", [])):
                            if rect.collidepoint(event.pos):
                                state = start_difficulty(state, DIFF_ORDER[i])
                                break

            control_x = None
            from_hand = False
            if state["hand_enabled"] and hand.enabled:
                x_norm = hand.update()
                if state["mode"] == "playing" and x_norm is not None:
                    control_x = x_norm * WIDTH
                    from_hand = True
                if state["mode"] == "upgrade":
                    update_gesture_choice(state, hand)
                if state["mode"] == "shop":
                    update_shop_gesture(state, hand)
                if state["mode"] == "menu":
                    state = update_menu_gesture(state, hand)
                state = try_fist_restart(state, hand)

            if (
                control_x is None
                and pygame.mouse.get_focused()
                and state["mode"] == "playing"
            ):
                control_x = pygame.mouse.get_pos()[0]

            pressed = pygame.key.get_pressed()
            move_left = (
                held[pygame.K_LEFT]
                or held[pygame.K_a]
                or pressed[pygame.K_LEFT]
                or pressed[pygame.K_a]
            )
            move_right = (
                held[pygame.K_RIGHT]
                or held[pygame.K_d]
                or pressed[pygame.K_RIGHT]
                or pressed[pygame.K_d]
            )

            update_playing(state, move_left, move_right, control_x, from_hand)
            tick_flash(state)
            draw(state, hand)
            clock.tick(FPS)
    finally:
        sfx.stop_bgm()
        hand.close()
        pygame.quit()


if __name__ == "__main__":
    main()
