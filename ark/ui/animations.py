"""Phase 36 — Utilities animations Qt.

Wrappers QPropertyAnimation pour interactions soft :
- fade_in / fade_out (opacity)
- slide_in (position)
- scale_pulse (status indicator breathing)
- bounce_in (notification subtle)
"""
from __future__ import annotations
from PySide6.QtCore import (
    QEasingCurve, QPropertyAnimation, QParallelAnimationGroup,
    QSequentialAnimationGroup, QObject, Property,
)
from PySide6.QtWidgets import QWidget, QGraphicsOpacityEffect

from ark.ui import design_tokens as T


def _easing(name: str) -> QEasingCurve:
    return QEasingCurve(getattr(QEasingCurve.Type, name, QEasingCurve.Type.OutCubic))


def fade_in(widget: QWidget, duration: int = T.DUR_NORMAL,
              easing: str = T.EASE_STANDARD) -> QPropertyAnimation:
    """Opacity 0 → 1 sur widget."""
    eff = widget.graphicsEffect()
    if not isinstance(eff, QGraphicsOpacityEffect):
        eff = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(eff)
    eff.setOpacity(0.0)
    anim = QPropertyAnimation(eff, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(_easing(easing))
    anim.start()
    return anim


def fade_out(widget: QWidget, duration: int = T.DUR_NORMAL,
               easing: str = T.EASE_STANDARD) -> QPropertyAnimation:
    """Opacity 1 → 0 sur widget."""
    eff = widget.graphicsEffect()
    if not isinstance(eff, QGraphicsOpacityEffect):
        eff = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(eff)
    anim = QPropertyAnimation(eff, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(eff.opacity())
    anim.setEndValue(0.0)
    anim.setEasingCurve(_easing(easing))
    anim.start()
    return anim


def pulse_opacity(widget: QWidget, min_op: float = 0.4, max_op: float = 1.0,
                    duration: int = T.DUR_PULSE) -> QSequentialAnimationGroup:
    """Pulse opacity continu (breathing) — pour status indicator training."""
    eff = widget.graphicsEffect()
    if not isinstance(eff, QGraphicsOpacityEffect):
        eff = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(eff)
    seq = QSequentialAnimationGroup(widget)
    a1 = QPropertyAnimation(eff, b"opacity")
    a1.setDuration(duration // 2)
    a1.setStartValue(max_op)
    a1.setEndValue(min_op)
    a1.setEasingCurve(_easing(T.EASE_IN_OUT))
    a2 = QPropertyAnimation(eff, b"opacity")
    a2.setDuration(duration // 2)
    a2.setStartValue(min_op)
    a2.setEndValue(max_op)
    a2.setEasingCurve(_easing(T.EASE_IN_OUT))
    seq.addAnimation(a1)
    seq.addAnimation(a2)
    seq.setLoopCount(-1)  # infini
    seq.start()
    return seq


def slide_in(widget: QWidget, from_offset: tuple[int, int] = (-40, 0),
                duration: int = T.DUR_NORMAL,
                easing: str = T.EASE_EMPHASIZED) -> QPropertyAnimation:
    """Slide widget depuis offset → position actuelle."""
    target_pos = widget.pos()
    start_pos = target_pos + type(target_pos)(*from_offset)
    widget.move(start_pos)
    anim = QPropertyAnimation(widget, b"pos", widget)
    anim.setDuration(duration)
    anim.setStartValue(start_pos)
    anim.setEndValue(target_pos)
    anim.setEasingCurve(_easing(easing))
    anim.start()
    return anim


def animate_property(target: QObject, prop: bytes,
                       end_value, duration: int = T.DUR_NORMAL,
                       easing: str = T.EASE_STANDARD,
                       start_value=None) -> QPropertyAnimation:
    """Helper générique animateur de propriété Qt."""
    anim = QPropertyAnimation(target, prop, target)
    anim.setDuration(duration)
    if start_value is not None:
        anim.setStartValue(start_value)
    anim.setEndValue(end_value)
    anim.setEasingCurve(_easing(easing))
    anim.start()
    return anim


# ============================================================
# QObject avec property animée (pour QGraphicsItem qui n'a pas de
# système property natif — proxy via QObject)
# ============================================================

class AnimatedFloat(QObject):
    """Proxy QObject pour animer une valeur float sur un QGraphicsItem."""

    def __init__(self, item, attr: str, parent=None):
        super().__init__(parent)
        self._item = item
        self._attr = attr

    def get_value(self) -> float:
        return float(getattr(self._item, self._attr, 0.0))

    def set_value(self, v: float):
        setattr(self._item, self._attr, v)
        if hasattr(self._item, "update"):
            self._item.update()

    value = Property(float, get_value, set_value)


def animate_item_float(item, attr: str, end_value: float,
                          duration: int = T.DUR_NORMAL,
                          easing: str = T.EASE_STANDARD) -> tuple:
    """Anime un attribut float d'un QGraphicsItem (n'a pas de QProperty natif)."""
    proxy = AnimatedFloat(item, attr)
    anim = QPropertyAnimation(proxy, b"value")
    anim.setDuration(duration)
    anim.setStartValue(getattr(item, attr, 0.0))
    anim.setEndValue(end_value)
    anim.setEasingCurve(_easing(easing))
    anim.start()
    return anim, proxy  # garde ref proxy pour qu'il survive
