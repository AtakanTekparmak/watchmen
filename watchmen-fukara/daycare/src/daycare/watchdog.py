"""SIGALRM wall-clock watchdog (K9).

Fires at ``budget_seconds - 600`` (10-min grace) so the evolution loop has
time to drain in-flight rollouts and run Phase 4 baselines without
overrunning the user's ``--budget``.

On Unix we use ``signal.SIGALRM`` — fires from the kernel timer, no
Python overhead between checks. On Windows (no SIGALRM) we fall back
to ``threading.Timer``, which is good enough at minute-scale precision.
"""

from __future__ import annotations

import signal
import sys
import threading


# 10-minute grace before the user's nominal budget — gives Phase 4
# baseline runs time to complete after evolution stops.
_GRACE_SECONDS = 600


class Watchdog:
    """Owns a single one-shot timer that sets ``_stop`` when budget nears.

    Usage::

        wd = Watchdog(budget_seconds=8 * 3600)
        wd.start()
        while not wd.should_stop():
            run_one_iter()
        wd.cancel()
    """

    def __init__(self, budget_seconds: int) -> None:
        self.budget_seconds = budget_seconds
        self._stop = False
        # Holds either the threading.Timer (Windows) or None (Unix; SIGALRM
        # state lives in the process-level signal handler instead).
        self._timer: threading.Timer | None = None
        # True if we registered a SIGALRM handler — needed so cancel()
        # only resets handlers we own.
        self._used_sigalrm = False

    def _handler(self, signum, frame) -> None:  # noqa: ARG002 (signal API)
        """SIGALRM handler — flip the stop flag and return."""
        self._stop = True

    def _timer_callback(self) -> None:
        """Cross-platform fallback callback (threading.Timer)."""
        self._stop = True

    def start(self) -> None:
        """Arm the timer/alarm. No-op if budget is already too small."""
        fire_in = max(1, self.budget_seconds - _GRACE_SECONDS)

        # signal.SIGALRM only exists on Unix. On Windows fall back to a
        # threading.Timer — signal.alarm raises AttributeError there.
        has_sigalrm = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm") and sys.platform != "win32"

        if has_sigalrm:
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(fire_in)
            self._used_sigalrm = True
        else:
            t = threading.Timer(fire_in, self._timer_callback)
            t.daemon = True
            t.start()
            self._timer = t

    def should_stop(self) -> bool:
        """Cheap flag read — call at every iter boundary."""
        return self._stop

    def cancel(self) -> None:
        """Cancel the pending alarm/timer. Idempotent."""
        if self._used_sigalrm:
            try:
                signal.alarm(0)
            except (AttributeError, ValueError):
                # Already cancelled or platform doesn't support — ignore.
                pass
            # Restore the default handler so we don't leave our handler
            # installed when the process keeps running (tests, daemon).
            try:
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
            except (AttributeError, ValueError):
                pass
            self._used_sigalrm = False

        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
