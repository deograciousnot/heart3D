// Shared page helpers: theme toggle (remembered per browser), token lookup, heartbeat pulse.
(function () {
  const KEY = "heart3d-theme";
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  } catch {}

  const isDark = () =>
    root.dataset.theme ? root.dataset.theme === "dark" : !matchMedia("(prefers-color-scheme: light)").matches;

  window.HeartUI = {
    /** Current value of a CSS colour token, e.g. token("--qrs"). */
    token: name => getComputedStyle(root).getPropertyValue(name).trim(),
    isDark,
    /** Calls fn whenever the theme changes (toggle or system setting). */
    onThemeChange(fn) {
      window.addEventListener("heart3d-theme", fn);
      matchMedia("(prefers-color-scheme: light)").addEventListener("change", fn);
    },
    /** One heartbeat of the logo; call on every QRS. */
    pulse() {
      const mark = document.querySelector(".brand-mark");
      if (!mark || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      mark.classList.remove("beat");
      void mark.offsetWidth; // restart the animation
      mark.classList.add("beat");
    },
  };

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("themeToggle");
    if (!btn) return;
    const sync = () => btn.setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme");
    sync();
    btn.addEventListener("click", () => {
      root.dataset.theme = isDark() ? "light" : "dark";
      try { localStorage.setItem(KEY, root.dataset.theme); } catch {}
      sync();
      window.dispatchEvent(new Event("heart3d-theme"));
    });
  });
})();
