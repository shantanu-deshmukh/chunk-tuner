/**
 * Persisted collapse for the left site nav (lg+ only).
 */
(function () {
  const LAYOUT_ID = "ct-doc-layout";
  const STORAGE_KEY = "chunktuner-docs-sidebar-collapsed";

  function layout() {
    return document.getElementById(LAYOUT_ID);
  }

  function toggleBtn() {
    return document.querySelector(".ct-sidebar-toggle");
  }

  function setCollapsed(collapsed) {
    const el = layout();
    const btn = toggleBtn();
    if (!el || !btn) return;

    el.classList.toggle("ct-sidebar-collapsed", collapsed);
    btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    btn.setAttribute(
      "aria-label",
      collapsed ? "Expand site navigation" : "Collapse site navigation",
    );

    const exp = btn.querySelector(".ct-sidebar-toggle-expanded");
    const col = btn.querySelector(".ct-sidebar-toggle-collapsed");
    if (exp && col) {
      exp.classList.toggle("d-none", collapsed);
      col.classList.toggle("d-none", !collapsed);
    }

    try {
      localStorage.setItem(STORAGE_KEY, collapsed ? "1" : "0");
    } catch {
      /* private mode */
    }
  }

  function readStored() {
    try {
      return localStorage.getItem(STORAGE_KEY) === "1";
    } catch {
      return false;
    }
  }

  function mqLg() {
    return window.matchMedia("(min-width: 992px)");
  }

  function init() {
    const btn = toggleBtn();
    if (!btn) return;

    if (mqLg().matches && readStored()) {
      setCollapsed(true);
    }

    btn.addEventListener("click", function () {
      const el = layout();
      if (!el) return;
      setCollapsed(!el.classList.contains("ct-sidebar-collapsed"));
    });

    mqLg().addEventListener("change", function (e) {
      if (!e.matches) {
        const el = layout();
        if (el) el.classList.remove("ct-sidebar-collapsed");
        const b = toggleBtn();
        if (b) {
          b.setAttribute("aria-expanded", "true");
          b.setAttribute("aria-label", "Collapse site navigation");
          const exp = b.querySelector(".ct-sidebar-toggle-expanded");
          const col = b.querySelector(".ct-sidebar-toggle-collapsed");
          if (exp && col) {
            exp.classList.remove("d-none");
            col.classList.add("d-none");
          }
        }
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
