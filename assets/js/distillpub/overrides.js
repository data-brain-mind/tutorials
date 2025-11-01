$(document).ready(function() {
    // Helper to safely override styles once distill custom elements are upgraded.
    function applyOverrides() {
        // Override styles of the footnotes.
        document.querySelectorAll("d-footnote").forEach(function(footnote) {
            if (!footnote.shadowRoot) return; // not yet upgraded
            try {
                const supSpan = footnote.shadowRoot.querySelector("sup > span");
                if (supSpan) {
                    supSpan.setAttribute("style", "color: var(--global-theme-color);");
                }
                const hoverStyle = footnote.shadowRoot.querySelector("d-hover-box")?.shadowRoot?.querySelector("style");
                if (hoverStyle?.sheet) {
                    hoverStyle.sheet.insertRule(".panel {background-color: var(--global-bg-color) !important;}");
                    hoverStyle.sheet.insertRule(".panel {border-color: var(--global-divider-color) !important;}");
                }
            } catch (e) {
                // swallow errors to avoid breaking citation rendering
            }
        });

        // Override styles of the citations.
        document.querySelectorAll("d-cite").forEach(function(cite) {
            if (!cite.shadowRoot) return; // not yet upgraded
            try {
                const span = cite.shadowRoot.querySelector("div > span");
                if (span) {
                    span.setAttribute("style", "color: var(--global-theme-color);");
                }
                const styleEl = cite.shadowRoot.querySelector("style");
                if (styleEl?.sheet) {
                    styleEl.sheet.insertRule("ul li a {color: var(--global-text-color) !important; text-decoration: none;}");
                    styleEl.sheet.insertRule("ul li a:hover {color: var(--global-theme-color) !important;}");
                }
                const hoverStyle = cite.shadowRoot.querySelector("d-hover-box")?.shadowRoot?.querySelector("style");
                if (hoverStyle?.sheet) {
                    hoverStyle.sheet.insertRule(".panel {background-color: var(--global-bg-color) !important;}");
                    hoverStyle.sheet.insertRule(".panel {border-color: var(--global-divider-color) !important;}");
                }
            } catch (e) {
                // swallow errors to avoid breaking citation rendering
            }
        });
    }

    // Try after DOM ready, and again once custom elements are defined.
    applyOverrides();
    if (window.customElements && customElements.whenDefined) {
        Promise.allSettled([
            customElements.whenDefined('d-cite'),
            customElements.whenDefined('d-footnote')
        ]).then(function() {
            // Run on next tick to give transforms time to attach shadow DOM.
            setTimeout(applyOverrides, 0);
        });
    }

    // Also observe for dynamically added citations/footnotes.
    const mo = new MutationObserver(function(mutations) {
        for (const m of mutations) {
            if (m.addedNodes && m.addedNodes.length) {
                applyOverrides();
                break;
            }
        }
    });
    mo.observe(document.body, { childList: true, subtree: true });
});