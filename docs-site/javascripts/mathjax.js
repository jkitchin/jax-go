// MathJax configuration. Markdown pages go through pymdownx.arithmatex, which
// emits \( \) and \[ \]; the notebook pages (mkdocs-jupyter) keep the raw $ and
// $$ delimiters, so both forms are recognized.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: "tex2jax_ignore",
    processHtmlClass: "tex2jax_process|arithmatex",
  },
};

// Material's instant navigation swaps the page without reloading; typeset again.
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.startup.output.clearCache();
      MathJax.typesetClear();
      MathJax.texReset();
      MathJax.typesetPromise();
    }
  });
}
