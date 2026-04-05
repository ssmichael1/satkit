window.MathJax = {
  tex: {
    // Accept both the pymdownx.arithmatex delimiters \(...\) / \[...\] (used
    // in plain Markdown files via the arithmatex extension) AND the
    // Jupyter / standard LaTeX $...$ and $$...$$ delimiters (used in the
    // notebook cells under docs/tutorials/). Both are needed because
    // mkdocs-jupyter passes raw notebook markdown through to MathJax
    // without going through pymdownx.arithmatex.
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  // Process the entire page. The previous `ignoreHtmlClass: ".*|"` +
  // `processHtmlClass: "arithmatex"` restriction caused MathJax to skip
  // mkdocs-jupyter notebook HTML entirely, since mkdocs-jupyter does not
  // wrap notebook-cell math in a `.arithmatex` span. pymdownx.arithmatex
  // (generic mode) still emits raw delimiters inside `span.arithmatex`,
  // which MathJax picks up with default scanning.
};

// Re-typeset MathJax after MkDocs Material instant navigation
document$.subscribe(function () {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
