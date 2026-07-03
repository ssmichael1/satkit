// Render KaTeX math after each (instant-navigation) page load.
//
// Handles both the pymdownx.arithmatex (generic) delimiters \( … \) / \[ … \]
// used in plain Markdown files AND the Jupyter / standard LaTeX $…$ / $$…$$
// delimiters used in notebook cells under docs/tutorials/ (mkdocs-jupyter
// passes raw notebook markdown through without arithmatex processing).
// auto-render skips <pre>/<code> blocks by default, so stray dollar signs in
// code cells are not misinterpreted.
document$.subscribe(() => {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
  });
});
