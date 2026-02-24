window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  startup: {
    // Don't auto-typeset on load; we control it via document$
    typeset: false,
  },
};

// Re-typeset MathJax after MkDocs Material instant navigation.
// On first load MathJax may not be ready yet, so wait for its promise.
document$.subscribe(function () {
  var typeset = function () {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  };

  if (typeof MathJax !== "undefined" && MathJax.startup && MathJax.startup.promise) {
    MathJax.startup.promise.then(typeset);
  }
});
