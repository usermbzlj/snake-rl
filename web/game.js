/* Compatibility loader for file:// and older HTML that still points at game.js. */
(function loadSnakeGameScripts() {
  var current = document.currentScript;
  var base = current && current.src ? current.src.replace(/[^/]+$/, "") : "./";
  var files = [
    "game/constants.js",
    "game/features.js",
    "game/snake_game.js",
    "game/remote_inference.js",
  ];
  for (var i = 0; i < files.length; i += 1) {
    document.write('<script src="' + base + files[i] + '"><\/script>');
  }
})();
