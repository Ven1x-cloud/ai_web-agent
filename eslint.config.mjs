import globals from "globals";

export default [
  {
    files: ["extension/**/*.js"],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: "module",
      globals: { ...globals.browser, ...globals.webextensions, marked: "readonly", DOMPurify: "readonly", katex: "readonly" },
    },
    rules: {
      "no-undef": "error",
      "no-unused-vars": ["warn", { args: "none", caughtErrors: "none", varsIgnorePattern: "^_" }],
      "no-unreachable": "error",
      "no-dupe-keys": "error",
      "no-constant-condition": ["error", { checkLoops: false }],
    },
  },
  {
    files: ["extension/content/content.js"],
    languageOptions: { sourceType: "script" },
  },
  {
    files: ["scripts/**/*.mjs", "test/**/*.mjs"],
    languageOptions: { ecmaVersion: 2023, sourceType: "module", globals: { ...globals.node } },
    rules: { "no-undef": "error", "no-unused-vars": "warn" },
  },
  { ignores: ["extension/vendor/**", "node_modules/**"] },
];
