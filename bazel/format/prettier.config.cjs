/**
 * @see https://prettier.io/docs/en/configuration.html
 */
const config = {
  tabWidth: 2,
  printWidth: 80,
  plugins: [
    require("../../deps/npm/node_modules/prettier-plugin-sql"),
    require("../../deps/npm/node_modules/@prettier/plugin-xml"),
    require("../../deps/npm/node_modules/prettier-plugin-gherkin"),
  ],
};

module.exports = config;
