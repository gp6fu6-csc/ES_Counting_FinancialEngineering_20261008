# Vendored front-end libraries

Served offline from `/ui/vendor/` (no CDN, no Node build). Downloaded once (2026-10-12) from the npm
registry tarballs; each tarball matched the registry `dist.integrity` (sha512) before extraction.

| File | Package | Version | Licence | Source tarball | SHA-256 of the file |
|---|---|---|---|---|---|
| `vue.global.prod.js` | vue (`dist/vue.global.prod.js`) | 3.5.43 | MIT (`LICENSE-vue`) | https://registry.npmjs.org/vue/-/vue-3.5.43.tgz | `b72394052eef1eeda1752db3758ce1be6f88016903f22e5241cc732ea307478e` |
| `echarts.min.js` | echarts (`dist/echarts.min.js`) | 6.1.0 | Apache-2.0 (`LICENSE-echarts`, `NOTICE-echarts`) | https://registry.npmjs.org/echarts/-/echarts-6.1.0.tgz | `b66b25aeb4df84e33199dc21694014d336d222cbd9deb0e5a7c14bd6aa0d0fd0` |

Bundled inside `echarts.min.js` (licences kept next to it):

- zrender 6.1.0, BSD-3-Clause (`LICENSE-zrender`, from https://registry.npmjs.org/zrender/-/zrender-6.1.0.tgz)
- parts of d3.js, BSD-3-Clause (`LICENSE-echarts-d3` = `licenses/LICENSE-d3` of the echarts package)
- tslib 2.3.0 helpers, 0BSD

Tarball integrity (npm `dist.integrity`):

- vue-3.5.43.tgz `sha512-o5qZoksdnjIKvW1srZ3ab7pcDNYAerBjRe54D0LBLfRdCYFrSgBHVXokMas35czQc0//lmx4/tuY4ZNQ+Rf2Ng==`
- echarts-6.1.0.tgz `sha512-q0yaFPggC9FUdsWH4blavRWFmxdrIodbkoKNAjJudAI6CA9gNPxHtV2RcZNEepZVlk4yvBYkOkbk6HIVpIyHZA==`
- zrender-6.1.0.tgz `sha512-oEGMDB6pOP2S6OwRR4PdVv610zrjnA3Bh+JnSG12fYJlBKjtNAoEb5fSUoCOOINlH96I2fU38/A2UpRKs67xYQ==`

Not used (forbidden by the project licence policy): lightweight-charts, Highcharts, PyQt, any CDN.
