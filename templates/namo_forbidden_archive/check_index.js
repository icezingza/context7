#!/usr/bin/env node
/*
  check_index.js
  Usage: node check_index.js [projectPath]
  Example: node check_index.js icezingza/namo_forbidden_archive

  This script fetches the rendered document (llms.txt) from Context7
  and reports status + presence of sensitive keywords. It does NOT
  modify any repository content.
*/

const project = process.argv[2] || "icezingza/namo_forbidden_archive";
const url = `https://context7.com/${project}/llms.txt?tokens=10000`;

(async () => {
  try {
    const res = await fetch(url, { headers: { "User-Agent": "context7-checker/1.0" } });
    console.log("URL:", url);
    console.log("Status:", res.status);
    if (res.status === 200) {
      const text = await res.text();
      console.log("Content length:", text.length);
      const keywords = ["NaMo Forbidden Archive", "Incest", "Horny", "Sister", "llms.txt"];
      const found = keywords.filter((k) => text.indexOf(k) !== -1);
      console.log("Found keywords:", found.length ? found.join(", ") : "None");
      if (found.length) {
        console.warn(
          "Warning: content contains sensitive keywords. Consider excluding the file via context7.json if you don't want it indexed."
        );
      }
    } else {
      console.log(
        "Not available (status != 200). If you just updated config, trigger a re-scan in the Context7 Dashboard and retry."
      );
    }
  } catch (err) {
    console.error("Fetch error:", err?.message || err);
  }
})();
