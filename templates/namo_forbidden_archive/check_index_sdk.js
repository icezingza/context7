/*
  check_index_sdk.js
  Usage: Set CONTEXT7_API_KEY in env, then:
    node check_index_sdk.js

  This example uses the @upstash/context7-sdk package to search for the
  library and fetch docs. Install the package if running outside the monorepo:
    npm install @upstash/context7-sdk

  Note: running this requires a valid Context7 API key with access to the
  project's data.
*/

import { Context7 } from "@upstash/context7-sdk";

async function main() {
  const apiKey = process.env.CONTEXT7_API_KEY;
  if (!apiKey) {
    console.error("Please set CONTEXT7_API_KEY in your environment.");
    process.exit(1);
  }

  const client = new Context7({ apiKey });
  const query = "NaMo Forbidden Archive";
  try {
    const search = await client.searchLibrary(query);
    console.log("searchLibrary result:", JSON.stringify(search, null, 2));
    const first = search?.results?.[0];
    if (first?.id) {
      console.log("Found library id:", first.id);
      const docs = await client.getDocs(first.id, { mode: "code", format: "txt", limit: 1 });
      console.log("getDocs sample:", JSON.stringify(docs?.items?.[0] || docs, null, 2));
    } else {
      console.log("Library not found via SDK search. It may not be indexed yet.");
    }
  } catch (err) {
    console.error("SDK error:", err?.message || err);
  }
}

main();
