/**
 * Local PR Review Agent (Ollama-powered)
 * ---------------------------------------
 * Fetches a PR's changed files/diff from GitHub, sends them to a local
 * Ollama model for review, and posts the result as a GitHub PR review
 * with inline, severity-tagged comments.
 *
 * Run locally (not in GitHub Actions) with:
 *   GITHUB_TOKEN=xxx OWNER=you REPO=your-repo PR_NUMBER=1 node review-pr.js
 *
 * Requires:
 *   - Ollama running locally (ollama serve) with a model pulled, e.g.
 *     ollama pull qwen2.5-coder:7b
 *   - A GitHub Personal Access Token with repo scope (for private repos)
 *     or public_repo scope (for public repos), set as GITHUB_TOKEN.
 */

require("dotenv").config();
const { Octokit } = require("@octokit/rest");
const fetch = require("node-fetch");

// ---------- Config ----------
const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
const OWNER = process.env.OWNER;
const REPO = process.env.REPO;
const PR_NUMBER = process.env.PR_NUMBER;
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434/api/chat";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "qwen2.5-coder:7b";

// Files we never want to send to the model (noise, generated, binary-ish)
const IGNORE_PATTERNS = [
  /package-lock\.json$/,
  /yarn\.lock$/,
  /pnpm-lock\.yaml$/,
  /\.min\.js$/,
  /\.svg$/,
  /\.png$|\.jpg$|\.jpeg$|\.gif$/,
  /dist\//,
  /build\//,
];

if (!GITHUB_TOKEN || !OWNER || !REPO || !PR_NUMBER) {
  console.error(
    "Missing required env vars. Need: GITHUB_TOKEN, OWNER, REPO, PR_NUMBER"
  );
  process.exit(1);
}

const octokit = new Octokit({ auth: GITHUB_TOKEN });

// ---------- Step 1: Fetch PR metadata + changed files ----------
async function fetchPRContext() {
  const { data: pr } = await octokit.pulls.get({
    owner: OWNER,
    repo: REPO,
    pull_number: PR_NUMBER,
  });

  const files = await octokit.paginate(octokit.pulls.listFiles, {
    owner: OWNER,
    repo: REPO,
    pull_number: PR_NUMBER,
    per_page: 100,
  });

  const relevantFiles = files.filter(
    (f) => !IGNORE_PATTERNS.some((re) => re.test(f.filename)) && f.patch
  );

  return { pr, files, relevantFiles };
}

// ---------- Step 2: Build the prompt ----------
function buildPrompt({ pr, files, relevantFiles }) {
  const fileSummaries = relevantFiles
    .map(
      (f) => `
### File: ${f.filename}
Status: ${f.status} | +${f.additions} / -${f.deletions}
\`\`\`diff
${f.patch}
\`\`\`
`
    )
    .join("\n");

  const skipped = files.length - relevantFiles.length;

  return `You are a senior software engineer performing a code review on a GitHub pull request.

PR Title: ${pr.title}
PR Description: ${pr.body || "(no description provided)"}
Total files changed: ${files.length}${skipped ? ` (${skipped} lockfile/binary/generated files skipped)` : ""}
Additions: +${pr.additions} / Deletions: -${pr.deletions}

Below are the diffs for each reviewable file.

${fileSummaries}

Review this PR and respond with ONLY valid JSON (no markdown fences, no prose outside the JSON) matching this exact schema:

{
  "summary": "2-4 sentence overview of what this PR does and overall assessment",
  "files_changed": [
    { "filename": "string", "change_description": "1-2 sentence description of what changed in this file" }
  ],
  "issues": [
    {
      "file": "string (must match a filename above)",
      "line_hint": "string, a short snippet of the exact line or nearby context from the diff so it can be located, or null if the issue is general/file-wide",
      "severity": "critical | major | minor | nit",
      "description": "clear explanation of the issue and why it matters",
      "suggestion": "concrete suggested fix, or null"
    }
  ],
  "overall_recommendation": "approve | request_changes | comment"
}

Severity guide:
- critical: bugs, security issues, data loss risk, breaking changes
- major: logic errors, missing error handling, significant design concerns
- minor: style inconsistencies, missing edge cases, minor inefficiencies
- nit: naming, formatting, trivial suggestions

If there are no issues, return an empty array for "issues". Output raw JSON only.`;
}

// ---------- Step 3: Call Ollama ----------
async function callOllama(prompt) {
  const res = await fetch(OLLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      messages: [
        {
          role: "system",
          content: "You are a precise, structured code review assistant. You always respond with raw JSON only, never markdown fences, never prose outside the JSON object.",
        },
        { role: "user", content: prompt },
      ],
      stream: false,
      format: "json",
      options: { temperature: 0.2 },
    }),
  });

  if (!res.ok) {
    throw new Error(`Ollama request failed: ${res.status} ${await res.text()}`);
  }

  const data = await res.json();
  return data.message.content;
}

// ---------- Step 4: Parse + validate model output ----------
function parseReview(raw) {
  let cleaned = raw.trim();
  // Safety net in case the model wraps output in ```json fences anyway
  cleaned = cleaned.replace(/^```json\s*/i, "").replace(/```$/, "").trim();

  let parsed;
  try {
    parsed = JSON.parse(cleaned);
  } catch (err) {
    throw new Error(
      `Failed to parse model output as JSON: ${err.message}\n---RAW OUTPUT---\n${raw}`
    );
  }

  if (!parsed.summary || !Array.isArray(parsed.issues) || !Array.isArray(parsed.files_changed)) {
    throw new Error(`Model output missing required fields.\n---RAW OUTPUT---\n${raw}`);
  }

  return parsed;
}

// ---------- Step 5: Map issues to file+line for inline comments ----------
// Ollama doesn't reliably give exact diff line numbers, so we locate the
// issue's line_hint inside the file's patch to find the right diff position.
function findLineInPatch(patch, lineHint) {
  if (!lineHint) return null;
  const lines = patch.split("\n");
  let diffLine = 0; // position within the diff hunk, which is what GitHub's API wants

  for (const line of lines) {
    if (line.startsWith("@@")) {
      diffLine = 0;
      continue;
    }
    if (line.startsWith("-")) continue; // removed lines don't count toward position
    diffLine++;
    const content = line.replace(/^[+ ]/, "");
    if (content.includes(lineHint.slice(0, 40))) {
      return diffLine;
    }
  }
  return null;
}

const SEVERITY_LABEL = {
  critical: "🔴 Critical",
  major: "🟠 Major",
  minor: "🟡 Minor",
  nit: "🔵 Nit",
};

function buildReviewPayload(review, relevantFiles) {
  const patchByFile = Object.fromEntries(
    relevantFiles.map((f) => [f.filename, f.patch])
  );

  const comments = [];
  const unplacedIssues = [];

  for (const issue of review.issues) {
    const patch = patchByFile[issue.file];
    const position = patch ? findLineInPatch(patch, issue.line_hint) : null;
    const label = SEVERITY_LABEL[issue.severity] || issue.severity;
    const body = `**${label}** — ${issue.description}${
      issue.suggestion ? `\n\n**Suggestion:** ${issue.suggestion}` : ""
    }`;

    if (patch && position) {
      comments.push({ path: issue.file, position, body });
    } else {
      // Fall back to listing it in the summary if we can't place it inline
      unplacedIssues.push({ ...issue, label });
    }
  }

  const fileList = review.files_changed
    .map((f) => `- \`${f.filename}\`: ${f.change_description}`)
    .join("\n");

  const unplacedList = unplacedIssues.length
    ? `\n\n### Additional issues (couldn't anchor to a specific line)\n` +
      unplacedIssues
        .map((i) => `- **${i.label}** \`${i.file}\`: ${i.description}`)
        .join("\n")
    : "";

  const bodySummary = `## 🤖 AI Review\n\n${review.summary}\n\n### Files changed\n${fileList}${unplacedList}`;

  const eventMap = {
    approve: "APPROVE",
    request_changes: "REQUEST_CHANGES",
    comment: "COMMENT",
  };

  return {
    body: bodySummary,
    event: eventMap[review.overall_recommendation] || "COMMENT",
    comments,
  };
}

// ---------- Step 6: Post the review ----------
async function postReview(payload) {
  await octokit.pulls.createReview({
    owner: OWNER,
    repo: REPO,
    pull_number: PR_NUMBER,
    body: payload.body,
    event: payload.event,
    comments: payload.comments,
  });
}

// ---------- Main ----------
async function main() {
  console.log(`Fetching PR #${PR_NUMBER} from ${OWNER}/${REPO}...`);
  const context = await fetchPRContext();
  console.log(
    `Found ${context.files.length} changed files (${context.relevantFiles.length} reviewable).`
  );

  const prompt = buildPrompt(context);

  console.log(`Sending diff to Ollama model "${OLLAMA_MODEL}"...`);
  const raw = await callOllama(prompt);

  console.log("Parsing model output...");
  const review = parseReview(raw);
  console.log(`Model returned ${review.issues.length} issue(s).`);

  const payload = buildReviewPayload(review, context.relevantFiles);

  console.log("Posting review to GitHub...");
  await postReview(payload);

  console.log("Done. Review posted.");
}

main().catch((err) => {
  console.error("Review agent failed:", err.message);
  process.exit(1);
});
