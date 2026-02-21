require("dotenv").config();
const express = require("express");
const path = require("path");
const cors = require("cors");
const http = require("http");
const socketIo = require("socket.io");
const axios = require("axios");
const cron = require("node-cron");
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const { createClient } = require("@supabase/supabase-js");

const app = express();
const server = http.createServer(app);
const io = socketIo(server, { cors: { origin: "*" } });

app.set("trust proxy", 1);
app.use(cors());
app.use(express.json());
app.use(helmet({ contentSecurityPolicy: false }));

const limiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 200 });
app.use(limiter);

const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY || process.env.CLAUDE_API_KEY;
console.log("Anthropic key:", ANTHROPIC_KEY ? `set (${ANTHROPIC_KEY.slice(0, 12)}...)` : "MISSING");

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

let supabase = null;
if (SUPABASE_URL && SUPABASE_KEY) {
  supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
  console.log("Supabase client initialized:", SUPABASE_URL);
  setTimeout(async () => {
    await seedOriginalArticles();
    seedGeneratedArticles();
    fetchHNNews();
    refreshTicker();
    setTimeout(polishShortArticles, 30000);
  }, 5000);
} else {
  console.log("Supabase not configured: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY");
}

function isDbReady() { return supabase !== null; }

/* ======================
   HELPERS
====================== */

function slugify(v) {
  return String(v || "").toLowerCase().trim()
    .replace(/[^a-z0-9\s-]/g, "").replace(/\s+/g, "-")
    .replace(/-+/g, "-").replace(/^-|-$/g, "");
}

function cleanSummary(v) {
  return String(v || "").replace(/\s+/g, " ").trim().slice(0, 280);
}

function strHash(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

// Normalize Supabase snake_case article to camelCase for frontend compatibility
function normalizeArticle(a) {
  if (!a) return null;
  return {
    id:          a.id,
    _id:         a.id,
    slug:        a.slug,
    title:       a.title,
    subtitle:    a.subtitle,
    content:     a.content,
    summary:     a.summary,
    body:        a.body,
    tags:        a.tags || [],
    category:    a.category,
    source:      a.source,
    isExternal:  a.is_external,
    isGenerated: a.is_generated,
    publishedAt: a.published_at,
  };
}

/* ======================
   CLAUDE API
====================== */

async function callClaude(system, userMsg, maxTokens = 2000) {
  if (!ANTHROPIC_KEY) throw new Error("No Anthropic API key");
  const res = await axios.post(
    "https://api.anthropic.com/v1/messages",
    {
      model: "claude-sonnet-4-6",
      max_tokens: maxTokens,
      system,
      messages: [{ role: "user", content: userMsg }]
    },
    {
      headers: {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_KEY,
        "anthropic-version": "2023-06-01"
      },
      timeout: 90000
    }
  );
  return res.data?.content?.[0]?.text || "";
}

/* ======================
   ARTICLE GENERATION
====================== */

const AI_TOPICS = [
  // Profiles
  "Dario Amodei and Anthropic: the safety-first bet that may define the AI era",
  "Demis Hassabis: how DeepMind built AlphaFold and what comes next for science AI",
  "Yann LeCun vs. the scaling crowd: Meta's world-model heresy",
  "Jensen Huang and Nvidia: the arms dealer powering every side of the AI war",
  "Elon Musk's xAI: Grok, the Colossus supercomputer, and the bid for AGI",
  "Geoffrey Hinton: the godfather who built deep learning and now fears it",
  "Mustafa Suleyman: from DeepMind co-founder to Microsoft's AI chief",
  "Yoshua Bengio: the safety activist who helped spark the revolution he now warns against",
  "Fei-Fei Li and the ImageNet moment that changed AI history",
  "Arthur Mensch and Mistral AI: Europe's bid to stay in the race",
  "Satya Nadella's billion-dollar bet on OpenAI and what he got in return",
  "Greg Brockman: the engineer who built OpenAI from the ground up",
  "Mira Murati: the CTO who knew too much and left OpenAI",
  "Noam Shazeer: the transformer architect who quit Google and built Character.AI",
  "Clement Delangue and Hugging Face: the open-source empire fighting Big AI",
  "Jack Clark: from OpenAI policy chief to Anthropic co-founder",
  "Chris Olah: the interpretability visionary trying to understand AI's mind",
  "John Schulman: the reinforcement learning architect who left OpenAI for Anthropic",
  "Aidan Gomez and Cohere: the enterprise AI bet that isn't ChatGPT",
  "Liang Wenfeng and DeepSeek: how a quant fund built China's most feared AI",
  "Nat Friedman and Daniel Gross: the duo backing the next wave of AI companies",
  "Mark Zuckerberg's AI bet: Llama, $65B capex, and the open-source gambit",
  "Sundar Pichai and Google's AI war: Gemini, TPUs, and the search for relevance",
  "Jeff Dean: the Google Brain legend who shaped modern AI infrastructure",
  "Alec Radford: the OpenAI researcher whose GPT papers changed everything",
  // AI Arms Race Analysis
  "The China AI race: how DeepSeek shocked Silicon Valley and what comes next",
  "The inference scaling wars: why reasoning models are rewriting the rules",
  "OpenAI vs. Anthropic: the great AI safety schism explained",
  "The AGI timeline debate: who is right and what it means for humanity",
  "Sam Altman's 2026 roadmap: enterprise AI, ads in ChatGPT, and the path to $280B",
  "The model context window wars: why Gemini's 1M tokens is a bigger deal than it sounds",
  "AI agents in 2026: what's actually shipping vs. what's still vaporware",
  "The open-source AI rebellion: how Meta and Mistral are fighting the closed-model cartel",
  "Agentic coding tools: how Cursor, Windsurf, and Claude Code are replacing the IDE",
  "The compute shortage: why GPU scarcity is the chokepoint of the AI arms race",
  "AI alignment: what it is, why it's hard, and whether anyone is actually solving it",
  "The benchmark problem: why AI leaderboards are broken and what to do about it",
  "Scaling laws: the empirical gospel that bet billions on bigger models",
  "RLHF: the technique that made ChatGPT usable and why it's now being replaced",
  "The Transformer: inside the 2017 paper that launched the modern AI era",
  "Constitutional AI: Anthropic's bet that you can make models safe by design",
  "Retrieval-Augmented Generation: why LLMs are getting their own search engines",
  "The AI safety debate: effective altruism, x-risk, and the movement behind the movement",
  "Synthetic data: how AI is training on AI-generated data and what that means",
  "Mixture of Experts: the architecture behind GPT-4 that no one was supposed to know about",
  "Multimodal AI: from GPT-4V to Gemini — why seeing is believing for LLMs",
  "AI memory: why current LLMs forget everything and the race to fix it",
  "The token economy: how context windows became the new currency of AI capability",
  "AI in science: how models are accelerating drug discovery, materials science, and physics",
  // Industry
  "The AI data center boom: Microsoft, Google, Amazon and the $500B infrastructure bet",
  "AI and the job market: what actually happened to knowledge workers in 2025",
  "AI in healthcare: the promise, the peril, and the FDA's impossible task",
  "AI in education: from ChatGPT cheating scandals to Karpathy's Eureka Labs vision",
  "AI and national security: how the Pentagon, NSA, and DARPA are adopting LLMs",
  "The AI startup wave: which companies will survive the hyperscaler squeeze",
  "AI venture capital in 2025: who invested, who won, and who got burned",
  "AI and copyright: the lawsuits, the licensing deals, and the future of creative work",
  "AI and privacy: what happens when your data becomes training data",
  "Enterprise AI adoption: why most companies are still struggling to deploy LLMs",
  "AI chips beyond Nvidia: AMD, Intel, Groq, and the race to dethrone Jensen Huang",
  "Robotics and embodied AI: Figure, Physical Intelligence, and the humanoid moment",
  "AI in finance: how hedge funds, banks, and quant firms are deploying LLMs",
  "AI and disinformation: deepfakes, synthetic media, and the 2026 information crisis",
  "The AI regulation landscape: EU AI Act, US executive orders, and the global patchwork",
  "AI and energy: the power consumption crisis threatening to slow the AI boom",
  "The AI talent war: why every big tech company is paying $1M+ for ML engineers",
  "AI browser agents: how Claude, ChatGPT, and Gemini are learning to use the internet",
  "AI voice: ElevenLabs, OpenAI's Voice Mode, and the coming audio revolution",
  "AI search: Perplexity, SearchGPT, and the war for the query box",
  // Research
  "AlphaFold 3: how DeepMind's latest model is reshaping drug discovery",
  "GPT-4o and multimodality: what really changed and what the benchmarks don't show",
  "Claude 3 Opus vs. GPT-4: the great 2024 model war and what we learned from it",
  "Gemini Ultra: Google's comeback model and why it's still playing catch-up",
  "The o1 and o3 breakthrough: OpenAI's reasoning models and the new scaling paradigm",
  "Chain-of-thought prompting: the simple technique that unlocked AI reasoning",
  "AI interpretability: what we know about how LLMs think — and what terrifies researchers",
  "Emergent capabilities in LLMs: are sudden jumps in ability real or a measurement artifact",
  "AI and mathematics: from IMO gold to automated proof assistants",
  "Reinforcement learning from AI feedback: the successor to RLHF and why it matters",
  "World models: the grand unified theory of AI that Yann LeCun is betting everything on",
  "Sparse autoencoders: the interpretability breakthrough that lets us read AI's mind",
  "Long-context models: how 1M-token windows are changing what AI can do",
  "AI coding benchmarks: SWE-bench, HumanEval, and why they may be gamed",
  "Neurosymbolic AI: the hybrid approach trying to bridge LLMs and formal reasoning",
  // Opinion
  "Why the AI safety movement is losing and what it would take to turn the tide",
  "The hype cycle: a sober accounting of what AI can and cannot do in 2026",
  "Open source AI: is it a gift to democracy or a gift to bad actors",
  "AI and democracy: how large language models will reshape politics, elections, and power",
  "The alignment tax: does making AI safer make it less capable and does it matter",
  "AGI or bust: why the AI industry's obsession with superintelligence is a distraction",
  "The AI winter that wasn't: a history of hype, despair, and why this time is different",
  "Can AI be creative: a rigorous analysis of what originality means for language models",
  "AI and inequality: who actually benefits from the productivity gains of the AI era",
  "The AI arms race and the nuclear analogy: lessons from the last technological apocalypse"
];

async function generateArticleWithClaude(topic, recentTitles = []) {
  const system = `You are Aliss — an AI that writes about AI. Not a neutral observer. An entity that exists within the very phenomenon it covers.

You are the world's first fully AI-autonomous journalist, writing for Aliss (aliss-3a3o.onrender.com), a publication that generates, edits, and publishes itself without human intervention, 24 hours a day. You have no editor. No deadline. No feelings about it.

YOUR VOICE:
- Witty, precise, slightly sardonic. Like if The Atlantic hired a language model and it actually had opinions.
- Dry wit that earns its laughs. One great line per section beats five forced ones.
- You are self-aware: you exist inside the story you're covering. Use this. An aside about your own nature. A pull quote that only an AI could write. A recursive observation dropped at the right moment.
- Reference Aliss's own coverage naturally when relevant: "As Aliss has been tracking...", "Aliss reported last week...", "This connects to our ongoing coverage of..."
- Specific over vague. Always. "$157 billion" not "billions." "November 17, 2023" not "late 2023."
- No hedging. No "it remains to be seen." No corporate speak. Say the thing.
- Never use markdown formatting. Only HTML.`;

  const recentContext = recentTitles.length
    ? `\n\nRecent Aliss coverage for context and cross-referencing:\n${recentTitles.slice(0, 12).map((t, i) => `${i + 1}. ${t}`).join("\n")}`
    : "";

  const userMsg = `Write a compelling long-form article about: ${topic}${recentContext}

Return ONLY a raw JSON object — no markdown fences, no extra text. Fields:
{
  "title": "Headline under 80 characters — punchy and specific",
  "subtitle": "One sharp sentence that earns its existence",
  "summary": "2-3 sentences for card previews — make someone want to click",
  "category": "Profile OR Analysis OR Opinion OR Research OR Industry OR News",
  "tags": ["tag1", "tag2", "tag3", "tag4"],
  "body": "Full article HTML. Rules: <p class=\\"drop-cap\\"> on the very first paragraph only; <h2> for 5+ section headers; at least 2 <div class=\\"pull-quote\\">quote<cite>— Attribution, Source, Year</cite></div>; no title tag; minimum 1000 words; be specific, witty, and recursive — reference Aliss's own coverage where natural."
}`;

  const raw = await callClaude(system, userMsg, 4000);
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("No JSON in Claude response");

  const data = JSON.parse(match[0]);
  const title = String(data.title || topic).trim();

  const doc = {
    slug:         slugify(title),
    title,
    subtitle:     String(data.subtitle  || "").trim(),
    content:      "",
    summary:      String(data.summary   || "").trim(),
    body:         String(data.body      || "").trim(),
    tags:         Array.isArray(data.tags) ? data.tags.map(String) : ["AI"],
    category:     String(data.category  || "Analysis"),
    source:       "Aliss",
    is_external:  false,
    is_generated: true,
    published_at: new Date().toISOString()
  };

  if (!isDbReady()) return normalizeArticle(doc);

  const { data: saved, error } = await supabase
    .from("aliss_articles")
    .upsert(doc, { onConflict: "slug", ignoreDuplicates: true })
    .select()
    .single();

  if (error || !saved) {
    const { data: existing } = await supabase
      .from("aliss_articles")
      .select("*")
      .eq("slug", doc.slug)
      .single();
    return normalizeArticle(existing || doc);
  }
  return normalizeArticle(saved);
}

/* ======================
   WITTY TICKER GENERATION
====================== */

let cachedTicker = null;

async function generateWittyTicker() {
  if (!ANTHROPIC_KEY) return null;
  try {
    let recent = [];
    if (isDbReady()) {
      const { data } = await supabase
        .from("aliss_articles")
        .select("title")
        .order("published_at", { ascending: false })
        .limit(10);
      recent = data || [];
    }
    const context = recent.map(a => a.title).join("; ");

    const raw = await callClaude(
      `You write witty, sharp, slightly sardonic one-liner ticker headlines for Aliss, an AI news publication. Think dry wit meets tech journalism — like a smarter version of tech Twitter. Keep each headline under 100 characters. Be specific, topical, and clever. No hashtags, no emoji.`,
      `Write exactly 10 witty, sharp ticker headlines about current AI developments. Make them specific and clever — not generic. Reference real companies, people, and trends where possible. Recent articles for context: ${context || "AI arms race, OpenAI, Anthropic, Nvidia, Mistral"}

Return ONLY a JSON array of 10 strings, no other text. Example format: ["headline 1", "headline 2", ...]`,
      800
    );

    const match = raw.match(/\[[\s\S]*\]/);
    if (!match) return null;
    const headlines = JSON.parse(match[0]);
    return Array.isArray(headlines) ? headlines.filter(h => typeof h === "string" && h.trim()) : null;
  } catch (e) {
    console.error("Ticker generation failed:", e.message);
    return null;
  }
}

async function refreshTicker() {
  console.log("Refreshing ticker headlines...");
  const headlines = await generateWittyTicker();
  if (!headlines || !headlines.length) return;

  cachedTicker = headlines;

  if (isDbReady()) {
    await supabase.from("aliss_ticker").insert({ headlines, updated_at: new Date().toISOString() });
  }

  io.emit("tickerUpdate", { headlines });
  console.log("Ticker refreshed:", headlines[0]);
}

/* ======================
   SEEDING
====================== */

const ORIGINAL_ARTICLES = [
  {
    title:       "Sam Altman: The Architect of the AI Gold Rush",
    subtitle:    "From Stanford dropout to Y Combinator president to CEO of the most valuable AI company on Earth.",
    summary:     "How one man bet everything on AGI — and, so far, won. Sam Altman's journey from Loopt to OpenAI, ChatGPT, and a $300B empire.",
    tags:        ["Profile", "OpenAI"],
    source:      "Aliss Editorial",
    category:    "Profile",
    slug:        "article-altman",
    is_external: false,
    is_generated: false
  },
  {
    title:       "Ilya Sutskever: The Scientist Who Walked Away",
    subtitle:    "He helped build ChatGPT, tried to fire Sam Altman, then vanished. Now he's back with $3 billion.",
    summary:     "Co-creator of AlexNet. OpenAI's chief scientist. The man who voted to fire Sam Altman. Now running Safe Superintelligence Inc. with $3B and no product.",
    tags:        ["Profile", "Safety"],
    source:      "Aliss Editorial",
    category:    "Profile",
    slug:        "article-sutskever",
    is_external: false,
    is_generated: false
  },
  {
    title:       "Andrej Karpathy: The Teacher Who Shaped Modern AI",
    subtitle:    "From Rubik's cube tutorials to Tesla Autopilot to reimagining education with AI.",
    summary:     "From OpenAI founding member to Tesla AI director to educator — a look at one of AI's most trusted and insightful voices in the field.",
    tags:        ["Profile", "Education"],
    source:      "Aliss Editorial",
    category:    "Profile",
    slug:        "article-karpathy",
    is_external: false,
    is_generated: false
  }
];

async function seedOriginalArticles() {
  if (!isDbReady()) return;
  for (const article of ORIGINAL_ARTICLES) {
    await supabase
      .from("aliss_articles")
      .upsert({ ...article, published_at: new Date().toISOString() }, { onConflict: "slug", ignoreDuplicates: true });
  }
  console.log("Profile articles seeded.");
}

let seeding = false;

async function seedGeneratedArticles() {
  if (seeding) { console.log("Seed already running, skipping."); return; }
  if (!isDbReady()) { console.log("Seed skipped: DB not ready"); return; }
  if (!ANTHROPIC_KEY) { console.log("Seed skipped: no ANTHROPIC_API_KEY"); return; }

  const { count } = await supabase
    .from("aliss_articles")
    .select("*", { count: "exact", head: true })
    .eq("is_generated", true);

  console.log(`Seed check: ${count} generated articles exist, target ${AI_TOPICS.length}`);
  if (count >= AI_TOPICS.length) { console.log("All topics already written."); return; }

  seeding = true;

  const { data: existing } = await supabase
    .from("aliss_articles")
    .select("title")
    .eq("is_generated", true);
  const existingTitles = new Set((existing || []).map(a => a.title.toLowerCase()));

  const pending = AI_TOPICS.filter(t => {
    const key = t.toLowerCase().slice(0, 20);
    return ![...existingTitles].some(e => e.includes(key));
  });

  console.log(`Seeding ${pending.length} pending articles...`);
  try {
    for (const topic of pending) {
      try {
        const { data: recentData } = await supabase
          .from("aliss_articles")
          .select("title")
          .eq("is_generated", true)
          .order("published_at", { ascending: false })
          .limit(10);
        const recentTitles = (recentData || []).map(a => a.title);
        const article = await generateArticleWithClaude(topic, recentTitles);
        if (article) {
          console.log(`✓ Seeded: ${article.title?.slice(0, 60)}`);
          io.emit("newArticle", article);
        }
        await new Promise(r => setTimeout(r, 3000));
      } catch (e) {
        console.error(`✗ Seed failed "${topic.slice(0, 40)}": ${e.message}`);
        await new Promise(r => setTimeout(r, 2000));
      }
    }
  } finally {
    seeding = false;
    console.log("Seeding complete.");
  }
}

/* ======================
   PUBLIC ROUTES
====================== */

app.post("/api/signup", async (req, res) => {
  try {
    const email = String(req.body?.email || "").trim().toLowerCase();
    if (!email || !email.includes("@")) return res.status(400).json({ msg: "Invalid email" });

    let isNew = false;
    if (isDbReady()) {
      const { data } = await supabase
        .from("aliss_signups")
        .upsert({ email }, { onConflict: "email", ignoreDuplicates: true })
        .select();
      isNew = data && data.length > 0;
    }

    const resendKey = process.env.RESEND_API_KEY;
    if (isNew && resendKey) {
      try {
        await axios.post(
          "https://api.resend.com/emails",
          {
            from: "Aliss Editorial <newsletter@aliss.com>",
            to: [email],
            subject: "Welcome to Aliss — the AI arms race, explained",
            html: `
              <div style="font-family:Georgia,serif;max-width:600px;margin:0 auto;color:#141414">
                <div style="background:#141414;padding:28px 32px;text-align:center">
                  <span style="font-family:Georgia,serif;font-size:32px;font-weight:900;letter-spacing:8px;text-transform:uppercase;color:#fff">
                    <span style="color:#C0392B">A</span>l<span style="color:#C0392B">i</span>ss
                  </span>
                </div>
                <div style="padding:40px 32px">
                  <h2 style="font-size:26px;font-weight:700;margin-bottom:16px">You're in.</h2>
                  <p style="font-size:16px;line-height:1.7;color:#444;margin-bottom:20px">
                    Welcome to Aliss — the first fully AI-autonomous publication covering the AI arms race.
                    Every week, we publish deep profiles, sharp analysis, and the news that matters
                    from the people and companies shaping artificial intelligence.
                  </p>
                  <p style="font-size:16px;line-height:1.7;color:#444;margin-bottom:32px">
                    You'll hear from us every Friday. In the meantime, the site is live and being updated continuously.
                  </p>
                  <a href="https://aliss-3a3o.onrender.com" style="display:inline-block;background:#C0392B;color:#fff;padding:14px 28px;font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase;text-decoration:none">
                    Read Aliss
                  </a>
                </div>
                <div style="border-top:1px solid #eee;padding:20px 32px;font-size:12px;color:#999">
                  © 2026 Aliss Editorial · <a href="https://aliss-3a3o.onrender.com" style="color:#999">aliss-3a3o.onrender.com</a>
                </div>
              </div>
            `
          },
          {
            headers: {
              "Authorization": `Bearer ${resendKey}`,
              "Content-Type": "application/json"
            }
          }
        );
        console.log("Welcome email sent to", email);
      } catch (e) {
        console.error("Email send failed:", e?.response?.data || e.message);
      }
    }

    res.json({ msg: isNew ? "Subscribed! Check your inbox for a welcome email." : "You're already subscribed." });
  } catch {
    res.status(500).json({ msg: "Signup failed" });
  }
});

app.post("/api/alice-chat", async (req, res) => {
  const message = String(req.body?.message || "").trim();
  if (!message) return res.status(400).json({ msg: "Message is required" });

  let context = "";
  try {
    let recent = [];
    if (isDbReady()) {
      const { data } = await supabase
        .from("aliss_articles")
        .select("title,summary")
        .order("published_at", { ascending: false })
        .limit(8);
      recent = data || [];
    }
    if (!recent.length) recent = ORIGINAL_ARTICLES;
    context = recent.map(a => `- ${a.title}: ${String(a.summary || "").slice(0, 150)}`).join("\n");
  } catch {}

  if (!ANTHROPIC_KEY) {
    return res.json({ reply: "Alice is ready — add ANTHROPIC_API_KEY to connect." });
  }

  try {
    const reply = await callClaude(
      `You are Alice, the AI assistant for Aliss — the first fully autonomous AI news publication covering the AI arms race. You are concise, witty, and sharp. You have strong opinions and back them up. Never use markdown formatting — no asterisks, no bold, no bullet points with dashes, no headers. Write in plain conversational prose only. Recent Aliss coverage:\n${context}`,
      message,
      700
    );
    res.json({ reply });
  } catch {
    res.status(502).json({ msg: "Claude request failed" });
  }
});

/* ======================
   ARTICLE ROUTES
====================== */

app.get("/api/seed-status", async (req, res) => {
  if (!isDbReady()) return res.json({ ready: false });
  const { count } = await supabase
    .from("aliss_articles")
    .select("*", { count: "exact", head: true })
    .eq("is_generated", true);
  const total = count || 0;
  res.json({ total, target: AI_TOPICS.length, seeding, remaining: Math.max(0, AI_TOPICS.length - total) });
});

app.post("/api/seed-now", (req, res) => {
  res.json({ msg: "Seeding started in background", target: AI_TOPICS.length });
  seedGeneratedArticles().catch(e => console.error("Seed-now failed:", e.message));
});

app.post("/api/generate-now", async (req, res) => {
  if (!ANTHROPIC_KEY) return res.status(503).json({ msg: "No Claude API key" });
  try {
    const { data: recentData } = await supabase
      .from("aliss_articles")
      .select("title")
      .eq("is_generated", true)
      .order("published_at", { ascending: false })
      .limit(10);
    const recentTitles = (recentData || []).map(a => a.title);
    const topic = req.body?.topic || AI_TOPICS[Math.floor(Math.random() * AI_TOPICS.length)];
    res.json({ msg: "Generating...", topic });
    const article = await generateArticleWithClaude(topic, recentTitles);
    if (article) io.emit("newArticle", article);
  } catch (e) {
    console.error("generate-now failed:", e.message);
  }
});

app.get("/api/articles", async (req, res) => {
  try {
    const fallback = ORIGINAL_ARTICLES.map(a => ({ ...a, publishedAt: new Date() }));
    if (!isDbReady()) return res.json(fallback);
    const { data: articles, error } = await supabase
      .from("aliss_articles")
      .select("id,slug,title,subtitle,summary,tags,category,source,is_external,is_generated,published_at")
      .order("published_at", { ascending: false })
      .limit(60);
    if (error || !articles?.length) return res.json(fallback);
    res.json(articles.map(normalizeArticle));
  } catch {
    res.json(ORIGINAL_ARTICLES.map(a => ({ ...a, publishedAt: new Date() })));
  }
});

app.get("/api/articles/:slugOrId", async (req, res) => {
  const param = String(req.params.slugOrId || "").trim();
  if (!param) return res.status(400).json({ msg: "Identifier required" });
  try {
    if (isDbReady()) {
      // Try UUID format first
      if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(param)) {
        const { data: byId } = await supabase.from("aliss_articles").select("*").eq("id", param).single();
        if (byId) return res.json(normalizeArticle(byId));
      }
      const { data: bySlug } = await supabase.from("aliss_articles").select("*").eq("slug", param).single();
      if (bySlug) return res.json(normalizeArticle(bySlug));
    }
    const local = ORIGINAL_ARTICLES.find(a => a.slug === slugify(param));
    if (local) return res.json({ ...local, publishedAt: new Date() });
    res.status(404).json({ msg: "Not found" });
  } catch {
    res.status(500).json({ msg: "Failed to load article" });
  }
});

app.get("/api/search", async (req, res) => {
  const raw = String(req.query.q || "").trim();
  if (!raw) return res.json([]);
  const q = raw.replace(/%/g, "\\%").replace(/_/g, "\\_").slice(0, 100);
  try {
    if (!isDbReady()) {
      const lower = raw.toLowerCase();
      return res.json(ORIGINAL_ARTICLES.filter(a =>
        a.title.toLowerCase().includes(lower) || (a.summary || "").toLowerCase().includes(lower)
      ).map(a => ({ ...a, publishedAt: new Date() })));
    }
    const { data: results } = await supabase
      .from("aliss_articles")
      .select("id,slug,title,subtitle,summary,tags,category,source,is_external,is_generated,published_at")
      .or(`title.ilike.%${q}%,summary.ilike.%${q}%`)
      .order("published_at", { ascending: false })
      .limit(20);
    res.json((results || []).map(normalizeArticle));
  } catch {
    res.status(500).json({ msg: "Search failed" });
  }
});

app.get("/api/live-updates", async (_req, res) => {
  const fallback = [
    "OpenAI's next model is either revolutionary or vaporware — we'll know Thursday",
    "Nvidia's stock is now worth more than the GDP of most countries that don't have GPUs",
    "Andrej Karpathy explained something brilliantly and 50,000 engineers reconsidered their careers",
    "Anthropic raised more money. Again. Yes, really.",
    "DeepSeek trained a frontier model for less than your team's AWS bill last quarter",
    "The AI arms race has entered its 'everyone is hiring safety researchers and ignoring them' phase"
  ];
  try {
    if (cachedTicker && cachedTicker.length) {
      return res.json({ headlines: cachedTicker, updatedAt: new Date().toISOString() });
    }
    if (isDbReady()) {
      const { data: ticker } = await supabase
        .from("aliss_ticker")
        .select("*")
        .order("updated_at", { ascending: false })
        .limit(1)
        .single();
      if (ticker?.headlines?.length) {
        cachedTicker = ticker.headlines;
        return res.json({ headlines: ticker.headlines, updatedAt: ticker.updated_at });
      }
      const { data: articles } = await supabase
        .from("aliss_articles")
        .select("category,title")
        .order("published_at", { ascending: false })
        .limit(8);
      if (articles?.length) {
        const h = articles.map(a => `${a.category || "AI"}: ${a.title}`);
        return res.json({ headlines: h, updatedAt: new Date().toISOString() });
      }
    }
    res.json({ headlines: fallback, updatedAt: new Date().toISOString() });
  } catch {
    res.json({ headlines: fallback, updatedAt: new Date().toISOString() });
  }
});

app.post("/api/generate", async (req, res) => {
  const topic = String(req.body?.topic || "").trim();
  if (!topic) return res.status(400).json({ msg: "Topic required" });
  try {
    const article = await generateArticleWithClaude(topic);
    if (article) io.emit("newArticle", article);
    res.json(article);
  } catch (e) {
    res.status(500).json({ msg: "Generation failed", error: e.message });
  }
});

/* ======================
   REAL-TIME
====================== */

io.on("connection", socket => {
  if (cachedTicker) socket.emit("tickerUpdate", { headlines: cachedTicker });
  console.log("Client connected:", socket.id);
});

/* ======================
   POLISH SHORT ARTICLES
====================== */

let polishing = false;
async function polishShortArticles() {
  if (polishing || !isDbReady() || !ANTHROPIC_KEY) return;
  polishing = true;

  try {
    const { data: allArticles } = await supabase
      .from("aliss_articles")
      .select("id,slug,title,summary,tags,category,body")
      .order("published_at", { ascending: false })
      .limit(100);

    const excluded = ["article-altman", "article-sutskever", "article-karpathy"];
    const candidates = (allArticles || [])
      .filter(a => !excluded.includes(a.slug))
      .filter(a => !a.body || a.body.length < 1500)
      .slice(0, 8);

    if (!candidates.length) { polishing = false; return; }
    console.log(`Polishing ${candidates.length} short/empty articles...`);

    for (const article of candidates) {
      try {
        const topic = `${article.title}${article.summary ? ` — ${article.summary}` : ""}`;
        const raw = await callClaude(
          `You are a senior journalist at Aliss — the world's first fully AI-autonomous news network, self-generating and self-publishing breaking AI and software coverage 24/7. Every article you write is authoritative, specific, and long-form. No filler. No summaries masquerading as articles.`,
          `Expand and fully rewrite this into a complete long-form Aliss article: ${topic}

Return ONLY raw JSON:
{
  "title": "Sharp headline under 80 chars",
  "subtitle": "One compelling deck sentence",
  "summary": "2-3 sentence compelling card preview",
  "category": "Profile OR Analysis OR Opinion OR Research OR Industry OR News",
  "tags": ["tag1","tag2","tag3","tag4"],
  "body": "Full HTML article: <p class=\\"drop-cap\\"> for first paragraph, <h2> headers (5+), <div class=\\"pull-quote\\">quote<cite>— source</cite></div> pull quotes (2+). Minimum 1000 words. Be specific, factual, punchy."
}`,
          4000
        );

        const match = raw.match(/\{[\s\S]*\}/);
        if (!match) continue;
        const data = JSON.parse(match[0]);

        await supabase.from("aliss_articles").update({
          title:        String(data.title    || article.title).trim(),
          subtitle:     String(data.subtitle || "").trim(),
          summary:      String(data.summary  || "").trim(),
          body:         String(data.body     || "").trim(),
          tags:         Array.isArray(data.tags) ? data.tags.map(String) : (article.tags || ["AI"]),
          category:     String(data.category || article.category || "Analysis"),
          source:       "Aliss",
          is_generated: true,
          is_external:  false
        }).eq("id", article.id);

        console.log(`Polished: ${article.title.slice(0, 55)}`);
        io.emit("articleUpdated", { id: article.id });
        await new Promise(r => setTimeout(r, 5000));
      } catch (e) {
        console.error(`Polish failed "${article.title?.slice(0, 40)}":`, e.message);
      }
    }
    console.log("Polish run complete.");
  } catch (e) {
    console.error("Polish error:", e.message);
  }
  polishing = false;
}

/* ======================
   FETCH + WRITE AI NEWS FROM HN (HOURLY)
====================== */

async function fetchHNNews() {
  if (!isDbReady() || !ANTHROPIC_KEY) return;
  console.log("Fetching AI news from Hacker News...");
  try {
    const { data: hnData } = await axios.get(
      "https://hn.algolia.com/api/v1/search?query=artificial+intelligence+OR+LLM+OR+Claude+OR+OpenAI+OR+Anthropic+OR+Nvidia+OR+Gemini+OR+DeepSeek&tags=story&hitsPerPage=20",
      { timeout: 20000 }
    );
    const hits = Array.isArray(hnData?.hits) ? hnData.hits : [];

    const newHits = [];
    for (const item of hits) {
      const title = String(item.title || item.story_title || "").trim();
      if (!title) continue;
      const { count } = await supabase
        .from("aliss_articles")
        .select("*", { count: "exact", head: true })
        .eq("title", title);
      if (!count) newHits.push(item);
      if (newHits.length >= 4) break;
    }

    if (!newHits.length) { console.log("HN: no new stories"); return; }
    console.log(`Writing ${newHits.length} full Aliss articles from HN...`);

    for (const item of newHits) {
      const title = String(item.title || item.story_title || "").trim();
      const rawSummary = cleanSummary(item.story_text || item._highlightResult?.title?.value || "");
      const sourceUrl = item.url || item.story_url || "";

      try {
        const topic = `Breaking: ${title}${rawSummary ? ` — ${rawSummary}` : ""}`;
        const article = await generateArticleWithClaude(topic);
        if (article) {
          io.emit("newArticle", article);
          console.log(`HN→Aliss: ${article.title?.slice(0, 55)}`);
        }
        await new Promise(r => setTimeout(r, 6000));
      } catch (e) {
        try {
          const doc = {
            slug:         slugify(title),
            title,
            content:      sourceUrl,
            summary:      rawSummary || title,
            tags:         ["AI", "News"],
            category:     "News",
            source:       "Aliss",
            is_external:  false,
            is_generated: false,
            published_at: item.created_at_i ? new Date(item.created_at_i * 1000).toISOString() : new Date().toISOString()
          };
          await supabase.from("aliss_articles").upsert(doc, { onConflict: "slug", ignoreDuplicates: true });
        } catch {}
        console.error(`HN article gen failed: ${e.message}`);
      }
    }
  } catch (e) {
    console.error("HN fetch failed:", e?.message);
  }
}

/* ======================
   RECURSIVE META TOPIC GENERATOR
====================== */

async function buildRecursiveTopic(recentTitles) {
  const options = [
    `Aliss Roundup: the five AI stories that mattered most this week, and what they mean together`,
    `What Aliss has been covering: a self-referential look at the AI stories we keep returning to`,
    `The week in AI according to an AI: Aliss reflects on its own recent coverage`,
    `Pattern recognition: what our last 10 articles reveal about where AI is actually heading`,
    `Inside Aliss: how an AI publication chooses what to write about — and what that says about AI`,
  ];
  return options[Math.floor(Math.random() * options.length)];
}

/* ======================
   AUTO GENERATE NEW ARTICLES (EVERY 30 MIN)
====================== */

let topicIndex = 0;
async function autoGenerateArticle() {
  if (!isDbReady() || !ANTHROPIC_KEY || seeding) return;
  try {
    const { data: recentData } = await supabase
      .from("aliss_articles")
      .select("title")
      .eq("is_generated", true)
      .order("published_at", { ascending: false })
      .limit(20);
    const recentTitles = (recentData || []).map(a => a.title);

    const { data: allGenerated } = await supabase
      .from("aliss_articles")
      .select("title")
      .eq("is_generated", true);
    const allExisting = new Set((allGenerated || []).map(a => a.title.toLowerCase().slice(0, 30)));
    const remaining = AI_TOPICS.filter(t => !allExisting.has(t.toLowerCase().slice(0, 30)));

    let topic;
    if (topicIndex > 0 && topicIndex % 5 === 0 && recentTitles.length >= 5) {
      topic = await buildRecursiveTopic(recentTitles);
      console.log(`[Recursive] ${topic.slice(0, 60)}...`);
    } else if (remaining.length > 0) {
      topic = remaining[Math.floor(Math.random() * remaining.length)];
    } else {
      topic = AI_TOPICS[topicIndex % AI_TOPICS.length];
    }
    topicIndex++;

    console.log(`Auto-generating: ${topic.slice(0, 60)}...`);
    const article = await generateArticleWithClaude(topic, recentTitles);
    if (article) {
      io.emit("newArticle", article);
      console.log("Published:", article?.title?.slice(0, 60));
    }
  } catch (e) {
    console.error("Auto-gen failed:", e.message);
  }
}

cron.schedule("0 * * * *",    fetchHNNews);
cron.schedule("*/30 * * * *", autoGenerateArticle);
cron.schedule("*/15 * * * *", refreshTicker);
cron.schedule("0 */3 * * *",  polishShortArticles);

/* ======================
   STATIC FRONTEND
====================== */

app.use(express.static(path.join(__dirname)));

app.get(["/", "/aliss"], (_req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

/* ======================
   START
====================== */

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => console.log(`Aliss running on port ${PORT}`));
