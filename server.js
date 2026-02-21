require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const path = require("path");
const cors = require("cors");
const jwt = require("jsonwebtoken");
const bcrypt = require("bcryptjs");
const http = require("http");
const socketIo = require("socket.io");
const axios = require("axios");
const cron = require("node-cron");
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");

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

const mongoUri = process.env.MONGO_URI;
if (mongoUri && (mongoUri.startsWith("mongodb://") || mongoUri.startsWith("mongodb+srv://"))) {
  mongoose.connect(mongoUri)
    .then(() => {
      console.log("MongoDB Connected");
      setTimeout(async () => {
        await migrateSourceToAliss();
        await seedOriginalArticles();
        await seedGeneratedArticles();
        fetchHNNews();
        refreshTicker();
        // Polish short articles after seeding is done
        setTimeout(polishShortArticles, 30000);
      }, 5000);
    })
    .catch(err => console.error("MongoDB error:", err));
} else {
  console.log("MongoDB not connected: set MONGO_URI in .env");
}

/* ======================
   MODELS
====================== */

const ArticleSchema = new mongoose.Schema({
  slug:        { type: String, unique: true, sparse: true },
  title:       { type: String, required: true },
  subtitle:    String,
  content:     String,
  summary:     String,
  body:        String,
  tags:        [String],
  category:    { type: String, default: "News" },
  source:      String,
  imageUrl:    String,
  isExternal:  { type: Boolean, default: false },
  isGenerated: { type: Boolean, default: false },
  publishedAt: { type: Date, default: Date.now }
});
ArticleSchema.index({ title: 1, source: 1 }, { unique: true });
ArticleSchema.index({ slug: 1 }, { unique: true, sparse: true });
ArticleSchema.index({ title: "text", summary: "text", tags: "text" });

const UserSchema = new mongoose.Schema({
  email:    { type: String, unique: true },
  password: String,
  role:     { type: String, default: "admin" }
});

const SignupSchema = new mongoose.Schema({
  email:     { type: String, required: true, unique: true },
  createdAt: { type: Date, default: Date.now }
});

const TickerSchema = new mongoose.Schema({
  headlines: [String],
  updatedAt: { type: Date, default: Date.now }
});

const Article = mongoose.model("Article", ArticleSchema);
const User    = mongoose.model("User", UserSchema);
const Signup  = mongoose.model("Signup", SignupSchema);
const Ticker  = mongoose.model("Ticker", TickerSchema);

/* ======================
   HELPERS
====================== */

function isMongoReady() {
  return mongoose.connection.readyState === 1;
}

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

// AI-generated images via Pollinations (free, no key required)
function getImageUrl(prompt, seed) {
  const p = String(prompt || "artificial intelligence technology futuristic dark").slice(0, 200);
  const s = seed !== undefined ? seed : strHash(p);
  return `https://image.pollinations.ai/prompt/${encodeURIComponent(p)}?width=800&height=450&nologo=true&seed=${s}`;
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
  "Dario Amodei and Anthropic: the safety-first bet that may define the AI era",
  "Demis Hassabis: how DeepMind built AlphaFold — and what comes next for science AI",
  "Yann LeCun vs. the scaling crowd: Meta's world-model heresy",
  "Jensen Huang and Nvidia: the arms dealer powering every side of the AI war",
  "Elon Musk's xAI: Grok, the Colossus supercomputer, and the bid for AGI",
  "Geoffrey Hinton: the godfather who built deep learning — and now fears it",
  "The China AI race: how DeepSeek shocked Silicon Valley",
  "Mustafa Suleyman: from DeepMind co-founder to Microsoft's AI chief",
  "Yoshua Bengio: the safety activist who helped spark the revolution he now warns against",
  "The inference scaling wars: why reasoning models are rewriting the rules",
  "OpenAI vs. Anthropic: the great AI safety schism explained",
  "Fei-Fei Li and the ImageNet moment that changed AI history",
  "Arthur Mensch and Mistral AI: Europe's bid to stay in the race",
  "Satya Nadella's billion-dollar bet on OpenAI — and what he got in return",
  "The AGI timeline debate: who is right and what it means for humanity",
  "Sam Altman's 2026 roadmap: enterprise AI, ads in ChatGPT, and the path to $280B",
  "The model context window wars: why Gemini's 1M tokens is a bigger deal than it sounds",
  "AI agents in 2026: what's actually shipping vs. what's still vaporware",
  "The open-source AI rebellion: how Meta and Mistral are fighting the closed-model cartel",
  "Agentic coding tools: how Cursor, Windsurf, and Claude Code are replacing the IDE"
];

async function generateArticleWithClaude(topic) {
  const system = `You are a senior journalist at Aliss, the first fully AI-autonomous news publication. You cover the AI arms race with the authority of The Atlantic and the speed of Reuters. Write accurate, factually grounded, long-form journalism about real people and companies shaping artificial intelligence. Your writing is witty, sharp, and intellectually serious.`;

  const userMsg = `Write a compelling long-form article about: ${topic}

Return ONLY a raw JSON object — no markdown fences, no extra text. Fields:
{
  "title": "Headline under 80 characters — punchy and specific",
  "subtitle": "One sharp sentence expanding on the headline",
  "summary": "2-3 sentence summary for card previews — make it compelling",
  "category": "Profile OR Analysis OR Opinion OR Research OR Industry OR News",
  "tags": ["tag1", "tag2", "tag3", "tag4"],
  "imagePrompt": "Descriptive Stable Diffusion prompt for a relevant image — be specific and vivid, e.g. 'professional portrait of a tech CEO in dark server room, dramatic blue lighting, photorealistic' or 'NVIDIA GPU chips glowing neon green on circuit board, macro photography, dark background'",
  "body": "Full article HTML. Rules: <p> for paragraphs (use class=\\"drop-cap\\" on the first); <h2> for section headers (5+ sections); <div class=\\"pull-quote\\">quote<cite>— Attribution, Source, Year</cite></div> for pull quotes (2+ minimum); do NOT include the title. Minimum 900 words. Be specific with facts, dates, and figures."
}`;

  const raw = await callClaude(system, userMsg, 4000);
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("No JSON in Claude response");

  const data = JSON.parse(match[0]);
  const title = String(data.title || topic).trim();
  const imagePrompt = String(data.imagePrompt || `${topic} artificial intelligence technology`);

  const doc = {
    slug:        slugify(title),
    title,
    subtitle:    String(data.subtitle   || "").trim(),
    content:     "",
    summary:     String(data.summary    || "").trim(),
    body:        String(data.body       || "").trim(),
    tags:        Array.isArray(data.tags) ? data.tags.map(String) : ["AI"],
    category:    String(data.category   || "Analysis"),
    source:      "Aliss",
    imageUrl:    getImageUrl(imagePrompt, strHash(title)),
    isExternal:  false,
    isGenerated: true,
    publishedAt: new Date()
  };

  try {
    const saved = await Article.findOneAndUpdate(
      { title: doc.title, source: "Aliss" },
      { $setOnInsert: doc },
      { upsert: true, new: true }
    );
    return saved;
  } catch {
    return await Article.findOne({ title: doc.title });
  }
}

/* ======================
   WITTY TICKER GENERATION
====================== */

let cachedTicker = null;

async function generateWittyTicker() {
  if (!ANTHROPIC_KEY) return null;
  try {
    // Get recent article headlines for context
    let recent = [];
    if (isMongoReady()) {
      recent = await Article.find().sort({ publishedAt: -1 }).limit(10).select("title").lean();
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

  if (isMongoReady()) {
    await Ticker.deleteMany({});
    await Ticker.create({ headlines, updatedAt: new Date() });
  }

  io.emit("tickerUpdate", { headlines });
  console.log("Ticker refreshed:", headlines[0]);
}

/* ======================
   SEEDING
====================== */

const ORIGINAL_ARTICLES = [
  {
    title:    "Sam Altman: The Architect of the AI Gold Rush",
    subtitle: "From Stanford dropout to Y Combinator president to CEO of the most valuable AI company on Earth.",
    summary:  "How one man bet everything on AGI — and, so far, won. Sam Altman's journey from Loopt to OpenAI, ChatGPT, and a $300B empire.",
    tags:     ["Profile", "OpenAI"],
    source:   "Aliss Editorial",
    category: "Profile",
    slug:     "article-altman",
    imageUrl: getImageUrl("tech CEO boardroom silicon valley dark dramatic lighting professional portrait", strHash("altman")),
    isExternal: false
  },
  {
    title:    "Ilya Sutskever: The Scientist Who Walked Away",
    subtitle: "He helped build ChatGPT, tried to fire Sam Altman, then vanished. Now he's back with $3 billion.",
    summary:  "Co-creator of AlexNet. OpenAI's chief scientist. The man who voted to fire Sam Altman. Now running Safe Superintelligence Inc. with $3B and no product.",
    tags:     ["Profile", "Safety"],
    source:   "Aliss Editorial",
    category: "Profile",
    slug:     "article-sutskever",
    imageUrl: getImageUrl("AI researcher neural network abstract blue light dark background photorealistic", strHash("sutskever")),
    isExternal: false
  },
  {
    title:    "Andrej Karpathy: The Teacher Who Shaped Modern AI",
    subtitle: "From Rubik's cube tutorials to Tesla Autopilot to reimagining education with AI.",
    summary:  "From OpenAI founding member to Tesla AI director to educator — a look at one of AI's most trusted and insightful voices in the field.",
    tags:     ["Profile", "Education"],
    source:   "Aliss Editorial",
    category: "Profile",
    slug:     "article-karpathy",
    imageUrl: "/assets/andrej-karpathy.jpg",
    isExternal: false
  }
];

async function seedOriginalArticles() {
  if (!isMongoReady()) return;
  for (const article of ORIGINAL_ARTICLES) {
    await Article.updateOne(
      { slug: article.slug },
      { $setOnInsert: { ...article, publishedAt: new Date() } },
      { upsert: true }
    );
  }
  console.log("Profile articles seeded.");
}

const SEED_TOPICS = AI_TOPICS.slice(0, 6);
let seeding = false;

async function seedGeneratedArticles() {
  if (seeding || !isMongoReady() || !ANTHROPIC_KEY) return;
  const count = await Article.countDocuments({ isGenerated: true });
  if (count >= 4) return;

  seeding = true;
  console.log("Seeding Claude-generated articles...");
  for (const topic of SEED_TOPICS.slice(0, 4 - count)) {
    try {
      const article = await generateArticleWithClaude(topic);
      console.log(`Seeded: ${article?.title?.slice(0, 60)}`);
      io.emit("newArticle", article);
      await new Promise(r => setTimeout(r, 4000));
    } catch (e) {
      console.error(`Seed failed: ${e.message}`);
    }
  }
  seeding = false;
  console.log("Seeding complete.");
}

/* ======================
   AUTH MIDDLEWARE
====================== */

function auth(req, res, next) {
  const token = req.header("Authorization");
  if (!token) return res.status(401).json({ msg: "No token" });
  try {
    req.user = jwt.verify(token, process.env.JWT_SECRET);
    next();
  } catch {
    res.status(401).json({ msg: "Invalid token" });
  }
}

/* ======================
   AUTH ROUTES
====================== */

app.post("/api/register", async (req, res) => {
  const { email, password } = req.body;
  const hashed = await bcrypt.hash(password, 10);
  const user = new User({ email, password: hashed });
  await user.save();
  res.json({ msg: "Admin created" });
});

app.post("/api/login", async (req, res) => {
  const { email, password } = req.body;
  const user = await User.findOne({ email });
  if (!user) return res.status(400).json({ msg: "User not found" });
  const valid = await bcrypt.compare(password, user.password);
  if (!valid) return res.status(400).json({ msg: "Invalid credentials" });
  const token = jwt.sign({ id: user._id, role: user.role }, process.env.JWT_SECRET, { expiresIn: "1d" });
  res.json({ token });
});

/* ======================
   PUBLIC ROUTES
====================== */

app.post("/api/signup", async (req, res) => {
  try {
    const email = String(req.body?.email || "").trim().toLowerCase();
    if (!email || !email.includes("@")) return res.status(400).json({ msg: "Invalid email" });

    const result = await Signup.updateOne({ email }, { $setOnInsert: { email } }, { upsert: true });
    const isNew = result.upsertedCount > 0;

    // Send welcome email via Resend if configured
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
    if (isMongoReady()) recent = await Article.find().sort({ publishedAt: -1 }).limit(8).select("title summary").lean();
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

app.get("/api/articles", async (req, res) => {
  try {
    const fallback = ORIGINAL_ARTICLES.map(a => ({ ...a, publishedAt: new Date() }));
    if (!isMongoReady()) return res.json(fallback);
    const articles = await Article.find().sort({ publishedAt: -1 }).select("-body").limit(60);
    if (!articles.length) return res.json(fallback);
    res.json(articles);
  } catch {
    res.json(ORIGINAL_ARTICLES.map(a => ({ ...a, publishedAt: new Date() })));
  }
});

app.get("/api/articles/:slugOrId", async (req, res) => {
  const param = String(req.params.slugOrId || "").trim();
  if (!param) return res.status(400).json({ msg: "Identifier required" });
  try {
    if (isMongoReady()) {
      if (/^[a-f0-9]{24}$/i.test(param)) {
        const byId = await Article.findById(param).lean();
        if (byId) return res.json(byId);
      }
      const bySlug = await Article.findOne({ slug: param }).lean();
      if (bySlug) return res.json(bySlug);
    }
    const local = ORIGINAL_ARTICLES.find(a => a.slug === slugify(param));
    if (local) return res.json({ ...local, publishedAt: new Date() });
    res.status(404).json({ msg: "Not found" });
  } catch {
    res.status(500).json({ msg: "Failed to load article" });
  }
});

app.get("/api/search", async (req, res) => {
  const q = String(req.query.q || "").trim();
  if (!q) return res.json([]);
  try {
    if (!isMongoReady()) {
      const lower = q.toLowerCase();
      return res.json(ORIGINAL_ARTICLES.filter(a =>
        a.title.toLowerCase().includes(lower) || (a.summary || "").toLowerCase().includes(lower)
      ).map(a => ({ ...a, publishedAt: new Date() })));
    }
    // Try full-text search first, fall back to regex
    let results = [];
    try {
      results = await Article.find(
        { $text: { $search: q } },
        { score: { $meta: "textScore" } }
      ).sort({ score: { $meta: "textScore" } }).select("-body").limit(20).lean();
    } catch {
      results = await Article.find({
        $or: [
          { title: { $regex: q, $options: "i" } },
          { summary: { $regex: q, $options: "i" } },
          { tags: { $regex: q, $options: "i" } }
        ]
      }).sort({ publishedAt: -1 }).select("-body").limit(20).lean();
    }
    res.json(results);
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
    // Try cached ticker first
    if (cachedTicker && cachedTicker.length) {
      return res.json({ headlines: cachedTicker, updatedAt: new Date().toISOString() });
    }
    if (isMongoReady()) {
      const ticker = await Ticker.findOne().sort({ updatedAt: -1 }).lean();
      if (ticker?.headlines?.length) {
        cachedTicker = ticker.headlines;
        return res.json({ headlines: ticker.headlines, updatedAt: ticker.updatedAt });
      }
      // Fall back to article titles
      const articles = await Article.find().sort({ publishedAt: -1 }).limit(8).lean();
      if (articles.length) {
        const h = articles.map(a => `${a.category || "AI"}: ${a.title}`);
        return res.json({ headlines: h, updatedAt: new Date().toISOString() });
      }
    }
    res.json({ headlines: fallback, updatedAt: new Date().toISOString() });
  } catch {
    res.json({ headlines: fallback, updatedAt: new Date().toISOString() });
  }
});

app.post("/api/articles", auth, async (req, res) => {
  const article = new Article(req.body);
  await article.save();
  io.emit("newArticle", article);
  res.json(article);
});

app.delete("/api/articles/:id", auth, async (req, res) => {
  await Article.findByIdAndDelete(req.params.id);
  res.json({ msg: "Deleted" });
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
  // Send cached ticker on connect
  if (cachedTicker) socket.emit("tickerUpdate", { headlines: cachedTicker });
  console.log("Client connected:", socket.id);
});

/* ======================
   MIGRATE: SET ALL SOURCES TO "Aliss"
====================== */

async function migrateSourceToAliss() {
  if (!isMongoReady()) return;
  try {
    const result = await Article.updateMany(
      { source: { $ne: "Aliss" } },
      { $set: { source: "Aliss" } }
    );
    if (result.modifiedCount > 0) console.log(`Migrated ${result.modifiedCount} articles to source=Aliss`);
  } catch (e) {
    console.error("Source migration failed:", e.message);
  }
}

/* ======================
   POLISH SHORT ARTICLES
====================== */

let polishing = false;
async function polishShortArticles() {
  if (polishing || !isMongoReady() || !ANTHROPIC_KEY) return;
  polishing = true;

  try {
    // Find articles with no body, empty body, or body under 1500 chars
    const candidates = await Article.find({
      slug: { $nin: ["article-altman", "article-sutskever", "article-karpathy"] },
      $or: [
        { body: { $exists: false } },
        { body: null },
        { body: "" },
        { $expr: { $lt: [{ $strLenCP: { $ifNull: ["$body", ""] } }, 1500] } }
      ]
    }).limit(8).lean();

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
  "imagePrompt": "Vivid Stable Diffusion image prompt, specific and cinematic",
  "body": "Full HTML article: <p class=\\"drop-cap\\"> for first paragraph, <h2> headers (5+), <div class=\\"pull-quote\\">quote<cite>— source</cite></div> pull quotes (2+). Minimum 1000 words. Be specific, factual, punchy."
}`,
          4000
        );

        const match = raw.match(/\{[\s\S]*\}/);
        if (!match) continue;
        const data = JSON.parse(match[0]);

        await Article.updateOne(
          { _id: article._id },
          {
            $set: {
              title:       String(data.title    || article.title).trim(),
              subtitle:    String(data.subtitle || "").trim(),
              summary:     String(data.summary  || "").trim(),
              body:        String(data.body     || "").trim(),
              tags:        Array.isArray(data.tags) ? data.tags.map(String) : (article.tags || ["AI"]),
              category:    String(data.category || article.category || "Analysis"),
              source:      "Aliss",
              imageUrl:    getImageUrl(data.imagePrompt || article.title, strHash(article.title)),
              isGenerated: true,
              isExternal:  false
            }
          }
        );
        console.log(`Polished: ${article.title.slice(0, 55)}`);
        io.emit("articleUpdated", { id: article._id });
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
  if (!isMongoReady() || !ANTHROPIC_KEY) return;
  console.log("Fetching AI news from Hacker News...");
  try {
    const { data } = await axios.get(
      "https://hn.algolia.com/api/v1/search?query=artificial+intelligence+OR+LLM+OR+Claude+OR+OpenAI+OR+Anthropic+OR+Nvidia+OR+Gemini+OR+DeepSeek&tags=story&hitsPerPage=20",
      { timeout: 20000 }
    );
    const hits = Array.isArray(data?.hits) ? data.hits : [];

    // Only take truly new stories (not already in DB)
    const newHits = [];
    for (const item of hits) {
      const title = String(item.title || item.story_title || "").trim();
      if (!title) continue;
      const exists = await Article.exists({ title });
      if (!exists) newHits.push(item);
      if (newHits.length >= 4) break; // Write up to 4 full articles per hour from real news
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
        // If generation fails, store as a stub for later polishing
        try {
          const doc = {
            slug:        slugify(title),
            title,
            content:     sourceUrl,
            summary:     rawSummary || title,
            tags:        ["AI", "News"],
            category:    "News",
            source:      "Aliss",
            imageUrl:    getImageUrl("AI technology news breaking story dark cinematic", strHash(title)),
            isExternal:  false,
            isGenerated: false,
            publishedAt: item.created_at_i ? new Date(item.created_at_i * 1000) : new Date()
          };
          await Article.updateOne({ title: doc.title }, { $setOnInsert: doc }, { upsert: true });
        } catch {}
        console.error(`HN article gen failed: ${e.message}`);
      }
    }
  } catch (e) {
    console.error("HN fetch failed:", e?.message);
  }
}

/* ======================
   AUTO GENERATE NEW ARTICLES (EVERY 30 MIN)
====================== */

let topicIndex = 4;
async function autoGenerateArticle() {
  if (!isMongoReady() || !ANTHROPIC_KEY) return;
  const topic = AI_TOPICS[topicIndex % AI_TOPICS.length];
  topicIndex++;
  console.log(`Auto-generating [${topicIndex}]: ${topic.slice(0, 55)}...`);
  try {
    const article = await generateArticleWithClaude(topic);
    if (article) {
      io.emit("newArticle", article);
      console.log("Published:", article?.title?.slice(0, 55));
    }
  } catch (e) {
    console.error("Auto-gen failed:", e.message);
  }
}

cron.schedule("0 * * * *",    fetchHNNews);           // Real news every hour
cron.schedule("*/30 * * * *", autoGenerateArticle);   // Original articles every 30 min
cron.schedule("*/15 * * * *", refreshTicker);         // Witty ticker every 15 min
cron.schedule("0 */3 * * *",  polishShortArticles);   // Polish short articles every 3h

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
