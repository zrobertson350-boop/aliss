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
        await seedOriginalArticles();
        await seedGeneratedArticles();
        fetchHNNews();
        refreshTicker();
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
    await Signup.updateOne({ email }, { $setOnInsert: { email } }, { upsert: true });
    res.json({ msg: "Signed up" });
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
   AUTO FETCH HN NEWS (HOURLY)
====================== */

async function fetchHNNews() {
  if (!isMongoReady()) return;
  console.log("Fetching AI news from Hacker News...");
  try {
    const { data } = await axios.get(
      "https://hn.algolia.com/api/v1/search?query=artificial+intelligence+OR+LLM+OR+Claude+OR+OpenAI+OR+Anthropic+OR+Nvidia+OR+Gemini&tags=story&hitsPerPage=15",
      { timeout: 20000 }
    );
    const hits = Array.isArray(data?.hits) ? data.hits : [];
    let added = 0;
    for (const item of hits) {
      const title = String(item.title || item.story_title || "").trim();
      if (!title) continue;
      const summary = cleanSummary(item.story_text || item._highlightResult?.title?.value || "");
      const doc = {
        slug:        slugify(title),
        title,
        content:     item.url || item.story_url || "",
        summary:     summary || "Latest AI news from Hacker News.",
        tags:        ["AI", "News"],
        category:    "News",
        source:      "Hacker News",
        imageUrl:    getImageUrl("AI technology news silicon valley computers dark blue abstract", strHash(title)),
        isExternal:  true,
        publishedAt: item.created_at_i ? new Date(item.created_at_i * 1000) : new Date()
      };
      const result = await Article.updateOne(
        { title: doc.title, source: doc.source },
        { $setOnInsert: doc },
        { upsert: true }
      );
      if (result.upsertedCount > 0) {
        added++;
        const saved = await Article.findOne({ title: doc.title, source: doc.source }).lean();
        if (saved) io.emit("newArticle", saved);
      }
    }
    console.log(`HN fetch: ${added} new article(s)`);
  } catch (e) {
    console.error("HN fetch failed:", e?.message);
  }
}

/* ======================
   AUTO GENERATE (EVERY 2H)
====================== */

let topicIndex = 4; // Start after seed topics
async function autoGenerateArticle() {
  if (!isMongoReady() || !ANTHROPIC_KEY) return;
  const topic = AI_TOPICS[topicIndex % AI_TOPICS.length];
  topicIndex++;
  console.log(`Auto-generating: ${topic.slice(0, 60)}...`);
  try {
    const article = await generateArticleWithClaude(topic);
    if (article) {
      io.emit("newArticle", article);
      console.log("Done:", article?.title?.slice(0, 60));
    }
  } catch (e) {
    console.error("Auto-gen failed:", e.message);
  }
}

cron.schedule("0 * * * *",    fetchHNNews);
cron.schedule("0 */2 * * *",  autoGenerateArticle);
cron.schedule("*/30 * * * *", refreshTicker);

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
