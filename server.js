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

// Support both ANTHROPIC_API_KEY (Render) and CLAUDE_API_KEY (legacy)
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY || process.env.CLAUDE_API_KEY;

const mongoUri = process.env.MONGO_URI;
if (mongoUri && (mongoUri.startsWith("mongodb://") || mongoUri.startsWith("mongodb+srv://"))) {
  mongoose.connect(mongoUri)
    .then(() => {
      console.log("MongoDB Connected");
      setTimeout(() => { seedOriginalArticles(); seedGeneratedArticles(); }, 8000);
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
  content:     String,   // External URL or short text for HN articles
  summary:     String,   // 2-3 sentence card preview
  body:        String,   // Full HTML for generated articles
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

const UserSchema = new mongoose.Schema({
  email:    { type: String, unique: true },
  password: String,
  role:     { type: String, default: "admin" }
});

const SignupSchema = new mongoose.Schema({
  email:     { type: String, required: true, unique: true },
  createdAt: { type: Date, default: Date.now }
});

const Article = mongoose.model("Article", ArticleSchema);
const User    = mongoose.model("User", UserSchema);
const Signup  = mongoose.model("Signup", SignupSchema);

/* ======================
   HELPERS
====================== */

function isMongoReady() {
  return mongoose.connection.readyState === 1;
}

function slugify(value) {
  return String(value || "")
    .toLowerCase().trim()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

function cleanSummary(v) {
  return String(v || "").replace(/\s+/g, " ").trim().slice(0, 260);
}

// Deterministic beautiful photos via picsum.photos
function getImageUrl(seed) {
  const s = (seed || "artificial-intelligence")
    .toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 50);
  return `https://picsum.photos/seed/${s}/800/450`;
}

/* ======================
   CLAUDE API
====================== */

async function callClaude(system, userMsg, maxTokens = 2000) {
  if (!ANTHROPIC_KEY) throw new Error("No Anthropic API key configured");
  const response = await axios.post(
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
  return response.data?.content?.[0]?.text || "";
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
  "The AGI timeline debate: who is right and what it means for humanity"
];

async function generateArticleWithClaude(topic) {
  const system = `You are a senior journalist at Aliss, an independent editorial publication covering the AI arms race. Your writing is authoritative, nuanced, and intellectually serious — like The Atlantic meets MIT Technology Review. Write accurate, factually grounded journalism about real people and companies shaping artificial intelligence.`;

  const userMsg = `Write a compelling long-form article about: ${topic}

Return ONLY a raw JSON object — no markdown code fences, no extra commentary, just valid JSON. Fields:
{
  "title": "Headline under 80 characters",
  "subtitle": "One compelling sentence expanding on the headline",
  "summary": "2-3 sentence summary for card previews",
  "category": "Profile OR Analysis OR Opinion OR Research OR Industry OR News",
  "tags": ["tag1", "tag2", "tag3", "tag4"],
  "imageSeed": "2-3 lowercase words for image, e.g. neural network, silicon chip",
  "body": "Full article HTML. Rules: <p> for paragraphs; <h2> for section headers (at least 5 sections); <div class=\\"pull-quote\\">quote text<cite>— Attribution, Source, Year</cite></div> for pull quotes (include at least 2); first <p> must have class=\\"drop-cap\\"; do NOT include the title. Minimum 900 words."
}`;

  const raw = await callClaude(system, userMsg, 4000);
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("No JSON in Claude response");

  const data = JSON.parse(match[0]);
  const title = String(data.title || topic).trim();

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
    imageUrl:    getImageUrl(data.imageSeed || "artificial intelligence"),
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
   SEED FUNCTIONS
====================== */

const ORIGINAL_ARTICLES = [
  {
    title:    "The New AI Product Stack in 2026",
    summary:  "Why orchestration, retrieval, and evaluation layers are becoming the real moat in AI products.",
    content:  "The first AI wave rewarded model access. The second rewards product architecture. In 2026, the most durable teams are not simply wrapping a model endpoint — they are building resilient stacks with retrieval controls, evaluation loops, and domain workflows that improve with every user interaction.",
    tags:     ["Industry", "Architecture"],
    source:   "Aliss Editorial",
    category: "Analysis",
    imageUrl: getImageUrl("product-stack-ai"),
    isExternal: false
  },
  {
    title:    "Why LLM Evaluation Is Finally Going Production-Grade",
    summary:  "Benchmarks are useful, but real products now need task-level reliability and continuous eval pipelines.",
    content:  "Static benchmarks helped compare models, but production systems expose a different truth: reliability is contextual. Teams now run evaluation suites tied to their own tasks — customer support, research synthesis, legal drafting, and code generation — then gate releases with pass/fail thresholds.",
    tags:     ["Research", "LLMs"],
    source:   "Aliss Editorial",
    category: "Research",
    imageUrl: getImageUrl("llm-evaluation"),
    isExternal: false
  },
  {
    title:    "AI Infrastructure Economics: The Margin Battle Has Begun",
    summary:  "As usage explodes, cost control and model routing decide who survives.",
    content:  "The unit economics of AI products are no longer theoretical. As request volumes increase, infrastructure decisions directly shape margins. Teams that route intelligently across model tiers, cache aggressively, and trim unnecessary context windows can cut serving costs dramatically without hurting user quality.",
    tags:     ["Analysis", "Infrastructure"],
    source:   "Aliss Editorial",
    category: "Industry",
    imageUrl: getImageUrl("infrastructure-economics"),
    isExternal: false
  },
  {
    title:    "Multimodal Agents Are Real, but Narrowly Useful",
    summary:  "Agent systems are improving quickly when tasks are constrained and verifiable.",
    content:  "The broad promise of autonomous agents remains ahead of reality, but constrained agents are already useful. Workflows like structured data extraction, incident triage, and document-heavy QA now benefit from multimodal models that reason across text, tables, screenshots, and logs.",
    tags:     ["Opinion", "Agents"],
    source:   "Aliss Editorial",
    category: "Opinion",
    imageUrl: getImageUrl("multimodal-agents"),
    isExternal: false
  },
  {
    title:    "Safety and Governance Moves From Policy to Product",
    summary:  "Governance is becoming embedded in release workflows, not just written in policy docs.",
    content:  "Safety has moved from high-level principle to implementation detail. Leading teams now attach policy checks to deployment pipelines, maintain model behavior audits, and track refusal quality alongside latency and cost.",
    tags:     ["Policy", "Safety"],
    source:   "Aliss Editorial",
    category: "Analysis",
    imageUrl: getImageUrl("ai-safety-governance"),
    isExternal: false
  },
  {
    title:    "Developer Platform Updates: The Rise of AI-Native Tooling",
    summary:  "The best developer platforms now treat AI as a first-class runtime, not a plugin.",
    content:  "Developer tooling is being rebuilt around AI primitives: structured prompts, eval suites, model routing, and observability for prompts and completions. The strongest platforms reduce cognitive load by giving teams versioned prompts, replayable test cases, and traceability across entire AI workflows.",
    tags:     ["Developers", "Platforms"],
    source:   "Aliss Editorial",
    category: "Industry",
    imageUrl: getImageUrl("developer-platform-ai"),
    isExternal: false
  }
];

async function seedOriginalArticles() {
  if (!isMongoReady()) return;
  const originals = ORIGINAL_ARTICLES.map(a => ({ ...a, slug: slugify(a.title), publishedAt: new Date() }));
  for (const article of originals) {
    await Article.updateOne({ slug: article.slug }, { $setOnInsert: article }, { upsert: true });
  }
  console.log("Original articles seeded.");
}

const SEED_TOPICS = AI_TOPICS.slice(0, 4);
let seeding = false;

async function seedGeneratedArticles() {
  if (seeding || !isMongoReady() || !ANTHROPIC_KEY) return;
  const count = await Article.countDocuments({ isGenerated: true });
  if (count > 0) return;

  seeding = true;
  console.log("Seeding initial Claude-generated articles...");
  for (const topic of SEED_TOPICS) {
    try {
      const article = await generateArticleWithClaude(topic);
      console.log(`Seeded: ${article?.title?.slice(0, 60)}`);
      await new Promise(r => setTimeout(r, 3000));
    } catch (e) {
      console.error(`Seed failed: ${e.message}`);
    }
  }
  console.log("Seeding complete.");
  seeding = false;
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

  // Build context from recent articles
  let contextBlock = "";
  try {
    let recent = [];
    if (isMongoReady()) {
      recent = await Article.find().sort({ publishedAt: -1 }).limit(6).lean();
    }
    if (!recent.length) recent = ORIGINAL_ARTICLES.slice(0, 6);
    contextBlock = recent.map(a => `- ${a.title}: ${String(a.summary || "").slice(0, 180)}`).join("\n");
  } catch {}

  if (!ANTHROPIC_KEY) {
    return res.json({ reply: "Alice is ready — add ANTHROPIC_API_KEY to enable live responses." });
  }

  try {
    const reply = await callClaude(
      `You are Alice, the AI assistant for Aliss — an editorial publication covering the AI arms race. Be concise, sharp, and accurate. Recent site context:\n${contextBlock}`,
      message,
      600
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
    if (!isMongoReady()) {
      return res.json(ORIGINAL_ARTICLES.map(a => ({ ...a, slug: slugify(a.title), publishedAt: new Date() })));
    }
    const articles = await Article.find().sort({ publishedAt: -1 }).select("-body").limit(50);
    if (!articles.length) {
      return res.json(ORIGINAL_ARTICLES.map(a => ({ ...a, slug: slugify(a.title), publishedAt: new Date() })));
    }
    res.json(articles);
  } catch {
    res.json(ORIGINAL_ARTICLES.map(a => ({ ...a, slug: slugify(a.title), publishedAt: new Date() })));
  }
});

// Fetch by slug (existing articles) OR MongoDB ObjectId (generated articles)
app.get("/api/articles/:slugOrId", async (req, res) => {
  const param = String(req.params.slugOrId || "").trim();
  if (!param) return res.status(400).json({ msg: "Identifier required" });

  try {
    if (isMongoReady()) {
      // Try ObjectId first (24-char hex)
      if (/^[a-f0-9]{24}$/i.test(param)) {
        const byId = await Article.findById(param).lean();
        if (byId) return res.json(byId);
      }
      // Try slug
      const bySlug = await Article.findOne({ slug: param }).lean();
      if (bySlug) return res.json(bySlug);
    }

    // Fallback to in-memory originals
    const slug = slugify(param);
    const local = ORIGINAL_ARTICLES.find(a => slugify(a.title) === slug);
    if (local) return res.json({ ...local, slug, publishedAt: new Date() });

    res.status(404).json({ msg: "Article not found" });
  } catch {
    res.status(500).json({ msg: "Failed to load article" });
  }
});

app.get("/api/live-updates", async (_req, res) => {
  const fallback = [
    "Dario Amodei: Anthropic's safety-first approach may define the next era of AI",
    "Nvidia's Jensen Huang: the GPU shortage is over — the AI race is just beginning",
    "DeepSeek continues to challenge frontier labs on reasoning benchmarks",
    "Meta's Llama 4 set to release with 400B parameter multimodal capabilities",
    "xAI's Colossus supercomputer now hosts 200,000 H100 GPUs"
  ];
  try {
    if (!isMongoReady()) return res.json({ headlines: fallback, updatedAt: new Date().toISOString() });
    const articles = await Article.find().sort({ publishedAt: -1 }).limit(12).lean();
    const headlines = articles.length
      ? articles.map(a => {
          const title = String(a.title || "").trim();
          const tag = Array.isArray(a.tags) && a.tags[0] ? String(a.tags[0]) : "AI";
          return `${tag}: ${title.length > 100 ? title.slice(0, 97) + "..." : title}`;
        })
      : fallback;
    res.json({ headlines, updatedAt: new Date().toISOString() });
  } catch {
    res.status(500).json({ headlines: fallback, updatedAt: new Date().toISOString() });
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

// Trigger Claude article generation
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
      "https://hn.algolia.com/api/v1/search?query=artificial+intelligence+OR+LLM+OR+Claude+OR+OpenAI+OR+Anthropic&tags=story&hitsPerPage=12",
      { timeout: 20000 }
    );
    const hits = Array.isArray(data?.hits) ? data.hits : [];
    let added = 0;
    for (const item of hits) {
      const title = String(item.title || item.story_title || "").trim();
      if (!title) continue;
      const doc = {
        slug:        slugify(title),
        title,
        content:     item.url || item.story_url || "",
        summary:     cleanSummary(item.story_text || item._highlightResult?.title?.value || "Latest AI news."),
        tags:        ["AI", "News"],
        category:    "News",
        source:      "Hacker News",
        imageUrl:    getImageUrl("tech-news"),
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
        const saved = await Article.findOne({ title: doc.title, source: doc.source });
        if (saved) io.emit("newArticle", saved);
      }
    }
    console.log(`HN fetch: ${added} new article(s)`);
  } catch (e) {
    console.error("HN fetch failed:", e?.message);
  }
}

/* ======================
   AUTO GENERATE (EVERY 6H)
====================== */

let topicIndex = 0;
async function autoGenerateArticle() {
  if (!isMongoReady() || !ANTHROPIC_KEY) return;
  const topic = AI_TOPICS[topicIndex % AI_TOPICS.length];
  topicIndex++;
  console.log(`Auto-generating: ${topic.slice(0, 60)}...`);
  try {
    const article = await generateArticleWithClaude(topic);
    if (article) io.emit("newArticle", article);
    console.log("Auto-generation done:", article?.title?.slice(0, 60));
  } catch (e) {
    console.error("Auto-generation failed:", e.message);
  }
}

cron.schedule("0 * * * *",   fetchHNNews);
cron.schedule("0 */6 * * *", autoGenerateArticle);

setTimeout(fetchHNNews, 10000);

/* ======================
   STATIC FRONTEND
====================== */

app.use(express.static(path.join(__dirname)));

app.get(["/", "/aliss"], (_req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

/* ======================
   START SERVER
====================== */

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));
