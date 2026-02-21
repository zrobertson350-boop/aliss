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
const io = socketIo(server, {
  cors: { origin: "*" }
});

app.set("trust proxy", 1);
app.use(cors());
app.use(express.json());
app.use(helmet({ contentSecurityPolicy: false }));

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100
});
app.use(limiter);

const mongoUri = process.env.MONGO_URI;
if (mongoUri && (mongoUri.startsWith("mongodb://") || mongoUri.startsWith("mongodb+srv://"))) {
  mongoose.connect(mongoUri)
    .then(() => console.log("MongoDB Connected"))
    .catch(err => console.error(err));
} else {
  console.log("MongoDB not connected: set a valid MONGO_URI in .env");
}

/* ======================
   MODELS
====================== */

const ArticleSchema = new mongoose.Schema({
  slug: { type: String, unique: true, sparse: true },
  title: { type: String, required: true },
  content: String,
  summary: String,
  tags: [String],
  source: String,
  isExternal: { type: Boolean, default: false },
  publishedAt: { type: Date, default: Date.now }
});
ArticleSchema.index({ title: 1, source: 1 }, { unique: true });
ArticleSchema.index({ slug: 1 }, { unique: true, sparse: true });

const UserSchema = new mongoose.Schema({
  email: { type: String, unique: true },
  password: String,
  role: { type: String, default: "admin" }
});

const SignupSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  createdAt: { type: Date, default: Date.now }
});

const Article = mongoose.model("Article", ArticleSchema);
const User = mongoose.model("User", UserSchema);
const Signup = mongoose.model("Signup", SignupSchema);

const ORIGINAL_ARTICLES = [
  {
    title: "The New AI Product Stack in 2026",
    summary: "Why orchestration, retrieval, and evaluation layers are becoming the real moat in AI products.",
    content: "The first AI wave rewarded model access. The second rewards product architecture. In 2026, the most durable teams are not simply wrapping a model endpoint—they are building resilient stacks with retrieval controls, evaluation loops, and domain workflows that improve with every user interaction. The practical lesson is clear: your moat is no longer just model quality, but the quality of your system design, feedback collection, and vertical-specific execution.",
    tags: ["Industry", "Architecture"],
    source: "Aliss Editorial",
    isExternal: false
  },
  {
    title: "Why LLM Evaluation Is Finally Going Production-Grade",
    summary: "Benchmarks are useful, but real products now need task-level reliability and continuous eval pipelines.",
    content: "Static benchmarks helped compare models, but production systems expose a different truth: reliability is contextual. Teams now run evaluation suites tied to their own tasks—customer support, research synthesis, legal drafting, and code generation—then gate releases with pass/fail thresholds. This shift turns evals from academic scorecards into operational controls. It also changes hiring: AI teams increasingly need evaluation engineers as much as prompt engineers.",
    tags: ["Research", "LLMs"],
    source: "Aliss Editorial",
    isExternal: false
  },
  {
    title: "AI Infrastructure Economics: The Margin Battle Has Begun",
    summary: "As usage explodes, cost control and model routing decide who survives.",
    content: "The unit economics of AI products are no longer theoretical. As request volumes increase, infrastructure decisions directly shape margins. Teams that route intelligently across model tiers, cache aggressively, and trim unnecessary context windows can cut serving costs dramatically without hurting user quality. In the coming year, infrastructure discipline—not hype—will separate sustainable businesses from expensive demos.",
    tags: ["Analysis", "Infrastructure"],
    source: "Aliss Editorial",
    isExternal: false
  },
  {
    title: "Multimodal Agents Are Real, but Narrowly Useful",
    summary: "Agent systems are improving quickly when tasks are constrained and verifiable.",
    content: "The broad promise of autonomous agents remains ahead of reality, but constrained agents are already useful. Workflows like structured data extraction, incident triage, and document-heavy QA now benefit from multimodal models that reason across text, tables, screenshots, and logs. The winning pattern is narrow scope plus explicit checks. When tasks are measurable, agent performance becomes tractable.",
    tags: ["Opinion", "Agents"],
    source: "Aliss Editorial",
    isExternal: false
  },
  {
    title: "Safety and Governance Moves From Policy to Product",
    summary: "Governance is becoming embedded in release workflows, not just written in policy docs.",
    content: "Safety has moved from high-level principle to implementation detail. Leading teams now attach policy checks to deployment pipelines, maintain model behavior audits, and track refusal quality alongside latency and cost. The result is pragmatic governance: testable controls instead of abstract commitments. For users, this means more consistent behavior and clearer boundaries in high-risk use cases.",
    tags: ["Policy", "Safety"],
    source: "Aliss Editorial",
    isExternal: false
  },
  {
    title: "Developer Platform Updates: The Rise of AI-Native Tooling",
    summary: "The best developer platforms now treat AI as a first-class runtime, not a plugin.",
    content: "Developer tooling is being rebuilt around AI primitives: structured prompts, eval suites, model routing, and observability for prompts and completions. The strongest platforms reduce cognitive load by giving teams versioned prompts, replayable test cases, and traceability across entire AI workflows. In short, we are watching the emergence of AI-native software engineering.",
    tags: ["Developers", "Platforms"],
    source: "Aliss Editorial",
    isExternal: false
  }
];

function slugify(value) {
  return String(value || "")
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

function buildOriginalArticles() {
  return ORIGINAL_ARTICLES.map((article) => ({
    ...article,
    slug: slugify(article.title),
    publishedAt: new Date()
  }));
}

/* ======================
   AUTH MIDDLEWARE
====================== */

function auth(req, res, next) {
  const token = req.header("Authorization");
  if (!token) return res.status(401).json({ msg: "No token" });

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch {
    res.status(401).json({ msg: "Invalid token" });
  }
}

/* ======================
   AUTH ROUTES
====================== */

// Register Admin (run once)
app.post("/api/register", async (req, res) => {
  const { email, password } = req.body;

  const hashed = await bcrypt.hash(password, 10);
  const user = new User({ email, password: hashed });

  await user.save();
  res.json({ msg: "Admin created" });
});

// Login
app.post("/api/login", async (req, res) => {
  const { email, password } = req.body;

  const user = await User.findOne({ email });
  if (!user) return res.status(400).json({ msg: "User not found" });

  const valid = await bcrypt.compare(password, user.password);
  if (!valid) return res.status(400).json({ msg: "Invalid credentials" });

  const token = jwt.sign(
    { id: user._id, role: user.role },
    process.env.JWT_SECRET,
    { expiresIn: "1d" }
  );

  res.json({ token });
});

/* ======================
   PUBLIC ROUTES
====================== */

app.post("/api/signup", async (req, res) => {
  try {
    const email = String(req.body?.email || "").trim().toLowerCase();
    if (!email || !email.includes("@")) {
      return res.status(400).json({ msg: "Invalid email" });
    }

    await Signup.updateOne(
      { email },
      { $setOnInsert: { email } },
      { upsert: true }
    );

    res.json({ msg: "Signed up" });
  } catch {
    res.status(500).json({ msg: "Signup failed" });
  }
});

app.post("/api/alice-chat", async (req, res) => {
  const message = String(req.body?.message || "").trim();
  if (!message) return res.status(400).json({ msg: "Message is required" });

  let contextArticles = [];
  try {
    if (isMongoReady()) {
      contextArticles = await Article.find()
        .sort({ publishedAt: -1 })
        .limit(6)
        .lean();
    }
  } catch {}

  if (!contextArticles.length) {
    contextArticles = buildOriginalArticles().slice(0, 6);
  }

  const contextBlock = contextArticles
    .map((article) => `- ${article.title}: ${String(article.summary || "").slice(0, 180)}`)
    .join("\n");

  const claudeKey = process.env.CLAUDE_API_KEY;
  if (!claudeKey) {
    return res.json({
      reply: `Alice is connected to Aliss content but missing a Claude key. Recent context:\n${contextBlock}`
    });
  }

  try {
    const response = await axios.post(
      "https://api.anthropic.com/v1/messages",
      {
        model: "claude-3-5-sonnet-latest",
        max_tokens: 500,
        system: `You are Alice, an AI assistant for the Aliss AI news website. Be concise, accurate, and editorial in tone. Use this recent site context when relevant:\n${contextBlock}`,
        messages: [{ role: "user", content: message }]
      },
      {
        headers: {
          "Content-Type": "application/json",
          "x-api-key": claudeKey,
          "anthropic-version": "2023-06-01"
        },
        timeout: 30000
      }
    );

    const text = response.data?.content?.[0]?.text || "I couldn't generate a response right now.";
    res.json({ reply: text });
  } catch {
    res.status(502).json({ msg: "Claude request failed" });
  }
});

/* ======================
   ARTICLE ROUTES
====================== */

// Get all articles
app.get("/api/articles", async (req, res) => {
  try {
    if (!isMongoReady()) {
      return res.json(buildOriginalArticles());
    }

    const articles = await Article.find().sort({ publishedAt: -1 });
    if (!articles.length) {
      return res.json(buildOriginalArticles());
    }
    res.json(articles);
  } catch {
    res.json(buildOriginalArticles());
  }
});

app.get("/api/articles/:slug", async (req, res) => {
  const slug = String(req.params.slug || "").trim();
  if (!slug) return res.status(400).json({ msg: "Slug is required" });

  try {
    if (isMongoReady()) {
      const article = await Article.findOne({ slug }).lean();
      if (article) return res.json(article);
    }

    const localArticle = buildOriginalArticles().find((article) => article.slug === slug);
    if (localArticle) return res.json(localArticle);

    return res.status(404).json({ msg: "Article not found" });
  } catch {
    return res.status(500).json({ msg: "Failed to load article" });
  }
});

function generateAiHeadline(article) {
  const title = String(article?.title || "").trim();
  const source = String(article?.source || "AI Wire").trim();
  if (!title) return "Live AI update: new developments are coming in.";

  const compactTitle = title.length > 120 ? `${title.slice(0, 117)}...` : title;
  const tag = Array.isArray(article?.tags) && article.tags.length ? String(article.tags[0]) : "AI";
  return `Live updates of AI · ${tag}: ${compactTitle} (${source})`;
}

app.get("/api/live-updates", async (_req, res) => {
  try {
    if (!isMongoReady()) {
      return res.json({
        headlines: [
          "Live updates of AI · models are evolving in real time.",
          "Live updates of AI · frontier labs are shipping weekly improvements.",
          "Live updates of AI · infrastructure and apps continue to scale."
        ],
        updatedAt: new Date().toISOString()
      });
    }

    const latestArticles = await Article.find()
      .sort({ publishedAt: -1 })
      .limit(12)
      .lean();

    const headlines = latestArticles.length
      ? latestArticles.map(generateAiHeadline)
      : [
          "Live updates of AI · no posts yet, first headlines are on the way.",
          "Live updates of AI · monitoring the latest model and product news.",
          "Live updates of AI · autonomous feed is warming up."
        ];

    res.json({
      headlines,
      updatedAt: new Date().toISOString()
    });
  } catch {
    res.status(500).json({
      headlines: ["Live updates of AI · feed temporarily unavailable."],
      updatedAt: new Date().toISOString()
    });
  }
});

// Create article (admin only)
app.post("/api/articles", auth, async (req, res) => {
  const article = new Article(req.body);
  await article.save();

  io.emit("newArticle", article); // Real-time update

  res.json(article);
});

// Delete article
app.delete("/api/articles/:id", auth, async (req, res) => {
  await Article.findByIdAndDelete(req.params.id);
  res.json({ msg: "Deleted" });
});

/* ======================
   REAL-TIME SOCKET
====================== */

io.on("connection", socket => {
  console.log("Client connected:", socket.id);
});

/* ======================
   AUTO FETCH AI NEWS
====================== */

function isMongoReady() {
  return mongoose.connection.readyState === 1;
}

function cleanSummary(value) {
  if (!value) return "";
  return String(value).replace(/\s+/g, " ").trim().slice(0, 260);
}

async function fetchAndPublishAINews() {
  if (!isMongoReady()) {
    console.log("Skipping AI news fetch: MongoDB is not connected yet");
    return;
  }

  console.log("Fetching hourly AI news posts...");

  try {
    const response = await axios.get(
      "https://hn.algolia.com/api/v1/search?query=artificial%20intelligence%20OR%20LLM%20OR%20GPT&tags=story",
      { timeout: 20000 }
    );

    const hits = Array.isArray(response.data?.hits) ? response.data.hits : [];
    const topHits = hits.slice(0, 10);
    let insertedCount = 0;

    for (const item of topHits) {
      const title = String(item.title || item.story_title || "").trim();
      if (!title) continue;

      const publishedAt = item.created_at_i
        ? new Date(item.created_at_i * 1000)
        : new Date();

      const articleDoc = {
        slug: slugify(title),
        title,
        content: item.url || item.story_url || "",
        summary: cleanSummary(item.story_text || item.comment_text || item._highlightResult?.title?.value || "AI news update"),
        tags: ["AI", "Auto"],
        source: "Hacker News",
        isExternal: true,
        publishedAt
      };

      const result = await Article.updateOne(
        { title: articleDoc.title, source: articleDoc.source },
        { $setOnInsert: articleDoc },
        { upsert: true }
      );

      if (result.upsertedCount > 0) {
        insertedCount += 1;
        const newArticle = await Article.findOne({ title: articleDoc.title, source: articleDoc.source });
        if (newArticle) io.emit("newArticle", newArticle);
      }
    }

    console.log(`AI hourly job completed: ${insertedCount} new post(s)`);
  } catch (err) {
    console.error("Auto fetch failed", err?.message || err);
  }
}

async function seedOriginalArticles() {
  if (!isMongoReady()) return;

  const originals = buildOriginalArticles();
  for (const article of originals) {
    await Article.updateOne(
      { slug: article.slug },
      { $setOnInsert: article },
      { upsert: true }
    );
  }
}

cron.schedule("0 * * * *", async () => {
  await fetchAndPublishAINews();
});

setTimeout(() => {
  seedOriginalArticles();
  fetchAndPublishAINews();
}, 8000);

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

server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
