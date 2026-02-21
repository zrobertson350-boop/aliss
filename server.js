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
app.use(helmet());

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
  title: { type: String, required: true },
  content: String,
  summary: String,
  tags: [String],
  source: String,
  isExternal: { type: Boolean, default: false },
  publishedAt: { type: Date, default: Date.now }
});
ArticleSchema.index({ title: 1, source: 1 }, { unique: true });

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

  const claudeKey = process.env.CLAUDE_API_KEY;
  if (!claudeKey) {
    return res.json({
      reply: "Alice is ready. Add CLAUDE_API_KEY in .env to enable live Claude responses."
    });
  }

  try {
    const response = await axios.post(
      "https://api.anthropic.com/v1/messages",
      {
        model: "claude-3-5-sonnet-latest",
        max_tokens: 500,
        system: "You are Alice, an AI assistant for the Aliss AI news website. Be concise and helpful.",
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
  const articles = await Article.find().sort({ publishedAt: -1 });
  res.json(articles);
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

cron.schedule("0 * * * *", async () => {
  await fetchAndPublishAINews();
});

setTimeout(() => {
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
