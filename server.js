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

cron.schedule("0 * * * *", async () => {
  console.log("Fetching AI news...");

  try {
    const res = await axios.get(
      "https://hn.algolia.com/api/v1/search?query=artificial%20intelligence"
    );

    for (let item of res.data.hits.slice(0, 5)) {
      const exists = await Article.findOne({ title: item.title });
      if (!exists && item.title) {
        const article = await Article.create({
          title: item.title,
          content: item.url,
          source: "Hacker News",
          isExternal: true
        });

        io.emit("newArticle", article);
      }
    }

  } catch (err) {
    console.error("Auto fetch failed");
  }
});

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
