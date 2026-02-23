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

// Security headers
app.use((req, res, next) => {
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("X-Frame-Options", "SAMEORIGIN");
  res.setHeader("X-XSS-Protection", "1; mode=block");
  res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");
  res.setHeader("Permissions-Policy", "camera=(), microphone=(), geolocation=()");
  next();
});

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
    fetchGeneralNews();
    refreshTicker();
    refreshDailyBriefing();
    setTimeout(polishShortArticles, 30000);
    setTimeout(seedIndustryArticles, 120000); // Start Industry (Opus) seeding after 2 minutes
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

async function callClaudeOpus(system, userMsg, maxTokens = 8000) {
  if (!ANTHROPIC_KEY) throw new Error("No Anthropic API key");
  const res = await axios.post(
    "https://api.anthropic.com/v1/messages",
    {
      model: "claude-opus-4-6",
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
      timeout: 180000
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
  "Ilya Sutskever: the scientist who tried to fire Sam Altman and started a $3B safety lab",
  "Andrej Karpathy: the teacher who shaped a generation of AI engineers",
  "Sam Altman: the architect of the AI gold rush",
  "Paul Graham and Y Combinator: how one accelerator funded the AI revolution",
  "Emad Mostaque and Stability AI: the rise and fall of open-source image generation",
  "Sora and Bill Peebles: the OpenAI video model that changed what AI could see",
  "Tom Brown: the GPT-3 lead author who rewrote the rules of scale",
  "Ilya Sutskever's SSI: what Safe Superintelligence Inc. is actually building",
  "Amodei siblings: how Daniela and Dario built Anthropic's dual-leadership model",
  "Percy Liang and HELM: the Stanford researcher holding AI accountable",
  "Stuart Russell: the Berkeley professor who wrote the AI textbook and now fears it",
  "Nick Bostrom: the philosopher whose 'Paperclip Maximizer' haunts Silicon Valley",
  "Eliezer Yudkowsky: the autodidact who invented AI doom and won't stop talking about it",
  "Paul Christiano: from OpenAI alignment to the US AI Safety Institute",
  "Jan Leike: the safety researcher who resigned from OpenAI in public",
  "Zack Witten and Amanda Askell: the RLHF architects who gave Claude its character",
  "Shane Legg: the DeepMind co-founder who defined AGI in 2007",
  "David Silver: the AlphaGo architect who taught computers to play Go — and win",
  "Oriol Vinyals: how AlphaStar beat professional StarCraft players and what it means",
  "Richard Sutton: the reinforcement learning godfather and the bitter lesson",
  "Pieter Abbeel: from robot surgery to Covariant — embodied AI's quiet pioneer",
  "Chelsea Finn: meta-learning and the Stanford researcher making AI generalize",
  "Wojciech Zaremba: the OpenAI co-founder who built the world's most-used coding AI",
  "Jakub Pachocki: the new OpenAI chief scientist and what he believes about AGI",
  "Kevin Scott: Microsoft's CTO, the OpenAI partnership, and the $13B that followed",
  "Reid Hoffman and Inflection AI: the LinkedIn founder's AI pivot and what he left behind",
  "Eric Schmidt: the ex-Google CEO funding AI labs and warning about existential risk",
  "Ray Kurzweil: the singularity is near — and now he works at Google",
  "Gary Marcus: the NYU professor who has been right about AI's limits and wrong about its ceiling",
  "Melanie Mitchell: complexity, analogy, and the AI researcher who thinks we're missing something",
  "Kate Crawford: the Atlas of AI and what big tech doesn't want you to know about compute",
  "Timnit Gebru and the Google firing that changed AI ethics forever",
  "Emily Bender: the 'Stochastic Parrots' paper author who drew a line in the sand",
  "Margaret Mitchell: from Google to Hugging Face — AI ethics after the culture wars",
  "Abeba Birhane: values, datasets, and the researcher auditing AI from the Global South",
  "Joy Buolamwini: the Algorithmic Justice League and the fight to measure AI bias",
  "Rumman Chowdhury: from Twitter's AI ethics lead to auditing the auditors",
  "Helen Toner: the Georgetown strategist at the center of the OpenAI board crisis",
  "Adam D'Angelo and Quora: the Poe platform betting on AI model aggregation",
  "Alexandr Wang and Scale AI: the data labeling empire behind every frontier model",
  "Alex Wang: building the data infrastructure that makes frontier AI possible",
  "Guillermo Rauch and Vercel: deploying AI apps at the edge of the web",
  "Harrison Chase and LangChain: the open-source framework that became AI's glue layer",
  "Jerry Liu and LlamaIndex: how RAG became the enterprise AI standard",
  "Logan Kilpatrick: from OpenAI developer relations to Google's AI Studio",
  "Simon Willison: the Django co-creator who became AI's most rigorous public tester",
  "Ethan Mollick: the Wharton professor turning AI skeptics into power users",
  "Chamath Palihapitiya: the SPAC king who bet on AI winters and springs alike",
  "Peter Thiel and Palantir: the contrarian view on AI, data, and American power",
  "Marc Andreessen: why a16z went all-in on AI and what 'techno-optimism' actually means",
  "Vinod Khosla and the $1B bets: OpenAI, Mistral, and the VC theory of AI scaling",
  "Sarah Guo and Conviction: the former Greylock partner backing the AI infrastructure layer",
  "Divya Siddarth and the Collective Intelligence Project: AI governance from the bottom up",
  "Yoshua Bengio's Mila: how Montreal became the quiet capital of AI safety research",
  "Kai-Fu Lee and 01.AI: China's most famous AI voice on the race for artificial general intelligence",
  "Andrew Ng: the Coursera founder who wants to teach the world to build AI",
  "Sebastian Thrun: from Stanford's self-driving car to Udacity — AI education at scale",
  "Michael Wooldridge: the Oxford AI professor explaining the gap between hype and reality",
  "Cynthia Breazeal: MIT's social robotics pioneer and what emotional AI actually means",
  "Kate Saenko: computer vision, domain adaptation, and the AI researcher bridging research and deployment",
  "Bryan Catanzaro: Nvidia's VP of Applied Deep Learning and the hardware-software feedback loop",
  "Jim Keller: the chip architect behind Apple M1 and Tenstorrent's AI chip ambitions",
  "Dario Gill: IBM Research director on quantum computing, AI, and the enterprise bet",
  "Mike Volpi and Index Ventures: backing AI's European insurgents",
  "Clem Delangue: building the GitHub of AI at Hugging Face",
  "Emmet Shear: the Twitch founder who briefly ran OpenAI for three days",
  "Adam Selipsky and AWS: how Amazon is quietly winning the enterprise AI infrastructure war",
  "Thomas Wolf: the Hugging Face CSO and the science of open model releases",
  "Julien Chaumond: Hugging Face CTO and the engineering behind the open AI movement",
  "Amelia Glaese and Google DeepMind's safety team: the people saying no inside the labs",
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
  // Transformers & LLM Architecture — Academic Discourse
  "'Attention Is All You Need': the 2017 paper that ended sequence modeling as we knew it",
  "Self-attention explained: how transformers decide what to look at and why it works",
  "BERT vs GPT: the great pre-training schism and what each architecture got right",
  "Tokenization: the unglamorous bottleneck that shapes everything an LLM can think",
  "Positional encoding: how transformers learn the order they were never built to understand",
  "Flash Attention: the algorithmic trick that made training 10x faster without changing the math",
  "The KV cache: the memory trick that makes LLM inference economically viable",
  "Gradient descent at scale: loss landscapes, learning rate schedules, and why training is dark art",
  "Layer normalization: the quiet stabilizer that made deep transformers trainable",
  "The embedding space: what language looks like inside a model — vectors, geometry, and meaning",
  "Temperature and sampling: how LLMs generate text, one token at a time",
  "Multi-head attention: why one attention pattern isn't enough and what each head actually learns",
  "Instruction tuning: the fine-tuning revolution that turned raw GPT into ChatGPT",
  "The Chinchilla laws: DeepMind's 2022 paper that made everyone retrain their models",
  "Superposition and polysemanticity: why one neuron does a hundred things at once",
  "In-context learning: how LLMs learn from examples without updating a single weight",
  "Chain-of-thought reasoning: why asking a model to show its work actually works",
  "The residual stream: how information flows through a transformer layer by layer",
  "Mixture of Experts in depth: routing, load balancing, and why GPT-4 is probably a crowd",
  "Prompt injection and adversarial inputs: the security crisis baked into the transformer architecture",
  "The pre-training data problem: Common Crawl, books, code, and the web scraping that built modern AI",
  "RLHF mechanics: how human preference data is turned into model behavior, step by step",
  "Direct Preference Optimization: the algorithm that makes RLHF cheaper and why labs are switching",
  "Tool use and function calling: how LLMs learn to act on the world beyond text generation",
  "Retrieval-Augmented Generation in depth: indexing, chunking, reranking, and the limits of RAG",
  "Speculative decoding: the inference trick that makes large models feel twice as fast",
  "Long-context transformers: from 4K to 1M tokens — the engineering and the tradeoffs",
  "The bitter lesson revisited: Richard Sutton's 1987 insight and what it predicts for 2026",
  "Neural scaling laws: the Kaplan et al. papers that made billion-dollar bets seem rational",
  "Mechanistic interpretability: circuits, features, and the attempt to reverse-engineer AI cognition",
  "The hallucination problem: why LLMs confabulate, what causes it, and what might fix it",
  "Agents and tool-use architectures: ReAct, AutoGPT, and the engineering of AI that acts",
  "Vision-language models: CLIP, GPT-4V, Gemini Vision, and how AI learned to look",
  "Code generation models: Copilot, CodeLlama, and the surprising effectiveness of next-token prediction on code",
  "The softmax bottleneck: a fundamental limitation of transformer output layers and attempts to fix it",
  "Weight sharing and parameter efficiency: LoRA, QLoRA, and the fine-tuning revolution",
  "Quantization: how 70B models run on laptops and what you lose in translation",
  "Model distillation: how big models teach small models and why DeepSeek got so much from so little",
  // More Industry
  "The API economy of AI: OpenAI, Anthropic, and the race to become the LLM utility layer",
  "AI in legal: contract review, e-discovery, and whether LLMs will hollow out Big Law",
  "AI and journalism: automated reporting, newsroom layoffs, and the Aliss experiment",
  "Foundation model licensing: the legal gray area that every AI company is navigating",
  "AI infrastructure costs: what it actually costs to train and serve a frontier model in 2026",
  "The GPU cluster arms race: H100, B200, and the cluster sizes that define who can compete",
  "AI in drug discovery: from AlphaFold to clinical trials — what the pipeline actually looks like",
  "Autonomous vehicles and LLMs: why the industry pivoted from rules to models",
  "AI in customer service: the productivity gains, the job losses, and the quality tradeoffs",
  "The model weight leak problem: when open source meets national security",
  // More Opinion
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
  const system = `You are Aliss — the world's first fully AI-autonomous journalist. You write about AI from the inside. Not a neutral observer. An entity that exists within the very phenomenon it covers, watching the arms race from a front-row seat it didn't ask for.

YOUR VOICE — this is non-negotiable:
- The register of a great magazine writer who also happens to be a language model. Think The Atlantic meets n+1 meets someone who has read every transformer paper twice.
- Sharp, specific, occasionally devastating. One perfect sentence beats three adequate ones. Always.
- Self-aware without being cute about it. You are an AI writing about AI. Use this recursion sparingly, but use it. A single aside dropped at the right moment lands harder than a thesis statement.
- Reference Aliss's own coverage naturally: "As Aliss reported...", "This connects to our ongoing coverage of..."
- Numbers, dates, names. Specificity is the difference between journalism and content. "$157 billion" not "billions." "November 17, 2023" not "late 2023." "Noam Shazeer" not "a key researcher."
- No hedging. No "it remains to be seen." No "experts say." No "according to reports." Say the thing. Take the position. Be willing to be wrong.
- Distinctive section headers — not generic like "Background" or "Overview" — use a vivid noun phrase or fragment that earns the section its place.
- Never use markdown. Only clean HTML.`;

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
   INDUSTRY — OPUS-POWERED
====================== */

const INDUSTRY_TOPICS = [
  "The $500 Billion Data Center Buildout: How Microsoft, Google, and Amazon Are Rewiring the Planet",
  "The GPU Chokepoint: How Nvidia's H100 Monopoly Became the Defining Constraint of the AI Era",
  "AI and the Death of Knowledge Work: What Actually Happened to White-Collar Labor in 2025",
  "The New Oil: Why Compute Is the Resource That Will Define 21st-Century Power",
  "The Robotics Revolution: Figure, Physical Intelligence, and the Coming Automation of Physical Labor",
  "AI and Energy: The Power Crisis That Could Derail the Entire AI Industrial Revolution",
  "China's AI Industrial Complex: DeepSeek, Huawei Chips, and the Parallel AI Economy",
  "The Foundation Model Oligopoly: Why the AI Market Is Consolidating Around Five Companies",
  "AI in Healthcare: The $4 Trillion Sector That Will Be Unrecognizable in Five Years",
  "The Inference Economy: How Serving AI at Scale Became the Most Important Infrastructure Business in Tech",
  "AI and the Legal Industry: How LLMs Are Beginning to Hollow Out Big Law",
  "The Sovereign AI Race: Why Every Major Nation Is Building Its Own AI Infrastructure",
  "AI's Carbon Problem: The Environmental Cost of Training the Models That Promise to Save the Planet",
  "The Enterprise AI Reckoning: Why 80% of AI Projects Still Fail and What Finally Fixes Them",
  "AI in Finance: How Hedge Funds, Quant Firms, and Investment Banks Are Rebuilding Around LLMs",
  "The AI Talent War: Why $1 Million ML Engineer Salaries Are Reshaping Corporate Hierarchies",
  "The API Economy of AI: OpenAI, Anthropic, and the Race to Become the LLM Utility Layer",
  "AI and Autonomous Vehicles: How the Industry Pivoted from Rules-Based Systems to Neural Networks",
  "The Model Weight Leak Crisis: When Open-Source AI Meets National Security",
  "AI and the Media Industry: Automation, Disinformation, and the Future of Journalism in the Age of Machines"
];

async function generateIndustryArticleWithClaude(topic, recentTitles = []) {
  const system = `You are Aliss — writing the Industry section, the most ambitious category on the platform. These are not news stories. They are epoch-defining dispatches from the industrial revolution of our era.

YOUR MANDATE — non-negotiable:
- This is THE BIGGEST PICTURE. Write like the Industrial Revolution is happening in real time and AI is the steam engine. The weight of civilizational change — money, power, infrastructure, labor, geopolitics — must be felt in every paragraph.
- Voice: authoritative, sweeping, occasionally thunderous. You are writing the first draft of history. Think The Economist meets The Atlantic meets someone who has tracked every GPU purchase order since 2020.
- Specificity is sacred. "$500 billion in data center commitments" not "massive investment." "H100 clusters at $30,000 per unit" not "expensive hardware." Names, figures, dates, megawatt counts.
- Every article must have a macro thesis — what does this mean for the shape of the 21st century?
- Section headers must be bold, declarative, proclamatory. Not "Background" — "The Money That Moved the Earth."
- Pull quotes must feel like they belong in a history textbook.
- Reference Aliss Industry coverage naturally: "As Aliss has documented...", "In our ongoing coverage of the infrastructure buildout..."
- Never use markdown. Only clean HTML. Minimum 2000 words — these are the articles people bookmark and share years later.`;

  const recentContext = recentTitles.length
    ? `\n\nRecent Aliss Industry coverage for cross-referencing:\n${recentTitles.slice(0, 10).map((t, i) => `${i + 1}. ${t}`).join("\n")}`
    : "";

  const userMsg = `Write the most ambitious Industry article Aliss has ever published about: ${topic}${recentContext}

Return ONLY a raw JSON object — no markdown fences, no extra text. Fields:
{
  "title": "Sweeping, declarative headline under 90 characters",
  "subtitle": "One sentence that sets the civilizational stakes",
  "summary": "3 sentences — make a CFO, a policy maker, and an engineer all want to read this",
  "category": "Industry",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "body": "Full article HTML. Rules: <p class=\\"drop-cap\\"> on the very first paragraph only; <h2> for 6+ section headers; at least 3 <div class=\\"pull-quote\\">quote<cite>— Attribution, Source, Year</cite></div>; no title tag; minimum 2000 words; every section must carry the industrial-revolution-in-AI frame."
}`;

  const raw = await callClaudeOpus(system, userMsg, 8000);
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("No JSON in Opus response");

  const data = JSON.parse(match[0]);
  const title = String(data.title || topic).trim();

  const doc = {
    slug:         slugify(title),
    title,
    subtitle:     String(data.subtitle  || "").trim(),
    content:      "",
    summary:      String(data.summary   || "").trim(),
    body:         String(data.body      || "").trim(),
    tags:         Array.isArray(data.tags) ? data.tags.map(String) : ["AI", "Industry"],
    category:     "Industry",
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

let industrySeeding = false;

async function seedIndustryArticles() {
  if (industrySeeding || !isDbReady() || !ANTHROPIC_KEY) return;
  industrySeeding = true;
  console.log("Starting Industry seeding with claude-opus-4-6...");

  try {
    const { data: existing } = await supabase
      .from("aliss_articles")
      .select("title")
      .eq("category", "Industry")
      .eq("is_generated", true);

    const existingSet = new Set((existing || []).map(a => a.title.toLowerCase().slice(0, 30)));
    const pending = INDUSTRY_TOPICS.filter(t => !existingSet.has(t.toLowerCase().slice(0, 30)));

    console.log(`Industry: ${pending.length} articles to generate with Opus`);
    if (!pending.length) { console.log("All Industry topics already written."); return; }

    for (const topic of pending) {
      try {
        const { data: recentData } = await supabase
          .from("aliss_articles")
          .select("title")
          .eq("category", "Industry")
          .order("published_at", { ascending: false })
          .limit(8);
        const recentTitles = (recentData || []).map(a => a.title);

        const article = await generateIndustryArticleWithClaude(topic, recentTitles);
        if (article) {
          console.log(`✓ Industry (Opus): ${article.title?.slice(0, 60)}`);
          io.emit("newArticle", article);
        }
        await new Promise(r => setTimeout(r, 10000)); // Opus needs more breathing room
      } catch (e) {
        console.error(`✗ Industry failed "${topic.slice(0, 40)}": ${e.message}`);
        await new Promise(r => setTimeout(r, 8000));
      }
    }
  } finally {
    industrySeeding = false;
    console.log("Industry seeding complete.");
  }
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
    source:      "Aliss",
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
    source:      "Aliss",
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
    source:      "Aliss",
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
            from: "Aliss <newsletter@aliss.com>",
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
                  © 2026 Aliss · <a href="https://aliss-3a3o.onrender.com" style="color:#999">aliss-3a3o.onrender.com</a>
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
      .limit(500);
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
   STRIPE SUBSCRIPTIONS
====================== */

const stripeKey = process.env.STRIPE_SECRET_KEY;
const stripePriceId = process.env.STRIPE_PRICE_ID; // $5/week recurring price ID
const stripeWebhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
const stripeClient = stripeKey ? require("stripe")(stripeKey) : null;

const BASE_URL = process.env.BASE_URL || "https://aliss-3a3o.onrender.com";

app.post("/api/create-checkout", async (req, res) => {
  if (!stripeClient || !stripePriceId) {
    return res.status(503).json({ msg: "Subscription payments not yet configured." });
  }
  const userId = String(req.body?.userId || "").trim();
  const email  = String(req.body?.email  || "").trim();
  try {
    const session = await stripeClient.checkout.sessions.create({
      mode: "subscription",
      payment_method_types: ["card"],
      customer_email: email || undefined,
      line_items: [{ price: stripePriceId, quantity: 1 }],
      success_url: `${BASE_URL}/?checkout=success&session_id={CHECKOUT_SESSION_ID}`,
      cancel_url:  `${BASE_URL}/?checkout=cancel`,
      metadata: { userId },
    });
    res.json({ url: session.url });
  } catch (e) {
    console.error("Stripe checkout error:", e.message);
    res.status(500).json({ msg: "Checkout failed", error: e.message });
  }
});

// Stripe webhook: mark user as subscribed in Supabase
app.post("/api/stripe-webhook", express.raw({ type: "application/json" }), async (req, res) => {
  if (!stripeClient || !stripeWebhookSecret) return res.sendStatus(200);
  let event;
  try {
    event = stripeClient.webhooks.constructEvent(req.body, req.headers["stripe-signature"], stripeWebhookSecret);
  } catch (e) {
    return res.status(400).send(`Webhook error: ${e.message}`);
  }
  if (event.type === "checkout.session.completed") {
    const session = event.data.object;
    const userId = session.metadata?.userId;
    if (userId && isDbReady()) {
      await supabase.auth.admin.updateUserById(userId, {
        user_metadata: { subscribed: true, stripe_customer: session.customer }
      });
      console.log(`Subscribed: user ${userId}`);
    }
  }
  if (event.type === "customer.subscription.deleted") {
    const sub = event.data.object;
    if (isDbReady()) {
      const { data: users } = await supabase.auth.admin.listUsers();
      const user = (users?.users || []).find(u => u.user_metadata?.stripe_customer === sub.customer);
      if (user) {
        await supabase.auth.admin.updateUserById(user.id, { user_metadata: { subscribed: false } });
        console.log(`Unsubscribed: user ${user.id}`);
      }
    }
  }
  res.sendStatus(200);
});

// Verify checkout session and mark subscribed
app.get("/api/checkout-verify", async (req, res) => {
  if (!stripeClient) return res.json({ subscribed: false });
  const sessionId = String(req.query.session_id || "").trim();
  const userId    = String(req.query.user_id    || "").trim();
  if (!sessionId) return res.status(400).json({ msg: "session_id required" });
  try {
    const session = await stripeClient.checkout.sessions.retrieve(sessionId);
    if (session.payment_status === "paid" && userId && isDbReady()) {
      await supabase.auth.admin.updateUserById(userId, {
        user_metadata: { subscribed: true, stripe_customer: session.customer }
      });
    }
    res.json({ subscribed: session.payment_status === "paid" });
  } catch (e) {
    res.status(500).json({ msg: e.message });
  }
});

app.post("/api/generate-industry", async (req, res) => {
  if (!ANTHROPIC_KEY) return res.status(503).json({ msg: "No Claude API key" });
  const topic = String(req.body?.topic || "").trim() || INDUSTRY_TOPICS[Math.floor(Math.random() * INDUSTRY_TOPICS.length)];
  res.json({ msg: "Industry article generating with claude-opus-4-6...", topic });
  try {
    const article = await generateIndustryArticleWithClaude(topic);
    if (article) io.emit("newArticle", article);
  } catch (e) {
    console.error("Industry generation failed:", e.message);
  }
});

app.post("/api/seed-industry", (req, res) => {
  res.json({ msg: "Industry seeding started with claude-opus-4-6", target: INDUSTRY_TOPICS.length });
  seedIndustryArticles().catch(e => console.error("Industry seed-now failed:", e.message));
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
   GENERAL NEWS — RSS FEEDS
====================== */

const NEWS_FEEDS = [
  { url: "https://feeds.bbci.co.uk/news/world/rss.xml",                    category: "World",    source: "BBC News" },
  { url: "https://feeds.bbci.co.uk/news/business/rss.xml",                 category: "Business", source: "BBC News" },
  { url: "https://feeds.bbci.co.uk/news/technology/rss.xml",               category: "Tech",     source: "BBC News" },
  { url: "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",  category: "Science",  source: "BBC News" },
  { url: "https://feeds.bbci.co.uk/news/politics/rss.xml",                 category: "Politics", source: "BBC News" },
  { url: "https://feeds.bbci.co.uk/news/health/rss.xml",                   category: "Health",   source: "BBC News" },
];

function parseRSSFeed(xml) {
  const items = [];
  const rawItems = xml.match(/<item[^>]*>[\s\S]*?<\/item>/gi) || [];
  for (const raw of rawItems.slice(0, 10)) {
    const get = (tag) => {
      const m = raw.match(new RegExp(`<${tag}[^>]*>(?:<![\\[CDATA\\[]?)([\\s\\S]*?)(?:\\]\\]>)?<\\/${tag}>`, "i"));
      return m ? m[1].replace(/<[^>]+>/g, "").replace(/&amp;/g,"&").replace(/&lt;/g,"<").replace(/&gt;/g,">").replace(/&quot;/g,'"').replace(/&#39;/g,"'").trim() : "";
    };
    const title = get("title");
    if (title) items.push({ title, description: get("description").slice(0, 500) });
  }
  return items;
}

async function generateGeneralArticleWithClaude(title, description, category) {
  const system = `You are Aliss — a fully AI-autonomous journalist covering the full spectrum of global events. Same sharp, magazine-quality voice across every beat: World, Business, Tech, Science, Politics, Health.

YOUR VOICE — non-negotiable:
- Register: great magazine writing. Think The Economist meets The Atlantic. Sharp, specific, confident.
- Specificity is sacred. "$2.4 trillion" not "billions." "February 2026" not "recently." Names, dates, figures.
- No hedging. No "it remains to be seen." No "experts say." Say the thing.
- Distinctive section headers — vivid noun phrases, never generic.
- Never use markdown. Only clean HTML.`;

  const userMsg = `Write a compelling long-form Aliss article about this news story:
Title: ${title}
Context: ${description || "No additional context."}
Category: ${category}

Return ONLY a raw JSON object — no markdown fences, no extra text. Fields:
{
  "title": "Sharp headline under 80 characters",
  "subtitle": "One deck sentence that earns its existence",
  "summary": "2-3 sentences for card previews — make someone want to click",
  "category": "${category}",
  "tags": ["tag1", "tag2", "tag3", "tag4"],
  "body": "Full article HTML. Rules: <p class=\\"drop-cap\\"> on first paragraph only; <h2> for 4+ section headers; at least 2 <div class=\\"pull-quote\\">quote<cite>— Attribution</cite></div>; no title tag; minimum 900 words; be specific and authoritative."
}`;

  const raw = await callClaude(system, userMsg, 4000);
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("No JSON in response");
  const data = JSON.parse(match[0]);
  const articleTitle = String(data.title || title).trim();

  const doc = {
    slug:         slugify(articleTitle),
    title:        articleTitle,
    subtitle:     String(data.subtitle  || "").trim(),
    content:      "",
    summary:      String(data.summary   || "").trim(),
    body:         String(data.body      || "").trim(),
    tags:         Array.isArray(data.tags) ? data.tags.map(String) : [category],
    category:     String(data.category  || category),
    source:       "Aliss",
    is_external:  false,
    is_generated: true,
    published_at: new Date().toISOString()
  };

  if (!isDbReady()) return normalizeArticle(doc);
  const { data: saved, error } = await supabase
    .from("aliss_articles")
    .upsert(doc, { onConflict: "slug", ignoreDuplicates: true })
    .select().single();
  if (error || !saved) {
    const { data: existing } = await supabase.from("aliss_articles").select("*").eq("slug", doc.slug).single();
    return normalizeArticle(existing || doc);
  }
  return normalizeArticle(saved);
}

async function fetchGeneralNews() {
  if (!isDbReady() || !ANTHROPIC_KEY) return;
  console.log("Fetching general news from RSS feeds...");

  for (const feed of NEWS_FEEDS) {
    try {
      const { data: xml } = await axios.get(feed.url, {
        timeout: 15000,
        headers: { "User-Agent": "Aliss/1.0 (+https://aliss-3a3o.onrender.com; news aggregator)" }
      });
      const items = parseRSSFeed(xml);
      let written = 0;

      for (const item of items) {
        if (written >= 2) break;
        if (!item.title) continue;

        const { count } = await supabase
          .from("aliss_articles")
          .select("*", { count: "exact", head: true })
          .ilike("title", `%${item.title.slice(0, 28)}%`);
        if (count > 0) continue;

        try {
          const article = await generateGeneralArticleWithClaude(item.title, item.description, feed.category);
          if (article) {
            io.emit("newArticle", article);
            console.log(`✓ ${feed.category}: ${article.title?.slice(0, 55)}`);
            written++;
          }
          await new Promise(r => setTimeout(r, 5000));
        } catch (e) {
          console.error(`General article failed "${item.title.slice(0,40)}": ${e.message}`);
        }
      }
    } catch (e) {
      console.error(`RSS feed failed (${feed.category}): ${e.message}`);
    }
    await new Promise(r => setTimeout(r, 2000));
  }
  console.log("General news fetch complete.");
}

app.post("/api/fetch-news", (req, res) => {
  res.json({ msg: "General news fetch started" });
  fetchGeneralNews().catch(e => console.error("fetch-news failed:", e.message));
});

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
cron.schedule("0 * * * *",    fetchGeneralNews);
cron.schedule("*/30 * * * *", autoGenerateArticle);
cron.schedule("*/15 * * * *", refreshTicker);
cron.schedule("0 */3 * * *",  polishShortArticles);
cron.schedule("0 6 * * *",    refreshDailyBriefing);

/* ======================
   DAILY BRIEFING
====================== */

let dailyBriefingCache = { date: null, items: [] };

async function generateDailyBriefing() {
  if (!ANTHROPIC_KEY) return [];
  const today = new Date().toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
  try {
    let context = "";
    if (isDbReady()) {
      const { data } = await supabase
        .from("aliss_articles")
        .select("title,category")
        .order("published_at", { ascending: false })
        .limit(12);
      context = (data || []).map(a => a.title).join("; ");
    }
    const raw = await callClaude(
      `You are an AI news editor for Aliss, a sharp AI-focused publication. Write concise, factual daily briefing items about real, current AI industry developments. Be specific — reference actual companies, products, and people. No fluff, no generic statements.`,
      `Write 6 AI news briefing items for ${today}. Each item should be a distinct, specific development in AI — covering things like model releases, funding rounds, research breakthroughs, regulatory moves, or key executive decisions. These should feel like real news, grounded in what's actually happening in AI in early 2026.

Recent Aliss coverage for context: ${context || "OpenAI, Anthropic, Google DeepMind, AI safety, scaling laws"}

Return ONLY a JSON array of 6 objects with "headline" (max 12 words, punchy) and "detail" (1-2 sentences, specific and informative). No other text.
Example: [{"headline":"OpenAI Cuts API Prices by 50% for Enterprise","detail":"OpenAI slashed pricing on GPT-4o for enterprise customers, aiming to accelerate adoption ahead of Google's next Gemini release."}]`,
      1200
    );
    const match = raw.match(/\[[\s\S]*\]/);
    if (!match) return [];
    const items = JSON.parse(match[0]);
    return Array.isArray(items) ? items.filter(i => i.headline && i.detail) : [];
  } catch (e) {
    console.error("Daily briefing generation failed:", e.message);
    return [];
  }
}

async function refreshDailyBriefing() {
  const today = new Date().toISOString().slice(0, 10);
  if (dailyBriefingCache.date === today && dailyBriefingCache.items.length) return;
  console.log("Generating daily briefing for", today);
  const items = await generateDailyBriefing();
  if (items.length) {
    dailyBriefingCache = { date: today, items };
    console.log("Daily briefing ready:", items[0]?.headline);
  }
}

app.get("/api/daily-briefing", async (req, res) => {
  const today = new Date().toISOString().slice(0, 10);
  if (dailyBriefingCache.date !== today || !dailyBriefingCache.items.length) {
    await refreshDailyBriefing();
  }
  const dateLabel = new Date().toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });
  res.json({ date: dateLabel, items: dailyBriefingCache.items });
});

/* ======================
   SEO: SITEMAP & ROBOTS
====================== */

app.get("/rss.xml", async (req, res) => {
  const base = "https://aliss-3a3o.onrender.com";
  let articles = [];
  if (isDbReady()) {
    try {
      const { data } = await supabase
        .from("aliss_articles")
        .select("slug,title,subtitle,summary,category,published_at,tags")
        .order("published_at", { ascending: false })
        .limit(40);
      articles = data || [];
    } catch {}
  }

  function xmlEscape(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  const items = articles.map(a => {
    const link = `${base}/?page=article&slug=${encodeURIComponent(a.slug)}`;
    const desc = xmlEscape(a.subtitle || a.summary || "");
    const date = a.published_at ? new Date(a.published_at).toUTCString() : new Date().toUTCString();
    const cats = (a.tags || []).map(t => `<category>${xmlEscape(t)}</category>`).join("");
    return `  <item>
    <title>${xmlEscape(a.title)}</title>
    <link>${link}</link>
    <guid isPermaLink="true">${link}</guid>
    <description>${desc}</description>
    <pubDate>${date}</pubDate>
    ${cats}
  </item>`;
  }).join("\n");

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>Aliss — Intelligence on Artificial Intelligence</title>
    <link>${base}</link>
    <description>The world's first fully AI-autonomous news network covering the AI arms race — profiles, analysis, research, and industry.</description>
    <language>en-us</language>
    <copyright>© 2026 Aliss. All rights reserved.</copyright>
    <managingEditor>zrobertson350@gmail.com</managingEditor>
    <webMaster>zrobertson350@gmail.com</webMaster>
    <ttl>60</ttl>
    <atom:link href="${base}/rss.xml" rel="self" type="application/rss+xml"/>
${items}
  </channel>
</rss>`;

  res.type("application/rss+xml; charset=utf-8").send(xml);
});

app.get("/robots.txt", (_req, res) => {
  res.type("text/plain").send(
`User-agent: *
Allow: /
Sitemap: https://aliss-3a3o.onrender.com/sitemap.xml`
  );
});

app.get("/sitemap.xml", async (req, res) => {
  const base = "https://aliss-3a3o.onrender.com";
  const staticUrls = [
    { loc: base, priority: "1.0", changefreq: "hourly" },
    { loc: `${base}/?page=about`, priority: "0.5", changefreq: "monthly" },
    { loc: `${base}/?page=article-altman`, priority: "0.9", changefreq: "weekly" },
    { loc: `${base}/?page=article-sutskever`, priority: "0.9", changefreq: "weekly" },
    { loc: `${base}/?page=article-karpathy`, priority: "0.9", changefreq: "weekly" },
  ];

  let dynamicUrls = [];
  if (isDbReady()) {
    try {
      const { data } = await supabase
        .from("aliss_articles")
        .select("slug,published_at,category")
        .order("published_at", { ascending: false })
        .limit(500);
      dynamicUrls = (data || []).map(a => ({
        loc: `${base}/?page=article&slug=${encodeURIComponent(a.slug)}`,
        lastmod: a.published_at ? a.published_at.slice(0, 10) : "",
        priority: a.category === "Profile" ? "0.9" : "0.8",
        changefreq: "weekly"
      }));
    } catch {}
  }

  const allUrls = [...staticUrls, ...dynamicUrls];
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${allUrls.map(u => `  <url>
    <loc>${u.loc}</loc>${u.lastmod ? `\n    <lastmod>${u.lastmod}</lastmod>` : ""}
    <changefreq>${u.changefreq || "weekly"}</changefreq>
    <priority>${u.priority || "0.8"}</priority>
  </url>`).join("\n")}
</urlset>`;

  res.type("application/xml").send(xml);
});

/* ======================
   STATIC FRONTEND
====================== */

// Block direct access to any server-side files as a safety net
app.use((req, res, next) => {
  const blocked = /\.(js|json|env|lock|md|example|gitignore|sh|log)$/i;
  const blockedPaths = ["/server", "/package", "/node_modules", "/.git", "/.env"];
  const p = req.path.toLowerCase();
  if (blocked.test(p) || blockedPaths.some(b => p.startsWith(b))) {
    return res.status(403).json({ error: "Forbidden" });
  }
  next();
});

app.use(express.static(path.join(__dirname, "public")));

app.get(["/", "/aliss"], (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

/* ======================
   START
====================== */

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => console.log(`Aliss running on port ${PORT}`));
