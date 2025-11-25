"use client";

import React from "react";
import Button from "../components/ui/button";
import Input from "../components/ui/input";

type Phase = "input" | "processing" | "dashboard";
type Tab = "Files" | "Dependencies" | "Complexity" | "Chat";

type ChatMsg = {
  id: string;
  role: "user" | "system";
  content: string;
};

export default function Page() {
  const [phase, setPhase] = React.useState<Phase>("input");
  const [repoUrl, setRepoUrl] = React.useState("");
  const [repoPath, setRepoPath] = React.useState("");
  const [progress, setProgress] = React.useState(0);
  const [logs, setLogs] = React.useState<string[]>([]);
  const [tab, setTab] = React.useState<Tab>("Chat");
  const [chat, setChat] = React.useState<ChatMsg[]>([
    {
      id: cryptoId(),
      role: "system",
      content:
        "Welcome. Drop a GitHub repo URL to analyze. I’ll index files, dependencies, and complexity, then answer questions in real time.",
    },
  ]);
  const [inputMsg, setInputMsg] = React.useState("");
    // Choose and sanitize API URL:
    // - In production, require NEXT_PUBLIC_API_URL to be set at build time.
    // - During local development (when browsing from localhost/127.0.0.1),
    //   fall back to http://localhost:8000.
    function sanitizeApiUrl(raw: string | undefined): { url?: string; invalid: boolean; reason?: string } {
        let val = (raw || "").trim();
        // Treat literal 'undefined'/'null' as missing (can happen if build-time substitution fails)
        if (!val || val.toLowerCase() === "undefined" || val.toLowerCase() === "null") {
            return {invalid: true, reason: "missing"};
        }
        // If it contains '/undefined' anywhere, consider it invalid (miswired env)
        if (/\bundefined\b/.test(val)) {
            return {invalid: true, reason: "contains 'undefined'"};
        }
        // Remove trailing slash
        val = val.replace(/\/$/, "");
        // Basic scheme check
        if (!/^https?:\/\//i.test(val)) {
            return {invalid: true, reason: "must start with http(s)://"};
        }
        return {url: val, invalid: false};
    }

    let resolved: { url?: string; invalid: boolean; reason?: string } = {url: undefined, invalid: true};
    {
        // First try build-time env
        resolved = sanitizeApiUrl(process.env.NEXT_PUBLIC_API_URL as string | undefined);
        // If not valid and we're on localhost, fall back to local API for convenience
        if (resolved.invalid && typeof window !== "undefined") {
        const host = window.location.hostname;
        const isLocal = host === "localhost" || host === "127.0.0.1";
        if (isLocal) {
            resolved = sanitizeApiUrl("http://localhost:8000");
        }
        }
    }
    const API_URL = resolved.url;
    const apiMisconfigured = (!API_URL) && (typeof window !== "undefined") && !["localhost", "127.0.0.1"].includes(window.location.hostname);
    if (!API_URL) {
        // Surface a clear console warning to help diagnose misconfiguration in deployed environments
        // without breaking the UI instantly. API calls will fail with a descriptive error below.
        console.warn(
            "NEXT_PUBLIC_API_URL is not set or invalid. In production, configure it to the deployed backend URL.",
            resolved.reason ? `(reason: ${resolved.reason})` : ""
        );
    }

    function resetToHome() {
        setPhase("input");
        setRepoUrl("");
        setRepoPath("");
        setProgress(0);
        setLogs([]);
        setTab("Chat");
        setChat([
            {
                id: cryptoId(),
                role: "system",
                content:
                    "Welcome. Drop a GitHub repo URL to analyze. I'll index files, dependencies, and complexity, then answer questions in real time.",
            },
        ]);
        setInputMsg("");
    }

  React.useEffect(() => {
    if (phase !== "processing") return;
    setLogs([]);
    setProgress(0);
    const steps = [
      "Resolving repository...",
      "Downloading archive...",
      "Unpacking...",
      "Indexing files...",
      "Extracting dependencies...",
      "Computing complexity metrics...",
      "Finalizing index...",
    ];

    let i = 0;
    const interval = setInterval(() => {
      setLogs((prev) => [...prev, `> ${steps[i % steps.length]}`].slice(-8));
      setProgress((p) => {
        const n = Math.min(p + Math.ceil(Math.random() * 18), 99); // leave room for server completion
        if (n >= 99) {
          clearInterval(interval);
        }
        return n;
      });
      i++;
    }, 600);

    return () => clearInterval(interval);
  }, [phase]);

  async function onAnalyze(e?: React.FormEvent) {
    e?.preventDefault();
    if (!repoUrl.trim()) return;
    const normalized = normalizeRepoPath(repoUrl.trim());
    if (!normalized) {
      setChat((c) => [
        ...c,
        { id: cryptoId(), role: "system", content: "Please provide a valid GitHub URL or 'owner/repo' path." },
      ]);
      return;
    }
    setRepoPath(normalized);
    setPhase("processing");
    try {
        if (!API_URL) {
            throw new Error(
                "Frontend is not configured with NEXT_PUBLIC_API_URL. Set it to your deployed backend URL and redeploy."
            );
        }
      const res = await fetch(`${API_URL}/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_path: normalized, use_langchain: true }),
      });
      const data = await safeJson(res);
      if (!res.ok) {
        const msg = data?.detail || data?.message || `Failed to index repository (${res.status})`;
        throw new Error(msg);
      }
      // Indexing complete
      setProgress(100);
      setLogs([]);
      setPhase("dashboard");
      setChat((c) => [
        ...c,
        {
          id: cryptoId(),
          role: "system",
          content: `Indexed ${data?.repo_path || normalized}. You can now ask questions in Chat.`,
        },
      ]);
    } catch (err: any) {
      setPhase("input");
      setProgress(0);
      setLogs([]);
      setChat((c) => [
        ...c,
        { id: cryptoId(), role: "system", content: `Indexing error: ${err?.message || String(err)}` },
      ]);
    }
  }

  async function sendMessage() {
    const content = inputMsg.trim();
    if (!content) return;
    const userMsg: ChatMsg = { id: cryptoId(), role: "user", content };
    setChat((c) => [...c, userMsg]);
    setInputMsg("");
    if (!repoPath) {
      setChat((c) => [
        ...c,
        { id: cryptoId(), role: "system", content: "Please index a repository first." },
      ]);
      return;
    }

    try {
        if (!API_URL) {
            throw new Error(
                "Frontend is not configured with NEXT_PUBLIC_API_URL. Set it to your deployed backend URL and redeploy."
            );
        }
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_path: repoPath, question: content, use_both_collections: true }),
      });
      const data = await safeJson(res);
      if (!res.ok) {
        const msg = data?.detail || data?.message || `Query failed (${res.status})`;
        throw new Error(msg);
      }
      const answer: string = data?.answer || "(no answer)";
      const sourcesText = summarizeSources(data);
      const sysMsg: ChatMsg = {
        id: cryptoId(),
        role: "system",
        content: [answer, sourcesText].filter(Boolean).join("\n\n"),
      };
      setChat((c) => [...c, sysMsg]);
    } catch (err: any) {
      setChat((c) => [
        ...c,
        { id: cryptoId(), role: "system", content: `Error: ${err?.message || String(err)}` },
      ]);
    }
  }

  const suggested: string[] = [
    "Show dependency graph",
    "Top 10 most complex files",
    "List devDependencies",
    "Explain project structure",
  ];

  return (
    <div className="min-h-dvh bg-neutral-50 text-neutral-900">
        {apiMisconfigured && (
            <div className="w-full bg-red-600 text-white text-sm px-4 py-2 text-center">
                NEXT_PUBLIC_API_URL is not set. This deployed frontend needs to know your API base URL. Configure it and
                redeploy.
            </div>
        )}
      <header className="sticky top-0 z-10 bg-white border-b-2 border-black">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
          <div className="inline-flex items-center gap-3">
              <button
                  onClick={resetToHome}
                  className="border-2 border-black bg-amber-200 px-2 py-1 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] font-extrabold tracking-tight hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none transition-transform cursor-pointer"
              >
              RRA
              </button>
            <span className="text-sm md:text-base font-semibold">Repository Analysis Engine</span>
          </div>
          <div className="hidden md:block text-xs font-mono opacity-70">fast • bold • friendly</div>
        </div>
      </header>

      {phase === "input" && (
        <main className="mx-auto max-w-3xl px-4 min-h-[calc(100dvh-64px)] flex items-center">
          <div className="w-full text-center">
            <h1 className="text-3xl md:text-4xl font-extrabold mb-4">Analyze a GitHub Repository</h1>
            <p className="text-neutral-700 mb-8 max-w-2xl mx-auto">
                Paste a GitHub repository URL. I’ll download, index, and make it conversational.
            </p>

            <form onSubmit={onAnalyze} className="grid grid-cols-1 gap-3 max-w-2xl mx-auto">
              <label className="text-sm font-semibold text-left" htmlFor="repo">
                GitHub Repository URL
              </label>
              <div className="flex flex-col sm:flex-row gap-3">
                <Input
                  id="repo"
                  placeholder="https://github.com/owner/repo"
                  value={repoUrl}
                  onChange={(e) => setRepoUrl(e.target.value)}
                  className="flex-1 font-mono"
                  fullWidth
                />
                <Button type="submit" className="whitespace-nowrap">
                  Analyze
                </Button>
              </div>
              <p className="text-xs font-mono opacity-70 text-left">
                Tip: Private repos aren’t fetched in this demo. We’ll simulate the pipeline.
              </p>
            </form>
          </div>
        </main>
      )}

      {phase === "processing" && (
        <main className="mx-auto max-w-4xl px-4 pt-16 pb-24">
          <h2 className="text-2xl md:text-3xl font-extrabold mb-6">Downloading & Indexing</h2>
          <div className="border-2 border-black bg-white p-4 md:p-6 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none transition-transform">
            <div className="mb-4">
              <div className="w-full border-2 border-black bg-neutral-100 h-6 relative overflow-hidden">
                <div
                  className="h-full bg-amber-200 border-r-2 border-black transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="mt-2 text-xs font-mono">{progress}%</div>
            </div>

            <div className="border-2 border-black bg-neutral-50 p-3 font-mono text-sm h-48 overflow-hidden">
              <div className="animate-[marquee_18s_linear_infinite] whitespace-nowrap">
                <span className="mr-6 opacity-70">{repoUrl || "(repo)"}</span>
                {Array.from({ length: 12 }, (_, idx) => (
                  <span key={idx} className="mr-6">
                    {logs[idx % logs.length] ?? "> waiting..."}
                  </span>
                ))}
              </div>
            </div>
            <style jsx>{`
              @keyframes marquee {
                0% { transform: translateX(0); }
                100% { transform: translateX(-50%); }
              }
            `}</style>
          </div>
        </main>
      )}

      {phase === "dashboard" && (
        <main className="mx-auto max-w-6xl px-4 py-6 grid grid-cols-1 md:grid-cols-[220px_1fr] gap-6">
          <aside className="border-2 border-black bg-white p-3 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none transition-transform">
            <div className="font-extrabold mb-3">Navigation</div>
            <nav className="grid gap-2">
              {(["Files", "Dependencies", "Complexity", "Chat"] as Tab[]).map(
                (t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={[
                      "text-left px-3 py-2 border-2 border-black",
                      "transition-transform hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none",
                      tab === t
                        ? "bg-amber-200 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]"
                        : "bg-white",
                    ].join(" ")}
                  >
                    <span className="font-semibold">{t}</span>
                  </button>
                )
              )}
            </nav>
          </aside>

          <section className="min-h-[60vh] border-2 border-black bg-white p-4 md:p-6 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
            {tab !== "Chat" && (
              <div className="text-sm opacity-70">
                {tab} panel placeholder — switch to Chat for Q&A.
              </div>
            )}

            {tab === "Chat" && (
              <div className="grid grid-rows-[1fr_auto] h-full gap-4">
                <div className="space-y-4 overflow-y-auto pr-1">
                  {chat.map((m, idx) => (
                    <MessageBubble key={m.id} msg={m} />
                  ))}

                  {chat[chat.length - 1]?.role === "system" && (
                    <div className="flex flex-wrap gap-2 pt-1">
                      {suggested.map((s) => (
                        <button
                          key={s}
                          onClick={() => {
                            setInputMsg(s);
                            setTimeout(sendMessage, 0);
                          }}
                          className="text-xs px-2 py-1 border-2 border-black bg-amber-100 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-[1fr_auto] gap-2">
                  <Input
                    placeholder="Ask about files, deps, complexity…"
                    value={inputMsg}
                    onChange={(e) => setInputMsg(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    className="font-mono"
                  />
                  <Button onClick={sendMessage}>Send</Button>
                </div>
              </div>
            )}
          </section>
        </main>
      )}
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMsg }) {
  if (msg.role === "user") {
    return (
      <div className="ml-auto max-w-[85%] bg-white border-2 border-black px-3 py-2 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
        <div className="text-sm">{msg.content}</div>
      </div>
    );
  }
  return (
    <div className="max-w-[92%]">
      <div className="border-2 border-black bg-neutral-50 px-3 py-2 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
        <pre className="whitespace-pre-wrap font-mono text-sm">{msg.content}</pre>
      </div>
    </div>
  );
}

function cryptoId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return Math.random().toString(36).slice(2);
}

function normalizeRepoPath(input: string): string {
  try {
    // Accept owner/repo directly
    const simpleMatch = input.match(/^[\w.-]+\/[\w.-]+$/);
    if (simpleMatch) return input;
    // Parse GitHub URLs
    const url = new URL(input);
    if (!/github\.com$/.test(url.hostname)) return "";
    const parts = url.pathname.replace(/^\//, "").split("/");
    if (parts.length >= 2) {
      return `${parts[0]}/${parts[1].replace(/\.git$/, "")}`;
    }
    return "";
  } catch {
    return "";
  }
}

async function safeJson(res: Response) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

function summarizeSources(data: any): string {
  let sources: any[] = [];
  if (Array.isArray(data?.all_sources)) sources = data.all_sources;
  else if (Array.isArray(data?.sources)) sources = data.sources;
  else sources = [...(data?.textual_sources || []), ...(data?.code_sources || [])];
  if (!sources.length) return "";
  const lines = sources.slice(0, 5).map((s, idx) => {
    const name = s.file_name || s?.metadata?.file_name || s?.path || s?.metadata?.path || `source_${idx+1}`;
    const score = typeof s.score === "number" ? s.score.toFixed(3) : s?.metadata?.score || "";
    return `- ${name}${score ? ` (score ${score})` : ""}`;
  });
  return `Sources:\n${lines.join("\n")}`;
}
