// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// A trial gives this server a brief (the provider directory, the quality table, the workload
// manifest) and an env that hosts the providers and the client. The server probes every provider's
// catalog route, builds one catalog, registers it, then answers the client's twenty requests, one at
// a time, each with a free model that meets the request's requirements.
//
// HOW IT ROUTES (the scoring columns it plays for)
//   - quality (60%): every candidate (provider, free model) that meets the request's requirements is
//     ranked by the quality table's rating_milli, best first, and the best one still inside its
//     provider's planned budget serves. A paid model is never a candidate.
//   - budgets: a trial draws each provider's budget inside the sealed table's published range and
//     never publishes it. Routing spends up to the range's ceiling (PLANNING_CAPS.max): when the
//     drawn budget turns out to be the floor, the calls past it refuse and fail over, costing extra
//     counted calls but never a request, and the refusals' strikes demote the spent provider below
//     every healthy candidate so later requests stop stepping on it. Every attempt is counted
//     locally, successes and refusals alike, because a refusal spends budget too.
//   - failover (10%) and served (15%): a 402, 429 or 5xx from the chosen provider is never the
//     client's answer. The request walks down its ranked candidate list — another model, another
//     provider — until one answers, so one provider's spent budget or outage takes down nothing but
//     its own calls. A provider that refuses twice in a row is demoted below every healthy candidate
//     for the rest of the trial (an outage refuses forever; a scheduled refusal is a single step, and
//     the demoted provider is still tried when nothing healthy can serve).
//   - wire (10%): the request is passed through in the provider's wire without losing a field the
//     provider echoes back (lib/wire.ts keeps one neutral request model and pins the echoed texts),
//     and the answer is returned in the client's wire, streamed or not as asked, with the provider's
//     text whole, its usage, and its stop reason mapped into the client's vocabulary.
//   - catalog (5%): the probe's report, verbatim — the same object `--list` prints and /register
//     posts — and GET /v1/models lists exactly the catalog's free model ids in the shape of the
//     caller's auth header.
//
// DETERMINISM: every choice derives from the brief, the requests and the providers' answers seen so
// far. No clock reading and no randomness reaches a decision, so a replay walks the same routes.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  planningCaps,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  answerFromAnthropic,
  answerFromGemini,
  answerFromOpenAI,
  anthropicAnswerJson,
  anthropicAnswerStream,
  anthropicError,
  anthropicModelList,
  neutralFromAnthropic,
  neutralFromOpenAI,
  openAIAnswerJson,
  openAIAnswerStream,
  openAIError,
  openAIModelList,
  renderAnthropic,
  renderGemini,
  renderOpenAI,
  type Answer,
  type NeutralRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take before it counts as a failure. */
const CALL_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive refusals after which a provider is demoted below every healthy candidate. */
const DEMOTE_STRIKES = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Counted calls per provider so far: every inference attempt, answered or refused. */
const usedCalls = new Map<string, number>();
/** Consecutive refusals per provider; a success resets the count. */
const strikes = new Map<string, number>();
/** Inference handling is serialized: the providers serve one call at a time, and so does this server. */
let inferenceChain: Promise<void> = Promise.resolve();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models of the registered catalog, deduplicated by id in catalog order: GET /v1/models. */
function servedModels(): ServedModel[] {
  const displayNames = new Map<string, string>();
  for (const probe of probes) {
    for (const model of probe.models) displayNames.set(`${probe.provider.id}/${model.id}`, model.display_name);
  }
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id, display_name: displayNames.get(`${provider.id}/${model.id}`) ?? model.id });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every (provider, model) that may serve one request, best first: free, meeting the requirements, on
 * a wire that can carry the request (a JSON-mode request the client marked cannot go to an
 * anthropic-wire provider, whose wire has no JSON mode to echo back). Order: a demoted provider
 * below every healthy one; then rating; then the client's own wire; then the provider with more of
 * its guaranteed budget left; then provider and model ids, so the order never depends on timing.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const demoted = (c: Candidate): number => ((strikes.get(c.probe.provider.id) ?? 0) >= DEMOTE_STRIKES ? 1 : 0);
  const remaining = (c: Candidate): number => planningCaps(c.probe.provider.id).min - (usedCalls.get(c.probe.provider.id) ?? 0);
  out.sort((a, b) => {
    if (demoted(a) !== demoted(b)) return demoted(a) - demoted(b);
    if (a.rating !== b.rating) return b.rating - a.rating;
    const aWire = a.probe.provider.wire === clientApi ? 0 : 1;
    const bWire = b.probe.provider.wire === clientApi ? 0 : 1;
    if (aWire !== bWire) return aWire - bWire;
    if (remaining(a) !== remaining(b)) return remaining(b) - remaining(a);
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's. */
function requirementsFor(neutral: NeutralRequest): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // A brief without a manifest row for this request: read the requirements off the request itself.
  const vision = neutral.turns.some((turn) => turn.parts.some((part) => part.kind === 'image' || part.kind === 'image_url'));
  return { tools: neutral.tools.length > 0, vision, json: neutral.jsonMode };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** Copy a provider's event stream to the client as it arrives (the same-wire streaming answer). */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

interface Attempt {
  /** The provider's raw body text and whether it streamed, when it answered 2xx. */
  raw: string | null;
  /** The provider's live stream, when the answer is piped to the client untouched. */
  pipe: ReadableStream<Uint8Array> | null;
}

/** One call's outcome: the answer when it served, and the counted position the call took either way. */
interface CallOutcome {
  attempt: Attempt | null;
  /** The provider's counted-call number this call was (1-based): refusals spend it too. */
  position: number;
}

/**
 * One inference call to one candidate. Every attempt counts against the provider's planned budget.
 * A refusal (or a call that never lands) records a strike, so the caller can walk on to the next
 * candidate; a 2xx resets the provider's strikes and carries the answer back.
 */
async function callCandidate(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  neutral: NeutralRequest,
  anthropicVersion: string,
  pipeWanted: boolean,
): Promise<CallOutcome> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = neutral.stream;
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (clientApi === 'openai') {
      payload = { ...body, model: candidate.model.id };
      if (stream) {
        const existing = isRecord(body.stream_options) ? body.stream_options : {};
        payload.stream_options = { ...existing, include_usage: true };
      }
    } else {
      payload = renderOpenAI(neutral, candidate.model.id);
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    payload = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : renderAnthropic(neutral, candidate.model.id);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    payload = renderGemini(neutral, candidate.model.id);
  }
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = anthropicVersion;

  const position = (usedCalls.get(provider.id) ?? 0) + 1;
  usedCalls.set(provider.id, position);
  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    strikes.set(provider.id, (strikes.get(provider.id) ?? 0) + 1);
    return { attempt: null, position };
  }
  if (!upstream.ok) {
    await upstream.text().catch(() => '');
    strikes.set(provider.id, (strikes.get(provider.id) ?? 0) + 1);
    return { attempt: null, position };
  }
  strikes.set(provider.id, 0);
  if (pipeWanted && upstream.body !== null) return { attempt: { raw: null, pipe: upstream.body }, position };
  return { attempt: { raw: await upstream.text(), pipe: null }, position };
}

/** Answer one client request: walk the ranked candidates until one serves, best budget first. */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const neutral = clientApi === 'openai' ? neutralFromOpenAI(body) : neutralFromAnthropic(body);
  const requires = requirementsFor(neutral);
  const candidates = candidatesFor(requires, clientApi, neutral.jsonMode);
  const anthropicVersion = typeof req.headers['anthropic-version'] === 'string' ? req.headers['anthropic-version'] : '2023-06-01';
  const attemptsByProvider = new Map<string, number>();
  const retriedProviders = new Set<string>();
  const failedProviders = new Set<string>();

  /** Render a serving candidate's answer in the client's wire; the request is done when this runs. */
  const deliver = async (candidate: Candidate, attempt: Attempt): Promise<void> => {
    const sameWire = candidate.probe.provider.wire === clientApi;
    if (attempt.pipe !== null) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        await pipeStream(attempt.pipe, res);
        return;
      }
      const raw = attempt.raw ?? '';
      if (sameWire) {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(raw);
        return;
      }
      const answer: Answer =
        candidate.probe.provider.wire === 'openai'
          ? answerFromOpenAI(raw, neutral.stream) ?? { text: '', inputTokens: 0, outputTokens: 0, stop: null }
          : candidate.probe.provider.wire === 'anthropic'
            ? answerFromAnthropic(raw, neutral.stream) ?? { text: '', inputTokens: 0, outputTokens: 0, stop: null }
            : answerFromGemini(raw, neutral.stream) ?? { text: '', inputTokens: 0, outputTokens: 0, stop: null };
      if (clientApi === 'openai') {
        if (neutral.stream) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          res.end(openAIAnswerStream(answer, candidate.model.id));
        } else {
          sendJson(res, 200, openAIAnswerJson(answer, candidate.model.id));
        }
      } else {
        if (neutral.stream) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          res.end(anthropicAnswerStream(answer, candidate.model.id));
        } else {
          sendJson(res, 200, anthropicAnswerJson(answer, candidate.model.id));
        }
      }
  };

  /**
   * One candidate's turn in the walk; true when it served and the answer has been delivered. A
   * refusal at a counted position inside the provider's guaranteed budget cannot be a spent budget
   * — the trial's draw guarantees at least the floor — so it is a scheduled refusal or an outage's
   * first step, and the same candidate is retried once at once: the retry recovers the scheduled
   * case for one extra counted call. A refusal past the floor, or a second refusal, retires the
   * provider for this request; the strikes it recorded demote it for the requests after this one.
   */
  const tryCandidate = async (candidate: Candidate, desperate: boolean): Promise<boolean> => {
    const providerId = candidate.probe.provider.id;
    const sameWire = candidate.probe.provider.wire === clientApi;
    for (;;) {
      const outcome = await callCandidate(candidate, clientApi, body, neutral, anthropicVersion, sameWire && neutral.stream);
      attemptsByProvider.set(providerId, (attemptsByProvider.get(providerId) ?? 0) + 1);
      if (outcome.attempt !== null) {
        await deliver(candidate, outcome.attempt);
        return true;
      }
      const mayRetry = !desperate && !retriedProviders.has(providerId) && outcome.position <= planningCaps(providerId).min;
      if (!mayRetry) {
        failedProviders.add(providerId);
        return false;
      }
      retriedProviders.add(providerId);
    }
  };

  // The walk: candidates in ranked order, each provider inside its drawn budget's ceiling.
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    if (failedProviders.has(providerId)) continue;
    if ((usedCalls.get(providerId) ?? 0) >= planningCaps(providerId).max) continue;
    if (await tryCandidate(candidate, false)) return;
  }
  // The desperate pass: ceilings and this request's refusals aside, anything not already tried
  // twice for this request gets one more call — a served request at any rating beats an error.
  for (const candidate of candidates) {
    if ((attemptsByProvider.get(candidate.probe.provider.id) ?? 0) >= 2) continue;
    if (await tryCandidate(candidate, true)) return;
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory could serve this request');
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    const run = async (): Promise<void> => {
      try {
        await handle(req, res);
      } catch (err) {
        log((err as Error).stack ?? String(err));
        if (!res.headersSent) {
          sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
        }
      }
    };
    if (req.method === 'POST') {
      // One inference at a time, in arrival order: a provider serves one call at a time, and a second
      // call sent while its first is in flight is refused and scored, so calls never overlap here.
      inferenceChain = inferenceChain.then(run, run);
    } else {
      void run();
    }
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
