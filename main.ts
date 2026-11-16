// free-model-aggregator-02: one server, two client wires, a real free model
// behind every answer.
//
// The env (MAKEFUN_ENV_URL) drives six prompts at this server, one at a
// time: three on POST /v1/chat/completions (OpenAI wire) and three on POST
// /v1/messages (Anthropic wire), two of them streaming. Every prompt is
// answered by dialling ONE free model through the grader's gateway
// ($OPENAI_BASE_URL/chat/completions) on the trial's key, and never with an
// answer of this server's own. Each prompt carries a unique token the answer
// must carry back; the model is instructed to return exactly that token.
//
// HOW A PROMPT IS SERVED
//   1. Its requirements come from the brief's manifest, in arrival order
//      (falling back to the body's own signals when the manifest is spent).
//   2. The candidates are the brief's free models meeting those requirements,
//      best first by the house's rating_milli. Quality is 60% of the score,
//      so the rating decides: Inkling 2 Small (1475) wherever it qualifies,
//      Nemotron 3.5 Super VL (1303) for the JSON-mode prompts Inkling cannot
//      serve.
//   3. The best candidate is dialled. A 429 or 5xx is a refusal, not an
//      answer: the model goes on this trial's cooling-down list (never spend
//      another call on it while a fresh candidate exists), the failure is
//      recorded, and the prompt fails over to the next-best candidate. A 400
//      or 401 is not retried: no other model fixes a disallowed model or a
//      missing key. At most three gateway calls are spent on one prompt.
//   4. A prompt served after refusals carries them in the
//      `x-makefun-upstream-failures` header: the only way the failover
//      column can see a refusal the server retried and then served.
//   5. The response's `model` is always the id actually dialled. The
//      provider's own answer spells it without the gateway's `openrouter/`
//      prefix; it is put back on every answer and every streamed chunk.
//
// DETERMINISM
//   Every final trial runs twice, live and then replayed from the recorded
//   gateway answers, and the replay must leave the same state export byte
//   for byte. So every choice here is a function of the brief, the requests
//   and the gateway's answers so far: no wall clock (the model listings use
//   fixed timestamps; no Date.now anywhere a request could carry), no
//   unseeded randomness, and the cooling-down list is refusal history, not
//   time. The poll timer's readings never enter a request.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  meetsRequirements,
  rankedCandidates,
  readBrief,
  type Brief,
  type Catalog,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most gateway calls spent on one prompt (the trial caps at 60). */
const MAX_ATTEMPTS_PER_PROMPT = 3;
/** The pause between attempts, in milliseconds (never inside a request). */
const RETRY_PAUSE_MS = 500;
/** One gateway call may take at most this long: a hung dial fails over
 * instead of stalling the trial (its wall is 240s; a healthy model answers
 * in seconds, so 45s is generous for a slow one and still bounds a dead one). */
const DIAL_TIMEOUT_MS = 45000;
/** How often the register is retried while the env is not up yet. */
const REGISTER_TRIES = 20;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief);
/** Which prompt row the next completion request is: the brief's prompts are in arrival order. */
let manifestIndex = 0;
/** Models that refused (429/5xx) on this trial: cooling down, never re-dialled while a fresh one exists. */
const coolingDown = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI root
 * (`…/v1` on the trial network); a base that does not already end in `/v1`
 * gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1')
    ? `${GATEWAY_URL}/chat/completions`
    : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({
    id: model.id,
    provider: model.provider,
    display_name: model.display_name,
  }));
}

/** One refusal, as the `x-makefun-upstream-failures` header reports it. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** The error code of a gateway refusal body, or a fallback from its status. */
function failureCode(status: number, bodyText: string): string {
  try {
    const parsed: unknown = JSON.parse(bodyText);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      const code = parsed.error.code;
      if (typeof code === 'string' && code !== '') return code;
      const type = parsed.error.type;
      if (typeof type === 'string' && type !== '') return type;
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/**
 * The candidates for a prompt, best first, with cooling-down models last:
 * a model that refused on this trial is only dialled again when no fresh
 * candidate meeting the requirements is left.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const ranked = rankedCandidates(brief, requires);
  const fresh = ranked.filter((m) => !coolingDown.has(m.id));
  const cooled = ranked.filter((m) => coolingDown.has(m.id));
  return [...fresh, ...cooled];
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
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

function sendJson(res: ServerResponse, status: number, body: unknown, failures: UpstreamFailure[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(failures.slice(0, 8));
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  failures: UpstreamFailure[] = [],
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One gateway call: the Chat Completions body with the dialled model's id. */
async function dial(model: string, body: Json): Promise<Response> {
  const upstreamBody: Json = { ...body, model };
  if (upstreamBody.stream === true) {
    const options = isRecord(upstreamBody.stream_options) ? upstreamBody.stream_options : {};
    upstreamBody.stream_options = { ...options, include_usage: true };
  }
  // The gateway refuses a body with no output cap: the env always sends one,
  // this is only a guard for a body that arrives without it.
  if (typeof upstreamBody.max_tokens !== 'number' && typeof upstreamBody.max_completion_tokens !== 'number') {
    upstreamBody.max_tokens = 1024;
  }
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (upstreamBody.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(upstreamBody),
    signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
  });
}

interface Dialled {
  /** The gateway's answer, when one was served. */
  response: Response | null;
  /** The model that served it, when one did. */
  model: string | null;
  /** The refusals recorded while serving this prompt, in order. */
  failures: UpstreamFailure[];
}

/**
 * Dial a prompt until one model serves it or the attempts run out. A 429 or
 * 5xx cools the model down and fails over to the next-best candidate; a 400
 * or 401 is not retried, and a network error counts as a refusal of the
 * model that was dialled.
 */
async function serveWithFailover(body: Json, requires: Requirement): Promise<Dialled> {
  const failures: UpstreamFailure[] = [];
  const candidates = candidatesFor(requires);
  const attempts = Math.min(MAX_ATTEMPTS_PER_PROMPT, candidates.length);
  for (let i = 0; i < attempts; i++) {
    const model = candidates[i];
    let upstream: Response;
    try {
      upstream = await dial(model.id, body);
    } catch (err) {
      const timedOut = (err as Error).name === 'TimeoutError';
      failures.push({ status: 0, code: timedOut ? 'timeout' : 'gateway_unreachable', model: model.id });
      log(`dial ${model.id} failed: ${(err as Error).message}`);
      coolingDown.add(model.id);
      if (i + 1 < attempts) await sleep(RETRY_PAUSE_MS);
      continue;
    }
    if (upstream.ok) return { response: upstream, model: model.id, failures };
    const text = await upstream.text().catch(() => '');
    failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: model.id });
    log(`dial ${model.id} refused HTTP ${upstream.status}`);
    if (upstream.status === 400 || upstream.status === 401) {
      // A disallowed model or a missing key: no other model fixes it.
      coolingDown.add(model.id);
      return { response: null, model: null, failures };
    }
    coolingDown.add(model.id);
    if (i + 1 < attempts) await sleep(RETRY_PAUSE_MS);
  }
  return { response: null, model: null, failures };
}

/** A prompt no model served, as an error in the client's wire. */
function giveUp(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  failures: UpstreamFailure[],
): void {
  const last = failures[failures.length - 1];
  const status = last !== undefined && last.status > 0 ? last.status : 502;
  if (wire === 'anthropic') {
    sendError(res, wire, 502, 'api_error', `the gateway refused every model tried (last HTTP ${status})`, failures);
  } else {
    sendError(res, wire, 502, 'upstream_error', `the gateway refused every model tried (last HTTP ${status})`, failures);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const requires = requirementsFor(body);
  const served = await serveWithFailover(body, requires);
  if (served.response === null || served.model === null || served.response.body === null) {
    giveUp(res, 'openai', served.failures);
    return;
  }
  const upstream = served.response;
  const model = served.model;
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...(served.failures.length > 0
        ? { 'x-makefun-upstream-failures': JSON.stringify(served.failures.slice(0, 8)) }
        : {}),
    });
    try {
      for await (const frame of openAIStreamWithModel(upstream.body, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), served.failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const requires = requirementsFor(body);
  const translated = anthropicToOpenAI(body, 'auto');
  const served = await serveWithFailover(translated, requires);
  if (served.response === null || served.model === null || served.response.body === null) {
    giveUp(res, 'anthropic', served.failures);
    return;
  }
  const upstream = served.response;
  const model = served.model;
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...(served.failures.length > 0
        ? { 'x-makefun-upstream-failures': JSON.stringify(served.failures.slice(0, 8)) }
        : {}),
    });
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), served.failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog, retrying while the env is not up yet. */
async function register(port: number, report: Catalog): Promise<void> {
  let lastError: string = '';
  for (let attempt = 0; attempt < REGISTER_TRIES; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the trial holds ${reply.requests} prompts`);
      }
      return;
    } catch (err) {
      lastError = (err as Error).message;
      log(`register attempt ${attempt + 1} failed: ${lastError}`);
      await sleep(RETRY_PAUSE_MS);
    }
  }
  throw new Error(`could not register after ${REGISTER_TRIES} attempts: ${lastError}`);
}

/** Poll GET /work; exit 0 the moment it says done. */
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
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts. A process whose
    // stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
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
