// The trial brief: the free model list, the six prompts, and the catalog.
//
// A trial hands this server one file (MAKEFUN_PROMPT_FILE). The file is prose
// wrapped around two JSON documents, each carrying a `schema` member under
// this competition's prefix. Nothing in the prose is parsed; the documents are
// found by the arrays they carry.
//
//   - models (`free-model-aggregator-02/models/v1`): one row per free model,
//     keyed by the id the gateway routes (`openrouter/<vendor>/<model>:free`),
//     with the house's quality rating in milli-points, what the model can do
//     (tools, vision, JSON mode, context length) and its free flag.
//   - prompts (`free-model-aggregator-02/prompts/v1`): the six prompts the
//     env will send, in arrival order, each with its id, wire, streaming
//     flag, weight and the requirements a model must meet to serve it.
//
// Everything in this file is a pure function of its inputs: no clock, no
// randomness. The same brief builds the same catalog on a live pass and on
// its replay, byte for byte.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** The schema prefix every document in this competition's brief carries. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** OpenRouter's public endpoint: the catalog names it, never the gateway's trial address. */
export const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';
/** The env var an entrant's OpenRouter key is pushed under. */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  /** The house's quality rating, milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs from the model that serves it. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the env will send, in the brief's (arrival) order. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as read. `error` is set when a document could not be found. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model row of the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** The one provider of the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the register body carries: the same document. */
export interface Catalog {
  providers: CatalogProvider[];
}

/** What a model can do, as the brief's row states it (null: unstated). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** The index of the `}` that closes the object opening at `start`, or -1. */
function closingBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') {
      inString = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function schemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = closingBrace(text, at);
    if (end === -1) continue;
    try {
      const parsed: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(parsed) && typeof parsed.schema === 'string') found.push(parsed);
    } catch {
      // Not a JSON object on its own: keep scanning.
    }
  }
  return found;
}

/**
 * The first document under this competition's prefix that holds one of `keys`
 * as an array, with the key it was found under. A brief that spells its
 * lists under the sibling contest's names still reads.
 */
function documentWithArray(
  docs: readonly Json[],
  keys: readonly string[],
): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!String(doc.schema ?? '').includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

function textOrEmpty(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider
 * that names it; a bare upstream id under `openrouter` is routed with the
 * gateway's prefix. A row is free unless it says `free: false`.
 */
function modelRowOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = textOrEmpty(v.id) !== '' ? textOrEmpty(v.id) : textOrEmpty(v.model);
  if (raw === '') return null;
  const named = textOrEmpty(v.provider) !== '' ? textOrEmpty(v.provider) : textOrEmpty(v.wire);
  const id =
    named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/')
      ? `openrouter/${raw}`
      : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: textOrEmpty(v.display_name) !== '' ? textOrEmpty(v.display_name) : raw,
    rating_milli: numOrNull(v.rating_milli) ?? 0,
    context_length: numOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptRowOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = textOrEmpty(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(req.tools === true ? { tools: true } : {}),
      ...(req.vision === true ? { vision: true } : {}),
      ...(req.json === true ? { json: true } : {}),
      ...(typeof req.min_context === 'number' && Number.isFinite(req.min_context)
        ? { min_context: req.min_context }
        : {}),
    },
  };
}

/** The brief at `path`: its model list and its prompt manifest. */
export function readBrief(path: string): Brief {
  if (path === '') {
    return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const modelsDoc = documentWithArray(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const row = modelRowOf(v);
    return row === null ? [] : [row];
  });
  const promptsDoc = documentWithArray(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const row = promptRowOf(v);
    return row === null ? [] : [row];
  });
  return { models, prompts, error: null };
}

/** The models this entry may dial: the brief's free ones, in brief order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** What a model can do, as the brief's row states it. */
export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/**
 * Whether a model's capabilities meet one prompt's requirements. An unknown
 * capability never meets a requirement.
 */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (
    requires.min_context !== undefined &&
    (caps.context_length === null || caps.context_length < requires.min_context)
  ) {
    return false;
  }
  return true;
}

/**
 * The free models that can serve a prompt, best first: highest rating_milli,
 * ties in the brief's order (a stable sort keeps the brief's order among
 * equals, so the choice is the same on a live pass and on its replay).
 *
 * Quality is 60% of the score, so the rating decides: for the trial-0 brief
 * this picks Inkling 2 Small (1475) wherever it qualifies, and Nemotron 3.5
 * Super VL (1303) for the JSON-mode prompts Inkling cannot serve.
 */
export function rankedCandidates(brief: Brief, requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
}

/**
 * The catalog the brief's free model list makes: one provider, every free
 * model once, in the brief's order. The base URL is OpenRouter's public
 * endpoint, never the gateway's trial-local address: the gateway's origin
 * differs between a live pass and its replay, and the catalog is part of the
 * state export a replay must reproduce byte for byte.
 */
export function buildCatalog(brief: Brief): Catalog {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_BASE_URL,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
