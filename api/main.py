import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from uuid import uuid4
import urllib.parse
from datetime import datetime, timezone
import io
import csv
import aiohttp
from aiohttp import ClientTimeout, FormData
import base64

print("--- DEBUG: api/main.py TOP LEVEL EXECUTION ---", flush=True)

try:
    from fastapi import FastAPI, HTTPException, Body, Response, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field, field_validator, ConfigDict # Import ConfigDict for Pydantic V2
    from openai import AsyncOpenAI # Use Async client
    from dotenv import load_dotenv
    import redis
    from redis.asyncio import ConnectionPool as AsyncConnectionPool, Redis as AsyncRedis # Use Async Redis
    from vercel_blob import put as blob_put, head as blob_head, delete as blob_delete # Import Async Blob functions
    print("--- DEBUG: Imported Async libs and FastAPI components ---", flush=True)

    app = FastAPI(title="Enhanced Bayesian Network API with UI Analysis")
    print("--- DEBUG: FastAPI app created ---", flush=True)

    app.add_middleware( CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"] )
    print("--- DEBUG: CORS middleware added ---", flush=True)

except Exception as e:
    print(f"--- DEBUG: ERROR during core imports: {e} ---", flush=True)
    raise e

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
redis_url = os.getenv("KV_URL") or os.getenv("REDIS_URL") # Vercel KV provides KV_URL
vercel_blob_token = os.getenv("BLOB_READ_WRITE_TOKEN") # Vercel Blob provides this

# Initialize async OpenAI client
openai_client = None
if openai_api_key:
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    logger.info("--- Async OpenAI client initialized ---")
else:
    logger.warning("OPENAI_API_KEY missing. OpenAI features disabled.")

# Async Redis Connection Pool
redis_client: Optional[AsyncRedis] = None
if redis_url:
    try:
        # Use redis.asyncio for async Redis client compatible with FastAPI async functions
        redis_pool = AsyncConnectionPool.from_url(redis_url, decode_responses=True, max_connections=10)
        redis_client = AsyncRedis(connection_pool=redis_pool)
        logger.info("--- Async Redis connection pool created ---")
        # Note: Ping might be done later within endpoints to ensure connection liveness
    except Exception as e:
        logger.error(f"--- Failed to create Async Redis connection pool: {e} ---", exc_info=True)
        redis_client = None
else:
    logger.warning("--- Redis URL missing. Config storage disabled. ---")

if not vercel_blob_token:
    logger.warning("BLOB_READ_WRITE_TOKEN missing. Blob storage disabled.")


# Constants
CONFIG_KEY_PREFIX = "bn_config:"
DEFAULT_CONFIG_KEY = "bn_default_config" # Key to store the ID of the default config
LOG_FILENAME_PREFIX = "bn_log_"
LOG_FILENAME_SUFFIX = ".csv"
# Updated base headers, specific node/feature headers added dynamically in log_data_to_blob
BASE_CSV_HEADERS = ['Timestamp', 'ConfigID', 'ConfigName']

# Default Graph Structure
DEFAULT_GRAPH_STRUCTURE = {
    "id": "default-config-001", # Keep a consistent ID for the default
    "name": "Default Cognitive Model",
    "graph_structure": {
        # ... (nodes and edges as defined previously) ...
         "nodes": [
            {"id": "A1", "fullName": "Domain Expertise", "nodeType": "input"},
            {"id": "A2", "fullName": "Web Literacy", "nodeType": "input"},
            {"id": "A3", "fullName": "Task Familiarity", "nodeType": "input"},
            {"id": "A4", "fullName": "Goal Clarity", "nodeType": "input"},
            {"id": "A5", "fullName": "Cognitive Capacity", "nodeType": "input"},
            {"id": "A6", "fullName": "Risk Aversion", "nodeType": "input"},
            {"id": "H", "fullName": "Recent History", "nodeType": "input"}, # Make sure H is in PersonaInputs model
            {"id": "IS1", "fullName": "Perceived UI Quality", "nodeType": "hidden"},
            {"id": "IS2", "fullName": "Task Understanding (Situational)", "nodeType": "hidden"},
            {"id": "IS3", "fullName": "Cognitive Load (Induced)", "nodeType": "hidden"},
            {"id": "IS4", "fullName": "Confidence / Self-Efficacy", "nodeType": "hidden"},
            {"id": "IS5", "fullName": "Attentional Focus", "nodeType": "hidden"},
            {"id": "IS6", "fullName": "Affective State", "nodeType": "hidden"},
            {"id": "O1", "fullName": "Action Success Likelihood", "nodeType": "hidden"},
            {"id": "O2", "fullName": "Action Efficiency", "nodeType": "hidden"},
            {"id": "O3", "fullName": "Help Seeking / Abandonment Likelihood", "nodeType": "hidden"},
            {"id": "O4", "fullName": "Exploration Likelihood", "nodeType": "hidden"},
        ],
        "edges": [
            {"source": "A2", "target": "IS1"},
            {"source": "A1", "target": "IS2"}, {"source": "A4", "target": "IS2"}, {"source": "IS1", "target": "IS2"},
            {"source": "A2", "target": "IS3"}, {"source": "A3", "target": "IS3"}, {"source": "A5", "target": "IS3"},
            {"source": "IS1", "target": "IS3"}, {"source": "IS2", "target": "IS3"},
            {"source": "A1", "target": "IS4"}, {"source": "A3", "target": "IS4"}, {"source": "A6", "target": "IS4"},
            {"source": "H", "target": "IS4"}, {"source": "IS1", "target": "IS4"}, {"source": "IS2", "target": "IS4"},
            {"source": "IS3", "target": "IS4"},
            {"source": "A4", "target": "IS5"}, {"source": "A5", "target": "IS5"}, {"source": "IS1", "target": "IS5"},
            {"source": "IS3", "target": "IS5"},
            {"source": "A6", "target": "IS6"}, {"source": "H", "target": "IS6"}, {"source": "IS1", "target": "IS6"},
            {"source": "IS3", "target": "IS6"}, {"source": "IS4", "target": "IS6"},
            {"source": "IS2", "target": "O1"}, {"source": "IS4", "target": "O1"}, {"source": "IS5", "target": "O1"},
            {"source": "IS3", "target": "O1"},
            {"source": "IS2", "target": "O2"}, {"source": "IS4", "target": "O2"}, {"source": "IS5", "target": "O2"},
            {"source": "IS3", "target": "O2"},
            {"source": "IS2", "target": "O3"}, {"source": "IS4", "target": "O3"}, {"source": "IS6", "target": "O3"},
            {"source": "A6", "target": "O4"}, {"source": "IS4", "target": "O4"}, {"source": "IS5", "target": "O4"},
            {"source": "IS6", "target": "O4"},
        ]
    }
}

# === Pydantic Models (Updated for V2 Config) ===

class NodeData(BaseModel):
    id: str
    fullName: str = Field(...)
    nodeType: str = Field(...) # 'input' or 'hidden'
    model_config = ConfigDict(extra='ignore') # Pydantic V2 config

    @field_validator('nodeType')
    def valid_node_type(cls, v): 
        if v not in ['input', 'hidden']: 
            raise ValueError("Invalid nodeType"); return v

class EdgeData(BaseModel):
    source: str
    target: str
    model_config = ConfigDict(extra='ignore')

class GraphStructure(BaseModel):
    nodes: List[NodeData]
    edges: List[EdgeData]
    model_config = ConfigDict(extra='ignore')

    @field_validator('nodes')
    def nodes_unique(cls, v): 
        ids = [n.id for n in v]; 
        if len(ids) != len(set(ids)): 
            raise ValueError("Node IDs must be unique"); 
        return v
    @field_validator('edges')
    def edges_valid(cls, v, values):
        # Use model_dump() in Pydantic V2 if values is a model instance
        nodes_data = values.data.get('nodes') if hasattr(values, 'data') else values.get('nodes')
        if not nodes_data: return v # Should not happen if validation runs after nodes
        node_ids = {n.id for n in nodes_data}
        for e in v:
            if e.source not in node_ids or e.target not in node_ids: raise ValueError(f"Edge {e.source}->{e.target} invalid");
        return v


class UiFeatures(BaseModel):
    Clarity: float = Field(..., ge=0.0, le=1.0)
    Familiarity: float = Field(..., ge=0.0, le=1.0)
    Findability: float = Field(..., ge=0.0, le=1.0)
    Complexity: float = Field(..., ge=0.0, le=1.0) # 0=Simple, 1=Complex
    AestheticTrust: float = Field(..., ge=0.0, le=1.0)
    model_config = ConfigDict(extra='ignore')

class CognitiveStateProbabilities(BaseModel):
    IS1: float = Field(..., alias="Perceived UI Quality", ge=0.0, le=1.0)
    IS2: float = Field(..., alias="Task Understanding (Situational)", ge=0.0, le=1.0)
    IS3: float = Field(..., alias="Cognitive Load (Induced)", ge=0.0, le=1.0)
    IS4: float = Field(..., alias="Confidence / Self-Efficacy", ge=0.0, le=1.0)
    IS5: float = Field(..., alias="Attentional Focus", ge=0.0, le=1.0)
    IS6: float = Field(..., alias="Affective State", ge=0.0, le=1.0)
    model_config = ConfigDict(populate_by_name=True, extra='ignore') # Updated Pydantic V2 config


class Stage2LLMResponse(BaseModel):
    cognitive_state_probabilities: CognitiveStateProbabilities
    user_perception_summary: str
    model_config = ConfigDict(extra='ignore')

class PersonaInputs(BaseModel):
    A1: float = Field(..., ge=0.0, le=1.0, alias="Domain Expertise")
    A2: float = Field(..., ge=0.0, le=1.0, alias="Web Literacy")
    A3: float = Field(..., ge=0.0, le=1.0, alias="Task Familiarity")
    A4: float = Field(..., ge=0.0, le=1.0, alias="Goal Clarity")
    A5: float = Field(..., ge=0.0, le=1.0, alias="Cognitive Capacity")
    A6: float = Field(..., ge=0.0, le=1.0, alias="Risk Aversion")
    H: Optional[float] = Field(0.5, ge=0.0, le=1.0, alias="Recent History") # Optional H node
    model_config = ConfigDict(populate_by_name=True, extra='ignore') # Updated Pydantic V2 config

class PredictionPayload(BaseModel):
    persona_inputs: PersonaInputs
    ui_features: UiFeatures # From stage 1
    graph_structure: GraphStructure # The BN definition
    config_id: Optional[str] = None # ID if saved, for logging
    config_name: Optional[str] = "Unsaved/Unknown" # Name for logging
    model_config = ConfigDict(extra='ignore')

class SaveConfigPayload(BaseModel):
    config_name: str = Field(..., min_length=1)
    graph_structure: GraphStructure
    model_config = ConfigDict(extra='ignore')

class LogPayload(BaseModel): # Used internally for blob logging format
    timestamp: str
    configId: str
    configName: str
    uiFeatures: Dict[str, float]
    nodeProbabilities: Dict[str, float] # All nodes (A, IS, O) with P(1)
    userPerceptionSummary: str
    llmReasoning: str
    model_config = ConfigDict(extra='ignore')

# --- Helper Functions ---
# (get_dynamic_node_info and is_dag definitions remain the same)
def get_dynamic_node_info(graph: GraphStructure) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], List[str], List[str], List[str]]:
    """ Parses graph structure to get node info needed for LLM prompts and logic. """
    node_parents: Dict[str, List[str]] = {}
    node_descriptions: Dict[str, str] = {}
    node_types: Dict[str, str] = {}
    nodes_set: Set[str] = set()
    for node in graph.nodes:
        if node and node.id: node_parents[node.id] = []; node_descriptions[node.id] = node.fullName; node_types[node.id] = node.nodeType; nodes_set.add(node.id)
        else: logger.warning(f"get_dynamic_node_info: Skipping invalid node: {node}")
    for edge in graph.edges:
        if edge and edge.source and edge.target and edge.source in nodes_set and edge.target in nodes_set:
            if edge.target in node_parents: node_parents[edge.target].append(edge.source)
            else: logger.warning(f"get_dynamic_node_info: Target node '{edge.target}' not in keys for edge {edge.source}->{edge.target}")
        else: logger.warning(f"get_dynamic_node_info: Skipping invalid edge: {edge}")
    all_nodes = sorted(list(nodes_set))
    target_nodes = sorted([node_id for node_id, node_type in node_types.items() if node_type == 'hidden'])
    input_nodes = sorted([node_id for node_id, node_type in node_types.items() if node_type == 'input'])
    return node_parents, node_descriptions, node_types, all_nodes, target_nodes, input_nodes

def is_dag(graph: GraphStructure) -> Tuple[bool, Optional[str]]:
    adj: Dict[str, List[str]] = {node.id: [] for node in graph.nodes}; nodes_set = {node.id for node in graph.nodes}
    for edge in graph.edges:
        if edge.source not in nodes_set or edge.target not in nodes_set: return False, f"Edge refs invalid node: {edge.source}->{edge.target}"
        if edge.source not in adj: adj[edge.source] = []
        adj[edge.source].append(edge.target)
    path: Set[str] = set(); visited: Set[str] = set()
    def dfs(node: str) -> bool:
        path.add(node); visited.add(node)
        for neighbor in adj.get(node, []):
            if neighbor in path: return False
            if neighbor not in visited:
                if not dfs(neighbor): return False
        path.remove(node); return True
    for node in nodes_set:
        if node not in visited:
            if not dfs(node): return False, f"Cycle detected near {node}"
    return True, None


# --- OpenAI Callers ---
async def call_openai_stage1_ui_features(image_base64: str, task_description: str) -> UiFeatures:
    # ... (Keep existing implementation) ...
    if not openai_client: raise HTTPException(503, "OpenAI client not available.")
    logger.info(f"Stage 1: Analyzing UI for task: '{task_description[:50]}...'")
    system_prompt = """You are a UI/UX expert analyzing a webpage screenshot for a specific user task. Evaluate the UI features relevant to the task on a 0.0 (very poor/absent/complex) to 1.0 (excellent/present/simple) scale. Output ONLY a JSON object containing float values for keys: "Clarity", "Familiarity", "Findability", "Complexity", "AestheticTrust". - Clarity: Visual clarity for the task (hierarchy, labels). - Familiarity: Use of common web patterns/conventions for task elements. - Findability: Ease of locating necessary elements for the task. - Complexity: Perceived steps/elements for the task (0.0=Very Simple, 1.0=Very Complex). - AestheticTrust: Professionalism and trustworthiness of the visual design."""
    user_prompt = f"Analyze the UI in the image for the task: \"{task_description}\""
    try:
        response = await openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": [{"type": "text", "text": user_prompt},{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}], response_format={"type": "json_object"}, max_tokens=300, temperature=0.1)
        content = response.choices[0].message.content; logger.info(f"Stage 1 LLM Raw Output: {content}"); features_dict = json.loads(content); return UiFeatures(**features_dict)
    except json.JSONDecodeError as e: logger.error(f"Stage 1 JSON Parse Error: {e}. Output: {content}"); raise HTTPException(500, f"LLM (Stage 1) invalid JSON: {e}")
    except Exception as e: logger.error(f"Stage 1 OpenAI Error: {e}", exc_info=True); raise HTTPException(502, f"OpenAI Stage 1 failed: {e}")

async def call_openai_stage2_cognitive_states(persona: PersonaInputs, ui_features: UiFeatures, graph: GraphStructure) -> Stage2LLMResponse:
    # ... (Keep existing implementation) ...
    if not openai_client: raise HTTPException(503, "OpenAI client not available.")
    logger.info("Stage 2: Estimating cognitive states..."); persona_desc = "\n".join([f"- {f.alias or name}: {getattr(persona, name):.2f}" for name, f in persona.model_fields.items()]); ui_features_desc = "\n".join([f"- {name}: {getattr(ui_features, name):.2f}" for name in ui_features.model_fields]); node_parents, node_descriptions, _, _, _, _ = get_dynamic_node_info(graph); is_nodes = ["IS1", "IS2", "IS3", "IS4", "IS5", "IS6"]; dependencies_desc = "\n".join([f"- {n} ({node_descriptions.get(n,n)}): <- {', '.join(node_parents.get(n,[])) or 'Inputs'}" for n in is_nodes]); system_prompt = """You simulate a user's cognitive state interacting with a UI for a specific task. Inputs: User persona probabilities (0-1), Objective UI feature scores (0-1), and Network dependencies. Task: Estimate P(Node=1) for intermediate cognitive state nodes (IS1-IS6) based on the interaction between persona and UI features, following the network dependencies. Also provide a brief (2-3 sentence) summary of how this user might perceive/struggle with this UI for the task, starting with "User Perception:". Output ONLY a single JSON object with two keys: 1. "cognitive_state_probabilities": An object mapping IS1-IS6 node IDs to their P(1) float values (0.0-1.0). 2. "user_perception_summary": A string containing the perception summary. Example: {"cognitive_state_probabilities": {"IS1": 0.6, ...}, "user_perception_summary": "User Perception: ..."}"""; user_prompt = f"""User Persona (P=1):\n{persona_desc}\n\nObjective UI Feature Scores (0-1):\n{ui_features_desc}\n\nCognitive Network Dependencies:\n{dependencies_desc}\n\nEstimate P(Node=1) for IS1, IS2, IS3, IS4, IS5, IS6 and provide the user perception summary in the specified JSON format.""";
    try:
        response = await openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"}, max_tokens=600, temperature=0.2); content = response.choices[0].message.content; logger.info(f"Stage 2 LLM Raw Output: {content}"); response_dict = json.loads(content); validated_response = Stage2LLMResponse(**response_dict); return validated_response
    except json.JSONDecodeError as e: logger.error(f"Stage 2 JSON Parse Error: {e}. Output: {content}"); raise HTTPException(500, f"LLM (Stage 2) invalid JSON: {e}")
    except Exception as e: logger.error(f"Stage 2 OpenAI/Validation Error: {e}", exc_info=True); raise HTTPException(502, f"OpenAI Stage 2 failed: {e}")


def calculate_outcome_nodes(persona: PersonaInputs, cognitive_states: CognitiveStateProbabilities) -> Dict[str, float]:
    # ... (Keep existing implementation) ...
     logger.info("Stage 3: Calculating outcome nodes..."); o1 = 0.3*cognitive_states.IS2 + 0.4*cognitive_states.IS4 + 0.3*cognitive_states.IS5 - 0.3*cognitive_states.IS3; o2 = 0.2*cognitive_states.IS2 + 0.4*cognitive_states.IS4 + 0.4*cognitive_states.IS5 - 0.2*cognitive_states.IS3; o3 = 0.4*(1-cognitive_states.IS2) + 0.4*(1-cognitive_states.IS4) + 0.5*(1-cognitive_states.IS6); o4 = -0.3*persona.A6 + 0.3*cognitive_states.IS4 + 0.2*cognitive_states.IS5 + 0.4*cognitive_states.IS6; base = 0.1; outcomes = {"O1": max(0.0, min(1.0, base + o1)), "O2": max(0.0, min(1.0, base + o2)), "O3": max(0.0, min(1.0, base + o3)), "O4": max(0.0, min(1.0, base + o4))}; logger.info(f"Calculated Outcomes: {outcomes}"); return outcomes

async def call_openai_stage4_reasoning(
    persona: PersonaInputs, ui_features: UiFeatures,
    cognitive_states: CognitiveStateProbabilities, outcomes: Dict[str, float],
    graph: GraphStructure, user_perception: str, task_description: str
    ) -> str:
    """Stage 4: Generate final reasoning explanation."""
    if not openai_client: return "Reasoning disabled: OpenAI client not available."
    logger.info("Stage 4: Generating reasoning...")

    # Format inputs
    persona_desc = "\n".join([f"- {f.alias or name}: {getattr(persona, name):.2f}" for name, f in persona.model_fields.items()])
    ui_features_desc = "\n".join([f"- {name}: {getattr(ui_features, name):.2f}" for name in ui_features.model_fields])
    cognitive_states_desc = "\n".join([f"- {name} ({f.alias or name}): {getattr(cognitive_states, name):.3f}" for name, f in cognitive_states.model_fields.items()])
    outcomes_desc = "\n".join([f"- {name}: {value:.3f}" for name, value in outcomes.items()])
    _, node_descriptions, _, _, _, _ = get_dynamic_node_info(graph) # Get descriptions

    # --- *** FIX IS HERE: Assign prompts to variables *** ---
    system_message = """
    You are a UX analyst explaining a simulation of user cognition.
    Given: User Persona, Objective UI scores, Task, User Perception Summary, Estimated Cognitive States (IS1-6), Predicted Outcomes (O1-4).
    Task: Provide detailed reasoning, explaining:
    1. Briefly, how UI scores reflect the visual + task.
    2. How Persona *interacted* with UI scores -> User Perception & Cognitive States (IS1-6). Reference specific inputs.
    3. How Cognitive States -> Predicted Outcomes (O1-4). Reference specific IS nodes.
    Be specific and link causes to effects based on standard UX principles. Use bullet points or structured paragraphs for clarity.
    """
    user_message = f"""
    Simulation Context:
    Task: "{task_description}"
    User Persona (P=1):
    {persona_desc}
    Objective UI Feature Scores (0-1):
    {ui_features_desc}
    User Perception Summary:
    {user_perception}
    Estimated Cognitive States (P=1):
    {cognitive_states_desc}
    Predicted Outcomes (P=1):
    {outcomes_desc}
    Node Descriptions:
    {json.dumps(node_descriptions, indent=2)}

    Please provide the detailed step-by-step reasoning for these results.
    """
    # --- *** END FIX *** ---

    logger.debug("Sending reasoning prompt to OpenAI...")
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            # Use the assigned variables here
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1000, # Allow longer reasoning
            temperature=0.4,
        )
        reasoning_text = response.choices[0].message.content.strip()
        logger.info(f"Stage 4 Reasoning Output received.")
        return reasoning_text
    except Exception as e:
        logger.error(f"Stage 4 Reasoning Error: {e}", exc_info=True)
        # Provide a more informative error message back
        return f"Could not generate reasoning due to API error: {str(e)[:200]}..." # Limit error length

# --- Blob Logging ---
async def log_data_to_blob(log_data: LogPayload, all_node_ids: List[str], ui_feature_ids: List[str]):
    # ... (Keep existing implementation - checked for correctness) ...
    if not vercel_blob_token: logger.warning("Blob token missing, skip log."); return
    if not log_data.configId or log_data.configId in ["unknown", "default-config-001", "unsaved"]: logger.info(f"Skipping log for configId: {log_data.configId}"); return
    log_filename = f"{LOG_FILENAME_PREFIX}{log_data.configId}{LOG_FILENAME_SUFFIX}"; logger.info(f"Attempting log append to: {log_filename}"); sorted_ui_feature_ids = sorted(ui_feature_ids); sorted_node_ids = sorted(all_node_ids); headers = BASE_CSV_HEADERS + sorted_ui_feature_ids + sorted_node_ids + ["UserPerceptionSummary", "LLMReasoning"]; data_row = [ log_data.timestamp, log_data.configId, log_data.configName ];
    for feature_id in sorted_ui_feature_ids: data_row.append(f"{log_data.uiFeatures.get(feature_id, ''):.4f}" if isinstance(log_data.uiFeatures.get(feature_id), float) else '')
    for node_id in sorted_node_ids: data_row.append(f"{log_data.nodeProbabilities.get(node_id, ''):.4f}" if isinstance(log_data.nodeProbabilities.get(node_id), float) else '')
    data_row.append(log_data.userPerceptionSummary); data_row.append(log_data.llmReasoning)
    try:
        existing_content = b""; needs_headers = False
        try:
            logger.debug(f"Checking blob: {log_filename}"); head_result = await blob_head(pathname=log_filename, options={'token': vercel_blob_token}); logger.debug(f"Blob exists, fetching URL: {head_result.get('url')}"); blob_url = head_result.get('url');
            if not blob_url: raise Exception("Blob URL missing")
            async with aiohttp.ClientSession(timeout=ClientTimeout(total=10)) as session:
                async with session.get(blob_url) as response:
                    if response.status == 200: 
                        existing_content = await response.read(); 
                    if existing_content.endswith(b'\n'): 
                        existing_content = existing_content[:-1]; 
                        logger.debug(f"Read {len(existing_content)} bytes.")
                    elif response.status == 404: 
                        needs_headers = True; 
                        logger.info(f"Log URL 404.")
                    else: 
                        raise Exception(f"HTTP {response.status} fetching log.")
        except Exception as head_or_download_error: logger.info(f"Log not found or download error ({head_or_download_error}). Creating new."); needs_headers = True
        output = io.StringIO(); writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        if existing_content: output.write(existing_content.decode('utf-8')); output.write('\n')
        if needs_headers: writer.writerow(headers); logger.info("Writing headers.")
        writer.writerow(data_row); final_csv_content = output.getvalue().encode('utf-8'); output.close()
        logger.debug(f"Uploading {len(final_csv_content)} bytes to {log_filename}...")
        put_result = await blob_put(pathname=log_filename, body=final_csv_content, options={'access': 'public', 'add_random_suffix': False, 'contentType': 'text/csv', 'token': vercel_blob_token})
        logger.info(f"Log write successful: {log_filename}")
    except Exception as e: logger.error(f"FAILED to log to blob {log_filename}: {e}", exc_info=True)


# === API Endpoints ===

@app.get("/api/ping")
async def ping(): # ... (keep existing ping) ...
    redis_status = "disabled";
    if redis_client: 
        try: 
            await redis_client.ping(); redis_status = "connected"
        except Exception: 
            redis_status = "disconnected"
    return {"message": "pong", "redis_status": redis_status}

@app.get("/api/configs/default", response_model=Dict[str, Any])
async def get_default_configuration():
    # ... (Keep corrected version from previous response) ...
    logger.info("Attempting to serve default configuration...")
    config_to_return = None
    if redis_client:
        try:
            await redis_client.ping(); logger.debug("Redis connection OK.")
            default_config_id = await redis_client.get(DEFAULT_CONFIG_KEY)
            if default_config_id:
                logger.info(f"User default ID found: {default_config_id}")
                config_json = await redis_client.get(default_config_id)
                if config_json:
                    try: config_to_return = json.loads(config_json);
                    except Exception: logger.error(f"Failed parse default JSON: {default_config_id}"); config_to_return=None
                    if config_to_return and all(k in config_to_return for k in ["id", "name", "graph_structure"]): logger.info(f"Loaded user default '{config_to_return['name']}'.")
                    else: logger.warning(f"Default data invalid: {default_config_id}. Falling back."); config_to_return=None
                else: logger.warning(f"Default ID '{default_config_id}' key exists, but no data found. Falling back.")
            else: logger.info("No user default key.")
        except redis.RedisError as e: logger.error(f"Redis error checking default: {e}. Falling back.") ; config_to_return = None
        except Exception as e: logger.error(f"Unexpected error checking default: {e}", exc_info=True); config_to_return = None
    if config_to_return is None: logger.info("Returning hardcoded default."); config_to_return = DEFAULT_GRAPH_STRUCTURE
    if not config_to_return or "id" not in config_to_return: logger.error("CRITICAL: Default config is invalid!"); raise HTTPException(500,"Server default config error.")
    return config_to_return

@app.post("/api/extract_ui_features", response_model=UiFeatures)
async def extract_ui_features(task_description: str = Form(...), image: UploadFile = File(...)):
    # ... (Keep existing implementation) ...
    logger.info(f"Received UI analysis request for task: {task_description}")
    if not image.content_type.startswith("image/"): raise HTTPException(400, "Invalid file type.")
    if not openai_client: raise HTTPException(503, "OpenAI client unavailable.")
    try: image_bytes = await image.read(); image_base64 = base64.b64encode(image_bytes).decode("utf-8"); features = await call_openai_stage1_ui_features(image_base64, task_description); return features
    except HTTPException as e: raise e
    except Exception as e: logger.error(f"Error processing UI image: {e}", exc_info=True); raise HTTPException(500, f"Failed UI analysis: {e}")

@app.post("/api/configs", status_code=201)
async def save_configuration(payload: SaveConfigPayload): # ... (use await redis_client.set) ...
    if not redis_client: raise HTTPException(503, "Storage unavailable.")
    is_valid, cycle_info = is_dag(payload.graph_structure);
    if not is_valid: raise HTTPException(400, f"Invalid graph: Not a DAG. {cycle_info or ''}")
    config_id = f"{CONFIG_KEY_PREFIX}{uuid4()}"; config_data = {"id": config_id, "name": payload.config_name, "graph_structure": payload.graph_structure.model_dump()} # Use model_dump for Pydantic V2
    try: await redis_client.set(config_id, json.dumps(config_data)); logger.info(f"Saved: {config_id}"); return {"id": config_id, "name": payload.config_name}
    except Exception as e: logger.error(f"Redis save error: {e}", exc_info=True); raise HTTPException(500, f"Save failed: {e}")

@app.get("/api/configs", response_model=List[Dict[str, str]])
async def list_configurations(): # ... (use await redis_client.scan_iter, mget) ...
    if not redis_client: return []
    configs_summary = []; keys = [];
    try: 
        async for key in redis_client.scan_iter(match=f"{CONFIG_KEY_PREFIX}*"): keys.append(key)
    except Exception as e: logger.error(f"Redis scan error: {e}"); return []
    if not keys: return []
    try: config_jsons = await redis_client.mget(keys)
    except Exception as e: logger.error(f"Redis mget error: {e}"); return [{"id": k, "name":"*Load Error*"} for k in keys]
    for key, config_json in zip(keys, config_jsons):
        if config_json: 
            try: 
                data = json.loads(config_json); configs_summary.append({"id": data.get("id", key), "name": data.get("name", "Unnamed")})
            except Exception: 
                configs_summary.append({"id": key, "name": "*Parse Error*"})
        else: 
            configs_summary.append({"id": key, "name": "*Not Found*"})
        return configs_summary

@app.get("/api/configs/{config_id}", response_model=Dict[str, Any])
async def load_configuration(config_id: str): # ... (use await redis_client.get) ...
    if not redis_client: raise HTTPException(503, "Storage unavailable.")
    config_id_with_prefix = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
    try: config_json = await redis_client.get(config_id_with_prefix);
    except Exception as e: logger.error(f"Redis load error: {e}"); raise HTTPException(500, "Storage error")
    if config_json is None: raise HTTPException(404, "Config not found.")
    try: return json.loads(config_json)
    except Exception: raise HTTPException(500, "Config data corrupted.")

@app.delete("/api/configs/{config_id}", status_code=200)
async def delete_configuration(config_id: str): # ... (use await redis_client.delete, blob_delete) ...
    if not redis_client: raise HTTPException(503, "Storage unavailable.")
    config_id_with_prefix = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
    log_filename = f"{LOG_FILENAME_PREFIX}{config_id_with_prefix}{LOG_FILENAME_SUFFIX}"
    redis_deleted_count = 0
    try: redis_deleted_count = await redis_client.delete(config_id_with_prefix)
    except Exception as e: logger.error(f"Redis delete error: {e}"); raise HTTPException(500,"Storage delete error (Redis)")
    if redis_deleted_count == 0: raise HTTPException(404, "Config not found in Redis.")
    try: await blob_delete(url=log_filename, token=vercel_blob_token) # Vercel blob expects url=pathname
    except Exception as blob_error: logger.warning(f"Failed to delete log file '{log_filename}': {blob_error}") # Log but don't fail delete
    try: current_default_id = await redis_client.get(DEFAULT_CONFIG_KEY);
    except Exception as e: logger.warning(f"Could not check default key: {e}"); current_default_id = None
    if current_default_id == config_id_with_prefix:
        try: await redis_client.delete(DEFAULT_CONFIG_KEY); logger.info(f"Removed deleted config {config_id_with_prefix} as default.")
        except Exception as e: logger.warning(f"Could not remove default key: {e}")
    logger.info(f"Deleted config '{config_id_with_prefix}'.")
    return {"message": "Configuration deleted successfully."}

@app.post("/api/configs/set_default", status_code=200)
async def set_default_configuration(config_id: str = Body(..., embed=True)): # Ensure body is parsed correctly
    # ... (use await redis_client.get, set) ...
    if not redis_client: raise HTTPException(503, "Storage unavailable.")
    config_id_with_prefix = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
    try:
        config_json = await redis_client.get(config_id_with_prefix)
        if config_json is None: raise HTTPException(404, "Config not found.")
        await redis_client.set(DEFAULT_CONFIG_KEY, config_id_with_prefix)
        config_data = json.loads(config_json)
        logger.info(f"Set '{config_data['name']}' as default.")
        return {"message": f"'{config_data['name']}' set as default."}
    except Exception as e: 
        logger.error(f"Set default error: {e}"); raise HTTPException(500, f"Failed to set default: {e}")

# --- Main Prediction Endpoint ---
@app.post("/api/predict_full_simulation")
async def predict_full_simulation(payload: PredictionPayload):
    # ... (Keep existing 7-step implementation using await for async calls) ...
    logger.info("Entered full simulation endpoint.")
    try:
        is_valid, cycle_info = is_dag(payload.graph_structure);
        if not is_valid: raise HTTPException(400, f"Invalid graph: Not a DAG. {cycle_info or ''}")
        _, node_descriptions, _, all_nodes, _, _ = get_dynamic_node_info(payload.graph_structure)
        ui_features = payload.ui_features
        stage2_result = await call_openai_stage2_cognitive_states(payload.persona_inputs, ui_features, payload.graph_structure)
        cognitive_states = stage2_result.cognitive_state_probabilities; user_perception = stage2_result.user_perception_summary
        outcomes = calculate_outcome_nodes(payload.persona_inputs, cognitive_states)
        reasoning_context_probs = {**payload.persona_inputs.model_dump(by_alias=True), **cognitive_states.model_dump(by_alias=True), **outcomes}
        reasoning_text = await call_openai_stage4_reasoning(payload.persona_inputs, ui_features, cognitive_states, outcomes, payload.graph_structure, user_perception, "User performing task on provided UI")
        all_final_node_probs_p1 = {**{name: getattr(payload.persona_inputs, name) for name in payload.persona_inputs.model_fields}, **{name: getattr(cognitive_states, name) for name in cognitive_states.model_fields}, **outcomes}
        final_response_probs = {}; log_probs_p1 = {}
        for node_id in all_nodes:
            p1 = all_final_node_probs_p1.get(node_id, 0.5); p1_clamped = max(0.0, min(1.0, p1))
            final_response_probs[node_id] = {"0": 1.0 - p1_clamped, "1": p1_clamped, "description": node_descriptions.get(node_id, node_id)}; log_probs_p1[node_id] = p1_clamped
        response_payload = {"probabilities": final_response_probs, "ui_features": ui_features.dict(), "user_perception_summary": user_perception, "llm_reasoning": reasoning_text, "debug_context": {"persona_inputs": payload.persona_inputs.model_dump(by_alias=True), "cognitive_states_est": cognitive_states.model_dump(by_alias=True), "outcomes_calc": outcomes}}
        log_entry = LogPayload(timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'), configId=payload.config_id or "unknown", configName=payload.config_name or "Unknown", uiFeatures=ui_features.dict(), nodeProbabilities=log_probs_p1, userPerceptionSummary=user_perception, llmReasoning=reasoning_text)
        try: logger.info(f"Logging prediction for configId: {log_entry.configId}"); ui_feature_ids = list(ui_features.model_fields.keys()); await log_data_to_blob(log_entry, all_nodes, ui_feature_ids)
        except Exception as log_err: logger.error(f"Logging failed: {log_err}", exc_info=True)
        logger.info("Full simulation successful."); return response_payload
    except HTTPException as e: logger.error(f"HTTPException: {e.detail}"); raise e
    except Exception as e: logger.error(f"Full Simulation Error: {e}", exc_info=True); raise HTTPException(500, "Simulation failed.")

# --- Download Log Endpoint ---
@app.get("/api/download_log/{config_id}")
async def download_log_file(config_id: str):
    # ... (Keep existing implementation - checked for correctness) ...
    if not vercel_blob_token: raise HTTPException(500, "BLOB token missing")
    if not config_id or config_id in ["unknown", "default-config-001"]: raise HTTPException(400, "Invalid config ID for log download.")
    config_id_with_prefix = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"; log_filename = f"{LOG_FILENAME_PREFIX}{config_id_with_prefix}{LOG_FILENAME_SUFFIX}"; logger.info(f"Downloading log: {log_filename}");
    try:
        head_result = await blob_head(pathname=log_filename, options={'token': vercel_blob_token}); blob_url = head_result.get('url')
        if not blob_url: raise HTTPException(404, f"No URL for log {log_filename}")
        async def stream_content():
            try:
                async with aiohttp.ClientSession(timeout=ClientTimeout(total=20)) as session:
                    async with session.get(blob_url) as response:
                        if response.status == 200:
                            while True: 
                                chunk = await response.content.read(8192);
                                if not chunk: 
                                    break; 
                                yield chunk
                        else: raise HTTPException(response.status, f"Error fetching log {response.status}")
            except Exception as stream_err: logger.error(f"Streaming err: {stream_err}"); raise HTTPException(500, "Streaming fail")
        safe_filename_part = config_id_with_prefix.replace(CONFIG_KEY_PREFIX, ""); download_filename = f"log_{safe_filename_part}.csv"
        return StreamingResponse(stream_content(), media_type='text/csv', headers={'Content-Disposition': f'attachment; filename="{download_filename}"'})
    except Exception as e:
        err_str = str(e); is_404 = isinstance(e, HTTPException) and e.status_code == 404; not_found_kw = "NotFound" in err_str or "not found" in err_str.lower()
        if is_404 or not_found_kw : logger.warning(f"Log not found: {log_filename}"); raise HTTPException(404, f"Log not found for {config_id}")
        else: logger.error(f"Failed download: {e}", exc_info=True); raise HTTPException(500, f"Failed download log")

# --- Root Endpoint ---
@app.get("/")
async def root(): logger.info("Root accessed."); return {"message": "BN API v2 running."}

print("--- DEBUG: Finished defining routes. ---", flush=True)
