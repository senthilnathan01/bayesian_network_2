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
    from pydantic import BaseModel, Field, validator
    from openai import AsyncOpenAI # Use Async client
    from dotenv import load_dotenv
    import redis
    from redis.asyncio import ConnectionPool as AsyncConnectionPool, Redis as AsyncRedis # Use Async Redis
    from vercel_blob import put as blob_put, head as blob_head, delete as blob_delete, list as blob_list # Import Async Blob functions
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
# Updated headers for row-per-prediction CSV - dynamically determined later based on actual nodes + features
BASE_CSV_HEADERS = ['Timestamp', 'ConfigID', 'ConfigName']

DEFAULT_GRAPH_STRUCTURE = {
    "id": "default-config-001", # Keep a consistent ID for the default
    "name": "Default Cognitive Model",
    "graph_structure": {
        "nodes": [
            # Input Nodes (Persona)
            {"id": "A1", "fullName": "Domain Expertise", "nodeType": "input"},
            {"id": "A2", "fullName": "Web Literacy", "nodeType": "input"},
            {"id": "A3", "fullName": "Task Familiarity", "nodeType": "input"},
            {"id": "A4", "fullName": "Goal Clarity", "nodeType": "input"},
            {"id": "A5", "fullName": "Cognitive Capacity", "nodeType": "input"},
            {"id": "A6", "fullName": "Risk Aversion", "nodeType": "input"},
            {"id": "H", "fullName": "Recent History", "nodeType": "input"}, # Make sure H is in PersonaInputs model
            # Intermediate Nodes (Cognitive States)
            {"id": "IS1", "fullName": "Perceived UI Quality", "nodeType": "hidden"},
            {"id": "IS2", "fullName": "Task Understanding (Situational)", "nodeType": "hidden"},
            {"id": "IS3", "fullName": "Cognitive Load (Induced)", "nodeType": "hidden"},
            {"id": "IS4", "fullName": "Confidence / Self-Efficacy", "nodeType": "hidden"},
            {"id": "IS5", "fullName": "Attentional Focus", "nodeType": "hidden"},
            {"id": "IS6", "fullName": "Affective State", "nodeType": "hidden"},
            # Outcome Nodes (Styled as Hidden)
            {"id": "O1", "fullName": "Action Success Likelihood", "nodeType": "hidden"},
            {"id": "O2", "fullName": "Action Efficiency", "nodeType": "hidden"},
            {"id": "O3", "fullName": "Help Seeking / Abandonment Likelihood", "nodeType": "hidden"},
            {"id": "O4", "fullName": "Exploration Likelihood", "nodeType": "hidden"},
        ],
        "edges": [
            # Edges based on the refined BN dependencies defined previously
            # IS1 (Perc. UI Quality) <- A2, (Objective UI Features handled separately)
            {"source": "A2", "target": "IS1"},
            # IS2 (Task Understand.) <- A1, A4, IS1, (Objective UI Features)
            {"source": "A1", "target": "IS2"}, {"source": "A4", "target": "IS2"}, {"source": "IS1", "target": "IS2"},
            # IS3 (Cognitive Load) <- A2, A3, A5, IS1, IS2, (Objective UI Features)
            {"source": "A2", "target": "IS3"}, {"source": "A3", "target": "IS3"}, {"source": "A5", "target": "IS3"},
            {"source": "IS1", "target": "IS3"}, {"source": "IS2", "target": "IS3"},
            # IS4 (Confidence) <- A1, A3, A6, H, IS1, IS2, IS3
            {"source": "A1", "target": "IS4"}, {"source": "A3", "target": "IS4"}, {"source": "A6", "target": "IS4"},
            {"source": "H", "target": "IS4"}, {"source": "IS1", "target": "IS4"}, {"source": "IS2", "target": "IS4"},
            {"source": "IS3", "target": "IS4"},
            # IS5 (Atten. Focus) <- A4, A5, IS1, IS3, (Objective UI Features)
            {"source": "A4", "target": "IS5"}, {"source": "A5", "target": "IS5"}, {"source": "IS1", "target": "IS5"},
            {"source": "IS3", "target": "IS5"},
            # IS6 (Affective State) <- A6, H, IS1, IS3, IS4
            {"source": "A6", "target": "IS6"}, {"source": "H", "target": "IS6"}, {"source": "IS1", "target": "IS6"},
            {"source": "IS3", "target": "IS6"}, {"source": "IS4", "target": "IS6"},
            # O1 (Success Like.) <- IS2, IS4, IS5, IS3(-)
            {"source": "IS2", "target": "O1"}, {"source": "IS4", "target": "O1"}, {"source": "IS5", "target": "O1"},
            {"source": "IS3", "target": "O1"}, # Representing dependency, directionality handled in heuristic/LLM
            # O2 (Action Effic.) <- IS2, IS4, IS5, IS3(-)
            {"source": "IS2", "target": "O2"}, {"source": "IS4", "target": "O2"}, {"source": "IS5", "target": "O2"},
            {"source": "IS3", "target": "O2"},
            # O3 (Help/Abandon) <- IS2, IS4(-), IS6(-)
            {"source": "IS2", "target": "O3"}, {"source": "IS4", "target": "O3"}, {"source": "IS6", "target": "O3"},
             # O4 (Explore Like.) <- A6(-), IS4, IS5, IS6
            {"source": "A6", "target": "O4"}, {"source": "IS4", "target": "O4"}, {"source": "IS5", "target": "O4"},
            {"source": "IS6", "target": "O4"},
        ]
    }
}

# === Pydantic Models ===

class NodeData(BaseModel):
    id: str
    fullName: str = Field(...)
    nodeType: str = Field(...) # 'input' or 'hidden'
    @validator('nodeType')
    def valid_node_type(cls, v): 
        if v not in ['input', 'hidden']: 
            raise ValueError("Invalid nodeType"); 
        return v

class EdgeData(BaseModel): source: str; target: str
class GraphStructure(BaseModel):
    nodes: List[NodeData]
    edges: List[EdgeData]
    @validator('nodes')
    def nodes_unique(cls, v): 
        ids = [n.id for n in v]; 
        if len(ids) != len(set(ids)): 
            raise ValueError("Node IDs must be unique"); 
        return v
    @validator('edges')
    def edges_valid(cls, v, values):
        if 'nodes' not in values: return v
        node_ids = {n.id for n in values['nodes']}
        for e in v:
            if e.source not in node_ids or e.target not in node_ids: raise ValueError(f"Edge {e.source}->{e.target} invalid");
        return v

class UiFeatures(BaseModel):
    # Removed UI_Guidance
    Clarity: float = Field(..., ge=0.0, le=1.0)
    Familiarity: float = Field(..., ge=0.0, le=1.0)
    Findability: float = Field(..., ge=0.0, le=1.0)
    Complexity: float = Field(..., ge=0.0, le=1.0) # 0=Simple, 1=Complex
    AestheticTrust: float = Field(..., ge=0.0, le=1.0)

class CognitiveStateProbabilities(BaseModel):
    IS1: float = Field(..., alias="Perceived UI Quality", ge=0.0, le=1.0)
    IS2: float = Field(..., alias="Task Understanding (Situational)", ge=0.0, le=1.0)
    IS3: float = Field(..., alias="Cognitive Load (Induced)", ge=0.0, le=1.0)
    IS4: float = Field(..., alias="Confidence / Self-Efficacy", ge=0.0, le=1.0)
    IS5: float = Field(..., alias="Attentional Focus", ge=0.0, le=1.0)
    IS6: float = Field(..., alias="Affective State", ge=0.0, le=1.0)
    # Allow potential extra fields from LLM if needed, though strict parsing is better
    class Config: allow_population_by_field_name = True; extra = 'ignore'


class Stage2LLMResponse(BaseModel):
    cognitive_state_probabilities: CognitiveStateProbabilities
    user_perception_summary: str

class PersonaInputs(BaseModel):
    # Define expected persona inputs (A nodes, H)
    A1: float = Field(..., ge=0.0, le=1.0, alias="Domain Expertise")
    A2: float = Field(..., ge=0.0, le=1.0, alias="Web Literacy")
    A3: float = Field(..., ge=0.0, le=1.0, alias="Task Familiarity")
    A4: float = Field(..., ge=0.0, le=1.0, alias="Goal Clarity")
    A5: float = Field(..., ge=0.0, le=1.0, alias="Cognitive Capacity")
    A6: float = Field(..., ge=0.0, le=1.0, alias="Risk Aversion")
    H: Optional[float] = Field(0.5, ge=0.0, le=1.0, alias="Recent History") # Optional H node
    class Config: allow_population_by_field_name = True

class PredictionPayload(BaseModel):
    persona_inputs: PersonaInputs
    ui_features: UiFeatures # From stage 1
    graph_structure: GraphStructure # The BN definition
    config_id: Optional[str] = None # ID if saved, for logging
    config_name: Optional[str] = "Unsaved/Unknown" # Name for logging

class SaveConfigPayload(BaseModel):
    config_name: str = Field(..., min_length=1)
    graph_structure: GraphStructure

class LogPayload(BaseModel): # Used internally for blob logging format
    timestamp: str
    configId: str
    configName: str
    uiFeatures: Dict[str, float]
    nodeProbabilities: Dict[str, float] # All nodes (A, IS, O) with P(1)
    userPerceptionSummary: str
    llmReasoning: str

def get_dynamic_node_info(graph: GraphStructure) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], List[str], List[str], List[str]]:
    """ Parses graph structure to get node info needed for LLM prompts and logic. """
    node_parents: Dict[str, List[str]] = {}
    node_descriptions: Dict[str, str] = {}
    node_types: Dict[str, str] = {}
    nodes_set: Set[str] = set()

    for node in graph.nodes:
        if node and node.id:
            node_parents[node.id] = [] # Initialize parent list
            node_descriptions[node.id] = node.fullName
            node_types[node.id] = node.nodeType
            nodes_set.add(node.id)
        else:
            logger.warning(f"get_dynamic_node_info: Skipping invalid node object: {node}")

    for edge in graph.edges:
        # Ensure source and target nodes exist in our set before adding edge dependency
        if edge and edge.source and edge.target and edge.source in nodes_set and edge.target in nodes_set:
            # Check if target is already initialized (should be)
            if edge.target in node_parents:
                node_parents[edge.target].append(edge.source)
            else:
                 logger.warning(f"get_dynamic_node_info: Target node '{edge.target}' not found in keys while processing edge {edge.source}->{edge.target}")
        else:
            logger.warning(f"get_dynamic_node_info: Skipping invalid edge or edge with missing node: {edge}")


    all_nodes = sorted(list(nodes_set)) # Sort for consistency
    # Target nodes for probability estimation are 'hidden' nodes (includes IS and O nodes based on current model)
    target_nodes = sorted([node_id for node_id, node_type in node_types.items() if node_type == 'hidden'])
    # Input nodes are those defined as 'input' type
    input_nodes = sorted([node_id for node_id, node_type in node_types.items() if node_type == 'input'])

    # REMOVED unused cognitive_state_nodes and outcome_nodes definitions

    # Return only the necessary information
    return node_parents, node_descriptions, node_types, all_nodes, target_nodes, input_nodes

# --- Graph Validation ---
def is_dag(graph: GraphStructure) -> Tuple[bool, Optional[str]]:
    # ... (Keep existing is_dag function - it's correct) ...
    adj: Dict[str, List[str]] = {node.id: [] for node in graph.nodes}
    nodes_set = {node.id for node in graph.nodes}
    for edge in graph.edges:
        if edge.source not in nodes_set or edge.target not in nodes_set: return False, f"Edge refs invalid node: {edge.source}->{edge.target}"
        if edge.source not in adj: adj[edge.source] = [] # Ensure source key exists
        adj[edge.source].append(edge.target)
    path: Set[str] = set(); visited: Set[str] = set()
    def dfs(node: str) -> bool:
        path.add(node); visited.add(node)
        for neighbor in adj.get(node, []):
            if neighbor in path: return False # Cycle
            if neighbor not in visited:
                if not dfs(neighbor): return False
        path.remove(node); return True
    for node in nodes_set:
        if node not in visited:
            if not dfs(node): return False, f"Cycle detected near {node}"
    return True, None



# --- OpenAI Callers ---

async def call_openai_stage1_ui_features(image_base64: str, task_description: str) -> UiFeatures:
    """Stage 1: Analyze UI image for objective features."""
    if not openai_client: raise HTTPException(503, "OpenAI client not available.")
    logger.info(f"Stage 1: Analyzing UI for task: '{task_description[:50]}...'")

    system_prompt = """
    You are a UI/UX expert analyzing a webpage screenshot for a specific user task.
    Evaluate the UI features relevant to the task on a 0.0 (very poor/absent/complex) to 1.0 (excellent/present/simple) scale.
    Output ONLY a JSON object containing float values for keys: "Clarity", "Familiarity", "Findability", "Complexity", "AestheticTrust".
    - Clarity: Visual clarity for the task (hierarchy, labels).
    - Familiarity: Use of common web patterns/conventions for task elements.
    - Findability: Ease of locating necessary elements for the task.
    - Complexity: Perceived steps/elements for the task (0.0=Very Simple, 1.0=Very Complex).
    - AestheticTrust: Professionalism and trustworthiness of the visual design.
    """
    user_prompt = f"Analyze the UI in the image for the task: \"{task_description}\""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o", # Ensure this model supports vision
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}, # Assuming PNG, adjust if needed
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.1,
        )
        content = response.choices[0].message.content
        logger.info(f"Stage 1 LLM Raw Output: {content}")
        features_dict = json.loads(content)
        # Validate and return as Pydantic model
        return UiFeatures(**features_dict)

    except json.JSONDecodeError as e:
        logger.error(f"Stage 1 JSON Parse Error: {e}. Output: {content}")
        raise HTTPException(500, f"LLM (Stage 1) produced invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Stage 1 OpenAI Error: {e}", exc_info=True)
        raise HTTPException(502, f"OpenAI Stage 1 request failed: {e}")


async def call_openai_stage2_cognitive_states(persona: PersonaInputs, ui_features: UiFeatures, graph: GraphStructure) -> Stage2LLMResponse:
    """Stage 2: Estimate IS nodes based on persona and UI features."""
    if not openai_client: raise HTTPException(503, "OpenAI client not available.")
    logger.info("Stage 2: Estimating cognitive states...")

    # Format inputs for prompt
    persona_desc = "\n".join([f"- {f.alias or name}: {getattr(persona, name):.2f}" for name, f in persona.__fields__.items()])
    ui_features_desc = "\n".join([f"- {name}: {getattr(ui_features, name):.2f}" for name in ui_features.__fields__])

    # Extract node info needed for prompt
    node_parents, node_descriptions, _, _, target_nodes, _ = get_dynamic_node_info(graph)
    is_nodes = ["IS1", "IS2", "IS3", "IS4", "IS5", "IS6"] # Explicitly list expected IS nodes
    dependencies_desc = "\n".join([f"- {n} ({node_descriptions.get(n,n)}): <- {', '.join(node_parents.get(n,[])) or 'Inputs'}" for n in is_nodes])

    system_prompt = """
    You simulate a user's cognitive state interacting with a UI for a specific task.
    Inputs: User persona probabilities (0-1), Objective UI feature scores (0-1), and Network dependencies.
    Task: Estimate P(Node=1) for intermediate cognitive state nodes (IS1-IS6) based on the interaction between persona and UI features, following the network dependencies.
    Also provide a brief (2-3 sentence) summary of how this user might perceive/struggle with this UI for the task, starting with "User Perception:".
    Output ONLY a single JSON object with two keys:
    1. "cognitive_state_probabilities": An object mapping IS1-IS6 node IDs to their P(1) float values (0.0-1.0).
    2. "user_perception_summary": A string containing the perception summary.
    Example: {"cognitive_state_probabilities": {"IS1": 0.6, ...}, "user_perception_summary": "User Perception: ..."}
    """
    user_prompt = f"""
    User Persona (P=1):
    {persona_desc}

    Objective UI Feature Scores (0-1):
    {ui_features_desc}

    Cognitive Network Dependencies:
    {dependencies_desc}

    Estimate P(Node=1) for IS1, IS2, IS3, IS4, IS5, IS6 and provide the user perception summary in the specified JSON format.
    """
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            max_tokens=600,
            temperature=0.2,
        )
        content = response.choices[0].message.content
        logger.info(f"Stage 2 LLM Raw Output: {content}")
        response_dict = json.loads(content)

        # Validate using Pydantic model
        validated_response = Stage2LLMResponse(**response_dict)
        return validated_response

    except json.JSONDecodeError as e:
        logger.error(f"Stage 2 JSON Parse Error: {e}. Output: {content}")
        raise HTTPException(500, f"LLM (Stage 2) produced invalid JSON: {e}")
    except Exception as e: # Catches Pydantic validation errors too
        logger.error(f"Stage 2 OpenAI/Validation Error: {e}", exc_info=True)
        raise HTTPException(502, f"OpenAI Stage 2 request/validation failed: {e}")

# Stage 3: Heuristic calculation (can be expanded)
def calculate_outcome_nodes(persona: PersonaInputs, cognitive_states: CognitiveStateProbabilities) -> Dict[str, float]:
    """Stage 3: Calculate Outcome nodes based on IS nodes using heuristics."""
    logger.info("Stage 3: Calculating outcome nodes...")
    # Example simple heuristics (Weights need tuning!)
    # O1 (Success Like.) <- IS2(+), IS4(+), IS5(+), IS3(-)
    o1 = 0.3*cognitive_states.IS2 + 0.4*cognitive_states.IS4 + 0.3*cognitive_states.IS5 - 0.3*cognitive_states.IS3
    # O2 (Action Effic.) <- IS2(+), IS4(+), IS5(+), IS3(-)
    o2 = 0.2*cognitive_states.IS2 + 0.4*cognitive_states.IS4 + 0.4*cognitive_states.IS5 - 0.2*cognitive_states.IS3
    # O3 (Help/Abandon) <- IS2(-), IS4(-), IS6(-)  -- Note: Depends on IS6 (Affective) which might be negative state (0)
    o3 = 0.4*(1-cognitive_states.IS2) + 0.4*(1-cognitive_states.IS4) + 0.5*(1-cognitive_states.IS6)
    # O4 (Explore Like.) <- A6(-), IS4(+), IS5(+), IS6(+) -- Note: Depends on A6 (Risk Aversion) and positive IS6 state (1)
    o4 = -0.3*persona.A6 + 0.3*cognitive_states.IS4 + 0.2*cognitive_states.IS5 + 0.4*cognitive_states.IS6

    # Clamp results between 0 and 1 and add base offset if needed
    # Base offset can prevent results always being near zero if weights are small
    base = 0.1 # Small base chance
    outcomes = {
        "O1": max(0.0, min(1.0, base + o1)),
        "O2": max(0.0, min(1.0, base + o2)),
        "O3": max(0.0, min(1.0, base + o3)),
        "O4": max(0.0, min(1.0, base + o4)),
    }
    logger.info(f"Calculated Outcomes: {outcomes}")
    return outcomes


async def call_openai_stage4_reasoning(
    persona: PersonaInputs, ui_features: UiFeatures,
    cognitive_states: CognitiveStateProbabilities, outcomes: Dict[str, float],
    graph: GraphStructure, user_perception: str, task_description: str
    ) -> str:
    """Stage 4: Generate final reasoning explanation."""
    if not openai_client: return "Reasoning disabled: OpenAI client not available."
    logger.info("Stage 4: Generating reasoning...")

    # Format inputs
    persona_desc = "\n".join([f"- {f.alias or name}: {getattr(persona, name):.2f}" for name, f in persona.__fields__.items()])
    ui_features_desc = "\n".join([f"- {name}: {getattr(ui_features, name):.2f}" for name in ui_features.__fields__])
    cognitive_states_desc = "\n".join([f"- {name} ({f.alias or name}): {getattr(cognitive_states, name):.3f}" for name, f in cognitive_states.__fields__.items()])
    outcomes_desc = "\n".join([f"- {name}: {value:.3f}" for name, value in outcomes.items()])
    _, node_descriptions, _, _, _, _ = get_dynamic_node_info(graph) # Get descriptions

    system_prompt = """
    You are a UX analyst explaining a simulation of user cognition.
    Given: User Persona, Objective UI scores, Task, User Perception Summary, Estimated Cognitive States (IS1-6), Predicted Outcomes (O1-4).
    Task: Provide detailed reasoning, explaining:
    1. Briefly, how UI scores reflect the visual + task.
    2. How Persona *interacted* with UI scores -> User Perception & Cognitive States (IS1-6). Reference specific inputs.
    3. How Cognitive States -> Predicted Outcomes (O1-4). Reference specific IS nodes.
    Be specific and link causes to effects based on standard UX principles. Use bullet points or structured paragraphs.
    """
    user_prompt = f"""
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
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=1000, # Allow longer reasoning
            temperature=0.4,
        )
        reasoning_text = response.choices[0].message.content.strip()
        logger.info(f"Stage 4 Reasoning Output received.")
        return reasoning_text
    except Exception as e:
        logger.error(f"Stage 4 Reasoning Error: {e}", exc_info=True)
        return f"Could not generate full reasoning: {e}"


# --- Blob Logging (Modified for new format) ---
async def log_data_to_blob(log_data: LogPayload, all_node_ids: List[str], ui_feature_ids: List[str]):
    """Appends log data (one row per prediction) to a CSV file in Vercel Blob."""
    if not vercel_blob_token: logger.warning("Blob token missing, skip log."); return
    if not log_data.configId or log_data.configId in ["unknown", "default-config-001", "unsaved"]: logger.info(f"Skipping log for configId: {log_data.configId}"); return

    log_filename = f"{LOG_FILENAME_PREFIX}{log_data.configId}{LOG_FILENAME_SUFFIX}"
    logger.info(f"Attempting log append to: {log_filename}")

    # --- Prepare Header and Data Row ---
    # Ensure consistent order: Base info, UI features (sorted), Nodes (sorted)
    sorted_ui_feature_ids = sorted(ui_feature_ids)
    sorted_node_ids = sorted(all_node_ids)
    headers = BASE_CSV_HEADERS + sorted_ui_feature_ids + sorted_node_ids + ["UserPerceptionSummary", "LLMReasoning"]

    data_row = [ log_data.timestamp, log_data.configId, log_data.configName ]
    # Add UI features
    for feature_id in sorted_ui_feature_ids:
        data_row.append(f"{log_data.uiFeatures.get(feature_id, ''):.4f}" if isinstance(log_data.uiFeatures.get(feature_id), float) else '')
    # Add Node probabilities
    for node_id in sorted_node_ids:
        data_row.append(f"{log_data.nodeProbabilities.get(node_id, ''):.4f}" if isinstance(log_data.nodeProbabilities.get(node_id), float) else '')
    # Add text fields (handle potential commas/newlines later with CSV writer)
    data_row.append(log_data.userPerceptionSummary)
    data_row.append(log_data.llmReasoning)


    # --- Handle File Write ---
    try:
        existing_content = b""
        needs_headers = False
        try:
            logger.debug(f"Checking blob: {log_filename}")
            head_result = await blob_head(pathname=log_filename, options={'token': vercel_blob_token})
            logger.debug(f"Blob exists, fetching URL: {head_result.get('url')}")
            blob_url = head_result.get('url')
            if not blob_url: raise Exception("Blob URL missing from head result")
            # Use aiohttp to download content
            async with aiohttp.ClientSession(timeout=ClientTimeout(total=10)) as session: # Increased timeout
                async with session.get(blob_url) as response:
                    if response.status == 200:
                        existing_content = await response.read()
                        if existing_content.endswith(b'\n'): existing_content = existing_content[:-1]
                        logger.debug(f"Read {len(existing_content)} bytes from existing log.")
                    elif response.status == 404: # Explicit 404 check
                         needs_headers = True
                         logger.info(f"Log file URL gave 404, creating new.")
                    else:
                         raise Exception(f"HTTP Error {response.status} fetching existing log.")
        except Exception as head_or_download_error:
            logger.info(f"Log file {log_filename} not found or download error ({head_or_download_error}). Creating new.")
            needs_headers = True

        # Use StringIO and csv.writer for proper CSV formatting
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL) # Use minimal quotes

        if existing_content:
             # Write existing content WITHOUT decoding/re-encoding if possible,
             # but decoding is safer if encoding isn't guaranteed UTF-8.
             # Assuming UTF-8 for now.
             output.write(existing_content.decode('utf-8'))
             # Add newline ONLY if existing content doesn't end with one (it was stripped)
             output.write('\n')

        if needs_headers:
            writer.writerow(headers) # Write dynamic headers

        writer.writerow(data_row) # Write the new data

        final_csv_content = output.getvalue().encode('utf-8')
        output.close()

        logger.debug(f"Uploading {len(final_csv_content)} bytes to {log_filename}...")
        put_result = await blob_put(
            pathname=log_filename, body=final_csv_content,
            options={'access': 'public', 'add_random_suffix': False, 'contentType': 'text/csv', 'token': vercel_blob_token}
        )
        logger.info(f"Log write successful: {log_filename}")

    except Exception as e:
        logger.error(f"FAILED to log to blob {log_filename}: {e}", exc_info=True)
        # Do not re-raise, logging failure shouldn't break prediction

# === API Endpoints ===

@app.get("/api/ping")
async def ping(): # ... (keep existing ping) ...
    redis_status = "disabled"
    if redis_client:
        try: await redis_client.ping(); redis_status = "connected" # Use await for async redis
        except Exception: redis_status = "disconnected"
    return {"message": "pong", "redis_status": redis_status}

# --- Stage 1 Endpoint ---
@app.post("/api/extract_ui_features", response_model=UiFeatures)
async def extract_ui_features(task_description: str = Form(...), image: UploadFile = File(...)):
    """Receives an image and task, returns objective UI feature scores from LLM."""
    logger.info(f"Received UI analysis request for task: {task_description}")
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid file type. Please upload an image.")
    if not openai_client:
        raise HTTPException(503, "OpenAI client not available.")

    try:
        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        features = await call_openai_stage1_ui_features(image_base64, task_description)
        return features
    except HTTPException as e:
        raise e # Re-raise HTTP exceptions from LLM calls
    except Exception as e:
        logger.error(f"Error processing UI image: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to analyze UI image: {e}")

# --- Config Management Endpoints ---
# (Use AsyncRedis methods: await redis_client.get(...), etc.)
@app.get("/api/configs/default", response_model=Dict[str, Any])
async def get_default_configuration(): # ... (keep existing logic, use await) ...
    logger.info("Serving default configuration.")
    if redis_client:
        try:
            if not await redis_client.ping(): raise Exception("Redis ping failed") # Check connection
            default_id = await redis_client.get(DEFAULT_CONFIG_KEY)
            if default_id:
                config_json = await redis_client.get(default_id)
                if config_json: return json.loads(config_json)
        except Exception as e: logger.error(f"Redis error fetching default: {e}", exc_info=True)
    logger.info("Returning hardcoded default configuration.")
    return DEFAULT_GRAPH_STRUCTURE

@app.post("/api/configs", status_code=201)
async def save_configuration(payload: SaveConfigPayload): # ... (use await redis_client.set) ...
    if not redis_client: raise HTTPException(503, "Storage unavailable.")
    is_valid, cycle_info = is_dag(payload.graph_structure);
    if not is_valid: raise HTTPException(400, f"Invalid graph: Not a DAG. {cycle_info or ''}")
    config_id = f"{CONFIG_KEY_PREFIX}{uuid4()}"; config_data = {"id": config_id, "name": payload.config_name, "graph_structure": payload.graph_structure.dict()}
    try: await redis_client.set(config_id, json.dumps(config_data)); logger.info(f"Saved: {config_id}"); return {"id": config_id, "name": payload.config_name}
    except Exception as e: logger.error(f"Redis save error: {e}", exc_info=True); raise HTTPException(500, f"Save failed: {e}")

@app.get("/api/configs", response_model=List[Dict[str, str]])
async def list_configurations(): # ... (use await redis_client.scan_iter, await redis_client.mget) ...
    if not redis_client: return []
    configs_summary = []; keys = [];
    try: 
        async for key in redis_client.scan_iter(match=f"{CONFIG_KEY_PREFIX}*"): 
            keys.append(key) # Collect keys first
    except Exception as e: logger.error(f"Redis scan error: {e}"); return []
    if not keys: return []
    try: config_jsons = await redis_client.mget(keys) # Fetch all at once
    except Exception as e: logger.error(f"Redis mget error: {e}"); return [{"id": k, "name":"*Load Error*"} for k in keys] # Indicate load error per key
    for key, config_json in zip(keys, config_jsons):
        if config_json:
            try: data = json.loads(config_json); configs_summary.append({"id": data.get("id", key), "name": data.get("name", "Unnamed")})
            except Exception: configs_summary.append({"id": key, "name": "*Parse Error*"})
        else: configs_summary.append({"id": key, "name": "*Not Found*"}) # Should be rare after scan
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
    try: deleted_count = await redis_client.delete(config_id_with_prefix) # Delete from Redis
    except Exception as e: logger.error(f"Redis delete error: {e}"); raise HTTPException(500,"Storage delete error (Redis)")
    if deleted_count == 0: raise HTTPException(404, "Config not found in Redis.")
    try: await blob_delete(pathname=log_filename, options={'token': vercel_blob_token}) # Delete log (best effort)
    except Exception as blob_error: logger.warning(f"Failed to delete log file '{log_filename}': {blob_error}")
    # Check if deleted config was the default
    try:
        current_default_id = await redis_client.get(DEFAULT_CONFIG_KEY)
        if current_default_id == config_id_with_prefix:
            await redis_client.delete(DEFAULT_CONFIG_KEY)
            logger.info(f"Removed deleted config {config_id_with_prefix} as default.")
    except Exception as e: logger.warning(f"Could not check/remove default config setting: {e}")
    logger.info(f"Deleted config '{config_id_with_prefix}'.")
    return {"message": "Configuration deleted successfully."}

@app.post("/api/configs/set_default", status_code=200)
async def set_default_configuration(config_id: str = Body(...)): # ... (use await redis_client.get, set) ...
    if not redis_client: raise HTTPException(503, "Storage unavailable.")
    config_id_with_prefix = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
    try:
        config_json = await redis_client.get(config_id_with_prefix) # Check if target config exists
        if config_json is None: raise HTTPException(404, "Configuration to set as default not found.")
        await redis_client.set(DEFAULT_CONFIG_KEY, config_id_with_prefix) # Store the ID in the default key
        config_data = json.loads(config_json)
        logger.info(f"Set '{config_data['name']}' as default config.")
        return {"message": f"Configuration '{config_data['name']}' set as default."}
    except Exception as e: logger.error(f"Error setting default: {e}"); raise HTTPException(500, f"Failed to set default: {e}")

# --- Main Prediction Endpoint (Modified for 4 Stages) ---
@app.post("/api/predict_full_simulation") # Changed route name for clarity
async def predict_full_simulation(payload: PredictionPayload):
    """Runs the full 4-stage simulation."""
    logger.info("Entered full simulation endpoint.")
    try:
        # 0. Validate Graph
        is_valid, cycle_info = is_dag(payload.graph_structure)
        if not is_valid: raise HTTPException(400, f"Invalid graph: Not a DAG. {cycle_info or ''}")

        # Get node info needed throughout
        _, node_descriptions, _, all_nodes, _, _ = get_dynamic_node_info(payload.graph_structure)

        # 1. Stage 1 (UI Features) - Already done by client, passed in payload.ui_features
        ui_features = payload.ui_features

        # 2. Stage 2 (Cognitive States + User Perception)
        stage2_result = await call_openai_stage2_cognitive_states(payload.persona_inputs, ui_features, payload.graph_structure)
        cognitive_states = stage2_result.cognitive_state_probabilities
        user_perception = stage2_result.user_perception_summary

        # 3. Stage 3 (Outcome Nodes)
        outcomes = calculate_outcome_nodes(payload.persona_inputs, cognitive_states)

        # 4. Stage 4 (Reasoning)
        reasoning_context_probs = { # Combine all probabilities for reasoning context
            **payload.persona_inputs.dict(by_alias=True), # Use aliases like Domain Expertise
            **cognitive_states.dict(by_alias=True), # Use aliases
            **outcomes
        }
        reasoning_text = await call_openai_stage4_reasoning(
            payload.persona_inputs, ui_features, cognitive_states, outcomes,
            payload.graph_structure, user_perception, "User performing task on provided UI" # Generic task desc for now
        )

        # 5. Combine All Node Probabilities for Response & Logging
        all_final_node_probs_p1 = {
            **{name: getattr(payload.persona_inputs, name) for name in payload.persona_inputs.__fields__}, # A nodes + H
            **{name: getattr(cognitive_states, name) for name in cognitive_states.__fields__}, # IS nodes
            **outcomes # O nodes
        }
        # Format for frontend { "ID": {"0": ..., "1": ...}, ...}
        final_response_probs = {}
        for node_id in all_nodes:
            p1 = all_final_node_probs_p1.get(node_id, 0.5) # Default if missing (shouldn't happen)
            p1_clamped = max(0.0, min(1.0, p1))
            final_response_probs[node_id] = {"0": 1.0 - p1_clamped, "1": p1_clamped, "description": node_descriptions.get(node_id, node_id)}

        # 6. Prepare Log Payload
        log_entry = LogPayload(
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            configId=payload.config_id or "unknown",
            configName=payload.config_name or "Unknown",
            uiFeatures=ui_features.dict(),
            nodeProbabilities=all_final_node_probs_p1, # Log the P(1) values
            userPerceptionSummary=user_perception,
            llmReasoning=reasoning_text
        )
        try:
            ui_feature_ids = list(ui_features.__fields__.keys())
            await log_data_to_blob(log_entry, all_nodes, ui_feature_ids) # Pass node/feature lists for headers
        except Exception as log_err:
            logger.error(f"Logging failed: {log_err}", exc_info=True) # Log error but continue

        # 7. Return Full Response
        logger.info("Full simulation successful.")
        return {
            "probabilities": final_response_probs, # For graph display
            "ui_features": ui_features.dict(), # For display
            "user_perception_summary": user_perception, # For display
            "llm_reasoning": reasoning_text, # For display
            # Optionally include raw context sent to LLM reasoning for debug/display
            "debug_context": {
                 "persona_inputs": payload.persona_inputs.dict(by_alias=True),
                 "cognitive_states_est": cognitive_states.dict(by_alias=True),
                 "outcomes_calc": outcomes,
            }
        }

    except HTTPException as e: logger.error(f"HTTPException: {e.detail}"); raise e
    except Exception as e: logger.error(f"Full Simulation Error: {e}", exc_info=True); raise HTTPException(500, "Simulation failed.")

# --- Download Log Endpoint ---
@app.get("/api/download_log/{config_id}")
async def download_log_file(config_id: str):
    # ... (Keep existing download logic - it reads the CSV as is) ...
    if not vercel_blob_token: raise HTTPException(status_code=500, detail="BLOB_READ_WRITE_TOKEN not configured")
    if not config_id or config_id == "unknown" or config_id == "default-config-001": raise HTTPException(status_code=400, detail="Cannot download logs for unsaved/default configs.")
    config_id_with_prefix = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
    log_filename = f"{LOG_FILENAME_PREFIX}{config_id_with_prefix}{LOG_FILENAME_SUFFIX}"
    logger.info(f"Downloading log: {log_filename}")
    try:
        head_result = await blob_head(pathname=log_filename, options={'token': vercel_blob_token})
        blob_url = head_result.get('url')
        if not blob_url: raise HTTPException(status_code=404, detail=f"No URL for log file {log_filename}")
        async def stream_content():
            try:
                async with aiohttp.ClientSession(timeout=ClientTimeout(total=20)) as session: # Longer timeout for download
                    async with session.get(blob_url) as response:
                        if response.status == 200:
                            while True:
                                chunk = await response.content.read(8192);
                                if not chunk: break; yield chunk
                        else: raise HTTPException(status_code=response.status, detail=f"Error fetching log content.")
            except Exception as stream_err: logger.error(f"Streaming error: {stream_err}"); raise HTTPException(500, "Streaming failed")
        safe_filename_part = config_id_with_prefix.replace(CONFIG_KEY_PREFIX, "")
        download_filename = f"log_{safe_filename_part}.csv"
        return StreamingResponse(stream_content(), media_type='text/csv', headers={'Content-Disposition': f'attachment; filename="{download_filename}"'})
    except Exception as e:
        err_str = str(e); is_404 = isinstance(e, HTTPException) and e.status_code == 404
        if "NotFound" in err_str or "not found" in err_str.lower() or is_404: logger.warning(f"Log file not found: {log_filename}"); raise HTTPException(status_code=404, detail=f"Log file not found for config ID {config_id}")
        else: logger.error(f"Failed download: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Failed download log file.")

# --- Root Endpoint ---
@app.get("/")
async def root(): logger.info("Root endpoint accessed."); return {"message": "BN API v2 (UI Analysis) is running."}

print("--- DEBUG: Finished defining routes. ---", flush=True)
