import sys  # Provides access to system-specific parameters and functions
import os  # Provides functions for interacting with the operating system
import json  # Provides methods for working with JSON data
import logging  # Provides a flexible framework for emitting log messages
from typing import Dict, Any, List, Optional, Set, Tuple  # Provides type hints for better code clarity and type checking
from uuid import uuid4  # Generates universally unique identifiers (UUIDs)
import urllib.parse  # Provides functions for parsing and manipulating URLs
from datetime import datetime, timezone  # Provides classes for working with dates and times, including timezone support
import io  # Provides tools for working with streams (e.g., in-memory file-like objects)
import csv  # Provides tools for reading and writing CSV files
import aiohttp  # Provides asynchronous HTTP client functionality
from aiohttp import ClientTimeout  # Allows setting timeouts for HTTP requests

print("--- DEBUG: api/main.py TOP LEVEL EXECUTION ---", flush=True)

try:
    from fastapi import FastAPI, HTTPException, Body, Response  # FastAPI framework for building APIs, HTTP exceptions, and request/response handling
    from fastapi.middleware.cors import CORSMiddleware  # Middleware to enable Cross-Origin Resource Sharing (CORS)
    from fastapi.responses import StreamingResponse  # Provides support for streaming responses
    from pydantic import BaseModel, Field, validator  # Pydantic for data validation and settings management
    from openai import AsyncOpenAI  # OpenAI client for interacting with OpenAI APIs
    from dotenv import load_dotenv  # Loads environment variables from a .env file
    import redis  # Redis client for interacting with a Redis database
    from redis.connection import ConnectionPool  # Manages connection pooling for Redis
    from vercel_blob import put as blob_put, head as blob_head, delete as blob_delete  # Vercel Blob API for file storage and management
    print("--- DEBUG: Imported FastAPI, CORS, Pydantic, OpenAI, Redis, Vercel Blob, aiohttp ---", flush=True)

    app = FastAPI(title="Dynamic Bayesian Network API")
    print("--- DEBUG: FastAPI app created ---", flush=True)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
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
redis_url = os.getenv("KV_URL") or os.getenv("REDIS_URL")
vercel_blob_token = os.getenv("BLOB_READ_WRITE_TOKEN")

# Initialize async OpenAI client
openai_client = None
if openai_api_key:
    openai_client = AsyncOpenAI(api_key=openai_api_key)
else:
    logger.warning("OPENAI_API_KEY environment variable not found. Prediction endpoints will fail.")

# Redis Connection with Connection Pool
redis_client = None
redis_pool = None
if redis_url:
    try:
        redis_pool = ConnectionPool.from_url(redis_url, decode_responses=True)
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_client.ping()
        logger.info("--- Successfully connected to Redis ---")
    except redis.ConnectionError as e:
        logger.error(f"--- Failed to connect to Redis: {e} ---", exc_info=True)
        redis_client = None
        redis_pool = None
else:
    logger.warning("--- Redis URL (KV_URL or REDIS_URL) not found. Config storage disabled. ---")

def reconnect_redis():
    global redis_client, redis_pool
    if redis_url and not redis_client:
        try:
            redis_pool = ConnectionPool.from_url(redis_url, decode_responses=True)
            redis_client = redis.Redis(connection_pool=redis_pool)
            redis_client.ping()
            logger.info("--- Reconnected to Redis ---")
        except redis.ConnectionError as e:
            logger.error(f"--- Failed to reconnect to Redis: {e} ---", exc_info=True)
            redis_client = None
            redis_pool = None

if not vercel_blob_token:
    logger.warning("BLOB_READ_WRITE_TOKEN environment variable not found. Blob storage will fail.")
if not redis_url:
    logger.warning("No Redis URL provided. Configuration storage will use hardcoded defaults.")

# Constants
CONFIG_KEY_PREFIX = "bn_config:"
DEFAULT_CONFIG_KEY = "bn_default_config"
LOG_FILENAME_PREFIX = "bn_log_"
LOG_FILENAME_SUFFIX = ".csv"
CSV_HEADERS = ['Timestamp', 'ConfigID', 'ConfigName', 'NodeID', 'ProbabilityP1']

# Default Graph Structure
DEFAULT_GRAPH_STRUCTURE = {
    "id": "default-config-001",
    "name": "Default Cognitive Model",
    "graph_structure": {
        "nodes": [
            {"id": "A1", "fullName": "Domain Expertise", "nodeType": "input"},
            {"id": "A2", "fullName": "Web Literacy", "nodeType": "input"},
            {"id": "A3", "fullName": "Task Familiarity", "nodeType": "input"},
            {"id": "A4", "fullName": "Goal Clarity", "nodeType": "input"},
            {"id": "A5", "fullName": "Motivation", "nodeType": "input"},
            {"id": "UI", "fullName": "UI State (Quality/Clarity)", "nodeType": "input"},
            {"id": "H", "fullName": "History (Relevant past interactions)", "nodeType": "input"},
            {"id": "IS3", "fullName": "Task Understanding", "nodeType": "hidden"},
            {"id": "IS4", "fullName": "Interaction Fluency", "nodeType": "hidden"},
            {"id": "IS5", "fullName": "Relevant Knowledge Activation", "nodeType": "hidden"},
            {"id": "IS2", "fullName": "Cognitive Load", "nodeType": "hidden"},
            {"id": "IS1", "fullName": "Confidence", "nodeType": "hidden"},
            {"id": "O1", "fullName": "Predicted Success Probability", "nodeType": "hidden"},
            {"id": "O2", "fullName": "Action Speed/Efficiency", "nodeType": "hidden"},
            {"id": "O3", "fullName": "Help Seeking Likelihood", "nodeType": "hidden"},
        ],
        "edges": [
            {"source": "A1", "target": "IS3"}, {"source": "A3", "target": "IS3"},
            {"source": "A4", "target": "IS3"}, {"source": "H", "target": "IS3"},
            {"source": "A2", "target": "IS4"}, {"source": "UI", "target": "IS4"},
            {"source": "H", "target": "IS4"}, {"source": "A1", "target": "IS5"},
            {"source": "A3", "target": "IS5"}, {"source": "IS3", "target": "IS5"},
            {"source": "IS4", "target": "IS2"}, {"source": "IS3", "target": "IS2"},
            {"source": "A5", "target": "IS1"}, {"source": "IS2", "target": "IS1"},
            {"source": "IS3", "target": "IS1"}, {"source": "IS4", "target": "IS1"},
            {"source": "IS5", "target": "IS1"}, {"source": "IS1", "target": "O1"},
            {"source": "IS2", "target": "O1"}, {"source": "IS3", "target": "O1"},
            {"source": "IS5", "target": "O1"}, {"source": "IS1", "target": "O2"},
            {"source": "IS2", "target": "O2"}, {"source": "A5", "target": "O2"},
            {"source": "IS1", "target": "O3"}, {"source": "IS2", "target": "O3"},
            {"source": "IS3", "target": "O3"}
        ]
    }
}

# Pydantic Models
class NodeData(BaseModel):
    id: str
    fullName: str = Field(..., description="User-friendly name for the node")
    nodeType: str = Field(..., description="Type of node: 'input', 'hidden'")

    @validator('nodeType')
    def node_type_must_be_valid(cls, v):
        if v not in ['input', 'hidden']:
            raise ValueError("nodeType must be 'input' or 'hidden'")
        return v

class EdgeData(BaseModel):
    source: str
    target: str

class GraphStructure(BaseModel):
    nodes: List[NodeData]
    edges: List[EdgeData]

    @validator('nodes')
    def nodes_must_be_unique(cls, v):
        ids = [node.id for node in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Node IDs must be unique")
        return v

    @validator('edges')
    def edges_must_reference_valid_nodes(cls, v, values):
        if 'nodes' not in values:
            return v
        node_ids = {node.id for node in values['nodes']}
        for edge in v:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(f"Edge {edge.source}->{edge.target} references invalid node")
        return v

class ContinuousUserInput(BaseModel):
    input_values: Dict[str, float]

class PredictionPayload(ContinuousUserInput):
    graph_structure: GraphStructure
    config_id: Optional[str] = None
    config_name: Optional[str] = "Unknown"

class SaveConfigPayload(BaseModel):
    config_name: str = Field(..., min_length=1, description="User-provided name for the configuration")
    graph_structure: GraphStructure

class LogPayload(BaseModel):
    timestamp: str
    configId: str
    configName: str
    probabilities: Dict[str, float]

# Graph Validation
def is_dag(graph: GraphStructure) -> Tuple[bool, Optional[str]]:
    adj: Dict[str, List[str]] = {node.id: [] for node in graph.nodes}
    nodes_set = {node.id for node in graph.nodes}
    for edge in graph.edges:
        if edge.source not in nodes_set or edge.target not in nodes_set:
            return False, f"Edge references non-existent node: {edge.source} -> {edge.target}"
        if edge.target not in adj:
            adj[edge.target] = []
        adj[edge.source].append(edge.target)

    path: Set[str] = set()
    visited: Set[str] = set()

    def dfs(node: str) -> bool:
        path.add(node)
        visited.add(node)
        for neighbor in adj.get(node, []):
            if neighbor in path:
                return False
            if neighbor not in visited:
                if not dfs(neighbor):
                    return False
        path.remove(node)
        return True

    for node in nodes_set:
        if node not in visited:
            if not dfs(node):
                return False, f"Cycle detected involving node path near {node}"
    return True, None

# Helper Functions
def get_dynamic_node_info(graph: GraphStructure):
    node_parents = {node.id: [] for node in graph.nodes}
    node_descriptions = {node.id: node.fullName for node in graph.nodes}
    node_types = {node.id: node.nodeType for node in graph.nodes}
    nodes_set = {node.id for node in graph.nodes}

    for edge in graph.edges:
        if edge.source in nodes_set and edge.target in nodes_set:
            if edge.target in node_parents:
                node_parents[edge.target].append(edge.source)

    all_nodes = list(nodes_set)
    target_nodes = [node_id for node_id, node_type in node_types.items() if node_type == 'hidden']
    input_nodes = [node_id for node_id, node_type in node_types.items() if node_type == 'input']

    return node_parents, node_descriptions, node_types, all_nodes, target_nodes, input_nodes

async def call_openai_for_probabilities(input_states: Dict[str, float], graph: GraphStructure) -> Dict[str, float]:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    node_parents, node_descriptions, _, _, target_nodes, input_nodes = get_dynamic_node_info(graph)

    missing_inputs = [node for node in input_nodes if node not in input_states]
    if missing_inputs:
        raise HTTPException(status_code=400, detail=f"Missing input values for nodes: {', '.join(missing_inputs)}")

    input_desc_list = []
    for node, value in input_states.items():
        if node in input_nodes:
            state_desc = "High" if value >= 0.66 else ("Medium" if value >= 0.33 else "Low")
            input_desc_list.append(f"- {node} ({node_descriptions.get(node, node)}): {state_desc} (P(1)={value:.2f})")
    input_context = "\n".join(input_desc_list)

    if not target_nodes:
        return {}

    # Simplified node info for prompt
    node_info = []
    for node_id in target_nodes:
        parents = node_parents.get(node_id, [])
        parent_desc = ", ".join([node_descriptions.get(p, p) for p in parents]) if parents else "none"
        node_info.append(f"- {node_id} ({node_descriptions[node_id]}): influenced by {parent_desc}")
    structure_description = f"Nodes:\n{ '\n'.join(node_info) }"

    system_message = """
    You are an expert probabilistic reasoner modeling a Bayesian Network.
    Given a Directed Acyclic Graph (DAG) with binary nodes (states 0 or 1), estimate P(Node=1) for hidden nodes based on input node probabilities.
    Use node descriptions and dependencies to infer probabilities.
    Return a JSON object mapping each hidden node to P(Node=1) (float, 0.0 to 1.0).
    Example: {"HiddenNode1": 0.7, "HiddenNode2": 0.4}
    Output ONLY the JSON object.
    """
    user_message = f"""
    Input Probabilities (P=1):
    {input_context}

    Network Structure:
    {structure_description}

    Estimate P(Node=1) for hidden nodes ({', '.join(target_nodes)}).
    """

    logger.debug("Sending probability prompt to OpenAI...")
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.1,
            n=1
        )
        llm_output_raw = response.choices[0].message.content.strip()
        logger.info(f"OpenAI Probability Output: {llm_output_raw}")
        estimated_probs = json.loads(llm_output_raw)
        validated_probs = {}
        for node in target_nodes:
            if node in estimated_probs and isinstance(estimated_probs[node], (float, int)):
                validated_probs[node] = max(0.0, min(1.0, float(estimated_probs[node])))
            else:
                logger.warning(f"Node '{node}' missing/invalid. Using default 0.5.")
                validated_probs[node] = 0.5
        return validated_probs
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON: {e}")
        raise HTTPException(status_code=500, detail=f"LLM JSON parse error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in OpenAI probability call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")

async def call_openai_for_reasoning(input_states: Dict[str, float], graph: GraphStructure, estimated_probabilities: Dict[str, float]) -> str:
    if not openai_client:
        return "Reasoning disabled: OpenAI API key not configured."

    node_parents, node_descriptions, _, _, target_nodes, _ = get_dynamic_node_info(graph)

    input_desc_list = []
    for node, value in input_states.items():
        state_desc = "High" if value >= 0.66 else ("Medium" if value >= 0.33 else "Low")
        input_desc_list.append(f"- {node} ({node_descriptions.get(node, node)}): {state_desc} (P(1)={value:.2f})")
    input_context = "\n".join(input_desc_list)

    probs_text = "\n".join([f"- {node} ({node_descriptions.get(node, node)}): P(1)={prob:.3f}" for node, prob in estimated_probabilities.items()])

    system_message = """
    You are an expert analyst explaining a Bayesian Network simulation.
    Given input probabilities, hidden node probabilities, and dependencies, explain why each hidden node received its P(Node=1).
    Focus on parent nodes' influence. Use a concise list format.
    Example:
    - Node IS1 (Confidence): P(1)=0.7 due to high Motivation and moderate Task Understanding.
    """
    user_message = f"""
    Inputs (P=1):
    {input_context}

    Estimated Probabilities:
    {probs_text}

    Explain why each hidden node ({', '.join(target_nodes)}) received its probability.
    """

    logger.debug("Sending reasoning prompt to OpenAI...")
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=800,
            temperature=0.3,
            n=1
        )
        reasoning_text = response.choices[0].message.content.strip()
        logger.info(f"OpenAI Reasoning Output received.")
        return reasoning_text
    except Exception as e:
        logger.error(f"Error in reasoning call: {e}", exc_info=True)
        return f"Could not generate reasoning: {str(e)}"

# API Endpoints
@app.get("/api/ping")
async def ping():
    redis_status = "disabled"
    if redis_client:
        try:
            redis_client.ping()
            redis_status = "connected"
        except redis.ConnectionError:
            redis_status = "disconnected"
    return {"message": "pong", "redis_status": redis_status}

@app.get("/api/vercel_blob_test")
async def vercel_blob_test():
    if not vercel_blob_token:
        raise HTTPException(status_code=500, detail="BLOB_READ_WRITE_TOKEN not configured")
    try:
        test_filename = "test_blob.csv"
        test_content = "Test,Timestamp,Data\n1,2023-10-01T00:00:00Z,TestData".encode('utf-8')
        put_result = await blob_put(
            pathname=test_filename,
            body=test_content,
            options={'access': 'public', 'add_random_suffix': False, 'content_type': 'text/csv', 'token': vercel_blob_token}
        )
        async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as session:
            async with session.get(put_result['url']) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch blob: HTTP {response.status}")
                content = await response.read()
        await blob_delete(pathname=test_filename, options={'token': vercel_blob_token})
        return {"status": "success", "content": content.decode('utf-8')}
    except Exception as e:
        logger.error(f"Vercel Blob test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vercel Blob test failed: {str(e)}")

@app.get("/api/configs/default", response_model=Dict[str, Any])
async def get_default_configuration():
    logger.info("Serving default configuration.")
    if redis_client:
        try:
            reconnect_redis()  # Attempt to reconnect if connection is lost
            if redis_client:
                default_id = redis_client.get(DEFAULT_CONFIG_KEY)
                if default_id:
                    config_json = redis_client.get(default_id)
                    if config_json:
                        return json.loads(config_json)
        except redis.RedisError as e:
            logger.error(f"Error fetching default config from Redis: {e}", exc_info=True)
    logger.info("Returning hardcoded default configuration due to Redis unavailability.")
    return DEFAULT_GRAPH_STRUCTURE

@app.post("/api/configs", status_code=201)
async def save_configuration(payload: SaveConfigPayload):
    if not redis_client:
        reconnect_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Storage connection unavailable.")
    is_valid_dag, cycle_info = is_dag(payload.graph_structure)
    if not is_valid_dag:
        raise HTTPException(status_code=400, detail=f"Invalid graph: Not a DAG. {cycle_info or ''}")

    config_id = f"{CONFIG_KEY_PREFIX}{uuid4()}"
    config_data = {"id": config_id, "name": payload.config_name, "graph_structure": payload.graph_structure.dict()}
    try:
        redis_client.set(config_id, json.dumps(config_data))
        logger.info(f"Saved config '{payload.config_name}' ID: {config_id}")
        return {"message": "Configuration saved", "config_id": config_id, "config_name": payload.config_name}
    except redis.RedisError as e:
        logger.error(f"Error saving to Redis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save: {str(e)}")

@app.get("/api/configs", response_model=List[Dict[str, str]])
async def list_configurations():
    logger.info("Listing configurations.")
    if not redis_client:
        reconnect_redis()
    if not redis_client:
        logger.warning("Redis unavailable. Returning empty config list.")
        return []
    try:
        config_keys = [key for key in redis_client.scan_iter(match=f"{CONFIG_KEY_PREFIX}*")]
        configs_summary = []
        for key in config_keys:
            try:
                config_json = redis_client.get(key)
                if config_json:
                    config_data = json.loads(config_json)
                    configs_summary.append({"id": config_data.get("id", key), "name": config_data.get("name", "Unnamed")})
                else:
                    configs_summary.append({"id": key, "name": "*Load Error*"})
            except (redis.RedisError, json.JSONDecodeError) as e:
                logger.error(f"Error processing config {key}: {e}", exc_info=True)
                configs_summary.append({"id": key, "name": "*Load Error*"})
        return configs_summary
    except redis.RedisError as e:
        logger.error(f"Error listing configs: {e}", exc_info=True)
        return []

@app.get("/api/configs/{config_id}", response_model=Dict[str, Any])
async def load_configuration(config_id: str):
    if not redis_client:
        reconnect_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Storage connection unavailable.")
    config_id_with_prefix = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
    try:
        config_json = redis_client.get(config_id_with_prefix)
        if config_json is None:
            raise HTTPException(status_code=404, detail="Configuration not found.")
        return json.loads(config_json)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON for {config_id_with_prefix}")
        raise HTTPException(status_code=500, detail="Config data corrupted.")
    except redis.RedisError as e:
        logger.error(f"Error loading {config_id_with_prefix}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")

@app.delete("/api/configs/{config_id}", status_code=200)
async def delete_configuration(config_id: str):
    if not redis_client:
        reconnect_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Storage connection unavailable.")
    config_id_with_prefix = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
    try:
        deleted_count = redis_client.delete(config_id_with_prefix)
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Configuration not found.")
        log_filename = f"{LOG_FILENAME_PREFIX}{config_id_with_prefix}{LOG_FILENAME_SUFFIX}"
        try:
            await blob_delete(pathname=log_filename, options={'token': vercel_blob_token})
            logger.info(f"Deleted log file '{log_filename}'.")
        except Exception as e:
            logger.warning(f"Failed to delete log file '{log_filename}': {e}")
        logger.info(f"Deleted config '{config_id_with_prefix}'.")
        return {"message": "Configuration deleted successfully."}
    except redis.RedisError as e:
        logger.error(f"Error deleting {config_id_with_prefix}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")

@app.post("/api/configs/set_default", status_code=200)
async def set_default_configuration(config_id: str = Body(...)):
    if not redis_client:
        reconnect_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Storage connection unavailable.")
    config_id_with_prefix = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
    try:
        config_json = redis_client.get(config_id_with_prefix)
        if config_json is None:
            raise HTTPException(status_code=404, detail="Configuration not found.")
        redis_client.set(DEFAULT_CONFIG_KEY, config_id_with_prefix)
        config_data = json.loads(config_json)
        logger.info(f"Set '{config_data['name']}' as default config.")
        return {"message": f"Configuration '{config_data['name']}' set as default."}
    except redis.RedisError as e:
        logger.error(f"Error setting default config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set default: {str(e)}")

@app.post("/api/predict_openai_bn_single_call")
async def predict_openai_bn_single_call(payload: PredictionPayload):
    """ Runs prediction: validates graph, calls LLM for probs & reasoning, logs result. """
    logger.info("Entered prediction endpoint.")
    try:
        # 1. Validate graph structure
        is_valid_dag, cycle_info = is_dag(payload.graph_structure)
        if not is_valid_dag:
            raise HTTPException(status_code=400, detail=f"Invalid graph: Not a DAG. {cycle_info or ''}")

        # Get node info early for logging and response prep
        node_parents, node_descriptions, _, all_nodes, target_nodes, input_nodes = get_dynamic_node_info(payload.graph_structure)

        # 2. Get Probabilities
        logger.info("Calling OpenAI for probabilities...")
        estimated_target_probs = await call_openai_for_probabilities(payload.input_values, payload.graph_structure)

        # 3. Get Reasoning
        logger.info("Calling OpenAI for reasoning...")
        # Combine inputs and estimates for reasoning prompt context
        reasoning_prob_context = {**payload.input_values, **estimated_target_probs}
        reasoning_text = await call_openai_for_reasoning(payload.input_values, payload.graph_structure, reasoning_prob_context)

        # 4. Combine results for final response
        all_current_probabilities = {**payload.input_values, **estimated_target_probs}
        final_result_probs = {}
        log_probs_p1 = {} # Prepare flattened P(1) for logging
        for node_id in all_nodes: # Iterate through ALL nodes in the graph structure
            node_desc = node_descriptions.get(node_id, node_id)
            p1 = all_current_probabilities.get(node_id, 0.5) # Default if somehow missing
            p1_clamped = max(0.0, min(1.0, p1))
            final_result_probs[node_id] = {"0": 1.0 - p1_clamped, "1": p1_clamped, "description": node_desc}
            log_probs_p1[node_id] = p1_clamped # Store P(1) for logging

        # 5. Prepare response payload
        response_payload = {
            "probabilities": final_result_probs,
            "llm_reasoning": reasoning_text,
            "llm_context": { # Context provided TO the LLM (for display/debug)
                 "input_states": [{ "node": n, "description": node_descriptions.get(n, n), "value": v, "state": ("High" if v >= 0.66 else ("Medium" if v >= 0.33 else "Low")) } for n,v in payload.input_values.items() if n in input_nodes], # Show only actual inputs used
                 "node_dependencies": node_parents,
                 "node_descriptions": node_descriptions
             }
        }

        # 6. Log the prediction (pass the full list of nodes for header consistency)
        log_entry = LogPayload(
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'), # Use UTC timestamp
            configId=payload.config_id or "unknown",
            configName=payload.config_name or "Unknown",
            probabilities=log_probs_p1 # Use the flattened P(1) dict
        )
        try:
            logger.info(f"Attempting to log prediction to blob for configId: {log_entry.configId}")
            # *** Pass `all_nodes` list here ***
            await log_data_to_blob(log_entry, all_nodes)
            logger.info("Logging successful (or skipped for unsaved/default).")
        except Exception as log_err:
            # Log the error but don't fail the main prediction request
            logger.error(f"Failed to log prediction data: {log_err}", exc_info=True)

        logger.info("Prediction completed successfully.")
        return response_payload

    except HTTPException as e: # Handle specific HTTP errors cleanly
        logger.error(f"HTTPException in prediction: {e.detail}", exc_info=False) # Log detail without full stack for HTTP known errors
        raise e
    except Exception as e: # Catch unexpected errors
        logger.error(f"Unexpected error in prediction: {e}", exc_info=True) # Log full stack trace
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction.")

async def log_data_to_blob(log_entry: LogPayload, all_node_ids_in_graph: List[str]):
    """Appends log data (one row per prediction) to a CSV file in Vercel Blob."""
    if not vercel_blob_token:
        logger.warning("BLOB_READ_WRITE_TOKEN not configured. Skipping log.")
        return
    # Do not log for unsaved ('unknown') or the default config
    if not log_entry.configId or log_entry.configId == "unknown" or log_entry.configId == "default-config-001":
         logger.info(f"Skipping persistent log for configId: {log_entry.configId}")
         return # Don't log for unsaved or default configs

    # Use the config ID (which includes prefix) in the filename
    log_filename = f"{LOG_FILENAME_PREFIX}{log_entry.configId}{LOG_FILENAME_SUFFIX}"
    logger.info(f"Attempting to log to blob: {log_filename}")

    # --- Prepare the single data row for this prediction ---
    # Use all_node_ids_in_graph passed from the prediction context for consistent ordering
    sorted_node_ids = sorted(all_node_ids_in_graph) # Ensure consistent order

    # Create the data row, getting values from the log_entry probabilities
    # Use empty string if a node from the graph structure wasn't in this prediction's results
    data_row = [
        log_entry.timestamp,
        log_entry.configId,
        log_entry.configName,
    ]
    for node_id in sorted_node_ids:
        prob_p1 = log_entry.probabilities.get(node_id) # Get P(1) if exists
        data_row.append(f"{prob_p1:.4f}" if isinstance(prob_p1, float) else "") # Format or leave blank

    if not data_row: # Should not happen if timestamp etc. are present
        logger.warning("No data generated for log row.")
        return

    # --- Handle File Writing (Check Existence, Headers, Append) ---
    try:
        existing_content = b""
        needs_headers = False
        try:
            # Check if file exists
            logger.debug(f"Checking existence of blob: {log_filename}")
            head_result = await blob_head(pathname=log_filename, options={'token': vercel_blob_token})
            logger.debug(f"Blob {log_filename} exists (size: {head_result.get('size', 'N/A')}). Fetching content...")
            # If exists, download content to append to
            blob_response = await aiohttp.ClientSession().get(head_result['url']) # Use head result url
            if blob_response.status == 200:
                existing_content = await blob_response.read()
                if existing_content.endswith(b'\n'): # Remove trailing newline for cleaner append
                    existing_content = existing_content[:-1]
                logger.debug(f"Read {len(existing_content)} bytes from existing log.")
            else:
                 logger.warning(f"Could not read existing log {log_filename} (Status: {blob_response.status}). Will overwrite.")
                 needs_headers = True # Treat as new if cannot read
            await blob_response.release() # Close connection

        except Exception as head_or_download_error:
            # If head fails (likely 404 Not Found), it means the file doesn't exist
            logger.info(f"Log file {log_filename} not found or error accessing ({head_or_download_error}). Will create with headers.")
            needs_headers = True

        # --- Build the final CSV content ---
        output = io.StringIO()
        writer = csv.writer(output)

        # Write headers ONLY if file is new
        if needs_headers:
            # Create headers based on the nodes in the *current graph structure*
            headers = ['Timestamp', 'ConfigID', 'ConfigName'] + sorted_node_ids
            writer.writerow(headers)
            logger.info("Writing headers for new log file.")

        # Append existing content (if any)
        if existing_content:
             # Write existing content, ensuring a newline before the new row
             output.write(existing_content.decode('utf-8'))
             output.write('\n')

        # Write the new data row
        writer.writerow(data_row)

        # --- Upload the final content (overwrite existing blob) ---
        final_csv_content = output.getvalue().encode('utf-8')
        output.close()

        logger.debug(f"Uploading {len(final_csv_content)} bytes to {log_filename}...")
        put_result = await blob_put(
            pathname=log_filename,
            body=final_csv_content,
            options={'access': 'public', 'add_random_suffix': False, 'contentType': 'text/csv', 'token': vercel_blob_token}
        )
        logger.info(f"Successfully wrote log data to: {log_filename} (URL: {put_result.get('url', 'N/A')})")

    except Exception as e:
        logger.error(f"Failed to put/append log data to blob {log_filename}: {e}", exc_info=True)

@app.get("/api/download_log/{config_id}")
async def download_log_file(config_id: str):
    """Downloads the full log CSV for a specific config ID from Vercel Blob."""
    if not vercel_blob_token: raise HTTPException(status_code=500, detail="BLOB_READ_WRITE_TOKEN not configured")
    if not config_id or config_id == "unknown" or config_id == "default-config-001": raise HTTPException(status_code=400, detail="Cannot download logs for unsaved/default configs.")

    config_id_with_prefix = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
    log_filename = f"{LOG_FILENAME_PREFIX}{config_id_with_prefix}{LOG_FILENAME_SUFFIX}"
    logger.info(f"Attempting to download log file: {log_filename}")

    try:
        # Use blob_head to get the actual URL (more reliable than assuming structure)
        head_result = await blob_head(pathname=log_filename, options={'token': vercel_blob_token})
        blob_url = head_result.get('url')
        if not blob_url:
            logger.error(f"Blob head succeeded but no URL found for {log_filename}")
            raise HTTPException(status_code=404, detail=f"Log file metadata found, but URL is missing.")

        # Stream the content from the blob URL
        async def stream_content():
            try:
                 # Increased timeout slightly
                async with aiohttp.ClientSession(timeout=ClientTimeout(total=10)) as session:
                    async with session.get(blob_url) as response:
                        if response.status == 200:
                            while True:
                                chunk = await response.content.read(8192) # Read in chunks
                                if not chunk:
                                    break
                                yield chunk
                        elif response.status == 404:
                             logger.warning(f"Log file URL returned 404: {blob_url}")
                             raise HTTPException(status_code=404, detail=f"Log file not found at storage URL.")
                        else:
                             error_text = await response.text()
                             logger.error(f"Failed to fetch log file content from {blob_url}: HTTP {response.status} - {error_text}")
                             raise HTTPException(status_code=response.status, detail=f"Error fetching log file content.")
            except Exception as stream_err:
                 logger.error(f"Error streaming log file {log_filename}: {stream_err}", exc_info=True)
                 # Yield an error message in the stream? Or just let it fail?
                 # For now, let the exception propagate to the main try/except
                 raise stream_err


        safe_filename_part = config_id_with_prefix.replace(CONFIG_KEY_PREFIX, "")
        download_filename = f"log_{safe_filename_part}.csv"

        return StreamingResponse(
            stream_content(), # Stream the response body
            media_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{download_filename}"'}
        )

    except Exception as e:
        # Catch errors from blob_head or potential streaming errors
        err_str = str(e)
        # Check if it's a likely "Not Found" error from blob_head
        if "NotFound" in err_str or "not found" in err_str.lower() or (isinstance(e, HTTPException) and e.status_code == 404):
             logger.warning(f"Log file not found: {log_filename}")
             raise HTTPException(status_code=404, detail=f"Log file not found for config ID {config_id}")
        else:
             logger.error(f"Failed to download log file {log_filename}: {e}", exc_info=True)
             # Raise a generic 500 if it wasn't a clear 404 or HTTP error from streaming
             if isinstance(e, HTTPException):
                 raise e
             else:
                 raise HTTPException(status_code=500, detail=f"Failed to download log file.")

@app.get("/")
async def root():
    logger.info("Entered / route.")
    return {"message": "BN API is running."}

print("DEBUG: Finished defining routes.", flush=True)
