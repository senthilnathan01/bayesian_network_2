import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from uuid import uuid4
import urllib.parse
from datetime import datetime
import io
import csv
import aiohttp

print("--- DEBUG: api/main.py TOP LEVEL EXECUTION ---", flush=True)

try:
    from fastapi import FastAPI, HTTPException, Body, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field, validator
    import openai
    from dotenv import load_dotenv
    import redis
    from redis.connection import ConnectionPool
    from vercel_blob import put as blob_put, head as blob_head, delete as blob_delete
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

if not openai_api_key:
    logger.warning("OPENAI_API_KEY environment variable not found. Prediction endpoints will fail.")
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

def call_openai_for_probabilities(input_states: Dict[str, float], graph: GraphStructure) -> Dict[str, float]:
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    openai.api_key = openai_api_key

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

    # Enhanced prompt with detailed node and relationship descriptions
    node_info = []
    for node_id in node_descriptions:
        parents = node_parents.get(node_id, [])
        node_type = "input" if node_id in input_nodes else "hidden"
        parent_desc = ", ".join([f"{p} ({node_descriptions.get(p, p)})" for p in parents]) if parents else "none"
        node_info.append(f"- {node_id} ({node_descriptions[node_id]}): {node_type} node, influenced by {parent_desc}")
    structure_description = f"""
    Bayesian Network Structure:
    Nodes and Relationships:
    { '\n'.join(node_info) }
    """

    system_message = """
    You are an expert probabilistic reasoner modeling a Bayesian Network for cognitive processes.
    You are given a Directed Acyclic Graph (DAG) with nodes representing cognitive factors or actions.
    Each node is binary (states 0 or 1), and you must estimate P(Node=1) for all 'hidden' nodes based on the provided P(Node=1) for 'input' nodes.
    Use the node descriptions and their dependency relationships to infer how input probabilities propagate through the network.
    Consider the semantic meaning of each node and its parents to make informed estimates.
    Return a JSON object mapping each hidden node to its estimated P(Node=1) (float, 0.0 to 1.0).
    Example: {"HiddenNode1": 0.7, "HiddenNode2": 0.4}
    Output ONLY the JSON object.
    """
    user_message = f"""
    Input Probabilities (P=1):
    {input_context}

    Network Structure:
    {structure_description}

    Estimate P(Node=1) for hidden nodes ({', '.join(target_nodes)}) based on the inputs and network structure.
    """

    logger.debug("Sending probability prompt to OpenAI...")
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            max_tokens=1000,
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
        raise HTTPException(status_code=500, detail=f"LLM JSON parse error: {e}")
    except openai.APIError as e:
        logger.error(f"OpenAI API Error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API Error: {e.body}")
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI Auth Error: {e}")
        raise HTTPException(status_code=401, detail=f"OpenAI Auth Failed: {e.body}")
    except openai.RateLimitError as e:
        logger.error(f"OpenAI Rate Limit: {e}")
        raise HTTPException(status_code=429, detail=f"OpenAI Rate Limit: {e.body}")
    except Exception as e:
        logger.error(f"Error in OpenAI call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {e}")

def call_openai_for_reasoning(input_states: Dict[str, float], graph: GraphStructure, estimated_probabilities: Dict[str, float]) -> str:
    if not openai_api_key:
        return "Reasoning disabled: OpenAI API key not configured."
    openai.api_key = openai_api_key

    node_parents, node_descriptions, _, all_nodes, target_nodes, input_nodes = get_dynamic_node_info(graph)

    input_desc_list = []
    for node, value in input_states.items():
        if node in input_nodes:
            state_desc = "High" if value >= 0.66 else ("Medium" if value >= 0.33 else "Low")
            input_desc_list.append(f"- {node} ({node_descriptions.get(node, node)}): {state_desc} (P(1)={value:.2f})")
    input_context = "\n".join(input_desc_list)

    node_info = []
    for node_id in node_descriptions:
        parents = node_parents.get(node_id, [])
        node_type = "input" if node_id in input_nodes else "hidden"
        parent_desc = ", ".join([f"{p} ({node_descriptions.get(p, p)})" for p in parents]) if parents else "none"
        node_info.append(f"- {node_id} ({node_descriptions[node_id]}): {node_type} node, influenced by {parent_desc}")
    structure_description = f"""
    Network Structure:
    { '\n'.join(node_info) }
    """

    probs_text = "\n".join([f"- {node} ({node_descriptions.get(node, node)}): P(1)={prob:.3f}" for node, prob in estimated_probabilities.items()])
    all_probs = {**input_states, **estimated_probabilities}
    all_probs_text = "\n".join([f"- {node} ({node_descriptions.get(node, node)}): P(1)={all_probs.get(node, 'N/A'):.3f}" for node in all_nodes])

    system_message = """
    You are an expert analyst explaining a Bayesian Network simulation for cognitive processes.
    Given the input probabilities, network structure (nodes, descriptions, dependencies), and estimated probabilities for hidden nodes,
    provide a concise explanation for why each hidden node received its estimated P(Node=1).
    Focus on how the input values and parent nodes' semantic meanings influenced each hidden node's probability.
    Structure the explanation as a list, one item per hidden node, with clear reasoning.
    Example:
    - Node IS1 (Confidence): Estimated P(1)=0.7 due to high Motivation (A5) and moderate Task Understanding (IS3).
    Keep it concise and avoid repeating the structure unnecessarily.
    """
    user_message = f"""
    Inputs (P=1):
    {input_context}

    Network Structure:
    {structure_description}

    Estimated Probabilities for Hidden Nodes:
    {probs_text}

    All Probabilities:
    {all_probs_text}

    Explain why each hidden node ({', '.join(target_nodes)}) received its estimated probability, considering the inputs and dependencies.
    """

    logger.debug("Sending reasoning prompt to OpenAI...")
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.3,
            n=1
        )
        reasoning_text = response.choices[0].message.content.strip()
        logger.info(f"OpenAI Reasoning Output received.")
        return reasoning_text
    except Exception as e:
        logger.error(f"Error in reasoning call: {e}", exc_info=True)
        return f"Could not generate reasoning: {e}"

# API Endpoints
@app.get("/api/ping")
def ping():
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
        async with aiohttp.ClientSession() as session:
            async with session.get(put_result['url']) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch blob: HTTP {response.status}")
                content = await response.read()
        await blob_delete(pathname=test_filename, options={'token': vercel_blob_token})
        return {"status": "success", "content": content.decode('utf-8')}
    except Exception as e:
        logger.error(f"Vercel Blob test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vercel Blob test failed: {e}")

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
        raise HTTPException(status_code=500, detail=f"Failed to save: {e}")

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
        raise HTTPException(status_code=500, detail=f"Load failed: {e}")

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
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")

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
        raise HTTPException(status_code=500, detail=f"Failed to set default: {e}")

@app.post("/api/predict_openai_bn_single_call")
async def predict_openai_bn_single_call(payload: PredictionPayload):
    logger.info("Entered prediction endpoint.")
    try:
        is_valid_dag, cycle_info = is_dag(payload.graph_structure)
        if not is_valid_dag:
            raise HTTPException(status_code=400, detail=f"Invalid graph: Not a DAG. {cycle_info or ''}")

        estimated_target_probs = call_openai_for_probabilities(payload.input_values, payload.graph_structure)
        reasoning_text = call_openai_for_reasoning(payload.input_values, payload.graph_structure, estimated_target_probs)
        all_current_probabilities = {**payload.input_values, **estimated_target_probs}
        _, node_descriptions, _, all_nodes, _, _ = get_dynamic_node_info(payload.graph_structure)
        final_result_probs = {}
        for node in all_nodes:
            node_desc = node_descriptions.get(node, node)
            p1 = all_current_probabilities.get(node, 0.5)
            p1_clamped = max(0.0, min(1.0, p1))
            final_result_probs[node] = {"0": 1.0 - p1_clamped, "1": p1_clamped, "description": node_desc}

        response_payload = {
            "probabilities": final_result_probs,
            "llm_reasoning": reasoning_text,
            "llm_context": {
                "input_states": [
                    {
                        "node": n,
                        "description": node_descriptions.get(n, n),
                        "value": v,
                        "state": "High" if v >= 0.66 else ("Medium" if v >= 0.33 else "Low")
                    } for n, v in payload.input_values.items()
                ],
                "node_dependencies": {
                    node.id: [e.source for e in payload.graph_structure.edges if e.target == node.id]
                    for node in payload.graph_structure.nodes
                },
                "node_descriptions": {n.id: n.fullName for n in payload.graph_structure.nodes}
            }
        }

        log_probs = {node_id: data["1"] for node_id, data in final_result_probs.items()}
        log_entry = LogPayload(
            timestamp=datetime.utcnow().isoformat() + "Z",
            configId=payload.config_id or "unknown",
            configName=payload.config_name or "Unknown",
            probabilities=log_probs
        )
        try:
            await log_data_to_blob(log_entry)
            logger.info("Prediction logged successfully.")
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}", exc_info=True)

        logger.info("Prediction completed successfully.")
        return response_payload
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

async def log_data_to_blob(log_entry: LogPayload):
    if not vercel_blob_token:
        logger.warning("BLOB_READ_WRITE_TOKEN not configured. Skipping log.")
        return
    if not log_entry.configId or log_entry.configId == "unknown" or log_entry.configId == "default-config-001":
        logger.warning(f"Skipping log for unsaved/default config: {log_entry.configId}")
        return

    log_filename = f"{LOG_FILENAME_PREFIX}{log_entry.configId}{LOG_FILENAME_SUFFIX}"
    logger.info(f"Logging to blob: {log_filename}")

    new_rows = [
        [log_entry.timestamp, log_entry.configId, log_entry.configName, node_id, f"{prob_p1:.4f}"]
        for node_id, prob_p1 in log_entry.probabilities.items()
    ]

    if not new_rows:
        logger.warning("No data rows to log.")
        return

    try:
        needs_headers = False
        try:
            await blob_head(pathname=log_filename, options={'token': vercel_blob_token})
        except Exception:
            needs_headers = True
            logger.info(f"Log file {log_filename} does not exist. Will create with headers.")

        output = io.StringIO()
        writer = csv.writer(output)

        existing_content = b""
        if not needs_headers:
            try:
                put_result = await blob_head(pathname=log_filename, options={'token': vercel_blob_token})
                blob_url = put_result.get('url')
                if not blob_url:
                    logger.warning(f"No URL returned for {log_filename}. Treating as new file.")
                    needs_headers = True
                else:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(blob_url) as response:
                            if response.status != 200:
                                logger.warning(f"Failed to fetch {log_filename}: HTTP {response.status}")
                                needs_headers = True
                            else:
                                existing_content = await response.read()
                                if existing_content.endswith(b'\n'):
                                    existing_content = existing_content[:-1]
            except Exception as e:
                logger.warning(f"Could not retrieve {log_filename}: {e}")
                needs_headers = True
                existing_content = b""

        if existing_content:
            output.write(existing_content.decode('utf-8'))
            output.write('\n')

        if needs_headers:
            writer.writerow(CSV_HEADERS)

        writer.writerows(new_rows)
        final_csv_content = output.getvalue().encode('utf-8')
        output.close()

        put_result = await blob_put(
            pathname=log_filename,
            body=final_csv_content,
            options={'access': 'public', 'add_random_suffix': False, 'content_type': 'text/csv', 'token': vercel_blob_token}
        )
        logger.info(f"Appended {len(new_rows)} entries to {log_filename} at {put_result['url']}")
    except Exception as e:
        logger.error(f"Failed to log to {log_filename}: {e}", exc_info=True)
        raise e

@app.get("/api/download_log/{config_id}")
async def download_log_file(config_id: str):
    if not vercel_blob_token:
        raise HTTPException(status_code=500, detail="BLOB_READ_WRITE_TOKEN not configured")
    if not config_id or config_id == "unknown" or config_id == "default-config-001":
        raise HTTPException(status_code=400, detail="Cannot download logs for unsaved/default configs.")

    config_id_with_prefix = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
    log_filename = f"{LOG_FILENAME_PREFIX}{config_id_with_prefix}{LOG_FILENAME_SUFFIX}"
    logger.info(f"Downloading log: {log_filename}")

    try:
        put_result = await blob_head(pathname=log_filename, options={'token': vercel_blob_token})
        blob_url = put_result.get('url')
        if not blob_url:
            raise HTTPException(status_code=404, detail=f"No URL found for log file {log_filename}")
        async with aiohttp.ClientSession() as session:
            async with session.get(blob_url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=404, detail=f"Failed to fetch log file: HTTP {response.status}")
                content = await response.read()
        safe_filename_part = config_id_with_prefix.replace(CONFIG_KEY_PREFIX, "")
        download_filename = f"log_{safe_filename_part}.csv"
        return StreamingResponse(
            io.BytesIO(content),
            media_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{download_filename}"'}
        )
    except Exception as e:
        logger.error(f"Failed to download {log_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"Log file not found for config ID {config_id}")

@app.get("/")
def root():
    logger.info("Entered / route.")
    return {"message": "BN API is running."}

print("DEBUG: Finished defining routes.", flush=True)
