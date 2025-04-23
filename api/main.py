import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from uuid import uuid4
import urllib.parse
from datetime import datetime
import io # For handling streams with Blob
import csv # For CSV processing

print("--- DEBUG: api/main.py TOP LEVEL EXECUTION ---", flush=True)

try:
    from fastapi import FastAPI, HTTPException, Body, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field, validator
    import openai
    from dotenv import load_dotenv
    import redis
    from vercel_blob import put as blob_put, head as blob_head, download as blob_download, delete as blob_delete # Import Vercel Blob functions
    print("--- DEBUG: Imported FastAPI, CORS, Pydantic, OpenAI, Redis, Vercel Blob ---", flush=True)

    app = FastAPI(title="Dynamic Bayesian Network API")
    print("--- DEBUG: FastAPI app created ---", flush=True)

    # Allow CORS
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
redis_url = os.getenv("UPSTASH_REDIS_URL") or os.getenv("REDIS_URL")
# Vercel Blob automatically uses BLOB_READ_WRITE_TOKEN from env

# --- Redis Connection ---
redis_client = None
if redis_url:
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("--- Successfully connected to Redis (Upstash) ---")
    except Exception as e:
        logger.error(f"--- Failed to connect to Redis using URL: {e} ---", exc_info=True)
        redis_client = None
else:
    logger.warning("--- Redis URL not found. Config storage/retrieval disabled. ---")

if not openai_api_key: logger.warning("OPENAI_API_KEY environment variable not found.")

# --- Constants ---
CONFIG_KEY_PREFIX = "bn_config:"
LOG_FILENAME_PREFIX = "bn_log_"
LOG_FILENAME_SUFFIX = ".csv"
CSV_HEADERS = ['Timestamp', 'ConfigID', 'ConfigName', 'NodeID', 'ProbabilityP1'] # Simpler log format for append

# --- Default Graph Structure ---
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
            {"id": "O1", "fullName": "Predicted Success Probability", "nodeType": "hidden"}, # Changed to hidden
            {"id": "O2", "fullName": "Action Speed/Efficiency", "nodeType": "hidden"}, # Changed to hidden
            {"id": "O3", "fullName": "Help Seeking Likelihood", "nodeType": "hidden"}, # Changed to hidden
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

# --- Pydantic Models ---

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

class ContinuousUserInput(BaseModel):
    input_values: Dict[str, float]

class PredictionPayload(ContinuousUserInput):
    graph_structure: GraphStructure
    config_id: Optional[str] = None # Needed for logging
    config_name: Optional[str] = "Unknown" # Needed for logging

class SaveConfigPayload(BaseModel):
    config_name: str = Field(..., min_length=1, description="User-provided name for the configuration")
    graph_structure: GraphStructure

class LogPayload(BaseModel):
    timestamp: str
    configId: str
    configName: str
    probabilities: Dict[str, float] # Dict of NodeID: P(1)

# --- Graph Validation ---

def is_dag(graph: GraphStructure) -> Tuple[bool, Optional[str]]:
    """ Checks if the graph is a Directed Acyclic Graph (DAG). Returns (is_dag, cycle_info). """
    adj: Dict[str, List[str]] = {node.id: [] for node in graph.nodes}
    nodes_set = {node.id for node in graph.nodes}
    for edge in graph.edges:
        # Ensure source and target nodes exist
        if edge.source not in nodes_set or edge.target not in nodes_set:
             return False, f"Edge references non-existent node: {edge.source} -> {edge.target}"
        if edge.target not in adj: adj[edge.target] = [] # Should not happen if nodes_set is correct
        adj[edge.source].append(edge.target)

    path: Set[str] = set()
    visited: Set[str] = set()

    def dfs(node: str) -> bool:
        path.add(node)
        visited.add(node)
        for neighbor in adj.get(node, []):
            if neighbor in path:
                return False # Cycle detected
            if neighbor not in visited:
                if not dfs(neighbor):
                    return False # Cycle detected downstream
        path.remove(node)
        return True

    all_nodes_list = list(nodes_set)
    for node in all_nodes_list:
        if node not in visited:
            if not dfs(node):
                return False, f"Cycle detected involving node path near {node}" # Basic cycle info

    return True, None

# --- Helper Functions ---

def get_dynamic_node_info(graph: GraphStructure):
    """ Parses graph structure to get node info. (No qualitative rules anymore) """
    node_parents = {node.id: [] for node in graph.nodes}
    node_descriptions = {node.id: node.fullName for node in graph.nodes}
    node_types = {node.id: node.nodeType for node in graph.nodes}
    nodes_set = {node.id for node in graph.nodes}

    for edge in graph.edges:
        # Basic check, more robust validation happens before this
        if edge.source in nodes_set and edge.target in nodes_set:
             if edge.target in node_parents:
                 node_parents[edge.target].append(edge.source)

    all_nodes = list(nodes_set)
    target_nodes = [node_id for node_id, node_type in node_types.items() if node_type == 'hidden']
    input_nodes = [node_id for node_id, node_type in node_types.items() if node_type == 'input']

    return node_parents, node_descriptions, node_types, all_nodes, target_nodes, input_nodes


def call_openai_for_probabilities(
    input_states: Dict[str, float],
    graph: GraphStructure
) -> Dict[str, float]:
    """ Calls OpenAI API JUST for probability estimates. """
    if not openai_api_key: raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    openai.api_key = openai_api_key

    node_parents, node_descriptions, _, _, target_nodes, input_nodes = get_dynamic_node_info(graph)

    missing_inputs = [node for node in input_nodes if node not in input_states]
    if missing_inputs: raise HTTPException(status_code=400, detail=f"Missing input values for nodes: {', '.join(missing_inputs)}")

    input_desc_list = []
    for node, value in input_states.items():
        if node in input_nodes:
            state_desc = "High" if value >= 0.66 else ("Medium" if value >= 0.33 else "Low")
            input_desc_list.append(f"- {node} ({node_descriptions.get(node, node)}): {state_desc} (input probability approx {value:.2f})")
    input_context = "\n".join(input_desc_list)

    if not target_nodes: return {} # No hidden nodes to predict

    structure_description = f"""
    Bayesian Network Structure:
    Node Descriptions: {json.dumps(node_descriptions, indent=2)}
    Node Types: {json.dumps({node.id: node.nodeType for node in graph.nodes}, indent=2)}
    Dependencies (Node <- Parents): {json.dumps(node_parents, indent=2)}
    """

    system_message = f"""
    You are an expert probabilistic reasoner simulating a user-defined Directed Acyclic Graph (DAG).
    Your task is to estimate the probability P(Node=1) for the 'hidden' type nodes, given the P(Node=1) for the 'input' type nodes.
    Use the provided DAG structure and node descriptions to guide your estimation. Consider how parent nodes influence children based on their descriptions and the overall structure.
    Provide the final estimated probabilities ONLY as a single, valid JSON object mapping the target node names ({', '.join(target_nodes)}) to their estimated P(Node=1) value (a float between 0.0 and 1.0).
    Ensure the JSON object contains *all* target nodes. Example format: {{"HiddenNode1": 0.7, "HiddenNode2": 0.4}}
    Output ONLY the JSON object.
    """
    user_message = f"""
    Initial Input Probabilities (P=1):
    {input_context}

    Bayesian Network Structure:
    {structure_description}

    Estimate the probability P(Node=1) for all target nodes ({', '.join(target_nodes)}) based ONLY on the provided inputs and network structure/descriptions. Return the result ONLY as a JSON object.
    """

    logger.debug("Constructing probability prompt for OpenAI...")
    try:
        response = openai.chat.completions.create(
            model="gpt-4o", messages=[ {"role": "system", "content": system_message}, {"role": "user", "content": user_message} ],
            response_format={"type": "json_object"}, max_tokens=1000, temperature=0.1, n=1 )
        llm_output_raw = response.choices[0].message.content.strip()
        logger.info(f"OpenAI Probability Output: {llm_output_raw}")
        # (Validation logic remains the same)
        estimated_probs = json.loads(llm_output_raw)
        validated_probs = {}
        missing_nodes = []
        for node in target_nodes:
            if node in estimated_probs and isinstance(estimated_probs[node], (float, int)):
                validated_probs[node] = max(0.0, min(1.0, float(estimated_probs[node])))
            else:
                logger.warning(f"Node '{node}' missing/invalid in probability JSON. Using default 0.5.")
                missing_nodes.append(node)
                validated_probs[node] = 0.5
        if missing_nodes: logger.warning(f"LLM probability output missing/invalid for: {', '.join(missing_nodes)}.")
        return validated_probs
    except json.JSONDecodeError as json_err: logger.error(f"Failed to parse LLM prob JSON: {llm_output_raw}. Error: {json_err}"); raise HTTPException(status_code=500, detail=f"LLM probability JSON parse error: {json_err}")
    except openai.APIError as e: logger.error(f"OpenAI Prob API Error: {e}"); raise HTTPException(status_code=502, detail=f"OpenAI API Error: {e.body}")
    except openai.AuthenticationError as e: logger.error(f"OpenAI Auth Error: {e}"); raise HTTPException(status_code=401, detail=f"OpenAI Auth Failed: {e.body}")
    except openai.RateLimitError as e: logger.error(f"OpenAI Rate Limit: {e}"); raise HTTPException(status_code=429, detail=f"OpenAI Rate Limit: {e.body}")
    except Exception as e: logger.error(f"Error in OpenAI prob call: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Error during OpenAI prob request: {e}")


def call_openai_for_reasoning(
    input_states: Dict[str, float],
    graph: GraphStructure,
    estimated_probabilities: Dict[str, float] # Pass the results from the first call
    ) -> str:
    """ Calls OpenAI API a second time to get reasoning text. """
    if not openai_api_key: return "Reasoning disabled: OpenAI API key not configured."
    openai.api_key = openai_api_key

    node_parents, node_descriptions, _, all_nodes, target_nodes, input_nodes = get_dynamic_node_info(graph)

    # Prepare context similar to the first call
    input_desc_list = []
    for node, value in input_states.items():
        if node in input_nodes:
            state_desc = "High" if value >= 0.66 else ("Medium" if value >= 0.33 else "Low")
            input_desc_list.append(f"- {node} ({node_descriptions.get(node, node)}): {state_desc} (input P(1)={value:.2f})")
    input_context = "\n".join(input_desc_list)

    structure_description = f"""
    Network Structure:
    Nodes: {json.dumps(node_descriptions, indent=2)}
    Dependencies (Node <- Parents): {json.dumps(node_parents, indent=2)}
    """

    # Format estimated probabilities nicely
    estimated_probs_text = "\n".join([f"- {node}: {prob:.3f}" for node, prob in estimated_probabilities.items()])
    all_probs = {**input_states, **estimated_probabilities}
    all_probs_text = "\n".join([f"- {node} ({node_descriptions.get(node, node)}): P(1)={all_probs.get(node, 'N/A'):.3f}" for node in all_nodes])


    system_message = """
    You are an expert analyst explaining the results of a Bayesian Network simulation.
    You will be given the initial input probabilities, the network structure (nodes, descriptions, dependencies), and the final estimated probabilities for the hidden nodes.
    Your task is to provide a concise explanation for *why* the hidden nodes likely received their estimated probabilities, considering the specific input values and the network's structure and node meanings.
    Focus on the key influences and propagation effects through the network. Explain the logic flow. Do not just repeat the inputs or structure. Provide insights.
    Keep the explanation clear and focused on the reasoning process.
    """
    user_message = f"""
    Simulation Context:
    Inputs (P=1):
    {input_context}

    Network Structure:
    {structure_description}

    Estimated Probabilities for Hidden Nodes (P=1):
    {estimated_probs_text}

    All Node Probabilities (Inputs + Estimated):
    {all_probs_text}

    Task: Please explain the reasoning behind the estimated probabilities for the hidden nodes ({', '.join(target_nodes)}). How did the inputs propagate through the network defined by the descriptions and dependencies to arrive at these results?
    """

    logger.debug("Constructing reasoning prompt for OpenAI...")
    try:
        response = openai.chat.completions.create(
            model="gpt-4o", # Can use a slightly cheaper/faster model if needed, but 4o is good
            messages=[ {"role": "system", "content": system_message}, {"role": "user", "content": user_message} ],
            max_tokens=500, # Adjust as needed for reasoning length
            temperature=0.3, # Slightly higher temp for more natural explanation
            n=1
        )
        reasoning_text = response.choices[0].message.content.strip()
        logger.info(f"OpenAI Reasoning Output received.")
        return reasoning_text
    # Separate error handling for reasoning - non-critical, return default text
    except Exception as e:
        logger.error(f"Error calling OpenAI for reasoning: {e}", exc_info=True)
        return f"Could not generate reasoning: {e}"


# --- API Endpoints ---

@app.get("/api/ping")
def ping():
    # (Unchanged from previous Redis version)
    redis_status = "disabled"
    if redis_client:
        try: redis_client.ping(); redis_status = "connected"
        except Exception as e: redis_status = f"disconnected ({e})"
    return {"message": "pong", "redis_status": redis_status}

# --- Config Management Endpoints ---

@app.get("/api/configs/default", response_model=Dict[str, Any])
async def get_default_configuration():
    """ Returns the hardcoded default graph configuration. """
    logger.info("Serving default configuration.")
    # No need to check Redis here
    return DEFAULT_GRAPH_STRUCTURE

@app.post("/api/configs", status_code=201)
async def save_configuration(payload: SaveConfigPayload):
    """ Validates and saves a graph configuration to Redis. """
    if not redis_client: raise HTTPException(status_code=503, detail="Storage connection unavailable.")

    # Validate graph structure
    is_valid_dag, cycle_info = is_dag(payload.graph_structure)
    if not is_valid_dag:
        raise HTTPException(status_code=400, detail=f"Invalid graph structure: Not a DAG. {cycle_info or ''}")

    config_id = f"{CONFIG_KEY_PREFIX}{uuid4()}"
    config_data = { "id": config_id, "name": payload.config_name, "graph_structure": payload.graph_structure.dict() }
    try:
        success = redis_client.set(config_id, json.dumps(config_data))
        if not success: raise Exception("Redis SET command failed.")
        logger.info(f"Saved config '{payload.config_name}' ID: {config_id}")
        return {"message": "Configuration saved", "config_id": config_id, "config_name": payload.config_name}
    except Exception as e: logger.error(f"Error saving to Redis: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Failed to save: {e}")

@app.get("/api/configs", response_model=List[Dict[str, str]])
async def list_configurations():
    """ Lists saved graph configurations from Redis. """
    if not redis_client: raise HTTPException(status_code=503, detail="Storage connection unavailable.")
    configs_summary = []
    try:
        config_keys = [key for key in redis_client.scan_iter(match=f"{CONFIG_KEY_PREFIX}*")]
        if not config_keys: return []
        # Use MGET for efficiency if many keys
        config_jsons = redis_client.mget(config_keys)
        for key, config_json in zip(config_keys, config_jsons):
             if config_json:
                 try:
                     config_data = json.loads(config_json)
                     configs_summary.append({ "id": config_data.get("id", key), "name": config_data.get("name", "Unnamed") })
                 except Exception as e: logger.warning(f"Failed parsing key {key}: {e}"); configs_summary.append({"id": key, "name": "*Load Error*"})
             else: logger.warning(f"Key {key} from scan not found with MGET.") # Should be rare
        return configs_summary
    except Exception as e: logger.error(f"Error listing from Redis: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Failed to list: {e}")

@app.get("/api/configs/{config_id}", response_model=Dict[str, Any])
async def load_configuration(config_id: str):
    """ Loads a specific graph configuration from Redis. """
    if not redis_client: raise HTTPException(status_code=503, detail="Storage connection unavailable.")
    if not config_id.startswith(CONFIG_KEY_PREFIX): config_id_with_prefix = f"{CONFIG_KEY_PREFIX}{config_id}"
    else: config_id_with_prefix = config_id
    try:
        config_json = redis_client.get(config_id_with_prefix)
        if config_json is None: raise HTTPException(status_code=404, detail="Configuration not found.")
        config_data = json.loads(config_json)
        return config_data
    except HTTPException as e: raise e
    except json.JSONDecodeError: logger.error(f"Invalid JSON for {config_id_with_prefix}"); raise HTTPException(status_code=500, detail="Config data corrupted.")
    except Exception as e: logger.error(f"Error loading {config_id_with_prefix} from Redis: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Load failed: {e}")

@app.delete("/api/configs/{config_id}", status_code=200)
async def delete_configuration(config_id: str):
    """ Deletes a specific graph configuration from Redis and its log from Blob. """
    if not redis_client: raise HTTPException(status_code=503, detail="Storage connection unavailable.")
    if not config_id.startswith(CONFIG_KEY_PREFIX): config_id_with_prefix = f"{CONFIG_KEY_PREFIX}{config_id}"
    else: config_id_with_prefix = config_id

    try:
        # Delete from Redis
        deleted_count = redis_client.delete(config_id_with_prefix)
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Configuration not found in Redis.")

        # Delete corresponding log file from Blob storage (best effort)
        log_filename = f"{LOG_FILENAME_PREFIX}{config_id_with_prefix}{LOG_FILENAME_SUFFIX}"
        try:
            await blob_delete(log_filename)
            logger.info(f"Deleted config '{config_id_with_prefix}' and corresponding log file '{log_filename}'.")
        except Exception as blob_error:
             # Log deletion failure is not fatal to the config deletion
             logger.warning(f"Deleted config '{config_id_with_prefix}' but failed to delete log file '{log_filename}': {blob_error}")

        return {"message": "Configuration deleted successfully."}

    except HTTPException as e: raise e
    except Exception as e:
        logger.error(f"Error deleting configuration '{config_id_with_prefix}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete configuration: {e}")


# --- Prediction Endpoint ---

@app.post("/api/predict_openai_bn_single_call")
async def predict_openai_bn_single_call(payload: PredictionPayload):
    """ Runs prediction: validates graph, calls LLM for probs, optionally calls for reasoning. """
    logger.info("Entered dynamic prediction endpoint.")
    try:
        # Validate graph structure
        is_valid_dag, cycle_info = is_dag(payload.graph_structure)
        if not is_valid_dag:
            raise HTTPException(status_code=400, detail=f"Invalid graph structure: Not a DAG. {cycle_info or ''}")

        # 1. Get Probabilities
        estimated_target_probs = call_openai_for_probabilities(payload.input_values, payload.graph_structure)

        # 2. Get Reasoning (Optional second call)
        reasoning_text = call_openai_for_reasoning(payload.input_values, payload.graph_structure, estimated_target_probs)

        # 3. Combine results
        all_current_probabilities = {**payload.input_values, **estimated_target_probs}
        _, node_descriptions, _, all_nodes, _, _ = get_dynamic_node_info(payload.graph_structure)
        final_result_probs = {}
        for node in all_nodes:
            node_desc = node_descriptions.get(node, node)
            if node in all_current_probabilities:
                p1 = all_current_probabilities[node]
                p1_clamped = max(0.0, min(1.0, p1))
                final_result_probs[node] = {"0": 1.0 - p1_clamped, "1": p1_clamped, "description": node_desc}
            else: # Should not happen if logic is correct
                final_result_probs[node] = {"0": 0.5, "1": 0.5, "description": node_desc}


        # Prepare response
        response_payload = {
            "probabilities": final_result_probs,
            "llm_reasoning": reasoning_text, # Add reasoning text
             # Keep context for display consistency, but remove qualitative rules part
             "llm_context": {
                 # Context info remains useful for debugging/display
                 "input_states": [{ "node": n, "description": node_descriptions.get(n, n), "value": v, "state": ("High" if v >= 0.66 else ("Medium" if v >= 0.33 else "Low")) } for n,v in payload.input_values.items()],
                 "node_dependencies": {node.id: [e.source for e in payload.graph_structure.edges if e.target == node.id] for node in payload.graph_structure.nodes},
                 "node_descriptions": {n.id: n.fullName for n in payload.graph_structure.nodes}
             }
        }

        # 4. Log the prediction (asynchronously if possible, but sync here for simplicity)
        # Use P(1) values for logging
        log_probs = {node_id: data["1"] for node_id, data in final_result_probs.items()}
        log_entry = LogPayload(
            timestamp=datetime.utcnow().isoformat() + "Z",
            configId=payload.config_id or "unknown",
            configName=payload.config_name or "Unknown",
            probabilities=log_probs
        )
        # Call logging endpoint internally or directly call logging function
        try:
            await log_data_to_blob(log_entry)
        except Exception as log_err:
            logger.error(f"Failed to log prediction data: {log_err}", exc_info=True)
            # Don't fail the whole request if logging fails, just log the error

        logger.info("Successfully generated prediction and reasoning.")
        return response_payload

    except HTTPException as e: raise e
    except Exception as e: logger.error(f"Error in prediction endpoint: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Logging Endpoints ---

async def log_data_to_blob(log_entry: LogPayload):
    """Appends log data to a CSV file in Vercel Blob storage."""
    if not log_entry.configId or log_entry.configId == "unknown" or log_entry.configId == "default-config-001":
         logger.warning(f"Skipping log for unsaved/default config: {log_entry.configId}")
         return # Don't log for unsaved or default configs

    # Use the config ID (which includes prefix) in the filename
    log_filename = f"{LOG_FILENAME_PREFIX}{log_entry.configId}{LOG_FILENAME_SUFFIX}"
    logger.info(f"Attempting to log to blob: {log_filename}")

    new_rows = []
    for node_id, prob_p1 in log_entry.probabilities.items():
         new_rows.append([
             log_entry.timestamp,
             log_entry.configId,
             log_entry.configName,
             node_id,
             f"{prob_p1:.4f}" # Format probability
         ])

    if not new_rows:
        logger.warning("No data rows to log.")
        return

    try:
        # Check if file exists to determine if headers are needed
        needs_headers = False
        try:
            await blob_head(pathname=log_filename)
            logger.debug(f"Log file {log_filename} exists.")
        except Exception as head_error:
            # Assuming 404 means file doesn't exist (Vercel Blob might not raise specific 404 error easily via head)
            logger.info(f"Log file {log_filename} likely doesn't exist (or HEAD error: {head_error}). Adding headers.")
            needs_headers = True

        # Use StringIO to build CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)

        existing_content = b""
        if not needs_headers:
             # Download existing content if file exists
             try:
                 blob_response = await blob_download(pathname=log_filename)
                 existing_content = await blob_response.read()
                 # Strip trailing newline if present to avoid blank rows
                 if existing_content.endswith(b'\n'):
                      existing_content = existing_content[:-1]
                 logger.debug(f"Read {len(existing_content)} bytes from existing log file.")
             except Exception as download_error:
                 logger.warning(f"Could not download existing log {log_filename} to append, creating new: {download_error}")
                 needs_headers = True # Treat as new file if download fails
                 existing_content = b""


        # Write existing content (if any) back first
        if existing_content:
             output.write(existing_content.decode('utf-8'))
             output.write('\n') # Ensure newline before new rows

        # Write headers if needed
        if needs_headers:
            writer.writerow(CSV_HEADERS)

        # Write new data rows
        writer.writerows(new_rows)

        # Upload the combined content
        final_csv_content = output.getvalue().encode('utf-8')
        output.close()

        await blob_put(
            pathname=log_filename,
            body=final_csv_content,
            options={'addRandomSuffix': False, 'contentType': 'text/csv'} # Overwrite with appended data
        )
        logger.info(f"Successfully appended {len(new_rows)} node entries to log: {log_filename}")

    except Exception as e:
        logger.error(f"Failed to put/append log data to blob {log_filename}: {e}", exc_info=True)
        # Re-raise so the calling function knows logging failed
        raise e


@app.get("/api/download_log/{config_id}")
async def download_log_file(config_id: str):
    """Downloads the full log CSV for a specific config ID from Vercel Blob."""
    if not config_id or config_id == "unknown" or config_id == "default-config-001":
         raise HTTPException(status_code=400, detail="Cannot download logs for unsaved or default configurations.")

    # Ensure config ID has the prefix if necessary (though it should from frontend)
    if not config_id.startswith(CONFIG_KEY_PREFIX):
         config_id_with_prefix = f"{CONFIG_KEY_PREFIX}{config_id}"
    else:
         config_id_with_prefix = config_id

    log_filename = f"{LOG_FILENAME_PREFIX}{config_id_with_prefix}{LOG_FILENAME_SUFFIX}"
    logger.info(f"Attempting to download log file: {log_filename}")

    try:
        # Use stream=True for potentially large files
        blob_response = await blob_download(pathname=log_filename, options={'stream': True})

        # Check if the download was successful (status code, etc.)
        # Note: vercel_blob download doesn't directly expose status code easily here,
        # it raises error on failure. We rely on that.

        # Get original filename for download header
        # Clean the config_id part for the download filename
        safe_filename_part = config_id_with_prefix.replace(CONFIG_KEY_PREFIX, "")
        download_filename = f"log_{safe_filename_part}.csv"

        return StreamingResponse(
            blob_response.iter_bytes(), # Stream the body
            media_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{download_filename}"'}
        )

    except Exception as e:
        # Handle file not found specifically if possible, otherwise generic error
        # Vercel Blob might raise a generic error, check message if needed
        err_str = str(e)
        if "404" in err_str or "NotFound" in err_str or "not found" in err_str.lower():
             logger.warning(f"Log file not found: {log_filename}")
             raise HTTPException(status_code=404, detail=f"Log file not found for config ID {config_id}")
        else:
             logger.error(f"Failed to download log file {log_filename}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Failed to download log file: {e}")


# --- Root Endpoint ---
@app.get("/")
def root():
    logger.info("Entered / route (API function).")
    return {"message": "BN API is running."}

print("DEBUG: Finished defining routes.", flush=True)
