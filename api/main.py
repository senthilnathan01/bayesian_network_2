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
import asyncio # For potential async operations with blob

print("--- DEBUG: api/main.py TOP LEVEL EXECUTION ---", flush=True)

try:
    from fastapi import FastAPI, HTTPException, Body, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse # For CSV download
    from pydantic import BaseModel, Field, validator
    from openai import AsyncOpenAI # Use Async client
    from dotenv import load_dotenv
    import redis # Keep using redis-py
    # from redis.asyncio import ConnectionPool, Redis # Use asyncio redis if available/needed later
    from redis.connection import ConnectionPool # Sync pool ok for now if redis client calls are simple
    from vercel_blob import put as blob_put, head as blob_head, list as blob_list, download as blob_download, delete as blob_delete # Import necessary Vercel Blob functions
    import aiohttp # Required for vercel_blob's internal async http calls
    from aiohttp import ClientTimeout # For specifying timeouts

    print("--- DEBUG: Imported FastAPI, CORS, Pydantic, AsyncOpenAI, Redis, Vercel Blob, aiohttp ---", flush=True)

    app = FastAPI(title="Dynamic Bayesian Network API")
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
# Prefer KV_URL if set (Vercel standard), fallback to REDIS_URL
redis_url = os.getenv("KV_URL") or os.getenv("REDIS_URL")
vercel_blob_token = os.getenv("BLOB_READ_WRITE_TOKEN") # Vercel automatically provides this

# Initialize async OpenAI client
openai_client = None
if openai_api_key:
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    logger.info("--- Async OpenAI client initialized ---")
else:
    logger.warning("OPENAI_API_KEY environment variable not found. Prediction endpoints will fail.")

# Redis Connection (using sync client for simplicity, ensure it doesn't block async FastAPI calls excessively)
redis_client = None
if redis_url:
    try:
        # decode_responses=True is important for getting strings
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("--- Successfully connected to Redis (sync client) ---")
    except Exception as e:
        logger.error(f"--- Failed to connect to Redis using URL {redis_url}: {e} ---", exc_info=True)
        redis_client = None # Ensure client is None if connection fails
else:
    logger.warning("--- Redis URL (KV_URL or REDIS_URL) not found. Config storage disabled. ---")

if not vercel_blob_token:
    logger.warning("BLOB_READ_WRITE_TOKEN environment variable not found. Blob storage operations will fail.")

# --- Constants ---
CONFIG_KEY_PREFIX = "bn_config:"
LOG_FILENAME_PREFIX = "bn_log_" # Store logs in a subfolder-like structure
LOG_FILENAME_SUFFIX = ".csv"
# Simpler CSV format for easier appending
CSV_HEADERS = ['Timestamp', 'ConfigID', 'ConfigName', 'NodeID', 'ProbabilityP1']

# --- Default Graph Structure (keep as before) ---
DEFAULT_GRAPH_STRUCTURE = { # ... keep definition ...
    "id": "default-config-001",
    "name": "Default Cognitive Model",
    "graph_structure": {
        "nodes": [
            {"id": "A1", "fullName": "Domain Expertise", "nodeType": "input"}, {"id": "A2", "fullName": "Web Literacy", "nodeType": "input"},
            {"id": "A3", "fullName": "Task Familiarity", "nodeType": "input"}, {"id": "A4", "fullName": "Goal Clarity", "nodeType": "input"},
            {"id": "A5", "fullName": "Motivation", "nodeType": "input"}, {"id": "UI", "fullName": "UI State (Quality/Clarity)", "nodeType": "input"},
            {"id": "H", "fullName": "History (Relevant past interactions)", "nodeType": "input"},
            {"id": "IS3", "fullName": "Task Understanding", "nodeType": "hidden"}, {"id": "IS4", "fullName": "Interaction Fluency", "nodeType": "hidden"},
            {"id": "IS5", "fullName": "Relevant Knowledge Activation", "nodeType": "hidden"}, {"id": "IS2", "fullName": "Cognitive Load", "nodeType": "hidden"},
            {"id": "IS1", "fullName": "Confidence", "nodeType": "hidden"},
            {"id": "O1", "fullName": "Predicted Success Probability", "nodeType": "hidden"}, {"id": "O2", "fullName": "Action Speed/Efficiency", "nodeType": "hidden"},
            {"id": "O3", "fullName": "Help Seeking Likelihood", "nodeType": "hidden"},
        ],
        "edges": [
            {"source": "A1", "target": "IS3"}, {"source": "A3", "target": "IS3"}, {"source": "A4", "target": "IS3"}, {"source": "H", "target": "IS3"},
            {"source": "A2", "target": "IS4"}, {"source": "UI", "target": "IS4"}, {"source": "H", "target": "IS4"},
            {"source": "A1", "target": "IS5"}, {"source": "A3", "target": "IS5"}, {"source": "IS3", "target": "IS5"},
            {"source": "IS4", "target": "IS2"}, {"source": "IS3", "target": "IS2"},
            {"source": "A5", "target": "IS1"}, {"source": "IS2", "target": "IS1"}, {"source": "IS3", "target": "IS1"}, {"source": "IS4", "target": "IS1"}, {"source": "IS5", "target": "IS1"},
            {"source": "IS1", "target": "O1"}, {"source": "IS2", "target": "O1"}, {"source": "IS3", "target": "O1"}, {"source": "IS5", "target": "O1"},
            {"source": "IS1", "target": "O2"}, {"source": "IS2", "target": "O2"}, {"source": "A5", "target": "O2"},
            {"source": "IS1", "target": "O3"}, {"source": "IS2", "target": "O3"}, {"source": "IS3", "target": "O3"}
        ]
    }
}


# --- Pydantic Models (keep as before) ---
class NodeData(BaseModel): # ... keep definition ...
    id: str; fullName: str; nodeType: str
    @validator('nodeType')
    def node_type_valid(cls, v):
        if v not in ['input', 'hidden']: raise ValueError("nodeType must be 'input' or 'hidden'"); return v
class EdgeData(BaseModel): source: str; target: str # ... keep definition ...
class GraphStructure(BaseModel): # ... keep definition ...
    nodes: List[NodeData]; edges: List[EdgeData]
    @validator('nodes')
    def nodes_unique(cls, v): ids = [n.id for n in v]; if len(ids) != len(set(ids)): raise ValueError("Node IDs must be unique"); return v
    @validator('edges')
    def edges_valid(cls, v, values):
        if 'nodes' not in values: return v
        node_ids = {n.id for n in values['nodes']};
        for edge in v:
            if edge.source not in node_ids or edge.target not in node_ids: raise ValueError(f"Edge {edge.source}->{edge.target} invalid");
        return v
class ContinuousUserInput(BaseModel): input_values: Dict[str, float] # ... keep definition ...
class PredictionPayload(ContinuousUserInput): # ... keep definition ...
    graph_structure: GraphStructure; config_id: Optional[str] = None; config_name: Optional[str] = "Unknown"
class SaveConfigPayload(BaseModel): # ... keep definition ...
    config_name: str = Field(..., min_length=1); graph_structure: GraphStructure
class LogPayload(BaseModel): # ... keep definition ...
    timestamp: str; configId: str; configName: str; probabilities: Dict[str, float] # NodeID: P(1)

# --- Graph Validation (keep as before) ---
def is_dag(graph: GraphStructure) -> Tuple[bool, Optional[str]]: # ... keep definition ...
    adj: Dict[str, List[str]] = {node.id: [] for node in graph.nodes}
    nodes_set = {node.id for node in graph.nodes}
    for edge in graph.edges:
        if edge.source not in nodes_set or edge.target not in nodes_set: return False, f"Edge refs invalid node: {edge.source}->{edge.target}"
        if edge.source in adj: adj[edge.source].append(edge.target)
        else: adj[edge.source] = [edge.target] # Initialize if needed (shouldn't be if nodes parsed first)
    path: Set[str] = set(); visited: Set[str] = set()
    def dfs(node: str) -> bool:
        path.add(node); visited.add(node)
        for neighbor in adj.get(node, []):
            if neighbor not in nodes_set: continue # Safety check
            if neighbor in path: return False # Cycle
            if neighbor not in visited:
                if not dfs(neighbor): return False # Cycle downstream
        path.remove(node); return True
    for node in nodes_set:
        if node not in visited:
            if not dfs(node): return False, f"Cycle detected near {node}"
    return True, None

# --- Helper Functions (keep as before) ---
def get_dynamic_node_info(graph: GraphStructure): # ... keep definition ...
    node_parents={n.id:[]for n in graph.nodes}; node_descriptions={n.id:n.fullName for n in graph.nodes}; node_types={n.id:n.nodeType for n in graph.nodes}; nodes_set={n.id for n in graph.nodes}
    for e in graph.edges:
        if e.source in nodes_set and e.target in nodes_set and e.target in node_parents: node_parents[e.target].append(e.source)
    all_nodes=list(nodes_set); target_nodes=[id for id,t in node_types.items() if t=='hidden']; input_nodes=[id for id,t in node_types.items() if t=='input']
    return node_parents,node_descriptions,node_types,all_nodes,target_nodes,input_nodes

# --- OpenAI Calls (Async now) ---
async def call_openai_for_probabilities(input_states: Dict[str, float], graph: GraphStructure) -> Dict[str, float]: # ... keep logic, make async ...
     if not openai_client: raise HTTPException(status_code=500, detail="OpenAI Client not initialized")
     node_parents, node_descriptions, _, _, target_nodes, input_nodes = get_dynamic_node_info(graph)
     missing_inputs=[n for n in input_nodes if n not in input_states];
     if missing_inputs: raise HTTPException(status_code=400,detail=f"Missing inputs: {', '.join(missing_inputs)}")
     input_desc_list=[f"- {n} ({node_descriptions.get(n,n)}): {'High' if v>=0.66 else ('Medium' if v>=0.33 else 'Low')} (P(1)={v:.2f})" for n,v in input_states.items() if n in input_nodes]
     input_context="\n".join(input_desc_list);
     if not target_nodes: return {}
     node_info = "\n".join([f"- {n} ({node_descriptions[n]}): influenced by {', '.join([node_descriptions.get(p, p) for p in node_parents.get(n,[])]) or 'none'}" for n in target_nodes])
     structure_description = f"Nodes & Dependencies:\n{node_info}"
     system_message = f"""You are an expert probabilistic reasoner simulating a DAG. Estimate P(Node=1) for hidden nodes based on inputs and structure. Return ONLY a JSON object mapping hidden node names ({', '.join(target_nodes)}) to their P(Node=1) float value (0.0-1.0). Example: {{"H1": 0.7}}"""
     user_message = f"Inputs (P=1):\n{input_context}\n\nNetwork Structure:\n{structure_description}\n\nEstimate P(Node=1) for hidden nodes. Return ONLY JSON."
     logger.debug("Sending probability prompt to OpenAI...")
     try:
         response = await openai_client.chat.completions.create(model="gpt-4o",messages=[{"role":"system","content":system_message},{"role":"user","content":user_message}],response_format={"type":"json_object"},max_tokens=1000,temperature=0.1)
         llm_output_raw = response.choices[0].message.content.strip(); logger.info(f"OpenAI Prob Output: {llm_output_raw}")
         estimated_probs = json.loads(llm_output_raw); validated_probs = {}
         for node in target_nodes:
             val = estimated_probs.get(node);
             if isinstance(val,(float,int)): validated_probs[node]=max(0.0,min(1.0,float(val)))
             else: logger.warning(f"Node '{node}' missing/invalid. Default 0.5."); validated_probs[node]=0.5
         return validated_probs
     except Exception as e: logger.error(f"OpenAI prob call error: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")

async def call_openai_for_reasoning(input_states: Dict[str, float], graph: GraphStructure, estimated_probabilities: Dict[str, float]) -> str: # ... keep logic, make async ...
     if not openai_client: return "Reasoning disabled: OpenAI Client not initialized."
     node_parents, node_descriptions, _, all_nodes, target_nodes, input_nodes = get_dynamic_node_info(graph)
     input_context="\n".join([f"- {n} ({node_descriptions.get(n,n)}): {'High' if v>=0.66 else ('Medium' if v>=0.33 else 'Low')} (P(1)={v:.2f})" for n,v in input_states.items() if n in input_nodes])
     all_probs = {**input_states, **estimated_probabilities}
     probs_text = "\n".join([f"- {node} ({node_descriptions.get(node, node)}): P(1)={all_probs.get(node, 0.5):.3f}" for node in all_nodes]) # Show all probs
     deps_text = "\n".join([f"- {n} <- {', '.join(node_parents.get(n,[])) or '(Input)'}" for n in all_nodes])
     system_message = """You are an analyst explaining a BN simulation. Given inputs, final probabilities, and dependencies, explain *concisely* why each hidden node likely got its probability, focusing on parent influence. Use a list format."""
     user_message = f"Inputs (P=1):\n{input_context}\n\nDependencies:\n{deps_text}\n\nFinal Probabilities (P=1):\n{probs_text}\n\nTask: Explain reasoning for hidden nodes ({', '.join(target_nodes)})."
     logger.debug("Sending reasoning prompt to OpenAI...")
     try:
         response = await openai_client.chat.completions.create(model="gpt-4o",messages=[{"role":"system","content":system_message},{"role":"user","content":user_message}],max_tokens=700,temperature=0.3)
         reasoning_text = response.choices[0].message.content.strip(); logger.info("OpenAI Reasoning Output received.")
         return reasoning_text
     except Exception as e: logger.error(f"OpenAI reasoning error: {e}", exc_info=True); return f"Could not generate reasoning: {str(e)}"

# --- NEW: Blob Logging Function ---
async def log_data_to_blob(log_entry: LogPayload):
    if not vercel_blob_token: logger.warning("BLOB_READ_WRITE_TOKEN not set. Skipping log."); return
    if not log_entry.configId or log_entry.configId == "unknown" or log_entry.configId == "default-config-001":
        logger.info(f"Skipping log for unsaved/default config: {log_entry.configId}"); return

    # Extract simple ID part for filename if prefix exists
    simple_config_id = log_entry.configId.replace(CONFIG_KEY_PREFIX, "")
    log_filename = f"{LOG_FILENAME_PREFIX}{simple_config_id}{LOG_FILENAME_SUFFIX}"
    logger.info(f"Attempting to log to blob: {log_filename}")

    new_rows_data = []
    for node_id, prob_p1 in log_entry.probabilities.items():
         new_rows_data.append([
             log_entry.timestamp, log_entry.configId, log_entry.configName, node_id, f"{prob_p1:.4f}"
         ])

    if not new_rows_data: logger.warning("No data rows generated for log entry."); return

    try:
        existing_content_str = ""
        needs_headers = False
        try:
            # Use head to check existence first (less bandwidth than download)
            head_result = await blob_head(pathname=log_filename, options={'token': vercel_blob_token})
            logger.debug(f"Log file {log_filename} exists (size: {head_result.get('size', 0)}). Downloading for append.")
            # If exists and size > 0, download
            if head_result.get('size', 0) > 0:
                 # Use aiohttp to download content as vercel_blob download stream can be tricky
                 async with aiohttp.ClientSession(timeout=ClientTimeout(total=10)) as session: # Increased timeout
                     async with session.get(head_result['url']) as response:
                         if response.status == 200:
                             existing_content_bytes = await response.read()
                             existing_content_str = existing_content_bytes.decode('utf-8').strip()
                             logger.debug(f"Read {len(existing_content_bytes)} bytes from existing log.")
                         else:
                             logger.warning(f"Failed to download existing log {log_filename} (Status {response.status}), will create new.")
                             needs_headers = True
            else:
                # Exists but empty, needs headers
                 needs_headers = True

        except Exception as e: # Catches errors from head (like 404) or download
            logger.info(f"Log file {log_filename} likely doesn't exist or download failed ({e}). Creating new with headers.")
            needs_headers = True

        # Build new CSV content in memory
        output = io.StringIO()
        writer = csv.writer(output)

        if needs_headers:
            writer.writerow(CSV_HEADERS)
        elif existing_content_str:
            # Write existing content first if it's not empty
            output.write(existing_content_str)
            output.write('\n') # Ensure newline before new rows

        # Write the new rows
        writer.writerows(new_rows_data)
        final_csv_content = output.getvalue().encode('utf-8')
        output.close()

        # Upload the combined/new content, overwriting the file
        put_result = await blob_put(
            pathname=log_filename,
            body=final_csv_content,
            options={'addRandomSuffix': False, 'contentType': 'text/csv', 'token': vercel_blob_token}
        )
        logger.info(f"Successfully logged to {log_filename} at {put_result['url']}")

    except Exception as e:
        logger.error(f"Failed to put/append log data to blob {log_filename}: {e}", exc_info=True)
        # Do not re-raise, just log the error. Prediction should still succeed for user.


# --- API Endpoints ---
@app.get("/api/ping")
async def ping(): # ... keep as before ...
    redis_status = "disabled";
    if redis_client: try: redis_client.ping(); redis_status = "connected"
    except Exception as e: redis_status = f"disconnected ({e})"
    return {"message": "pong", "redis_status": redis_status}

@app.get("/api/configs/default")
async def get_default_configuration(): # ... keep as before ...
     logger.info("Serving default configuration."); return DEFAULT_GRAPH_STRUCTURE

@app.post("/api/configs")
async def save_configuration(payload: SaveConfigPayload): # ... keep sync redis call for simplicity ...
    if not redis_client: raise HTTPException(status_code=503, detail="Storage unavailable.")
    is_valid_dag, cycle_info = is_dag(payload.graph_structure);
    if not is_valid_dag: raise HTTPException(status_code=400, detail=f"Invalid graph: Not a DAG. {cycle_info or ''}")
    config_id = f"{CONFIG_KEY_PREFIX}{uuid4()}"; config_data = { "id": config_id, "name": payload.config_name, "graph_structure": payload.graph_structure.dict() }
    try: redis_client.set(config_id, json.dumps(config_data)); logger.info(f"Saved config '{payload.config_name}' ID: {config_id}"); return {"message": "Configuration saved", "config_id": config_id, "config_name": payload.config_name}
    except Exception as e: logger.error(f"Redis save error: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Failed to save: {str(e)}")

@app.get("/api/configs")
async def list_configurations(): # ... keep sync redis call ...
     if not redis_client: logger.warning("Redis unavailable for list."); return []
     configs_summary = [];
     try:
         config_keys = [key for key in redis_client.scan_iter(match=f"{CONFIG_KEY_PREFIX}*")]
         if not config_keys: return []
         config_jsons = redis_client.mget(config_keys) # Use mget for efficiency
         for key, config_json in zip(config_keys, config_jsons):
             if config_json: try: config_data = json.loads(config_json); configs_summary.append({ "id": config_data.get("id", key), "name": config_data.get("name", "Unnamed") })
             except Exception as e: logger.warning(f"Parse fail key {key}: {e}"); configs_summary.append({"id": key, "name": "*Load Error*"})
             else: logger.warning(f"Key {key} from scan not found with MGET.")
         return configs_summary
     except Exception as e: logger.error(f"Redis list error: {e}", exc_info=True); return [] # Return empty list on error

@app.get("/api/configs/{config_id}")
async def load_configuration(config_id: str): # ... keep sync redis call ...
     if not redis_client: raise HTTPException(status_code=503, detail="Storage unavailable.")
     key = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
     try: config_json = redis_client.get(key);
     if config_json is None: raise HTTPException(status_code=404, detail="Config not found.")
     return json.loads(config_json)
     except Exception as e: logger.error(f"Redis load error {key}: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")

@app.delete("/api/configs/{config_id}")
async def delete_configuration(config_id: str): # Delete now needs to be async for blob delete
    if not redis_client: raise HTTPException(status_code=503, detail="Storage unavailable.")
    key = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
    log_filename = f"{LOG_FILENAME_PREFIX}{key.replace(CONFIG_KEY_PREFIX, '')}{LOG_FILENAME_SUFFIX}"
    redis_deleted = False
    try: # Delete from Redis first
        deleted_count = redis_client.delete(key);
        if deleted_count == 0: raise HTTPException(status_code=404, detail="Config not found in Redis.")
        redis_deleted = True
        logger.info(f"Deleted config '{key}' from Redis.")
        # Delete log file from Blob storage (best effort)
        try:
             if vercel_blob_token: # Only attempt if token exists
                 await blob_delete(pathname=log_filename, options={'token': vercel_blob_token})
                 logger.info(f"Deleted corresponding log file '{log_filename}'.")
             else: logger.warning(f"Skipping log deletion for {log_filename} - No Blob Token.")
        except Exception as blob_error: logger.warning(f"Config {key} deleted, but failed to delete log file '{log_filename}': {blob_error}")
        return {"message": "Configuration deleted successfully."}
    except HTTPException as e: raise e # Propagate 404 etc.
    except Exception as e: logger.error(f"Error deleting config {key}: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")


@app.post("/api/configs/set_default")
async def set_default_configuration(config_id: str = Body(...)): # ... keep sync redis ...
     # This endpoint seems unused in current frontend, but keep it for now
     if not redis_client: raise HTTPException(status_code=503, detail="Storage unavailable.")
     key = config_id if config_id.startswith(CONFIG_KEY_PREFIX) else f"{CONFIG_KEY_PREFIX}{config_id}"
     try:
         if not redis_client.exists(key): raise HTTPException(status_code=404, detail="Config not found.")
         # Storing the *key* of the default config under a known key name
         # DEFAULT_CONFIG_KEY = "bn_default_config" # Define this constant if not already done
         # redis_client.set(DEFAULT_CONFIG_KEY, key)
         # logger.info(f"Set {key} as default config marker.")
         # return {"message": f"Configuration marked as default."}
         # NOTE: The current default logic loads a *hardcoded* structure.
         # Setting a dynamic default requires changing GET /api/configs/default logic.
         # For now, this endpoint doesn't actually change the default load behavior.
         logger.warning("Set default endpoint called, but default load logic is currently hardcoded.")
         return {"message": "Set default functionality not fully implemented in load logic."}

     except Exception as e: logger.error(f"Set default error: {e}"); raise HTTPException(status_code=500, detail=f"Failed to set default: {str(e)}")


# --- Prediction Endpoint (Main async logic) ---
@app.post("/api/predict_openai_bn_single_call")
async def predict_openai_bn_single_call(payload: PredictionPayload):
    logger.info("Prediction request received.")
    try:
        is_valid_dag, cycle_info = is_dag(payload.graph_structure);
        if not is_valid_dag: raise HTTPException(status_code=400, detail=f"Invalid graph: Not a DAG. {cycle_info or ''}")

        logger.info("Calling OpenAI for probabilities...")
        estimated_target_probs = await call_openai_for_probabilities(payload.input_values, payload.graph_structure)

        logger.info("Calling OpenAI for reasoning...")
        reasoning_text = await call_openai_for_reasoning(payload.input_values, payload.graph_structure, estimated_target_probs)

        # Combine results
        all_current_probabilities = {**payload.input_values, **estimated_target_probs}
        _, node_descriptions, _, all_nodes, _, input_nodes = get_dynamic_node_info(payload.graph_structure)
        final_result_probs = {}
        log_probs_p1 = {} # For logging P(1) values
        for node in all_nodes:
            node_desc = node_descriptions.get(node, node)
            p1 = all_current_probabilities.get(node, 0.5) # Default to 0.5 if missing
            p1_clamped = max(0.0, min(1.0, p1))
            final_result_probs[node] = {"0": 1.0 - p1_clamped, "1": p1_clamped, "description": node_desc}
            log_probs_p1[node] = p1_clamped # Log the clamped P(1)

        # Prepare response payload
        response_payload = {
            "probabilities": final_result_probs,
            "llm_reasoning": reasoning_text,
            "llm_context": { # Context used for reasoning call might be more useful here
                 "input_states": [{ "node": n, "description": node_descriptions.get(n,n), "value": v, "state": ("High" if v >= 0.66 else ("Medium" if v >= 0.33 else "Low")) } for n,v in payload.input_values.items() if n in input_nodes],
                 "node_dependencies": {node_id: parents for node_id, parents in get_dynamic_node_info(payload.graph_structure)[0].items() if parents}, # Show only deps for non-inputs
                 "node_descriptions": node_descriptions
            }
        }

        # Log the prediction result to Blob (best effort)
        log_entry = LogPayload(
            timestamp=datetime.utcnow().isoformat() + "Z",
            configId=payload.config_id or "unknown", # Get from payload
            configName=payload.config_name or "Unknown", # Get from payload
            probabilities=log_probs_p1 # Use the prepared P(1) dict
        )
        try:
            logger.info(f"Attempting to log prediction for config: {log_entry.configId}")
            await log_data_to_blob(log_entry) # Call the async logging function
            logger.info("Logging task completed (or skipped).")
        except Exception as log_err:
            # Log the error but don't fail the request
            logger.error(f"Failed during background logging: {log_err}", exc_info=True)

        logger.info("Prediction request completed successfully.")
        return response_payload

    except HTTPException as e: raise e # Re-raise HTTP exceptions directly
    except Exception as e: logger.error(f"Error in prediction endpoint: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- NEW: Download All Logs Endpoint ---
@app.get("/api/download_log/{config_id}")
async def download_log_file(config_id: str):
    """Downloads the full log CSV for a specific config ID from Vercel Blob."""
    if not vercel_blob_token: raise HTTPException(status_code=503, detail="Blob storage not configured.")
    if not config_id or config_id == "unknown" or config_id == "default-config-001":
        raise HTTPException(status_code=400, detail="Cannot download logs for unsaved or default configurations.")

    # Construct filename using the simple ID part
    simple_config_id = config_id.replace(CONFIG_KEY_PREFIX, "")
    log_filename = f"{LOG_FILENAME_PREFIX}{simple_config_id}{LOG_FILENAME_SUFFIX}"
    logger.info(f"Requesting download for log file: {log_filename}")

    try:
        # Use head to get metadata including the URL
        head_result = await blob_head(pathname=log_filename, options={'token': vercel_blob_token})
        blob_url = head_result.get('url')
        content_length = head_result.get('size', 0)

        if not blob_url:
            logger.error(f"Blob URL not found in head result for {log_filename}")
            raise HTTPException(status_code=404, detail=f"Log file URL not found for {config_id}")
        if content_length == 0:
             logger.warning(f"Log file {log_filename} is empty.")
             # Return an empty CSV or a 404? Let's return empty CSV.
             empty_csv = io.BytesIO(f"{','.join(CSV_HEADERS)}\n".encode('utf-8'))
             return StreamingResponse(empty_csv, media_type='text/csv', headers={'Content-Disposition': f'attachment; filename="log_{simple_config_id}_empty.csv"'})

        # Stream the download using aiohttp
        async def stream_generator():
             # Increased timeout for potentially larger files
             async with aiohttp.ClientSession(timeout=ClientTimeout(total=60)) as session:
                 async with session.get(blob_url) as response:
                     if response.status == 200:
                         while True:
                             chunk = await response.content.read(8192) # Read in chunks
                             if not chunk:
                                 break
                             yield chunk
                     elif response.status == 404:
                          logger.warning(f"Blob URL returned 404: {blob_url}")
                          raise HTTPException(status_code=404, detail=f"Log file not found at URL for {config_id}")
                     else:
                         error_text = await response.text()
                         logger.error(f"Failed to download log {log_filename}: HTTP {response.status} - {error_text}")
                         raise HTTPException(status_code=502, detail=f"Failed to download log file (upstream error {response.status})")

        download_filename = f"log_{simple_config_id}.csv"
        return StreamingResponse(
            stream_generator(),
            media_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{download_filename}"',
                     'Content-Length': str(content_length)} # Provide content length if available
        )

    except HTTPException as e: raise e # Re-raise explicit HTTP exceptions
    except Exception as e:
        # Handle potential 404 from blob_head if file doesn't exist
        err_str = str(e).lower()
        if "not_found" in err_str or "not found" in err_str or "404" in err_str:
             logger.warning(f"Log file not found via head/download: {log_filename}")
             raise HTTPException(status_code=404, detail=f"Log file not found for config ID {config_id}")
        else:
             logger.error(f"Failed to initiate log download for {log_filename}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Failed to download log file: {str(e)}")


# --- Root Endpoint ---
@app.get("/")
async def root(): logger.info("Root endpoint '/' accessed."); return {"message": "BN API is running."}

print("--- DEBUG: Finished defining routes and helper functions ---", flush=True)
